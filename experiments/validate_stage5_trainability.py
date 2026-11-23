"""Validation script for Stage 5: Trainability Analysis results.

This script loads results from stage5_trainability/summary.json and performs
physics-aware sanity checks:

1. **Value integrity**: Trainability estimate is in [0, 1] and not NaN.
2. **Gradient variance**: Finite and strictly positive. The measured gradient
   is with respect to the classifier's variational angles, which every
   encoding leaves differentiable, so a zero variance is a bug, not physics.
3. **Risk level**: barren_plateau_risk is one of {"low", "medium", "high", "unknown"}.
4. **Non-entangling encodings**: Product-state encodings should not exhibit
   barren plateaus (risk != "high").
5. **Depth-trainability correlation**: Shallow circuits should generally have
   higher trainability than deep circuits (with tolerance for noise).
6. **Per-parameter variance**: Individual parameter variances are non-negative.
7. **Light cone**: parameters outside ``live_parameter_mask`` have zero
   variance, and the provenance fields (``observable``, ``n_var_layers``,
   ``n_live_parameters``) are present and consistent.
8. **Width invariance**: for encodings that prepare a product of per-feature
   qubit states, the read-out sees the same two-qubit reduced state at every
   width, so the gradient variance must not change with ``n_features``.
9. **Sample quality**: Sufficient successful samples for reliable estimation.

Physics Notes
-------------
- **What is measured.** ``Var[∂C/∂θ]`` for the Stage 6a variational
  classifier, ``C(x, θ) = ⟨Z₀⟩`` after ``U(x)`` and two layers of ``RY(θ)`` +
  a linear CNOT chain, at random ``x`` and random initial ``θ``, averaged over
  the parameters inside the read-out's light cone.

- **Light cone.** ``Z₀`` commutes with the chain's first CNOT (qubit 0 is its
  control), so only ``θ[l, q]`` with ``q ≤ n_var_layers - 1 - l`` can move the
  cost: three parameters at any width. The remaining entries of
  ``per_parameter_variance`` are zero up to round-off (~1e-33).

- **No structural zeros.** Earlier releases differentiated the data ``x``
  with a ±π/2 shift and measured ``P(0…0)``. That produced exact zeros for
  BasisEncoding (a step function of ``x``) and for single-layer IQP/ZZ/Pauli
  maps (``P(0…0) = 2⁻ⁿ`` for every ``x``), which this script used to exempt.
  Neither is a barren plateau, and neither arises for ``∂C/∂θ``.

Usage:
    python experiments/validate_stage5_trainability.py
    python experiments/validate_stage5_trainability.py --results-path custom/path/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

# Compute paths relative to this script's location
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add project root to path for imports
sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_RESULTS_PATH = str(_SCRIPT_DIR / "results" / "raw" / "stage5_trainability" / "summary.json")

# Non-entangling encodings (should have high trainability)
NON_ENTANGLING_ENCODINGS: set[str] = {
    "angle",
    "basis",
    "higher_order_angle",
}

# Entangling encodings (may have barren plateau issues).
# Names must match the encoding_name field in summary.json exactly.
ENTANGLING_ENCODINGS: set[str] = {
    "iqp",
    "zz_feature_map",
    "pauli_feature_map",
    "data_reuploading",
    "hardware_efficient",
    "qaoa_encoding",
    "hamiltonian_encoding",
    "symmetry_inspired",
    "trainable_encoding",
    "so2_equivariant",
    "cyclic_equivariant",
    "swap_equivariant",
    "amplitude",
}

# Valid barren plateau risk levels
VALID_RISK_LEVELS = {"low", "medium", "high", "unknown"}

# Encoding families for grouping
ENCODING_FAMILIES: dict[str, list[str]] = {
    "Non-Entangling": ["angle", "basis", "higher_order_angle"],
    "IQP-Based": ["iqp", "zz_feature_map"],
    "Pauli-Based": ["pauli_feature_map"],
    "Data Re-uploading": ["data_reuploading"],
    "Hardware-Efficient": ["hardware_efficient"],
    "QAOA/Hamiltonian": ["qaoa_encoding", "hamiltonian_encoding"],
    "Symmetry-Based": ["symmetry_inspired"],
    "Trainable": ["trainable_encoding"],
    "Equivariant": ["so2_equivariant", "cyclic_equivariant", "swap_equivariant"],
    "Amplitude": ["amplitude"],
}

# Encodings whose qubit q depends on feature q alone, so the read-out's
# reduced state on qubits {0, 1} — and therefore the gradient variance — is
# the same at every width. HigherOrderAngleEncoding is excluded: it is a
# product state too, but its qubit angles combine features from across the
# input, so the reduced state does depend on the width.
WIDTH_INVARIANT_ENCODINGS: set[str] = {"angle", "basis"}

# Relative tolerance for the width-invariance check. The variance estimate's
# standard error is ~σ²·√(2/(3(n-1))) ≈ 4% at 500 samples and three live
# parameters, so 25% is a > 5σ band.
WIDTH_INVARIANCE_RTOL: float = 0.25

# Upper bound on the variance of a parameter outside the light cone. Exact
# arithmetic gives zero; double-precision round-off gives ~1e-33.
DEAD_PARAMETER_VARIANCE_ATOL: float = 1e-20

VALID_OBSERVABLES: set[str] = {"pauli_z", "global_z", "computational"}


# =============================================================================
# Logging Setup
# =============================================================================

# Configured in main(), so importing this module (e.g. from the test suite)
# does not reconfigure the caller's logging.
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None


@dataclass
class EncodingValidation:
    """Validation results for a single encoding configuration."""

    encoding_name: str
    params: dict[str, Any]
    results: list[ValidationResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_trainability_result(
    encoding_name: str,
    params: dict[str, Any],
    result: dict[str, Any],
) -> EncodingValidation:
    """Validate a single trainability result.

    Parameters
    ----------
    encoding_name : str
        Name of the encoding.
    params : dict[str, Any]
        Encoding parameters.
    result : dict[str, Any]
        Result data from summary.json.

    Returns
    -------
    EncodingValidation
        Validation results.
    """
    checks: list[ValidationResult] = []

    # Extract trainability values
    trainability_estimate = result.get("trainability_estimate")
    gradient_variance = result.get("gradient_variance")
    barren_plateau_risk = result.get("barren_plateau_risk", "unknown")
    per_parameter_variance = result.get("per_parameter_variance", [])
    n_successful_samples = result.get("n_successful_samples")

    # Check 1: Trainability estimate is not NaN
    if trainability_estimate is not None:
        is_nan = math.isnan(trainability_estimate) if isinstance(trainability_estimate, float) else False
        checks.append(ValidationResult(
            check_name="trainability_not_nan",
            passed=not is_nan,
            message=f"trainability_estimate={trainability_estimate} ({'NaN' if is_nan else 'valid'})",
        ))

    # Check 2: Trainability estimate is in valid range [0, 1]
    if trainability_estimate is not None and not math.isnan(trainability_estimate):
        in_range = 0.0 <= trainability_estimate <= 1.0
        checks.append(ValidationResult(
            check_name="trainability_in_range",
            passed=in_range,
            message=f"trainability_estimate={trainability_estimate:.4f} ({'in [0,1]' if in_range else 'OUT OF RANGE'})",
            details={"value": trainability_estimate, "min": 0.0, "max": 1.0},
        ))

    # Check 3: Gradient variance is finite and strictly positive.
    #
    # The gradient is taken with respect to the classifier's RY angles, each
    # of which enters one gate, so every live parameter has a gradient that
    # is not identically zero for any encoding. A genuine barren plateau makes
    # the variance small, never exactly zero; zero or NaN indicates a bug.
    if gradient_variance is not None:
        is_finite = isinstance(gradient_variance, (int, float)) and math.isfinite(gradient_variance)
        is_positive = is_finite and gradient_variance > 0
        checks.append(ValidationResult(
            check_name="gradient_variance_positive",
            passed=is_positive,
            message=(
                f"gradient_variance={gradient_variance:.6e}"
                if is_finite
                else f"gradient_variance={gradient_variance} (not finite)"
            ),
            details={"value": gradient_variance},
        ))

    # Check 4: Valid barren plateau risk level
    is_valid_risk = barren_plateau_risk in VALID_RISK_LEVELS
    checks.append(ValidationResult(
        check_name="valid_risk_level",
        passed=is_valid_risk,
        message=f"barren_plateau_risk='{barren_plateau_risk}' is {'valid' if is_valid_risk else 'invalid'}",
    ))

    # Check 5: Flag high barren plateau risk
    if barren_plateau_risk == "high":
        # This is a warning, not a failure - just flagging for attention
        checks.append(ValidationResult(
            check_name="high_risk_flagged",
            passed=True,  # Not a failure, just flagging
            message=f"WARNING: {encoding_name} has barren_plateau_risk='high'",
            details={
                "encoding": encoding_name,
                "params": params,
                "risk": "high",
            },
        ))

    # Check 6: Non-entangling encodings should not exhibit barren plateaus.
    #
    # PHYSICAL RATIONALE:
    # A non-entangling encoding hands the read-out a product state. The
    # read-out's light cone covers qubits {0, 1} only, so the cost depends on
    # a two-qubit product state whatever the width, and its gradient variance
    # cannot decay with n. A "high" risk here would indicate a bug.
    if encoding_name in NON_ENTANGLING_ENCODINGS and trainability_estimate is not None:
        if not math.isnan(trainability_estimate):
            is_not_high_risk = barren_plateau_risk != "high"
            checks.append(ValidationResult(
                check_name="non_entangling_no_barren_plateau",
                passed=is_not_high_risk,
                message=(
                    f"{encoding_name} (non-entangling) "
                    f"barren_plateau_risk='{barren_plateau_risk}', "
                    f"trainability={trainability_estimate:.4f}"
                ),
                details={
                    "encoding": encoding_name,
                    "trainability": trainability_estimate,
                    "risk": barren_plateau_risk,
                    "n_features": params.get("n_features"),
                },
            ))

    # Check 7: Per-parameter variance values are finite and non-negative
    if per_parameter_variance:
        invalid_indices = [
            i
            for i, val in enumerate(per_parameter_variance)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0
        ]
        checks.append(ValidationResult(
            check_name="per_param_variance_non_negative",
            passed=not invalid_indices,
            message=(
                "per_parameter_variance all finite and non-negative"
                if not invalid_indices
                else f"per_parameter_variance invalid at {invalid_indices}"
            ),
            details={
                "n_params": len(per_parameter_variance),
                "invalid_indices": invalid_indices,
            },
        ))

    # Check 8: Provenance and light cone.
    checks.extend(_validate_light_cone(result))

    # Check 9: n_successful_samples is reasonable
    if n_successful_samples is not None:
        is_reasonable = n_successful_samples >= 10  # At least some samples succeeded
        checks.append(ValidationResult(
            check_name="samples_reasonable",
            passed=is_reasonable,
            message=f"n_successful_samples={n_successful_samples} (min: 10)",
        ))

    return EncodingValidation(encoding_name, params, checks)


def _validate_light_cone(result: dict[str, Any]) -> list[ValidationResult]:
    """Check the provenance fields and that dead parameters really are dead."""
    observable = result.get("observable")
    n_var_layers = result.get("n_var_layers")
    mask = result.get("live_parameter_mask")
    n_live = result.get("n_live_parameters")
    variances = result.get("per_parameter_variance")

    missing = [
        name
        for name, value in (
            ("observable", observable),
            ("n_var_layers", n_var_layers),
            ("live_parameter_mask", mask),
            ("n_live_parameters", n_live),
            ("per_parameter_variance", variances),
        )
        if value is None
    ]
    if missing:
        return [ValidationResult(
            check_name="provenance_present",
            passed=False,
            message=f"result lacks {missing}; was it produced before the θ-gradient definition?",
        )]

    assert isinstance(mask, list) and isinstance(variances, list)
    assert isinstance(n_var_layers, int)
    consistent = (
        observable in VALID_OBSERVABLES
        and n_var_layers >= 1
        and len(mask) == len(variances)
        and len(mask) % n_var_layers == 0
        and n_live == sum(bool(m) for m in mask)
        and n_live >= 1
    )
    checks = [ValidationResult(
        check_name="provenance_consistent",
        passed=consistent,
        message=(
            f"observable={observable!r}, n_var_layers={n_var_layers}, "
            f"n_live_parameters={n_live} of {len(mask)}"
        ),
    )]
    if consistent:
        dead = [v for v, live in zip(variances, mask) if not live]
        worst = max(dead, default=0.0)
        checks.append(ValidationResult(
            check_name="dead_parameters_have_zero_variance",
            passed=worst <= DEAD_PARAMETER_VARIANCE_ATOL,
            message=(
                f"max variance outside the light cone = {worst:.2e} "
                f"(<= {DEAD_PARAMETER_VARIANCE_ATOL:.0e})"
            ),
            details={"max_dead_variance": worst, "n_dead": len(dead)},
        ))
    return checks


def _depth_parameter(params: dict[str, Any]) -> int | None:
    """The repetition count that sets an encoding's depth, if it has one."""
    for key in ("reps", "n_layers"):
        if key in params:
            return int(params[key])
    return None


def validate_depth_trainability_correlation(results: list[dict[str, Any]]) -> list[ValidationResult]:
    """Validate that shallow circuits have higher trainability than deep circuits.

    Compares configurations of one encoding that differ only in their
    repetition count (``reps`` or ``n_layers``) at a fixed ``n_features``.

    Parameters
    ----------
    results : list[dict]
        All results from summary.json.

    Returns
    -------
    list[ValidationResult]
        Validation results for depth-trainability correlation.
    """
    checks: list[ValidationResult] = []

    # Group by encoding name and n_features
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)

    for entry in results:
        if entry.get("status") != "success":
            continue

        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})
        result = entry.get("result", {})

        depth = _depth_parameter(params)
        trainability = result.get("trainability_estimate")
        if depth is None or trainability is None or math.isnan(trainability):
            continue
        grouped[(enc_name, params.get("n_features", 4))].append((depth, trainability))

    # For each encoding, check if trainability decreases with depth
    for (enc_name, n_features), values in grouped.items():
        if len(values) < 2:
            continue

        values.sort(key=lambda x: x[0])
        depth_list = [v[0] for v in values]
        train_list = [v[1] for v in values]

        # Allow increases up to 0.3 (some noise is expected)
        reasonable = all(
            train_list[i] <= train_list[i - 1] + 0.3 for i in range(1, len(train_list))
        )

        checks.append(ValidationResult(
            check_name=f"depth_correlation_{enc_name}_n{n_features}",
            passed=reasonable,
            message=f"{enc_name}(n={n_features}): reps={depth_list} -> train={[f'{t:.3f}' for t in train_list]}",
            details={
                "encoding": enc_name,
                "n_features": n_features,
                "reps": depth_list,
                "trainability": train_list,
            },
        ))

    return checks


def validate_width_invariance(results: list[dict[str, Any]]) -> list[ValidationResult]:
    """Product-of-feature-state encodings must not change with width.

    For :data:`WIDTH_INVARIANT_ENCODINGS` the read-out sees the same reduced
    state on qubits {0, 1} at every ``n_features``, so the gradient variance
    may differ between widths only by sampling noise. A systematic decay
    would mean the estimator averages over parameters outside the light cone.

    Parameters
    ----------
    results : list[dict]
        All results from summary.json.

    Returns
    -------
    list[ValidationResult]
        One check per width-invariant encoding with at least two widths.
    """
    by_encoding: dict[str, dict[int, float]] = defaultdict(dict)
    for entry in results:
        if entry.get("status") != "success":
            continue
        name = entry.get("encoding_name", "unknown")
        if name not in WIDTH_INVARIANT_ENCODINGS:
            continue
        gv = entry.get("result", {}).get("gradient_variance")
        if gv is not None:
            by_encoding[name][int(entry.get("encoding_params", {}).get("n_features", 4))] = float(gv)

    checks: list[ValidationResult] = []
    for name, by_width in sorted(by_encoding.items()):
        if len(by_width) < 2:
            continue
        lo, hi = min(by_width.values()), max(by_width.values())
        spread = (hi - lo) / hi if hi > 0 else float("inf")
        checks.append(ValidationResult(
            check_name=f"width_invariance_{name}",
            passed=spread <= WIDTH_INVARIANCE_RTOL,
            message=(
                f"{name}: gradient_variance by width "
                f"{ {w: round(v, 4) for w, v in sorted(by_width.items())} } "
                f"(relative spread {spread:.1%}, tolerance {WIDTH_INVARIANCE_RTOL:.0%})"
            ),
            details={"by_width": by_width, "relative_spread": spread},
        ))
    return checks


def collect_high_risk_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect all encodings with high barren plateau risk.

    Parameters
    ----------
    results : list[dict]
        All results from summary.json.

    Returns
    -------
    list[dict]
        High-risk cases.
    """
    high_risk = []

    for entry in results:
        if entry.get("status") != "success":
            continue

        result = entry.get("result", {})
        if result.get("barren_plateau_risk") == "high":
            high_risk.append({
                "encoding": entry.get("encoding_name"),
                "params": entry.get("encoding_params"),
                "trainability": result.get("trainability_estimate"),
                "gradient_variance": result.get("gradient_variance"),
            })

    return high_risk


# =============================================================================
# Table Generation
# =============================================================================

def format_params(params: dict[str, Any]) -> str:
    """Format parameters as a short string."""
    if not params:
        return "default"
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        key_abbrev = {
            "n_features": "n",
            "reps": "r",
            "entanglement": "ent",
            "n_layers": "L",
        }.get(k, k)
        parts.append(f"{key_abbrev}={v}")
    return ", ".join(parts)


def get_encoding_family(encoding_name: str) -> str:
    """Get the family name for an encoding."""
    for family, encodings in ENCODING_FAMILIES.items():
        if encoding_name in encodings:
            return family
    return "Other"


def generate_markdown_table(results: list[dict[str, Any]]) -> str:
    """Generate a publication-ready Markdown table.

    Parameters
    ----------
    results : list[dict]
        Results from summary.json.

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "# Stage 5: Trainability Analysis Results",
        "",
        "## Summary Table",
        "",
        "Sorted by trainability (descending).",
        "",
        "| Encoding | Parameters | Trainability | Gradient Var | BP Risk | n_samples |",
        "|----------|------------|--------------|--------------|---------|-----------|",
    ]

    # Sort by trainability (descending)
    sorted_results = sorted(
        [r for r in results if r.get("result", {}).get("trainability_estimate") is not None],
        key=lambda x: x.get("result", {}).get("trainability_estimate", 0),
        reverse=True,
    )

    for entry in sorted_results:
        result = entry.get("result", {})
        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})

        trainability = result.get("trainability_estimate", 0)
        gradient_var = result.get("gradient_variance", 0)
        bp_risk = result.get("barren_plateau_risk", "—")
        n_samples = result.get("n_successful_samples", "—")

        # Format values
        train_str = f"{trainability:.4f}" if isinstance(trainability, (int, float)) else "—"
        gv_str = f"{gradient_var:.6f}" if isinstance(gradient_var, (int, float)) else "—"

        # Add risk indicator emoji
        risk_emoji = "🟢" if bp_risk == "low" else ("🟡" if bp_risk == "medium" else "🔴" if bp_risk == "high" else "⚪")

        # Add indicator for non-entangling
        entangling_marker = "" if enc_name in ENTANGLING_ENCODINGS else " (NE)"

        lines.append(
            f"| {enc_name}{entangling_marker} | {format_params(params)} | "
            f"{train_str} | {gv_str} | {risk_emoji} {bp_risk} | {n_samples} |"
        )

    # Add high-risk section
    high_risk = collect_high_risk_cases(
        [r for r in results if r.get("status") == "success"]
    )

    if high_risk:
        lines.extend([
            "",
            "## High Barren Plateau Risk Cases",
            "",
            "The following configurations have high barren plateau risk:",
            "",
        ])
        for case in high_risk:
            lines.append(f"- **{case['encoding']}** ({format_params(case['params'])}): "
                        f"trainability={case['trainability']:.4f}, "
                        f"gradient_variance={case['gradient_variance']:.6f}")

    # Add legend
    lines.extend([
        "",
        "## Legend",
        "",
        "- (NE): Non-entangling encoding (expected high trainability)",
        "- Trainability: Higher values indicate easier optimization",
        "- Gradient Var: Variance of gradients (low = barren plateau)",
        "- BP Risk: Barren plateau risk (🟢 low, 🟡 medium, 🔴 high)",
    ])

    return "\n".join(lines)


def generate_latex_table(results: list[dict[str, Any]]) -> str:
    """Generate a publication-ready LaTeX table.

    Parameters
    ----------
    results : list[dict]
        Results from summary.json.

    Returns
    -------
    str
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Stage 5: Trainability Analysis Results}",
        r"\label{tab:stage5-trainability}",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Encoding & Parameters & Trainability & Gradient Var. & BP Risk \\",
        r"\midrule",
    ]

    # Sort by trainability (descending)
    sorted_results = sorted(
        [r for r in results if r.get("result", {}).get("trainability_estimate") is not None],
        key=lambda x: x.get("result", {}).get("trainability_estimate", 0),
        reverse=True,
    )

    for entry in sorted_results[:20]:  # Top 20 for space
        result = entry.get("result", {})
        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})

        # Escape underscores for LaTeX
        enc_name_tex = enc_name.replace("_", r"\_")
        params_tex = format_params(params).replace("_", r"\_")

        trainability = result.get("trainability_estimate", 0)
        gradient_var = result.get("gradient_variance", 0)
        bp_risk = result.get("barren_plateau_risk", "—")

        train_str = f"{trainability:.4f}" if isinstance(trainability, (int, float)) else "—"
        gv_str = f"{gradient_var:.6f}" if isinstance(gradient_var, (int, float)) else "—"

        lines.append(
            f"  {enc_name_tex} & {params_tex} & {train_str} & {gv_str} & {bp_risk} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Main Validation
# =============================================================================

def load_results(path: str) -> dict[str, Any]:
    """Load results from summary.json file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_validation(results_path: str) -> tuple[int, int, list[EncodingValidation]]:
    """Run all validation checks.

    Parameters
    ----------
    results_path : str
        Path to summary.json file.

    Returns
    -------
    tuple[int, int, list[EncodingValidation]]
        (total_passed, total_failed, all_validations)
    """
    data = load_results(results_path)
    results = data.get("results", [])

    if not results:
        logger.warning("No results found in file")
        return 0, 0, []

    logger.info(f"Loaded {len(results)} results from {results_path}")

    all_validations: list[EncodingValidation] = []
    total_passed = 0
    total_failed = 0

    # Validate individual results
    for entry in results:
        if entry.get("status") != "success":
            continue

        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})
        result = entry.get("result", {})

        validation = validate_trainability_result(enc_name, params, result)
        all_validations.append(validation)
        total_passed += validation.pass_count
        total_failed += validation.fail_count

    # Global validation checks
    global_checks: list[ValidationResult] = []

    # Check depth-trainability correlation
    depth_checks = validate_depth_trainability_correlation(results)
    global_checks.extend(depth_checks)

    # Check width invariance of product-of-feature-state encodings
    global_checks.extend(validate_width_invariance(results))

    # Add global checks to a synthetic validation
    if global_checks:
        global_validation = EncodingValidation("_global_checks_", {}, global_checks)
        all_validations.append(global_validation)
        total_passed += global_validation.pass_count
        total_failed += global_validation.fail_count

    return total_passed, total_failed, all_validations


def print_validation_summary(
    total_passed: int,
    total_failed: int,
    validations: list[EncodingValidation],
    results_path: str,
) -> None:
    """Print validation summary to console."""

    print("\n" + "=" * 70)
    print("STAGE 5: TRAINABILITY ANALYSIS VALIDATION")
    print("=" * 70)

    # Summary statistics
    total_encodings = len([v for v in validations if not v.encoding_name.startswith("_")])
    encodings_passed = sum(1 for v in validations if v.all_passed and not v.encoding_name.startswith("_"))
    encodings_failed = total_encodings - encodings_passed

    print(f"\nSummary:")
    print(f"  Total encoding configurations: {total_encodings}")
    print(f"  Configurations passed: {encodings_passed}")
    print(f"  Configurations with failures: {encodings_failed}")
    print(f"  Total checks passed: {total_passed}")
    print(f"  Total checks failed: {total_failed}")

    # Print trainability statistics
    train_values: list[float] = []
    for v in validations:
        if v.encoding_name.startswith("_"):
            continue
        for r in v.results:
            if r.check_name == "trainability_in_range" and r.details:
                val = r.details.get("value")
                if val is not None:
                    train_values.append(val)

    if train_values:
        print(f"\nTrainability Statistics:")
        print(f"  Min: {min(train_values):.4f}")
        print(f"  Max: {max(train_values):.4f}")
        print(f"  Mean: {sum(train_values)/len(train_values):.4f}")

    # Print high-risk cases
    try:
        data = load_results(results_path)
        high_risk = collect_high_risk_cases(
            [r for r in data.get("results", []) if r.get("status") == "success"]
        )
        if high_risk:
            print(f"\nHigh Barren Plateau Risk Cases ({len(high_risk)}):")
            for case in high_risk[:5]:  # Top 5
                print(f"  - {case['encoding']} ({format_params(case['params'])})")
    except Exception:
        pass

    # Print failures
    if total_failed > 0:
        print("\n" + "-" * 70)
        print("FAILED CHECKS:")
        print("-" * 70)

        for validation in validations:
            if not validation.all_passed:
                display_name = validation.encoding_name if not validation.encoding_name.startswith("_") else "Global Checks"
                print(f"\n{display_name} ({format_params(validation.params)}):")
                for check in validation.results:
                    if not check.passed:
                        print(f"  [FAIL] {check.check_name}: {check.message}")

    # Print pass/fail summary
    print("\n" + "-" * 70)
    if total_failed == 0:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print(f"RESULT: {total_failed} CHECKS FAILED")
    print("-" * 70)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Stage 5 Trainability Analysis results",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to summary.json file (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated tables (default: same as results)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Run validation
        total_passed, total_failed, validations = run_validation(args.results_path)

        # Print summary
        print_validation_summary(total_passed, total_failed, validations, args.results_path)

        # Generate tables if we have results
        if validations:
            data = load_results(args.results_path)
            results = [r for r in data.get("results", []) if r.get("status") == "success"]

            # Determine output directory
            output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate Markdown table
            md_table = generate_markdown_table(results)
            md_path = output_dir / "stage5_trainability_table.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_table)
            print(f"\nGenerated Markdown table: {md_path}")

            # Generate LaTeX table
            latex_table = generate_latex_table(results)
            latex_path = output_dir / "stage5_trainability_table.tex"
            with open(latex_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
            print(f"Generated LaTeX table: {latex_path}")

        return 0 if total_failed == 0 else 1

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Stage 5 results not yet available. Run the experiment first:")
        print("  python -m experiments.run_stage --config experiments/configs/stage5_trainability.json")
        return 1
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
