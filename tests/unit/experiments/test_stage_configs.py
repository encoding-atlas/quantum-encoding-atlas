"""The tracked stage configurations measure the circuits the atlas reports.

The atlas attributes every per-encoding metric to the circuit Stages 6a/6b
benchmarked. That attribution is only possible if each analysis stage
actually measured that circuit. Two did not: Stage 5b measured
``AngleEncoding(reps=2)`` and Stage 2 ``DataReuploading(n_layers=3)``, while
the benchmark ran ``reps=1`` and ``n_layers=2``. These tests read the
committed configs (no pipeline output needed) so such a gap fails CI.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from experiments.config import load_config
from experiments.tradeoff import DEFAULT_N_FEATURES, WIDTH_PARAM, config_key

CONFIGS = Path("experiments/configs")

#: Stages whose per-encoding values the atlas reports.
ANALYSIS_STAGES = (
    "stage1_resources.json",
    "stage2_simulability.json",
    "stage3_expressibility.json",
    "stage4_entanglement.json",
    "stage5_trainability.json",
    "stage5b_noise.json",
)

pytestmark = pytest.mark.skipif(
    not CONFIGS.is_dir(), reason="experiments/configs not available"
)


def _benchmark_families() -> dict[str, str]:
    families: dict[str, set[str]] = {}
    for stage in ("stage6a_vqc.json", "stage6b_kernel.json"):
        for spec in load_config(str(CONFIGS / stage)).encoding_specs:
            families.setdefault(spec.name, set()).add(
                config_key(spec.name, spec.params, drop=(WIDTH_PARAM,))
            )
    assert all(len(keys) == 1 for keys in families.values()), families
    return {name: next(iter(keys)) for name, keys in families.items()}


def _headline_width(name: str) -> int:
    return 2 if name == "so2_equivariant" else DEFAULT_N_FEATURES


def test_vqc_and_kernel_benchmark_the_same_circuits() -> None:
    assert len(_benchmark_families()) == 16


@pytest.mark.parametrize("stage", ANALYSIS_STAGES)
def test_each_stage_measures_the_benchmarked_circuit(stage: str) -> None:
    families = _benchmark_families()
    specs = load_config(str(CONFIGS / stage)).encoding_specs
    measured: dict[str, set[str]] = {}
    for spec in specs:
        if spec.params.get(WIDTH_PARAM, DEFAULT_N_FEATURES) == _headline_width(
            spec.name
        ):
            measured.setdefault(spec.name, set()).add(
                config_key(spec.name, spec.params, drop=(WIDTH_PARAM,))
            )

    missing = sorted(
        name
        for name in {spec.name for spec in specs}
        if families.get(name) not in measured.get(name, set())
    )
    assert not missing, (
        f"{stage} lists these encodings but never measures their benchmarked "
        f"circuit at the headline width: {missing}"
    )


def test_stage5_measures_the_stage6a_classifier() -> None:
    """Trainability must describe the model whose accuracy is reported."""
    stage5 = json.loads(
        (CONFIGS / "stage5_trainability.json").read_text(encoding="utf-8")
    )
    stage6a = json.loads((CONFIGS / "stage6a_vqc.json").read_text(encoding="utf-8"))
    params = stage5["analysis_params"]
    assert (
        params["n_var_layers"]
        == stage6a["analysis_params"]["vqc_params"]["n_var_layers"]
    )
    assert params["observable"] == "pauli_z"  # the classifier reads out <Z_0>


def test_stage5_matches_the_estimator_defaults() -> None:
    """``profile_encoding``/``compare_to_atlas`` call the estimator with its
    defaults and claim to share the atlas's definition; that holds only while
    Stage 5 uses those same defaults."""
    from encoding_atlas.analysis import estimate_trainability

    defaults = {
        name: p.default
        for name, p in inspect.signature(estimate_trainability).parameters.items()
    }
    params = json.loads(
        (CONFIGS / "stage5_trainability.json").read_text(encoding="utf-8")
    )["analysis_params"]
    assert params["observable"] == defaults["observable"]
    assert params["n_var_layers"] == defaults["n_var_layers"]
    assert tuple(params["input_range"]) == pytest.approx(defaults["input_range"])
    assert params["n_samples"] == defaults["n_samples"]
