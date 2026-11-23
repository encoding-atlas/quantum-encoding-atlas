"""Trainability analysis for quantum encodings.

This module estimates how trainable a variational model built on an encoding
is, through the variance of its cost gradient at random initialisation — the
quantity whose exponential decay with qubit count defines a barren plateau
(McClean et al. 2018; Cerezo et al. 2021).

What is measured
----------------
An encoding ``U(x)`` has no trainable parameters of its own to differentiate:
its angles are the data. Trainability is a property of the *model* trained on
top of it, so the encoding is measured inside the model the atlas benchmarks —
the variational classifier of :class:`encoding_atlas.benchmark.VQCClassifier`:

.. math::

    C(x, \\theta) = \\langle 0 |\\, U(x)^\\dagger W(\\theta)^\\dagger \\, O \\,
                    W(\\theta) U(x) \\,| 0 \\rangle,

where ``W(θ)`` is ``n_var_layers`` layers of ``RY(θ[l, q])`` on every qubit
followed by a linear CNOT chain, and ``O`` is the read-out observable
(``Z`` on qubit 0 by default, the classifier's own read-out). The estimator
draws ``x`` uniformly from ``input_range`` and ``θ`` uniformly from
``[-π, π)`` — the classifier's initialisation — and reports

.. math::

    \\mathrm{Var}\\left[\\partial C / \\partial \\theta_k\\right]
    \\quad\\text{averaged over the live parameters } k.

Each ``θ_k`` enters exactly one ``RY`` gate, so the gradient is exact; it is
computed by the adjoint method in :mod:`encoding_atlas.analysis._variational`
and agrees with PennyLane's automatic differentiation of the classifier's
circuit to floating-point precision.

*Live* parameters are those inside the read-out's backward light cone; the
others have ``∂C/∂θ_k ≡ 0`` for every input. With the default ``Z₀`` read-out,
``θ[l, q]`` is live exactly when ``q ≤ n_var_layers - 1 - l`` — three
parameters for the benchmark's two layers, at any width. Averaging over all
``n_var_layers · n`` parameters instead would dilute the variance by
``3 / (2n)``: a decay with width that comes from the read-out, not the
encoding, and would read as a barren plateau. A corollary worth knowing: the
classifier's output, and so this estimate, depends on the encoded state only
through its first ``n_var_layers`` qubits.

This is the barren-plateau quantity of the literature, evaluated with the data
distribution averaged in. A data-dependent average matters: an encoding can
itself concentrate the states it produces, and a concentrated input state
flattens the landscape of any model trained on top of it.

.. note::

   Earlier releases differentiated the *data* ``x`` with a ``±π/2`` shift.
   That shift is exact only for a feature that enters a single unit-frequency
   rotation (plain angle encoding); for the other encodings it did not return
   a derivative at all, and a derivative with respect to data measures input
   sensitivity rather than trainability. See the changelog.

Observables
-----------
- ``"pauli_z"`` (default): ``Z`` on qubit 0. Local, and the benchmark
  classifier's read-out.
- ``"global_z"``: ``Z⊗Z⊗…⊗Z``. Global.
- ``"computational"``: the projector ``|0…0⟩⟨0…0|``. Global — its support is
  every qubit — so it exhibits cost-function-induced barren plateaus even for
  shallow circuits (Cerezo et al. 2021).

References
----------
.. [1] McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H.
       (2018). "Barren plateaus in quantum neural network training
       landscapes." Nature Communications, 9(1), 4812.

.. [2] Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. (2021).
       "Cost function dependent barren plateaus in shallow parametrized
       quantum circuits." Nature Communications, 12(1), 1791.

.. [3] Holmes, Z., Sharma, K., Cerezo, M., & Coles, P. J. (2022).
       "Connecting ansatz expressibility to gradient magnitudes and barren
       plateaus." PRX Quantum, 3(1), 010313.

.. [4] Jones, T., & Gacon, J. (2020). "Efficient calculation of gradients in
       classical simulations of variational quantum algorithms."
       arXiv:2009.02823.

Examples
--------
Basic trainability estimation:

>>> from encoding_atlas import AngleEncoding, IQPEncoding
>>> from encoding_atlas.analysis import estimate_trainability
>>>
>>> angle = estimate_trainability(AngleEncoding(n_features=4), n_samples=100, seed=42)
>>> iqp = estimate_trainability(IQPEncoding(n_features=4, reps=2), n_samples=100, seed=42)
>>> print(f"angle: {angle:.3f}  iqp: {iqp:.3f}")  # doctest: +SKIP

Detailed analysis with barren plateau risk:

>>> result = estimate_trainability(
...     IQPEncoding(n_features=4, reps=2),
...     n_samples=200,
...     seed=42,
...     return_details=True,
... )
>>> print(f"Gradient variance: {result['gradient_variance']:.2e}")  # doctest: +SKIP
>>> print(f"Barren plateau risk: {result['barren_plateau_risk']}")  # doctest: +SKIP

Notes
-----
The trainability score is a relative measure for comparing encodings, not a
guarantee of training success. Optimiser, learning rate, cost function and
noise all matter as well.

See Also
--------
compute_expressibility : Measure Hilbert space coverage.
compute_entanglement_capability : Measure entanglement generation.
count_resources : Count computational resources.
encoding_atlas.benchmark.VQCClassifier : The model whose gradients are measured.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Literal, TypedDict, cast, overload

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.analysis._ci import validate_ci_args
from encoding_atlas.analysis._parallel import ParallelArg, resolve_parallel_mode
from encoding_atlas.analysis._sampling import (
    SamplingMethod,
    generate_sample_batch,
    validate_sampling,
)
from encoding_atlas.analysis._utils import (
    create_rng,
    simulate_encoding_statevector,
    validate_encoding_for_analysis,
)
from encoding_atlas.analysis._variational import (
    VARIATIONAL_OBSERVABLES,
    live_parameter_mask,
    variational_gradient,
    variational_parameter_count,
)
from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.exceptions import (
    InsufficientSamplesError,
    NumericalInstabilityError,
    SimulationError,
)

# =============================================================================
# Module-Level Logger
# =============================================================================

_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "estimate_trainability",
    "compute_gradient_variance",
    "detect_barren_plateau",
    "TrainabilityResult",
]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Default number of random (x, θ) samples for gradient-variance estimation.
#
# Each sample costs one encoding simulation plus one adjoint sweep through the
# variational read-out (O(n_var_layers · n_qubits) gate applications on 2ⁿ
# amplitudes), so the sample count — not the parameter count — sets the cost.
# The standard error of a variance estimate is ≈ σ²·√(2/(n-1)), so 500
# samples resolve the variance to ~6%, ample for barren-plateau detection
# where orders of magnitude matter. Barren-plateau studies typically use
# 100-1000 samples (McClean et al., 2018; Cerezo et al., 2021).
_DEFAULT_N_SAMPLES: int = 500

# Warning threshold for sample count. 50 samples resolves a variance to ~20%,
# the minimum for a meaningful order-of-magnitude estimate.
_MIN_SAMPLES_WARNING: int = 50

# Error threshold for sample count. Below this, the variance estimate is
# statistically meaningless.
_MIN_SAMPLES_ERROR: int = 10

# Default data sampling range: a full rotation period, which is also the
# empirical pipeline's feature scaling.
_DEFAULT_INPUT_RANGE: tuple[float, float] = (0.0, 2.0 * np.pi)

# Default number of variational layers in the read-out. Matches
# ``VQCClassifier(n_var_layers=2)`` and the benchmark's Stage 6a configuration.
_DEFAULT_N_VAR_LAYERS: int = 2

# Variational angles are drawn from the classifier's own initialisation,
# ``rng.uniform(-π, π)`` (see ``VQCClassifier.fit``).
_THETA_RANGE: tuple[float, float] = (-np.pi, np.pi)

# Barren plateau detection thresholds.
# These thresholds are calibrated based on literature values and empirical
# testing. They should be adjusted based on the specific circuit architecture.
#
# HIGH RISK: Variance below this threshold indicates severe barren plateau.
# The circuit is likely untrainable with gradient-based optimization.
_BP_HIGH_RISK_THRESHOLD: float = 1e-6

# MEDIUM RISK: Variance below this threshold indicates potential trainability
# issues. Training may be slow or require careful hyperparameter tuning.
_BP_MEDIUM_RISK_THRESHOLD: float = 1e-3

# Reference variance for trainability scoring.
#
# MATHEMATICAL JUSTIFICATION:
# On one qubit, angle encoding ``RY(x)`` followed by the read-out
# ``RY(θ₁)…RY(θ_L)`` gives ``C = ⟨Z⟩ = cos(x + Σθ)``, so
# ``∂C/∂θ_k = -sin(x + Σθ)``. With θ uniform over a full period the argument
# is uniform too, and
#
#     Var[sin(φ)] = E[sin²(φ)] - E[sin(φ)]² = 1/2 - 0 = 0.5
#
# Multi-qubit circuits have lower variance: the CNOT chain spreads each
# parameter's influence over qubits a local read-out does not see.
#
# The value 0.1 is chosen as a conservative reference because:
# - It's below the single-parameter maximum (0.5)
# - It's representative of shallow, structured circuits (2-4 qubits, 1-2 layers)
# - It's well above the barren plateau regime (< 10⁻³)
#
# The trainability score sigmoid maps variance relative to this reference:
# - variance >> 0.1 → trainability → 1 (highly trainable)
# - variance ≈ 0.1 → trainability ≈ 0.5 (moderately trainable)
# - variance << 0.1 → trainability → 0 (poorly trainable)
_REFERENCE_VARIANCE: float = 0.1

# Numerical stability constant to prevent division by zero.
_EPSILON: float = 1e-15

# Maximum number of failed gradient computations before aborting.
# If more than this fraction of samples fail, the analysis is unreliable.
_MAX_FAILURE_FRACTION: float = 0.2

# Sigmoid steepness for variance-to-trainability mapping.
# Higher values create sharper transition between trainable/untrainable.
_SIGMOID_STEEPNESS: float = 2.0

# Default number of bootstrap resamples for the trainability /
# variance CIs. 200 stabilizes the percentile endpoints to ~1% jitter
# at negligible cost next to the per-sample gradient computation.
_DEFAULT_N_BOOTSTRAP_CI: int = 200

# Default two-sided confidence level for reported CIs.
_DEFAULT_CONFIDENCE_LEVEL: float = 0.95

_VALID_BACKENDS: frozenset[str] = frozenset({"pennylane", "qiskit", "cirq"})

TrainabilityObservable = Literal["pauli_z", "global_z", "computational"]


# =============================================================================
# Type Definitions
# =============================================================================


class TrainabilityResult(TypedDict):
    """Result of trainability estimation.

    This TypedDict defines the structure of the detailed result returned
    by :func:`estimate_trainability` when ``return_details=True``.

    Attributes
    ----------
    trainability_estimate : float
        Overall trainability score in the range [0, 1], where higher values
        indicate better trainability. This score is derived from the gradient
        variance using a logistic transformation.
    gradient_variance : float
        ``Var[∂C/∂θ_k]`` averaged over the live variational parameters (see
        ``live_parameter_mask``), computed over successful samples only. This
        is the primary metric for barren plateau detection. Values near zero
        indicate a barren plateau.
    barren_plateau_risk : {"low", "medium", "high"}
        Categorical assessment of barren plateau risk based on gradient
        variance and circuit size:

        - "low": Variance is healthy, training should proceed normally
        - "medium": Variance is borderline, training may be slow
        - "high": Variance is very low, likely barren plateau

    effective_dimension : float
        Estimated effective parameter dimension (HEURISTIC METRIC), computed
        as the count of parameters with variance exceeding 1% of the mean
        variance. This indicates how many parameters contribute meaningfully
        to the optimization landscape. Note: This is not a standard metric
        in barren plateau literature; it is provided for diagnostic purposes.
        For rigorous analysis, examine ``per_parameter_variance`` directly.
    n_samples : int
        Total number of random ``(x, θ)`` samples requested for the estimate.
    n_successful_samples : int
        Number of samples where gradient computation succeeded. This is the
        actual sample size used for variance computation. Equal to
        ``n_samples - n_failed_samples``.
    per_parameter_variance : NDArray[np.floating]
        Variance of each variational parameter's gradient, shape
        ``(n_var_layers * n_qubits,)``. Entry ``l * n_qubits + q`` belongs to
        ``θ[l, q]`` — the ``RY`` on qubit ``q`` in layer ``l``. Entries
        outside ``live_parameter_mask`` are zero up to round-off.
    n_failed_samples : int
        Number of samples where gradient computation failed. High values
        may indicate issues with the encoding or simulation backend.
        Failed samples are excluded from variance computation to ensure
        unbiased statistical estimates.
    trainability_ci_lower : float
        Lower bound of the percentile bootstrap CI on
        ``trainability_estimate`` at ``confidence_level``. Computed by
        resampling the per-sample gradient matrix and re-running the
        variance → trainability mapping on each resample.
    trainability_ci_upper : float
        Upper bound of the bootstrap CI on ``trainability_estimate``.
    gradient_variance_ci_lower : float
        Lower bound of the bootstrap CI on ``gradient_variance``.
    gradient_variance_ci_upper : float
        Upper bound of the bootstrap CI on ``gradient_variance``.
    confidence_level : float
        Two-sided confidence level used for the CI bounds above
        (default 0.95 → 95% CIs).
    sampling : {'uniform', 'sobol'}
        Sampling strategy that produced the per-sample ``(x, θ)`` draws. See
        :func:`estimate_trainability` for details.
    n_var_layers : int
        Number of variational layers in the read-out that was differentiated.
    observable : {"pauli_z", "global_z", "computational"}
        Read-out observable whose gradient was measured.
    n_live_parameters : int
        Number of parameters inside the read-out's light cone, over which
        ``gradient_variance`` is averaged.
    live_parameter_mask : NDArray[np.bool_]
        Shape ``(n_var_layers * n_qubits,)``; ``True`` where the gradient is
        not identically zero. Depends on the read-out only, never on the
        encoding.
    """

    trainability_estimate: float
    gradient_variance: float
    barren_plateau_risk: Literal["low", "medium", "high"]
    effective_dimension: float
    n_samples: int
    n_successful_samples: int
    per_parameter_variance: NDArray[np.floating[Any]]
    n_failed_samples: int
    trainability_ci_lower: float
    trainability_ci_upper: float
    gradient_variance_ci_lower: float
    gradient_variance_ci_upper: float
    confidence_level: float
    sampling: str
    n_var_layers: int
    observable: str
    n_live_parameters: int
    live_parameter_mask: NDArray[np.bool_]


# =============================================================================
# Process-pool worker plumbing (top-level for picklability)
# =============================================================================
#
# These globals are populated *once per worker process* by
# ``_trainability_worker_init`` so the encoding + observable label travel
# across the wire only once per worker. The worker function returns either
# the gradient array or a sentinel ``None`` for failed samples; the main
# process aggregates and packs successful gradients contiguously, matching
# the sequential implementation byte-for-byte.

_TRAIN_WORKER_ENCODING: BaseEncoding | None = None
_TRAIN_WORKER_BACKEND: str | None = None
_TRAIN_WORKER_OBSERVABLE: str | None = None


def _trainability_worker_init(
    encoding: BaseEncoding, backend: str, observable: str
) -> None:
    """ProcessPoolExecutor initializer — runs once per worker process."""
    global _TRAIN_WORKER_ENCODING, _TRAIN_WORKER_BACKEND, _TRAIN_WORKER_OBSERVABLE
    _TRAIN_WORKER_ENCODING = encoding
    _TRAIN_WORKER_BACKEND = backend
    _TRAIN_WORKER_OBSERVABLE = observable


def _trainability_worker_compute(
    x: NDArray[np.floating[Any]],
    theta: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]] | None, str | None]:
    """Worker entrypoint: compute gradients for one ``(x, θ)`` sample.

    Returns ``(gradients, None)`` on success or ``(None, error_message)``
    on a tolerated failure (``SimulationError`` /
    ``NumericalInstabilityError`` / unexpected ``Exception``). Sample-level
    failures must NOT raise out of the worker because that would tear down
    the executor; the sequential code path tolerates these failures and
    we preserve that behavior.
    """
    assert (
        _TRAIN_WORKER_ENCODING is not None
        and _TRAIN_WORKER_BACKEND is not None
        and _TRAIN_WORKER_OBSERVABLE is not None
    ), "Process pool worker invoked before initializer ran"
    return _compute_one_gradient_sample(
        _TRAIN_WORKER_ENCODING,
        x,
        theta,
        _TRAIN_WORKER_BACKEND,
        _TRAIN_WORKER_OBSERVABLE,
    )


def _compute_one_gradient_sample(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    theta: NDArray[np.floating[Any]],
    backend: str,
    observable: str,
) -> tuple[NDArray[np.floating[Any]] | None, str | None]:
    """Compute ``∂C/∂θ`` for one sample, tolerating expected failures.

    The encoded state ``U(x)|0⟩`` is simulated on ``backend``; the
    variational read-out and its exact gradient are then evaluated in NumPy.
    Every parallel mode calls this same function once per sample, which is
    what keeps their outputs bit-identical.

    Returns
    -------
    (gradients, None) on success.
    (None, error_message) on a tolerated failure — the caller is
    responsible for counting failures and bumping the appropriate logger.
    """
    try:
        state = simulate_encoding_statevector(
            encoding,
            x,
            backend=cast("Literal['pennylane', 'qiskit', 'cirq']", backend),
        )
        gradients = variational_gradient(
            state, theta, cast("TrainabilityObservable", observable)
        )
        return gradients, None
    except (SimulationError, NumericalInstabilityError) as e:
        return None, f"expected:{type(e).__name__}:{e}"
    except Exception as e:  # noqa: BLE001 - matches sequential tolerance
        return None, f"unexpected:{type(e).__name__}:{e}"


def _validate_n_var_layers(n_var_layers: object) -> int:
    """Return ``n_var_layers`` as an int, or raise ``ValueError``."""
    if (
        isinstance(n_var_layers, bool)
        or not isinstance(n_var_layers, (int, np.integer))
        or int(n_var_layers) < 1
    ):
        raise ValueError(
            f"n_var_layers must be a positive integer, got {n_var_layers!r}."
        )
    return int(n_var_layers)


# =============================================================================
# Main Public Function
# =============================================================================


@overload
def estimate_trainability(
    encoding: BaseEncoding,
    n_samples: int = ...,
    input_range: tuple[float, float] = ...,
    observable: TrainabilityObservable = ...,
    seed: int | None = ...,
    backend: Literal["pennylane", "qiskit", "cirq"] = ...,
    return_details: Literal[False] = ...,
    verbose: bool = ...,
    parallel: ParallelArg = ...,
    max_workers: int | None = ...,
    sampling: SamplingMethod = ...,
    confidence_level: float = ...,
    n_bootstrap_ci: int = ...,
    n_var_layers: int = ...,
) -> float: ...


@overload
def estimate_trainability(
    encoding: BaseEncoding,
    n_samples: int = ...,
    input_range: tuple[float, float] = ...,
    observable: TrainabilityObservable = ...,
    seed: int | None = ...,
    backend: Literal["pennylane", "qiskit", "cirq"] = ...,
    return_details: Literal[True] = ...,
    verbose: bool = ...,
    parallel: ParallelArg = ...,
    max_workers: int | None = ...,
    sampling: SamplingMethod = ...,
    confidence_level: float = ...,
    n_bootstrap_ci: int = ...,
    n_var_layers: int = ...,
) -> TrainabilityResult: ...


def estimate_trainability(
    encoding: BaseEncoding,
    n_samples: int = _DEFAULT_N_SAMPLES,
    input_range: tuple[float, float] = _DEFAULT_INPUT_RANGE,
    observable: TrainabilityObservable = "pauli_z",
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
    return_details: bool = False,
    verbose: bool = False,
    parallel: ParallelArg = False,
    max_workers: int | None = None,
    sampling: SamplingMethod = "uniform",
    confidence_level: float = _DEFAULT_CONFIDENCE_LEVEL,
    n_bootstrap_ci: int = _DEFAULT_N_BOOTSTRAP_CI,
    n_var_layers: int = _DEFAULT_N_VAR_LAYERS,
) -> float | TrainabilityResult:
    """Estimate the trainability of a variational model built on an encoding.

    Measures ``Var[∂C/∂θ_k]`` for the benchmark's variational classifier
    ``C(x, θ) = ⟨0| U(x)† W(θ)† O W(θ) U(x) |0⟩`` at random initialisation,
    with the data ``x`` drawn from ``input_range`` and the variational angles
    ``θ`` drawn from ``[-π, π)``, averaged over the parameters inside the
    read-out's light cone. Low variance — especially variance that decays
    exponentially with qubit count — indicates a barren plateau.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to analyze. Must be a valid encoding that
        implements the ``get_circuit()`` method.
    n_samples : int, default=500
        Number of random ``(x, θ)`` samples. Higher values give more accurate
        estimates but take longer to compute. Minimum value is 10,
        recommended minimum is 50.
    input_range : tuple[float, float], default=(0, 2π)
        Range from which each data feature is drawn, as ``(min, max)``. The
        default is a full rotation period, matching the benchmark's feature
        scaling.
    observable : {"pauli_z", "global_z", "computational"}, default="pauli_z"
        Read-out observable ``O``:

        - ``"pauli_z"``: ``Z`` on qubit 0 (local; the classifier's read-out)
        - ``"global_z"``: ``Z⊗Z⊗…⊗Z`` (global)
        - ``"computational"``: projector onto ``|0…0⟩`` (global)

        Global observables exhibit cost-function-induced barren plateaus even
        in shallow circuits; use them to study that effect specifically.
    seed : int, optional
        Random seed for reproducibility. If None, uses system entropy.
        For reproducible results, always specify a seed.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Backend that simulates the encoded state ``U(x)|0⟩``. The variational
        read-out and its gradient are evaluated exactly in NumPy, so every
        backend yields the same result up to floating-point precision.
    return_details : bool, default=False
        If True, return full result dictionary with detailed statistics.
        If False, return only the trainability estimate as a float.
    verbose : bool, default=False
        If True, log progress information during computation.
    parallel : bool or {'thread', 'process'}, default=False
        Parallel-dispatch mode for the per-sample computation (one encoding
        simulation plus one adjoint gradient sweep per sample).

        - ``False`` (default) — sequential, no executor overhead.
        - ``True`` or ``'thread'`` — :class:`ThreadPoolExecutor`.
        - ``'process'`` — :class:`ProcessPoolExecutor` with the encoding
          pickled once per worker. Workers exchange only NumPy float
          arrays, so process-pool parallelism works with **all** three
          backends here.

        Output is numerically identical across all modes for a fixed
        ``seed`` — the RNG is fully consumed in the main process before
        any work is dispatched. Failed samples are still counted (so the
        ``failure_fraction`` check produces the same result as
        sequential), and successful gradients are stored contiguously
        from index 0 just like the sequential version.
    max_workers : int or None, default=None
        Maximum number of workers when ``parallel`` is enabled.
    sampling : {'uniform', 'sobol'}, default='uniform'
        Strategy for drawing the per-sample ``(x, θ)`` pairs:

        - ``'uniform'`` (default): pseudo-random i.i.d. uniform draws.
        - ``'sobol'``: a scrambled Sobol' sequence over the joint
          ``(x, θ)`` hypercube, seeded from the same ``seed`` argument.
          Typically gives the same variance accuracy with fewer samples.
          For best statistical properties choose ``n_samples`` as a power
          of two.
    confidence_level : float, default=0.95
        Two-sided confidence level for the bootstrap CIs on
        ``trainability_estimate`` and ``gradient_variance`` reported
        in the result dict (only used when ``return_details=True``).
        Must lie strictly in ``(0, 1)``.
    n_bootstrap_ci : int, default=200
        Number of bootstrap resamples used to estimate the
        percentile CIs (only when ``return_details=True``). 200
        stabilizes the percentile endpoints to ~1% jitter at
        negligible cost.
    n_var_layers : int, default=2
        Number of variational layers ``L`` in the read-out ``W(θ)``. The
        default matches :class:`~encoding_atlas.benchmark.VQCClassifier`
        and the benchmark. The circuit has ``L * n_qubits`` parameters.

    Returns
    -------
    float or TrainabilityResult
        If ``return_details=False``: Trainability estimate in [0, 1],
        where higher values indicate better trainability.

        If ``return_details=True``: a :class:`TrainabilityResult` with the
        score, ``gradient_variance``, ``barren_plateau_risk``,
        ``effective_dimension``, sample counts, ``per_parameter_variance``
        of shape ``(n_var_layers * n_qubits,)``, bootstrap CIs, and the
        ``sampling``, ``n_var_layers`` and ``observable`` that produced them,
        and the ``live_parameter_mask`` the variance was averaged over.

    Raises
    ------
    InsufficientSamplesError
        If ``n_samples`` < 10.
    ValueError
        If ``input_range``, ``observable``, ``backend``, ``parallel``,
        ``sampling``, the CI arguments or ``n_var_layers`` are invalid.
    SimulationError
        If too many gradient computations fail (> 20% of samples).
    AnalysisError
        If the encoding is invalid or cannot be analyzed.

    Warns
    -----
    UserWarning
        If ``n_samples`` < 50, a warning is issued about potential
        statistical unreliability.

    Notes
    -----
    **Interpretation:**

    The trainability score should be interpreted as a relative measure
    for comparing encodings, not an absolute guarantee. A score of 0.8
    means the encoding appears more trainable than one with score 0.5,
    but actual training success depends on many other factors.

    An encoding's own internal parameters (for instance
    :class:`~encoding_atlas.TrainableEncoding`'s rotation angles) are held
    fixed: the encoding is treated as a feature map, exactly as the benchmark
    trains it.

    **Barren Plateau Risk Levels:**

    - "low" (variance ≥ 1e-3): Training should proceed normally
    - "medium" (1e-6 ≤ variance < 1e-3): May need careful tuning
    - "high" (variance < 1e-6): Likely barren plateau, consider:
      - Using fewer qubits
      - Shallower circuits
      - Local cost functions
      - Parameter initialization strategies

    Thresholds are relaxed for circuits wider than four qubits; see
    :func:`detect_barren_plateau`.

    **Computational Cost:**

    Each sample costs one simulation of the encoding plus one adjoint sweep
    through the read-out, ``O(n_var_layers · n_qubits · 2^n_qubits)``.

    **Failed Sample Handling:**

    Gradient computation may fail for certain inputs due to numerical
    instabilities or backend issues. Failed samples are completely excluded
    from variance computation to ensure unbiased statistical estimates. If
    more than 20% of samples fail, a :exc:`SimulationError` is raised. The
    number of successful samples used is reported in
    ``n_successful_samples`` when ``return_details=True``.

    **Best Practices:**

    1. Use at least 100 samples for reliable estimates
    2. Compare encodings using the same seed for fair comparison
    3. Keep the default local observable unless specifically studying
       global cost functions
    4. For publication, report both trainability score and gradient
       variance with confidence intervals

    Examples
    --------
    Basic usage:

    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> train = estimate_trainability(enc, seed=42)
    >>> 0.0 <= train <= 1.0
    True

    Detailed analysis:

    >>> result = estimate_trainability(
    ...     enc, n_samples=200, seed=42, return_details=True
    ... )
    >>> result["per_parameter_variance"].shape  # n_var_layers * n_qubits
    (8,)
    >>> result["n_live_parameters"]  # inside the Z₀ read-out's light cone
    3
    >>> result["barren_plateau_risk"]
    'low'

    See Also
    --------
    compute_gradient_variance : Compute only gradient variance.
    detect_barren_plateau : Detect barren plateau from variance.
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    validate_encoding_for_analysis(encoding)

    if n_samples < _MIN_SAMPLES_ERROR:
        raise InsufficientSamplesError(
            f"n_samples must be at least {_MIN_SAMPLES_ERROR}, got {n_samples}. "
            f"For reliable trainability estimates, use at least "
            f"{_MIN_SAMPLES_WARNING} samples.",
            requested_samples=n_samples,
            minimum_samples=_MIN_SAMPLES_ERROR,
            metric="trainability",
        )

    if n_samples < _MIN_SAMPLES_WARNING:
        warnings.warn(
            f"n_samples={n_samples} is below the recommended minimum of "
            f"{_MIN_SAMPLES_WARNING}. Trainability estimates may be unreliable. "
            f"Consider increasing n_samples for more accurate results.",
            UserWarning,
            stacklevel=2,
        )

    if input_range[0] >= input_range[1]:
        raise ValueError(
            f"input_range must satisfy min < max, got {input_range}. "
            f"Provide a valid range like (0.0, 2*np.pi)."
        )

    if observable not in VARIATIONAL_OBSERVABLES:
        raise ValueError(
            f"observable must be one of {sorted(VARIATIONAL_OBSERVABLES)}, "
            f"got {observable!r}."
        )

    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {sorted(_VALID_BACKENDS)}, got {backend!r}."
        )

    n_var_layers = _validate_n_var_layers(n_var_layers)

    # Validate parallel argument upfront for a clean ValueError on bad input.
    mode = resolve_parallel_mode(parallel)

    # Validate sampling / CI arguments for the same reason.
    validate_sampling(sampling)
    validate_ci_args(confidence_level, n_bootstrap_ci)

    # =========================================================================
    # Setup
    # =========================================================================
    rng = create_rng(seed)
    n_features = int(encoding.n_features)
    n_qubits = int(encoding.n_qubits)
    n_params = variational_parameter_count(n_qubits, n_var_layers)

    if verbose:
        _logger.info(
            "Estimating trainability for %s (n_qubits=%d, n_var_layers=%d, "
            "n_params=%d, n_samples=%d, observable=%r, backend=%r)",
            encoding.__class__.__name__,
            n_qubits,
            n_var_layers,
            n_params,
            n_samples,
            observable,
            backend,
        )

    # =========================================================================
    # Draw (x, θ) Samples
    # =========================================================================
    # One draw over the joint hypercube, so 'sobol' is low-discrepancy in
    # (x, θ) jointly, and the main process consumes the RNG completely before
    # any work is dispatched — which is what makes every parallel mode
    # reproduce the sequential result exactly.
    unit = generate_sample_batch(
        n_samples, n_features + n_params, (0.0, 1.0), rng, sampling
    )
    x_low, x_high = float(input_range[0]), float(input_range[1])
    t_low, t_high = _THETA_RANGE
    X_samples = x_low + unit[:, :n_features] * (x_high - x_low)
    theta_samples = (t_low + unit[:, n_features:] * (t_high - t_low)).reshape(
        n_samples, n_var_layers, n_qubits
    )

    # =========================================================================
    # Sample Gradients
    # =========================================================================
    # Successful samples are packed contiguously from index 0; failed samples
    # are not stored, so the variance uses valid data only.
    gradient_samples: NDArray[np.float64] = np.zeros(
        (n_samples, n_params), dtype=np.float64
    )
    n_successful = 0
    n_failed = 0

    # Log progress at 10% intervals
    log_interval = max(1, n_samples // 10)

    def _record(
        index: int,
        gradients: NDArray[np.floating[Any]] | None,
        error: str | None,
    ) -> None:
        nonlocal n_successful, n_failed
        if gradients is not None:
            gradient_samples[n_successful, :] = gradients
            n_successful += 1
            return
        n_failed += 1
        # Expected failures at DEBUG, unexpected at WARNING.
        assert error is not None
        if error.startswith("expected:"):
            _logger.debug(
                "Gradient computation failed at sample %d/%d: %s",
                index + 1,
                n_samples,
                error.split(":", 2)[2],
            )
        else:
            _logger.warning(
                "Unexpected error during gradient computation at sample %d: %s",
                index + 1,
                error.split(":", 2)[2],
            )

    if mode == "sequential" or n_samples <= 1:
        # Sequential path keeps the inline loop so the per-sample log
        # messages remain available with their sample indices intact.
        for i in range(n_samples):
            gradients, error = _compute_one_gradient_sample(
                encoding, X_samples[i], theta_samples[i], backend, observable
            )
            _record(i, gradients, error)
            if verbose and (i + 1) % log_interval == 0:
                _logger.info(
                    "Progress: %d/%d samples completed (%d successful, %d failed)",
                    i + 1,
                    n_samples,
                    n_successful,
                    n_failed,
                )
    else:
        # Parallel path. Results come back in input order thanks to
        # ``executor.map``, so packing successful gradients contiguously
        # from index 0 matches the sequential implementation exactly.
        if mode == "thread":
            with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
                results = list(
                    thread_pool.map(
                        lambda x, t: _compute_one_gradient_sample(
                            encoding, x, t, backend, observable
                        ),
                        X_samples,
                        theta_samples,
                    )
                )
        else:  # mode == "process"
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_trainability_worker_init,
                initargs=(encoding, backend, observable),
            ) as process_pool:
                results = list(
                    process_pool.map(
                        _trainability_worker_compute, X_samples, theta_samples
                    )
                )

        for i, (gradients, error) in enumerate(results):
            _record(i, gradients, error)

    # Check if too many samples failed
    failure_fraction = n_failed / n_samples
    if failure_fraction > _MAX_FAILURE_FRACTION:
        raise SimulationError(
            f"Too many gradient computations failed: {n_failed}/{n_samples} "
            f"({100*failure_fraction:.1f}%). This may indicate issues with the "
            f"encoding or the simulation backend. Try a different backend or "
            f"check if the encoding is valid.",
            backend=backend,
            details={
                "n_failed": n_failed,
                "n_samples": n_samples,
                "n_successful": n_successful,
                "failure_fraction": failure_fraction,
            },
        )

    # =========================================================================
    # Compute Statistics
    # =========================================================================
    # Validate we have enough successful samples for meaningful statistics
    if n_successful < _MIN_SAMPLES_ERROR:
        raise SimulationError(
            f"Only {n_successful} successful samples, need at least "
            f"{_MIN_SAMPLES_ERROR}. Gradient computation is failing too often.",
            backend=backend,
            details={"n_successful": n_successful, "n_failed": n_failed},
        )

    # Extract only the successful samples for variance computation.
    # This ensures unbiased estimation - failed samples are completely excluded.
    successful_gradients = gradient_samples[:n_successful, :]

    # Per-parameter variance across successful samples only.
    # Using ddof=1 for unbiased sample variance (Bessel's correction).
    per_param_variance = np.var(successful_gradients, axis=0, ddof=1)

    # Overall gradient variance: the mean over the parameters that can move
    # the cost. Parameters outside the read-out's light cone contribute
    # identically-zero gradients, and averaging them in would dilute the
    # variance by a width-dependent factor unrelated to the encoding.
    live = live_parameter_mask(n_qubits, n_var_layers, observable)
    n_live = int(np.count_nonzero(live))
    gradient_variance = float(np.mean(per_param_variance[live]))

    # =========================================================================
    # Effective Dimension Computation (Heuristic Metric)
    # =========================================================================
    #
    # DEFINITION:
    # Effective dimension counts how many parameters have "meaningful"
    # gradient variance, indicating they contribute to the optimization.
    #
    # THRESHOLD RATIONALE:
    # We use 1% of the mean variance as the significance threshold. This is
    # a HEURISTIC choice, not derived from theory, with the following logic:
    #
    # 1. Parameters with variance < 1% of mean contribute negligibly to
    #    the average gradient signal and may be effectively "frozen"
    # 2. The 1% threshold (two orders of magnitude) is commonly used in
    #    statistics for distinguishing signal from noise
    # 3. Using a relative threshold (vs. absolute) adapts to different
    #    circuit architectures and observable choices
    #
    # INTERPRETATION:
    # - effective_dimension ≈ n_params: All parameters are trainable
    # - effective_dimension << n_params: Many parameters are frozen/redundant
    # - effective_dimension = 0: Complete barren plateau (all variances tiny)
    #
    # NOTE: This metric is provided for diagnostic purposes only. It is not
    # a standard measure in the barren plateau literature and should be
    # interpreted with caution. For rigorous analysis, examine the full
    # per_parameter_variance distribution instead.
    #
    variance_threshold = max(gradient_variance * 0.01, _EPSILON)
    effective_dimension = float(np.sum(per_param_variance > variance_threshold))

    _logger.debug(
        "Gradient statistics: variance=%.2e, effective_dim=%.1f, "
        "n_successful=%d, n_failed=%d",
        gradient_variance,
        effective_dimension,
        n_successful,
        n_failed,
    )

    # =========================================================================
    # Barren Plateau Detection
    # =========================================================================
    barren_plateau_risk = detect_barren_plateau(
        gradient_variance=gradient_variance,
        n_qubits=n_qubits,
        n_params=n_live,
    )

    # =========================================================================
    # Compute Trainability Score
    # =========================================================================
    trainability_estimate = _variance_to_trainability(variance=gradient_variance)

    if verbose:
        _logger.info(
            "Trainability estimation complete: score=%.4f, variance=%.2e, "
            "risk=%s, effective_dim=%.1f, samples_used=%d/%d",
            trainability_estimate,
            gradient_variance,
            barren_plateau_risk,
            effective_dimension,
            n_successful,
            n_samples,
        )

    if not return_details:
        return trainability_estimate

    # =========================================================================
    # Bootstrap CIs (only for the detailed result path)
    # =========================================================================
    # We resample rows of ``successful_gradients`` with replacement and
    # re-run the same per-parameter-variance → mean → variance-to-
    # trainability chain that produced the point estimates. Doing the
    # bootstrap on the gradient matrix (rather than on the variance
    # scalar) properly propagates the variance estimator's own
    # non-linearity through the CI.
    indices = rng.integers(0, n_successful, size=(n_bootstrap_ci, n_successful))
    boot_variances: NDArray[np.float64] = np.empty(n_bootstrap_ci, dtype=np.float64)
    for b in range(n_bootstrap_ci):
        resample = successful_gradients[indices[b]][:, live]
        boot_variances[b] = float(np.mean(np.var(resample, axis=0, ddof=1)))
    boot_trainabilities = np.array(
        [_variance_to_trainability(variance=v) for v in boot_variances],
        dtype=np.float64,
    )
    alpha = 1.0 - float(confidence_level)
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    gradient_variance_ci_lower = float(np.percentile(boot_variances, lo_q))
    gradient_variance_ci_upper = float(np.percentile(boot_variances, hi_q))
    trainability_ci_lower = float(np.percentile(boot_trainabilities, lo_q))
    trainability_ci_upper = float(np.percentile(boot_trainabilities, hi_q))
    # Guard against tiny numerical inversions of (lower, upper).
    gradient_variance_ci_upper = max(
        gradient_variance_ci_upper, gradient_variance_ci_lower
    )
    trainability_ci_upper = max(trainability_ci_upper, trainability_ci_lower)

    # =========================================================================
    # Return Results
    # =========================================================================
    return TrainabilityResult(
        trainability_estimate=trainability_estimate,
        gradient_variance=gradient_variance,
        barren_plateau_risk=barren_plateau_risk,
        effective_dimension=effective_dimension,
        n_samples=n_samples,
        n_successful_samples=n_successful,
        per_parameter_variance=per_param_variance,
        n_failed_samples=n_failed,
        trainability_ci_lower=trainability_ci_lower,
        trainability_ci_upper=trainability_ci_upper,
        gradient_variance_ci_lower=gradient_variance_ci_lower,
        gradient_variance_ci_upper=gradient_variance_ci_upper,
        confidence_level=float(confidence_level),
        sampling=str(sampling),
        n_var_layers=n_var_layers,
        observable=str(observable),
        n_live_parameters=n_live,
        live_parameter_mask=live,
    )


# =============================================================================
# Supporting Public Functions
# =============================================================================


def compute_gradient_variance(
    encoding: BaseEncoding,
    n_samples: int = _DEFAULT_N_SAMPLES,
    input_range: tuple[float, float] = _DEFAULT_INPUT_RANGE,
    observable: TrainabilityObservable = "pauli_z",
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
    n_var_layers: int = _DEFAULT_N_VAR_LAYERS,
) -> float:
    """Compute ``Var[∂C/∂θ_k]`` averaged over the live variational parameters.

    This is the core metric used for trainability estimation: how much the
    cost gradient of the variational classifier built on ``encoding`` varies
    across random initialisations. Low variance indicates a flat
    optimization landscape (barren plateau).

    This function is a convenience wrapper that calls :func:`estimate_trainability`
    and extracts only the gradient variance.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    n_samples : int, default=500
        Number of random ``(x, θ)`` samples.
    input_range : tuple[float, float], default=(0, 2π)
        Range for sampling each data feature.
    observable : {"pauli_z", "global_z", "computational"}, default="pauli_z"
        Read-out observable; see :func:`estimate_trainability`.
    seed : int, optional
        Random seed for reproducibility.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Backend that simulates the encoded state.
    n_var_layers : int, default=2
        Number of variational layers in the read-out.

    Returns
    -------
    float
        Average gradient variance across parameters. Values close to zero
        indicate a barren plateau.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> variance = compute_gradient_variance(enc, n_samples=100, seed=42)
    >>> variance > 0
    True

    See Also
    --------
    estimate_trainability : Full trainability analysis with risk assessment.
    detect_barren_plateau : Categorize variance into risk levels.
    """
    result = estimate_trainability(
        encoding=encoding,
        n_samples=n_samples,
        input_range=input_range,
        observable=observable,
        seed=seed,
        backend=backend,
        return_details=True,
        n_var_layers=n_var_layers,
    )
    return result["gradient_variance"]


def detect_barren_plateau(
    gradient_variance: float,
    n_qubits: int,
    n_params: int,
) -> Literal["low", "medium", "high"]:
    """Detect barren plateau risk based on gradient variance.

    Compares observed gradient variance to theoretical scaling for
    barren plateaus (exponential decay with qubit count) and returns
    a categorical risk assessment.

    The thresholds are adjusted based on system size to account for
    the natural scaling of gradient variance with qubit count.

    Parameters
    ----------
    gradient_variance : float
        Observed gradient variance from gradient sampling. Must be
        non-negative.
    n_qubits : int
        Number of qubits in the circuit. Used to adjust thresholds
        based on expected scaling.
    n_params : int
        Number of parameters in the circuit. Currently used for
        logging but may be used for more sophisticated detection
        in future versions.

    Returns
    -------
    {"low", "medium", "high"}
        Barren plateau risk level:

        - "low": Variance is healthy, gradient-based training should work
        - "medium": Variance is borderline, may need careful tuning
        - "high": Variance is very low, likely barren plateau

    Notes
    -----
    **Risk Threshold Calibration:**

    The risk thresholds are calibrated based on literature values:

    - McClean et al. (2018) showed variance scales as O(1/2ⁿ) for
      random circuits with global cost functions
    - Cerezo et al. (2021) showed local cost functions have better
      scaling: O(1/n) for certain circuit architectures

    The thresholds are adjusted as follows:

    - Base high-risk threshold: 1e-6
    - Base medium-risk threshold: 1e-3
    - Size adjustment: thresholds scale with 2^(-(n-4)/2) above four qubits
      to account for natural variance decay

    **Practical Interpretation:**

    - "low" risk: Proceed with standard optimization
    - "medium" risk: Consider using:
      - Adaptive learning rates
      - Better parameter initialization (e.g., identity-initialized)
      - Noise injection during training
    - "high" risk: Fundamental changes needed:
      - Reduce circuit depth
      - Use local cost functions
      - Try hardware-efficient ansatze
      - Consider layer-wise training

    Examples
    --------
    >>> risk = detect_barren_plateau(
    ...     gradient_variance=1e-2,
    ...     n_qubits=4,
    ...     n_params=16
    ... )
    >>> print(f"Risk level: {risk}")
    Risk level: low

    >>> risk = detect_barren_plateau(
    ...     gradient_variance=1e-8,
    ...     n_qubits=10,
    ...     n_params=40
    ... )
    >>> print(f"Risk level: {risk}")
    Risk level: high

    See Also
    --------
    estimate_trainability : Complete trainability analysis.
    compute_gradient_variance : Compute gradient variance.
    """
    # Validate inputs
    if gradient_variance < 0:
        raise ValueError(
            f"gradient_variance must be non-negative, got {gradient_variance}"
        )
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be at least 1, got {n_qubits}")
    if n_params < 1:
        raise ValueError(f"n_params must be at least 1, got {n_params}")

    # =========================================================================
    # Threshold Scaling Based on System Size
    # =========================================================================
    #
    # THEORETICAL BACKGROUND:
    # Barren plateau severity depends on system size. McClean et al. (2018)
    # showed that for deep random circuits with global cost functions:
    #
    #     Var[∂⟨O⟩/∂θ] ∝ 2^(-n)  [exponential decay]
    #
    # However, Cerezo et al. (2021) demonstrated that local cost functions
    # have more favorable scaling:
    #
    #     Var[∂⟨O⟩/∂θ] ∝ 1/poly(n)  [polynomial decay]
    #
    # SCALING FORMULA DERIVATION:
    # We use a moderate scaling factor: 2^(-(n-4)/2) for n > 4
    #
    # This formula is chosen because:
    # 1. It interpolates between full exponential (2^(-n)) and no decay (1)
    # 2. The exponent -(n-4)/2 means:
    #    - n=4: factor = 1.0 (baseline, no adjustment)
    #    - n=6: factor = 2^(-1) = 0.5
    #    - n=8: factor = 2^(-2) = 0.25
    #    - n=10: factor = 2^(-3) = 0.125
    # 3. This ~O(2^(-n/2)) scaling matches the geometric mean of:
    #    - Worst case: O(2^(-n)) for random circuits [McClean et al.]
    #    - Best case: O(1) for very shallow/structured circuits
    # 4. The offset (n-4) ensures small circuits (≤4 qubits) use
    #    unmodified thresholds, as barren plateaus are less severe there.
    #
    # PRACTICAL EFFECT:
    # As system size increases, we lower our expectations for gradient
    # variance, making the detection more lenient for larger systems
    # where some natural variance decay is expected even for trainable circuits.
    #
    # References:
    # - McClean et al. (2018), Nature Communications 9(1), 4812
    # - Cerezo et al. (2021), Nature Communications 12(1), 1791
    #
    if n_qubits <= 4:  # noqa: SIM108
        # Small systems: use base thresholds without adjustment
        size_factor = 1.0
    else:
        # Larger systems: apply moderate exponential scaling
        # Formula: 2^(-(n-4)/2) gives ~O(2^(-n/2)) overall scaling
        size_factor = 2.0 ** (-(n_qubits - 4) / 2.0)

    high_threshold = _BP_HIGH_RISK_THRESHOLD * size_factor
    medium_threshold = _BP_MEDIUM_RISK_THRESHOLD * size_factor

    # Enforce minimum thresholds to prevent numerical issues.
    # These floors ensure we don't classify everything as "low risk"
    # for very large systems where size_factor becomes tiny.
    # - 1e-12 for high risk: ~40 qubits at full 2^(-n) scaling
    # - 1e-9 for medium risk: maintains 3 orders of magnitude gap
    high_threshold = max(high_threshold, 1e-12)
    medium_threshold = max(medium_threshold, 1e-9)

    _logger.debug(
        "Barren plateau detection: variance=%.2e, n_qubits=%d, "
        "high_thresh=%.2e, medium_thresh=%.2e",
        gradient_variance,
        n_qubits,
        high_threshold,
        medium_threshold,
    )

    if gradient_variance < high_threshold:
        return "high"
    elif gradient_variance < medium_threshold:
        return "medium"
    else:
        return "low"


# =============================================================================
# Private Helper Functions
# =============================================================================


def _variance_to_trainability(variance: float) -> float:
    """Convert gradient variance to trainability score in [0, 1].

    Uses a logistic (sigmoid) transformation on the log-ratio of variance
    to a reference value, mapping the infinite range of possible variances
    to a bounded [0, 1] score.

    Mathematical Formulation
    ------------------------
    The transformation is:

        r = log₁₀(σ² / σ²_ref)
        trainability = 1 / (1 + exp(-k · r))

    where:
    - σ² is the observed gradient variance
    - σ²_ref = 0.1 is the reference variance (see _REFERENCE_VARIANCE)
    - k = 2.0 is the sigmoid steepness (see _SIGMOID_STEEPNESS)

    Properties:
    - σ² = σ²_ref → r = 0 → trainability = 0.5
    - σ² >> σ²_ref → r >> 0 → trainability → 1
    - σ² << σ²_ref → r << 0 → trainability → 0

    The log-ratio ensures equal multiplicative changes in variance produce
    equal additive changes in the sigmoid input, making the score invariant
    to the scale of variance (only relative magnitude matters).

    Parameters
    ----------
    variance : float
        Gradient variance (must be non-negative).

    Returns
    -------
    float
        Trainability score in [0, 1].
    """
    # Handle edge case: zero or negative variance
    if variance <= 0:
        return 0.0

    # Reference variance for comparison
    reference = _REFERENCE_VARIANCE

    # Compute log ratio
    # Positive when variance > reference (good trainability)
    # Negative when variance < reference (poor trainability)
    log_ratio = np.log10(variance / reference)

    # Sigmoid transformation: maps (-∞, +∞) to (0, 1)
    # trainability = 1 / (1 + exp(-k * log_ratio))
    # Higher k = sharper transition
    trainability = 1.0 / (1.0 + np.exp(-_SIGMOID_STEEPNESS * log_ratio))

    # Clamp to [0, 1] for numerical safety
    trainability = float(np.clip(trainability, 0.0, 1.0))

    return trainability
