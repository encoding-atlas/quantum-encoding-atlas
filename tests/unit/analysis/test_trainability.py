"""Unit tests for ``encoding_atlas.analysis.trainability``.

``estimate_trainability`` measures ``Var[∂C/∂θ]`` for the benchmark's
variational classifier built on an encoding (encoding → ``n_var_layers``
layers of RY + a linear CNOT chain → ⟨Z₀⟩), averaged over the parameters
inside the read-out's light cone. These tests pin:

- **The estimand.** Closed forms (one qubit: ``Var = 1/2``), per-sample
  gradients equal PennyLane's automatic differentiation of the classifier,
  the variance is the mean over exactly the live parameters, and every
  backend gives the same answer.
- **Physics that the previous definition got wrong.** No encoding has an
  identically-zero variance; product-state encodings are width-invariant;
  a global observable produces the cost-function-dependent barren plateau.
- **The plumbing.** Sampling ranges, reproducibility, parallel modes,
  Sobol sampling, failure accounting, bootstrap CIs, validation, logging.

The engine itself (adjoint gradients, light cone) is tested in
``test_variational.py``.

References
----------
.. [1] McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
       network training landscapes." Nature Communications, 9(1), 4812.
.. [2] Cerezo, M., et al. (2021). "Cost function dependent barren plateaus in
       shallow parametrized quantum circuits." Nature Communications, 12, 1791.
"""

from __future__ import annotations

import logging
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from encoding_atlas import (
    AngleEncoding,
    BasisEncoding,
    HardwareEfficientEncoding,
    IQPEncoding,
    TrainableEncoding,
    ZZFeatureMap,
    get_encoding,
)
from encoding_atlas.analysis._utils import simulate_encoding_statevector
from encoding_atlas.analysis._variational import live_parameter_mask
from encoding_atlas.analysis.trainability import (
    _BP_HIGH_RISK_THRESHOLD,
    _BP_MEDIUM_RISK_THRESHOLD,
    _DEFAULT_N_SAMPLES,
    _DEFAULT_N_VAR_LAYERS,
    _EPSILON,
    _MAX_FAILURE_FRACTION,
    _MIN_SAMPLES_ERROR,
    _MIN_SAMPLES_WARNING,
    _REFERENCE_VARIANCE,
    _SIGMOID_STEEPNESS,
    _THETA_RANGE,
    _compute_one_gradient_sample,
    _variance_to_trainability,
    compute_gradient_variance,
    detect_barren_plateau,
    estimate_trainability,
)
from encoding_atlas.core.exceptions import (
    AnalysisError,
    InsufficientSamplesError,
    NumericalInstabilityError,
    SimulationError,
)

_MODULE = "encoding_atlas.analysis.trainability"

# Most tests use fewer than the recommended 50 samples to stay fast; the
# low-sample warning itself is asserted in TestValidation.
pytestmark = pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")

RESULT_KEYS = {
    "trainability_estimate",
    "gradient_variance",
    "barren_plateau_risk",
    "effective_dimension",
    "n_samples",
    "n_successful_samples",
    "per_parameter_variance",
    "n_failed_samples",
    "trainability_ci_lower",
    "trainability_ci_upper",
    "gradient_variance_ci_lower",
    "gradient_variance_ci_upper",
    "confidence_level",
    "sampling",
    "n_var_layers",
    "observable",
    "n_live_parameters",
    "live_parameter_mask",
}


def _record_samples(encoding, **kwargs):
    """Run the estimator while recording every (x, θ, gradient) it computes."""
    recorded: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    real = _compute_one_gradient_sample

    def spy(enc, x, theta, backend, observable):
        gradients, error = real(enc, x, theta, backend, observable)
        recorded.append((np.array(x), np.array(theta), np.array(gradients)))
        return gradients, error

    with patch(f"{_MODULE}._compute_one_gradient_sample", side_effect=spy):
        result = estimate_trainability(encoding, return_details=True, **kwargs)
    return result, recorded


# =============================================================================
# The estimand
# =============================================================================


class TestEstimand:
    """What ``gradient_variance`` is, checked against independent references."""

    def test_one_qubit_closed_form(self) -> None:
        """RY(x)·RY(θ₁)·RY(θ₂) gives ∂C/∂θ = -sin(x + θ₁ + θ₂): Var = 1/2."""
        result = estimate_trainability(
            AngleEncoding(n_features=1), n_samples=4000, seed=0, return_details=True
        )
        # Var[sin²φ] = 1/8, so the standard error at n=4000 is ~0.006.
        assert result["gradient_variance"] == pytest.approx(0.5, abs=0.03)
        assert result["n_live_parameters"] == 2  # both layers act on qubit 0

    def test_per_sample_gradients_equal_classifier_autodiff(self) -> None:
        """Each sample's gradient is PennyLane's gradient of VQCClassifier."""
        qml = pytest.importorskip("pennylane")
        from encoding_atlas.benchmark import VQCClassifier

        encoding = IQPEncoding(n_features=3, reps=2)
        clf = VQCClassifier(encoding, n_var_layers=2, epochs=1, seed=0)
        clf.fit(np.random.default_rng(0).uniform(0, 6, (4, 3)), np.array([0, 1, 0, 1]))

        _, recorded = _record_samples(encoding, n_samples=12, seed=3)
        assert len(recorded) == 12
        for x, theta, gradient in recorded:
            reference = qml.grad(lambda p, x=x: clf._qnode(x, p))(
                qml.numpy.array(theta, requires_grad=True)
            )
            assert_allclose(gradient, np.asarray(reference).ravel(), atol=1e-10)

    def test_variance_is_the_mean_over_live_parameters(self) -> None:
        result, recorded = _record_samples(
            HardwareEfficientEncoding(n_features=4, reps=2), n_samples=40, seed=1
        )
        gradients = np.stack([g for _, _, g in recorded])
        per_parameter = np.var(gradients, axis=0, ddof=1)
        mask = live_parameter_mask(4, 2, "pauli_z")

        assert_allclose(result["per_parameter_variance"], per_parameter, rtol=1e-12)
        assert np.array_equal(result["live_parameter_mask"], mask)
        assert result["gradient_variance"] == pytest.approx(
            float(np.mean(per_parameter[mask])), rel=1e-12
        )
        assert result["trainability_estimate"] == pytest.approx(
            _variance_to_trainability(result["gradient_variance"])
        )

    def test_samples_are_drawn_from_the_documented_ranges(self) -> None:
        _, recorded = _record_samples(
            AngleEncoding(n_features=3), n_samples=200, seed=5, input_range=(1.0, 2.0)
        )
        X = np.stack([x for x, _, _ in recorded])
        theta = np.stack([t for _, t, _ in recorded])
        assert X.shape == (200, 3) and theta.shape == (200, 2, 3)
        assert X.min() >= 1.0 and X.max() < 2.0
        assert theta.min() >= _THETA_RANGE[0] and theta.max() < _THETA_RANGE[1]
        # The draws cover the range rather than sitting at a constant.
        assert theta.std() > 1.5 and X.std() > 0.2

    def test_backends_agree_to_floating_point(self) -> None:
        """The backend only prepares U(x)|0⟩; the read-out is exact NumPy."""
        pytest.importorskip("qiskit")
        pytest.importorskip("cirq")
        values = [
            estimate_trainability(
                ZZFeatureMap(n_features=3, reps=2),
                n_samples=30,
                seed=4,
                backend=backend,
                return_details=True,
            )["gradient_variance"]
            for backend in ("pennylane", "qiskit", "cirq")
        ]
        assert_allclose(values, values[0], rtol=1e-10)

    def test_encoding_parameters_are_held_fixed(self) -> None:
        """TrainableEncoding's own angles are a feature map, as in the benchmark."""
        encoding = TrainableEncoding(n_features=3, n_layers=2, seed=0)
        result = estimate_trainability(
            encoding, n_samples=20, seed=0, return_details=True
        )
        assert result["per_parameter_variance"].shape == (2 * encoding.n_qubits,)


class TestLightCone:
    """Only parameters inside the observable's backward light cone count."""

    @pytest.mark.parametrize("n_features", [2, 4, 6])
    def test_pauli_z_has_three_live_parameters_at_any_width(
        self, n_features: int
    ) -> None:
        result = estimate_trainability(
            AngleEncoding(n_features=n_features),
            n_samples=20,
            seed=0,
            return_details=True,
        )
        assert result["n_live_parameters"] == 3
        dead = ~result["live_parameter_mask"]
        assert np.all(result["per_parameter_variance"][dead] < 1e-25)
        assert np.all(result["per_parameter_variance"][~dead] > 1e-3)

    def test_more_layers_widen_the_light_cone(self) -> None:
        result = estimate_trainability(
            AngleEncoding(n_features=4),
            n_samples=20,
            seed=0,
            n_var_layers=3,
            return_details=True,
        )
        assert result["per_parameter_variance"].shape == (12,)
        assert result["n_live_parameters"] == 3 + 2 + 1

    def test_product_state_encoding_is_width_invariant(self) -> None:
        """The read-out sees qubits {0, 1} only; for a product state of
        per-feature qubits that reduced state does not depend on the width.

        Averaging over all 2n parameters instead would halve the variance from
        n=4 to n=8 — a barren plateau that is not there.
        """
        variances = [
            estimate_trainability(
                AngleEncoding(n_features=n), n_samples=1500, seed=7, return_details=True
            )["gradient_variance"]
            for n in (2, 4, 8)
        ]
        assert max(variances) / min(variances) < 1.15

    def test_effective_dimension_counts_live_parameters_only(self) -> None:
        result = estimate_trainability(
            AngleEncoding(n_features=6), n_samples=50, seed=0, return_details=True
        )
        assert result["effective_dimension"] == 3.0


class TestPhysics:
    """Regressions against the previous input-derivative definition."""

    def test_basis_encoding_is_trainable(self) -> None:
        """A step function of x has zero x-derivative; ∂C/∂θ does not care."""
        result = estimate_trainability(
            BasisEncoding(n_features=4), n_samples=200, seed=0, return_details=True
        )
        assert result["gradient_variance"] > 0.1
        assert result["barren_plateau_risk"] == "low"

    def test_single_layer_diagonal_maps_are_not_flat(self) -> None:
        """IQP/ZZ at reps=1 make P(0…0) constant in x; their classifiers train."""
        for encoding in (
            IQPEncoding(n_features=4, reps=1),
            ZZFeatureMap(n_features=4, reps=1),
        ):
            result = estimate_trainability(
                encoding, n_samples=100, seed=0, return_details=True
            )
            assert result["gradient_variance"] > 1e-2

    def test_global_observable_shows_the_cost_function_barren_plateau(self) -> None:
        """Cerezo et al. 2021: a global cost flattens even a shallow circuit."""
        encoding = AngleEncoding(n_features=6)
        local = estimate_trainability(
            encoding, n_samples=200, seed=0, return_details=True
        )["gradient_variance"]
        projector = estimate_trainability(
            encoding,
            n_samples=200,
            seed=0,
            observable="computational",
            return_details=True,
        )["gradient_variance"]
        assert projector < local / 100

    def test_all_to_all_entangling_encoding_concentrates_with_width(self) -> None:
        """IQP entangles qubits 0 and 1 with every other qubit, so the reduced
        state the read-out sees approaches maximally mixed as the width grows:
        an encoding-induced barren plateau (~37x lower variance from 2 to 8)."""
        narrow = estimate_trainability(
            IQPEncoding(n_features=2, reps=2),
            n_samples=300,
            seed=0,
            return_details=True,
        )["gradient_variance"]
        wide = estimate_trainability(
            IQPEncoding(n_features=8, reps=2),
            n_samples=300,
            seed=0,
            return_details=True,
        )["gradient_variance"]
        assert wide < narrow / 10

    def test_deeper_hardware_efficient_encoding_is_less_trainable(self) -> None:
        shallow = estimate_trainability(
            HardwareEfficientEncoding(n_features=4, reps=1), n_samples=400, seed=0
        )
        deep = estimate_trainability(
            HardwareEfficientEncoding(n_features=4, reps=3), n_samples=400, seed=0
        )
        assert deep < shallow


# =============================================================================
# Return value, reproducibility, options
# =============================================================================


class TestReturnValue:
    def test_returns_float_by_default(self, sample_encoding_2q) -> None:
        value = estimate_trainability(sample_encoding_2q, n_samples=20, seed=0)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_detailed_result_keys_and_types(self, sample_encoding_2q) -> None:
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=0, return_details=True
        )
        assert set(result) == RESULT_KEYS
        assert result["n_samples"] == 20
        assert result["n_successful_samples"] == 20
        assert result["n_failed_samples"] == 0
        assert result["n_var_layers"] == _DEFAULT_N_VAR_LAYERS
        assert result["observable"] == "pauli_z"
        assert result["sampling"] == "uniform"
        assert result["barren_plateau_risk"] in {"low", "medium", "high"}
        assert result["per_parameter_variance"].dtype == np.float64
        assert result["live_parameter_mask"].dtype == bool

    def test_float_return_equals_detailed_estimate(self, sample_encoding_4q) -> None:
        value = estimate_trainability(sample_encoding_4q, n_samples=30, seed=2)
        result = estimate_trainability(
            sample_encoding_4q, n_samples=30, seed=2, return_details=True
        )
        assert value == result["trainability_estimate"]

    def test_confidence_intervals_bracket_the_estimates(
        self, sample_encoding_4q
    ) -> None:
        result = estimate_trainability(
            sample_encoding_4q, n_samples=100, seed=0, return_details=True
        )
        assert (
            result["gradient_variance_ci_lower"]
            <= result["gradient_variance"]
            <= result["gradient_variance_ci_upper"]
        )
        assert (
            result["trainability_ci_lower"]
            <= result["trainability_estimate"]
            <= result["trainability_ci_upper"]
        )

    def test_confidence_level_widens_the_interval(self, sample_encoding_4q) -> None:
        def width(level: float) -> float:
            r = estimate_trainability(
                sample_encoding_4q,
                n_samples=100,
                seed=0,
                return_details=True,
                confidence_level=level,
            )
            return r["gradient_variance_ci_upper"] - r["gradient_variance_ci_lower"]

        assert width(0.99) > width(0.5)


class TestReproducibility:
    def test_same_seed_same_result(self, entangling_encoding_4q) -> None:
        a = estimate_trainability(
            entangling_encoding_4q, n_samples=30, seed=11, return_details=True
        )
        b = estimate_trainability(
            entangling_encoding_4q, n_samples=30, seed=11, return_details=True
        )
        for key in RESULT_KEYS - {"per_parameter_variance", "live_parameter_mask"}:
            assert a[key] == b[key], key
        assert np.array_equal(a["per_parameter_variance"], b["per_parameter_variance"])

    def test_different_seeds_differ(self, entangling_encoding_4q) -> None:
        a = estimate_trainability(entangling_encoding_4q, n_samples=30, seed=1)
        b = estimate_trainability(entangling_encoding_4q, n_samples=30, seed=2)
        assert a != b

    def test_thread_mode_is_bit_identical(self, entangling_encoding_4q) -> None:
        seq = estimate_trainability(
            entangling_encoding_4q, n_samples=20, seed=3, return_details=True
        )
        par = estimate_trainability(
            entangling_encoding_4q,
            n_samples=20,
            seed=3,
            return_details=True,
            parallel="thread",
            max_workers=2,
        )
        assert np.array_equal(
            seq["per_parameter_variance"], par["per_parameter_variance"]
        )
        assert seq["gradient_variance"] == par["gradient_variance"]


class TestOptions:
    def test_sobol_sampling_is_reproducible_and_distinct(
        self, sample_encoding_4q
    ) -> None:
        a = estimate_trainability(
            sample_encoding_4q, n_samples=64, seed=0, sampling="sobol"
        )
        b = estimate_trainability(
            sample_encoding_4q, n_samples=64, seed=0, sampling="sobol"
        )
        c = estimate_trainability(sample_encoding_4q, n_samples=64, seed=0)
        assert a == b
        assert a != c

    def test_sobol_matches_the_closed_form(self) -> None:
        result = estimate_trainability(
            AngleEncoding(n_features=1),
            n_samples=1024,
            seed=0,
            sampling="sobol",
            return_details=True,
        )
        assert result["gradient_variance"] == pytest.approx(0.5, abs=0.02)
        assert result["sampling"] == "sobol"

    @pytest.mark.parametrize("observable", ["pauli_z", "global_z", "computational"])
    def test_observable_is_recorded_and_changes_the_result(
        self, observable: str
    ) -> None:
        encoding = HardwareEfficientEncoding(n_features=3, reps=1)
        result = estimate_trainability(
            encoding, n_samples=30, seed=0, observable=observable, return_details=True
        )
        assert result["observable"] == observable
        assert np.array_equal(
            result["live_parameter_mask"], live_parameter_mask(3, 2, observable)
        )

    def test_observables_give_different_variances(self) -> None:
        encoding = HardwareEfficientEncoding(n_features=3, reps=1)
        values = {
            obs: estimate_trainability(encoding, n_samples=30, seed=0, observable=obs)
            for obs in ("pauli_z", "global_z", "computational")
        }
        assert len(set(values.values())) == 3

    def test_n_var_layers_accepts_numpy_integers(self, sample_encoding_2q) -> None:
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=0,
            n_var_layers=np.int64(3),
            return_details=True,
        )
        assert result["n_var_layers"] == 3
        assert isinstance(result["n_var_layers"], int)

    def test_input_range_changes_the_result(self) -> None:
        encoding = IQPEncoding(n_features=3, reps=2)
        full = estimate_trainability(encoding, n_samples=50, seed=0)
        narrow = estimate_trainability(
            encoding, n_samples=50, seed=0, input_range=(0.0, 0.1)
        )
        assert full != narrow

    def test_negative_input_range_is_allowed(self, sample_encoding_2q) -> None:
        value = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=0, input_range=(-np.pi, 0.0)
        )
        assert 0.0 <= value <= 1.0


class TestComputeGradientVariance:
    def test_matches_the_detailed_result(self, sample_encoding_4q) -> None:
        variance = compute_gradient_variance(sample_encoding_4q, n_samples=30, seed=1)
        result = estimate_trainability(
            sample_encoding_4q, n_samples=30, seed=1, return_details=True
        )
        assert variance == result["gradient_variance"]

    def test_forwards_every_option(self, sample_encoding_4q) -> None:
        kwargs = dict(
            n_samples=25,
            seed=2,
            observable="global_z",
            input_range=(0.0, 1.0),
            n_var_layers=3,
        )
        variance = compute_gradient_variance(sample_encoding_4q, **kwargs)
        result = estimate_trainability(
            sample_encoding_4q, return_details=True, **kwargs
        )
        assert variance == result["gradient_variance"]

    def test_invalid_n_var_layers_propagates(self, sample_encoding_2q) -> None:
        with pytest.raises(ValueError, match="n_var_layers"):
            compute_gradient_variance(sample_encoding_2q, n_samples=20, n_var_layers=0)


# =============================================================================
# Validation
# =============================================================================


class TestValidation:
    def test_invalid_encoding_raises(self) -> None:
        with pytest.raises((AnalysisError, TypeError, AttributeError)):
            estimate_trainability("not an encoding", n_samples=20)  # type: ignore[arg-type]

    @pytest.mark.parametrize("n_samples", [-1, 0, 9])
    def test_too_few_samples_raise(self, sample_encoding_2q, n_samples: int) -> None:
        with pytest.raises(InsufficientSamplesError) as info:
            estimate_trainability(sample_encoding_2q, n_samples=n_samples)
        assert str(_MIN_SAMPLES_ERROR) in str(info.value)

    def test_minimum_samples_warn_but_run(self, sample_encoding_2q) -> None:
        with pytest.warns(UserWarning, match="below the recommended minimum"):
            value = estimate_trainability(
                sample_encoding_2q, n_samples=_MIN_SAMPLES_ERROR, seed=0
            )
        assert 0.0 <= value <= 1.0

    def test_no_warning_at_recommended_minimum(self, sample_encoding_2q) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            estimate_trainability(
                sample_encoding_2q, n_samples=_MIN_SAMPLES_WARNING, seed=0
            )

    @pytest.mark.parametrize("input_range", [(1.0, 1.0), (2.0, 1.0)])
    def test_invalid_input_range(self, sample_encoding_2q, input_range) -> None:
        with pytest.raises(ValueError, match="input_range"):
            estimate_trainability(
                sample_encoding_2q, n_samples=20, input_range=input_range
            )

    @pytest.mark.parametrize("observable", ["local_z", "Z", ""])
    def test_invalid_observable_lists_the_options(
        self, sample_encoding_2q, observable
    ) -> None:
        with pytest.raises(ValueError, match="pauli_z"):
            estimate_trainability(
                sample_encoding_2q, n_samples=20, observable=observable
            )

    def test_invalid_backend(self, sample_encoding_2q) -> None:
        with pytest.raises(ValueError, match="backend"):
            estimate_trainability(sample_encoding_2q, n_samples=20, backend="braket")  # type: ignore[arg-type]

    @pytest.mark.parametrize("n_var_layers", [0, -1, 1.5, True, "2", None])
    def test_invalid_n_var_layers(self, sample_encoding_2q, n_var_layers) -> None:
        with pytest.raises(ValueError, match="n_var_layers"):
            estimate_trainability(
                sample_encoding_2q, n_samples=20, n_var_layers=n_var_layers
            )

    def test_invalid_parallel_sampling_and_ci_arguments(
        self, sample_encoding_2q
    ) -> None:
        with pytest.raises(ValueError, match="parallel"):
            estimate_trainability(sample_encoding_2q, n_samples=20, parallel="gpu")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="sampling"):
            estimate_trainability(sample_encoding_2q, n_samples=20, sampling="lhs")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            estimate_trainability(
                sample_encoding_2q, n_samples=20, confidence_level=1.0
            )

    def test_validation_happens_before_any_simulation(self, sample_encoding_2q) -> None:
        with (
            patch(f"{_MODULE}._compute_one_gradient_sample") as worker,
            pytest.raises(ValueError),
        ):
            estimate_trainability(sample_encoding_2q, n_samples=20, n_var_layers=0)
        worker.assert_not_called()


# =============================================================================
# Per-sample worker and failure accounting
# =============================================================================


class TestPerSampleWorker:
    def test_success_returns_the_exact_gradient(self) -> None:
        encoding = IQPEncoding(n_features=3, reps=2)
        rng = np.random.default_rng(0)
        x, theta = rng.uniform(0, 6, 3), rng.uniform(-np.pi, np.pi, (2, 3))
        gradients, error = _compute_one_gradient_sample(
            encoding, x, theta, "pennylane", "pauli_z"
        )
        assert error is None
        from encoding_atlas.analysis._variational import variational_gradient

        expected = variational_gradient(
            simulate_encoding_statevector(encoding, x), theta
        )
        assert np.array_equal(gradients, expected)

    @pytest.mark.parametrize(
        "exception, prefix",
        [
            (SimulationError("boom"), "expected:SimulationError"),
            (NumericalInstabilityError("nan"), "expected:NumericalInstabilityError"),
            (RuntimeError("odd"), "unexpected:RuntimeError"),
        ],
    )
    def test_failures_are_reported_not_raised(self, exception, prefix) -> None:
        with patch(f"{_MODULE}.simulate_encoding_statevector", side_effect=exception):
            gradients, error = _compute_one_gradient_sample(
                AngleEncoding(n_features=2),
                np.zeros(2),
                np.zeros((2, 2)),
                "pennylane",
                "pauli_z",
            )
        assert gradients is None
        assert error is not None and error.startswith(prefix)


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestFailureAccounting:
    @staticmethod
    def _failing_after(n_failures: int, exception: Exception):
        """Simulator stub: the first ``n_failures`` calls raise ``exception``."""
        calls = {"n": 0}

        def stub(encoding, x, backend="pennylane"):
            calls["n"] += 1
            if calls["n"] <= n_failures:
                raise exception
            return simulate_encoding_statevector(encoding, x, backend=backend)

        return stub

    @pytest.mark.parametrize(
        "exception",
        [SimulationError("sim"), NumericalInstabilityError("nan"), ValueError("bug")],
    )
    def test_failed_samples_are_counted_and_excluded(self, exception) -> None:
        stub = self._failing_after(3, exception)
        with patch(f"{_MODULE}.simulate_encoding_statevector", side_effect=stub):
            result = estimate_trainability(
                AngleEncoding(n_features=2), n_samples=20, seed=0, return_details=True
            )
        assert result["n_failed_samples"] == 3
        assert result["n_successful_samples"] == 17
        assert np.isfinite(result["gradient_variance"])

    def test_exactly_the_tolerated_fraction_is_allowed(self) -> None:
        n_samples = 20
        allowed = int(_MAX_FAILURE_FRACTION * n_samples)
        stub = self._failing_after(allowed, SimulationError("sim"))
        with patch(f"{_MODULE}.simulate_encoding_statevector", side_effect=stub):
            result = estimate_trainability(
                AngleEncoding(n_features=2),
                n_samples=n_samples,
                seed=0,
                return_details=True,
            )
        assert result["n_failed_samples"] == allowed

    def test_too_many_failures_raise_with_details(self) -> None:
        stub = self._failing_after(5, SimulationError("sim"))
        with (
            patch(f"{_MODULE}.simulate_encoding_statevector", side_effect=stub),
            pytest.raises(
                SimulationError, match="Too many gradient computations failed"
            ) as info,
        ):
            estimate_trainability(AngleEncoding(n_features=2), n_samples=20, seed=0)
        assert info.value.details["n_failed"] == 5
        assert info.value.details["n_samples"] == 20

    def test_too_few_successes_raise_even_within_the_failure_budget(self) -> None:
        """2 of 11 failed is 18% (tolerated), but 9 successes is below 10."""
        stub = self._failing_after(2, SimulationError("sim"))
        with (
            patch(f"{_MODULE}.simulate_encoding_statevector", side_effect=stub),
            pytest.raises(SimulationError, match="Only 9 successful samples") as info,
        ):
            estimate_trainability(AngleEncoding(n_features=2), n_samples=11, seed=0)
        assert info.value.details == {"n_successful": 9, "n_failed": 2}

    def test_non_finite_state_is_a_failed_sample(self) -> None:
        """A backend that silently returns NaN amplitudes must not poison the variance."""
        calls = {"n": 0}

        def stub(encoding, x, backend="pennylane"):
            calls["n"] += 1
            state = simulate_encoding_statevector(encoding, x, backend=backend)
            return np.full_like(state, np.nan) if calls["n"] == 1 else state

        with patch(f"{_MODULE}.simulate_encoding_statevector", side_effect=stub):
            result = estimate_trainability(
                AngleEncoding(n_features=2), n_samples=20, seed=0, return_details=True
            )
        assert result["n_failed_samples"] == 1
        assert np.isfinite(result["gradient_variance"])

    def test_unexpected_failures_log_a_warning(self, caplog) -> None:
        stub = self._failing_after(1, ValueError("bug"))
        with (
            patch(f"{_MODULE}.simulate_encoding_statevector", side_effect=stub),
            caplog.at_level(logging.WARNING, logger=_MODULE),
        ):
            estimate_trainability(AngleEncoding(n_features=2), n_samples=20, seed=0)
        assert any("Unexpected error" in r.getMessage() for r in caplog.records)


class TestVerboseLogging:
    def test_verbose_logs_setup_progress_and_summary(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger=_MODULE):
            estimate_trainability(
                AngleEncoding(n_features=2), n_samples=20, seed=0, verbose=True
            )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "n_var_layers=2" in messages
        assert "Progress: 20/20" in messages
        assert "Trainability estimation complete" in messages

    def test_quiet_by_default(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger=_MODULE):
            estimate_trainability(AngleEncoding(n_features=2), n_samples=20, seed=0)
        assert not [r for r in caplog.records if r.levelno >= logging.INFO]


# =============================================================================
# Unchanged helpers: detect_barren_plateau, _variance_to_trainability, constants
# =============================================================================


class TestDetectBarrenPlateau:
    """Tests for detect_barren_plateau function."""

    def test_low_risk_for_high_variance(self) -> None:
        """High variance gives low risk."""
        risk = detect_barren_plateau(gradient_variance=0.1, n_qubits=4, n_params=16)
        assert risk == "low"

    def test_high_risk_for_low_variance(self) -> None:
        """Very low variance gives high risk."""
        risk = detect_barren_plateau(gradient_variance=1e-10, n_qubits=4, n_params=16)
        assert risk == "high"

    def test_medium_risk_for_borderline_variance(self) -> None:
        """Borderline variance gives medium risk.

        For n_qubits=4 (size_factor=1.0):
        - high_threshold = 1e-6
        - medium_threshold = 1e-3
        Variance 1e-5 is between them, so risk is 'medium'.
        """
        risk = detect_barren_plateau(gradient_variance=1e-5, n_qubits=4, n_params=16)
        assert risk == "medium"

    def test_scaling_with_qubits(self) -> None:
        """Thresholds scale with qubit count."""
        risk_small = detect_barren_plateau(
            gradient_variance=1e-5, n_qubits=2, n_params=8
        )
        risk_large = detect_barren_plateau(
            gradient_variance=1e-5, n_qubits=8, n_params=32
        )
        # Both should be valid risk levels
        assert risk_small in {"low", "medium", "high"}
        assert risk_large in {"low", "medium", "high"}

    def test_invalid_variance_raises(self) -> None:
        """Negative variance raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            detect_barren_plateau(gradient_variance=-1.0, n_qubits=4, n_params=16)

    def test_invalid_n_qubits_raises(self) -> None:
        """Invalid n_qubits raises ValueError."""
        with pytest.raises(ValueError, match="n_qubits"):
            detect_barren_plateau(gradient_variance=0.1, n_qubits=0, n_params=16)

    def test_invalid_n_params_raises(self) -> None:
        """Invalid n_params raises ValueError."""
        with pytest.raises(ValueError, match="n_params"):
            detect_barren_plateau(gradient_variance=0.1, n_qubits=4, n_params=0)

    def test_zero_variance_high_risk(self) -> None:
        """Zero variance gives high risk."""
        risk = detect_barren_plateau(gradient_variance=0.0, n_qubits=4, n_params=16)
        assert risk == "high"

    def test_returns_valid_literal(self) -> None:
        """Return value is always a valid literal for all input combinations."""
        for variance in [0.0, 1e-10, 1e-5, 1e-3, 0.01, 0.1, 1.0]:
            for n_qubits in [1, 2, 4, 8]:
                risk = detect_barren_plateau(
                    gradient_variance=variance,
                    n_qubits=n_qubits,
                    n_params=n_qubits * 4,
                )
                assert risk in {"low", "medium", "high"}

    def test_exact_high_threshold_boundary(self) -> None:
        """Variance exactly at high_threshold is medium, not high.

        The condition is 'variance < high_threshold' for high risk.
        At the boundary (variance == threshold), it falls through to medium.
        For n_qubits=4: high_threshold = 1e-6.
        """
        risk = detect_barren_plateau(
            gradient_variance=_BP_HIGH_RISK_THRESHOLD,
            n_qubits=4,
            n_params=16,
        )
        assert risk == "medium"

    def test_exact_medium_threshold_boundary(self) -> None:
        """Variance exactly at medium_threshold is low, not medium.

        The condition is 'variance < medium_threshold' for medium risk.
        At the boundary (variance == threshold), it falls through to low.
        For n_qubits=4: medium_threshold = 1e-3.
        """
        risk = detect_barren_plateau(
            gradient_variance=_BP_MEDIUM_RISK_THRESHOLD,
            n_qubits=4,
            n_params=16,
        )
        assert risk == "low"

    def test_threshold_floor_large_system(self) -> None:
        """Threshold floors prevent everything becoming 'low' for large systems.

        For n_qubits=100, the size_factor would make thresholds ~0, but
        the floor enforces: high_threshold >= 1e-12, medium_threshold >= 1e-9.
        """
        # Variance 1e-13 should still be classified as high risk
        risk_high = detect_barren_plateau(
            gradient_variance=1e-13, n_qubits=100, n_params=400
        )
        assert risk_high == "high"

        # Variance 1e-10 should be medium (between 1e-12 and 1e-9 floors)
        risk_med = detect_barren_plateau(
            gradient_variance=1e-10, n_qubits=100, n_params=400
        )
        assert risk_med == "medium"

        # Variance 1e-8 should be low (above 1e-9 floor)
        risk_low = detect_barren_plateau(
            gradient_variance=1e-8, n_qubits=100, n_params=400
        )
        assert risk_low == "low"

    def test_monotonicity_with_increasing_variance(self) -> None:
        """Risk severity never increases as variance increases.

        For a sequence of increasing variances, the risk transitions
        from high → medium → low and never goes backwards.
        """
        severity_map = {"high": 3, "medium": 2, "low": 1}
        variances = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0]
        n_qubits = 4
        n_params = 16

        prev_severity = 4  # Start higher than any level
        for var in variances:
            risk = detect_barren_plateau(var, n_qubits, n_params)
            severity = severity_map[risk]
            assert severity <= prev_severity, (
                f"Monotonicity violated: variance {var} has risk {risk} "
                f"(severity {severity}) which is more severe than previous "
                f"(severity {prev_severity})"
            )
            prev_severity = severity

    def test_single_qubit_system(self) -> None:
        """Single-qubit system uses base thresholds (n_qubits <= 4)."""
        risk = detect_barren_plateau(gradient_variance=0.01, n_qubits=1, n_params=1)
        assert risk == "low"

        risk = detect_barren_plateau(gradient_variance=1e-8, n_qubits=1, n_params=1)
        assert risk == "high"


class TestDetectBarrenPlateauScaling:
    """Extended tests for threshold scaling in detect_barren_plateau."""

    def test_size_factor_at_4_qubits(self) -> None:
        """At n_qubits=4, size_factor=1.0 (base thresholds unchanged)."""
        risk = detect_barren_plateau(gradient_variance=5e-7, n_qubits=4, n_params=16)
        assert risk == "high"

    def test_size_factor_at_6_qubits(self) -> None:
        """At n_qubits=6: size_factor=2^(-1)=0.5.

        high_threshold = 1e-6 * 0.5 = 5e-7
        medium_threshold = 1e-3 * 0.5 = 5e-4
        """
        # Variance 3e-7: below 5e-7 -> high
        assert detect_barren_plateau(3e-7, 6, 24) == "high"
        # Variance 1e-5: between 5e-7 and 5e-4 -> medium
        assert detect_barren_plateau(1e-5, 6, 24) == "medium"
        # Variance 1e-3: above 5e-4 -> low
        assert detect_barren_plateau(1e-3, 6, 24) == "low"

    def test_size_factor_at_8_qubits(self) -> None:
        """At n_qubits=8: size_factor=2^(-2)=0.25.

        high_threshold = 1e-6 * 0.25 = 2.5e-7
        medium_threshold = 1e-3 * 0.25 = 2.5e-4
        """
        assert detect_barren_plateau(1e-7, 8, 32) == "high"
        assert detect_barren_plateau(1e-5, 8, 32) == "medium"
        assert detect_barren_plateau(1e-2, 8, 32) == "low"

    def test_threshold_floors_activated(self) -> None:
        """For very large qubit counts, floors prevent near-zero thresholds.

        At n_qubits=50: size_factor = 2^(-23) ~ 1.2e-7
        Without floors: high_threshold ~ 1.2e-13 (below 1e-12 floor)
        With floors: high_threshold = 1e-12, medium_threshold = 1e-9
        """
        assert detect_barren_plateau(5e-13, 50, 200) == "high"
        assert detect_barren_plateau(5e-11, 50, 200) == "medium"
        assert detect_barren_plateau(1e-8, 50, 200) == "low"


class TestVarianceToTrainability:
    """Tests for the _variance_to_trainability sigmoid mapping.

    This function maps gradient variance to a [0, 1] trainability score
    using a logistic transformation on the log-ratio to a reference variance.
    """

    def test_zero_variance_returns_zero(self) -> None:
        """Zero variance maps to trainability 0.0."""
        assert _variance_to_trainability(0.0) == 0.0

    def test_negative_variance_returns_zero(self) -> None:
        """Negative variance maps to trainability 0.0 (edge guard)."""
        assert _variance_to_trainability(-1.0) == 0.0
        assert _variance_to_trainability(-1e-15) == 0.0

    def test_reference_variance_gives_half(self) -> None:
        """Variance equal to reference gives exactly 0.5.

        At variance = _REFERENCE_VARIANCE:
          log_ratio = log10(ref/ref) = 0
          sigmoid(0) = 1/(1+exp(0)) = 0.5
        """
        score = _variance_to_trainability(_REFERENCE_VARIANCE)
        assert_allclose(score, 0.5, atol=1e-10)

    def test_large_variance_approaches_one(self) -> None:
        """Very large variance gives score close to 1.0."""
        score = _variance_to_trainability(100.0)
        assert score > 0.99
        assert score <= 1.0

    def test_small_variance_approaches_zero(self) -> None:
        """Very small variance gives score close to 0.0."""
        score = _variance_to_trainability(1e-10)
        assert score < 0.01
        assert score >= 0.0

    def test_monotonically_increasing(self) -> None:
        """Larger variance always produces higher trainability score."""
        variances = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
        scores = [_variance_to_trainability(v) for v in variances]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Monotonicity violated: variance {variances[i]} "
                f"(score {scores[i]}) >= variance {variances[i+1]} "
                f"(score {scores[i+1]})"
            )

    def test_output_always_in_unit_interval(self) -> None:
        """Output is always in [0, 1] for any non-negative input."""
        test_values = [0.0, 1e-300, 1e-15, 1e-10, 1e-5, 0.001, 0.1, 1.0, 1e5, 1e100]
        for v in test_values:
            score = _variance_to_trainability(v)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for variance {v}"


class TestVarianceToTrainabilityExtended:
    """Extended tests for _variance_to_trainability sigmoid mapping."""

    def test_very_tiny_positive_variance(self) -> None:
        """Very small positive variance gives score near 0."""
        score = _variance_to_trainability(1e-300)
        assert 0.0 <= score <= 1.0
        assert score < 0.01

    def test_exact_sigmoid_formula(self) -> None:
        """Verify exact sigmoid formula matches implementation.

        For variance v: score = 1 / (1 + exp(-k * log10(v / ref)))
        """
        variance = 0.05
        expected_log_ratio = np.log10(variance / _REFERENCE_VARIANCE)
        expected_score = 1.0 / (1.0 + np.exp(-_SIGMOID_STEEPNESS * expected_log_ratio))
        actual = _variance_to_trainability(variance)
        assert_allclose(actual, expected_score, atol=1e-12)

    def test_symmetry_around_reference(self) -> None:
        """Score at 10*ref and ref/10 are symmetric around 0.5.

        If v1 = ref * k and v2 = ref / k, then:
        log_ratio(v1) = log10(k) and log_ratio(v2) = -log10(k)
        So: score(v1) + score(v2) = 1.0 (sigmoid symmetry)
        """
        k = 10.0
        v_high = _REFERENCE_VARIANCE * k
        v_low = _REFERENCE_VARIANCE / k
        score_high = _variance_to_trainability(v_high)
        score_low = _variance_to_trainability(v_low)
        assert_allclose(score_high + score_low, 1.0, atol=1e-10)

    def test_double_reference_variance(self) -> None:
        """Variance = 2 * reference gives score > 0.5."""
        score = _variance_to_trainability(2.0 * _REFERENCE_VARIANCE)
        assert score > 0.5

    def test_half_reference_variance(self) -> None:
        """Variance = 0.5 * reference gives score < 0.5."""
        score = _variance_to_trainability(0.5 * _REFERENCE_VARIANCE)
        assert score < 0.5


class TestModuleConstants:
    """Tests that module-level constants have expected values and types.

    These tests serve as guardrails against accidental changes to
    calibration constants that would silently alter behavior.
    """

    def test_default_n_samples(self) -> None:
        """Default sample count is 500."""
        assert _DEFAULT_N_SAMPLES == 500

    def test_min_samples_warning(self) -> None:
        """Warning threshold is 50."""
        assert _MIN_SAMPLES_WARNING == 50

    def test_min_samples_error(self) -> None:
        """Error threshold is 10."""
        assert _MIN_SAMPLES_ERROR == 10

    def test_bp_high_risk_threshold(self) -> None:
        """High risk threshold is 1e-6."""
        assert _BP_HIGH_RISK_THRESHOLD == 1e-6

    def test_bp_medium_risk_threshold(self) -> None:
        """Medium risk threshold is 1e-3."""
        assert _BP_MEDIUM_RISK_THRESHOLD == 1e-3

    def test_reference_variance(self) -> None:
        """Reference variance is 0.1."""
        assert _REFERENCE_VARIANCE == 0.1

    def test_sigmoid_steepness(self) -> None:
        """Sigmoid steepness is 2.0."""
        assert _SIGMOID_STEEPNESS == 2.0

    def test_epsilon(self) -> None:
        """Epsilon is 1e-15."""
        assert _EPSILON == 1e-15

    def test_max_failure_fraction(self) -> None:
        """Max failure fraction is 0.2."""
        assert _MAX_FAILURE_FRACTION == 0.2

    def test_threshold_ordering(self) -> None:
        """High risk threshold < medium risk threshold."""
        assert _BP_HIGH_RISK_THRESHOLD < _BP_MEDIUM_RISK_THRESHOLD

    def test_sample_thresholds_ordering(self) -> None:
        """Error threshold < warning threshold < default."""
        assert _MIN_SAMPLES_ERROR < _MIN_SAMPLES_WARNING < _DEFAULT_N_SAMPLES


class TestPublicAPI:
    """Tests that the module's public API is correctly defined."""

    def test_all_exports(self) -> None:
        """__all__ contains the expected public symbols."""
        from encoding_atlas.analysis import trainability

        assert "estimate_trainability" in trainability.__all__
        assert "compute_gradient_variance" in trainability.__all__
        assert "detect_barren_plateau" in trainability.__all__
        assert "TrainabilityResult" in trainability.__all__

    def test_all_has_exactly_four_entries(self) -> None:
        """__all__ has exactly 4 entries (no accidental additions)."""
        from encoding_atlas.analysis import trainability

        assert len(trainability.__all__) == 4

    def test_all_exports_resolvable(self) -> None:
        """Every name in __all__ is actually defined in the module."""
        from encoding_atlas.analysis import trainability

        for name in trainability.__all__:
            assert hasattr(
                trainability, name
            ), f"{name!r} is in __all__ but not defined in the module"

    def test_trainability_result_in_analysis_init(self) -> None:
        """TrainabilityResult is re-exported from analysis __init__."""
        from encoding_atlas.analysis import TrainabilityResult

        assert TrainabilityResult is not None

    def test_estimate_trainability_in_analysis_init(self) -> None:
        """estimate_trainability is re-exported from analysis __init__."""
        from encoding_atlas.analysis import estimate_trainability as et

        assert callable(et)

    def test_detect_barren_plateau_in_analysis_init(self) -> None:
        """detect_barren_plateau is re-exported from analysis __init__."""
        from encoding_atlas.analysis import detect_barren_plateau as dbp

        assert callable(dbp)

    def test_compute_gradient_variance_in_analysis_init(self) -> None:
        """compute_gradient_variance is re-exported from analysis __init__."""
        from encoding_atlas.analysis import compute_gradient_variance as cgv

        assert callable(cgv)


class TestNewModuleConstants:
    """Constants introduced with the classifier-gradient definition."""

    def test_default_n_var_layers_matches_the_benchmark_classifier(self) -> None:
        import inspect

        from encoding_atlas.benchmark import VQCClassifier

        default = inspect.signature(VQCClassifier).parameters["n_var_layers"].default
        assert _DEFAULT_N_VAR_LAYERS == default == 2

    def test_theta_range_is_the_classifier_initialisation(self) -> None:
        low, high = _THETA_RANGE
        assert (low, high) == (-np.pi, np.pi)
