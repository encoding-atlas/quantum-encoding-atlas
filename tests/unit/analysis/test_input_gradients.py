"""Correctness of ``compute_parameter_gradient`` / ``compute_all_parameter_gradients``.

Both functions are documented to return ``∂⟨O⟩/∂xᵢ``, the derivative of an
observable of the encoded state with respect to an input feature. They
previously applied the two-term parameter-shift rule to the features. That
rule is exact only for a parameter entering a single ``exp(−iθG/2)`` gate, so
for most encodings it returned no derivative at all — and the only existing
check used ``AngleEncoding(n_features=1)``, the one case where it happens to
be exact.

These tests check the value, not just the shape:

- analytic derivatives (angle encoding, including large angles);
- every benchmarked encoding against an *independent* reference — sixth-order
  Richardson extrapolation of central differences at a different step — which
  converges to the true derivative by a different route;
- an explicit regression showing the old shift rule is far from the truth.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from encoding_atlas import AngleEncoding, BasisEncoding, IQPEncoding, get_encoding
from encoding_atlas.analysis import (
    compute_all_parameter_gradients,
    compute_parameter_gradient,
)
from encoding_atlas.analysis._utils import (
    _expectations_batch,
    simulate_encoding_statevectors_batch,
)
from encoding_atlas.core.exceptions import NumericalInstabilityError
from encoding_atlas.guide._candidates import BENCHMARK_PARAMS

OBSERVABLES = ("computational", "global_z", "local_z")


def _observable(encoding, X: np.ndarray, observable: str) -> np.ndarray:
    states = simulate_encoding_statevectors_batch(encoding, np.atleast_2d(X))
    return _expectations_batch(states, observable, encoding.n_qubits)  # type: ignore[arg-type]


def _richardson_derivative(encoding, x: np.ndarray, i: int, observable: str) -> float:
    """Sixth-order Richardson extrapolation of central differences at h = 4e-3.

    Independent of the function under test: a different stencil and a step
    forty times larger, so the two agree only if both converge to the true
    derivative.
    """

    def central(h: float) -> float:
        e = np.zeros_like(x)
        e[i] = h
        values = _observable(encoding, np.stack([x - e, x + e]), observable)
        return float(values[1] - values[0]) / (2 * h)

    h = 4e-3
    d1, d2, d3 = central(h), central(h / 2), central(h / 4)
    r1, r2 = (4 * d2 - d1) / 3, (4 * d3 - d2) / 3
    return (16 * r2 - r1) / 15


def _benchmark_cases() -> list:
    even_only = {"swap_equivariant", "symmetry_inspired"}
    cases = []
    for name, params in BENCHMARK_PARAMS.items():
        if name == "basis":
            continue  # a step function: covered separately below
        n_features = 2 if name == "so2_equivariant" else 4 if name in even_only else 3
        cases.append(pytest.param(name, {"n_features": n_features, **params}, id=name))
    return cases


class TestAgainstAnalyticDerivatives:
    @pytest.mark.parametrize("rotation", ["X", "Y"])
    @pytest.mark.parametrize("scale", [1.0, 30.0])
    def test_angle_encoding_computational_observable(
        self, rotation: str, scale: float
    ) -> None:
        """P(0…0) = Πⱼ cos²(xⱼ/2), so ∂/∂xᵢ = −½ sin(xᵢ) Π_{j≠i} cos²(xⱼ/2)."""
        encoding = AngleEncoding(n_features=3, rotation=rotation)
        rng = np.random.default_rng(0)
        for _ in range(10):
            x = rng.uniform(-scale, scale, 3)
            c2 = np.cos(x / 2) ** 2
            expected = [
                -0.5 * np.sin(x[i]) * np.prod(np.delete(c2, i)) for i in range(3)
            ]
            assert_allclose(
                compute_all_parameter_gradients(encoding, x), expected, atol=1e-10
            )

    def test_angle_encoding_local_z(self) -> None:
        """⟨Z₀⟩ = cos(x₀) depends on the first feature only."""
        encoding = AngleEncoding(n_features=3)
        x = np.array([0.4, 1.3, -2.2])
        assert_allclose(
            compute_all_parameter_gradients(encoding, x, observable="local_z"),
            [-np.sin(0.4), 0.0, 0.0],
            atol=1e-10,
        )

    def test_rz_rotation_leaves_probabilities_flat(self) -> None:
        """RZ only adds phases to |0⟩: every probability is constant in x."""
        encoding = AngleEncoding(n_features=2, rotation="Z")
        assert_allclose(
            compute_all_parameter_gradients(encoding, np.array([0.7, 2.1])),
            0.0,
            atol=1e-10,
        )


class TestAgainstIndependentReference:
    @pytest.mark.parametrize("name, params", _benchmark_cases())
    @pytest.mark.parametrize("observable", OBSERVABLES)
    def test_every_benchmarked_encoding(
        self, name: str, params: dict, observable: str
    ) -> None:
        encoding = get_encoding(name, **params)
        rng = np.random.default_rng(1)
        x = rng.uniform(0.3, 2 * np.pi - 0.3, encoding.n_features)
        expected = [
            _richardson_derivative(encoding, x, i, observable)
            for i in range(encoding.n_features)
        ]
        assert_allclose(
            compute_all_parameter_gradients(encoding, x, observable=observable),
            expected,
            atol=1e-8,
        )

    def test_the_old_shift_rule_was_not_a_derivative(self) -> None:
        """IQP's features enter RZ(2x) and (π−xᵢ)(π−xⱼ): a ±π/2 shift is wrong."""
        encoding = IQPEncoding(n_features=4, reps=2)
        x = np.array([0.9, 2.3, 4.1, 5.0])
        true = compute_all_parameter_gradients(encoding, x)
        shifted = np.empty(4)
        for i in range(4):
            e = np.zeros(4)
            e[i] = np.pi / 2
            values = _observable(encoding, np.stack([x + e, x - e]), "computational")
            shifted[i] = (values[0] - values[1]) / 2
        reference = [
            _richardson_derivative(encoding, x, i, "computational") for i in range(4)
        ]
        assert_allclose(true, reference, atol=1e-8)
        assert np.max(np.abs(shifted - true)) > 0.1


class TestNonDifferentiableEncodings:
    def test_basis_encoding_is_flat_away_from_its_threshold(self) -> None:
        encoding = BasisEncoding(n_features=3)  # threshold 0.5
        x = np.array([0.1, 0.9, 2.0])
        assert_allclose(compute_all_parameter_gradients(encoding, x), 0.0, atol=1e-12)


class TestContract:
    @pytest.mark.parametrize("observable", OBSERVABLES)
    def test_batched_is_bit_identical_to_per_call(self, observable: str) -> None:
        encoding = IQPEncoding(n_features=3, reps=2)
        x = np.array([0.4, 1.1, 1.9])
        batched = compute_all_parameter_gradients(encoding, x, observable=observable)
        per_call = [
            compute_parameter_gradient(encoding, x, i, observable=observable)
            for i in range(3)
        ]
        assert np.array_equal(batched, np.array(per_call))

    def test_input_is_not_mutated(self) -> None:
        x = np.array([0.3, 0.7])
        original = x.copy()
        compute_all_parameter_gradients(AngleEncoding(n_features=2), x)
        compute_parameter_gradient(AngleEncoding(n_features=2), x, 1)
        assert np.array_equal(x, original)

    def test_empty_input_returns_empty_gradient(self) -> None:
        grads = compute_all_parameter_gradients(
            AngleEncoding(n_features=1), np.zeros(0)
        )
        assert grads.shape == (0,) and grads.dtype == np.float64

    def test_non_finite_expectations_raise(self) -> None:
        def nan_states(encoding, X, backend="pennylane"):
            return np.full((len(X), 2**encoding.n_qubits), np.nan, dtype=complex)

        from unittest.mock import patch

        with (
            patch(
                "encoding_atlas.analysis._utils.simulate_encoding_statevectors_batch",
                side_effect=nan_states,
            ),
            pytest.raises(NumericalInstabilityError) as info,
        ):
            compute_all_parameter_gradients(
                AngleEncoding(n_features=2), np.array([0.1, 0.2])
            )
        assert info.value.details["param_index"] == 0
        assert info.value.details["step"] > 0
