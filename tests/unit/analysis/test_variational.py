"""Tests for the exact variational read-out engine (``analysis._variational``).

The trainability axis is only as correct as this engine, so it is checked
against three independent references:

1. **PennyLane automatic differentiation** of the identical circuit, for every
   registry encoding, several depths, and all three observables.
2. **The benchmark's own models.** The circuit must be exactly the one
   :class:`encoding_atlas.benchmark.VQCClassifier` (and the pipeline's
   ``experiments/vqc.py``) trains; if either model's ansatz changes, these
   tests fail rather than letting trainability silently describe a different
   model.
3. **Closed forms**: the two-term parameter-shift rule (exact here, because
   every angle enters one ``RY`` gate), analytic one-qubit values, and the
   analytic light cone of the ``Z₀`` read-out.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from encoding_atlas import get_encoding
from encoding_atlas.analysis._utils import simulate_encoding_statevector
from encoding_atlas.analysis._variational import (
    VARIATIONAL_OBSERVABLES,
    apply_variational_circuit,
    live_parameter_mask,
    variational_expectation,
    variational_gradient,
    variational_parameter_count,
)
from encoding_atlas.core.exceptions import NumericalInstabilityError
from encoding_atlas.guide._candidates import BENCHMARK_PARAMS

qml = pytest.importorskip("pennylane")
pnp = qml.numpy

# Floating-point agreement between two exact computations of the same
# quantity (adjoint vs autodiff vs shift rule): a few ULPs of O(1) values.
EXACT = 1e-12


def _random_state(rng: np.random.Generator, n_qubits: int) -> np.ndarray:
    state = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return state / np.linalg.norm(state)


def _pennylane_reference(state: np.ndarray, n_var_layers: int, observable: str):
    """A PennyLane QNode for W(θ)|state⟩ with the requested read-out."""
    n_qubits = int(np.log2(state.size))
    dev = qml.device("default.qubit", wires=n_qubits)

    def measurement():
        if observable == "pauli_z":
            return qml.expval(qml.PauliZ(0))
        if observable == "global_z":
            op = qml.PauliZ(0)
            for q in range(1, n_qubits):
                op = op @ qml.PauliZ(q)
            return qml.expval(op)
        return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(params):
        qml.StatePrep(state, wires=range(n_qubits))
        for layer in range(n_var_layers):
            for q in range(n_qubits):
                qml.RY(params[layer, q], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return measurement()

    return circuit


def _shift_rule_gradient(state, theta, observable):
    """Two-term parameter-shift gradient — exact for single-RY parameters."""
    flat = theta.reshape(-1)
    grad = np.empty_like(flat)
    for k in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[k] += np.pi / 2
        minus[k] -= np.pi / 2
        grad[k] = (
            variational_expectation(state, plus.reshape(theta.shape), observable)
            - variational_expectation(state, minus.reshape(theta.shape), observable)
        ) / 2
    return grad


# =============================================================================
# Against PennyLane, for every encoding the atlas benchmarks
# =============================================================================


def _registry_cases() -> list:
    # An odd width where the encoding allows one, so the CNOT chain and the
    # light cone are exercised off the symmetric case.
    even_only = {"swap_equivariant", "symmetry_inspired"}
    cases = []
    for name, params in BENCHMARK_PARAMS.items():
        n_features = 2 if name == "so2_equivariant" else 4 if name in even_only else 3
        cases.append(pytest.param(name, {"n_features": n_features, **params}, id=name))
    return cases


class TestAgainstPennyLane:
    """Engine output equals PennyLane on the encoding's own simulated state."""

    @pytest.mark.parametrize("name, params", _registry_cases())
    @pytest.mark.parametrize("observable", VARIATIONAL_OBSERVABLES)
    def test_expectation_and_gradient_match_autodiff(
        self, name: str, params: dict, observable: str
    ) -> None:
        rng = np.random.default_rng(7)
        encoding = get_encoding(name, **params)
        x = rng.uniform(0, 2 * np.pi, encoding.n_features)
        state = simulate_encoding_statevector(encoding, x)
        for n_var_layers in (1, 2, 3):
            theta = rng.uniform(-np.pi, np.pi, (n_var_layers, encoding.n_qubits))
            circuit = _pennylane_reference(state, n_var_layers, observable)
            ref_value = float(circuit(pnp.array(theta, requires_grad=True)))
            ref_grad = np.asarray(
                qml.grad(circuit)(pnp.array(theta, requires_grad=True))
            ).reshape(-1)

            assert variational_expectation(state, theta, observable) == pytest.approx(
                ref_value, abs=EXACT
            )
            assert_allclose(
                variational_gradient(state, theta, observable), ref_grad, atol=EXACT
            )

    def test_forward_state_matches_pennylane_state(self) -> None:
        rng = np.random.default_rng(1)
        state = _random_state(rng, 3)
        theta = rng.uniform(-np.pi, np.pi, (2, 3))
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=range(3))
            for layer in range(2):
                for q in range(3):
                    qml.RY(theta[layer, q], wires=q)
                for q in range(2):
                    qml.CNOT(wires=[q, q + 1])
            return qml.state()

        assert_allclose(apply_variational_circuit(state, theta), circuit(), atol=EXACT)


# =============================================================================
# Against the benchmark's own classifiers
# =============================================================================


class TestMatchesTheBenchmarkClassifier:
    """The read-out must be the circuit the benchmark trains, gate for gate."""

    @pytest.mark.parametrize(
        "name, params",
        [
            ("angle", {"n_features": 4}),
            ("iqp", {"n_features": 3, "reps": 2}),
            ("data_reuploading", {"n_features": 3, "n_layers": 2}),
            ("amplitude", {"n_features": 4}),
        ],
    )
    @pytest.mark.parametrize("n_var_layers", [1, 2, 3])
    def test_packaged_vqc_classifier(
        self, name: str, params: dict, n_var_layers: int
    ) -> None:
        from encoding_atlas.benchmark import VQCClassifier

        rng = np.random.default_rng(3)
        encoding = get_encoding(name, **params)
        X = rng.uniform(0, 2 * np.pi, (6, encoding.n_features))
        y = np.array([0, 1, 0, 1, 0, 1])
        clf = VQCClassifier(encoding, n_var_layers=n_var_layers, epochs=1, seed=0)
        clf.fit(X, y)

        for x in rng.uniform(0, 2 * np.pi, (3, encoding.n_features)):
            expected = float(clf._qnode(x, clf.params_))
            state = simulate_encoding_statevector(encoding, x)
            assert variational_expectation(
                state, np.asarray(clf.params_), "pauli_z"
            ) == pytest.approx(expected, abs=1e-10)

    def test_pipeline_vqc_classifier(self) -> None:
        """``experiments/vqc.py`` produced the atlas's VQC accuracies."""
        vqc = pytest.importorskip("experiments.vqc")

        rng = np.random.default_rng(5)
        encoding = get_encoding("iqp", n_features=3, reps=2)
        model = vqc.VQCClassifier(encoding, n_var_layers=2, epochs=1, seed=0)
        model.fit(rng.uniform(0, 2 * np.pi, (6, 3)), np.array([0, 1, 0, 1, 0, 1]))

        for x in rng.uniform(0, 2 * np.pi, (3, 3)):
            expected = float(model._qnode(x, model.params_))
            state = simulate_encoding_statevector(encoding, x)
            assert variational_expectation(
                state, np.asarray(model.params_), "pauli_z"
            ) == pytest.approx(expected, abs=1e-10)


# =============================================================================
# Closed forms
# =============================================================================


class TestClosedForms:
    @pytest.mark.parametrize("observable", VARIATIONAL_OBSERVABLES)
    @pytest.mark.parametrize("n_qubits, n_var_layers", [(1, 1), (2, 3), (4, 2)])
    def test_adjoint_equals_parameter_shift(
        self, observable: str, n_qubits: int, n_var_layers: int
    ) -> None:
        rng = np.random.default_rng(11)
        for _ in range(3):
            state = _random_state(rng, n_qubits)
            theta = rng.uniform(-np.pi, np.pi, (n_var_layers, n_qubits))
            assert_allclose(
                variational_gradient(state, theta, observable),
                _shift_rule_gradient(state, theta, observable),
                atol=EXACT,
            )

    def test_one_qubit_angle_encoding_is_a_cosine(self) -> None:
        """RY(x) then RY(θ₁)RY(θ₂): ⟨Z⟩ = cos(x + θ₁ + θ₂)."""
        x, theta = 0.7, np.array([[0.3], [-1.1]])
        state = simulate_encoding_statevector(
            get_encoding("angle", n_features=1), np.array([x])
        )
        phase = x + theta.sum()
        assert variational_expectation(state, theta) == pytest.approx(
            np.cos(phase), abs=EXACT
        )
        assert_allclose(
            variational_gradient(state, theta), [-np.sin(phase)] * 2, atol=EXACT
        )

    def test_observables_on_basis_states(self) -> None:
        """θ = 0 is the identity except for the CNOT chain."""
        theta = np.zeros((1, 2))
        # |10⟩ → CNOT(0,1) → |11⟩: Z₀ = -1, Z⊗Z = +1, P(00) = 0
        state = np.array([0, 0, 1, 0], dtype=complex)
        assert variational_expectation(state, theta, "pauli_z") == pytest.approx(-1.0)
        assert variational_expectation(state, theta, "global_z") == pytest.approx(1.0)
        assert variational_expectation(state, theta, "computational") == 0.0
        # |00⟩ is fixed: every observable reads its maximum
        origin = np.array([1, 0, 0, 0], dtype=complex)
        for observable in VARIATIONAL_OBSERVABLES:
            assert variational_expectation(origin, theta, observable) == pytest.approx(
                1.0
            )

    def test_pauli_z_reads_the_most_significant_qubit(self) -> None:
        """Qubit 0 is the MSB, as everywhere else in the library."""
        theta = np.zeros((1, 3))
        msb_set = np.zeros(8, dtype=complex)
        msb_set[0b100] = 1.0  # qubit 0 in |1⟩ — CNOTs then flip 1 and 2
        lsb_set = np.zeros(8, dtype=complex)
        lsb_set[0b001] = 1.0  # qubit 2 in |1⟩ — untouched by the chain
        assert variational_expectation(msb_set, theta) == pytest.approx(-1.0)
        assert variational_expectation(lsb_set, theta) == pytest.approx(1.0)

    def test_parameter_layout_matches_theta(self) -> None:
        """Flat index l * n_qubits + q is θ[l, q]."""
        rng = np.random.default_rng(2)
        state = _random_state(rng, 3)
        theta = rng.uniform(-np.pi, np.pi, (2, 3))
        grad = variational_gradient(state, theta, "global_z").reshape(2, 3)
        for layer in range(2):
            for q in range(3):
                bumped = theta.copy()
                bumped[layer, q] += 1e-6
                fd = (
                    variational_expectation(state, bumped, "global_z")
                    - variational_expectation(state, theta, "global_z")
                ) / 1e-6
                assert grad[layer, q] == pytest.approx(fd, abs=1e-5)

    def test_parameter_count(self) -> None:
        assert variational_parameter_count(5, 3) == 15


# =============================================================================
# Light cone
# =============================================================================


class TestLiveParameterMask:
    @pytest.mark.parametrize("n_qubits", range(1, 9))
    @pytest.mark.parametrize("n_var_layers", range(1, 5))
    def test_pauli_z_light_cone_is_analytic(
        self, n_qubits: int, n_var_layers: int
    ) -> None:
        """θ[l, q] is live exactly when q ≤ L - 1 - l."""
        expected = np.array(
            [
                [q <= n_var_layers - 1 - layer for q in range(n_qubits)]
                for layer in range(n_var_layers)
            ]
        )
        mask = live_parameter_mask(n_qubits, n_var_layers, "pauli_z")
        assert np.array_equal(mask.reshape(n_var_layers, n_qubits), expected)

    @pytest.mark.parametrize("observable", VARIATIONAL_OBSERVABLES)
    @pytest.mark.parametrize("n_qubits, n_var_layers", [(2, 2), (4, 2), (5, 3)])
    def test_dead_parameters_have_identically_zero_gradient(
        self, observable: str, n_qubits: int, n_var_layers: int
    ) -> None:
        mask = live_parameter_mask(n_qubits, n_var_layers, observable)
        rng = np.random.default_rng(99)  # independent of the mask's own seed
        for _ in range(10):
            state = _random_state(rng, n_qubits)
            theta = rng.uniform(-np.pi, np.pi, (n_var_layers, n_qubits))
            grad = variational_gradient(state, theta, observable)
            assert np.all(np.abs(grad[~mask]) < 1e-13)
            assert np.all(np.abs(grad[mask]) > 0)

    def test_every_observable_has_a_live_parameter(self) -> None:
        for observable in VARIATIONAL_OBSERVABLES:
            for n_qubits in (1, 2, 6):
                assert live_parameter_mask(n_qubits, 2, observable).any()

    def test_returns_a_fresh_array(self) -> None:
        mask = live_parameter_mask(3, 2)
        mask[:] = False
        assert live_parameter_mask(3, 2).any()

    @pytest.mark.parametrize(
        "args, match",
        [
            ((0, 2, "pauli_z"), "positive"),
            ((2, 0, "pauli_z"), "positive"),
            ((2, 2, "local"), "observable"),
        ],
    )
    def test_invalid_arguments(self, args: tuple, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            live_parameter_mask(*args)


# =============================================================================
# Validation and safety
# =============================================================================


class TestValidation:
    def test_theta_must_be_two_dimensional(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            variational_gradient(np.array([1, 0], dtype=complex), np.zeros(1))

    def test_theta_dimensions_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            variational_expectation(np.array([1, 0], dtype=complex), np.zeros((0, 1)))

    def test_state_length_must_match_theta(self) -> None:
        with pytest.raises(ValueError, match="amplitudes"):
            variational_gradient(np.ones(8) / np.sqrt(8), np.zeros((2, 2)))

    def test_unknown_observable(self) -> None:
        state = np.array([1, 0], dtype=complex)
        with pytest.raises(ValueError, match="observable"):
            variational_gradient(state, np.zeros((1, 1)), "local_z")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="observable"):
            variational_expectation(state, np.zeros((1, 1)), "x")  # type: ignore[arg-type]

    def test_non_finite_state_raises_numerical_instability(self) -> None:
        state = np.array([np.nan, 0, 0, 0], dtype=complex)
        with pytest.raises(NumericalInstabilityError):
            variational_gradient(state, np.zeros((1, 2)))

    def test_inputs_are_not_mutated(self) -> None:
        rng = np.random.default_rng(4)
        state = _random_state(rng, 3)
        theta = rng.uniform(-np.pi, np.pi, (2, 3))
        state_copy, theta_copy = state.copy(), theta.copy()
        variational_gradient(state, theta)
        apply_variational_circuit(state, theta)
        assert np.array_equal(state, state_copy)
        assert np.array_equal(theta, theta_copy)

    def test_gradient_dtype_and_shape(self) -> None:
        grad = variational_gradient(
            np.array([1, 0, 0, 0], dtype=complex), np.zeros((3, 2))
        )
        assert grad.dtype == np.float64
        assert grad.shape == (6,)
