"""Exact simulation and gradients of the benchmark's variational read-out.

The atlas measures every encoding inside one fixed model: the variational
quantum classifier of :class:`encoding_atlas.benchmark.VQCClassifier` (and the
pipeline's ``experiments/vqc.py``, which builds the identical circuit). For an
encoding ``U(x)`` on ``n`` qubits that model prepares

.. math::

    |\\psi(x, \\theta)\\rangle = W(\\theta)\\, U(x)\\, |0\\rangle^{\\otimes n},
    \\qquad
    W(\\theta) = \\prod_{l=L-1}^{0} C \\, R(\\theta_l),

where each layer ``l`` applies ``RY(θ[l, q])`` to every qubit ``q`` and then
the linear chain ``C = CNOT(n-2, n-1) ··· CNOT(1, 2) · CNOT(0, 1)``. The model
reads out the cost

.. math::

    C(x, \\theta) = \\langle\\psi(x, \\theta)|\\, O \\,|\\psi(x, \\theta)\\rangle .

This module evaluates ``C`` and its exact gradient ``∂C/∂θ`` on a statevector
``U(x)|0⟩`` that has already been simulated, so the encoding may come from
any backend while the variational part is plain NumPy.

Why the gradient is exact
-------------------------
Every ``θ[l, q]`` enters the circuit through exactly one gate,
``RY(θ) = exp(-iθY/2)`` with ``Y² = I``. The derivative is therefore available
in closed form, and is computed here by the *adjoint method* [1]_: one forward
sweep, then one backward sweep that applies each gate's adjoint and reads off
``∂C/∂θ_k = 2 Re⟨λ_k| ∂G_k |φ_{k-1}⟩``. The result equals the two-term
parameter-shift rule [2]_ to floating-point precision while costing one
backward sweep instead of ``2·L·n`` forward ones.

That exactness is the point of differentiating ``θ`` rather than the data
``x``: a feature typically enters an encoding through several gates, with
scale factors and products such as ``(π - xᵢ)(π - xⱼ)``, so no two-term
shift rule applies to it.

Light cone
----------
Not every ``θ[l, q]`` can move the cost. A rotation outside the observable's
backward light cone commutes with the evolved observable, and its gradient is
identically zero for every input state. With the classifier's ``Z₀`` read-out
the CNOT chain's first gate has qubit 0 as *control*, which ``Z₀`` commutes
with, so the light cone grows by one qubit per layer going backwards:
``θ[l, q]`` is live exactly when ``q ≤ L - 1 - l``. For the benchmark's
``L = 2`` that is three parameters at any width. :func:`live_parameter_mask`
identifies the live set for any observable.

Qubit ordering
--------------
Statevectors follow the library-wide MSB convention: wire ``q`` is axis ``q``
of ``state.reshape((2,) * n)``, so wire 0 is the most significant bit. This
matches PennyLane, which is what :class:`~encoding_atlas.benchmark.VQCClassifier`
uses, and it is what :func:`~encoding_atlas.analysis.simulate_encoding_statevector`
returns for every backend.

References
----------
.. [1] Jones, T., & Gacon, J. (2020). "Efficient calculation of gradients in
       classical simulations of variational quantum algorithms."
       arXiv:2009.02823.
.. [2] Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., & Killoran, N. (2019).
       "Evaluating analytic gradients on quantum hardware." Physical Review A,
       99(3), 032331.
"""

from __future__ import annotations

import functools
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.analysis._utils import _global_z_eigenvalues
from encoding_atlas.core.exceptions import NumericalInstabilityError

__all__ = [
    "VARIATIONAL_OBSERVABLES",
    "VariationalObservable",
    "variational_parameter_count",
    "apply_variational_circuit",
    "variational_expectation",
    "variational_gradient",
    "live_parameter_mask",
]

VariationalObservable = Literal["pauli_z", "global_z", "computational"]

#: Observables the read-out can measure.
#:
#: - ``"pauli_z"`` — ``Z`` on qubit 0, the benchmark classifier's read-out. A
#:   local observable.
#: - ``"global_z"`` — ``Z⊗Z⊗…⊗Z`` on every qubit. A global observable.
#: - ``"computational"`` — the projector ``|0…0⟩⟨0…0|``. A global observable:
#:   it has support on all ``n`` qubits.
VARIATIONAL_OBSERVABLES: tuple[str, ...] = ("pauli_z", "global_z", "computational")

_Tensor = NDArray[np.complex128]

# A gate is ("ry", layer, qubit) or ("cnot", control, target).
_Gate = tuple[str, int, int]


def variational_parameter_count(n_qubits: int, n_var_layers: int) -> int:
    """Return the number of variational parameters, ``n_var_layers * n_qubits``."""
    return int(n_var_layers) * int(n_qubits)


def _gate_sequence(n_qubits: int, n_var_layers: int) -> list[_Gate]:
    """The read-out's gates in application order (first gate applied first)."""
    gates: list[_Gate] = []
    for layer in range(n_var_layers):
        gates.extend(("ry", layer, q) for q in range(n_qubits))
        gates.extend(("cnot", q, q + 1) for q in range(n_qubits - 1))
    return gates


def _axis_index(n_qubits: int, fixed: dict[int, int]) -> tuple[Any, ...]:
    """Index tuple selecting the given basis value on the given wires."""
    index: list[Any] = [slice(None)] * n_qubits
    for wire, value in fixed.items():
        index[wire] = value
    return tuple(index)


def _apply_ry_like(psi: _Tensor, c: float, s: float, wire: int) -> _Tensor:
    """Apply ``[[c, -s], [s, c]]`` to ``wire`` and return a new tensor.

    With ``c = cos(θ/2)``, ``s = sin(θ/2)`` this is ``RY(θ)``; with
    ``s → -s`` it is ``RY(θ)† = RY(-θ)``; with ``(c, s) = (-sin(θ/2)/2,
    cos(θ/2)/2)`` it is ``dRY/dθ``.
    """
    n_qubits = psi.ndim
    i0 = _axis_index(n_qubits, {wire: 0})
    i1 = _axis_index(n_qubits, {wire: 1})
    out = np.empty_like(psi)
    out[i0] = c * psi[i0] - s * psi[i1]
    out[i1] = s * psi[i0] + c * psi[i1]
    return out


def _apply_cnot(psi: _Tensor, control: int, target: int) -> _Tensor:
    """Apply ``CNOT(control, target)`` and return a new tensor (self-adjoint)."""
    n_qubits = psi.ndim
    i10 = _axis_index(n_qubits, {control: 1, target: 0})
    i11 = _axis_index(n_qubits, {control: 1, target: 1})
    out = psi.copy()
    out[i10] = psi[i11]
    out[i11] = psi[i10]
    return out


def _apply_gate(
    psi: _Tensor, gate: _Gate, theta: NDArray[np.float64], *, adjoint: bool
) -> _Tensor:
    """Apply one read-out gate, or its adjoint."""
    kind, a, b = gate
    if kind == "cnot":
        return _apply_cnot(psi, a, b)
    half = 0.5 * float(theta[a, b])
    s = np.sin(half)
    return _apply_ry_like(psi, float(np.cos(half)), float(-s if adjoint else s), b)


def _apply_observable(psi: _Tensor, observable: str) -> _Tensor:
    """Return ``O|ψ⟩`` for a Hermitian read-out observable."""
    n_qubits = psi.ndim
    if observable == "pauli_z":
        out = psi.copy()
        out[_axis_index(n_qubits, {0: 1})] *= -1.0
        return out
    if observable == "global_z":
        eigenvalues = _global_z_eigenvalues(n_qubits).reshape((2,) * n_qubits)
        return psi * eigenvalues
    if observable == "computational":
        out = np.zeros_like(psi)
        origin = (0,) * n_qubits
        out[origin] = psi[origin]
        return out
    raise ValueError(
        f"observable must be one of {list(VARIATIONAL_OBSERVABLES)}, "
        f"got {observable!r}."
    )


def _prepare(
    state: NDArray[Any], theta: NDArray[Any]
) -> tuple[_Tensor, NDArray[np.float64], int, int]:
    """Validate inputs and return ``(tensor, theta, n_qubits, n_var_layers)``."""
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 2 or theta_arr.shape[0] < 1 or theta_arr.shape[1] < 1:
        raise ValueError(
            "theta must have shape (n_var_layers, n_qubits) with both "
            f"dimensions at least 1, got shape {theta_arr.shape}."
        )
    n_var_layers, n_qubits = (int(d) for d in theta_arr.shape)

    flat = np.asarray(state, dtype=np.complex128).reshape(-1)
    if flat.shape[0] != 1 << n_qubits:
        raise ValueError(
            f"state has {flat.shape[0]} amplitudes, but theta describes "
            f"{n_qubits} qubits (expected {1 << n_qubits})."
        )
    # Copy so no caller-owned buffer is ever written to.
    tensor = flat.copy().reshape((2,) * n_qubits)
    return tensor, theta_arr, n_qubits, n_var_layers


def apply_variational_circuit(
    state: NDArray[Any], theta: NDArray[Any]
) -> NDArray[np.complex128]:
    """Apply the read-out ``W(θ)`` to an encoded statevector.

    Parameters
    ----------
    state : array_like, shape (2**n_qubits,)
        Encoded state ``U(x)|0⟩`` in MSB order.
    theta : array_like, shape (n_var_layers, n_qubits)
        Variational angles, laid out as ``VQCClassifier.params_``.

    Returns
    -------
    NDArray[np.complex128], shape (2**n_qubits,)
        ``W(θ) U(x)|0⟩``.

    Raises
    ------
    ValueError
        If ``theta`` is not 2-D or ``state`` has the wrong length.
    """
    psi, theta_arr, n_qubits, n_var_layers = _prepare(state, theta)
    for gate in _gate_sequence(n_qubits, n_var_layers):
        psi = _apply_gate(psi, gate, theta_arr, adjoint=False)
    return psi.reshape(-1)


def variational_expectation(
    state: NDArray[Any],
    theta: NDArray[Any],
    observable: VariationalObservable = "pauli_z",
) -> float:
    """Return ``C(x, θ) = ⟨ψ|O|ψ⟩`` for ``|ψ⟩ = W(θ) U(x)|0⟩``.

    Parameters
    ----------
    state : array_like, shape (2**n_qubits,)
        Encoded state ``U(x)|0⟩`` in MSB order.
    theta : array_like, shape (n_var_layers, n_qubits)
        Variational angles.
    observable : {"pauli_z", "global_z", "computational"}, default="pauli_z"
        Read-out observable; see :data:`VARIATIONAL_OBSERVABLES`.

    Returns
    -------
    float
        The expectation value.
    """
    psi, theta_arr, n_qubits, n_var_layers = _prepare(state, theta)
    for gate in _gate_sequence(n_qubits, n_var_layers):
        psi = _apply_gate(psi, gate, theta_arr, adjoint=False)
    return float(np.real(np.vdot(psi, _apply_observable(psi, observable))))


def variational_gradient(
    state: NDArray[Any],
    theta: NDArray[Any],
    observable: VariationalObservable = "pauli_z",
) -> NDArray[np.float64]:
    """Return the exact gradient ``∂C/∂θ`` by the adjoint method.

    Parameters
    ----------
    state : array_like, shape (2**n_qubits,)
        Encoded state ``U(x)|0⟩`` in MSB order.
    theta : array_like, shape (n_var_layers, n_qubits)
        Variational angles.
    observable : {"pauli_z", "global_z", "computational"}, default="pauli_z"
        Read-out observable; see :data:`VARIATIONAL_OBSERVABLES`.

    Returns
    -------
    NDArray[np.float64], shape (n_var_layers * n_qubits,)
        ``∂C/∂θ[l, q]`` at flat index ``l * n_qubits + q`` — the row-major
        flattening of ``theta``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``observable`` is unknown.
    NumericalInstabilityError
        If the gradient is not finite (for instance because the encoded
        state contained NaN amplitudes).

    Notes
    -----
    With ``|φ_k⟩`` the state after the first ``k`` gates and
    ``|λ_k⟩ = G_{k+1}† ··· G_K† O |φ_K⟩``, the derivative with respect to the
    angle of gate ``k`` is ``2 Re⟨λ_k| ∂G_k |φ_{k-1}⟩``. A single backward
    sweep maintains both ``φ`` and ``λ``.
    """
    psi, theta_arr, n_qubits, n_var_layers = _prepare(state, theta)
    if observable not in VARIATIONAL_OBSERVABLES:
        raise ValueError(
            f"observable must be one of {list(VARIATIONAL_OBSERVABLES)}, "
            f"got {observable!r}."
        )
    gates = _gate_sequence(n_qubits, n_var_layers)

    phi = psi
    for gate in gates:
        phi = _apply_gate(phi, gate, theta_arr, adjoint=False)
    lam = _apply_observable(phi, observable)

    gradient = np.zeros(n_var_layers * n_qubits, dtype=np.float64)
    for gate in reversed(gates):
        phi = _apply_gate(phi, gate, theta_arr, adjoint=True)
        kind, layer, qubit = gate
        if kind == "ry":
            half = 0.5 * float(theta_arr[layer, qubit])
            d_phi = _apply_ry_like(
                phi, -0.5 * float(np.sin(half)), 0.5 * float(np.cos(half)), qubit
            )
            gradient[layer * n_qubits + qubit] = 2.0 * float(
                np.real(np.vdot(lam, d_phi))
            )
        lam = _apply_gate(lam, gate, theta_arr, adjoint=True)

    if not np.all(np.isfinite(gradient)):
        bad = int(np.argmax(~np.isfinite(gradient)))
        raise NumericalInstabilityError(
            f"Variational gradient produced a non-finite value: {gradient[bad]}",
            value=float(gradient[bad]),
            operation="variational_adjoint_gradient",
            details={"param_index": bad, "n_qubits": n_qubits},
        )
    return gradient


# Liveness is decided by evaluating the exact gradient at random points. A
# gradient that is identically zero is zero at every point up to round-off
# (~1e-16 per gate); a gradient that is not identically zero is a non-zero
# trigonometric polynomial in θ and the input state, which vanishes at a
# random point with probability zero. For a Haar-random state its typical
# magnitude is ~2^(-n/2) ≥ 1e-3 at the library's 20-qubit ceiling, eight
# orders of magnitude above the tolerance. Several independent trials, with
# a fixed seed, make the mask exact in practice and deterministic.
_LIVENESS_TOLERANCE: float = 1e-10
_LIVENESS_TRIALS: int = 4
_LIVENESS_SEED: int = 20260601


@functools.lru_cache(maxsize=256)
def _live_parameter_mask(
    n_qubits: int, n_var_layers: int, observable: str
) -> tuple[bool, ...]:
    """Cached worker for :func:`live_parameter_mask`."""
    rng = np.random.default_rng(_LIVENESS_SEED)
    dim = 1 << n_qubits
    live = np.zeros(n_var_layers * n_qubits, dtype=bool)
    for _ in range(_LIVENESS_TRIALS):
        state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        state /= np.linalg.norm(state)
        theta = rng.uniform(-np.pi, np.pi, size=(n_var_layers, n_qubits))
        gradient = variational_gradient(
            state, theta, cast("VariationalObservable", observable)
        )
        live |= np.abs(gradient) > _LIVENESS_TOLERANCE
    return tuple(bool(v) for v in live)


def live_parameter_mask(
    n_qubits: int,
    n_var_layers: int,
    observable: VariationalObservable = "pauli_z",
) -> NDArray[np.bool_]:
    """Which variational parameters can change the cost at all.

    A parameter outside the observable's backward light cone has
    ``∂C/∂θ ≡ 0`` for every input state and every ``θ``. Including such
    parameters in an average gradient variance dilutes it by a factor that
    depends only on the circuit width, which would read as a barren plateau
    that is not there.

    The mask depends on the read-out circuit and observable only — never on
    the encoding — and is computed once per ``(n_qubits, n_var_layers,
    observable)``.

    Parameters
    ----------
    n_qubits : int
        Circuit width, ``1 <= n_qubits``.
    n_var_layers : int
        Number of read-out layers, ``1 <= n_var_layers``.
    observable : {"pauli_z", "global_z", "computational"}, default="pauli_z"
        Read-out observable.

    Returns
    -------
    NDArray[np.bool_], shape (n_var_layers * n_qubits,)
        ``True`` for parameters whose gradient is not identically zero, in the
        flat order of :func:`variational_gradient`. A fresh array on each call.

    Raises
    ------
    ValueError
        If the dimensions are not positive or ``observable`` is unknown.

    Notes
    -----
    For ``observable="pauli_z"`` the live set is exactly
    ``{θ[l, q] : q ≤ n_var_layers - 1 - l}``; see the module docstring.
    """
    if n_qubits < 1 or n_var_layers < 1:
        raise ValueError(
            "n_qubits and n_var_layers must be positive, got "
            f"n_qubits={n_qubits}, n_var_layers={n_var_layers}."
        )
    if observable not in VARIATIONAL_OBSERVABLES:
        raise ValueError(
            f"observable must be one of {list(VARIATIONAL_OBSERVABLES)}, "
            f"got {observable!r}."
        )
    return np.array(
        _live_parameter_mask(int(n_qubits), int(n_var_layers), str(observable)),
        dtype=bool,
    )
