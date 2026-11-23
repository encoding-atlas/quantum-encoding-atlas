# Encoding Properties

Every quantum encoding can be characterised by a set of measurable properties that determine its suitability for a given task. The Quantum Encoding Atlas computes these properties automatically through the `analysis` module and exposes them via the `EncodingProperties` dataclass.

This page defines each property, explains what it measures, and provides intuition for how to interpret the values.

---

## Property Summary

| Property | What It Measures | Range | Higher Is... |
|:---------|:----------------|:-----:|:-------------|
| **Expressibility** | Hilbert space coverage | [0, +inf) | More expressive |
| **Entanglement Capability** | Meyer-Wallach entanglement | [0, 1] | More entangled |
| **Trainability** | Gradient variance of a classifier built on the encoding (barren plateaus) | [0, 1] | Easier to train |
| **Circuit Depth** | Sequential gate layers | [1, +inf) | Deeper circuit |
| **Gate Count** | Total quantum gates | [1, +inf) | More expensive |
| **Simulability** | Classical simulation feasibility | Boolean | — |

---

## Expressibility

**What it measures:** How uniformly the encoding covers the space of quantum states (Hilbert space). An encoding that can only produce a small subset of possible states is *less* expressive than one that can reach states throughout the full space.

**How it's computed:** Sample random inputs, generate the corresponding quantum states, compute the distribution of pairwise fidelities, and compare against the Haar-random (maximally uniform) distribution using KL divergence.

```
  Low Expressibility                High Expressibility
  (e.g., Angle Encoding)           (e.g., IQP Encoding)

  States cluster in a              States spread across
  small region:                    the full Hilbert space:

       ·····                              ·   ·
      ·······                           ·   ·   ·
     ·········                        ·   ·   ·   ·
      ·······                           ·   ·   ·
       ·····                              ·   ·
```

!!! note
    Lower KL divergence = higher expressibility (closer to Haar random). The analysis module reports expressibility such that **lower values indicate more expressive encodings**.

---

## Entanglement Capability

**What it measures:** The average amount of entanglement produced across random inputs, quantified by the Meyer-Wallach measure \( Q \).

- \( Q = 0 \): product states (no entanglement) — e.g., Angle Encoding
- \( Q = 1 \): maximally entangled states

**Why it matters:** Entanglement enables quantum correlations that have no classical analogue. Encodings that produce entangled states can represent functions that separable encodings cannot.

---

## Trainability

**What it measures:** Whether gradient-based optimisation of a model *built on* the encoding is feasible. An encoding has no trainable parameters of its own — its angles are the data — so trainability is measured inside the model the atlas benchmarks: the variational classifier `VQCClassifier`, which applies the encoding, then `n_var_layers` layers of `RY(θ)` rotations and a linear CNOT chain, and reads out ⟨Z₀⟩.

`estimate_trainability` reports the variance of that model's cost gradient, \( \mathrm{Var}[\partial C / \partial \theta_k] \), at random initialisation — with the data \(x\) drawn from `input_range` and \(\theta\) from \([-\pi, \pi)\), the classifier's own initialisation. If that variance vanishes exponentially with the number of qubits (a "barren plateau"), the model is effectively untrainable.

```
  High Trainability                 Low Trainability
  (gradients are informative)       (barren plateau)

  ∂L/∂θ                            ∂L/∂θ
    │   ╱╲                            │
    │  ╱  ╲   ╱╲                      │  ─────────────  ≈ 0
    │ ╱    ╲ ╱  ╲                     │
    │╱      ╲    ╲                    │
    └───────────────                  └───────────────
```

**How it's computed:** each \(\theta_k\) enters exactly one `RY` gate, so the gradient is exact. It is evaluated by the adjoint method and agrees with PennyLane's automatic differentiation of the classifier's circuit to floating-point precision. The variance is averaged over the parameters that can move the cost at all. With the ⟨Z₀⟩ read-out, only \(\theta[l, q]\) with \(q \le L - 1 - l\) lie in the observable's backward light cone — three parameters for the benchmark's two layers, at any width. The rest have identically zero gradient, and averaging them in would make the variance fall like \(1/n\) for reasons that have nothing to do with the encoding.

!!! note "The classifier sees only its first qubits"
    A consequence of that light cone: the benchmark classifier's output depends on the encoded state only through its first `n_var_layers` qubits. For a product encoding such as `AngleEncoding` on four features, features `x[2]` and `x[3]` cannot affect the prediction at all; an entangling encoding can route their information onto qubits 0 and 1.

!!! warning "Changed definition"
    Releases before this one differentiated the *data* \(x\) with a fixed \(\pm\pi/2\) shift. That shift returns a derivative only when each feature enters a single unit-frequency rotation (plain angle encoding); for the other encodings it returned no derivative at all, and a derivative with respect to data measures input sensitivity rather than trainability. The atlas's trainability column has been re-measured under the definition above.

**Rule of thumb:** Shallow circuits with limited entanglement are more trainable. Deep, highly entangled circuits risk barren plateaus.

---

## Circuit Depth and Gate Count

**What they measure:** The hardware cost of implementing the encoding.

- **Depth**: number of sequential gate layers (determines execution time)
- **Gate count**: total gates, split into single-qubit and two-qubit (CNOT) gates

Two-qubit gates are significantly noisier than single-qubit gates on current hardware, so the CNOT count is often the more relevant metric for NISQ devices.

---

## Simulability

**What it measures:** Whether the encoding can be efficiently simulated by a classical computer.

- **Simulable encodings** (Angle, Basis, Higher-Order Angle): A classical computer can compute measurement outcomes in polynomial time. These cannot provide quantum advantage.
- **Non-simulable encodings** (IQP, Amplitude, ZZ, etc.): Classical simulation is believed to require exponential resources. These are candidates for quantum advantage.

**Detection method:** The atlas checks for Clifford circuits, matchgate circuits, and product-state circuits, which are known to be classically simulable.

---

## Accessing Properties

```python
from encoding_atlas import IQPEncoding

enc = IQPEncoding(n_features=4, reps=2)
props = enc.properties

print(props.n_qubits)        # 4
print(props.depth)            # Circuit depth
print(props.gate_count)       # Total gates
print(props.is_entangling)    # True
print(props.simulability)     # 'not_simulable'
```

For detailed analysis:

```python
from encoding_atlas.analysis import (
    compute_expressibility,
    compute_entanglement_capability,
    estimate_trainability,
    count_resources,
)
```

See the [API Reference](../api/index.md) for the full analysis interface.

---

## What's Next

- [Quantum Advantage](quantum-advantage.md) — when these properties translate into real-world benefits
- [Encodings Reference](../encodings/index.md) — see the properties of each encoding
