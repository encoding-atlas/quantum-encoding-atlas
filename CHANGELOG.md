# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

#### Attribution
- The project is attributed to the **Quantum Encoding Atlas Authors**
  throughout: the `LICENSE` copyright line, `CITATION.cff` (an entity author),
  the `pyproject.toml` authors and maintainers (and so the package metadata),
  `AUTHORS.md`, the README's BibTeX entry, the documentation's author and
  footer, and the notebooks' author lines. The contact address is unchanged.

### Fixed

#### Trainability did not measure a derivative, or trainability
- `estimate_trainability` differentiated the encoding's *data* `x` with the
  two-term shift `(f(x + π/2) − f(x − π/2)) / 2` and documented the result as
  the exact `∂⟨O⟩/∂xᵢ`. That rule is exact only for a parameter entering a
  single unit-frequency rotation, which holds for `angle` alone. A feature that
  enters `RZ(2x)` cancels exactly; one that enters `(π − xᵢ)(π − xⱼ)` is not
  periodic at all. Against a converged finite-difference reference, **13 of 15
  encodings** returned values off by more than the derivative itself — for IQP
  at one point `[0.001, −0.013, 0.015, −0.018]` where the derivative is
  `[−1.05, −0.92, −1.17, −0.83]` — and the stored gradient variances were off by
  factors from 0.5 to 152, unevenly enough to reorder the encodings. The only
  check was a finite-difference test on `AngleEncoding(n_features=1)`, the one
  case where the shift is exact.
- Even exactly computed, a derivative with respect to data measures input
  sensitivity, not trainability. `basis` scored **0.000** ("untrainable")
  because thresholded inputs have zero `x`-derivative, and 16 single-layer
  IQP/ZZ/Pauli and Hamiltonian configurations scored ~1e-33 because
  `P(0…0) = 2⁻ⁿ` for every input — including two-qubit circuits, where a
  barren plateau is not meaningful. The default observable, `P(0…0)`, was
  documented as local; it is a global projector.
- Trainability is now `Var[∂C/∂θ]` for the model the atlas benchmarks — the
  encoding followed by `VQCClassifier`'s `n_var_layers` layers of `RY(θ)` and a
  linear CNOT chain, read out as `⟨Z₀⟩` — with `x` drawn from `input_range` and
  `θ` from `[−π, π)`, the classifier's own initialisation. Each `θ` enters one
  `RY` gate, so the gradient is exact; it is computed by the adjoint method in
  the new private module `analysis._variational` and agrees with PennyLane's
  automatic differentiation of `VQCClassifier` to floating-point precision.
- The variance is averaged over the parameters inside the read-out's light
  cone. `⟨Z₀⟩` commutes with the chain's first CNOT, so only `θ[l, q]` with
  `q ≤ L − 1 − l` can move the cost — three parameters at any width. Averaging
  over all `L·n` would dilute the variance by `3 / (2n)`, a decay with width
  that would itself read as a barren plateau. A consequence for the benchmark
  itself: the classifier depends on the encoded state only through qubits
  `0 … L − 1`. For a product encoding such as `angle` on the four-feature
  datasets, features `x[2]` and `x[3]` cannot affect the Stage 6a VQC's
  predictions (verified: its output is unchanged to 12 decimals when they
  change). Stage 6a is unchanged; this is recorded so its VQC accuracies are
  read accordingly.
- API: `estimate_trainability` and `compute_gradient_variance` gain
  `n_var_layers` (default 2, matching `VQCClassifier`), appended last so
  positional calls are unaffected. The default `observable` is now `"pauli_z"`,
  the classifier's read-out. `TrainabilityResult` gains `n_var_layers`,
  `observable`, `n_live_parameters` and `live_parameter_mask`;
  `per_parameter_variance` now has shape `(n_var_layers * n_qubits,)`. All
  three backends give the same result to ~1e-15, since the backend only
  prepares `U(x)|0⟩`. Each sample costs one encoding simulation rather than
  `2 · n_features`, so Stage 5 now runs in about 90 s.

#### `compute_parameter_gradient` and `compute_all_parameter_gradients` were not derivatives
- Both public helpers applied the same `±π/2` shift to the input features and
  carried the same "exact" claim. They now use a fourth-order central
  difference with an absolute step `h = 1e-4`. They agree with analytic
  derivatives of angle encoding to 6e-12 (including angles up to ±30), and
  with an independent sixth-order Richardson reference to ~1e-11 on the IQP,
  ZZ and higher-order-angle maps, the hardest case. Each component remains
  bit-identical between the batched and per-feature functions.

#### Stage 7 joined metrics from different circuits
- `build_encoding_profiles` keyed every stage's results on `n_features`
  alone, while Stages 1-5 sweep `reps`, `order`, `n_layers` and
  `entanglement` at each width, so the configuration written last won. H4
  paired Stage 1 depth with Stage 5 gradient variance from *different*
  circuits in **16 of 58** pairs, and the per-encoding headline metrics came
  from circuits other than the benchmarked one for `angle` (reps=2 instead of
  1: depth 2 instead of 1), `higher_order_angle` (order=1 instead of 2) and
  `so2_equivariant` (`max_angular_momentum=1` instead of 2: depth 6 instead of
  10, entanglement capability 0.969 instead of 0.060).
- Every join is now keyed on the full constructor configuration — recorded
  parameters completed with the constructor's defaults — and anchored to the
  configuration Stages 6a/6b benchmarked (`benchmarked_configurations`).
  Scaling curves hold that configuration fixed and vary only the width. H4
  pairs depth and gradient variance per measured circuit
  (`trainability_records`): **121** points, against 55 before. A stage that
  never measured the benchmarked circuit now leaves the metric `None` instead
  of borrowing a neighbour's value.
- That exposed two stage configurations that never measured the benchmarked
  circuit: Stage 5b ran `AngleEncoding(reps=2)` and Stage 2
  `DataReuploading(n_layers=3)`, while the benchmark ran `reps=1` and
  `n_layers=2`. Both are corrected and re-run, one task each.

#### The bundled atlas predated the corrections described above
- `atlas/data/master_summary.json` was last regenerated before the
  missing-metric fix, so the rank changes this changelog reports for it
  (`higher_order_angle` 3 → 5, `cyclic_equivariant` 4 → 3, `amplitude` 5 → 4)
  were never in the shipped data. It is now regenerated from the corrected
  pipeline, so it carries those corrections as well as the ones above.

#### The independent Stage 7 verifier still used the pre-correction scoring rule
- `experiments/verify_stage7_results.py` re-derived the 7.9 scores by dropping
  every absent metric, as 7.9 did before the explicit missing-metric policy, so
  it disagreed with the pipeline for every encoding with a structural zero. It
  now implements the policy independently and compares against the stored
  four-decimal scores at half a unit in the last place.

### Changed

#### Corrected atlas results
- **Trainability column**: re-measured for all 121 configurations; the
  validator now rejects zero variances instead of exempting them, and checks
  that parameters outside the light cone have zero variance and that
  product-state encodings are width-invariant. All 1028 checks pass on the new
  data; the old data fails 133.
- **H4** (depth drives barren plateaus) remains supported, more strongly:
  depth partial ρ = −0.738 (p = 4.4e-22) after controlling for family, against
  −0.472 published; family partial ρ = −0.205 (p = 0.024). The variance of
  the all-to-all maps (IQP, ZZ) decays with *width* rather than with `reps` —
  an encoding-induced concentration — while hardware-efficient and
  re-uploading maps lose variance with depth.
- **H7** (no encoding dominates) is now **refuted**: with every objective taken
  from the benchmarked circuit, `angle` is at least as good as every other
  encoding on accuracy, inverse depth, trainability and noise resilience, and
  is the whole Pareto front (previously `angle`, `basis`,
  `higher_order_angle`, `swap_equivariant`). Its trainability margin over
  `swap_equivariant`, `higher_order_angle`, `basis` and `amplitude` (0.722
  against 0.712–0.717) lies inside their bootstrap 95% confidence intervals,
  so this verdict rests partly on a statistical tie.
- **H5** stands (U = 39, r = 1.0); its p-value moves from 0.0036 to 0.0105
  because `angle` and `higher_order_angle` now have *identical* fidelity decay
  — four single-qubit gates lose fidelity independently of the state — and the
  tie switches SciPy from the exact to the tie-corrected test.
- **Ranking (7.9)**: `angle` stays first. Trainability carries 20% of the
  weight, so the order below it moves: `amplitude` 2, `swap_equivariant` 3,
  `higher_order_angle` 4, `trainable_encoding` 5, `hardware_efficient` 6,
  `cyclic_equivariant` 7, `basis` 10 (from 14), `qaoa_encoding` 11.
- **Weight sweep (7.8)**: still no encoding is robustly top-3. `angle` is in
  the top 3 in 87.1% of weightings in the regenerated Stage 7 run (88.2% at the
  test suite's seed), below the 90% bar.
- **Guide**: 6 of 72 `recommend_encoding` queries change, all at
  `n_features=8` on a simulator, from `higher_order_angle` to `angle`
  (priorities `trainability`, `speed` and `noise_resilience`), because
  `angle`'s depth and noise resilience now describe its benchmarked circuit.

### Added

#### Tests
- `test_variational.py`: the read-out engine against PennyLane autodiff for
  every benchmarked encoding, all three observables and one to three layers;
  against `VQCClassifier` and the pipeline's `experiments/vqc.py` gate for
  gate; against the shift rule and closed forms; and the analytic light cone.
- `test_input_gradients.py`: the input-gradient helpers against analytic
  derivatives and an independent Richardson reference for every benchmarked
  encoding, with a regression showing the old shift rule's error.
- `tests/unit/experiments/`: same-circuit joins on fixtures built so that a
  width-keyed join gives a different answer; a check over the committed stage
  configurations that each one measures the benchmarked circuit (it fails on
  the two configurations above); and the Stage 5 handler and validator.
  Reintroducing the width-keyed join, the stale configurations or the
  all-parameter average was confirmed to fail these tests.

### Fixed

#### The guide recommended encodings that cannot be constructed
- `recommend_encoding(n_features=5, symmetry="general")` returned
  `symmetry_inspired` with confidence 0.88, and building it raised
  `ValueError: rotation symmetry requires even n_features`. Across a sweep of
  the full documented input space, **8400 of 190080 queries (4.42%)** named an
  encoding the caller could not construct at the stated feature count:
  `symmetry_inspired` at every odd width (4224 queries), plus
  `cyclic_equivariant`, `higher_order_angle` and `symmetry_inspired` at
  `n_features=1` (4176).
- The hard-filter mechanism was never wrong — `_passes_hard_constraints`
  already checked `requires_even_features`, `requires_n_features` and
  `max_features`. The rule *data* was: `symmetry_inspired` never declared
  `requires_even_features` (although `swap_equivariant`, which has the same
  constraint, did), and no field could express "needs at least two features"
  at all.
- `EncodingRule` gains `min_features`, populated for all 16 encodings from each
  constructor's measured admissible widths, and enforced by the hard filter.
  `requires_even_features` is now set on `symmetry_inspired` and
  `so2_equivariant` (the latter was masked by `requires_n_features=2`, so
  latent rather than live). The sweep now reports **0** unbuildable
  recommendations.
- Nothing else moves: across 103680 queries at already-valid feature counts,
  **no recommendation changed**. Encoding diversity rose from 13 to 15 of 16,
  because filtering an inapplicable candidate now surfaces the next best one
  instead of a name that raises.

### Added

#### The rule table is now checked against the encodings themselves
- `tests/unit/guide/test_rule_consistency.py` asserts the invariant directly:
  every `(encoding, n_features)` pair the hard filter admits must be
  constructible. It is exhaustive over all 16 encodings and widths 1-16, so a
  new encoding or a changed constructor precondition is caught without anyone
  updating a fixture.
- The invariant is deliberately one-directional. The rules may be *stricter*
  than the constructor: `max_features` is advisory for five of the six
  encodings that set it (`higher_order_angle`, `iqp`, `zz_feature_map`,
  `pauli_feature_map`, `data_reuploading` all build one feature above their
  stated ceiling), and that is pinned so a future change cannot widen the rules
  to match the constructors and undo the deliberate recommendation ceilings.
- An end-to-end sweep additionally checks that no recommendation, nor any
  alternative it offers, is unbuildable; and the published constraint table in
  `docs/guide/recommendation-architecture.md` is checked against
  `ENCODING_RULES`. A stale table is how the missing flag stayed invisible: the
  docs accurately described the rules and inaccurately described the encoding.
- Reintroducing any of the three defects was confirmed to fail the suite
  (9, 10 and 5 tests respectively).


#### A median imputation manufactured the atlas's only robustness claim
- Stage 7's Monte Carlo weight sweep filled every absent metric with the
  **column median** before ranking. For `entanglement_capability` the absent
  entries belong to `angle`, `higher_order_angle` and `basis` — the atlas's own
  `Non-Entangling` family, which Stage 4 never attempted because a circuit with
  no entangling gates has a capability of exactly 0. The median substituted
  **0.605**, the median of the encodings that do entangle.
- That one substitution was the whole claim. Published: `robustly_top3 ==
  ['angle']`, with angle in the top 3 in **97.8%** of 1000 random weightings.
  With the structural zero: angle drops to **80.8%**, below the 90% threshold,
  and `robustly_top3` becomes **`[]`** — no encoding survives reweighting. Mean
  rank moves 1.71 to 3.26; `higher_order_angle` 3.56 to 6.04; `basis` 7.68 to
  12.10. Computed by calling the pipeline's own functions with nothing changed
  but that classification.
- **This corrects a published conclusion**, unlike the preceding fixes, which
  left their results standing. The headline ranking is not affected: angle
  remains first at 0.733. `higher_order_angle` moves from rank 3 to 5,
  `cyclic_equivariant` 4 to 3 and `amplitude` 5 to 4.

#### Three analyses of the same data used three different missing-metric rules
- Stage 7.8 imputed the column median; Stage 7.9 omitted the metric and
  renormalised the weights; the rank figures imputed 0.0 (fixed previously).
  The same profile data was therefore ranked against three different
  objectives depending on which analysis read it.
- Under 7.9's rule, 4 of 16 encodings were scored on a reduced objective —
  `basis` on 70% of the intended weight, `amplitude` 85%, `angle` and
  `higher_order_angle` 95% — with nothing in the output saying so. Both stages
  now share one policy, and `effective_weight` reports the shortfall.

#### The weight sweep was not reproducible across NumPy versions
- The sweep drew weights from `numpy.random.default_rng`, whose stream NumPy
  explicitly does not guarantee between releases. Re-running the published
  analysis from unchanged code (`tradeoff.py` last modified 2026-02-11, before
  the 2026-02-23 run) and byte-identical inputs reproduced the deterministic
  equal-weight ranking exactly while returning 97.2% where 97.8% was published.
  It now uses `numpy.random.RandomState`, whose stream NumPy does guarantee.

### Added

#### An explicit missing-metric policy for weighted scoring
- `encoding_atlas.benchmark.scoring` distinguishes the two absences that both
  arrive as `None`: `structurally_zero` (defined, known, and equal to zero, so
  never measured) and `not_measured` (undefined or never computed). A
  structural zero resolves to `0.0` and takes part in normalisation; an
  unmeasured metric contributes nothing and is *reported* rather than filled or
  silently dropped. Exposed as `score_encodings`, `weight_sensitivity`,
  `readings_from_values`, `measured`, `structurally_zero`, `not_measured`,
  `MetricReading`, `MetricSpec`, `ScoringResult` and `SensitivityResult`.
- `ScoringResult.effective_weight` records the fraction of the intended weight
  each encoding was judged on, so ranking against a shorter yardstick is
  visible instead of absorbed into the score. `on_unusable="exclude"` drops
  such encodings instead. The failure mode is sharp: an encoding measured only
  on the axis it wins scores 1.0 and ranks first on half the objective.
- Ranking order is decided on scores rounded to 12 decimal places. Weighted
  sums accumulate in different orders for different encodings, so
  mathematically tied scores can differ by one ULP; without this, rounding
  noise decided rank 1 from rank 3 among genuinely tied encodings.
- New concepts page, `docs/concepts/missing-metrics.md`, and API reference
  entries. Reintroducing any of the three defects was confirmed to fail the
  test suite, including with the unpackaged raw pipeline output absent.

#### Multi-dataset statistical comparison (Demsar 2006)
- `encoding_atlas.benchmark.comparison` implements the procedure for comparing
  many encodings across many datasets: per-dataset ranking, the Friedman
  omnibus with the Iman-Davenport correction, and Nemenyi or Bonferroni-Dunn
  post-hoc tests. Exposed as `compare_over_datasets`, `friedman_test`,
  `iman_davenport`, `average_ranks`, `critical_difference` and
  `complete_cases`.
- This replaces a method that could not work. The previously available route —
  all-pairs `wilcoxon_test` with `holm_bonferroni` — is *arithmetically*
  incapable of a positive result at atlas scale, not merely underpowered: a
  paired Wilcoxon over N=8 datasets has an exact two-sided minimum attainable
  p of `2/2**8 = 0.0078`, while Holm's strictest threshold over the
  `15*14/2 = 105` comparisons is `0.05/105 = 0.00048`. Measured on the atlas's
  kernel results: 0 of 105 significant, smallest raw p exactly at the 0.0078
  floor. Friedman on the same matrix gives `chi2(14) = 79.09, p = 4.2e-11` and
  separates 5 encodings from the best.
- **The omnibus test was previously absent entirely.** The journal figure set
  drew Nemenyi critical-difference diagrams — a *post-hoc* test — with no
  omnibus behind them and no statistic reported. Running the missing test
  confirms the published conclusions stand (VQC `chi2(14) = 63.46,
  p = 2.9e-08`, Iman-Davenport `F(14,98) = 9.15`; kernel `chi2(14) = 79.09,
  p = 4.2e-11`, `F(14,98) = 16.83`), so this closes a reporting and procedure
  gap rather than changing a result. `compare_over_datasets` now returns
  `posthoc=None` when the omnibus fails to reject, instead of a table that
  should not be quoted.
- Critical values come from the studentised range distribution rather than a
  lookup table, so any number of methods and any alpha work. Agreement with
  Demsar Table 5 is within that table's own rounding (max deviation 0.0007
  across k = 2..20), and the published table is used as a test oracle. The
  Bonferroni-Dunn values match Demsar Table 6 except at k = 9, where the
  published 2.724 does not follow from `alpha/(k-1) = 0.00625`; the value is
  2.7344 and the neighbouring entries confirm the table has a typo.
- New concepts page, `docs/concepts/statistical-comparison.md`, and API
  reference entries.

### Fixed

#### Rank figures imputed a missing result as accuracy zero
- Two journal figures built their rank matrix with `acc.get(dataset, 0.0)`, so
  an encoding with no result on a dataset was ranked *worst* there. Missing
  cells are structural, not accidental: `SO2EquivariantFeatureMap` accepts only
  two features and therefore has no result on the four 4-feature datasets
  (`iris`, `wine`, `breast_cancer`, `digits_01`).
- Measured cost: SO(2)'s average rank moved from 5.00 to 10.50 under VQC and
  from 4.38 to 10.19 under kernel-SVM — top third to bottom half. Every other
  encoding was unaffected, so the distortion fell entirely on the encoding with
  the missing cells. It is also the encoding the guide recommends for
  `symmetry="rotation"`.
- Nothing is imputed now. Incomplete methods are excluded from the omnibus and
  post-hoc and reported in `ComparisonResult.excluded` alongside the datasets
  they lacked, and the critical-difference figures carry that exclusion plus
  the omnibus statistics as an on-figure caption. Descriptive ranks for such
  methods remain available via `average_ranks(..., missing="available")`, which
  the documentation marks as not comparable across differing competitor counts.
- The bundled atlas (`master_summary.json`) is unaffected: it stores scalar
  per-encoding accuracies, not the per-dataset rank matrix these figures build.

#### CI failed on Windows: backend checks were written as bash heredocs
- The backend-availability guards added alongside the SO(2) fix were embedded
  in `.github/workflows/ci.yml` as `python - <<'EOF'` heredocs. GitHub's
  `windows-latest` runners default to **PowerShell**, which cannot parse that
  syntax, so both Windows jobs died with a `ParserError` at the guard step.
  That step ran *before* the test step, so **the test suite did not execute on
  Windows at all** for that commit. Every other job passed, including the new
  Backend Consistency job; the `.[dev,all]` install itself was fine (Cirq
  installed cleanly on Windows Server 2025 for Python 3.11 and 3.12).
- The policy now lives in the test suite instead of in shell, so it is
  platform-independent by construction and can be reproduced locally:

  ```bash
  ENCODING_ATLAS_REQUIRE_ALL_BACKENDS=1 pytest
  ```

  Unset, a missing optional backend still skips, which is what a contributor
  with a partial install needs. Set, it is a failure. `tests/conftest.py`
  refuses to start a strict session when any advertised backend is missing, and
  `tests/_backends.py::require_backend` turns individual skips into failures.
  Both branches are covered by `tests/test_backend_policy.py`, and both were
  verified end-to-end by shadowing a backend with an unimportable stub:
  strict + missing exits 4 before collection, permissive + missing exits 0 with
  named skips.
- No shell heredoc remains in any workflow, and the only job that runs on
  Windows now contains just two `run:` steps, both unchanged from versions
  already proven green on `windows-latest`.

#### SO2EquivariantFeatureMap produced the wrong state on Qiskit
- `SO2EquivariantFeatureMap._to_qiskit` passed its amplitude vector straight to
  `QuantumCircuit.initialize`, which reads amplitudes **LSB-first**, while
  `_to_pennylane` hands the same vector to `qml.StatePrep`, which reads it
  **MSB-first**. The two backends therefore prepared different physical states,
  and the library's global MSB conversion compounded the error rather than
  correcting it. Measured PennyLane-vs-Qiskit fidelity was **0.0006** — very
  nearly orthogonal — and wrong on every input tested. PennyLane and Cirq
  always agreed with each other, so only the Qiskit path was affected.
- Anyone running this encoding on Qiskit got silently incorrect states, and so
  incorrect kernels, accuracies and diagnostics. It is the encoding the guide
  recommends for `symmetry="rotation"`. The bundled atlas is unaffected: the
  empirical pipeline runs on PennyLane.
- The MSB-to-LSB permutation now lives in one tested place,
  `encoding_atlas.encodings._qubit_order.msb_to_lsb_amplitudes`, used by both
  encodings that prepare a state from an amplitude vector
  (`SO2EquivariantFeatureMap` and `AmplitudeEncoding`, which already had the
  conversion inline and is unchanged in behaviour).

### Added

#### Cross-backend consistency is now verified systematically
- `tests/integration/test_all_backends_consistency.py` checks that **every**
  registry encoding prepares the same state on PennyLane, Qiskit and Cirq,
  across several feature counts and fixed inputs. The parametrisation is
  generated from the registry, so a newly added encoding is covered
  automatically. Previously only five encodings were spot-checked, which is
  how the SO(2) bug shipped. On failure the test reports whether the states are
  bit-reversed (a qubit-ordering bug) or genuinely different.
- Unit tests for the amplitude permutation, checked against an explicit
  bit-reversal reference rather than against itself, plus its involution,
  permutation and norm-preservation properties.

### Changed

#### CI verifies what the package advertises
- The test job installs `.[dev,all]`, so Cirq is present. It previously
  installed `.[dev,qiskit]`, which meant **157 tests silently skipped** —
  covering 14 of the 16 encodings' Cirq implementations — while CI reported
  green. Measured: 1 test skips locally with all three backends, 158 without
  Cirq.
- Both CI jobs that exercise backends set
  `ENCODING_ATLAS_REQUIRE_ALL_BACKENDS=1`, so a backend that failed to install
  is a loud failure rather than a green run with an unverified guarantee. A new
  **Backend Consistency** job runs the cross-backend suite under that flag,
  giving the multi-framework claim its own named result.
- The coverage gate is enabled at 80% (`--cov-fail-under=80`, previously `=0`,
  which overrode the `fail_under = 80` already set in `pyproject.toml`).
  Current coverage is 87.7%.
- Removed the stale comment claiming no tests used the backend markers; the
  tests existed and guarded themselves with availability checks, they were
  simply never given the dependency.

### Added

#### scikit-learn estimator compatibility
- `VQCClassifier`, `VQCRegressor`, `QuantumKernelClassifier` and
  `QuantumKernelRegressor` now follow scikit-learn's estimator contract,
  inheriting `BaseEstimator` plus `ClassifierMixin` / `RegressorMixin`. They
  compose with `cross_val_score`, `cross_validate`, `GridSearchCV`,
  `learning_curve`, `Pipeline`, `VotingClassifier`, `CalibratedClassifierCV`
  and anything else that clones or introspects an estimator. Previously only a
  bare `Pipeline` worked: everything that calls `clone` or `get_params`
  failed.
- The encoding is itself a tunable hyper-parameter, so
  `GridSearchCV(clf, {"encoding": [...], "C": [...]})` searches over encodings
  and model settings together.
- `QuantumKernelClassifier.decision_function`, enabling ROC-AUC and
  probability calibration for the kernel path.
- Standard fitted attributes: `classes_` (classifiers) and `n_features_in_`
  (all four), with `predict` checking the feature count against `fit`.

### Changed
- **Hyper-parameters are validated at `fit` time rather than in `__init__`.**
  `QuantumKernelClassifier(enc, C=-1)` now constructs and raises when fitted,
  exactly as `SVC(C=-1)` does. scikit-learn requires this: `clone` and
  `set_params` rebuild an estimator from `get_params()`, so the constructor
  must store its arguments verbatim. The error types and messages are
  unchanged — only where they are raised.
- Fitted state no longer exists before `fit`. `params_`, `loss_history_` and
  `status_` are set by `fit`; `get_final_loss()` still returns `None` on an
  unfitted model. Calling `predict` before `fit` now raises
  `NotFittedError`, which subclasses `ValueError`, so existing handlers keep
  working.
- The VQC estimators read `encoding.n_qubits` at `fit` time instead of
  caching it in `__init__`, where it would have gone stale after
  `set_params(encoding=...)`.
- `QuantumKernelRegressor.score` and `VQCRegressor.score` are inherited from
  `RegressorMixin` instead of being reimplemented. The value is unchanged
  (both were already `sklearn.metrics.r2_score`) and `sample_weight` is now
  supported. `score` on the classifiers is likewise `accuracy_score` via
  `ClassifierMixin`.
- Private fitted state on the kernel estimators is exposed under scikit-learn's
  trailing-underscore convention (`svm_`, `model_`, `X_train_`,
  `train_states_`).

#### Feature-scaling sensitivity
- `encoding_atlas.analysis.scan_feature_ranges` — measures an encoding's
  kernel-target alignment and kernel concentration across candidate
  feature-scaling ranges, training-free.
  `encoding_atlas.analysis.recommend_feature_range` returns the
  highest-aligned range; `scale_to_range` is the per-feature min-max helper
  behind both, with a `reference=` argument so a scaler can be fitted on a
  training split and applied to a test split.
- `FeatureRangeResult`, `FeatureRangeScan` (with `best`, `best_range`,
  `alignment_spread`, `at`) and `DEFAULT_FEATURE_RANGES`.
- `screen_encodings(..., feature_ranges=...)` ranks over (encoding, range)
  pairs rather than encodings alone; `ScreenedEncoding` gains `feature_range`
  and `ScreeningResult` gains `feature_ranges`.
- New concept page: *Feature Scaling*.

  Why it matters: the range features are scaled into decides how much of each
  qubit's rotation period the data sweeps. A full `[0, 2π]` sweep is the regime
  the concentration scan identifies as maximally Haar-like, so it drives
  kernels onto their concentration floor. Measured across four ranges, IQP
  swings **34 accuracy points**, and `[0, 2π]` is the worst range for every
  entangling map — while `angle` is the one encoding that prefers it.

#### Bundled scaling-sensitivity dataset
- `encoding_atlas/atlas/data/scaling_sensitivity.json`, generated by the new
  deterministic `experiments/scaling_scan.py` (`--check` verifies the committed
  file still matches the code).
- Atlas API: `get_scaling_profile`, `list_scaling_profiles`,
  `scaling_sensitive_encodings`, `expressibility_accuracy_correlation`,
  `scaling_metadata`, plus the `ScalingProfile` / `ScalingPoint` types.
- The dataset records the study's expressibility-versus-accuracy correlation
  **recomputed at each range**, with expressibility re-measured in the same
  regime. It reverses sign: ρ = +0.78 (p = 0.0009) at `[0, π/2]` against
  ρ = −0.54 (p = 0.046) at `[0, 2π]`, the range the empirical pipeline uses,
  both on the 14 encodings the atlas records an expressibility for. The
  published direction is reproduced at the published range; the point is that
  the association is set by a preprocessing choice rather than being a
  scaling-invariant property, so the range belongs in any report of it.

### Fixed
- The benchmark fitted its feature scaler on the whole dataset *before*
  splitting, so each test fold's own minimum and maximum set the scale it was
  later evaluated under — a mild but real leak in an otherwise carefully
  paired protocol. `evaluate_encoding` and `EncodingBenchmark` now scale per
  fold, fitting on the training split only (`_scale_fold`). Test values may
  land outside the target range as a result, which is correct and matches
  `sklearn.preprocessing.MinMaxScaler`.

### Changed
- `benchmark.runner._scale_features` delegates to
  `analysis.scaling.scale_to_range`, so the benchmark and the sensitivity
  analysis share one definition of min-max scaling.
- Binary-label validation moved to
  `encoding_atlas.analysis.validate_binary_labels` (now public), where the
  alignment functions it guards live; `guide.screening` and `analysis.scaling`
  both use it instead of keeping separate copies.

#### Data-driven encoding screening
- `encoding_atlas.guide.screen_encodings` — ranks candidate encodings by
  kernel-target alignment measured on the caller's own ``(X, y)``, with no
  training. This is the workflow the benchmark's validated predictor implies:
  score, keep the encodings that do well, train only those. All 16 candidates
  on 100 samples take about a second.
- `ScreeningResult` (with `top()`, `names()`, `best()`, `get()`) and
  `ScreenedEncoding`, which carries the built instance so the shortlist can be
  trained without rebuilding. Encodings that cannot be constructed at the
  caller's feature count are reported in `skipped` with the reason instead of
  raising.
- Supports restricted candidate sets, stratified and seeded sub-sampling for
  large inputs, finite-shot kernels via `shots=`, and optional kernel
  concentration annotation. Concentration deliberately does **not** affect the
  ranking: at a fixed circuit width alignment already ranks the Haar-floor
  encodings last, so a veto would be redundant.
- Documented honestly: on the benchmark's eight datasets the top-3 shortlist
  reaches 0.973 mean accuracy against an oracle's 0.974, but the single top
  pick (0.960) is not meaningfully better than always choosing `angle`
  (0.958). The shortlist is the deliverable, not the top-1.
- New guide page: *Screening on Your Data*.

#### Kernel-target alignment in the atlas
- `kernel_target_alignment` is now a first-class, rankable atlas metric:
  `rank_encodings(by="kernel_target_alignment")`. Previously the atlas
  exposed `expressibility` — the predictor the study *refutes* (rho = -0.68) —
  while the predictor it *validates* (rho = +0.91) was measured by the pipeline
  and then dropped during consolidation.
- `experiments/report.py` now carries the measured alignment through, so the
  column is derived from existing Stage 6b measurements with no new simulation,
  averaged over (configuration, dataset) pairs — the identical rule the
  `kernel_accuracy` column already used.

#### Analysis
- `encoding_atlas.analysis.summarize_kernel_concentration` — derives the
  concentration statistics from an already-computed kernel, so a caller that
  holds one (screening, for instance) does not pay for a second simulation.

### Fixed
- Kernel-target alignment assumed labels were literally `{0, 1}` and silently
  returned wrong values for any other two-class convention: an ideal kernel
  scored 1.00 for `{0, 1}` but 0.20 for `{1, 2}` and 0.80 for `{-1, +1}`.
  Two-class labels are now mapped to `{-1, +1}` by partition, so the score
  depends on the split rather than on how the classes are spelled. `{0, 1}`
  behaviour, and therefore every published number, is unchanged; continuous
  and multi-class targets keep the previous linear mapping.
- `encoding_atlas.benchmark.kernel` had a second, independently maintained copy
  of both alignment functions that had drifted from the analysis package's.
  It now re-exports the single definition, so the metric the benchmark records
  and the metric users screen with cannot disagree.

### Changed
- The benchmark-matched encoding parameters now live in the package
  (`encoding_atlas.guide._candidates.BENCHMARK_PARAMS`) as the single source of
  truth for both screening and `experiments/concentration_scan.py`, which
  re-exports them as `ENCODING_PARAMS` for backwards compatibility.

#### Kernel-concentration analysis
- `encoding_atlas.analysis.compute_kernel_concentration` — measures how close an
  encoding's fidelity kernel sits to the Haar floor at a given circuit width.
  The order parameter is `concentration_ratio`, the off-diagonal kernel
  variance divided by the Haar-random variance `(d-1)/(d²(d+1))`; a value near
  1 means the kernel has collapsed to the identity up to sampling noise and a
  kernel method cannot generalize from it at any shot budget.
- `encoding_atlas.analysis.estimate_concentration_scaling` — sweeps a factory
  across circuit widths and fits a log-linear decay model, returning
  `decay_rate`, `mean_decay_rate`, `haar_normalized_slope`, `r_squared`, a
  `concentration_horizon()`, and `shots_per_entry_at()` extrapolation. Widths a
  factory rejects are recorded in `skipped` rather than aborting the sweep.
- `encoding_atlas.analysis.haar_kernel_moments` — the Haar reference moments.
- `ConcentrationResult` (with `shots_for_dataset()`) and `ScalingResult`.
- New concept page: *Kernel Concentration*, documenting why the kernel variance
  and not its mean is the order parameter — a product of independent random
  single-qubit states has mean overlap exactly `2⁻ⁿ`, identical to Haar.

#### Finite-shot kernel estimation
- `encoding_atlas.analysis.sample_shot_kernel` — models the compute-uncompute
  estimator, whose all-zeros count is exactly `Binomial(shots, K)`, so shot
  realism costs one random draw rather than a second simulation.
- `shots=` / `seed=` on `compute_fidelity_kernel`,
  `compute_kernel_target_alignment`, `compute_geometric_difference`, and
  `compute_effective_dimension`. Sampled kernels stay symmetric with an exact
  unit diagonal but are not PSD; project with `benchmark.kernel.ensure_psd`.

#### Bundled concentration dataset
- `encoding_atlas/atlas/data/concentration.json` — the scan for all 16
  encodings across 2–8 qubits, generated by the new deterministic
  `experiments/concentration_scan.py` (`--check` verifies the committed file
  still matches the code).
- Atlas API: `get_concentration_profile`, `list_concentration_profiles`,
  `concentrated_encodings`, `concentration_metadata`, plus the
  `ConcentrationProfile` / `ConcentrationPoint` types.
- The four encodings whose kernels reach the Haar floor (`iqp`,
  `zz_feature_map`, `pauli_feature_map`, `hamiltonian`) are exactly the four
  with expressibility ≈ 0.999 and four of the five worst-ranked in the
  benchmark — the mechanism behind the refuted hypothesis H1.

### Changed
- `profile_encoding` now always computes a concentration axis
  (`kernel_concentration_ratio`, `kernel_offdiagonal_mean`,
  `kernel_shots_per_entry`, `kernel_is_concentrated`) and accepts
  `concentration_samples`. The axis is data-free by default and uses `X` when
  supplied. Existing metrics and failure handling are unchanged.
- `guide.Recommendation` gains a `scale_warning` field (default `None`), set
  when the recommended encoding is measured at the Haar floor for the caller's
  feature count. It is kept out of `explanation` so existing callers that
  format the rationale are unaffected.

## [0.1.0] - 2026-02-07

### Added

#### Core Framework
- Abstract base class `BaseEncoding` with thread-safe property caching
- `EncodingProperties` dataclass for encoding metrics (qubits, depth, gates, simulability)
- Encoding registry system with `register_encoding`, `get_encoding`, `list_encodings`
- Type definitions and protocols for static type checking (`py.typed` marker)

#### Encoding Implementations (16 encodings)
- **AngleEncoding**: Single-qubit rotations (RX, RY, RZ) with configurable repetitions
- **AmplitudeEncoding**: Logarithmic qubit encoding via state amplitudes
- **BasisEncoding**: Binary encoding into computational basis states
- **IQPEncoding**: Instantaneous Quantum Polynomial circuits with diagonal gates
- **ZZFeatureMap**: Pauli-ZZ entangling feature map
- **PauliFeatureMap**: Configurable Pauli rotation feature maps
- **HardwareEfficientEncoding**: NISQ-friendly ansatz with hardware-native gates
- **DataReuploading**: Multi-layer data re-uploading with trainable parameters
- **SymmetryInspiredFeatureMap**: Symmetry-inspired quantum feature maps
- **HamiltonianEncoding**: Time-evolution based encoding
- **HigherOrderAngleEncoding**: Higher-order polynomial angle encoding
- **QAOAEncoding**: QAOA-inspired encoding structure
- **TrainableEncoding**: Parameterized encoding with learnable gate parameters
- **SO2EquivariantFeatureMap**: SO(2) rotation-equivariant quantum feature map
- **CyclicEquivariantFeatureMap**: Cyclic group equivariant quantum feature map
- **SwapEquivariantFeatureMap**: Permutation-equivariant quantum feature map

#### Multi-Backend Support
- PennyLane backend (primary)
- Qiskit backend with full circuit generation
- Cirq backend with moment-based circuit construction
- Consistent quantum states across all backends (verified by cross-backend tests)

#### Analysis Tools
- Expressibility calculation
- Entanglement capability metrics
- Trainability estimation via variance of parameter-shift gradients
- Classical simulability analysis (Clifford detection, matchgate detection, entanglement-based)
- Resource counting and estimation (gate counts, depth, qubit requirements)

#### Experiment Framework
- Experiment runner with checkpointing support
- VQC (Variational Quantum Classifier) experiment pipeline
- Kernel-based classification experiment pipeline
- Noise model support for realistic hardware simulation

#### Utilities
- Benchmarking framework for encoding comparison
- Decision guide system for encoding selection
- Visualization tools (`pip install encoding-atlas[visualization]` for matplotlib support)

#### Quality & Infrastructure
- Comprehensive test suite with 80%+ coverage requirement
- Thread safety with double-checked locking pattern
- Pickle serialization support for distributed computing
- Input validation with clear error messages (NaN, Inf, complex, shape)
- Defensive copying for thread-safe input handling

#### Documentation
- NumPy-style docstrings with mathematical background
- Academic references for each encoding
- Usage examples in all public APIs
- Module-level documentation with preprocessing guidance

### Backend Requirements
- **Core**: NumPy ≥1.21, SciPy ≥1.7, PennyLane ≥0.33, scikit-learn ≥1.0
- **Optional**: Qiskit ≥1.0 (`pip install encoding-atlas[qiskit]`)
- **Optional**: Cirq ≥1.0 (`pip install encoding-atlas[cirq]`)
- **Optional**: matplotlib ≥3.5 (`pip install encoding-atlas[visualization]`)
- **All backends**: `pip install encoding-atlas[all]`

### Python Support
- Python 3.9, 3.10, 3.11, 3.12

---

[Unreleased]: https://github.com/encoding-atlas/quantum-encoding-atlas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/encoding-atlas/quantum-encoding-atlas/releases/tag/v0.1.0
