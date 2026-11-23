"""Stage 7 joins metrics from the same circuit, never just the same width.

Stages 1-5 sweep each encoding over several configurations at one feature
count (``reps``, ``order``, ``n_layers``, ``entanglement``). Keying the join on
``n_features`` alone let whichever configuration was written last win, which

- paired Stage 1 depth with Stage 5 gradient variance from *different*
  circuits in H4 (16 of 58 published pairs), and
- took the atlas's per-encoding depth, expressibility, entanglement and
  trainability from circuits other than the one whose accuracy it reports
  (angle reps=2 instead of 1, SO(2) L=1 instead of 2, ...), moving the
  Pareto front.

The synthetic stage directories below are built so that a width-keyed join
gives a different, wrong answer: neighbouring configurations are written both
before and after the benchmarked one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.tradeoff import (
    analyze_barren_plateau_onset,
    benchmarked_configurations,
    build_encoding_profiles,
    build_trainability_records,
    config_key,
    resolve_encoding_params,
)


def _record(name: str, params: dict[str, Any], **result: Any) -> dict[str, Any]:
    return {
        "encoding_name": name,
        "encoding_params": params,
        "status": "success",
        "result": result,
    }


def _write_stage(root: Path, stage: str, records: list[dict[str, Any]]) -> str:
    directory = root / stage
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps({"results": records}), encoding="utf-8"
    )
    return str(directory)


def _accuracy(value: float) -> dict[str, Any]:
    return {"datasets": {"moons": {"status": "success", "mean_test_accuracy": value}}}


@pytest.fixture
def stage_dirs(tmp_path: Path) -> dict[str, str]:
    """IQP benchmarked at reps=2 (entanglement left at its default, 'full'),
    and angle at its default reps=1 — which a "prefer reps=2" heuristic, or a
    last-write-wins join, both get wrong."""
    vqc = [
        _record("iqp", {"n_features": 2, "reps": 2}, **_accuracy(0.70)),
        _record("iqp", {"n_features": 4, "reps": 2}, **_accuracy(0.80)),
        _record("angle", {"n_features": 4, "rotation": "Y"}, **_accuracy(0.90)),
        _record("classical_svm_rbf", {}, **_accuracy(0.99)),
    ]
    kernel = [_record("iqp", {"n_features": 4, "reps": 2}, **_accuracy(0.85))]
    # Stage 1 sweeps an extra axis and writes the benchmarked circuit between
    # two neighbours at every width.
    resources = []
    for nf in (2, 4, 6):
        for reps, ent in ((1, "full"), (2, "full"), (3, "full"), (2, "linear")):
            resources.append(
                _record(
                    "iqp",
                    {"n_features": nf, "reps": reps, "entanglement": ent},
                    depth=100 * reps + nf + (50 if ent == "linear" else 0),
                    gate_count=1,
                    two_qubit_gates=1,
                    parameter_count=0,
                )
            )
    resources += [
        _record(
            "angle",
            {"n_features": nf, "reps": reps},
            depth=reps,
            gate_count=nf * reps,
            two_qubit_gates=0,
            parameter_count=0,
        )
        for nf in (2, 4)
        for reps in (1, 2, 3)
    ]
    trainability = [
        _record(
            "iqp",
            {"n_features": nf, "reps": reps},
            gradient_variance=1.0 / (10 * reps + nf),
            trainability_estimate=0.1 * reps + 0.01 * nf,
            barren_plateau_risk="low",
        )
        for nf in (2, 4, 6)
        for reps in (1, 2, 3)
    ] + [
        _record(
            "angle",
            {"n_features": nf},
            gradient_variance=0.3,
            trainability_estimate=0.7,
            barren_plateau_risk="low",
        )
        for nf in (2, 4)
    ]
    expressibility = [
        _record(
            "iqp",
            {"n_features": nf, "reps": reps},
            expressibility=0.1 * reps + 0.01 * nf,
        )
        for nf in (2, 4)
        for reps in (2, 1)  # benchmarked first, neighbour last
    ]
    return {
        "vqc": _write_stage(tmp_path, "vqc", vqc),
        "kernel": _write_stage(tmp_path, "kernel", kernel),
        "resources": _write_stage(tmp_path, "resources", resources),
        "trainability": _write_stage(tmp_path, "trainability", trainability),
        "expressibility": _write_stage(tmp_path, "expressibility", expressibility),
    }


# =============================================================================
# Configuration identity
# =============================================================================


class TestConfigIdentity:
    def test_defaults_complete_an_implicit_parameter(self) -> None:
        """Stage 5 omits IQP's entanglement; Stage 1 records the default."""
        assert config_key("iqp", {"n_features": 4, "reps": 1}) == config_key(
            "iqp", {"n_features": 4, "reps": 1, "entanglement": "full"}
        )
        assert config_key("iqp", {"n_features": 4, "reps": 1}) != config_key(
            "iqp", {"n_features": 4, "reps": 1, "entanglement": "linear"}
        )

    def test_explicit_default_equals_omitted(self) -> None:
        assert config_key("angle", {"n_features": 4}) == config_key(
            "angle", {"n_features": 4, "rotation": "Y", "reps": 1}
        )
        assert config_key("angle", {"n_features": 4}) != config_key(
            "angle", {"n_features": 4, "reps": 2}
        )

    def test_drop_identifies_a_family_across_widths(self) -> None:
        a = config_key("iqp", {"n_features": 2, "reps": 2}, drop=("n_features",))
        b = config_key("iqp", {"n_features": 8, "reps": 2}, drop=("n_features",))
        assert a == b

    def test_json_round_trip_does_not_change_the_key(self) -> None:
        params = {"n_features": 4, "paulis": ("Z", "ZZ")}
        round_tripped = json.loads(json.dumps(params))
        assert config_key("pauli_feature_map", params) == config_key(
            "pauli_feature_map", round_tripped
        )

    def test_unknown_names_are_compared_as_recorded(self) -> None:
        assert resolve_encoding_params("classical_svm_rbf", {"C": 1.0}) == {"C": 1.0}

    def test_resolution_covers_every_constructor_parameter(self) -> None:
        resolved = resolve_encoding_params("so2_equivariant", {})
        assert resolved["max_angular_momentum"] == 1
        assert resolved["n_features"] == 2


class TestBenchmarkedConfigurations:
    def test_anchor_comes_from_stage_six(self, stage_dirs: dict[str, str]) -> None:
        anchors = benchmarked_configurations(stage_dirs)
        assert set(anchors) == {"iqp", "angle"}  # baselines are not encodings
        assert anchors["iqp"] == config_key("iqp", {"reps": 2}, drop=("n_features",))

    def test_two_benchmarked_configurations_are_rejected(self, tmp_path: Path) -> None:
        dirs = {
            "vqc": _write_stage(
                tmp_path,
                "vqc",
                [_record("iqp", {"n_features": 4, "reps": 2}, **_accuracy(0.8))],
            ),
            "kernel": _write_stage(
                tmp_path,
                "kernel",
                [_record("iqp", {"n_features": 4, "reps": 3}, **_accuracy(0.8))],
            ),
        }
        with pytest.raises(ValueError, match="more than one configuration"):
            benchmarked_configurations(dirs)

    def test_no_stage_six_means_no_anchor(self) -> None:
        assert benchmarked_configurations({}) == {}


# =============================================================================
# Headline metrics and scaling curves
# =============================================================================


class TestHeadlineMetrics:
    def test_metrics_come_from_the_benchmarked_circuit(
        self, stage_dirs: dict[str, str]
    ) -> None:
        profile = build_encoding_profiles(stage_dirs)["iqp"]
        # reps=2, entanglement='full', n_features=4 — not reps=3 (written
        # after it) nor entanglement='linear' (written last).
        assert profile["depth"] == 204
        assert profile["expressibility"] == pytest.approx(0.24)
        assert profile["trainability_estimate"] == pytest.approx(0.24)
        assert profile["gradient_variance"] == pytest.approx(1.0 / 24)

    def test_default_configuration_is_not_mistaken_for_reps_two(
        self, stage_dirs: dict[str, str]
    ) -> None:
        """angle was benchmarked at reps=1; the atlas used to report reps=2."""
        profile = build_encoding_profiles(stage_dirs)["angle"]
        assert profile["depth"] == 1
        assert profile["gate_count"] == 4
        assert profile["depth_scaling"] == {2: 1, 4: 1}

    def test_unmeasured_circuit_stays_unmeasured(
        self, stage_dirs: dict[str, str], tmp_path: Path
    ) -> None:
        """Only reps=1 measured: borrowing it would misattribute the metric."""
        stage_dirs = dict(stage_dirs)
        stage_dirs["noise"] = _write_stage(
            tmp_path,
            "noise",
            [
                _record(
                    "iqp",
                    {"n_features": 4, "reps": 1},
                    noisy_medium_fidelity_decay=0.3,
                    noisy_medium_expressibility_change=0.1,
                )
            ],
        )
        profile = build_encoding_profiles(stage_dirs)["iqp"]
        assert profile["noise_resilience"] is None
        assert profile["noisy_medium_fidelity_decay"] is None

    def test_conflicting_duplicates_are_an_error(
        self, stage_dirs: dict[str, str], tmp_path: Path
    ) -> None:
        stage_dirs = dict(stage_dirs)
        stage_dirs["simulability"] = _write_stage(
            tmp_path,
            "simulability",
            [
                _record(
                    "iqp",
                    {"n_features": 4, "reps": 2},
                    simulability_class="a",
                    is_simulable=False,
                ),
                _record(
                    "iqp", {"n_features": 4}, simulability_class="b", is_simulable=True
                ),
            ],
        )
        with pytest.raises(ValueError, match="Two different results"):
            build_encoding_profiles(stage_dirs)

    def test_scaling_curves_hold_the_configuration_fixed(
        self, stage_dirs: dict[str, str]
    ) -> None:
        profile = build_encoding_profiles(stage_dirs)["iqp"]
        assert profile["depth_scaling"] == {2: 202, 4: 204, 6: 206}
        assert profile["trainability_scaling"] == pytest.approx(
            {2: 1 / 22, 4: 1 / 24, 6: 1 / 26}
        )
        assert profile["expressibility_scaling"] == pytest.approx({2: 0.22, 4: 0.24})

    def test_encodings_without_a_benchmark_have_no_headline(
        self, stage_dirs: dict[str, str]
    ) -> None:
        profiles = build_encoding_profiles(stage_dirs)
        assert profiles["basis"]["depth"] is None
        assert profiles["basis"]["trainability_records"] is None


# =============================================================================
# H4 pairs depth and gradient variance of the same circuit
# =============================================================================


class TestTrainabilityRecords:
    def test_every_stage_five_circuit_gets_its_own_depth(
        self, stage_dirs: dict[str, str]
    ) -> None:
        records = build_encoding_profiles(stage_dirs)["iqp"]["trainability_records"]
        assert len(records) == 9  # 3 widths x 3 reps, all measured by Stage 1
        for record in records:
            reps, nf = record["config"]["reps"], record["n_features"]
            assert record["config"]["entanglement"] == "full"
            assert record["depth"] == 100 * reps + nf
            assert record["gradient_variance"] == pytest.approx(1.0 / (10 * reps + nf))

    def test_circuits_without_a_depth_are_left_out(self) -> None:
        resources = [_record("iqp", {"n_features": 4, "reps": 2}, depth=6)]
        trainability = [
            _record("iqp", {"n_features": 4, "reps": 2}, gradient_variance=0.1),
            _record("iqp", {"n_features": 4, "reps": 5}, gradient_variance=0.2),
        ]
        records = build_trainability_records(resources, trainability)["iqp"]
        assert [r["config"]["reps"] for r in records] == [2]

    def test_h4_uses_one_point_per_circuit(self, stage_dirs: dict[str, str]) -> None:
        profiles = build_encoding_profiles(stage_dirs)
        result = analyze_barren_plateau_onset(profiles)
        assert result["n_points"] == 9 + 2  # IQP's 9 circuits plus angle's 2
        # Same-circuit pairs: variance falls strictly with depth.
        slope = result["per_family"]["IQP-Based"]["slope"]
        assert slope < 0

    def test_h4_falls_back_to_the_headline_configuration(self) -> None:
        profiles = {
            f"enc{i}": {
                "family": "A" if i % 2 else "B",
                "trainability_records": None,
                "gradient_variance": 10.0 ** (-i),
                "depth": i + 1,
            }
            for i in range(8)
        }
        assert analyze_barren_plateau_onset(profiles)["n_points"] == 8
