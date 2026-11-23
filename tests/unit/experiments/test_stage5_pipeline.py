"""Stage 5 handler and validator under the classifier-gradient definition."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from encoding_atlas import AngleEncoding, IQPEncoding
from experiments.runner import _handle_trainability
from experiments.validate_stage5_trainability import (
    _validate_light_cone,
    validate_trainability_result,
    validate_width_invariance,
)


def _config(**analysis_params: Any) -> SimpleNamespace:
    return SimpleNamespace(analysis_params=analysis_params, backend="pennylane")


class TestHandler:
    def test_result_is_json_serialisable_with_provenance(self) -> None:
        result = _handle_trainability(
            IQPEncoding(n_features=3, reps=2), _config(n_samples=12), seed=0
        )
        json.dumps(result)  # raises if any ndarray leaked through
        assert result["observable"] == "pauli_z"
        assert result["n_var_layers"] == 2
        assert (
            len(result["live_parameter_mask"])
            == len(result["per_parameter_variance"])
            == 6
        )
        assert result["n_live_parameters"] == 3

    def test_config_values_reach_the_estimator(self) -> None:
        result = _handle_trainability(
            AngleEncoding(n_features=2),
            _config(n_samples=12, observable="global_z", n_var_layers=3),
            seed=0,
        )
        assert result["observable"] == "global_z"
        assert result["n_var_layers"] == 3
        assert len(result["per_parameter_variance"]) == 6


def _valid_result(**overrides: Any) -> dict[str, Any]:
    result = {
        "trainability_estimate": 0.7,
        "gradient_variance": 0.3,
        "barren_plateau_risk": "low",
        "per_parameter_variance": [0.4, 0.2, 1e-33, 1e-34, 0.3, 1e-33, 1e-34, 1e-35],
        "live_parameter_mask": [True, True, False, False, True, False, False, False],
        "n_live_parameters": 3,
        "n_var_layers": 2,
        "observable": "pauli_z",
        "n_successful_samples": 500,
    }
    result.update(overrides)
    return result


class TestValidator:
    def test_a_valid_result_passes(self) -> None:
        validation = validate_trainability_result(
            "angle", {"n_features": 4}, _valid_result()
        )
        assert validation.all_passed, [
            c.message for c in validation.results if not c.passed
        ]

    def test_zero_variance_is_a_failure_for_every_encoding(self) -> None:
        """The old exemptions for basis and reps=1 IQP must not come back."""
        for name, params in (
            ("basis", {"n_features": 4}),
            ("iqp", {"n_features": 4, "reps": 1}),
        ):
            validation = validate_trainability_result(
                name, params, _valid_result(gradient_variance=0.0)
            )
            failed = {c.check_name for c in validation.results if not c.passed}
            assert "gradient_variance_positive" in failed

    def test_results_from_the_old_definition_are_rejected(self) -> None:
        old = {
            k: v
            for k, v in _valid_result().items()
            if k
            not in {
                "observable",
                "n_var_layers",
                "live_parameter_mask",
                "n_live_parameters",
            }
        }
        checks = _validate_light_cone(old)
        assert [c.check_name for c in checks] == ["provenance_present"]
        assert not checks[0].passed

    def test_a_live_dead_parameter_is_caught(self) -> None:
        variances = _valid_result()["per_parameter_variance"]
        variances[2] = 1e-4  # outside the light cone, but not zero
        checks = {
            c.check_name: c
            for c in _validate_light_cone(
                _valid_result(per_parameter_variance=variances)
            )
        }
        assert not checks["dead_parameters_have_zero_variance"].passed

    def test_inconsistent_live_count_is_caught(self) -> None:
        checks = {
            c.check_name: c
            for c in _validate_light_cone(_valid_result(n_live_parameters=8))
        }
        assert not checks["provenance_consistent"].passed

    def test_width_dilution_is_caught(self) -> None:
        """Averaging over all 2n parameters halves the variance from n=4 to 8."""

        def entry(nf: int, gv: float) -> dict[str, Any]:
            return {
                "status": "success",
                "encoding_name": "angle",
                "encoding_params": {"n_features": nf},
                "result": {"gradient_variance": gv},
            }

        diluted = validate_width_invariance(
            [entry(2, 0.22), entry(4, 0.11), entry(8, 0.055)]
        )
        flat = validate_width_invariance(
            [entry(2, 0.294), entry(4, 0.294), entry(8, 0.304)]
        )
        assert [c.passed for c in diluted] == [False]
        assert [c.passed for c in flat] == [True]

    @pytest.mark.parametrize("risk", ["high"])
    def test_non_entangling_barren_plateau_is_a_failure(self, risk: str) -> None:
        validation = validate_trainability_result(
            "basis", {"n_features": 4}, _valid_result(barren_plateau_risk=risk)
        )
        failed = {c.check_name for c in validation.results if not c.passed}
        assert "non_entangling_no_barren_plateau" in failed
