"""The rule table must never admit an encoding the user cannot build.

The guide answers "which encoding should I use?" with a name the caller then
passes to ``get_encoding``. If the rule table admits a width the constructor
rejects, the guide hands out a name that raises — which is exactly what it did:
``recommend_encoding(n_features=5, symmetry="general")`` returned
``symmetry_inspired`` with confidence 0.88, and building it raised
``ValueError: rotation symmetry requires even n_features``. Across the
documented input space, 8400 of 190080 queries (4.42%) were unbuildable.

The filter mechanism was never wrong; the data was. ``symmetry_inspired``
lacked ``requires_even_features``, and no field could express "needs at least
two features" at all. So the guard here is a *consistency* check between the
rule table and the encodings themselves, rather than a list of known-bad cases:
a new encoding, or a changed constructor precondition, is caught without anyone
remembering to update a fixture.

The invariant is deliberately one-directional. The rules may be *stricter* than
the constructor — ``max_features`` is advisory for five of the six encodings
that set it, a recommendation ceiling well below what still builds — but they
must never be looser.
"""

from __future__ import annotations

import functools
import itertools
import pathlib
import re
import warnings

import pytest

from encoding_atlas import get_encoding
from encoding_atlas.guide import recommend_encoding
from encoding_atlas.guide.rules import ENCODING_RULES, _passes_hard_constraints

#: Widths to check exhaustively. Spans both parities, the 1-feature degenerate
#: case, and powers of two where amplitude-style encodings change behaviour.
WIDTHS = tuple(range(1, 17))

#: Encodings whose constructor admits only even widths, measured not assumed.
EVEN_ONLY = frozenset({"so2_equivariant", "swap_equivariant", "symmetry_inspired"})

#: Smallest width each constructor actually accepts, measured not assumed.
EXPECTED_MIN = {
    "amplitude": 1,
    "angle": 1,
    "basis": 1,
    "cyclic_equivariant": 2,
    "data_reuploading": 1,
    "hamiltonian": 1,
    "hardware_efficient": 1,
    "higher_order_angle": 2,
    "iqp": 1,
    "pauli_feature_map": 1,
    "qaoa": 1,
    "so2_equivariant": 2,
    "swap_equivariant": 2,
    "symmetry_inspired": 2,
    "trainable": 1,
    "zz_feature_map": 1,
}


def _builds(name: str, n_features: int) -> bool:
    """Whether ``get_encoding`` succeeds at this width, warnings aside."""
    try:
        with warnings.catch_warnings():
            # Encodings warn about impractical widths; a warning is not a refusal.
            warnings.simplefilter("ignore")
            get_encoding(name, n_features=n_features)
    except Exception:
        return False
    return True


def _admits(name: str, n_features: int, **kwargs: object) -> bool:
    """Whether the rule table lets this encoding through its hard filter."""
    rules = ENCODING_RULES[name]
    return _passes_hard_constraints(
        rules,
        n_features=n_features,
        data_type=(kwargs.get("data_type") or (rules["requires_data_type"] or ["continuous"])[0]),  # type: ignore[index]
        symmetry=rules["requires_symmetry"],
        trainable=rules["requires_trainable"],
    )


class TestRulesNeverAdmitAnUnbuildableEncoding:
    """The core invariant, checked exhaustively over every encoding and width."""

    @pytest.mark.parametrize("name", sorted(ENCODING_RULES))
    def test_every_admitted_width_is_constructible(self, name: str) -> None:
        offenders = [w for w in WIDTHS if _admits(name, w) and not _builds(name, w)]
        assert not offenders, (
            f"the rule table admits {name} at n_features={offenders}, but the "
            f"constructor refuses there. Declare the real limit via "
            f"min_features / requires_even_features / requires_n_features."
        )

    def test_the_invariant_has_real_work_to_do(self) -> None:
        """Guard against the check passing because nothing is ever admitted."""
        admitted = sum(1 for name in ENCODING_RULES for w in WIDTHS if _admits(name, w))
        assert admitted > 100, f"only {admitted} (encoding, width) pairs admitted"

    def test_rules_may_be_stricter_than_the_constructor(self) -> None:
        """max_features is advisory, so one-directionality is intentional.

        Pinning this stops a future "fix" from widening the rules to match the
        constructors and undoing the deliberate recommendation ceilings.
        """
        advisory = [
            name
            for name, rules in ENCODING_RULES.items()
            if rules["max_features"] is not None
            and _builds(name, rules["max_features"] + 1)
        ]
        assert sorted(advisory) == [
            "data_reuploading",
            "higher_order_angle",
            "iqp",
            "pauli_feature_map",
            "zz_feature_map",
        ]


class TestDeclaredLimitsMatchReality:
    """The declared constraints must be the measured ones, not approximations."""

    @pytest.mark.parametrize("name", sorted(ENCODING_RULES))
    def test_min_features_is_the_real_minimum(self, name: str) -> None:
        declared = ENCODING_RULES[name]["min_features"]
        assert declared == EXPECTED_MIN[name]
        assert _builds(name, declared) or ENCODING_RULES[name]["requires_even_features"]
        if declared > 1:
            assert not _builds(name, declared - 1)

    @pytest.mark.parametrize("name", sorted(ENCODING_RULES))
    def test_even_flag_matches_the_constructor(self, name: str) -> None:
        buildable = [w for w in WIDTHS if _builds(name, w)]
        really_even = bool(buildable) and all(w % 2 == 0 for w in buildable)
        assert ENCODING_RULES[name]["requires_even_features"] == really_even
        assert (name in EVEN_ONLY) == really_even

    def test_min_features_is_present_and_sane_everywhere(self) -> None:
        for name, rules in ENCODING_RULES.items():
            assert isinstance(rules["min_features"], int), name
            assert rules["min_features"] >= 1, name
            if rules["max_features"] is not None:
                assert rules["min_features"] <= rules["max_features"], name

    def test_exact_feature_count_agrees_with_the_bounds(self) -> None:
        for name, rules in ENCODING_RULES.items():
            exact = rules["requires_n_features"]
            if exact is not None:
                assert exact >= rules["min_features"], name


class TestHardFilterEnforcesMinFeatures:
    """The filter must act on the new field, not merely store it."""

    def test_below_the_minimum_is_rejected(self) -> None:
        assert not _passes_hard_constraints(
            ENCODING_RULES["cyclic_equivariant"], n_features=1, symmetry="cyclic"
        )

    def test_at_the_minimum_is_accepted(self) -> None:
        assert _passes_hard_constraints(
            ENCODING_RULES["cyclic_equivariant"], n_features=2, symmetry="cyclic"
        )

    def test_unknown_feature_count_skips_the_check(self) -> None:
        """``n_features=None`` means "unspecified", not "zero"."""
        assert _passes_hard_constraints(
            ENCODING_RULES["cyclic_equivariant"], n_features=None, symmetry="cyclic"
        )

    def test_odd_width_is_rejected_for_even_only_encodings(self) -> None:
        for name in sorted(EVEN_ONLY):
            rules = ENCODING_RULES[name]
            assert not _passes_hard_constraints(
                rules, n_features=5, symmetry=rules["requires_symmetry"]
            )


class TestRecommendationsAreBuildable:
    """End-to-end: whatever the guide returns, the caller can construct."""

    #: Axes that can change which encoding is chosen at a given width. The
    #: remaining parameters (n_samples, task, hardware) are swept more coarsely
    #: because they shift scores without touching the hard filter.
    GRID = dict(
        n_features=[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16],
        n_samples=[50, 5000],
        task=["classification", "regression"],
        hardware=["simulator", "ibm"],
        priority=["accuracy", "trainability", "speed", "noise_resilience"],
        data_type=["continuous", "binary", "discrete"],
        symmetry=[None, "rotation", "cyclic", "permutation_pairs", "general"],
        trainable=[False, True],
        problem_structure=[None, "combinatorial", "physics_simulation", "time_series"],
        feature_interactions=[None, "polynomial", "custom_pauli"],
    )

    @classmethod
    @functools.lru_cache(maxsize=1)
    def _recommendations(cls) -> frozenset[tuple[str, int]]:
        """Distinct (encoding, n_features) pairs the guide can ever return.

        Cached: the sweep is the expensive part of this module and several
        tests consume the same result.
        """
        keys = list(cls.GRID)
        pairs: set[tuple[str, int]] = set()
        for values in itertools.product(*(cls.GRID[k] for k in keys)):
            kwargs = dict(zip(keys, values))
            result = recommend_encoding(**kwargs)  # type: ignore[arg-type]
            pairs.add((result.encoding_name, int(kwargs["n_features"])))  # type: ignore[arg-type]
        return frozenset(pairs)

    def test_no_recommendation_is_unbuildable(self) -> None:
        offenders = sorted(
            pair for pair in self._recommendations() if not _builds(*pair)
        )
        assert not offenders, (
            f"the guide recommends these (encoding, n_features) pairs, which "
            f"cannot be constructed: {offenders}"
        )

    def test_the_sweep_actually_covers_ground(self) -> None:
        pairs = self._recommendations()
        assert len({name for name, _ in pairs}) >= 12
        assert len({width for _, width in pairs}) == len(self.GRID["n_features"])

    def test_alternatives_are_buildable_too(self) -> None:
        """A user may take the runner-up; it must build at the same width."""
        offenders = []
        for n_features in (1, 3, 5, 7, 9):
            for symmetry in (None, "general", "cyclic"):
                result = recommend_encoding(
                    n_features=n_features, n_samples=500, symmetry=symmetry
                )
                for alternative in getattr(result, "alternatives", ()) or ():
                    name = getattr(alternative, "encoding_name", alternative)
                    if isinstance(name, str) and not _builds(name, n_features):
                        offenders.append((name, n_features))
        assert not offenders, f"unbuildable alternatives: {sorted(set(offenders))}"


class TestReportedRegression:
    """The exact query from the bug report, pinned."""

    def test_odd_width_with_general_symmetry_builds(self) -> None:
        result = recommend_encoding(n_features=5, n_samples=500, symmetry="general")
        assert result.encoding_name != "symmetry_inspired"
        assert _builds(result.encoding_name, 5)

    @pytest.mark.parametrize("n_features", [1, 3, 5, 7, 9])
    def test_every_odd_width_is_safe_under_general_symmetry(
        self, n_features: int
    ) -> None:
        result = recommend_encoding(
            n_features=n_features, n_samples=500, symmetry="general"
        )
        assert _builds(result.encoding_name, n_features)

    def test_single_feature_queries_are_safe(self) -> None:
        for symmetry in (None, "rotation", "cyclic", "permutation_pairs", "general"):
            result = recommend_encoding(n_features=1, n_samples=500, symmetry=symmetry)
            assert _builds(result.encoding_name, 1), (result.encoding_name, symmetry)


ARCHITECTURE_DOC = pathlib.Path("docs/guide/recommendation-architecture.md")


@pytest.mark.skipif(
    not ARCHITECTURE_DOC.exists(), reason="docs not present in this checkout"
)
class TestDocumentedTableMatchesTheRules:
    """The published constraint table must not drift from the rule base.

    A stale table is how the missing ``requires_even_features`` stayed
    invisible: the docs showed ``symmetry_inspired`` with no even-width
    constraint, which was an accurate description of the *rules* and a wrong
    description of the *encoding*.
    """

    @staticmethod
    def _documented() -> dict[str, dict[str, str]]:
        text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
        match = re.search(
            r"^\| Encoding \| requires_data_type \|.*?\n\|[-| ]+\|\n((?:\|.*\n)+)",
            text,
            re.M,
        )
        assert match, "constraint table not found in the architecture doc"
        header = re.search(r"^\| Encoding \|.*$", text, re.M).group(0)
        columns = [c.strip() for c in header.strip("|").split("|")][1:]
        rows: dict[str, dict[str, str]] = {}
        for line in match.group(1).strip().splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0]] = dict(zip(columns, cells[1:]))
        return rows

    def test_every_encoding_appears_exactly_once(self) -> None:
        assert set(self._documented()) == set(ENCODING_RULES)

    @pytest.mark.parametrize("field", ["min_features", "max_features"])
    def test_feature_bounds_match(self, field: str) -> None:
        for name, row in self._documented().items():
            declared = ENCODING_RULES[name][field]  # type: ignore[literal-required]
            shown = row[field]
            expected = (
                "--"
                if declared is None or (field == "min_features" and declared <= 1)
                else str(declared)
            )
            assert shown == expected, f"{name}.{field}: doc says {shown!r}"

    def test_even_flag_matches(self) -> None:
        for name, row in self._documented().items():
            expected = "YES" if ENCODING_RULES[name]["requires_even_features"] else "--"
            assert row["requires_even"] == expected, name

    def test_symmetry_and_trainable_match(self) -> None:
        for name, row in self._documented().items():
            rules = ENCODING_RULES[name]
            assert row["requires_symmetry"] == (
                rules["requires_symmetry"] or "--"
            ), name
            assert row["requires_trainable"] == (
                "YES" if rules["requires_trainable"] else "--"
            ), name
