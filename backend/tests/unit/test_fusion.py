from __future__ import annotations

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_scores_follow_the_rrf_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_items_found_by_both_retrievers_rank_first() -> None:
    vector = ["v1", "shared", "v2", "v3"]
    keyword = ["k1", "k2", "shared"]
    ranked = [item for item, _ in reciprocal_rank_fusion([vector, keyword])]
    assert ranked[0] == "shared"
    assert set(ranked) == {"v1", "v2", "v3", "k1", "k2", "shared"}


def test_ties_are_broken_deterministically_by_first_appearance() -> None:
    ranked = [item for item, _ in reciprocal_rank_fusion([["a"], ["b"]])]
    assert ranked == ["a", "b"]


def test_weights_shift_the_balance() -> None:
    ranked = [item for item, _ in reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0])]
    assert ranked == ["b", "a"]


def test_duplicates_within_one_list_vote_once() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=10))
    assert fused["a"] == pytest.approx(1 / 11)
    assert fused["b"] == pytest.approx(1 / 13)


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([]) == []


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
