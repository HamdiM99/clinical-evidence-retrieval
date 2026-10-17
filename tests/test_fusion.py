import pytest

from cer.retrieve.fusion import aggregate_max, rrf
from cer.schema import Hit


def test_rrf_both_lists_beat_one():
    f = rrf([["A", "B", "C"], ["A", "C"]], k=60)
    assert sorted(f, key=f.get, reverse=True) == ["A", "C", "B"]
    assert f["A"] == pytest.approx(2 / 61) and f["B"] == pytest.approx(1 / 62)
    assert f["C"] == pytest.approx(1 / 63 + 1 / 62)


def test_rrf_weights_and_zero_weight():
    f = rrf([["A"], ["B"]], k=60, weights=[2.0, 0.0])
    assert f == {"A": pytest.approx(2 / 61)}


def test_rrf_weight_count_checked():
    with pytest.raises(ValueError):
        rrf([["A"], ["B"]], weights=[1.0])


def test_aggregate_max_keeps_best_chunk_and_order():
    hits = [Hit("c1", "A", 0.9), Hit("c2", "B", 0.8), Hit("c3", "A", 0.7), Hit("c4", "C", 0.6)]
    agg = aggregate_max(hits)
    assert [d for d, _ in agg] == ["A", "B", "C"] and agg[0][1].chunk_id == "c1"
