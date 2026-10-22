import pytest

from src.split import partition_qrels, split_query_ids, verify_no_leakage

IDS = [str(i) for i in range(100)]


def test_split_sizes_disjoint_and_complete():
    tr, va = split_query_ids(IDS, 0.2, 42)
    assert len(va) == 20 and len(tr) == 80
    assert not set(tr) & set(va)
    assert set(tr) | set(va) == set(IDS)


def test_split_deterministic_and_seed_sensitive():
    assert split_query_ids(IDS, 0.2, 42) == split_query_ids(list(reversed(IDS)), 0.2, 42)
    assert split_query_ids(IDS, 0.2, 42)[1] != split_query_ids(IDS, 0.2, 7)[1]


def test_partition_keeps_all_positives_of_a_query_together():
    rows = [("1", "a", 1), ("1", "b", 1), ("2", "c", 1)]
    tr, va = partition_qrels(rows, ["1"], ["2"])
    assert tr == rows[:2] and va == rows[2:]


def test_partition_raises_instead_of_dropping_rows():
    with pytest.raises(ValueError, match="neither partition"):
        partition_qrels([("3", "a", 1)], ["1"], ["2"])


def test_overlapping_partitions_rejected():
    with pytest.raises(ValueError, match="overlap"):
        partition_qrels([], ["1"], ["1"])


def test_leakage_gates_detect_test_overlap():
    gates = verify_no_leakage(["1"], ["2"], ["2"], [("1", "a", 1)], [("2", "b", 1)])
    assert gates["val_test_disjoint"] is False


def test_real_split_matches_expected_counts_if_data_present():
    from src.config import find_repo_root
    from src.data_io import read_qrels

    p = find_repo_root() / "data/raw/train.tsv"
    if not p.is_file():
        pytest.skip("raw data not present")
    qids = {q for q, _, _ in read_qrels(p)}
    tr, va = split_query_ids(sorted(qids), 0.2, 42)
    assert (len(tr), len(va)) == (647, 162)
