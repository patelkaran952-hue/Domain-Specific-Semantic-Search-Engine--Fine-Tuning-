import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from src.index_faiss import IndexAlignmentError, build_index, check_unit_norm, load_index, save_index, search
from src.embed import l2_normalize

# NOTE: all vectors in this file are SYNTHETIC random data used to test index mechanics only.
rng = np.random.default_rng(0)
EMB = l2_normalize(rng.normal(size=(50, 16)))
IDS = [f"d{i:03d}" for i in range(50)]


def test_unit_norm_enforced():
    with pytest.raises(ValueError, match="not L2-normalized"):
        build_index(rng.normal(size=(5, 4)).astype(np.float32) * 3)
    assert check_unit_norm(EMB) < 1e-5


def test_self_query_ranks_itself_first_with_score_one():
    index = build_index(EMB)
    res = search(index, IDS, EMB[7], 3)[0]
    assert res[0][0] == "d007" and res[0][1] == pytest.approx(1.0, abs=1e-5)


def test_inner_product_equals_cosine():
    from sklearn.metrics.pairwise import cosine_similarity

    index = build_index(EMB)
    q = EMB[:5]
    res = search(index, IDS, q, 50)
    cos = cosine_similarity(q, EMB)
    for qi, row in enumerate(res):
        for doc, score in row:
            assert score == pytest.approx(cos[qi, IDS.index(doc)], abs=1e-5)


def test_scores_sorted_descending_and_topk_clipped():
    index = build_index(EMB)
    row = search(index, IDS, EMB[0], 1000)[0]
    assert len(row) == 50 and [s for _, s in row] == sorted((s for _, s in row), reverse=True)


def test_save_load_roundtrip_identical_ranking(tmp_path):
    index = build_index(EMB)
    save_index(index, IDS, tmp_path, "t", {"note": "synthetic"})
    index2, ids2 = load_index(tmp_path, "t", expected_doc_ids=IDS)
    q = EMB[:10]
    assert search(index, IDS, q, 10) == search(index2, ids2, q, 10)


def test_alignment_errors(tmp_path):
    index = build_index(EMB)
    save_index(index, IDS, tmp_path, "t", {})
    with pytest.raises(IndexAlignmentError, match="differs"):
        load_index(tmp_path, "t", expected_doc_ids=list(reversed(IDS)))
    (tmp_path / "t.document_ids.json").write_text('["a"]')
    with pytest.raises(IndexAlignmentError, match="rows"):
        load_index(tmp_path, "t")


def test_missing_index_file_names_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.faiss"):
        load_index(tmp_path, "nope")
