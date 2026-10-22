import json
import pytest

from src.data_io import DataFormatError, read_jsonl, read_qrels, write_qrels
from src.preprocess import build_doc_text, build_processed_corpus
from src.validate_data import DataValidationError, check_cross_qrels, validate_corpus, validate_qrels, validate_queries


def doc(i, title="T", text="abstract"):
    return {"_id": i, "title": title, "text": text, "metadata": {}}


def test_duplicate_corpus_ids_rejected():
    with pytest.raises(DataValidationError, match="duplicate corpus"):
        validate_corpus([doc("1"), doc("1")])


def test_integer_id_rejected_ids_must_be_strings():
    with pytest.raises(DataValidationError, match="non-blank string"):
        validate_corpus([doc(1)])


def test_missing_field_rejected():
    with pytest.raises(DataValidationError, match="missing field"):
        validate_corpus([{"_id": "1", "title": "t"}])


def test_blank_title_reported_not_fatal():
    audit = validate_corpus([doc("1", title=" "), doc("2")])
    assert audit["blank_title_ids"] == ["1"]


def test_duplicate_query_ids_rejected():
    with pytest.raises(DataValidationError, match="duplicate query"):
        validate_queries([{"_id": "1", "text": "a", "metadata": {}}, {"_id": "1", "text": "b", "metadata": {}}])


def test_missing_corpus_id_in_qrels_rejected():
    with pytest.raises(DataValidationError, match="not in corpus"):
        validate_qrels("t", [("q1", "d9", 1)], {"d1"}, {"q1"})


def test_missing_query_id_in_qrels_rejected():
    with pytest.raises(DataValidationError, match="not in queries"):
        validate_qrels("t", [("q9", "d1", 1)], {"d1"}, {"q1"})


def test_duplicate_qrel_pair_rejected():
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_qrels("t", [("q1", "d1", 1), ("q1", "d1", 1)], {"d1"}, {"q1"})


def test_train_test_query_overlap_rejected():
    with pytest.raises(DataValidationError):
        check_cross_qrels([("q1", "d1", 1)], [("q1", "d2", 1)])


def test_qrels_crlf_and_string_ids(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_bytes(b"query-id\tcorpus-id\tscore\r\n007\t0123\t1\r\n")
    assert read_qrels(p) == [("007", "0123", 1)]  # leading zeros preserved => no int coercion


@pytest.mark.parametrize(
    "content,msg",
    [
        ("a\tb\tc\n1\t2\t1\n", "expected header"),
        ("query-id\tcorpus-id\tscore\n1\t2\n", "3 tab-separated"),
        ("query-id\tcorpus-id\tscore\n1\t2\tx\n", "non-integer"),
        ("query-id\tcorpus-id\tscore\n\t2\t1\n", "blank"),
        ("", "empty"),
    ],
)
def test_malformed_tsv(tmp_path, content, msg):
    p = tmp_path / "bad.tsv"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(DataFormatError, match=msg):
        read_qrels(p)


def test_missing_file_error_names_path(tmp_path):
    with pytest.raises(DataFormatError, match="nope.tsv"):
        read_qrels(tmp_path / "nope.tsv")


def test_bad_json_line_reports_line_number(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"_id": "1"}\n{bad}\n', encoding="utf-8")
    with pytest.raises(DataFormatError, match=r"c.jsonl:2"):
        read_jsonl(p)


def test_qrels_roundtrip(tmp_path):
    rows = [("2", "10", 1), ("1", "20", 1)]
    write_qrels(rows, tmp_path / "o.tsv")
    assert read_qrels(tmp_path / "o.tsv") == rows


def test_doc_text_preserves_terms_and_orders_by_string_id():
    assert build_doc_text(" Title ", " body ") == "Title\nbody"
    ordered = build_processed_corpus([doc("9"), doc("10"), doc("2")])
    assert [d["_id"] for d in ordered] == ["10", "2", "9"]  # lexicographic, not numeric
