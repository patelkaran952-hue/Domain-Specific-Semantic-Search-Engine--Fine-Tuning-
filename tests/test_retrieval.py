from src.retrieve import TfidfProbe

DOCS = [
    {"_id": "1", "title": "Wnt signalling", "doc_text": "Wnt signalling in embryonic stem cells"},
    {"_id": "2", "title": "Cardiac arrhythmia", "doc_text": "Sudden cardiac death arrhythmia"},
]


def test_tfidf_probe_ranks_relevant_doc_first_and_handles_empty_query():
    p = TfidfProbe(DOCS)
    assert p.search("embryonic stem cell Wnt", 2)[0]["corpus_id"] == "1"
    assert p.search("   ") == []
