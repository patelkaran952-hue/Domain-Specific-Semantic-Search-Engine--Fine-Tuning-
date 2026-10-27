"""Scientific paper search UI (Streamlit). Run from the repo root:  streamlit run app/streamlit_app.py

Thin layer over src/ui_support.py + src/search_service.py. Models are loaded lazily and ONCE per process (st.cache_resource);
the baseline is loaded only if the user selects it or turns on the side-by-side comparison.
Scores are cosine similarities, NOT probabilities that a paper is relevant or that a claim is true.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import ui_support as ui  # noqa: E402


@st.cache_resource(show_spinner="Loading model and FAISS index (first use only; about 20 s on CPU)…")
def get_service(key: str):
    """One SearchService per model key for the whole process (no duplicate model initialization across reruns/sessions)."""
    return ui.load_service(ui.get_config(), key)


def set_query(q: str) -> None:
    st.session_state["query"] = q


def render_hit(h, svc, other_rank: int | None, other_label: str, show_abstract: bool) -> None:
    with st.container(border=True):
        st.markdown(f"**{h.rank}. {ui.md_escape(h.title)}**")
        tag = f" · also in {other_label} list (rank {other_rank})" if other_rank else ""
        st.caption(f"Paper ID `{h.corpus_id}` · cosine similarity **{h.similarity_score:.3f}**{tag}")
        st.write(ui.md_escape(h.snippet))
        if show_abstract:
            with st.expander("Full abstract"):
                st.write(ui.md_escape(svc.get_document(h.corpus_id)["text"]))


def render_panel(title: str, outcome, svc, other_ranks: dict, other_label: str, show_abstract: bool) -> None:
    st.subheader(title)
    for level, text in outcome.messages:
        getattr(st, level)(text)
    if outcome.level != "results":
        return
    r = outcome.response
    st.caption(f"{len(r.hits)} results in {r.latency_ms['total']:.0f} ms (query encoding {r.latency_ms['encode']:.0f} ms)")
    for h in r.hits:
        render_hit(h, svc, other_ranks.get(h.corpus_id), other_label, show_abstract)


def main() -> None:
    cfg = ui.get_config()
    A = cfg["app"]
    st.set_page_config(page_title=A["title"], page_icon="🔎", layout="wide")
    st.title("🔎 " + A["title"])
    st.caption("Semantic search over 5,183 SciFact paper abstracts. Results are ranked by embedding similarity to your query; "
               "this is not a verdict on whether a claim is true.")

    models = ui.discover_models(cfg)
    usable = [m for m in models if m.available]
    if not models or not models[0].available:
        reason = models[0].reason if models else "no frozen model selection found (complete Phases 5 and 6)"
        st.error(f"The search package is not available: {reason}")
        st.stop()
    labels = {m.key: m.label for m in usable}

    with st.sidebar:
        st.header("Settings")
        if len(usable) > 1:
            model_key = st.radio("Model", [m.key for m in usable], format_func=labels.get, key="model_key")
            compare = st.checkbox("Compare fine-tuned and baseline side by side", key="compare")
        else:
            model_key, compare = usable[0].key, False
            st.caption(f"Model: {labels[model_key]}")
        top_k = st.slider("Number of results (top-k)", 1, int(A["max_top_k_ui"]), int(cfg["service"]["default_top_k"]), key="top_k")
        st.subheader("Try an example")
        for i, q in enumerate(A["example_queries"]):
            st.button(q, key=f"example_{i}", on_click=set_query, args=(q,), use_container_width=True)
        with st.expander("About these scores and models"):
            st.markdown("**Cosine similarity** between the query and abstract embeddings (higher = more semantically similar). "
                        "It is **not** a probability of relevance or of a claim being true.")
            for m in usable:
                rep = m.reported.get("heldout_test_ndcg@10")
                st.markdown(f"- **{m.label}**" + (f": held-out test nDCG@10 {rep:.3f} (300 SciFact queries, sparse judgments)" if rep else ""))
            st.caption("Research prototype; not medical or clinical advice.")

    query = st.text_input("Search query", key="query", max_chars=int(cfg["service"]["max_query_chars"]),
                          placeholder="Type a scientific claim or topic, then press Enter")
    keys = [usable[0].key, "baseline"] if (compare and len(usable) > 1) else [model_key]
    try:
        services = {k: get_service(k) for k in keys}
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the search service ({type(exc).__name__}): {exc}")
        st.stop()

    outcomes = {k: ui.run_search(services[k], query, top_k) for k in keys}
    if all(o.level == "empty" for o in outcomes.values()) and not query.strip():
        for level, text in next(iter(outcomes.values())).messages:
            getattr(st, level)(text)
        return
    ranks, other = {}, {}
    if len(keys) == 2 and all(o.level == "results" for o in outcomes.values()):
        s = ui.overlap_summary(outcomes[keys[0]].response.hits, outcomes[keys[1]].response.hits)
        st.info(f"{s['n_shared']} of {s['k_a']} papers appear in both lists; the top result is "
                f"{'the same' if s['same_top1'] else 'different'}.")
        ranks = {keys[0]: s["rank_in_b"], keys[1]: s["rank_in_a"]}
        other = {keys[0]: "baseline", keys[1]: "fine-tuned"}
    cols = st.columns(len(keys)) if len(keys) > 1 else [st.container()]
    for col, k in zip(cols, keys):
        with col:
            render_panel(labels[k], outcomes[k], services[k], ranks.get(k, {}), other.get(k, ""), bool(A["show_full_abstract"]))
    st.divider()
    st.caption("Searches only the 5,183 SciFact abstracts; unlisted papers are not considered. Scores are cosine similarities.")


main()
