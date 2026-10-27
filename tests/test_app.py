from pathlib import Path

import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import ui_support as ui  # noqa: E402
from src.app_smoke import run_app_smoke  # noqa: E402
from tests.test_ui_support import fake_service  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
CFG = {"app": {"title": "Test Search", "max_top_k_ui": 30, "show_full_abstract": True, "compare_enabled": True,
               "example_queries": ["alpha1 topic3", "beta2 topic9"], "warmup_query": "w"},
       "service": {"default_top_k": 10, "max_top_k": 30, "max_query_chars": 200}}
TITLES = {0: "Effect of *p53* on $x_1$ signalling"}


class Env:
    """Fake context: fine-tuned + (optionally) baseline services; counts how often each is initialized."""
    def __init__(self, baseline=True, fail=None):
        self.loads, self.baseline, self.fail = [], baseline, fail

    def install(self, mp):
        opts = [ui.ModelOption("ft", "Fine-tuned model", True, reported={"heldout_test_ndcg@10": 0.713})]
        if self.baseline:
            opts.append(ui.ModelOption("baseline", "Baseline model", True, reported={"heldout_test_ndcg@10": 0.645}))
        mp.setattr(ui, "get_config", lambda: CFG)
        mp.setattr(ui, "discover_models", lambda cfg: opts)

        def load(cfg, key, **kw):
            self.loads.append(key)
            if self.fail == key:
                raise RuntimeError("boom while loading")
            return fake_service(32 if key == "ft" else 64, titles=TITLES)
        mp.setattr(ui, "load_service", load)
        return self


@pytest.fixture(autouse=True)
def _clear_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_app():
    return AppTest.from_file(str(APP), default_timeout=30).run()


def n_results(at):
    return sum(1 for c in at.caption if "Paper ID" in str(c.value))


def test_initial_render_prompts_without_exception(monkeypatch):
    Env().install(monkeypatch)
    at = run_app()
    assert not at.exception and at.title[0].value.endswith("Test Search")
    assert n_results(at) == 0 and any("Type a claim" in i.value for i in at.info)


def test_query_shows_ranked_cited_results_with_labeled_scores(monkeypatch):
    Env().install(monkeypatch)
    at = run_app()
    at = at.text_input(key="query").set_value("alpha1 topic3").run()
    assert not at.exception and n_results(at) == 10
    caps = [str(c.value) for c in at.caption if "Paper ID" in str(c.value)]
    assert all("cosine similarity" in c for c in caps) and "`p" in caps[0]
    bolds = [str(m.value) for m in at.markdown if str(m.value).startswith("**1.")]
    assert len(bolds) == 1


def test_dataset_text_is_escaped_not_rendered_as_markdown_or_math(monkeypatch):
    Env().install(monkeypatch)
    at = run_app().text_input(key="query").set_value("topic0 alpha0 beta0").run()
    titles = [str(m.value) for m in at.markdown if "p53" in str(m.value)]
    assert titles and "\\*p53\\*" in titles[0] and "\\$x\\_1\\$" in titles[0]


def test_slider_whitespace_and_example_button(monkeypatch):
    Env().install(monkeypatch)
    at = run_app().text_input(key="query").set_value("alpha1 topic3").run()
    at = at.slider(key="top_k").set_value(3).run()
    assert n_results(at) == 3
    at = at.text_input(key="query").set_value("   ").run()
    assert not at.exception and n_results(at) == 0 and any("Type a claim" in i.value for i in at.info)
    at = at.button(key="example_1").click().run()
    assert at.text_input(key="query").value == "beta2 topic9" and n_results(at) == 3 and not at.exception


def test_each_model_is_initialized_once_across_reruns_and_baseline_is_lazy(monkeypatch):
    env = Env().install(monkeypatch)
    at = run_app()
    for q in ("alpha1 topic3", "beta2 topic9", "alpha2"):
        at = at.text_input(key="query").set_value(q).run()
    assert env.loads == ["ft"]                                  # one init despite many reruns; baseline never loaded yet
    at = at.radio(key="model_key").set_value("baseline").run()
    at = at.text_input(key="query").set_value("alpha1").run()
    assert env.loads == ["ft", "baseline"] and n_results(at) == 10


def test_compare_mode_shows_both_lists_and_overlap(monkeypatch):
    Env().install(monkeypatch)
    at = run_app().text_input(key="query").set_value("alpha1 topic3").run()
    at = at.checkbox(key="compare").check().run()
    assert not at.exception and n_results(at) == 20
    assert any("appear in both lists" in i.value for i in at.info)
    assert len(at.columns) == 2


def test_no_toggle_when_baseline_unavailable(monkeypatch):
    Env(baseline=False).install(monkeypatch)
    at = run_app().text_input(key="query").set_value("alpha1").run()
    assert not at.exception and len(at.radio) == 0 and len(at.checkbox) == 0 and n_results(at) == 10


def test_missing_package_shows_error_not_traceback(monkeypatch):
    monkeypatch.setattr(ui, "get_config", lambda: CFG)
    monkeypatch.setattr(ui, "discover_models", lambda cfg: [ui.ModelOption("ft", "F", False, "package not built (run scripts/build_search_package.py)")])
    at = run_app()
    assert not at.exception and any("not available" in e.value for e in at.error)


def test_load_failure_shows_error_not_traceback(monkeypatch):
    Env(fail="ft").install(monkeypatch)
    at = run_app()
    assert not at.exception and any("Could not load the search service" in e.value and "boom" in e.value for e in at.error)


def test_smoke_harness_passes_on_fake_env_and_detects_failures(monkeypatch):
    Env().install(monkeypatch)
    rep = run_app_smoke(CFG, APP, 30, ["alpha1 topic3", "beta2 topic9"], 10)
    assert rep["status"] == "PASS", [s for s in rep["steps"] if not s["ok"]]
    assert {s["step"] for s in rep["steps"]} >= {"initial_load_no_exception", "query_returns_default_top_k", "compare_side_by_side_shows_both_lists"}
    st.cache_resource.clear()
    Env(fail="ft").install(monkeypatch)
    assert run_app_smoke(CFG, APP, 30, ["alpha1"], 10)["status"] == "FAIL"
