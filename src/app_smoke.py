"""Headless smoke test of the Streamlit app using Streamlit's AppTest (executes the real app script, simulates widget actions).
This verifies the script runs and behaves; it does NOT render in a browser (that remains a manual check)."""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

from .utils import utc_now


def _n_results(at) -> int:
    return sum(1 for c in at.caption if "Paper ID" in str(c.value))


def _info_has(at, text: str) -> bool:
    return any(text in str(i.value) for i in at.info)


def run_app_smoke(cfg: dict, app_path: Path, timeout_s: float, queries: list[str], default_top_k: int) -> dict:
    from streamlit.testing.v1 import AppTest

    steps: list[dict] = []
    state: dict = {"at": None}

    def step(name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        at = state["at"]
        if ok and at is not None and len(at.exception):
            ok, detail = False, "app raised: " + "; ".join(str(e.value)[:200] for e in at.exception)
        steps.append({"step": name, "ok": bool(ok), "seconds": round(time.perf_counter() - t0, 2), "detail": detail})

    def initial():
        state["at"] = AppTest.from_file(str(app_path), default_timeout=timeout_s).run()
        at = state["at"]
        errs = [str(e.value)[:160] for e in at.error]
        return (not errs and len(at.title) >= 1), f"title={at.title[0].value if len(at.title) else None!r} errors={errs}"

    def empty_prompt():
        at = state["at"]
        return (_n_results(at) == 0 and _info_has(at, "Type a claim")), f"results={_n_results(at)}"

    def query_default():
        state["at"] = state["at"].text_input(key="query").set_value(queries[0]).run()
        n = _n_results(state["at"])
        return n == default_top_k, f"results={n} expected={default_top_k}"

    def slider():
        state["at"] = state["at"].slider(key="top_k").set_value(3).run()
        n = _n_results(state["at"])
        return n == 3, f"results={n} expected=3"

    def whitespace():
        state["at"] = state["at"].text_input(key="query").set_value("   ").run()
        return (_n_results(state["at"]) == 0 and _info_has(state["at"], "Type a claim")), f"results={_n_results(state['at'])}"

    def example_button():
        state["at"] = state["at"].slider(key="top_k").set_value(default_top_k).run()
        at = state["at"].button(key="example_0").click().run()
        state["at"] = at
        n = _n_results(at)
        return (at.text_input(key="query").value != "" and n == default_top_k), f"query={at.text_input(key='query').value!r} results={n}"

    step("initial_load_no_exception", initial)
    if state["at"] is None:
        return _finish(steps, app_path)
    step("empty_query_prompts_user", empty_prompt)
    step("query_returns_default_top_k", query_default)
    step("slider_changes_result_count", slider)
    step("whitespace_query_is_graceful", whitespace)
    step("example_button_fills_query_and_searches", example_button)
    if any(r.key == "model_key" for r in state["at"].radio):
        def baseline():
            state["at"] = state["at"].text_input(key="query").set_value(queries[-1]).run()
            state["at"] = state["at"].radio(key="model_key").set_value("baseline").run()
            n = _n_results(state["at"])
            return n == default_top_k, f"baseline results={n}"

        def compare():
            at = state["at"].checkbox(key="compare").check().run()
            state["at"] = at
            n = _n_results(at)
            return (n == 2 * default_top_k and _info_has(at, "appear in both lists")), f"results={n} expected={2 * default_top_k}"

        step("switch_to_baseline_returns_results", baseline)
        step("compare_side_by_side_shows_both_lists", compare)
    else:
        steps.append({"step": "baseline_toggle", "ok": True, "seconds": 0.0, "detail": "not offered (baseline assets unavailable): correct"})
    return _finish(steps, app_path)


def _finish(steps: list[dict], app_path: Path) -> dict:
    return {"status": "PASS" if steps and all(s["ok"] for s in steps) else "FAIL", "generated_utc": utc_now(), "app": str(app_path),
            "method": "streamlit.testing.v1.AppTest (headless; not a browser)", "steps": steps}


def serve_check(root: Path, app_rel: str, port: int, timeout_s: float) -> dict:
    """Start the real `streamlit run` server headlessly, poll its health endpoint, fetch the index page, then stop it."""
    cmd = [sys.executable, "-m", "streamlit", "run", app_rel, "--server.headless", "true", "--server.port", str(port),
           "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0, health, index_status, detail = time.perf_counter(), None, None, ""
    try:
        while time.perf_counter() - t0 < timeout_s:
            if proc.poll() is not None:
                detail = f"server exited early with code {proc.returncode}: {(proc.stdout.read() or '')[-400:]}"
                break
            try:
                health = urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).read().decode().strip()
                index_status = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).status
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    ok = health == "ok" and index_status == 200
    return {"step": "streamlit_run_server_health", "ok": ok, "seconds": round(time.perf_counter() - t0, 2),
            "detail": detail or f"health={health!r} index_http={index_status} port={port}"}
