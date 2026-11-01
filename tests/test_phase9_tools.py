import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src import envtools, export_tools, hygiene, readme_results
from src.search_service import build_package
from src.service_validation import run_service_validation
from tests.test_search_service import built, encoder_factory

# Everything here runs on SYNTHETIC trees/reports: it tests the publishing/reproducibility tooling, not model quality.
SECRET = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3"          # assembled at runtime so this file itself stays clean
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(root, rel, text="x", binary=False):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text if binary else text.encode("utf-8"))
    return p


def kinds(rep, sev=None):
    return {(f["severity"], f["kind"]) for f in rep["findings"] if sev in (None, f["severity"])}


# ---- hygiene ------------------------------------------------------------------------------------------------------------
def test_hygiene_detects_and_masks_secrets_and_flags_assets(tmp_path):
    write(tmp_path, "src/a.py", f'KEY = "{SECRET}"\n')
    write(tmp_path, ".env", "ANTHROPIC_API_KEY=abc\n")
    write(tmp_path, "data/raw/corpus.jsonl", "{}\n")
    write(tmp_path, "notes.md", "see C:\\Users\\bob\\proj\\x for details\n")  # hygiene:ok (deliberate fixture)
    write(tmp_path, "ok.py", "print(1)\n")
    write(tmp_path, "idx.faiss", "binary-ish")
    rep = hygiene.scan(tmp_path, "repo")
    assert rep["status"] == "FAIL"
    assert ("FAIL", "secret:api_key_sk") in kinds(rep) and ("FAIL", "env_file_present") in kinds(rep)
    assert ("WARN", "licensed_or_binary_asset") in kinds(rep) and ("WARN", "private_absolute_path") in kinds(rep) and ("WARN", "no_license_file") in kinds(rep)
    assert SECRET not in json.dumps(rep)                                                     # never printed in full
    strict = hygiene.scan(tmp_path, "export")
    assert ("FAIL", "licensed_or_binary_asset") in kinds(strict) and ("FAIL", "private_absolute_path") in kinds(strict)


def test_hygiene_clean_tree_allowlist_and_size_limits(tmp_path):
    write(tmp_path, "LICENSE", "MIT")
    write(tmp_path, "src/a.py", "x = 1\n")
    write(tmp_path, "src/b.py", f'k = "{SECRET}"  # hygiene:ok demo value\n')
    write(tmp_path, "tests/t.py", 'SECRET = "unit-test-value-123456789"\n')              # assigned-secret pattern is skipped inside tests/
    write(tmp_path, ".env.example", "# ANTHROPIC_API_KEY=\n")
    assert hygiene.scan(tmp_path, "export")["status"] == "PASS"
    write(tmp_path, "big.bin", b"0" * 2_000_000, binary=True)
    r = hygiene.scan(tmp_path, "repo", max_mb=1, warn_mb=0.5)
    assert ("FAIL", "file_too_large") in kinds(r)
    r2 = hygiene.scan(tmp_path, "repo", max_mb=5, warn_mb=1)
    assert ("WARN", "file_large") in kinds(r2) and r2["status"] == "PASS"


def test_hygiene_assigned_secret_outside_tests_is_flagged_but_placeholders_are_not(tmp_path):
    write(tmp_path, "LICENSE", "x")
    write(tmp_path, "cfg.py", 'password = "Zx81Qw93Lm20Pa55Vb76"\n')
    assert ("FAIL", "secret:assigned_value") in kinds(hygiene.scan(tmp_path, "repo"))
    write(tmp_path, "cfg.py", 'password = "your_password_goes_here_1234"\n')
    assert hygiene.scan(tmp_path, "repo")["status"] == "PASS"


# ---- scrubbing / export ---------------------------------------------------------------------------------------------------
def test_scrub_text_handles_root_variants_homes_and_stray_drive_paths(tmp_path):
    root = Path(tmp_path)
    txt = f"a {root} b {root.as_posix()}/x c {str(root).replace(chr(92), chr(92) * 2)} d C:\\Users\\Alice\\p e /home/bob/q f D:\\Anaconda\\Lib\\x.py g file:///C:/b/abs_1"  # hygiene:ok (deliberate fixture)
    out, n = export_tools.scrub_text(txt, root)
    assert str(root) not in out and "Alice" not in out and "/home/bob" not in out and "Anaconda" not in out and "abs_1" not in out
    assert "<REPO_ROOT>" in out and "<HOME>" in out and n >= 5


SPEC = {"include": ["README.md", "LICENSE", "src/*.py", ".env.example", "notebooks/*.ipynb", "data/processed/document_ids.json", "data/processed/*_query_ids.json",
                    "artifacts/reports/*.json", "experiments/*.csv", "checkpoints/manifest.json", "checkpoints/phase_*/manifest.json", "docs/**/*"],
        "exclude": ["artifacts/reports/*run_top100*"],
        "remap": [["artifacts/reports/", "results/reports/"], ["experiments/", "results/experiments/"], ["checkpoints/", "results/checkpoints/"]]}


def make_repo(root, secret=False):
    root = Path(root)
    write(root, "README.md", "# x\n"); write(root, "LICENSE", "MIT"); write(root, "src/a.py", (f'k = "{SECRET}"\n' if secret else "x = 1\n"))
    write(root, ".env", "ANTHROPIC_API_KEY=real\n"); write(root, ".env.example", "# KEY=\n")
    write(root, "data/raw/corpus.jsonl", "{}\n"); write(root, "data/processed/corpus_processed.jsonl", "{}\n")
    write(root, "data/processed/document_ids.json", "[]\n"); write(root, "data/processed/val_query_ids.json", "[]\n")
    write(root, "artifacts/models/m/model.safetensors", "w"); write(root, "artifacts/embeddings/e.npy", "w")
    write(root, "artifacts/reports/test_results.json", json.dumps({"p": str(root), "v": 1}))
    write(root, "artifacts/reports/x_run_top100.json", "{}"); write(root, "experiments/runs.csv", "a,b\n1,2\n")
    write(root, "checkpoints/manifest.json", "{}"); write(root, "checkpoints/phase_01/manifest.json", "{}")
    nb = {"cells": [{"cell_type": "code", "source": "1", "metadata": {}, "execution_count": 1, "outputs": [
        {"output_type": "stream", "name": "stderr", "text": f"warning in {root}"}, {"output_type": "stream", "name": "stdout", "text": f"cwd={root}"}]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    write(root, "notebooks/n.ipynb", json.dumps(nb))
    return root


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def test_export_whitelists_remaps_scrubs_and_never_touches_the_source(tmp_path):
    root = make_repo(tmp_path / "repo")
    before = tree_hashes(root)
    dest = tmp_path / "out"
    m = export_tools.export(root, dest, SPEC)
    after = tree_hashes(root)
    assert before == after                                                                   # source untouched
    got = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert {"README.md", "LICENSE", "src/a.py", ".env.example", "data/processed/document_ids.json", "data/processed/val_query_ids.json",
            "results/reports/test_results.json", "results/experiments/runs.csv", "results/checkpoints/manifest.json",
            "results/checkpoints/phase_01/manifest.json", "notebooks/n.ipynb", "PUBLISH_MANIFEST.json"} <= got
    assert not any(x in got for x in (".env", "data/raw/corpus.jsonl", "data/processed/corpus_processed.jsonl", "artifacts/models/m/model.safetensors", "results/reports/x_run_top100.json"))
    rep = json.loads((dest / "results/reports/test_results.json").read_text())
    assert rep["v"] == 1 and "<REPO_ROOT>" in rep["p"] and str(root) not in rep["p"]
    nb = json.loads((dest / "notebooks/n.ipynb").read_text())
    outs = nb["cells"][0]["outputs"]
    assert [o["name"] for o in outs] == ["stdout"] and str(root) not in json.dumps(nb)           # stderr stream dropped, paths scrubbed
    man = json.loads((dest / "PUBLISH_MANIFEST.json").read_text())
    assert man["hygiene_status"] == "PASS" and all(sha(dest / f["path"]) == f["sha256"] for f in man["files"])
    assert m["n_files"] == len(man["files"])
    with pytest.raises(SystemExit, match="never overwrite"):
        export_tools.export(root, dest, SPEC)


def test_export_rejected_and_quarantined_when_a_secret_slips_in(tmp_path):
    root = make_repo(tmp_path / "repo", secret=True)
    dest = tmp_path / "out"
    with pytest.raises(SystemExit, match="REJECTED") as e:
        export_tools.export(root, dest, SPEC)
    assert SECRET not in str(e.value) and not dest.exists() and (tmp_path / "out.REJECTED").is_dir()


def test_export_zip_option_and_empty_whitelist(tmp_path):
    root = make_repo(tmp_path / "repo")
    export_tools.export(root, tmp_path / "o2", SPEC, make_zip=True)
    assert (tmp_path / "o2.zip").is_file()
    with pytest.raises(SystemExit, match="matched no files"):
        export_tools.export(root, tmp_path / "o3", {"include": ["nothing/*"], "exclude": []})


# ---- environment tools --------------------------------------------------------------------------------------------------------
def test_requirements_parsing_lite_filter_and_lock_rendering(tmp_path):
    f = write(tmp_path, "r.txt", "# c\nnumpy>=1.26\nsentence-transformers[train]>=3.0   # x\n-r other.txt\n\ntorch>=2.1\nPyYAML>=6\nfaiss_cpu\n")
    reqs = envtools.parse_requirements(f)
    assert [n for n, _ in reqs] == ["numpy", "sentence-transformers", "torch", "pyyaml", "faiss-cpu"]
    lite = envtools.filter_lite(reqs)
    assert lite == ["numpy>=1.26", "PyYAML>=6", "faiss_cpu"]
    lock = envtools.render_lock(["numpy", "torch", "pyyaml"], version_of=lambda n: {"numpy": "2.0.0", "pyyaml": "6.0"}.get(n))
    assert "numpy==2.0.0" in lock and "pyyaml==6.0" in lock and "not installed in this environment: torch" in lock and "file:" not in lock


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"s9_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_verify_fresh_env_orchestration_with_stubbed_subprocess(tmp_path, monkeypatch, capsys):
    mod = load_script("verify_fresh_env")
    root = tmp_path / "repo"; write(root, "configs/base.yaml", "x: 1\n")
    write(root, "requirements.txt", "numpy>=1\ntorch>=2\n"); write(root, "requirements-dev.txt", "pytest>=8\njupyter>=1\n")
    calls = []

    def fake_run(cmd, cwd=None, timeout=3600):
        calls.append(cmd)
        if "pytest" in cmd:
            return 0, "12 passed in 1.0s\n", 1.0
        if "list" in cmd:
            return 0, '[{"name": "numpy", "version": "2.0"}]', 0.1
        return 0, "", 0.1
    monkeypatch.setattr(mod, "run", fake_run); monkeypatch.setattr(mod, "find_repo_root", lambda: root); monkeypatch.setattr(sys, "argv", ["v"])
    assert mod.main() == 0
    rep = json.loads((root / "artifacts/reports/fresh_env_report_lite.json").read_text())
    assert rep["status"] == "PASS" and rep["mode"] == "lite" and rep["installed_in_fresh_env"] == {"numpy": "2.0"}
    assert [s["ok"] for s in rep["steps"]] == [True, True, True] and rep["steps"][2]["summary"].startswith("12 passed")
    installed = [c for c in calls if "install" in c]
    req_text = [c for c in calls if "-r" in c]
    assert installed and req_text
    def failing(cmd, cwd=None, timeout=3600):
        return (1, "boom", 0.1) if "pytest" in cmd else fake_run(cmd, cwd, timeout)
    monkeypatch.setattr(mod, "run", failing)
    assert mod.main() == 1


# ---- README result generation ---------------------------------------------------------------------------------------------------
@pytest.fixture()
def reports(tmp_path):
    cfg = built(tmp_path)
    build_package(cfg)
    run_service_validation(cfg, cfg["_abs"]["service_dir"] / "pkg", encoder_factory)
    return tmp_path, cfg


def test_generated_blocks_contain_exactly_the_measured_numbers(reports):
    root, cfg = reports
    rd = cfg["_abs"]["reports_dir"]
    R = readme_results.load_reports(rd)
    test = json.loads((rd / "test_results.json").read_text())
    sel = test["selected_model"]
    nd = test["models"][sel]["test_metrics"]["ndcg@10"]
    head = readme_results.headline(R)
    assert f"{nd:.4f}" in head and f"{test['eval_setup']['n_queries']} held-out" in head
    body = readme_results.results(R)
    val = json.loads((rd / "model_comparison_val.json").read_text())
    for k, m in val["models"].items():
        assert m["label"] in body and f"{m['metrics']['ndcg@10']:.4f}" in body
    assert "Held-out test (evaluated once)" in body and "Serving" in body
    assert "Not generated yet: run `python scripts/prepare_data.py`" in body                          # fixture audit is partial: degrade, never crash
    assert "Not generated yet: run `python scripts/calibrate_rag_gate.py`" in body                  # RAG not run in this fixture: no invented numbers
    sv = json.loads((rd / "search_service_report.json").read_text())
    assert f"{sv['package']['size_mb']} MB" in body and f"{sv['reload_consistency']['exact']}/{sv['reload_consistency']['n_queries']}" in body
    assert "No fine-tuning reports found" in body                                                  # fixture holds no real run reports


def test_blocks_missing_reports_say_so_and_marker_replacement_is_idempotent(tmp_path):
    R = readme_results.load_reports(tmp_path)
    assert "Not generated yet" in readme_results.results(R) and "generated from the measured reports" in readme_results.headline(R)
    text = "a\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nz\n"
    once = readme_results.replace_block(text, "RESULTS", "NEW\n")
    assert readme_results.replace_block(once, "RESULTS", "NEW\n") == once and "old" not in once and once.startswith("a\n")
    with pytest.raises(ValueError, match="markers"):
        readme_results.replace_block("no markers", "RESULTS", "x")


def test_generate_readme_script_check_mode(reports, monkeypatch, capsys):
    root, cfg = reports
    (root / "docs").mkdir(exist_ok=True)
    (root / "README.md").write_text("<!-- HEADLINE:START -->\nh\n<!-- HEADLINE:END -->\n<!-- RESULTS:START -->\nr\n<!-- RESULTS:END -->\n<!-- SCREENSHOTS:START -->\n<!-- SCREENSHOTS:END -->\n", encoding="utf-8")
    (root / "docs/MODEL_CARD.md").write_text("<!-- MODEL_METRICS:START -->\nm\n<!-- MODEL_METRICS:END -->\n", encoding="utf-8")
    mod = load_script("generate_readme_results")
    monkeypatch.setattr(mod, "find_repo_root", lambda: root)
    monkeypatch.setattr(sys, "argv", ["g", "--check"])
    assert mod.main() == 1 and "OUT OF DATE" in capsys.readouterr().out                           # placeholders are stale
    monkeypatch.setattr(sys, "argv", ["g"])
    assert mod.main() == 0
    assert "held-out" in (root / "README.md").read_text(encoding="utf-8") and "Held-out test" in (root / "docs/MODEL_CARD.md").read_text(encoding="utf-8").replace("Held-out test (", "Held-out test (")
    monkeypatch.setattr(sys, "argv", ["g", "--check"])
    assert mod.main() == 0 and "up to date" in capsys.readouterr().out
    (root / "docs/screenshots").mkdir(parents=True, exist_ok=True)
    (root / "docs/screenshots/demo_view.png").write_bytes(b"\x89PNG")
    monkeypatch.setattr(sys, "argv", ["g"]); mod.main()
    assert "![demo view](docs/screenshots/demo_view.png)" in (root / "README.md").read_text(encoding="utf-8")


def test_python_source_is_copied_verbatim_and_a_path_in_code_blocks_the_export(tmp_path):
    root = make_repo(tmp_path / "repo")
    write(root, "src/b.py", "P = '/home/someone/project/'\n")  # hygiene:ok (deliberate fixture)
    with pytest.raises(SystemExit, match="REJECTED") as e:
        export_tools.export(root, tmp_path / "o", SPEC)
    assert "private_absolute_path" in str(e.value) and (tmp_path / "o.REJECTED" / "src/b.py").read_text() == "P = '/home/someone/project/'\n"  # hygiene:ok (deliberate fixture)


def test_license_file_is_the_unmodified_apache_2_0_text():
    """Pins the canonical Apache-2.0 text (CRLF-normalized so a Windows checkout does not break the hash)."""
    raw = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    text = raw.decode("utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    for section in ("1. Definitions.", "2. Grant of Copyright License.", "3. Grant of Patent License.", "4. Redistribution.", "5. Submission of Contributions.",
                    "6. Trademarks.", "7. Disclaimer of Warranty.", "8. Limitation of Liability.", "9. Accepting Warranty or Additional Liability."):
        assert section in text, section
    assert "END OF TERMS AND CONDITIONS" in text and "APPENDIX: How to apply the Apache License to your work." in text


def test_docs_state_the_apache_choice_and_no_longer_say_license_is_pending():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Apache License 2.0" in readme and "chosen by its author" not in readme
    assert "Apache License 2.0" in (root / "docs/DATA_AND_LICENSES.md").read_text(encoding="utf-8")
    assert "Choose a license for your code" not in (root / "docs/PUBLISHING.md").read_text(encoding="utf-8")
