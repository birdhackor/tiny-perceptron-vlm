"""Bounded source14 course callback; retain full machine lists outside Git."""
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import ast
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import build_course, export_course, reading_time

ART = ROOT / "docs/technical-reviews/artifacts/omni-technical-course-v1"
CACHE = Path("/workspace/work/technical-course-current-v2")
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
BUNDLE = "docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v4.json"
PRIOR = "docs/course-revision-20261029-omni/reviews/technical/technical_course/technical-current-v1.json"
INTAKE = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v5.json"
EXECUTION = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-execution-v5.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def ref(path):
    return {"path": str(path), "sha256": sha(ROOT / path)}


def bind(value):
    assert sha(ROOT / value["path"]) == value["sha256"], value["path"]
    return load(value["path"])


def section(text, pid):
    start = re.search(r"^## " + re.escape(pid) + r" .+$", text, re.M)
    assert start, pid
    end = re.search(r"^## ", text[start.end():], re.M)
    return text[start.start():start.end() + end.start()] if end else text[start.start():]


def main():
    torch.set_num_threads(1)
    CACHE.mkdir(parents=True, exist_ok=True)
    previous = load(PRIOR)
    assert sha(ROOT / PRIOR) == "78184edd9b1809e385abf01e9dba4e3c2b9ecca0b653b86235e8dc50af6169d2"
    freeze = load(FREEZE)
    assert sha(ROOT / FREEZE) == "caaa502b629d60e77c1f6d06373419e007a47bb1a089501d7bf10f063a9b5e29"
    old = {p["page_id"]: p for p in bind(previous["source_freeze"])["pages"]}
    pages = {p["page_id"]: p for p in freeze["pages"]}
    assert set(old) == set(pages)
    changed = [pid for pid, p in pages.items() if any(p.get(k) != old[pid].get(k) for k in ["source_sha256", "figures_sha256", "notebook_sha256"])]
    assert changed == ["12.2"], changed
    for pid, p in pages.items():
        assert sha(ROOT / p["snapshot"]) == p["source_sha256"], pid
        for path, digest in p["figures_sha256"].items():
            assert sha(ROOT / path) == digest, path
    course = pages["course"]
    body = (ROOT / course["snapshot"]).read_text()
    assert sha(ROOT / course["source"]) == course["source_sha256"] == previous["source_sha256"]
    assert course["figures_sha256"] == {}
    for item in previous["artifacts"]:
        assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
    repo_sources = []
    for source in previous["sources"]:
        if source.get("kind") == "repository_code":
            assert sha(ROOT / source["path"]) == source["sha256"], source["path"]
            repo_sources.append(ref(source["path"]))

    bundle = load(BUNDLE)
    assert sha(ROOT / BUNDLE) == "1ddabe842cbb17502ae1fb08783825c16f5669c22b8c294c0e9f83046c8b19da"
    assert bind(bundle["canonical_source_freeze"]) == freeze
    numbered = {p["page_id"]: p for p in bind(bundle["section_freeze"])["pages"]}
    collection = bind(bundle["actual_collection_snapshot"])
    raw_collection = bind(collection["raw_collection_snapshot"])
    assert all(row in raw_collection["collection_index"] for row in collection["collection_index"])
    issues = bind(bundle["issue_inventory"])
    prior_bundle = bind(bundle["previous_bundle"])
    prior_sections = {r["lesson_id"]: r for r in prior_bundle["sections"]}
    owners, rows, changes, details = [], [], [], {}
    for row in bundle["sections"]:
        pid = row["lesson_id"]
        current = bind(row["report"])
        original = bind(row["original_report"])
        op = next(p for p in bind(row["original_section_freeze"])["pages"] if p["page_id"] == pid)
        assert original["source_sha256"] == op["source_sha256"] == sha(ROOT / op["snapshot"])
        assert current["source_sha256"] == numbered[pid]["source_sha256"] == sha(ROOT / numbered[pid]["snapshot"])
        assert current["figure_sha256"] == numbered[pid]["figures_sha256"]
        assert current["reviewer_task"] == original["reviewer_task"]
        owners.append(current["reviewer_task"])
        for role, report in [("original", original), ("current", current)]:
            rr = row["original_report"] if role == "original" else row["report"]
            receipt = row[role + "_collection"]
            matches = [v for v in collection["collection_index"] if all(v[k] == rr[k] for k in ["path", "sha256"]) and v["received_at"] == receipt["received_at"]]
            assert len(matches) == 1 and matches[0]["reviewer_task"] == current["reviewer_task"]
            assert datetime.fromisoformat(report["created_at"]) <= datetime.fromisoformat(receipt["received_at"])
        if row["report"] != prior_sections[pid]["report"]:
            changes.append(pid)
        if pid.startswith("R."):
            assert hashlib.sha256(section(body, pid).encode()).hexdigest() == current["source_sha256"]
            rows.append({"lesson_id": pid, "owner": current["reviewer_task"], "original": row["original_report"], "current": row["report"], "current_created_at": current["created_at"], "current_received_at": row["current_collection"]["received_at"], "source_sha256": current["source_sha256"]})
            details[pid] = {"issues": current["issues"], "original_issue_resolutions": current.get("original_issue_resolutions", [])}
    assert len(owners) == len(set(owners)) == 26
    assert previous["reviewer_task"] not in owners and "/root/omni_reader_entries" not in owners
    intake, intake_run = load(INTAKE), load(EXECUTION)
    assert intake["source_freeze"] == ref(FREEZE)
    assert bind(intake_run["manifest"]) == intake and intake_run["returncode"] == 0
    result = bind(intake_run["result"])
    assert result["status"] == "schema_checked_collection_pending"

    prior_audit = bind(next(a for a in previous["artifacts"] if a["id"] == "a_current_audit"))
    raw_ref = prior_audit["raw_machine_metadata_outside_git"]
    assert sha(raw_ref["path"]) == raw_ref["sha256"]
    prior_raw = load(raw_ref["path"])
    for path, digest in prior_raw["verified_reference_hashes"].items():
        assert sha(ROOT / path) == digest, path
    exact_reader, reader_exceptions = [], []
    for row in prior_raw["reader_rows"]:
        pid = row["page_id"]
        if row["status"] == "current_owner_source_identity" and row["source_sha256"] == pages[pid]["source_sha256"]:
            exact_reader.append(pid)
        else:
            reader_exceptions.append({"page_id": pid, "status": "current_full_reader_pending" if pid == "12.2" else row["status"]})
    assert len(exact_reader) == 363

    index = load("course/lesson-index.json")
    sync = subprocess.run([sys.executable, "scripts/build_course.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert sync.returncode == 0, sync.stderr
    site = ROOT / "outputs/omni-review-site-technical-fix14"
    stored = ROOT / "outputs/omni-notebooks-technical-fix14-current"
    cumulative_path = "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix14-cumulative.json"
    cumulative = load(cumulative_path)
    assert cumulative["source_freeze"] == ref(FREEZE)
    records = {v["lesson"]: v for v in cumulative["records"]}
    counts = Counter()
    notebook_rows = []
    for item in index:
        pid = item["id"]
        p = pages[pid]
        source = (ROOT / p["snapshot"]).read_text()
        assert source == section((ROOT / item["source"]).read_text(), pid)
        notebook = load(item["notebook"])
        executed_path = stored / Path(item["notebook"]).relative_to("notebooks")
        executed = export_course.checked_notebook(notebook, executed_path)
        record = records[pid]
        assert sha(ROOT / item["notebook"]) == record["notebook_sha256"] == p["notebook_sha256"]
        assert sha(executed_path) == sha(record["source"]) == record["executed_sha256"]
        assert (site / item["notebook"]).read_bytes() == (ROOT / item["notebook"]).read_bytes()
        html = (site / (pid + ".html")).read_text()
        assert "https://colab.research.google.com/github/" not in html
        for token in ["https://colab.research.google.com/", "尚未發布的預覽版", "Upload notebook", "File → Upload notebook", "雲端環境未經本課實測"]:
            assert token in html, (pid, token)
        for suffix in [pid + ".html", item["notebook"]]:
            response = urllib.request.urlopen("http://127.0.0.1:8775/" + suffix, timeout=15)
            assert response.status == 200 and response.read() == (site / suffix).read_bytes()
            counts["HTTP_GET200_exact_bytes"] += 1
        counts["notebooks"] += 1
        counts["html_with_CPU_output"] += "實際執行結果 · CPU" in html
        for cell in executed["cells"]:
            if cell["cell_type"] == "code":
                counts["recorded_code_cells"] += 1
                counts["recorded_output_blocks"] += len(cell.get("outputs", []))
                if not cell.get("metadata", {}).get("course_setup") and "course_figure" not in cell.get("metadata", {}):
                    counts["teaching_cells"] += 1
                    for output in cell.get("outputs", []):
                        if output["output_type"] == "stream" and "".join(output.get("text", [])).strip():
                            text = "".join(output["text"]).strip()
                            assert export_course.html.escape(text) in html or all(export_course.html.escape(line) in html for line in text.splitlines())
                            counts["teaching_text_outputs"] += 1
        notebook_rows.append({"page_id": pid, "notebook_sha256": record["notebook_sha256"], "executed_sha256": record["executed_sha256"], "method": record["method"]})
    assert counts["notebooks"] == 331 and counts["teaching_cells"] == 423
    assert Counter(r["method"] for r in records.values()) == {"reuse_matching_cells": 330, "new_independent_kernel": 1}
    kernel_validation = load("outputs/omni-notebooks-technical-fix14-run/validation-kernel.json")
    assert kernel_validation["passed"] == 1 and kernel_validation["failed"] == 0

    estimates = load("course/reading-times.json")
    stale = [r["page_id"] for r in estimates["pages"] if any(r[k] != pages[r["page_id"]][k] for k in ["source_sha256", "figures_sha256"])]
    preview_path = "outputs/omni-reading-times-technical-fix14-preview.json"
    preview_estimates = reading_time.load_estimates(ROOT / preview_path, freeze)
    assert sorted(stale) == ["12.2", "first-steps"] and len(preview_estimates) == 367
    assert all(v in estimates["pages"] for v in preview_estimates.values())
    estimate_guards = []
    for path, complete in [("course/reading-times.json", False), (preview_path, True)]:
        try:
            reading_time.load_estimates(ROOT / path, freeze, require_complete=complete)
        except ValueError as error:
            estimate_guards.append({"path": path, "require_complete": complete, "error": str(error)})
        else:
            raise AssertionError("stale/incomplete estimates unexpectedly accepted")
    build_info = load("outputs/omni-review-site-technical-fix14/build-info.json")
    assert build_info["reading_time_pages"] == 367 and build_info["publication_ready"] is False and build_info["colab_mode"] == "upload"

    texts = {(pid): (ROOT / pages[pid]["snapshot"]).read_text() for pid in ["12.1", "12.2"]}
    blocks = re.findall(r"```python\n(.*?)```", texts["12.2"], re.S)
    assert blocks == re.findall(r"```python\n(.*?)```", (ROOT / old["12.2"]["snapshot"]).read_text(), re.S)
    tone_code = re.findall(r"```python\n(.*?)```", texts["12.1"], re.S)[1]
    inline_tone = ast.parse(tone_code).body[0]
    repository_tone = next(n for n in ast.parse((ROOT / "tiny_perceptron/multimodal.py").read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "tone")
    assert ast.dump(inline_tone) == ast.dump(repository_tone)
    demonstrations = []
    for name, codes in [("exact_source", blocks), ("exercise_with_both_code_and_print_labels", [b.replace("9000.0", "9500.0").replace("7000.0", "6500.0").replace("9000Hz", "9500Hz").replace("7000Hz", "6500Hz") for b in blocks]), ("same_source_with_explicit_inline_precursor", [blocks[0], "import math\nimport torch\n" + tone_code + blocks[1].replace("from tiny_perceptron.multimodal import tone\n", "")])]:
        ns, stdout = {}, io.StringIO()
        with redirect_stdout(stdout):
            for code in codes:
                exec(compile(code, "12.2:" + name, "exec"), ns)
        assert len(ns["high"]) == 16 and torch.allclose(ns["high"], -ns["low"], atol=1e-5)
        assert "每一點都只差正負號 True" in stdout.getvalue()
        if name.startswith("exercise"):
            assert "9500Hz前四點" in stdout.getvalue() and "6500Hz前四點" in stdout.getvalue() and "9000Hz" not in stdout.getvalue()
        demonstrations.append({"case": name, "stdout": stdout.getvalue(), "scope": "Exact short CPU source/practice only; not sampling-theory validation, training, recording or cloud run."})

    links = []
    for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body):
        if "://" in target:
            continue
        rel, _, fragment = target.partition("#")
        p = (ROOT / course["source"]).parent / rel if rel else ROOT / course["source"]
        assert p.is_file()
        if fragment:
            assert fragment in re.findall(r"^## ([A-Z\d]+\.\d+) ", p.read_text(), re.M)
        links.append(target)
    guards = []
    index_one = [next(v for v in index if v["id"] == "1.1")]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, items, revision, expected in [("symbolic_ref", index, "main", "rejected"), ("nonexistent_commit", index, "0" * 40, "rejected"), ("true_commit_one_notebook_subset", index_one, head, "accepted"), ("old_pin_all331", index, build_info["revision"], "rejected")]:
        try:
            export_course.verify_colab_notebooks(items, revision)
            outcome, error = "accepted", ""
        except ValueError as exc:
            outcome, error = "rejected", str(exc).splitlines()[0]
        assert outcome == expected
        guards.append({"case": name, "result": outcome, "error_first_line": error})

    raw_path = CACHE / "current-audit-full-v2.json"
    raw = {"notebooks": notebook_rows, "section_details": details, "source14_exact_reader_page_ids": exact_reader, "reused_reference_hashes": prior_raw["verified_reference_hashes"]}
    with raw_path.open("x") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
        f.write("\n")
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": previous["reviewer_task"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU; bounded source demonstrations; read-only files/Git/HTTP"},
        "previous_current": ref(PRIOR), "source_freeze": ref(FREEZE), "changed_canonical_pages": changed, "canonical_snapshot_count_checked": len(pages), "course_source_sha256": course["source_sha256"],
        "sidecar": ref(BUNDLE), "issue_inventory": bundle["issue_inventory"], "issue_inventory_counts": {"reports": len(issues["reports"]), "issues": len(issues["issues"])}, "section_count": len(owners), "unique_section_owners": len(set(owners)), "section_report_changes": changes, "own_R_section_bindings": rows,
        "actual_collection": bundle["actual_collection_snapshot"], "collection_rows": len(collection["collection_index"]), "collection_rows_verbatim_in_raw_ledger": True, "collection_scope": collection["scope"], "intake_manifest": ref(INTAKE), "intake_execution": ref(EXECUTION), "intake_result_status": result["status"],
        "unchanged_previous_artifacts_count": len(previous["artifacts"]), "unchanged_repository_source_count": len(repo_sources), "unchanged_exporter": ref("scripts/export_course.py"), "reused_reader_reference_hashes_checked": len(prior_raw["verified_reference_hashes"]), "current_reader_source_identity_count": len(exact_reader), "reader_exceptions": reader_exceptions,
        "builder_check": {"returncode": sync.returncode, "stdout": sync.stdout.strip()}, "current_delivery_counts": dict(counts), "cumulative_receipt": ref(cumulative_path), "execution_methods": dict(Counter(r["method"] for r in records.values())), "new_12_2_raw_record": records["12.2"], "new_12_2_raw_kernel_validation": kernel_validation,
        "reading_times": {"metadata": ref("course/reading-times.json"), "metadata_count": len(estimates["pages"]), "current_identity_count": len(estimates["pages"]) - len(stale), "stale_pages": stale, "preview_metadata": ref(preview_path), "preview_current_entries": len(preview_estimates), "kind": estimates["estimate_kind"], "actual_guards": estimate_guards}, "build_info": build_info,
        "new_12_2_source_sha256": pages["12.2"]["source_sha256"], "12_2_python_fences_unchanged": True, "12_1_inline_tone_matches_current_AST": True, "12_2_demonstrations": demonstrations, "local_links_verified": len(links), "release_guard_cases": guards,
        "raw_machine_metadata_outside_git": {"path": str(raw_path), "sha256": sha(raw_path), "bytes": raw_path.stat().st_size},
        "scope": "source14 bounded whole-course dependency callback. Reuses same-byte own prior substantive reading and primary evidence, with complete course-context reread and new12.2/12.1 text/code reading. All331 delivery identity and records rechecked, not331 new executions or new semantic readings. No formal training, GPU, cloud, publication, stage seal,210-tool approval or whole-book blind read."
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
