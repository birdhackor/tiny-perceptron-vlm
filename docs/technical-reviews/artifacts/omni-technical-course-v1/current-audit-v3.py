"""Bounded source15 course callback; retain full machine lists outside Git."""
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
CACHE = Path("/workspace/work/technical-course-current-v3")
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"
BUNDLE = "docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v5.json"
PRIOR = "docs/course-revision-20261029-omni/reviews/technical/technical_course/technical-current-v2.json"
INTAKE = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v6.json"
EXECUTION = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-execution-v6.json"


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
    assert sha(ROOT / PRIOR) == "28b05d17fafe732bc265642dc15ecb6b3fa34ff75879b901b965cab0e10e7a69"
    freeze = load(FREEZE)
    assert sha(ROOT / FREEZE) == "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
    old = {p["page_id"]: p for p in bind(previous["source_freeze"])["pages"]}
    pages = {p["page_id"]: p for p in freeze["pages"]}
    assert set(old) == set(pages)
    changed = [pid for pid, p in pages.items() if any(p.get(k) != old[pid].get(k) for k in ["source_sha256", "figures_sha256", "notebook_sha256"])]
    assert changed == ["5.7", "chapter-10", "chapter-14"], changed
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
    assert sha(ROOT / BUNDLE) == "0077db6122efdd4371d02334fa8aab5ab7002cd3312431b48fc57660bcb66d4f"
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
    assert INTAKE in intake_run["command"] and intake_run["returncode"] == 0
    result = bind(intake_run["result"])
    assert result["status"] == "schema_checked_collection_pending"

    assignments = bind(intake["assignments"])
    report_metadata, seen_reports, legal_old_sources = [], set(), []
    for row in bundle["sections"]:
        current = bind(row["report"])
        tasks = [a for a in assignments["tasks"] if a.get("task") == current["reviewer_task"] and a.get("role") == "technical" and a.get("lesson_id") == row["lesson_id"] and a.get("context") == "fresh"]
        assert len(tasks) == 1
        if row.get("current_dispatch_window"):
            dispatches = assignments.get("callback_dispatches", []) + assignments["tasks"]
            assert any(a.get("task") == current["reviewer_task"] and a.get("dispatch_window") == row["current_dispatch_window"] for a in dispatches)
            assert datetime.fromisoformat(current["created_at"]) >= datetime.fromisoformat(row["current_dispatch_window"]["not_before"])
        original_freeze = bind(row["original_section_freeze"])
        canonical = bind(original_freeze["canonical_source_freeze"])
        snapshots = {v["source"]: v["snapshot"] for v in row.get("original_source_snapshots", [])}
        for role, key in [("original", "original_report"), ("current", "report")]:
            rr = row[key]
            if rr["path"] in seen_reports:
                continue
            seen_reports.add(rr["path"])
            report = bind(rr)
            sources = {v["id"]: v for v in report["sources"]}
            artifacts = {v["id"]: v for v in report["artifacts"]}
            assert len(sources) == len(report["sources"]) and len(artifacts) == len(report["artifacts"])
            for artifact in artifacts.values():
                assert sha(ROOT / artifact["path"]) == artifact["sha256"], artifact["path"]
            for claim in report["claims"]:
                assert claim["kind"] in {"concept", "numeric", "software", "empirical"}
                assert claim["status"] in {"verified", "unresolved", "contradicted"}
                for field in ["statement", "location", "scope"]:
                    assert isinstance(claim[field], str) and claim[field].strip()
                assert isinstance(claim["evidence"], list)
                for evidence in claim["evidence"]:
                    assert evidence["source_id"] in sources and evidence["locator"] and evidence["supports"]
                for aid in claim.get("artifact_ids", []):
                    assert aid in artifacts
            for issue in report["issues"]:
                assert issue["source_quote"] and issue["status"]
            for source in sources.values():
                if source.get("kind") != "repository_code":
                    continue
                path, want = source["path"], source["sha256"]
                if sha(ROOT / path) == want:
                    if source.get("authored_source") and source.get("selector", "").startswith("section:"):
                        pid = source["selector"].removeprefix("section:")
                        assert (ROOT / path).read_text() == section((ROOT / source["authored_source"]).read_text(), pid)
                    continue
                assert role == "original", (rr["path"], path)
                snapshot = snapshots.get(path)
                if snapshot:
                    assert sha(ROOT / snapshot["path"]) == snapshot["sha256"] == want
                    method = {"method": "registered_original_source_snapshot", "snapshot": snapshot}
                else:
                    pieces = [v for v in canonical["pages"] if v.get("source") == path]
                    intros = [v for v in pieces if v.get("selector") == "before-first-section"]
                    sections = [v for v in pieces if v.get("selector", "").startswith("section:")]
                    assert len(intros) == 1 and len(sections) + 1 == len(pieces)
                    for part in intros + sections:
                        assert sha(ROOT / part["snapshot"]) == part["source_sha256"]
                    reconstructed = (ROOT / intros[0]["snapshot"]).read_text() + "".join((ROOT / v["snapshot"]).read_text() for v in sections)
                    assert hashlib.sha256(reconstructed.encode()).hexdigest() == want
                    method = {"method": "exact_original_canonical_freeze_chapter_reconstruction", "canonical_freeze": original_freeze["canonical_source_freeze"], "part_count": len(pieces)}
                legal_old_sources.append({"report": rr, "path": path, "sha256": want, **method})
            report_metadata.append({"report": rr, "lesson_id": row["lesson_id"], "role": role, "claim_count": len(report["claims"]), "claim_status_counts": dict(Counter(v["status"] for v in report["claims"])), "issue_count": len(report["issues"]), "source_count": len(sources), "artifact_count": len(artifacts)})
    for rr in issues["reports"]:
        bind(rr)
    for issue in issues["issues"]:
        doc = bind(issue["report"])
        target = doc
        for key in issue["json_pointer"].strip("/").split("/"):
            target = target[int(key)] if isinstance(target, list) else target[key]
        assert target["source_quote"]
    assert len(report_metadata) == 36

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
            reader_exceptions.append({"page_id": pid, "status": "current_full_reader_pending" if pid in {"12.2", "5.7", "chapter-10", "chapter-14"} else row["status"]})
    assert len(exact_reader) == 360

    index = load("course/lesson-index.json")
    sync = subprocess.run([sys.executable, "scripts/build_course.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert sync.returncode == 0, sync.stderr
    site = ROOT / "outputs/omni-review-site-technical-fix15"
    stored = ROOT / "outputs/omni-notebooks-technical-fix15-current"
    cumulative_path = "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix15-cumulative.json"
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
            response = urllib.request.urlopen("http://127.0.0.1:8776/" + suffix, timeout=15)
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
    kernel_validation = load("outputs/omni-notebooks-technical-fix15-run/validation-kernel.json")
    assert kernel_validation["passed"] == 1 and kernel_validation["failed"] == 0

    estimates = load("course/reading-times.json")
    stale = [r["page_id"] for r in estimates["pages"] if any(r[k] != pages[r["page_id"]][k] for k in ["source_sha256", "figures_sha256"])]
    preview_path = "outputs/omni-reading-times-technical-fix15-preview.json"
    preview_estimates = reading_time.load_estimates(ROOT / preview_path, freeze)
    assert sorted(stale) == ["12.2", "5.7", "chapter-10", "chapter-14", "first-steps"] and len(preview_estimates) == 364
    assert all(v in estimates["pages"] for v in preview_estimates.values())
    estimate_guards = []
    for path, complete in [("course/reading-times.json", False), (preview_path, True)]:
        try:
            reading_time.load_estimates(ROOT / path, freeze, require_complete=complete)
        except ValueError as error:
            estimate_guards.append({"path": path, "require_complete": complete, "error": str(error)})
        else:
            raise AssertionError("stale/incomplete estimates unexpectedly accepted")
    build_info = load("outputs/omni-review-site-technical-fix15/build-info.json")
    assert build_info["reading_time_pages"] == 364 and build_info["publication_ready"] is False and build_info["colab_mode"] == "upload"

    text = (ROOT / pages["5.7"]["snapshot"]).read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks == re.findall(r"```python\n(.*?)```", (ROOT / old["5.7"]["snapshot"]).read_text(), re.S)
    assert len(blocks) == 1
    demonstrations = []
    for name, code, expected in [("exact_source", blocks[0], "completed"), ("exercise_omit_optimizer_history", blocks[0].replace('    restored_optimizer.load_state_dict(payload["optimizer"])', '    # restored_optimizer.load_state_dict(payload["optimizer"])'), "second_assertion_failed")]:
        ns, stdout = {}, io.StringIO()
        assertion_lines = [n.lineno for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Assert)]
        outcome, failed_line = "completed", None
        try:
            with redirect_stdout(stdout):
                exec(compile(code, "5.7:" + name, "exec"), ns)
        except AssertionError as error:
            tb = error.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            failed_line = tb.tb_lineno
            assert failed_line == max(assertion_lines)
            outcome = "second_assertion_failed"
        assert outcome == expected
        demonstrations.append({"case": name, "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "outcome": outcome, "assertion_lines": assertion_lines, "failed_line": failed_line, "stdout": stdout.getvalue(), "scope": "Exact tiny CPU checkpoint demonstration/practice, three visible AdamW step calls per case; not formal or long training, GPU, historical experiment reproduction, safety guarantee or cloud."})
    context_routes = []
    for pid, expected_links in [("chapter-10", ["10.9.html", "10.12.html"]), ("chapter-14", ["4.6.html", "14.19.html"]), ("course", [])]:
        htmlpath = site / (pid + ".html")
        response = urllib.request.urlopen("http://127.0.0.1:8776/" + pid + ".html", timeout=15)
        response_bytes = response.read()
        assert response.status == 200 and response_bytes == htmlpath.read_bytes()
        for link in expected_links:
            assert link in response_bytes.decode()
        context_routes.append({"page_id": pid, "status": response.status, "sha256": hashlib.sha256(response_bytes).hexdigest(), "expected_links_present": expected_links})

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

    raw_path = CACHE / "current-audit-full-v3.json"
    raw = {"notebooks": notebook_rows, "section_details": details, "source15_exact_reader_page_ids": exact_reader, "reused_reference_hashes": prior_raw["verified_reference_hashes"], "all_section_report_metadata": report_metadata}
    with raw_path.open("x") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
        f.write("\n")
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": previous["reviewer_task"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU; bounded source demonstrations; read-only files/Git/HTTP"},
        "previous_current": ref(PRIOR), "source_freeze": ref(FREEZE), "changed_canonical_pages": changed, "canonical_snapshot_count_checked": len(pages), "course_source_sha256": course["source_sha256"],
        "sidecar": ref(BUNDLE), "issue_inventory": bundle["issue_inventory"], "issue_inventory_counts": {"reports": len(issues["reports"]), "issues": len(issues["issues"])}, "section_count": len(owners), "unique_section_owners": len(set(owners)), "section_report_changes": changes, "own_R_section_bindings": rows, "all_section_unique_reports_checked": len(report_metadata), "all_section_claim_metadata_checked": sum(v["claim_count"] for v in report_metadata), "all_section_issue_metadata_checked": sum(v["issue_count"] for v in report_metadata), "legal_original_source_bindings": legal_old_sources, "assignments": intake["assignments"],
        "actual_collection": bundle["actual_collection_snapshot"], "collection_rows": len(collection["collection_index"]), "collection_rows_verbatim_in_raw_ledger": True, "collection_scope": collection["scope"], "intake_manifest": ref(INTAKE), "intake_execution": ref(EXECUTION), "intake_result_status": result["status"],
        "unchanged_previous_artifacts_count": len(previous["artifacts"]), "unchanged_repository_source_count": len(repo_sources), "unchanged_exporter": ref("scripts/export_course.py"), "reused_reader_reference_hashes_checked": len(prior_raw["verified_reference_hashes"]), "current_reader_source_identity_count": len(exact_reader), "reader_exceptions": reader_exceptions,
        "builder_check": {"returncode": sync.returncode, "stdout": sync.stdout.strip()}, "current_delivery_counts": dict(counts), "cumulative_receipt": ref(cumulative_path), "execution_methods": dict(Counter(r["method"] for r in records.values())), "new_5_7_raw_record": records["5.7"], "new_5_7_raw_kernel_validation": kernel_validation,
        "reading_times": {"metadata": ref("course/reading-times.json"), "metadata_count": len(estimates["pages"]), "current_identity_count": len(estimates["pages"]) - len(stale), "stale_pages": stale, "preview_metadata": ref(preview_path), "preview_current_entries": len(preview_estimates), "kind": estimates["estimate_kind"], "actual_guards": estimate_guards}, "build_info": build_info,
        "new_5_7_source_sha256": pages["5.7"]["source_sha256"], "5_7_python_fences_unchanged": True, "5_7_demonstrations": demonstrations, "new_context_routes": context_routes, "local_links_verified": len(links), "release_guard_cases": guards,
        "raw_machine_metadata_outside_git": {"path": str(raw_path), "sha256": sha(raw_path), "bytes": raw_path.stat().st_size},
        "scope": "source15 bounded whole-course dependency callback. Reuses same-byte own prior substantive reading/primary evidence, with complete course-context,5.7,two changed introductions and necessary10.12/14.19 full text/code read. All331 delivery and36 original/current reports claim/source/artifact/issue metadata checked; not section subject approvals,331 new executions or new semantic readings. No formal/long training, GPU, cloud, publication, stage seal,210-tool approval or whole-book blind read."
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
