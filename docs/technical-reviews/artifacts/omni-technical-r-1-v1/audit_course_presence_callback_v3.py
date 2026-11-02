"""R.1 current-v3: recompute fix13 identity and manual-upload preview scope."""

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
import html
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
REVIEWS = ROOT / "docs/course-revision-20261029-omni/reviews"
EVIDENCE = ROOT / "docs/course-revision-20261029-omni/evidence"
FREEZE = REVIEWS / "source-freeze-technical-fix13.json"
OLD_FREEZE = REVIEWS / "source-freeze-technical-fix11.json"
CATALOGUE = REVIEWS / "reader-evidence-catalogue-technical-fix07-v1.json"
CUMULATIVE = EVIDENCE / "notebooks-technical-fix13-cumulative.json"
EXECUTED = ROOT / "outputs/omni-notebooks-technical-fix13-current"
SITE = ROOT / "outputs/omni-review-site-technical-fix13"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def reference(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def joined(value):
    return "".join(value) if isinstance(value, list) else value


def pointer(data, path):
    for part in path.strip("/").split("/"):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs).get("href", ""))


    def handle_data(self, data):
        self.text.append(data)


freeze = json.loads(FREEZE.read_text())
old = {p["page_id"]: p for p in json.loads(OLD_FREEZE.read_text())["pages"]}
by_id = {p["page_id"]: p for p in freeze["pages"]}
lessons = [p for p in freeze["pages"] if p["kind"] == "lesson"]
cumulative = json.loads(CUMULATIVE.read_text())
assert cumulative["source_freeze"] == reference(FREEZE)
records = {r["lesson"]: r for r in cumulative["records"]}
assert len(lessons) == len(records) == 331
index = json.loads((ROOT / "course/lesson-index.json").read_text())
assert {p["id"] for p in index} == {p["page_id"] for p in lessons}
index = {p["id"]: p for p in index}
build_info = json.loads((SITE / "build-info.json").read_text())
pin = build_info["revision"]
errors, counts, metadata, changed = [], Counter(), Counter(), []

for page in freeze["pages"]:
    snapshot = ROOT / page["snapshot"]
    if digest(snapshot) != page["source_sha256"]:
        errors.append([page["page_id"], "frozen source SHA mismatch"])
    if any(page.get(k) != old[page["page_id"]].get(k) for k in ["source_sha256", "notebook_sha256", "figures_sha256"]):
        changed.append(page["page_id"])

for page in lessons:
    pid = page["page_id"]
    raw = (ROOT / page["snapshot"]).read_text()
    live = re.search(r"^## " + re.escape(pid) + r"\s.*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S).group()
    if sha256(live.encode()).hexdigest() != page["source_sha256"]:
        errors.append([pid, "live source differs from freeze"])
    nbpath = ROOT / page["notebook"]
    assert digest(nbpath) == page["notebook_sha256"]
    assert index[pid]["source"] == page["source"] and index[pid]["notebook"] == page["notebook"]
    nb = json.loads(nbpath.read_text())
    assert nb["metadata"]["lesson_id"] == pid
    metadata[(nb["nbformat"], nb["nbformat_minor"], nb["metadata"]["kernelspec"]["name"], nb["metadata"]["language_info"]["version"])] += 1
    blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", raw, re.S)]
    teaching = [c for c in nb["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and not c.get("metadata", {}).get("course_figure")]
    assert [joined(c["source"]).strip() for c in teaching] == blocks
    counts["lessons"] += 1
    counts["python_fences"] += len(blocks)
    counts["teaching_cells"] += len(teaching)
    record = records[pid]
    executed_path = EXECUTED / Path(page["notebook"]).relative_to("notebooks")
    assert str(executed_path) == record["bundle_path"]
    assert record["notebook_sha256"] == page["notebook_sha256"]
    assert digest(executed_path) == record["executed_sha256"] == digest(Path(record["source"]))
    executed = json.loads(executed_path.read_text())
    assert [(c["cell_type"], joined(c["source"])) for c in nb["cells"]] == [(c["cell_type"], joined(c["source"])) for c in executed["cells"]]
    assert digest(SITE / page["notebook"]) == page["notebook_sha256"]
    rendered = html.unescape((SITE / (pid + ".html")).read_text())
    links = Links()
    links.feed(rendered)
    assert page["notebook"] in links.links
    assert "https://colab.research.google.com/" in links.links
    assert not any(u.startswith("https://colab.research.google.com/github/") for u in links.links)
    text = " ".join(links.text)
    for phrase in ["尚未發布的預覽版", "Colab 直連尚未提供", "Upload notebook", "File → Upload notebook", "需要 Google 帳號", "雲端環境未經本課實測"]:
        assert phrase in text, (pid, phrase)
    assert 'id="colab-upload-instructions"' in rendered
    counts["candidate_downloads_matching_current_bytes"] += 1
    counts["manual_upload_footers_with_all_required_text"] += 1
    has_text = False
    for cell in executed["cells"]:
        if cell["cell_type"] != "code":
            continue
        counts["recorded_executed_code_cells"] += 1
        assert cell.get("execution_count") is not None
        assert not any(o["output_type"] == "error" for o in cell.get("outputs", []))
        is_teaching = not cell.get("metadata", {}).get("course_setup") and not cell.get("metadata", {}).get("course_figure")
        if not is_teaching:
            continue
        for output in cell.get("outputs", []):
            text = joined(output["text"]) if output["output_type"] == "stream" else joined(output.get("data", {}).get("text/plain", ""))
            if text:
                assert text in rendered
                has_text = True
                counts["teaching_text_outputs_in_candidate_html"] += 1
    if has_text:
        assert "實際執行結果 · CPU" in rendered
        counts["lesson_pages_with_teaching_text_output"] += 1

assert build_info["colab_mode"] == "upload"
assert build_info["publication_ready"] is False
assert build_info["colab_git_objects_verified"] is False
estimate_path = Path("/workspace/work/omni-reading-times-preview-fix13.json")
estimates = json.loads(estimate_path.read_text())["pages"]
estimate_errors = []
for estimate in estimates:
    pid = estimate["page_id"]
    if estimate["source_sha256"] != by_id[pid]["source_sha256"] or estimate["figures_sha256"] != by_id[pid]["figures_sha256"]:
        estimate_errors.append(pid)
assert not estimate_errors
pending_reader_pages = ["9.12", "9.15", "first-steps"]
assert sorted(set(by_id) - {e["page_id"] for e in estimates}) == sorted(pending_reader_pages)
assert len(estimates) == build_info["reading_time_pages"] == 366

catalogue = json.loads(CATALOGUE.read_text())
overlays, overlay_receipts = {}, []
for number in [5, 6, 7]:
    path = REVIEWS / f"reader-late-callback-intake-v{number}.json"
    data = json.loads(path.read_text())
    overlay_receipts.append(reference(path))
    for entry in data["entries"]:
        assert digest(ROOT / entry["path"]) == entry["sha256"]
        if entry["kind"] == "reader-current-v1":
            report = json.loads((ROOT / entry["path"]).read_text())
            page = next(p for p in report["pages"] if p["page_id"] == entry["page_id"])
            assert page["source_sha256"] == by_id[entry["page_id"]]["source_sha256"]
            for item in [page["basis_report"], page["session_evidence"]["opening"], *page["session_evidence"]["notes"]]:
                assert digest(ROOT / item["path"]) == item["sha256"]
            overlays[entry["page_id"]] = {"current_report": {"path": entry["path"], "sha256": entry["sha256"]}, "source_sha256": page["source_sha256"], "basis_report": page["basis_report"], "session": page["session"], "raw_notes_hash_checked": len(page["session_evidence"]["notes"])}

reader_counts, retained = Counter(), 0
for row in catalogue["rows"]:
    pid = row["page_id"]
    reader_counts[row["status"]] += 1
    if pid in ["9.12", "9.15"]:
        reader_counts["current_owner_source_identity"] -= 1
        reader_counts["current_full_reader_pending"] += 1
        continue
    if pid in overlays or row["status"] == "preserved_inherit_not_new_read":
        continue
    for item in [row["current_report"], row["basis"]["report"]]:
        assert digest(ROOT / item["path"]) == item["sha256"]
    report = pointer(json.loads((ROOT / row["current_report"]["path"]).read_text()), row["current_pointer"])
    if row["status"] == "current_owner_source_identity":
        assert report["source_sha256"] == by_id[pid]["source_sha256"]
        retained += 1

chains = []
for suffix, pid in [("06", "9.11"), ("09", "9.18"), ("11", "12.17"), ("12", "9.12"), ("13", "9.15")]:
    path = EVIDENCE / f"notebooks-technical-fix{suffix}-cumulative.json"
    data = json.loads(path.read_text())
    record = next(r for r in data["records"] if r["lesson"] == pid)
    assert record["method"] == "new_independent_kernel"
    assert digest(Path(record["source"])) == record["executed_sha256"]
    assert record["executed_sha256"] == records[pid]["executed_sha256"]
    validation_path = ROOT / f"outputs/omni-notebooks-technical-fix{suffix}-run/validation-kernel.json"
    validation = json.loads(validation_path.read_text())
    assert validation["mode"] == "kernel" and validation["passed"] == 1 and validation["failed"] == 0
    chains.append({"page_id": pid, "receipt": reference(path), "kernel_record": record, "validation": reference(validation_path), "validation_result": validation, "kernel_command": data["kernel_command"], "latest_bundle_sha256": records[pid]["executed_sha256"]})

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/audit_course_presence_callback_v3.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; static files and read-only git object inspection; no lesson or model execution"},
    "source_freeze": reference(FREEZE), "comparison_freeze": reference(OLD_FREEZE),
    "cumulative_execution_receipt": reference(CUMULATIVE),
    "candidate_site_root": str(SITE.relative_to(ROOT)), "candidate_build_info": build_info,
    "changed_canonical_sources_since_v1": changed,
    "counts": dict(counts),
    "notebook_metadata_profiles": [{"nbformat": k[0], "minor": k[1], "kernel": k[2], "language_version": k[3], "count": v} for k, v in metadata.items()],
    "reader_mapping": {"navigation_catalogue": reference(CATALOGUE), "new_intakes": overlay_receipts, "retained_exact_source_reader_rows": retained, "explicit_current_overlays": overlays, "current_source_identity_total_after_overlays": retained + len(overlays), "other_status_counts": {k:v for k,v in reader_counts.items() if k != "current_owner_source_identity"}, "note":"359 retained mappings plus three real current overlays = 362 exact-source reader identities. New 9.12/9.15 readers and the existing first-steps reader remain pending; no old 364 mapping or blind/human reading is claimed."},
    "specific_kernel_chains_inspected": chains,
    "preview_colab_route": {"mode": "upload", "active_github_pin_links": 0, "manual_upload_footers": 331, "publication_ready": False, "colab_git_objects_verified": False, "scope": "All current download bytes precede the official-home Upload notebook route; no stale committed Notebook link remains active. This is an uncommitted preview, not a published release or cloud execution."},
    "current_estimates": {"path_outside_git": str(estimate_path), "sha256": digest(estimate_path), "count": len(estimates), "source_and_figure_mismatches": estimate_errors, "pending_pages": pending_reader_pages},
    "identity_errors": errors,
    "scope":"Full 331-lesson identity/metadata/download/output/manual-upload audit plus three reader callback overlays, five actual raw kernel chains, and 366 source/figure-matched AI estimates. Existing author kernel results are inspected, not attributed to this reviewer. Semantic primary text and note reading is separately recorded. This does not establish publication, cloud execution, whole-book rereading, or current first-steps full-reader completion.",
}
with Path(__file__).with_name("course-presence-callback-v3.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"counts": dict(counts), "changed": changed, "reader_retained": retained, "reader_updated": list(overlays), "reader_identity_total": retained + len(overlays), "manual_upload_footers": 331, "AI_estimates": len(estimates), "pending_pages": pending_reader_pages, "identity_errors": errors}, ensure_ascii=False, indent=2))
