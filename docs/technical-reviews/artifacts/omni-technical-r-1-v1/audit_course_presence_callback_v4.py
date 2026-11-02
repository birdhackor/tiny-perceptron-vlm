"""R.1 bounded source14 dependency audit; no historical receipt is rewritten."""

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import html
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import build_course, export_course, reading_time

REVIEWS = ROOT / "docs/course-revision-20261029-omni/reviews"
EVIDENCE = ROOT / "docs/course-revision-20261029-omni/evidence"
FREEZE = REVIEWS / "source-freeze-technical-fix14.json"
OLD = REVIEWS / "source-freeze-technical-fix13.json"
NUMBERED = REVIEWS / "numbered-freeze-v12.json"
SITE = ROOT / "outputs/omni-review-site-technical-fix14"
BASE_URL = "http://127.0.0.1:8775/"
CUMULATIVE = EVIDENCE / "notebooks-technical-fix14-cumulative.json"
PREVIEW = ROOT / "outputs/omni-reading-times-technical-fix14-preview.json"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def joined(value):
    return "".join(value) if isinstance(value, list) else value


def pointer(value, path):
    for part in path.strip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.text = [], []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs).get("href", ""))

    def handle_data(self, data):
        self.text.append(data)


freeze = json.loads(FREEZE.read_text())
by_id = {p["page_id"]: p for p in freeze["pages"]}
old = {p["page_id"]: p for p in json.loads(OLD.read_text())["pages"]}
index = json.loads((ROOT / "course/lesson-index.json").read_text())
index_by_id = {p["id"]: p for p in index}
lessons = [p for p in freeze["pages"] if p["kind"] == "lesson"]
inventory = reading_time.build_inventory(ROOT, index, export_course.DOCUMENTS, export_course.home_introduction(index, True))
live = {p["page_id"]: p for p in inventory["pages"]}
assert set(live) == set(by_id) and len(lessons) == len(index) == 331
changed, figures = [], {}
for pid, page in by_id.items():
    assert digest(ROOT / page["snapshot"]) == page["source_sha256"]
    assert all(live[pid][k] == page[k] for k in ["source_sha256", "figures_sha256", "source", "selector"])
    differences = [k for k in ["source_sha256", "figures_sha256", "notebook_sha256"] if page.get(k) != old[pid].get(k)]
    if differences:
        changed.append({"page_id": pid, "fields": differences})
    figures.update(page["figures_sha256"])
for path, expected in figures.items():
    assert digest(ROOT / path) == expected
    assert digest(SITE / "figures" / Path(path).name) == expected
assert changed == [{"page_id": "12.2", "fields": ["source_sha256", "notebook_sha256"]}]
numbered = json.loads(NUMBERED.read_text())
assert numbered["canonical_source_freeze"] == ref(FREEZE)
r1 = next(p for p in numbered["pages"] if p["page_id"] == "R.1")
current_r1 = re.search(r"^## R\.1\s.*?(?=^## |\Z)", (ROOT / r1["source"]).read_text(), re.M | re.S).group()
assert sha256(current_r1.encode()).hexdigest() == digest(ROOT / r1["snapshot"]) == r1["source_sha256"]
assert r1["figures_sha256"] == {}

cumulative = json.loads(CUMULATIVE.read_text())
assert cumulative["source_freeze"] == ref(FREEZE)
records = {r["lesson"]: r for r in cumulative["records"]}
assert len(records) == 331
assert Counter(r["method"] for r in records.values()) == {"reuse_matching_cells": 330, "new_independent_kernel": 1}
counts, profiles, http_rows = Counter(), Counter(), []
for page in lessons:
    pid = page["page_id"]
    raw = (ROOT / page["snapshot"]).read_text()
    notebook_path = ROOT / page["notebook"]
    assert digest(notebook_path) == page["notebook_sha256"]
    nb = json.loads(notebook_path.read_text())
    assert nb["metadata"]["lesson_id"] == pid
    item = index_by_id[pid]
    assert all(item[k] == page[k] for k in ["source", "notebook", "title"])
    parts = re.split(r"^## ([\dABC]+\.\d+) (.+)$", raw, flags=re.M)
    generated = build_course.notebook(pid, parts[2], parts[3], ROOT / page["source"])
    assert nb == generated
    blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", raw, re.S)]
    teaching = [c for c in nb["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and not c.get("metadata", {}).get("course_figure")]
    assert blocks == [joined(c["source"]).strip() for c in teaching]
    counts.update(lessons=1, python_fences=len(blocks), teaching_cells=len(teaching))
    profiles[(nb["nbformat"], nb["nbformat_minor"], nb["metadata"]["kernelspec"]["name"], nb["metadata"]["language_info"]["version"])] += 1
    record = records[pid]
    executed_path = Path(record["bundle_path"])
    assert executed_path == ROOT / "outputs/omni-notebooks-technical-fix14-current" / Path(page["notebook"]).relative_to("notebooks")
    assert record["notebook_sha256"] == page["notebook_sha256"]
    assert digest(executed_path) == record["executed_sha256"] == digest(Path(record["source"]))
    executed = json.loads(executed_path.read_text())
    assert [(c["cell_type"], joined(c["source"])) for c in nb["cells"]] == [(c["cell_type"], joined(c["source"])) for c in executed["cells"]]
    assert digest(SITE / page["notebook"]) == page["notebook_sha256"]
    # Actual HTTP responses are also compared with the saved new site, for all331.
    with urlopen(BASE_URL + pid + ".html", timeout=10) as response:
        web = response.read()
        assert response.status == 200 and web == (SITE / (pid + ".html")).read_bytes()
    with urlopen(BASE_URL + page["notebook"], timeout=10) as response:
        download = response.read()
        assert response.status == 200 and sha256(download).hexdigest() == page["notebook_sha256"]
    http_rows.append({"page_id": pid, "html_sha256": sha256(web).hexdigest(), "download_sha256": sha256(download).hexdigest(), "status": 200})
    rendered = html.unescape(web.decode())
    links = Links()
    links.feed(rendered)
    assert page["notebook"] in links.links and "https://colab.research.google.com/" in links.links
    assert not any(u.startswith("https://colab.research.google.com/github/") for u in links.links)
    visible_text = " ".join(links.text)
    for phrase in ["尚未發布的預覽版", "Colab 直連尚未提供", "Upload notebook", "File → Upload notebook", "需要 Google 帳號", "雲端環境未經本課實測"]:
        assert phrase in visible_text, (pid, phrase)
    assert 'id="colab-upload-instructions"' in rendered
    counts.update(candidate_downloads_matching_current_bytes=1, manual_upload_footers_with_all_required_text=1, actual_HTTP_page_and_download_pairs=1)
    has_text = False
    for cell in executed["cells"]:
        if cell["cell_type"] != "code":
            continue
        counts["recorded_executed_code_cells"] += 1
        assert cell.get("execution_count") is not None
        assert not any(o["output_type"] == "error" for o in cell.get("outputs", []))
        if cell.get("metadata", {}).get("course_setup") or cell.get("metadata", {}).get("course_figure"):
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

build_info = json.loads((SITE / "build-info.json").read_text())
assert build_info["colab_mode"] == "upload" and build_info["publication_ready"] is False and build_info["colab_git_objects_verified"] is False
with urlopen(BASE_URL + "build-info.json", timeout=10) as response:
    assert response.read() == (SITE / "build-info.json").read_bytes()
preview = reading_time.load_estimates(PREVIEW, inventory)
metadata_path = ROOT / "course/reading-times.json"
metadata = json.loads(metadata_path.read_text())["pages"]
stale = [e["page_id"] for e in metadata if e["source_sha256"] != by_id[e["page_id"]]["source_sha256"] or e["figures_sha256"] != by_id[e["page_id"]]["figures_sha256"]]
assert sorted(stale) == ["12.2", "first-steps"]
assert len(preview) == build_info["reading_time_pages"] == len(metadata) - len(stale) == 367
assert sorted(set(by_id) - set(preview)) == sorted(stale)
preview_receipt_path = EVIDENCE / "preview-estimates-technical-fix14-v1.json"
preview_receipt = json.loads(preview_receipt_path.read_text())
assert digest(PREVIEW) == preview_receipt["preview_metadata_sha256"]
assert digest(metadata_path) == preview_receipt["original_metadata_sha256"]

# Recalculate old-pin bytes from actual Git objects, rather than reuse v2/v3 counts.
pin = build_info["revision"]
requests = "".join(f"{pin}:{p['notebook']}\n" for p in lessons).encode()
batch = subprocess.run(["git", "cat-file", "--batch"], input=requests, capture_output=True, cwd=ROOT, check=True, timeout=30)
stream = BytesIO(batch.stdout)
pin_rows = {"same_current_bytes": [], "different_current_bytes": [], "absent_at_pin": []}
for page in lessons:
    header = stream.readline().rstrip(b"\n").split()
    row = {"page_id": page["page_id"], "notebook": page["notebook"], "current_sha256": page["notebook_sha256"]}
    if len(header) == 2 and header[-1] == b"missing":
        pin_rows["absent_at_pin"].append(row)
        continue
    assert len(header) == 3 and header[1] == b"blob"
    data = stream.read(int(header[2])); assert stream.read(1) == b"\n"
    row["pinned_sha256"] = sha256(data).hexdigest()
    pin_rows["same_current_bytes" if row["pinned_sha256"] == row["current_sha256"] else "different_current_bytes"].append(row)
assert not stream.read()

# Catalogue is navigation only; inspect actual current source identity and raw refs.
catalogue_path = REVIEWS / "reader-evidence-catalogue-technical-fix07-v1.json"
catalogue = json.loads(catalogue_path.read_text())
overlays, overlay_receipts, unique_current_reports = {}, [], set()
for number in [5, 6, 7, 8]:
    path = REVIEWS / f"reader-late-callback-intake-v{number}.json"
    data = json.loads(path.read_text()); overlay_receipts.append(ref(path))
    for entry in data["entries"]:
        assert digest(ROOT / entry["path"]) == entry["sha256"]
        if entry["kind"] != "reader-current-v1":
            continue
        unique_current_reports.add(entry["path"])
        report = json.loads((ROOT / entry["path"]).read_text())
        for pid in entry.get("page_ids", [entry.get("page_id")]):
            page = next(p for p in report["pages"] if p["page_id"] == pid)
            assert page["source_sha256"] == by_id[pid]["source_sha256"] and page["figures_sha256"] == by_id[pid]["figures_sha256"]
            for item in [page["basis_report"], page["session_evidence"]["opening"], *page["session_evidence"]["notes"]]:
                assert digest(ROOT / item["path"]) == item["sha256"]
            overlays[pid] = {"current_report": {"path": entry["path"], "sha256": entry["sha256"]}, "source_sha256": page["source_sha256"], "session": page["session"], "raw_notes_hash_checked": len(page["session_evidence"]["notes"])}
retained, statuses = 0, Counter()
for row in catalogue["rows"]:
    pid = row["page_id"]
    if pid in ["first-steps", "12.2"]:
        statuses["current_full_reader_pending"] += 1
        continue
    if pid in overlays:
        continue
    statuses[row["status"]] += 1
    if row["status"] == "preserved_inherit_not_new_read":
        continue
    for item in [row["current_report"], row["basis"]["report"]]:
        assert digest(ROOT / item["path"]) == item["sha256"]
    page = pointer(json.loads((ROOT / row["current_report"]["path"]).read_text()), row["current_pointer"])
    if row["status"] == "current_owner_source_identity":
        assert page["source_sha256"] == by_id[pid]["source_sha256"]
        assert page.get("figures_sha256", page.get("figure_sha256")) == by_id[pid]["figures_sha256"]
        retained += 1
assert retained == 358 and len(overlays) == 5 and len(unique_current_reports) == 4

chains = []
for suffix, pid in [("06", "9.11"), ("09", "9.18"), ("11", "12.17"), ("12", "9.12"), ("13", "9.15"), ("14", "12.2")]:
    path = EVIDENCE / f"notebooks-technical-fix{suffix}-cumulative.json"
    data = json.loads(path.read_text()); record = next(r for r in data["records"] if r["lesson"] == pid)
    assert record["method"] == "new_independent_kernel"
    assert digest(Path(record["source"])) == record["executed_sha256"] == records[pid]["executed_sha256"]
    validation_path = ROOT / f"outputs/omni-notebooks-technical-fix{suffix}-run/validation-kernel.json"
    validation = json.loads(validation_path.read_text())
    assert validation["mode"] == "kernel" and validation["total"] == validation["passed"] == 1 and validation["failed"] == 0
    chains.append({"page_id": pid, "receipt": ref(path), "record": record, "validation": ref(validation_path), "validation_result": validation, "command": data["kernel_command"]})

prior_hashes = []
for name in ["technical-original-v1.json", "technical-current-v1.json", "technical-current-v2.json", "technical-current-v3.json"]:
    path = REVIEWS / "technical/technical_r_1" / name
    report = json.loads(path.read_text())
    for item in report["artifacts"]:
        assert digest(ROOT / item["path"]) == item["sha256"]
    prior_hashes.append(ref(path))
for source in json.loads((REVIEWS / "technical/technical_r_1/technical-current-v3.json").read_text())["sources"]:
    if source["kind"] == "repository_code":
        assert digest(ROOT / source["path"]) == source["sha256"]
result = {
    "created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/audit_course_presence_callback_v4.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; static and live HTTP identity inspection; no lesson execution or cloud"},
    "source_freeze": ref(FREEZE), "comparison_freeze": ref(OLD), "numbered_freeze": ref(NUMBERED),
    "changed_canonical_sources_since_v3": changed, "r1": r1,
    "cumulative_execution_receipt": ref(CUMULATIVE), "execution_methods": dict(Counter(r["method"] for r in records.values())),
    "candidate_site_root": str(SITE.relative_to(ROOT)), "candidate_base_url": BASE_URL, "candidate_build_info": build_info, "build_info_reference": ref(SITE / "build-info.json"),
    "counts": dict(counts), "canonical_source_identities_checked": len(by_id), "current_figure_byte_identities_checked": len(figures),
    "notebook_metadata_profiles": [{"nbformat": k[0], "minor": k[1], "kernel": k[2], "language_version": k[3], "count": v} for k, v in profiles.items()],
    "actual_HTTP_rows": http_rows,
    "preview_colab_pin": {"revision": pin, "active_github_pin_links": 0, "counts": {k: len(v) for k, v in pin_rows.items()}, **pin_rows, "scope": "The old pin is inert preview metadata, not a direct Colab route or current full release."},
    "reader_mapping": {"navigation_catalogue": ref(catalogue_path), "actual_intakes": overlay_receipts, "retained_exact_source_reader_rows": retained, "overlays": overlays, "unique_overlay_report_files": len(unique_current_reports), "current_source_identity_total": retained + len(overlays), "non_overlay_status_counts": dict(statuses), "pending_pages": ["first-steps", "12.2"], "scope": "Source/figure identity plus prior scoped note reading, not a semantic proof or a new whole-book blind/human reading.1.11 author patch and3 inherited references remain explicitly limited."},
    "estimates": {"preview": ref(PREVIEW), "live_metadata": ref(metadata_path), "preview_filter_receipt": ref(preview_receipt_path), "preview_source_and_figure_matching": len(preview), "live_rows": len(metadata), "live_source_and_figure_matching": len(metadata)-len(stale), "live_stale": stale},
    "specific_author_kernel_chains_inspected": chains, "prior_reports_and_artifacts_hash_checked": prior_hashes,
    "scope": "New source14 complete331 Notebook/body/output/download/HTTP/manual-footer identity audit and369 current canonical source identities. Source13 receipts remain historical. Current12.2 raw kernel and330 source-matching reuse are not claimed as this reviewer's execution. New semantic reading and own CPU run recorded separately. No release, commit, training, upload, or Colab cloud execution."
}
with Path(__file__).with_name("course-presence-callback-v4.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2); handle.write("\n")
print(json.dumps({"counts": dict(counts), "changed": changed, "pin_counts": result["preview_colab_pin"]["counts"], "reader_source_identities": retained+len(overlays), "AI_estimates": len(preview), "pending": stale}, ensure_ascii=False, indent=2))
