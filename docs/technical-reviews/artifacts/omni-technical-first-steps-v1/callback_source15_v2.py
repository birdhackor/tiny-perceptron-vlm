"""Same-owner first-steps source15 callback; read-only bounded delivery audit.

Unchanged science, CPU and actual figure inspection are reused by exact version.
Per-page comparisons stay outside Git; this receipt is not a reader/stage seal.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
REV = ROOT / "docs/course-revision-20261029-omni/reviews"
COORD = ROOT / "docs/technical-reviews/artifacts/omni-technical-coordinator-v1"
SITE = ROOT / "outputs/omni-review-site-technical-fix15"
RAW = ROOT / "outputs/omni-notebooks-technical-fix15-current"
PRIVATE = Path("/workspace/work/omni-technical-first-steps/source15-v2")
TASK = "/root/omni_technical_coordinator/technical_first_steps"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def ref(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_bytes())


def verify(value):
    path = ROOT / value["path"]
    assert sha(path) == value["sha256"], str(path)
    return path


def keep(name, value):
    target = PRIVATE / (name + ".json")
    with target.open("x") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return ref(target)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Page(HTMLParser):
    """Read exact visible Python and download actions from actual rendered HTML."""

    def __init__(self):
        super().__init__()
        self.divs = []
        self.capture = None
        self.codes = []
        self.actions = []
        self.ids = set()
        self.text = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "div":
            self.divs.append(set(attrs.get("class", "").split()))
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "code" and any("language-python" in classes for classes in self.divs):
            assert self.capture is None
            self.capture = []
        if tag == "a" and any("actions" in classes for classes in self.divs):
            self.actions.append((attrs.get("href"), "download" in attrs))

    def handle_endtag(self, tag):
        if tag == "code" and self.capture is not None:
            self.codes.append("".join(self.capture).rstrip("\n"))
            self.capture = None
        if tag == "div":
            assert self.divs
            self.divs.pop()

    def handle_data(self, value):
        self.text.append(value)
        if self.capture is not None:
            self.capture.append(value)


started = datetime.now(UTC).isoformat()
PRIVATE.mkdir(parents=True, exist_ok=False)
canonical_path = REV / "source-freeze-technical-fix15.json"
section_path = REV / "numbered-freeze-v13.json"
bundle_path = COORD / "numbered-section-reviews-v5.json"
issue_path = COORD / "numbered-section-issues-v5.json"
assert sha(canonical_path) == "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
assert sha(section_path) == "588e97fd8b6647d8cb4c69c7015416d508016402e40e88d4934a461faca0586f"
assert sha(bundle_path) == "0077db6122efdd4371d02334fa8aab5ab7002cd3312431b48fc57660bcb66d4f"
assert sha(issue_path) == "0c434bc585e15b69404eab334864cfca143d8744a1078a09ef4ac30340579c82"
canonical, numbered, bundle, issues = map(read, [canonical_path, section_path, bundle_path, issue_path])
oldcanon = read(REV / "source-freeze-technical-fix14.json")
oldsections = read(REV / "numbered-freeze-v12.json")
oldbundle = read(COORD / "numbered-section-reviews-v4.json")
own_paths = [REV / "technical/technical_first_steps" / name for name in ["technical-original-v1.json", "technical-current-v1.json", "technical-current-v2.json"]]
assert [sha(path) for path in own_paths] == [
    "b306b90afd771bc18937f8d5d370498e7d52185860dad745415c28539f519d73",
    "cb54d25acd61156497bff2c95d95b85ed4f0d58b9585d8908137d021b4e8a26a",
    "7434cfdee8c802ab425b2d9f3f1412d4224e289e58d53906c95d07cd8f2bbf60",
]
own = read(own_paths[-1])
for artifact in own["artifacts"]:
    verify(artifact)
assert len(own["claims"]) == 102 and len(own["original_issue_resolutions"]) == 2
pages = {page["page_id"]: page for page in canonical["pages"]}
previous_pages = {page["page_id"]: page for page in oldcanon["pages"]}
whole = pages["first-steps"]
assert sha(ROOT / whole["source"]) == sha(ROOT / whole["snapshot"]) == previous_pages["first-steps"]["source_sha256"] == own["source_sha256"]
assert whole["figures_sha256"] == own["figure_sha256"] == previous_pages["first-steps"]["figures_sha256"]
for path, expected in whole["figures_sha256"].items():
    assert sha(ROOT / path) == expected
text = (ROOT / whole["source"]).read_text()
assert '向下取整的除法；`-7 // 2` 會取成 `-4`' in text
assert '把這次的錯誤物件暫時叫作 `error`' in text
assert '| `tiny_perceptron/natural_concepts.py` | [11.14](chapters/11.md#11.14)–11.16、[12.14](chapters/12.md#12.14) |' in text
canonical_delta = []
for lesson, page in pages.items():
    changed = {key: {"previous": previous_pages[lesson].get(key), "current": page.get(key)} for key in ["source_sha256", "figures_sha256", "notebook_sha256"] if previous_pages[lesson].get(key) != page.get(key)}
    if changed:
        canonical_delta.append({"page_id": lesson, "changes": changed})
assert {row["page_id"] for row in canonical_delta} == {"5.7", "chapter-10", "chapter-14"}
oldsec = {page["page_id"]: page for page in oldsections["pages"]}
for page in numbered["pages"]:
    assert page["source_sha256"] == oldsec[page["page_id"]]["source_sha256"] == sha(ROOT / page["snapshot"])
    assert page["figures_sha256"] == oldsec[page["page_id"]]["figures_sha256"]
    for path, expected in page["figures_sha256"].items():
        assert sha(ROOT / path) == expected
repo_reuse = []
for source in own["sources"]:
    if source["kind"] == "repository_code":
        assert sha(ROOT / source["path"]) == source["sha256"]
        repo_reuse.append({"id": source["id"], **ref(ROOT / source["path"])})
primary_receipt = read(OUT / "primary-source-receipts-v1.json")
fixed_authorities = [ref(verify(row)) for row in primary_receipt["official_sources"] if row.get("ok")]
installed_sources = []
for source in own["sources"]:
    if source["id"].startswith("s_torch_"):
        note = source["inspection_note"]
        match = re.search(r"Installed source (.*?); whole-file SHA-256 ([a-f0-9]{64})", note)
        assert match, source["id"]
        path = Path(match[1]); assert sha(path) == match[2]
        installed_sources.append({"id": source["id"], **ref(path)})

# Check the full immutable original/current bundle as a collection, not an
# independent science signature for the other three canonical pages.
assert len(bundle["sections"]) == len({entry["lesson_id"] for entry in bundle["sections"]}) == 26
assert len(issues["reports"]) == 36 and len(issues["issues"]) == 5
assert bundle["canonical_source_freeze"] == ref(canonical_path) and bundle["section_freeze"] == ref(section_path)
selected_path = verify(bundle["actual_collection_snapshot"]); selected = read(selected_path)
raw_collection_path = verify(selected["raw_collection_snapshot"]); raw_collection = read(raw_collection_path)
assert len(selected["collection_index"]) == 48 and len(raw_collection["collection_index"]) == 236
assert all(row in raw_collection["collection_index"] for row in selected["collection_index"])
assert {row["lesson_id"] for row in selected["collection_index"]} == {entry["lesson_id"] for entry in bundle["sections"]}
assignment_path = REV / "technical-stage/assignments-numbered-intake-v6.json"; assignments = read(assignment_path)
reports = []; history_differences = []; artifact_count = 0; claim_count = 0
for pointer in issues["reports"]:
    path = verify(pointer); report = read(path)
    source_ids = {source["id"] for source in report["sources"]}
    artifact_ids = {artifact["id"] for artifact in report["artifacts"]}
    assert len(source_ids) == len(report["sources"]) and len(artifact_ids) == len(report["artifacts"])
    for artifact in report["artifacts"]:
        verify(artifact); artifact_count += 1
    for claim in report["claims"]:
        assert all(evidence["source_id"] in source_ids for evidence in claim["evidence"])
        assert all(identifier in artifact_ids for identifier in claim.get("artifact_ids", []))
        assert claim["statement"] and claim["scope"] and claim["location"]
    claim_count += len(report["claims"])
    for source in report["sources"]:
        if source["kind"] == "repository_code" and sha(ROOT / source["path"]) != source["sha256"]:
            history_differences.append({"lesson_id": report["lesson_id"], "report": pointer, "source": source["path"], "original_sha256": source["sha256"], "current_sha256": sha(ROOT / source["path"])})
    assert any(row["path"] == pointer["path"] and row["sha256"] == pointer["sha256"] and row["reviewer_task"] == report["reviewer_task"] for row in selected["collection_index"])
    reports.append({"report": pointer, "lesson_id": report["lesson_id"], "reviewer_task": report["reviewer_task"], "claims": len(report["claims"]), "sources": len(report["sources"]), "artifacts": len(report["artifacts"]), "verdict": report["verdict"], "source_sha256": report["source_sha256"], "figure_sha256": report["figure_sha256"]})
assert artifact_count == 303 and claim_count == 1525
assert len(history_differences) == 6
section_rows = []; changed_owners = []
oldentries = {entry["lesson_id"]: entry for entry in oldbundle["sections"]}
actors = set()
for entry in bundle["sections"]:
    original = read(verify(entry["original_report"])); current = read(verify(entry["report"]))
    original_freeze = read(verify(entry["original_section_freeze"]))
    original_page = next(page for page in original_freeze["pages"] if page["page_id"] == entry["lesson_id"])
    current_page = next(page for page in numbered["pages"] if page["page_id"] == entry["lesson_id"])
    assert original["source_sha256"] == original_page["source_sha256"] == sha(ROOT / original_page["snapshot"])
    assert original["figure_sha256"] == original_page["figures_sha256"]
    assert current["source_sha256"] == current_page["source_sha256"] and current["figure_sha256"] == current_page["figures_sha256"]
    assert original["reviewer_task"] == current["reviewer_task"] and current["verdict"] == "pass"
    actors.add(original["reviewer_task"])
    assert any(task.get("task") == original["reviewer_task"] and task.get("lesson_id") == entry["lesson_id"] and task.get("role") == "technical" for task in assignments["tasks"])
    for label, report in [("original", original), ("current", current)]:
        receipt = entry[label + "_collection"]
        assert any(row["path"] == entry["original_report" if label == "original" else "report"]["path"] and all(row.get(key) == value for key, value in receipt.items()) for row in selected["collection_index"])
    for snapshot in entry.get("original_source_snapshots", []):
        verify(snapshot["snapshot"])
    if entry["report"] != oldentries[entry["lesson_id"]]["report"]:
        changed_owners.append({"lesson_id": entry["lesson_id"], "current": entry["report"], "previous": oldentries[entry["lesson_id"]]["report"]})
    section_rows.append({"lesson_id": entry["lesson_id"], "actor": original["reviewer_task"], "original": entry["original_report"], "current": entry["report"], "original_claims": len(original["claims"]), "current_claims": len(current["claims"]), "source_figure_and_actual_receipt_binding": True})
assert len(actors) == 26 and {row["lesson_id"] for row in changed_owners} == {"R.1", "T.2", "W.1"}
issue_rows = []
for entry in issues["issues"]:
    report = read(verify(entry["report"])); issue = report["issues"][int(entry["json_pointer"].rsplit("/", 1)[1])]
    assert issue["source_quote"] and issue.get("necessity") in ["necessary", "required", "optional"]
    issue_rows.append({"lesson_id": entry["lesson_id"], "report": entry["report"], "json_pointer": entry["json_pointer"], "id": issue["id"], "necessity": issue["necessity"], "status_in_referenced_report": issue["status"], "source_quote": issue["source_quote"]})
w1entry = next(entry for entry in bundle["sections"] if entry["lesson_id"] == "W.1")
w1 = read(ROOT / w1entry["report"]["path"])
w1discovery = read(REV / "technical/technical_w_1/technical-current-v1.json")
assert read(REV / "technical/technical_w_1/technical-original-v1.json")["issues"] == []
assert w1["issues"][0]["id"] == "w1-current-dependency-drift" and w1["issues"][0]["status"] == "resolved"
for key in ["source_quote", "details", "location"]:
    assert w1["issues"][0][key] == w1discovery["issues"][0][key]
r1 = read(ROOT / next(entry["report"]["path"] for entry in bundle["sections"] if entry["lesson_id"] == "R.1"))
r1discovery = read(REV / "technical/technical_r_1/technical-current-v2.json")
assert r1["issues"][0]["id"] == "R1-COLAB-PIN-01" and r1["issues"][0]["status"] == "resolved"
for key in ["source_quote", "details", "location"]:
    assert r1["issues"][0][key] == r1discovery["issues"][0][key]
historical_export = ROOT / "docs/technical-reviews/artifacts/omni-author-colab-preview-fix01/export-before.py"
assert sha(historical_export) == "792f6b269d0c1afd34619d2ea10e1db52a6c19a67ce3365539a612f6ef625874"
for difference in history_differences:
    if difference["source"] == "scripts/export_course.py":
        assert difference["original_sha256"] == sha(historical_export)
        difference["authorized_original_snapshot"] = ref(historical_export)
        difference["scope"] = "Original authentication only; current exporter is separately bound and checked at662355."
    else:
        lesson = difference["lesson_id"]
        entry = next(entry for entry in bundle["sections"] if entry["lesson_id"] == lesson)
        frozen = read(verify(entry["original_section_freeze"]))
        original_canonical = read(verify(frozen["canonical_source_freeze"]))
        chapter_pages = [page for page in original_canonical["pages"] if page["source"] == difference["source"]]
        intro = next(page for page in chapter_pages if page["selector"] == "before-first-section")
        ordered = [intro] + [page for page in chapter_pages if page["selector"].startswith("section:")]
        reconstructed = b"".join((ROOT / page["snapshot"]).read_bytes() for page in ordered)
        for page in ordered:
            assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
        assert hashlib.sha256(reconstructed).hexdigest() == difference["original_sha256"]
        difference["historical_canonical"] = frozen["canonical_source_freeze"]
        difference["scope"] = "Original whole chapter reconstructed from its original freeze; effective current report has exact current sources and retains its bounded claims."

# Inspect raw successful current W1 proof and its failed observations without
# replacing this collector's independent current comparisons with that pass.
w1proof_path = ROOT / "docs/technical-reviews/artifacts/omni-technical-w-1-v1/fix15-current-recheck-v2.json"
w1proof = read(w1proof_path)
verify(w1proof["review_driver_scope_observation"]["failure"])
oldbad_path = REV / "technical/technical_w_1/technical-current-v4.json"
oldcorrected_path = REV / "technical/technical_w_1/technical-current-v4-schema-corrected-v1.json"
bad, corrected = read(oldbad_path), read(oldcorrected_path)
for key in ["claims", "sources", "issues", "checks", "review_scope"]:
    assert bad[key] == corrected[key]
schema_fail_path = ROOT / "docs/technical-reviews/artifacts/omni-technical-w-1-v1/schema-current-v4.json"
assert read(schema_fail_path)["exit_code"] == 1
old_w1proof = read(ROOT / "docs/technical-reviews/artifacts/omni-technical-w-1-v1/fix14-current-recheck-v3.json")
for value in old_w1proof["new_test_failure_and_diagnosis"].values():
    if isinstance(value, dict) and "path" in value and "sha256" in value:
        verify(value)

sys.path.insert(0, str(ROOT))
from scripts import build_course, export_course
from scripts.reading_time import build_inventory, load_estimates

index = read(ROOT / "course/lesson-index.json")
inventory = build_inventory(ROOT, index, export_course.DOCUMENTS, export_course.home_introduction(index, True))
assert {page["page_id"] for page in inventory["pages"]} == set(pages)
for page in inventory["pages"]:
    assert page["source_sha256"] == pages[page["page_id"]]["source_sha256"]
    assert page["figures_sha256"] == pages[page["page_id"]]["figures_sha256"]
dom_path = ROOT / "docs/technical-reviews/artifacts/omni-technical-w-1-v1/output_dom_v1.py"
assert sha(dom_path) == "55e67c3e848cb5f7aad28da80915cfb16f87ac5bec1271eb64a46691323c4597"
dom = load("collector_source15_output_dom", dom_path)


def joined(value):
    return "".join(value) if isinstance(value, list) else value


def expected_blocks(cells):
    blocks = []
    for cell in cells:
        parts = []
        for output in cell.get("outputs", []):
            kind = output["output_type"]
            assert kind != "error"
            if kind == "stream":
                parts.append(["pre", joined(output["text"])])
            elif kind in ["display_data", "execute_result"]:
                data = output["data"]
                if "image/png" in data:
                    parts.append(["image", "data:image/png;base64," + joined(data["image/png"]), "本段程式的實際輸出圖"])
                elif "text/plain" in data:
                    parts.append(["pre", joined(data["text/plain"])])
        if parts:
            blocks.append([["label", "實際執行結果 · CPU"], *parts])
    return blocks


author_path = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix15-cumulative.json"
author = read(author_path)
assert author["source_freeze"] == ref(canonical_path)
verify(author["script"])
assert author["reexecuted_this_fix"] == ["5.7"] and author["reused_matching_cells"] == 330
author_rows = {row["lesson"]: row for row in author["records"]}
assert len(index) == len(author_rows) == 331
rows = []; wrapper_only = []; code_count = 0; displayed = 0; executed_count = 0
for item in index:
    lesson = item["id"]; notebook_path = ROOT / item["notebook"]
    raw_path = RAW / Path(item["notebook"]).relative_to("notebooks")
    author_row = author_rows[lesson]
    assert sha(notebook_path) == pages[lesson]["notebook_sha256"] == author_row["notebook_sha256"]
    assert author_row["method"] == ("new_independent_kernel" if lesson == "5.7" else "reuse_matching_cells")
    expected_origin = ROOT / "outputs" / ("omni-notebooks-technical-fix15-run" if lesson == "5.7" else "omni-notebooks-technical-fix14-current") / Path(item["notebook"]).relative_to("notebooks")
    assert Path(author_row["source"]).resolve() == expected_origin.resolve()
    assert Path(author_row["bundle_path"]).resolve() == raw_path.resolve()
    assert sha(raw_path) == author_row["executed_sha256"] == sha(expected_origin)
    notebook = read(notebook_path); executed = export_course.checked_notebook(notebook, raw_path)
    parts = re.split(r"^## ([\dABC]+\.\d+) (.+)$", (ROOT / item["source"]).read_text(), flags=re.M)
    position = next(i for i in range(1, len(parts), 3) if parts[i] == lesson)
    generated = build_course.notebook(lesson, parts[position + 1], parts[position + 2], ROOT / item["source"])
    assert json.dumps(generated, ensure_ascii=False, indent=1) + "\n" == notebook_path.read_text()
    teaching = [cell for cell in executed["cells"] if cell["cell_type"] == "code" and not cell.get("metadata", {}).get("course_setup") and "course_figure" not in cell.get("metadata", {})]
    html_path = SITE / (lesson + ".html"); body = html_path.read_text()
    parser = Page(); parser.feed(body); parser.close()
    assert parser.capture is None and parser.divs == []
    assert parser.codes == [joined(cell["source"]).rstrip("\n") for cell in teaching], (lesson, "visible code")
    expected = expected_blocks(teaching)
    assert expected == dom.outputs(body), (lesson, "exact label/pre/image/order")
    assert set(parser.actions) == {(item["notebook"], True), ("https://colab.research.google.com/", False)}
    assert "colab-upload-instructions" in parser.ids
    assert all(value in " ".join(parser.text) for value in ["尚未發布的預覽版", "先按「下載本節 .ipynb」", "Upload notebook", "File → Upload notebook", "雲端環境未經本課實測"])
    assert notebook_path.read_bytes() == (SITE / item["notebook"]).read_bytes()
    serialized = [export_course.output_html(cell.get("outputs", [])) for cell in teaching]
    if any(value and value not in body for value in serialized):
        wrapper_only.append({"lesson_id": lesson, "exact_DOM_content_equal": True, "DOM_sha256": digest(expected)})
    code_count += len(teaching); displayed += len(expected)
    executed_count += sum(cell["cell_type"] == "code" for cell in executed["cells"])
    rows.append({"lesson_id": lesson, "authored_sha256": pages[lesson]["source_sha256"], "notebook": item["notebook"], "notebook_sha256": sha(notebook_path), "raw_sha256": sha(raw_path), "HTML_sha256": sha(html_path), "visible_code_sha256": digest(parser.codes), "output_DOM_sha256": digest(expected), "teaching_cells": len(teaching), "displayed_cells": len(expected), "method": author_row["method"]})
assert code_count == 423 and displayed == 363 and executed_count == 871
assert sum(row["teaching_cells"] > 0 for row in rows) == 319
assert {row["lesson_id"] for row in wrapper_only} == {"15.8", "19.6"}


def actual_http(row):
    value = {"lesson_id": row["lesson_id"]}
    for kind, path in [("HTML", row["lesson_id"] + ".html"), ("notebook", row["notebook"])]:
        with urlopen("http://127.0.0.1:8776/" + path, timeout=15) as response:
            data = response.read(); assert response.status == 200
        assert data == (SITE / path).read_bytes()
        if kind == "notebook":
            assert data == (ROOT / path).read_bytes()
        value[kind + "_sha256"] = hashlib.sha256(data).hexdigest()
    return value


with ThreadPoolExecutor(max_workers=6) as pool:
    http_rows = list(pool.map(actual_http, rows))
whole_html = SITE / "first-steps.html"
with urlopen("http://127.0.0.1:8776/first-steps.html", timeout=15) as response:
    assert response.status == 200 and response.read() == whole_html.read_bytes()
whole_parser = Page(); whole_parser.feed(whole_html.read_text()); whole_parser.close()
assert whole_parser.codes == [code.rstrip("\n") for code in re.findall(r"```python\n(.*?)```", text, flags=re.S)]
assert {"W." + str(i) for i in range(1, 8)}.issubset(whole_parser.ids)
prior_routes = read(OUT / "callback-audit-v1.json")["routes"]
assert len(prior_routes) == 65
changed_route_targets = {row["lesson_id"] for row in prior_routes}.intersection(row["page_id"] for row in canonical_delta)
assert changed_route_targets == {"5.7"}
route57 = (ROOT / pages["5.7"]["snapshot"]).read_text()
assert 'tiny_perceptron/training.py' in route57 and 'save_checkpoint' in route57 and 'load_checkpoint' in route57
assert '降低載入時執行任意程式的風險' in route57
old57 = read(ROOT / "outputs/omni-notebooks-technical-fix14-current/05/5.7.ipynb")
new57 = read(ROOT / "notebooks/05/5.7.ipynb")
changed_cells = [i for i, (left, right) in enumerate(zip(old57["cells"], new57["cells"], strict=True)) if left["cell_type"] != right["cell_type"] or left["source"] != right["source"]]
assert changed_cells == [4] and new57["cells"][4]["cell_type"] == "markdown"
kernel_path = ROOT / "outputs/omni-notebooks-technical-fix15-run/validation-kernel.json"
kernel = read(kernel_path); assert kernel["total"] == kernel["passed"] == 1 and kernel["failed"] == 0 and kernel["failures"] == []
new_raw57 = ROOT / "outputs/omni-notebooks-technical-fix15-run/05/5.7.ipynb"
assert sha(new_raw57) == sha(RAW / "05/5.7.ipynb")
stdout57 = "".join(joined(output["text"]) for cell in read(new_raw57)["cells"] for output in cell.get("outputs", []) if output["output_type"] == "stream" and output.get("name") == "stdout")
assert "保存步數 1" in stdout57 and "trainable_parameters" in stdout57
lineage = []
for fix, lesson in [("06", "9.11"), ("09", "9.18"), ("11", "12.17"), ("12", "9.12"), ("12", "9.15"), ("14", "12.2"), ("15", "5.7")]:
    item = next(item for item in index if item["id"] == lesson)
    original_raw = ROOT / "outputs" / ("omni-notebooks-technical-fix" + fix + "-run") / Path(item["notebook"]).relative_to("notebooks")
    assert sha(original_raw) == sha(RAW / Path(item["notebook"]).relative_to("notebooks"))
    lineage.append({"lesson_id": lesson, "original_raw": ref(original_raw), "current_raw_same_bytes": True})

base_times_path = ROOT / "course/reading-times.json"; base_times = read(base_times_path)
preview_times_path = ROOT / "outputs/omni-reading-times-technical-fix15-preview.json"; preview_times = read(preview_times_path)
base_rows = {row["page_id"]: row for row in base_times["pages"]}; preview_rows = {row["page_id"]: row for row in preview_times["pages"]}
stale = sorted(lesson for lesson, row in base_rows.items() if row["source_sha256"] != pages[lesson]["source_sha256"] or row["figures_sha256"] != pages[lesson]["figures_sha256"])
assert stale == ["12.2", "5.7", "chapter-10", "chapter-14", "first-steps"]
assert set(base_rows) - set(preview_rows) == set(stale) and len(preview_rows) == 364
assert all(value == base_rows[lesson] for lesson, value in preview_rows.items())
assert all(preview_times.get(key) == value for key, value in base_times.items() if key != "pages")
assert set(preview_times) - set(base_times) == {"preview_scope"}
assert preview_times["preview_scope"] == {"source_freeze": ref(canonical_path), "kind": "ignored_nonformal_review_preview_only", "omitted_stale_or_missing": stale, "new_estimates_fabricated": False, "formal_publication_ready": False}
assert len(load_estimates(preview_times_path, inventory, False)) == 364
guards = []
for lesson in stale:
    path = PRIVATE / ("stale-" + lesson + ".json"); path.write_text(json.dumps({**base_times, "pages": [base_rows[lesson]]}))
    try:
        load_estimates(path, inventory, False)
        raise AssertionError("stale estimate accepted")
    except ValueError as exc:
        guards.append({"case": "reject_stale", "page_id": lesson, "message": str(exc)})
try:
    load_estimates(preview_times_path, inventory, True)
    raise AssertionError("incomplete formal metadata accepted")
except ValueError as exc:
    guards.append({"case": "reject_missing5_formal", "message": str(exc)})
rendered_times = [path.stem for path in SITE.glob("*.html") if '<p class="reading-time">' in path.read_text()]
assert len(rendered_times) == 364 and not set(stale).intersection(rendered_times)
info_path = SITE / "build-info.json"; info = read(info_path)
assert info["colab_mode"] == "upload" and info["publication_ready"] is False and info["colab_git_objects_verified"] is False and info["reading_time_pages"] == 364
site_receipt_path = ROOT / "docs/course-revision-20261029-omni/evidence/preview-site-technical-fix15-v1.json"
site_receipt = read(site_receipt_path)
assert site_receipt["source_freeze"] == ref(canonical_path) and site_receipt["notebook_execution_receipt"] == ref(author_path)
assert site_receipt["build_info"] == info and site_receipt["build_info_sha256"] == sha(info_path)

# Repeat own comparator controls; expected output above was independently
# derived from raw fields, rather than comparing the exporter with itself.
sample = '<div class="output"><p class="output-label">實際執行結果 · CPU</p><pre>0.64755\n</pre><pre>second\n</pre><img src="data:image/png;base64,abc" alt="chart"></div>'
baseline = dom.outputs(sample)
control_inputs = [
    ("wrapper_newlines_only", sample.replace("><", ">\n<"), True),
    ("changed_number", sample.replace("0.64755", "0.64756"), False),
    ("missing_pre", sample.replace('<pre>second\n</pre>', ''), False),
    ("changed_pre_whitespace", sample.replace('0.64755\n', '0.64755 \n'), False),
    ("changed_image_src", sample.replace('base64,abc', 'base64,abcd'), False),
    ("changed_image_alt", sample.replace('alt="chart"', 'alt="other"'), False),
    ("changed_block_order", sample.replace('<pre>0.64755\n</pre><pre>second\n</pre>', '<pre>second\n</pre><pre>0.64755\n</pre>'), False),
    ("missing_label", sample.replace('<p class="output-label">實際執行結果 · CPU</p>', ''), False),
    ("unexpected_visible_text", sample.replace('</div>', 'unexpected</div>'), False),
]
controls = []
for name, value, expected in control_inputs:
    try:
        observed = dom.outputs(value) == baseline; message = ""
    except AssertionError as exc:
        observed = False; message = str(exc)
    assert observed == expected
    controls.append({"case": name, "expected_equal": expected, "observed_equal": observed, "rejection": message})

commands = []


def run(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=50)
    value = {"command": args, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    commands.append(value)
    return value


intake_output = OUT / "numbered-intake-source15-v1.json"
run(["uv", "run", "--extra", "cpu", "python", "scripts/omni_revision/review_setup.py", "check-numbered", "--manifest", str((REV / "technical-stage/numbered-intake-manifest-v6.json").relative_to(ROOT)), "--intake", "--output", str(intake_output.relative_to(ROOT))])
run(["uv", "run", "--extra", "cpu", "python", "scripts/build_course.py", "--check"])
run(["uv", "run", "--extra", "cpu", "python", "scripts/check_site.py", "--site", str(SITE.relative_to(ROOT))])
run(["uv", "run", "--extra", "cpu", "python", "docs/course-review-20261019/tools/check_technical_report.py", "W.1", w1entry["report"]["path"], "--freeze", str(section_path.relative_to(ROOT)), "--reader-task", "/root/rv1019_reader_entry"])
audit = {
    "started_at": started, "created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK,
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-first-steps-v1/callback_source15_v2.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "platform": platform.platform()},
    "canonical_source_freeze": ref(canonical_path), "section_freeze": ref(section_path), "bundle": ref(bundle_path), "issue_index": ref(issue_path),
    "preserved_own_reports": [ref(path) for path in own_paths], "unchanged_whole_source": whole["source_sha256"], "unchanged_figures": whole["figures_sha256"],
    "canonical_delta": canonical_delta, "unchanged_26section_source_figures": True,
    "reuse": {"whole_claims": 102, "source13_visual_and_CPU_report": ref(own_paths[1]), "source14_bounded_report": ref(own_paths[2]), "official_cached_originals_rehashed": len(fixed_authorities), "installed_originals_rehashed": installed_sources, "unchanged_direct_repository_sources": repo_reuse,
      "science_scope": "Original derivations/official originals, actual8warmupCPU blocks and two actual render/view_image checks are explicitly reused by unchanged bytes and interpreter/PyTorch version; not newly performed science or image viewing."},
    "whole_HTML_HTTP": {**ref(whole_html), "status": 200, "all8_python_blocks_equal_current_authored_source": True, "all7_W_anchors": True},
    "route_continuation": {"prior65route_proof": ref(OUT / "callback-audit-v1.json"), "unchanged_target_count": 64, "changed_target": "5.7", "current57": ref(ROOT / pages["5.7"]["snapshot"]), "assessment": "Complete current5.7 read as W.7 route prerequisite: still explicitly explains training.py/save_checkpoint/load_checkpoint. New wording lowers loading risk instead of guaranteeing arbitrary-code absence. No fresh whole5.7 scientific/GPU review or training claimed."},
    "bundle_audit": {"distinct_actors": len(actors), "reports": 36, "claim_records": claim_count, "artifact_references_verified": artifact_count, "sections": section_rows, "changed_effective_reports": changed_owners, "historical_repository_source_differences": history_differences, "five_issue_entries": issue_rows,
      "complete_claim_inventory": keep("36report-claim-inventory", [{"report": pointer, "claims": read(ROOT / pointer["path"])["claims"]} for pointer in issues["reports"]]),
      "scope": "All original/current claim inventories, issues, source/figure bindings, evidence identifiers,303actualartifact hashes and actual collection rows inspected. Foreign R/T/G claims remain those owners' scoped evidence; this collector does not add an independent science signature for those pages."},
    "collection": {"selected": ref(selected_path), "selected_rows": 48, "section_scope": 26, "verbatim_subset_of_raw": True, "raw_ledger": ref(raw_collection_path), "raw_rows": 236, "assignments": ref(assignment_path), "status": "schema_checked_collection_pending", "scope": "Numbered26 only; raw ledger separately pinned. Not canonical369/fullstage collection or seal."},
    "W1_retained_history": {"first_discovery": ref(REV / "technical/technical_w_1/technical-current-v1.json"), "original_issues_empty": True, "discovery_quote_details_location_preserved": True, "effective_current": w1entry["report"], "current_raw_proof": ref(w1proof_path), "invalid_current4": ref(oldbad_path), "schema_failure": ref(schema_fail_path), "effective_current4": ref(oldcorrected_path), "substantive_format_correction_unchanged": True, "fix14_failed_observations": old_w1proof["new_test_failure_and_diagnosis"], "fix15_scope_observation": w1proof["review_driver_scope_observation"]},
    "R1_retained_history": {"first_discovery": ref(REV / "technical/technical_r_1/technical-current-v2.json"), "discovery_quote_details_location_preserved": True, "effective_current": next(entry["report"] for entry in bundle["sections"] if entry["lesson_id"] == "R.1"), "status": "resolved_for_current_manual_upload_preview_not_formal_release"},
    "author_notebook_receipt": ref(author_path), "current_delivery": {"notebooks": 331, "teaching_sections": 319, "teaching_cells": code_count, "displayed_cells": displayed, "executed_code_cells": executed_count, "source_code_output_download_rows": keep("331-source-code-output-download", rows), "HTTP331_HTML_and_downloads": keep("331-HTTP-HTML-and-downloads", http_rows), "serialized_wrapper_only_differences": wrapper_only, "output_DOM_parser": ref(dom_path), "own_controls": controls},
    "new57_kernel": {"validation": ref(kernel_path), "result": kernel, "raw": ref(new_raw57), "same_bytes_as_current": True, "changed_notebook_cells": changed_cells, "only_markdown_changed": True, "stdout": stdout57}, "seven_author_kernel_lineages": lineage,
    "metadata": {"original": ref(base_times_path), "preview": ref(preview_times_path), "retained_rows": 364, "all_retained_rows_exact": True, "omitted_stale": stale, "preview_scope": preview_times["preview_scope"], "guards": guards, "rendered_current_estimates": len(rendered_times)},
    "actual_preview": {"site": str(SITE.relative_to(ROOT)), "base_url": "http://127.0.0.1:8776/", "build_info": ref(info_path), "info": info, "author_receipt": ref(site_receipt_path)}, "commands": commands,
    "own_preserved_probe_failure": {"driver": ref(OUT / "callback_source15_v1.py"), "actual_execution": ref(OUT / "callback-source15-execution-v1.json"), "exit_code": 1, "diagnosis": "Collector extra metadata assumption accepted only necessity=necessary/optional; genuine W1 later-discovery issue uses required, an explicit necessary classification. Original issue and all report bytes remain unchanged. This v2 accepts the existing required synonym verbatim and repeats the complete audit; no product/checker relaxation or source mismatch is hidden."},
    "limits": ["Informed bounded source15 callback after original/current1/current2; unchanged science and true prior figures/CPU explicitly reused; not a new blind read.", "331 current source/raw/visiblecode/outputDOM/HTML/HTTPdownloads independently checked;1new5.7author kernel+330matchingreuse, not331newkernels or cloud execution.", "Whole firststeps and12.2/5.7/ch10/ch14 properreaderwave02 pending; paused0–10aux is not pass.364exactoldestimates do not authenticate readers or supply missingminutes.", "48selectedactualrows belong to26scope, raw236separately preserved; pending collection is not whole approval,fullstage collection or seal.", "Original792f is original authentication only; current662355 and084027 contracts read/reused exactbytes. No native210tool audit delegated or claimed.", "No source/checker/model/priorreport edits, newexport/build, training, GPU, paid operation, commit, Colab runtime or publication."],
}
target = OUT / "callback-source15-v2.json"
with target.open("x") as file:
    json.dump(audit, file, ensure_ascii=False, indent=2); file.write("\n")
print(json.dumps({"artifact": ref(target), "counts": {"reports": 36, "claim_records": claim_count, "artifact_refs": artifact_count, "notebooks": 331, "HTTP_requests": 663, "estimates": 364, "selected_receipts": 48}, "commands": [{"exit_code": row["exit_code"], "stdout": row["stdout"]} for row in commands]}, ensure_ascii=False))
