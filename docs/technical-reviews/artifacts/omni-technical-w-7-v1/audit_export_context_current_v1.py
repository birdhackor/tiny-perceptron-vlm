"""Read-only W.7 callback evidence for current exporter, local preview and bytes."""

import ast
import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SITE = ROOT / "outputs/omni-review-site-technical-fix13"
BASE = "http://127.0.0.1:8774/"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path.read_bytes())}


def section(path, identifier):
    text = (ROOT / path).read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## (\S+) .+$", text, re.M))
    start = next(h for h in headings if h[1] == identifier)
    end = next((h.start() for h in headings if h.start() > start.start()), len(text))
    return text[start.start():end], text[:start.start()].count("\n") + 1


class Article(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.text = []
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "article":
            self.active = True
        if self.active:
            if tag == "a":
                self.links.append(attrs)
            if attrs.get("id"):
                self.ids.append(attrs["id"])

    def handle_endtag(self, tag):
        if tag == "article":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.text.append(data)


def http(path):
    with urlopen(BASE + path, timeout=15) as response:
        raw = response.read()
        return raw, {"url": BASE + path, "status": response.status, "bytes": len(raw), "sha256": digest(raw)}


def exception_result(function):
    try:
        function()
    except Exception as error:
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "accepted", "message": "returned normally"}


def main():
    started = datetime.datetime.now(datetime.UTC).isoformat()
    sys.path.insert(0, str(ROOT))
    from scripts import export_course

    index_path = ROOT / "course/lesson-index.json"
    index = json.loads(index_path.read_text())
    freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
    numbered_path = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v11.json"
    freeze = json.loads(freeze_path.read_text())
    numbered = json.loads(numbered_path.read_text())
    assert numbered["canonical_source_freeze"] == ref(freeze_path)
    pages = {p["page_id"]: p for p in freeze["pages"]}
    assert len(index) == 331
    build_raw, build_http = http("build-info.json")
    build_info = json.loads(build_raw)
    assert build_raw == (SITE / "build-info.json").read_bytes()
    assert build_info["lessons"] == len(index)
    assert build_info["colab_mode"] == "upload"
    assert build_info["publication_ready"] is False
    assert build_info["colab_git_objects_verified"] is False
    bundles = []
    for item in index:
        path = item["notebook"]
        raw = (ROOT / path).read_bytes()
        rendered = (SITE / path).read_bytes()
        downloaded, served = http(path)
        expected = pages[item["id"]]["notebook_sha256"]
        assert pages[item["id"]]["notebook"] == path
        assert digest(raw) == expected and raw == rendered == downloaded, path
        metadata = json.loads(raw)["metadata"]
        assert metadata["lesson_id"] == item["id"], path
        bundles.append({"lesson_id": item["id"], "notebook": path, "sha256": expected,
                        "bytes": len(raw), "frozen_equals_working_equals_site_equals_http": True,
                        "http_status": served["status"]})

    first_raw, first_http = http("first-steps.html")
    first_article = Article()
    first_article.feed(first_raw.decode())
    first_text = " ".join(" ".join(first_article.text).split())
    w7 = first_text[first_text.index("W.7 檔案放哪裡"):]
    assert "pyproject.toml" in w7 and "dry-run-no-weight-update" in w7
    assert "W.7" in first_article.ids
    routing = json.loads((HERE / "route-inspection-current-v1.json").read_text())
    route_links = []
    for row in routing["rows"]:
        for target in row["targets"]:
            page = SITE / (target["lesson_id"] + ".html")
            assert page.is_file()
            route_links.append({"lesson_id": target["lesson_id"], "site_path": page.relative_to(ROOT).as_posix(),
                                "exists": True})
    # Direct links start ranges; subsequent section IDs in a range are prose,
    # so do not claim each expanded route is a separate clickable link.
    direct = re.findall(r"\[[^]]+\]\(chapters/[^#]+#([^)]*)\)", section("course/first-steps.md", "W.7")[0])
    hrefs = {link.get("href") for link in first_article.links}
    for identifier in set(direct):
        assert identifier + ".html" in hrefs, identifier

    item = next(i for i in index if i["id"] == "1.1")
    notebook = json.loads((ROOT / item["notebook"]).read_text())
    preview_footer = export_course.section_footer(item, notebook, "main", True, direct_colab=False)
    empty_footer = export_course.section_footer(item, {"cells": []}, "main", False, direct_colab=False)
    for footer in [preview_footer, empty_footer]:
        assert 'href="https://colab.research.google.com/"' in footer
        assert "尚未發布的預覽版" in footer
        assert "首頁按 **Upload notebook**" in footer
        assert "**File → Upload notebook**" in footer
        assert "雲端環境未經本課實測" in footer
        assert "colab.research.google.com/github/" not in footer
    page_raw, page_http = http("1.1.html")
    article = Article()
    article.feed(page_raw.decode())
    page_text = " ".join(" ".join(article.text).split())
    action_offset = page_text.index("下載本節 .ipynb")
    shown_footer = page_text[action_offset:]
    assert "尚未發布的預覽版" in shown_footer
    assert "首頁按 Upload notebook" in shown_footer
    assert "File → Upload notebook" in shown_footer
    assert any(link.get("href") == item["notebook"] and "download" in link for link in article.links)
    assert any(link.get("href") == "https://colab.research.google.com/" for link in article.links)
    assert notebook["metadata"]["kernelspec"] == {"display_name": "Tiny Perceptron", "language": "python", "name": "tiny-perceptron"}
    setup = next(cell for cell in notebook["cells"] if cell.get("metadata", {}).get("course_setup"))
    setup_source = "".join(setup["source"])
    assert "root = Path.cwd().resolve()" in setup_source
    assert 'while not (root / "pyproject.toml").exists()' in setup_source
    assert "sys.path.insert(0, str(root))" in setup_source

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    guard_results = {
        "reject_symbolic_main": exception_result(lambda: export_course.verify_colab_notebooks(index, "main")),
        "reject_nonexistent_40hex": exception_result(lambda: export_course.verify_colab_notebooks(index, "0" * 40)),
        "reject_current_head_when_any_of_331_notebooks_differs": exception_result(lambda: export_course.verify_colab_notebooks(index, head)),
        "reject_site_destination_outside_outputs_before_writes": exception_result(lambda: export_course.build(ROOT / "__w7_invalid_site_destination__")),
    }
    assert guard_results["reject_symbolic_main"]["type"] == "ValueError"
    assert guard_results["reject_nonexistent_40hex"]["type"] == "ValueError"
    assert guard_results["reject_current_head_when_any_of_331_notebooks_differs"]["type"] == "ValueError"
    assert guard_results["reject_site_destination_outside_outputs_before_writes"]["type"] == "ValueError"
    old_bytes = subprocess.check_output(["git", "show", head + ":" + item["notebook"]], cwd=ROOT)
    assert old_bytes == (ROOT / item["notebook"]).read_bytes()
    guard_results["accept_real_commit_and_matching_one_notebook_subset"] = exception_result(lambda: export_course.verify_colab_notebooks([item], head))
    assert guard_results["accept_real_commit_and_matching_one_notebook_subset"]["type"] == "accepted"

    nine, line = section("course/chapters/09.md", "9.12")
    blocks = re.findall(r"```python\n(.*?)```", nine, re.S)
    trees = [ast.parse(code) for code in blocks]
    imports = [ast.unparse(n) for tree in trees for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == ["import torch"]
    assert "tiny_perceptron" not in nine
    assert "top = logits.max()" in blocks[0] and "chosen = min(tied)" in blocks[0]

    exporter_text = (ROOT / "scripts/export_course.py").read_text()
    lines = exporter_text.splitlines()
    contexts = []
    for first, last in [(192, 232), (234, 287), (317, 341), (394, 402), (463, 491)]:
        contexts.append({"path": "scripts/export_course.py", "start_line": first, "end_line": last,
                         "text": "\n".join(lines[first - 1:last])})
    slices = []
    for path, first, last in [("course/first-steps.md", 23, 28), ("course/first-steps.md", 31, 47),
                              ("course/first-steps.md", 59, 61), ("course/chapters/05.md", 257, 259),
                              ("course/chapters/05.md", 294, 307), ("course/training.md", 41, 61)]:
        text = "\n".join((ROOT / path).read_text().splitlines()[first - 1:last]) + "\n"
        slices.append({"path": path, "start_line": first, "end_line": last, "text": text,
                       "slice_sha256": digest(text.encode())})
    result = {"started_at": started, "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
              "reviewer_task": "/root/omni_technical_coordinator/technical_w_7",
              "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-7-v1/audit_export_context_current_v1.py",
              "environment": {"python": sys.version, "device": "cpu", "platform": platform.platform(),
                              "git_head": head, "local_preview": BASE},
              "current_exporter": ref(ROOT / "scripts/export_course.py"),
              "canonical_source_freeze": ref(freeze_path), "section_freeze": ref(numbered_path),
              "lesson_index": ref(index_path), "build_info_http": build_http, "build_info": build_info,
              "bundle": {"indexed_notebooks": len(index), "matched_frozen_working_site_http": len(bundles), "files": bundles},
              "first_steps_http": first_http, "w7_rendered_text": w7,
              "w7_route_pages": route_links, "w7_direct_route_links_checked": sorted(set(direct)),
              "example_page_http": page_http, "example_footer_text": shown_footer,
              "generated_preview_footer_with_code": preview_footer,
              "generated_preview_footer_without_code": empty_footer,
              "example_notebook_metadata": notebook["metadata"],
              "example_setup_source_sha256": digest(setup_source.encode()),
              "example_setup_source": setup_source,
              "release_guard_actual_results": guard_results,
              "exporter_necessary_context": contexts, "necessary_current_prerequisite_slices": slices,
              "section_9_12_routing_assessment": {"path": "course/chapters/09.md", "start_line": line,
                  "section_sha256": digest(nine.encode()), "imports": imports,
                  "source_quote": "推論時交換候選，只會交換相應分數和機率的位置。實數的softmax保留分數高低，但電腦的有限精度可能把很接近的機率捨入為平手；因此下面從softmax前的實際分數挑選。",
                  "code": blocks[0], "conclusion": "This is a local torch/raw-logit/min-ID illustration, with no tiny_perceptron import, definition excerpt, or named library implementation; it adds no route to the W.7 file implementation table. The raw-logit numerical theorem is outside this W.7 routing check."},
              "conclusion": "Current exporter still emits site products under outputs/; actual preview is explicitly unpublished and uses download plus Colab home Upload notebook (or File -> Upload notebook). All 331 currently served notebooks match source13 and local bytes. Release is refused until a real 40-hex commit contains matching bytes for the entire supplied index.",
              "limitations": ["HTTP/HTML and local Python checks only; no graphical layout or figure inspection claimed.",
                              "No Colab login, upload, cloud execution, deployment, release, commit, build or training performed.",
                              "The one-notebook positive commit test does not establish that all 331 are committed or publication-ready.",
                              "This W.7 callback does not complete or expand the first-steps whole-page wave02 reader scope."]}
    path = HERE / "exporter-context-current-v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"evidence": path.relative_to(ROOT).as_posix(), "sha256": digest(path.read_bytes()),
                      "exporter": result["current_exporter"], "bundle_count": len(bundles),
                      "preview": build_info, "release_guard_types": {k: v["type"] for k, v in guard_results.items()},
                      "9.12_routing": result["section_9_12_routing_assessment"]["conclusion"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
