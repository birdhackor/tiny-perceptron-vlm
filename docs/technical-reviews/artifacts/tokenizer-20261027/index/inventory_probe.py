"""Read-only checks for the frozen index introduction; does not execute lessons."""

import argparse
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts import build_course, export_course  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nav_paths(entries):
    for entry in entries:
        if isinstance(entry, str):
            yield entry
        else:
            for value in entry.values():
                yield from nav_paths(value) if isinstance(value, list) else [value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    freeze = json.loads((ROOT / "docs/course-review-20261027-tokenizer/freeze-01.json").read_text())
    page = next(p for p in freeze["pages"] if p["page_id"] == "index")
    snapshot = (ROOT / page["snapshot"]).read_text()
    assert export_course.home_introduction(index, True) == snapshot
    assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
    assert "尚未附上執行結果" in export_course.home_introduction(index, False)
    counts = {"index": len(index), "unique_ids": len({x["id"] for x in index})}
    ids = {x["id"] for x in index}
    source_ids = set()
    for path in (ROOT / "course/chapters").glob("*.md"):
        source_ids.update(re.findall(r"^## ([\dABC]+\.\d+) ", path.read_text(), re.M))
    assert source_ids == ids
    counts["chapter_heading_ids"] = len(source_ids)
    contract_ids = json.loads((ROOT / "course/lesson-contract.json").read_text())["lesson_ids"]
    assert set(contract_ids) == ids and len(contract_ids) == len(ids)
    nav = list(nav_paths(tomllib.loads((ROOT / "zensical.toml").read_text())["project"]["nav"]))
    lesson_nav = [x for x in nav if re.fullmatch(r"[\dABC]+\.\d+\.md", x)]
    assert set(lesson_nav) == {x + ".md" for x in ids} and len(lesson_nav) == len(ids)
    counts["lesson_navigation_entries"] = len(lesson_nav)
    guide = (ROOT / "course/README.md").read_text()
    guide_targets = {"personality": "08", "safety": "09", "image": "10", "quantization": "17"}
    for chapter in guide_targets.values():
        assert f"chapters/{chapter}.md" in guide
    assert all(x in ids for x in ["1.4", "1.5", "1.12", "10.1", "10.7", "10.8", "12.1"])
    counts["executed_notebooks_verified"] = 0
    counts["footer_pairs_verified"] = 0
    counts["teaching_code_cells"] = 0
    counts["teaching_cells_with_renderable_outputs"] = 0
    fingerprints = []
    sample = None
    for item in index:
        original_path = ROOT / item["notebook"]
        executed_path = args.executed / Path(item["notebook"]).relative_to("notebooks")
        original = json.loads(original_path.read_text())
        executed = export_course.checked_notebook(original, executed_path)
        counts["executed_notebooks_verified"] += 1
        footer = export_course.section_footer(item, original, "review-only", True)
        assert f'href="{item["notebook"]}" download' in footer
        assert f'/blob/review-only/{item["notebook"]}' in footer
        counts["footer_pairs_verified"] += 1
        for cell in executed["cells"]:
            if cell["cell_type"] != "code" or cell.get("metadata", {}).get("course_setup"):
                continue
            if "course_figure" in cell.get("metadata", {}):
                continue
            counts["teaching_code_cells"] += 1
            if export_course.output_html(cell.get("outputs", [])):
                counts["teaching_cells_with_renderable_outputs"] += 1
        fingerprints.append([item["id"], sha(original_path), sha(executed_path)])
        if item["id"] == "1.1":
            sample = [export_course.joined(out.get("text", "")) for cell in executed["cells"]
                      for out in cell.get("outputs", []) if out["output_type"] == "stream"]
    sync = io.StringIO()
    with contextlib.redirect_stdout(sync):
        build_course.build(check=True)
    assert counts["index"] == 313
    result = {
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "zensical": version("zensical"), "device": "CPU; file/renderer validation only"},
        "canonical": {"source": page["source"], "selector": page["selector"],
                      "source_sha256": page["source_sha256"], "generated_executed_branch_equals_freeze": True,
                      "nonexecuted_branch_discloses_absent_outputs": True},
        "counts": counts,
        "guide_theme_chapters": guide_targets,
        "curriculum_presence": ["1.4", "1.5", "1.12", "10.1", "10.7", "10.8", "12.1"],
        "build_course_check": sync.getvalue().strip(),
        "all_notebook_pair_fingerprint_sha256": hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest(),
        "sample_1_1_stream_outputs": sample,
        "scope": "Verified stored execution copies match every current cell, all code has execution_count and no error outputs; this probe did not rerun kernels or validate learning quality. Silent code need not produce a visible output.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
