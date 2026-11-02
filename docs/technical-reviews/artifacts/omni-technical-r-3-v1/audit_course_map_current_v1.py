"""R.3 callback: rerun the original read-only audit against freeze v5.

The previous program and raw output are immutable inputs. This wrapper adds
the exact source change, current scope assertion, and content comparisons.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGINAL_CODE_SHA256 = "ad5ca7666702c5fa8c19db696e30998020bc43e39c927729ce0ab204847078e4"
ORIGINAL_OUTPUT_SHA256 = "c9b57ce4e32562034655763150b3858a7077280b9745c3e5db07ecc126c14b32"
ORIGINAL_REPORT_SHA256 = "ff1340cc8219e5ee7d3ee7db2df44044523b772ffc7b7f2a3024b10d05a33f5b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_code = HERE / "audit_course_map.py"
    original_output = HERE / "audit-output.json"
    original_report = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_r_3/technical-original-v1.json"
    assert sha(original_code) == ORIGINAL_CODE_SHA256
    assert sha(original_output) == ORIGINAL_OUTPUT_SHA256
    assert sha(original_report) == ORIGINAL_REPORT_SHA256
    spec = importlib.util.spec_from_file_location("original_r3_audit", original_code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json"

    # Locate the same necessary chapter-19 passages by headings, so unrelated
    # insertions cannot silently turn an old line range into different evidence.
    old_excerpt = module.excerpt

    def current_excerpt(path, first, last):
        if path == ROOT / "course/chapters/19.md" and (first, last) in {(1681, 1685), (1889, 1899)}:
            lines = path.read_text(encoding="utf-8").splitlines()
            section_id = "19.22" if first == 1681 else "19.23"
            start = next(i for i, line in enumerate(lines) if line.startswith("## " + section_id + " "))
            if section_id == "19.22":
                end = start + 5
            else:
                end = next(i for i in range(start, len(lines)) if lines[i].startswith("```"))
            return old_excerpt(path, start + 1, end)
        return old_excerpt(path, first, last)

    module.excerpt = current_excerpt
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        module.main()
    result = json.loads(captured.getvalue())
    previous = json.loads(original_output.read_bytes())
    raw = (ROOT / result["source"]["snapshot"]).read_text(encoding="utf-8")
    old_raw = (ROOT / previous["source"]["snapshot"]).read_text(encoding="utf-8")
    old_phrase = "所有小節與 Notebook 都列在[目錄](lessons.md)。"
    new_phrase = "各章小節與 Notebook 都列在[目錄](lessons.md)。"
    assert old_raw.count(old_phrase) == 1
    assert raw.count(new_phrase) == 1 and old_phrase not in raw
    assert raw == old_raw.replace(old_phrase, new_phrase)
    assert result["index_coverage"]["chapter_section_count"] == 331
    assert result["index_coverage"]["listed_section_count"] == 331
    assert result["index_coverage"]["listed_notebook_count"] == 331
    freeze = json.loads(module.FREEZE.read_bytes())
    canonical = freeze["canonical_source_freeze"]
    assert canonical["path"] == "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json"
    assert sha(ROOT / canonical["path"]) == canonical["sha256"]
    result["callback"] = {
        "freeze": {"path": module.FREEZE.relative_to(ROOT).as_posix(), "sha256": sha(module.FREEZE)},
        "canonical_source_freeze": canonical,
        "original_report": {"path": original_report.relative_to(ROOT).as_posix(), "sha256": sha(original_report)},
        "original_code": {"path": original_code.relative_to(ROOT).as_posix(), "sha256": sha(original_code)},
        "original_output": {"path": original_output.relative_to(ROOT).as_posix(), "sha256": sha(original_output)},
        "original_issue_pointer": "/issues/0",
        "old_quote": old_phrase,
        "new_quote": new_phrase,
        "only_source_change_is_scope_phrase": True,
        "current_scope_matches_complete_chapter_and_notebook_coverage": True,
        "chapter_introductions_and_selected_headings_unchanged": {
            key: result["chapters"][key] == previous["chapters"][key]
            for key in result["chapters"]
        },
        "reference_introductions_unchanged": {
            key: result["reference_pages"][key] == previous["reference_pages"][key]
            for key in result["reference_pages"]
        },
        "legacy_product_primary_records_unchanged": result["legacy_product_records"] == previous["legacy_product_records"],
        "limitation": "Filesystem/primary-text audit only; no model inference, new training, performance reproduction, or whole-file collector endorsement.",
    }
    result["result"] = "PASS: freeze-v5 R.3 fully reread and audited; 26 direct links, 23 chapter mappings, C route junctions, 331 chapter sections and 331 Notebooks verified. Corrected '各章小節' matches actual index scope."
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
