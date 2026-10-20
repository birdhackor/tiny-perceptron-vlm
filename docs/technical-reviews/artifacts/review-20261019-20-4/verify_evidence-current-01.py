"""唯讀複查凍結版本、既有收據與 AISHELL 判準的直接來源。"""
import ast
import difflib
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_20-4/technical-20-4-original-01.json"
OLD_SOURCE = "docs/course-review-20261019/freeze-03/sources/20.4.md"
NEW_SOURCE = "docs/course-review-20261019/freeze-06/sources/20.4.md"
FREEZE = "docs/course-review-20261019/freeze-06.json"
FIGURE = "course/figures/natural-v4-training-cat.svg"
PREPARER = "scripts/prepare_natural_v4_voice_questions.py"
METADATA = "docs/natural-assistant/v4/data/voice-question-sources.json"
RESEARCH = "docs/natural-assistant-v4/research/chat-speech.json"
PROTOCOLS = ["docs/natural-assistant/v4/validation-protocol.json", "docs/natural-assistant/v4/validation-protocol-lower-lr.json"]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def run():
    original = read(ORIGINAL)
    source_checks = [{"id": s["id"], "path": s["path"], "recorded_sha256": s["sha256"], "current_sha256": sha(s["path"]), "unchanged": s["sha256"] == sha(s["path"])} for s in original["sources"] if s["kind"] == "repository_code"]
    artifact_checks = [{"id": a["id"], "path": a["path"], "recorded_sha256": a["sha256"], "current_sha256": sha(a["path"]), "unchanged": a["sha256"] == sha(a["path"])} for a in original["artifacts"]]
    script = ROOT / "docs/technical-reviews/artifacts/review-20261019-20-4/verify_evidence.py"
    spec = importlib.util.spec_from_file_location("retained_verifier", script)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    replayed = verifier.local()
    tree = ast.parse((ROOT / PREPARER).read_text())
    selected = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in n.targets))
    metadata = read(METADATA)
    by_id = {r["source_utterance_id"]: r for r in metadata["audio_rows"]}
    matches = [{"utterance_id": utterance, "split_matches": by_id[utterance]["split"] == split, "rubric_matches": by_id[utterance]["semantic_rubric"] == rubric, "rubric": rubric} for split, utterance, rubric in selected]
    provenance_fields = ["rubric_author", "rubric_author_task", "label_author", "author_kind", "human_review", "human_reviewed", "human_review_status", "human_audit"]
    research = read(RESEARCH)
    body = (ROOT / NEW_SOURCE).read_text()
    chapter = (ROOT / "course/chapters/20.md").read_text()
    current_section = next(part for part in chapter.split("\n## ") if part.startswith("20.4 "))
    current_section = "## " + current_section
    raw_old = (ROOT / OLD_SOURCE).read_text()
    result = {
        "bound_inputs": {p: sha(p) for p in [ORIGINAL, OLD_SOURCE, NEW_SOURCE, FREEZE, FIGURE]},
        "original_issues": original["issues"],
        "all_original_repo_sources_unchanged": all(r["unchanged"] for r in source_checks),
        "repo_source_checks": source_checks,
        "all_original_artifacts_unchanged": all(r["unchanged"] for r in artifact_checks),
        "artifact_checks": artifact_checks,
        "source_diff": "".join(difflib.unified_diff(raw_old.splitlines(keepends=True), body.splitlines(keepends=True), fromfile=OLD_SOURCE, tofile=NEW_SOURCE)),
        "current_section_matches_freeze": current_section == body,
        "python_code_block_count": body.count("```python"),
        "replayed_local_checks": replayed,
        "aishell_rubric_trace": {
            "preparer": PREPARER,
            "preparer_sha256": sha(PREPARER),
            "selected_count": len(selected),
            "all_spec_rubrics_and_splits_match_metadata": all(r["split_matches"] and r["rubric_matches"] for r in matches),
            "matches": matches,
            "metadata_modification_notice": metadata["modification_notice"],
            "metadata_reference_kinds": sorted({r["reference_kind"] for r in metadata["audio_rows"]}),
            "checked_provenance_field_names": provenance_fields,
            "explicit_provenance_at_document_level": {k: metadata[k] for k in provenance_fields if k in metadata},
            "explicit_provenance_at_row_level": [{"id": r["id"], "fields": {k: r[k] for k in provenance_fields if k in r}} for r in metadata["audio_rows"]],
            "research_task": research["reviewer_task"],
            "research_aishell_record": research["current_delivery"]["AISHELL_human_read_questions"] if "current_delivery" in research else None,
            "research_root_keys": list(research),
            "grading_protocols": [{"path": p, "sha256": sha(p), "grader": read(p)["manual_grading"]["grader"], "source_description": read(p)["manual_grading"]["sources"]} for p in PROTOCOLS],
            "evidence_limit": "資料準備程式證明判準預先存在且被逐字採用。研究 task 與 AI 評分者身分，不能單獨證明特定判準由哪個作者撰寫，也不能證明未記錄的真人逐題審核從未發生；缺欄位本身不是歷史不存在的證據。",
        },
        "additional_source_sha256": {p: sha(p) for p in [PREPARER, METADATA, RESEARCH] + PROTOCOLS},
    }
    assert result["all_original_repo_sources_unchanged"]
    assert result["all_original_artifacts_unchanged"]
    assert result["current_section_matches_freeze"]
    assert result["python_code_block_count"] == 0
    assert result["aishell_rubric_trace"]["all_spec_rubrics_and_splits_match_metadata"]
    return result


if __name__ == "__main__":
    print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)), "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；資料與 AST 分流檢查，無模型載入"}, "accessed_at": datetime.now(timezone.utc).isoformat(), "result": run()}, ensure_ascii=False, indent=2))
