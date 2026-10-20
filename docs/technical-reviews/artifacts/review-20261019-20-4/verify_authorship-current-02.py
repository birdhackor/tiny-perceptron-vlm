"""唯讀核對作者聲明的範圍、檔案指紋與逐筆判準。"""
import ast
import hashlib
import importlib.util
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PREVIOUS = "docs/course-review-20261019/reviews/rv1019_technical_20-4/technical-20-4-current-01.json"
DECLARATION = "docs/authorship.md"
PREPARER = "scripts/prepare_natural_v4_voice_questions.py"
METADATA = "docs/natural-assistant/v4/data/voice-question-sources.json"
MANIFEST = "docs/natural-assistant/v4/manifest.json"
RETAINED_VERIFIER = "docs/technical-reviews/artifacts/review-20261019-20-4/verify_evidence-current-02.py"


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def run():
    previous = read(PREVIOUS)
    source_checks = [{"path": s["path"], "recorded_sha256": s["sha256"], "current_sha256": sha(s["path"]), "unchanged": sha(s["path"]) == s["sha256"]} for s in previous["sources"] if s["kind"] == "repository_code"]
    artifact_checks = [{"path": a["path"], "recorded_sha256": a["sha256"], "current_sha256": sha(a["path"]), "unchanged": sha(a["path"]) == a["sha256"]} for a in previous["artifacts"]]
    assert all(s["unchanged"] for s in source_checks)
    assert all(a["unchanged"] for a in artifact_checks)
    assert sha(PREVIOUS) == "6a0de33e7e2a3f6a925da54b27f17cda9115a0e8ccb2910bb057f3765f7d8db9"
    assert sha(DECLARATION) == "58092b2cabcac23e39b935b2aaf3716ec2a68eee8ad3d8f937483b8a5fe8fb20"
    text = (ROOT / DECLARATION).read_text()
    bindings = dict(re.findall(r"^- `([^`]+)`：`([0-9a-f]{64})`$", text, re.M))
    assert set(bindings) == {PREPARER, METADATA}
    bound_checks = [{"path": p, "declared_sha256": digest, "current_sha256": sha(p), "matches": sha(p) == digest} for p, digest in bindings.items()]
    assert all(item["matches"] for item in bound_checks)
    spec = importlib.util.spec_from_file_location("retained_verifier", ROOT / RETAINED_VERIFIER)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    replayed = verifier.run()
    tree = ast.parse((ROOT / PREPARER).read_text())
    selected = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in n.targets))
    rows = {r["source_utterance_id"]: r for r in read(METADATA)["audio_rows"]}
    runtime_rows = {r["source_utterance_id"]: r for r in read(MANIFEST)["audio_rows"] if r.get("source_utterance_id") in rows}
    matched = [{"utterance_id": uid, "split": split, "source_matches_spec": rows[uid]["semantic_rubric"] == rubric and rows[uid]["split"] == split, "manifest_matches_spec": runtime_rows[uid]["references"]["rubric"] == rubric and runtime_rows[uid]["split"] == split} for split, uid, rubric in selected]
    assert len(selected) == len(rows) == len(runtime_rows) == 8
    assert all(r["source_matches_spec"] and r["manifest_matches_spec"] for r in matched)
    assert "這些判準由 AI 工具撰寫，沒有逐題經人審核。" in text
    assert "本課也沒有安排其他人逐條審核" in text
    return {
        "previous_report": {"path": PREVIOUS, "sha256": sha(PREVIOUS)},
        "previous_issue_ids": [item["claim_id"] for item in previous["issues"]],
        "source_checks": source_checks,
        "artifact_checks": artifact_checks,
        "declaration": {"path": DECLARATION, "sha256": sha(DECLARATION), "statement_date": "2026-10-20", "locators": ["作者的說明", "涵蓋的判準"], "binds_ai_authorship_and_no_per_question_human_review": True, "bindings": bound_checks, "evidence_class": "作者對本人及本課安排的具名流程聲明；不是獨立調查或品質驗證。"},
        "rubric_matches": matched,
        "split_counts": dict(Counter(item["split"] for item in matched)),
        "replayed_checks": replayed,
        "scope": "可確認現行判準與作者聲明指名的版本相同；接受作者對撰寫方式和安排人審狀態的直接說明。未重放歷史對話，未把聲明當成標註正確性或模型品質的證明。",
    }


if __name__ == "__main__":
    print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)), "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；唯讀資料核對與 AST 分流，無模型載入"}, "accessed_at": datetime.now(timezone.utc).isoformat(), "result": run()}, ensure_ascii=False, indent=2))
