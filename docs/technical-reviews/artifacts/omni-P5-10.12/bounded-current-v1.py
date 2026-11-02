"""同原 technical owner 的有限差異 callback；只比對 bytes，不跑模型或 kernel。"""

import difflib
import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
REVIEW = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}


def section(raw, lesson_id):
    return re.search(r"^## " + re.escape(lesson_id) + r" .+?(?=^## |\Z)", raw, re.M | re.S)[0]


def main():
    original_path = REVIEW / "technical-10.12/technical-original-v1.json"
    freeze_path = REVIEW / "source-freeze-v12.json"
    repair_path = REVIEW / "coherence-three-author-repairs-source12-v1.json"
    assert sha(original_path) == "98c2678bcb6b942fb26a79c2971bba3cf4a8a60c424b882c812748eed15be613"
    assert sha(freeze_path) == "1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1"
    assert sha(repair_path) == "76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7"
    original = json.loads(original_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    repairs = json.loads(repair_path.read_text())
    page = next(p for p in freeze["pages"] if p["page_id"] == "10.12")
    assert original["source_sha256"] == page["source_sha256"]
    assert original["figure_sha256"] == page["figures_sha256"]
    assert original["notebook_sha256"] == page["notebook_sha256"]
    assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
    current_unit = section((ROOT / page["source"]).read_text(), "10.12")
    assert hashlib.sha256(current_unit.encode()).hexdigest() == original["source_sha256"]
    assert sha(ROOT / page["notebook"]) == original["notebook_sha256"]
    for path, expected in original["figure_sha256"].items():
        assert sha(ROOT / path) == expected

    source_comparison = []
    for source in original["sources"]:
        if source["kind"] != "repository_code":
            continue
        actual = sha(ROOT / source["path"])
        source_comparison.append({"source_id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
    assert [s["source_id"] for s in source_comparison if not s["unchanged"]] == ["s_ch12"]
    artifact_comparison = []
    for artifact in original["artifacts"]:
        actual = sha(ROOT / artifact["path"])
        assert actual == artifact["sha256"]
        artifact_comparison.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})

    repair = next(r for r in repairs["changes"] if r["page_id"] == "12.17")
    before_path, after_path = ROOT / repair["before"]["path"], ROOT / repair["after"]["path"]
    assert sha(before_path) == repair["before"]["sha256"]
    assert sha(after_path) == repair["after"]["sha256"]
    before, after = before_path.read_text(), after_path.read_text()
    actual_diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="source11/12.17", tofile="source12/12.17", n=2))
    assert actual_diff == repair["diff"]
    addition = "本例的區塊能量取振幅平方平均，即[12.4](12.md#12.4)的平方和除以該塊樣本數；後面的0.01門檻與正規化都依這個尺度計算。\n\n"
    assert after.count(addition) == 1 and after.replace(addition, "", 1) == before
    chapter_path = ROOT / "course/chapters/12.md"
    chapter = chapter_path.read_text()
    assert section(chapter, "12.17") == after
    assert chapter.count(addition) == 1
    restored_chapter_sha = hashlib.sha256(chapter.replace(addition, "", 1).encode()).hexdigest()
    old_chapter_sha = next(s["sha256"] for s in original["sources"] if s["id"] == "s_ch12")
    assert restored_chapter_sha == old_chapter_sha
    old_inspection = json.loads((ARTIFACTS / "inspection-and-derivations-v1.json").read_text())
    current_opening = "\n\n".join(after.split("\n\n")[:3])
    assert current_opening == old_inspection["explicit_roadmap_checks"]["12.17"]["supporting_excerpt"]
    assert re.findall(r"```python\n(.*?)```", before, re.S) == re.findall(r"```python\n(.*?)```", after, re.S)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_p5_technical_10_12",
        "mode": "same-owner bounded reuse; not a fresh blind or complete rereview",
        "environment": {"python": platform.python_version(), "device": "no neural computation; stdlib byte/diff inspection only", "platform": platform.platform()},
        "original_report": reference(original_path),
        "current_source_freeze": reference(freeze_path),
        "author_repair_provenance": reference(repair_path),
        "assigned_page": page,
        "assigned_source_figure_notebook_unchanged": True,
        "repository_source_comparison": source_comparison,
        "original_artifact_integrity": artifact_comparison,
        "changed_dependency": {
            "source_id": "s_ch12",
            "path": "course/chapters/12.md",
            "original_sha256": old_chapter_sha,
            "current_sha256": sha(chapter_path),
            "sha256_after_undoing_exact_single_addition": restored_chapter_sha,
            "before_12_17": reference(before_path),
            "after_12_17": reference(after_path),
            "actual_diff": actual_diff,
            "original_cited_opening_exactly_unchanged": True,
            "current_cited_opening": current_opening,
            "all_12_17_python_blocks_unchanged": True,
        },
        "owner_assessment": "親核的唯一chapter12差異是在12.17人工答案規則示範前補說區塊能量採振幅平方平均，並連12.4及既有0.01尺度。10.12的c26只引用12.17開頭兩段所述共同時間軸與沿用10.12影格保持規則；該完整引文與原審查摘錄逐byte相同。新增能量尺度句不改變此關係，也不改10.12內嵌碼、數字、圖或方法限制。",
        "retained_findings": "26項原有主張及五項checks沿用本人original實讀／CPU／原始來源證據；本次只更新s_ch12真實檔案hash和c26的當前支持定位，未把沿用說成新全文審閱。",
        "limits": ["未重跑原CPU probe、全部Notebook kernel、NN、模型、訓練、調參或final家族。", "未審9.17或19.23修訂，也未對12.17全節另給technical／reader／coherence判定。", "author packet只用作變更定位；本次結論來自獨立凍結前後bytes、原審引用與實際diff比對。"],
        "all_assertions_passed": True,
    }
    target = ARTIFACTS / "bounded-current-result-v1.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(reference(target), ensure_ascii=False))


if __name__ == "__main__":
    main()
