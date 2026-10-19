"""讀者交原始判斷前的自我檢查：每頁的 session 用 gate 本身的 reader_session 核對，再查報告欄位。

驗證：頁面依閱讀計畫的順序與範圍、每頁來源與圖的 SHA 等於凍結檔、session 的 opening 與全部 notes
（數量、順序、單元 SHA、owner、時間、五欄觀察與四題欄位）通過 incremental_revision.reader_session，以及每頁
報告的必要欄位與 ai_estimate 形式。
不涵蓋：判斷是否正確、note 是否真的在讀當下單元時寫成、閱讀計畫的先備是否真的讀過；seal、收件與 closure
的時序。
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SESSIONS = "docs/course-revision-20261016/reader-sessions"
VERDICTS = {"pass", "需補說明"}
PAGE_FIELDS = ("reader_summary", "missing_visuals")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def reference(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def check_page(review, freeze, page, task):
    errors = []
    pid = page.get("page_id")
    frozen = next((p for p in freeze["pages"] if p["page_id"] == pid), None)
    if frozen is None:
        return [f"{pid}: 凍結檔沒有這一頁"]
    if page.get("source_sha256") != frozen["source_sha256"]:
        errors.append(f"{pid}: source_sha256 與凍結檔不同")
    if page.get("figures_sha256") != frozen["figures_sha256"]:
        errors.append(f"{pid}: figures_sha256 與凍結檔不同")
    if page.get("verdict") not in VERDICTS:
        errors.append(f"{pid}: verdict 要是 pass 或 需補說明")
    for field in PAGE_FIELDS:
        if not isinstance(page.get(field), str) or not page[field].strip():
            errors.append(f"{pid}: 缺少 {field}")
    for key in ("checks", "issues"):
        if not isinstance(page.get(key), list):
            errors.append(f"{pid}: {key} 必須是清單")
    if page.get("verdict") == "需補說明" and not any(i.get("necessary") is True for i in page.get("issues", [])):
        errors.append(f"{pid}: verdict 是需補說明，issues 卻沒有 necessary=true 的問題")
    estimate = page.get("ai_estimate", {})
    lo, hi = estimate.get("minutes_min"), estimate.get("minutes_max")
    if not (type(lo) is int and type(hi) is int and 0 < lo <= hi and estimate.get("reason")):
        errors.append(f"{pid}: ai_estimate 要有正整數 minutes_min ≤ minutes_max 與 reason")
    evidence = page.get("session_evidence", {})
    session = page.get("session")
    if not session or evidence.get("opening", {}).get("path") != f"{SESSIONS}/{session}/opening.json":
        errors.append(f"{pid}: session_evidence.opening 要指向這一頁 session 的 opening.json")
        return errors
    figures = {
        name: freeze["figure_snapshots"][name]
        for name in frozen["figures_sha256"]
        if name in freeze["figure_snapshots"]
    }
    proof = {
        "opening": evidence["opening"],
        "source": {"path": frozen["snapshot"], "sha256": frozen["source_sha256"]},
        "figures": figures,
        "notes": evidence.get("notes", []),
    }
    try:
        found, _ = review.reader_session(ROOT, proof, task)
        if found != pid:
            errors.append(f"{pid}: session 屬於另一頁 {found}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        errors.append(f"{pid}: reader_session 拒絕：{error}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    task = report.get("reviewer_task")
    errors = []
    for key in ("source_freeze", "reading_plan"):
        ref = report.get(key, {})
        if not ref.get("path") or reference(ref["path"]) != ref:
            errors.append(f"{key} 要是 repo 內相對路徑與現行 SHA")
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False))
        raise SystemExit(1)
    plan = json.loads((ROOT / report["reading_plan"]["path"]).read_bytes())
    freeze = json.loads((ROOT / report["source_freeze"]["path"]).read_bytes())
    if plan.get("reviewer_task") != task:
        errors.append("reviewer_task 與閱讀計畫不同")
    pages = report.get("pages", [])
    if [p.get("page_id") for p in pages] != plan.get("main_pages"):
        errors.append("pages 要依閱讀計畫 main_pages 的順序，一頁一筆")
    review = module("incremental_review_for_reader_check", "docs/review-tools/incremental_revision.py")
    for page in pages:
        errors.extend(check_page(review, freeze, page, task))
    print(json.dumps({"pages": len(pages), "errors": errors}, ensure_ascii=False, indent=1))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
