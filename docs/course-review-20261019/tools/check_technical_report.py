"""技術審閱者交件前的自我檢查：對凍結原文跑 check_technical_reviews._validate，並核對圖清單。

驗證：報告的 schema、來源與 artifact 的路徑及 SHA、與凍結原文相同的 source_sha256，以及 figure_sha256
是否恰好等於該頁在凍結檔裡的圖清單（incremental_revision.audit_manifest 對頁面證據的要求）。
不涵蓋：主張是否正確、是否涵蓋本頁全部實質主張、來源是否真的讀過；派工、收件與 seal 的時序。
verdict 為 revise 的原始報告，`_validate` 本來就會列出未核實與未解決的問題，那不是格式錯誤。
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_validator():
    spec = importlib.util.spec_from_file_location("technical_validator", ROOT / "scripts/check_technical_reviews.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page_id")
    parser.add_argument("report", type=Path)
    parser.add_argument("--freeze", type=Path, required=True, help="這次審閱用的凍結檔")
    parser.add_argument("--reader-task", action="append", default=[], help="同一 wave 的 reader task，可重複")
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_bytes())
    page = next((p for p in freeze["pages"] if p["page_id"] == args.page_id), None)
    if page is None:
        raise SystemExit("凍結檔沒有這一頁：" + args.page_id)
    raw = (ROOT / page["snapshot"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != page["source_sha256"]:
        raise SystemExit("凍結原文與凍結檔記錄的 SHA 不符：" + page["snapshot"])
    report = json.loads(args.report.read_bytes())
    owners = {report.get("reviewer_task"): [args.page_id]}
    errors = load_validator()._validate(
        ROOT, ROOT / page["source"], args.page_id, raw.decode("utf-8"), report, owners, set(args.reader_task)
    )
    if report.get("figure_sha256") != page["figures_sha256"]:
        errors.append("figure_sha256 必須恰好等於凍結檔裡這一頁的圖：" + json.dumps(page["figures_sha256"]))
    print(json.dumps({"page_id": args.page_id, "verdict": report.get("verdict"), "errors": errors}, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
