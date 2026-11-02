"""Inspect explicit homepage destinations and topic coverage without reviewing their results."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"


def record(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    freeze = json.loads(FREEZE.read_bytes())
    pages = {x["page_id"]: x for x in freeze["pages"]}
    guide = (ROOT / pages["course"]["snapshot"]).read_text()
    groups = {
        "先讀前文": ["2.2", "3.1", "4.1"],
        "回答與停止": ["7.11", "7.12", "7.17"],
        "決策與資訊是否足夠": ["9.11", "9.14", "9.18"],
        "圖片與聲音時間": ["10.12", "12.17", "14.19"],
        "長度效率與記憶體": ["14.7", "16.15", "17.1"],
        "個性文風安全圖片量化閱讀入口": ["8.1", "9.1", "10.1", "17.1"],
    }
    rows = []
    for topic, ids in groups.items():
        rows.append({"topic": topic, "destinations": [{"id": i, "title": pages[i]["title"],
                    "preview_exists": (ROOT / "outputs/omni-P5-preview-v8" / f"{i}.html").is_file()} for i in ids]})
    result = {
        "reviewer_task": "/root/omni_p5_technical_index",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "device": "CPU; read-only canonical title/link inspection"},
        "freeze": record(FREEZE), "guide_snapshot": record(ROOT / pages["course"]["snapshot"]),
        "guide_snapshot_matches_freeze": record(ROOT / pages["course"]["snapshot"])["sha256"] == pages["course"]["source_sha256"],
        "guide_r2_quotes": [
            "第一次接觸語言模型，先沿第1–4章閱讀。",
            "接續文章之後，還要用問答示範分清哪一段是提問、哪一段該回覆及何時停止。",
            "回覆要用來選服務時，還得判斷任務是否支援、資料是否足夠；",
            "線索若在圖片、聲音或某一時刻的畫面裡，則得保留內容和發生時間。",
        ],
        "quotes_present": all(q in guide for q in (
            "第一次接觸語言模型，先沿第1–4章閱讀。",
            "分清哪一段是提問、哪一段該回覆及何時停止。",
            "判斷任務是否支援、資料是否足夠；",
            "線索若在圖片、聲音或某一時刻的畫面裡，則得保留內容和發生時間。",
        )),
        "topic_destination_inspection": rows,
        "guide_route_labels_present": {x: x in guide for x in ("文風", "安全", "圖片", "量化")},
        "limits": "Coverage of advertised topics and explicit navigation only. This is not an independent review of other lessons' methods, measurements, trained capabilities or every prerequisite.",
    }
    dest = Path(__file__).with_name("route-audit-v1.json")
    with dest.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"path": str(dest.relative_to(ROOT)), "quotes_present": result["quotes_present"],
                      "route_labels": result["guide_route_labels_present"], "groups": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
