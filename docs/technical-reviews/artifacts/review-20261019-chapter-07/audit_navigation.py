"""核對指定章首的凍結版本、示意算術與課程導覽，保存實際讀取的短段落。"""

import hashlib
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-review-20261019/freeze-04.json"
PAGE_ID = "chapter-07"
EXPECTED_FREEZE = "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
EXPECTED_SOURCE = "9d1701085139c003d95eb401aef3abbd1d258dab8f5b9b551d09ea8f30b901b0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    freeze = json.loads(FREEZE.read_bytes())
    require(digest(FREEZE) == EXPECTED_FREEZE, "凍結清單版本不符")
    page = next(p for p in freeze["pages"] if p["page_id"] == PAGE_ID)
    snapshot = ROOT / page["snapshot"]
    raw = snapshot.read_text(encoding="utf-8")
    require(digest(snapshot) == EXPECTED_SOURCE, "凍結章首版本不符")
    require(page["source_sha256"] == EXPECTED_SOURCE, "凍結清單與章首雜湊不符")
    current_source = ROOT / page["source"]
    current = current_source.read_text(encoding="utf-8")
    current_intro = re.split(r"^## ", current, maxsplit=1, flags=re.M)[0]
    require(current_intro == raw, "目前章首與凍結章首不同")
    require(page["figures_sha256"] == {}, "凍結圖清單應為空")
    require(not re.search(r"!\[[^\]]*\]\([^)]*\)", raw), "章首含圖像引用")
    require("```" not in raw, "章首含程式或命令區塊")

    links = []
    for label, target in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", raw):
        path_text, separator, anchor = target.partition("#")
        resolved = (current_source.parent / path_text).resolve()
        require(resolved.is_relative_to(ROOT), "導覽超出專案")
        require(resolved.is_file(), "導覽目標不存在")
        text = resolved.read_text(encoding="utf-8")
        heading = None
        if separator:
            matches = re.findall(r"^## ([A-Z0-9]+\.\d+) (.+)$", text, re.M)
            heading = next((title for identifier, title in matches if identifier == anchor), None)
            require(heading is not None, "導覽目標沒有對應的標題")
        links.append({"label": label, "target": target, "path": resolved.relative_to(ROOT).as_posix(),
                      "sha256": digest(resolved), "heading": heading, "exists": True})

    requirements = [
        ("對話怎麼表示？", ["1+1=?", "role", "content"]),
        ("回答第一個 token 在哪個位置被預測？", ["assistant", "第一個 A"]),
        ("哪些位置應該學？", ["讀到", "計代價"]),
        ("不算 loss 的內容就看不見嗎？", ["仍在輸入裡", "沒有直接代價"]),
        ("補齊 batch 的位置該算 loss 嗎？", ["PAD"]),
        ("填充的位置會被當線索嗎？", ["PAD"]),
        ("補了 PAD 該取哪個輸出？", ["最後一個真實位置"]),
        ("模型怎麼知道回答結束？", ["EOS"]),
        ("答案被截光了還在訓練嗎？", ["沒有任何回答目標"]),
        ("預訓練模型怎麼學會對話？", ["監督式微調", "SFT"]),
        ("模型會回答新組合嗎？", ["留出", "訓練、驗證、測試"]),
        ("為什麼先大量讀文章，再練習當助手？", ["預訓練", "後訓練"]),
        ("後訓練有哪些不同教法？", ["示範", "偏好比較", "回饋"]),
        ("回答能寫多長，和前文能放多少，有何不同？", ["P＋G≤T", "上下文視窗"]),
        ("下一個之外，還能一起練習預測什麼？", ["再下一步", "同一個前文位置"]),
        ("選了 high，就是固定多想幾秒嗎？", ["thinking effort", "請求檔位", "實際耗用"]),
        ("同一題，怎樣教不同預算的回答？", ["條件", "示範"]),
        ("怎樣判斷多花計算值得？", ["品質與成本", "凍結之後才碰測試資料", "當作必修"]),
        ("寫出中間步驟能幫忙嗎？", ["長度"]),
        ("最後答對代表過程正確嗎？", ["驗證器"]),
        ("多試幾次能找到正確答案嗎？", ["候選"]),
        ("找得到也選得中嗎？", ["多數決", "驗證器"]),
        ("花更多計算值得嗎？", ["驗證的成本", "回答當下的計算"]),
    ]
    pages_by_title = {p["title"]: p for p in freeze["pages"]
                      if p["source"] in {"course/chapters/07.md", "course/chapters/0C.md"}}
    coverage = []
    for title, terms in requirements:
        target = pages_by_title[title]
        path = ROOT / target["snapshot"]
        body = path.read_text(encoding="utf-8")
        require(digest(path) == target["source_sha256"], "導覽目標凍結內容與雜湊不符")
        paragraphs = body.split("\n\n")
        matching = []
        for term in terms:
            paragraph = next((p for p in paragraphs if term in p), None)
            require(paragraph is not None, "導覽所述主題未在目標課文找到")
            matching.append({"term": term, "excerpt": paragraph})
        coverage.append({"page_id": target["page_id"], "title": title,
                         "path": target["snapshot"], "sha256": digest(path),
                         "opening": paragraphs[:3], "matches": matching})

    arithmetic = {"expression": "1 + 1", "expected": 2, "observed": 1 + 1,
                  "arithmetic": "Python 整數精確加法", "equal": 1 + 1 == 2}
    require(arithmetic["equal"], "示意題的整數答案不符")
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
                   "docs/technical-reviews/artifacts/review-20261019-chapter-07/audit_navigation.py",
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "device": "CPU", "third_party_packages": "未匯入"},
        "freeze": {"path": FREEZE.relative_to(ROOT).as_posix(), "sha256": digest(FREEZE)},
        "source": {"path": page["snapshot"], "sha256": digest(snapshot), "text": raw,
                   "current_intro_matches_snapshot": True, "figures": {}, "code_fences": False},
        "links": links,
        "navigation_coverage": coverage,
        "arithmetic": arithmetic,
        "scope": "只核對章首算術、導覽內容對應與目標存在；不判定被連結小節的方法或實驗結論。",
        "result": "所有章首版本、示意題、連結與主題對應檢查通過",
    }
    output = ARTIFACT_DIR / "audit_navigation_output.json"
    require(not output.exists(), "執行紀錄已存在，請使用新的輸出檔名")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": result["result"], "links_checked": len(links),
                      "coverage_targets": len(coverage), "arithmetic": arithmetic,
                      "output": output.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
