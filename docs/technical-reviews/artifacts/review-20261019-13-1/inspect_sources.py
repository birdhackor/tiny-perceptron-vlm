import hashlib
import json
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


specifications = [
    {
        "id": "dpo",
        "url": "https://arxiv.org/html/2305.18290v3",
        "version": "arXiv:2305.18290v3，2024-07-29",
        "locator": "Preliminaries：Reward Modelling Phase；偏好分布與靜態比較資料集",
        "excerpts": [
            "These are then presented to human labelers who express preferences for one answer",
            r"y_{w}\succ y_{l}\mid x",
        ],
        "inspection": "偏好由標註者提供，較佳與較差回答都以同一個提問 x 為條件。資料包含提問與兩份候選；不是無條件宣告某種寫法永遠較好。公式只描述相對偏好，不保證勝者是理想示範。",
    },
    {
        "id": "human_preferences",
        "url": "https://arxiv.org/html/1706.03741v4",
        "version": "arXiv:1706.03741v4，2023-02-17",
        "locator": "Preference Elicitation；Instructions Provided to Contractors：MuJoCo",
        "excerpts": [
            "the two segments are equally good",
            "they are unable to compare the two segments",
            "not included in the database",
        ],
        "inspection": "原文允許相等與無法比較，並分別處理；任務指示也指定優先判準。這是軌跡比較的原始研究，僅支持偏好蒐集可保留平手／不確定與明示任務判準，沒有聲稱本專案 DPO loss 接受平手。",
    },
    {
        "id": "evaluation",
        "url": "https://scikit-learn.org/1.7/modules/cross_validation.html",
        "version": "scikit-learn 1.7.2 documentation",
        "locator": "Cross-validation: evaluating estimator performance 開頭；Cross-validation iterators for grouped data",
        "excerpts": [
            "A test set should still be held out for final evaluation",
            "same group is not represented in both testing and training sets",
        ],
        "inspection": "已讀開頭有關用測試集調 hyperparameter 造成評估洩漏的說明，以及 grouped data 對未見家族的切分要求。這支持訓練／驗證／測試用途與家族不跨組的原則，不證明某份具體資料已正確切分；具體切分另外執行查核。",
    },
]
receipts = []
for specification in specifications:
    command = ["curl", "-fLsS", "--max-time", "40", specification["url"]]
    raw = subprocess.check_output(command)
    parser = TextParser()
    parser.feed(raw.decode("utf-8"))
    normalized = " ".join(" ".join(parser.parts).split())
    for excerpt in specification["excerpts"]:
        if excerpt not in normalized:
            raise ValueError("原始來源不含預期摘錄：" + specification["id"])
    receipts.append(
        {
            **specification,
            "command_argv": command,
            "accessed_at": datetime.now(UTC).isoformat(),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            "checked_original": True,
            "full_text_retained": False,
        }
    )
print(json.dumps({"sources": receipts}, ensure_ascii=False, indent=2))
