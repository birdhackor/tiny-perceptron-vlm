"""用 curl 讀取原始來源，只保存查核所需的短摘錄與收據。"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


SOURCES = [
    {
        "id": "transformer",
        "url": "https://arxiv.org/html/1706.03762v7",
        "version": "arXiv:1706.03762v7, 2023-08-02",
        "locator": "Optimizer，learning-rate formula 與緊接的說明",
        "phrases": ["This corresponds to increasing the learning rate linearly for the first"],
        "inspect": "This corresponds to increasing the learning rate linearly",
        "note": "原式為 d_model^(-1/2) min(step_num^(-1/2), step_num warmup_steps^(-3/2))；起步線性增加，之後反平方根下降。只支持 warmup 的用法，不支持本課餘弦配置必然更好。",
    },
    {
        "id": "sgdr",
        "url": "https://arxiv.org/html/1608.03983v5",
        "version": "arXiv:1608.03983v5",
        "locator": "Stochastic Gradient Descent with warm restarts，餘弦式與上下限的說明",
        "phrases": ["we decay the learning rate with a cosine annealing for each batch as follows"],
        "inspect": "Within the",
        "note": "原式 eta_min + (eta_max-eta_min)(1+cos(pi T_cur/T_i))/2。核對一段下降的形狀與上下限；SGDR 另有重啟，本課函式沒有重啟，兩者不可混同。",
    },
    {
        "id": "bengio",
        "url": "https://arxiv.org/html/1206.5533v2",
        "version": "arXiv:1206.5533v2",
        "locator": "Gradient Descent and Learning Rate；Hyper-Parameters 的 out-of-sample 選擇原則",
        "phrases": [
            "hyper-parameter that is called the learning rate",
            "select its value based on out-of-sample data",
        ],
        "inspect": "For any hyper-parameter that has an impact",
        "note": "讀取梯度更新式及 hyper-parameter 選擇段落。對固定更新方向，學習率縮放參數改動；驗證資料用來選擇配置，但不是品質保證，也不是獨立測試資料。",
    },
    {
        "id": "python-math",
        "url": "https://docs.python.org/3.13/library/math.html",
        "version": "Python 3.13.16 documentation",
        "locator": "math.cos 與 math.pi",
        "phrases": ["Return the cosine of x radians."],
        "inspect": "math.cos",
        "note": "math.cos 以弧度為輸入；math.pi 是可用精度的圓周率常數。配合透明的半週餘弦推導，不把浮點近似當作精確實數。",
    },
]


def main():
    receipts = []
    for spec in SOURCES:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", spec["url"]]
        completed = subprocess.run(command, capture_output=True, check=True)
        raw = completed.stdout
        parser = PlainText()
        parser.feed(raw.decode("utf-8"))
        text = re.sub(r"\s+", " ", " ".join(parser.parts))
        excerpts = []
        for phrase in spec["phrases"]:
            if phrase not in text:
                raise ValueError("無法定位來源摘錄：" + spec["id"] + " " + phrase)
            excerpts.append(phrase)
        offset = text.find(spec["inspect"])
        # 原文只在本次工具輸出顯示；持久檔保存短摘錄。
        print(json.dumps({"id": spec["id"], "inspection_text": text[max(0, offset - 100):offset + 1800]}, ensure_ascii=False))
        receipts.append({
            **{k: v for k, v in spec.items() if k not in {"phrases", "inspect"}},
            "command_argv": command,
            "exit_code": completed.returncode,
            "fetched_at_utc": datetime.now(UTC).isoformat(),
            "accessed_on": "2026-10-20",
            "download_bytes": len(raw),
            "download_sha256": hashlib.sha256(raw).hexdigest(),
            "checked_original": True,
            "excerpts": excerpts,
            "full_text_storage": "完整原文僅在記憶體中讀取，未保存至 repo。",
        })
    target = Path(__file__).with_name("source-receipts.json")
    with target.open("x", encoding="utf-8") as handle:
        json.dump({"sources": receipts}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
