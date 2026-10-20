"""以 curl 讀原始文件，只保存短摘錄與來源收據。"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


specs = [
    ("typesafe-primitives", "https://docs.typesafe.ai/primitives.md", "Primitives (Questions)，Choose a question type／What comes back", "Noul | Is this true?", "官方動態文件；本次回應以 SHA-256 定位"),
    ("typesafe-confidence", "https://docs.typesafe.ai/confidence.md", "How confidence is calculated，Noul／Choice", r"\text{confidence} = \frac{p_{\max} - \frac{1}{n}}{1 - \frac{1}{n}}", "官方動態文件；本次回應以 SHA-256 定位"),
    ("guo-temperature", "https://arxiv.org/html/1706.04599v2", "Calibration Methods；Extension to Multiclass Models，Temperature scaling，公式 (9)", "T>0", "arXiv:1706.04599v2，2017-08-03"),
]
receipts = []
for name, url, locator, fragment, version in specs:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    text = raw.decode()
    if name == "guo-temperature":
        parsed = Text()
        parsed.feed(text)
        text = " ".join(" ".join(parsed.parts).split())
        start = text.rfind("Temperature scaling,")
        inspected = text[start:text.index("4.3 Other Related Works", start)]
        assert "optimized with respect to NLL on the validation set" in inspected
        excerpt = "T T is optimized with respect to NLL on the validation set."
        inspection = "讀取 Calibration Methods 的同分布假設，以及 Temperature scaling 全段：共同正溫度除 logits、驗證集 NLL 選溫、argmax 不變。固定網格是本課實作，非論文聲稱的精確最佳化。"
    elif name == "typesafe-primitives":
        assert "The probability that the answer is yes." in text
        excerpt = "The probability that the answer is yes."
        inspection = "讀取型別表、Choose a question type、What comes back；Choice 是封閉選項，Noul 回 yes 機率，等級需要定義規準，外層程式組合答案。沒有從介面推定 Jev 內部網路。"
    else:
        assert fragment in text
        excerpt = fragment
        inspection = "讀取 How confidence is calculated 的 Noul／Choice：Choice 摘要取決於最大機率與候選數；Noul 沒有另附 confidence。分布摘要不是經實測保證的答對率。"
    assert excerpt in text
    receipts.append({"id": name, "url": url, "version": version, "retrieved_at": datetime.now(UTC).isoformat(), "command_argv": command, "response_sha256": hashlib.sha256(raw).hexdigest(), "bytes_read": len(raw), "locator": locator, "excerpt": excerpt, "inspection_note": inspection, "checked_original": True})
print(json.dumps({"result": "所有原始來源讀取及定位成功", "sources": receipts}, ensure_ascii=False, indent=2))
