import hashlib
import json
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.suppressed += 1
        if tag in ("p", "h2", "h3", "h4", "li", "figcaption"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.suppressed -= 1

    def handle_data(self, text):
        if not self.suppressed:
            self.parts.append(text)


specs = [
    {
        "id": "mtp",
        "url": "https://arxiv.org/html/2404.19737v1",
        "version": "arXiv:2404.19737v1，2024-04-30",
        "locators": ["Method", "Inference", "Experiments on real data", "Benefits scale with model size", "Faster inference"],
        "excerpts": ["independent output heads implemented in terms of transformer layers", "a shared unembedding matrix"],
        "receipt_notes": [
            "Method 把下一步目標 x[t+1] 改為同一前文下的多個 x[t+i]；各分支是 f_hi，共用 f_s 與 f_u。",
            "Inference 允許只留下一步分支作普通自回歸，也提出額外分支搭配 self-speculative decoding。",
            "Experiments on real data 的參數配平段落說明：增加 n−1 個分支層時，從核心移除 n−1 層。",
            "Benefits scale with model size 與 Figure 3 記錄小模型未必受益，不能由多個目標保證品質改善。",
            "Faster inference 使用 greedy self-speculative decoding，沒有把平行預測本身當成加速的完整程序。",
        ],
    },
    {
        "id": "transformer",
        "url": "https://arxiv.org/html/1706.03762v7",
        "version": "arXiv:1706.03762v7，2023-08-02；原論文發表於 2017",
        "locators": ["Encoder and Decoder Stacks", "Scaled Dot-Product Attention", "Multi-Head Attention", "Position-wise Feed-Forward Networks", "Embeddings and Softmax"],
        "excerpts": ["prevent positions from attending to subsequent positions", "learned linear transformation and softmax function"],
        "receipt_notes": [
            "Decoder 段落限制對後續位置的注意力；本課以輸入位置 t 預測 t+1，因此對應 j≤t。",
            "Scaled Dot-Product Attention 將多個 query 合成 Q 矩陣同時計算，與因果限制相容。",
            "Multi-Head Attention 的 head 是 Q/K/V 投影與注意力運算；與 MTP 的未來分支、最終輸出矩陣功能不同。",
            "Position-wise Feed-Forward Networks、Embeddings and Softmax 分別支持逐位置 FFN、token 向量與最終候選分數投影。",
            "Model Architecture 說明每步生成一個符號，生成過的符號成為下一步輸入。",
        ],
    },
    {
        "id": "blockwise",
        "url": "https://arxiv.org/html/1811.03115v1",
        "version": "arXiv:1811.03115v1，2018-11-07",
        "locators": ["Abstract", "Blockwise Parallel Decoding", "verify"],
        "excerpts": ["back off to the longest prefix validated by a scoring model"],
        "receipt_notes": [
            "Abstract 與 Blockwise Parallel Decoding 分別描述平行提出未來候選，再驗證可接受的最長前綴。",
            "算法的 predict、verify、accept 是生成流程，與多目標訓練分開；本節沒有宣稱任何固定加速倍率。",
        ],
    },
]
receipts = []
for spec in specs:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", spec["url"]]
    result = subprocess.run(command, check=True, capture_output=True)
    raw = result.stdout
    parser = TextParser()
    parser.feed(raw.decode("utf-8"))
    text = "".join(parser.parts)
    for phrase in spec["excerpts"] + spec["locators"]:
        assert phrase.lower() in text.lower(), "來源定位或摘錄不在這次下載的原文中"
    receipts.append({**spec, "accessed_at": datetime.now(UTC).isoformat(), "command_argv": command,
                     "returncode": result.returncode, "response_bytes": len(raw),
                     "response_sha256": hashlib.sha256(raw).hexdigest(),
                     "storage_policy": "全文只在記憶體中解析；只保存定位、短摘錄、雜湊與核對筆記。"})
print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-20/fetch_sources.py", "receipts": receipts}, ensure_ascii=False, indent=2))
