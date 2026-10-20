"""以 curl 讀取原始來源，只保存版本收據、短摘錄與核對說明。"""

import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime

sources = [
    {
        "id": "bpe", "url": "https://arxiv.org/html/1508.07909v5", "version": "arXiv:1508.07909v5，2016-06-10",
        "locator": "Subword Translation / Related Work / Byte Pair Encoding / Evaluation",
        "quotes": ["The best choice of subword units may be task-specific.", "We iteratively count all symbol pairs"],
        "inspection_note": "原文以詞頻加權的相鄰對決定合併；字表大小與文字序列長度是不同成本。新文字依訓練時學得的規則切分，沒有出現的相鄰對不會因預算增加而自動變成一個 token。其研究為英德與英俄翻譯，不是本教材的中英 byte-level BPE 實測。罕見完整詞可拆為在其他詞出現過的片段；這支持共享觀測例子的動機，不保證任何字表大小的品質。",
    },
    {
        "id": "attention", "url": "https://arxiv.org/html/1706.03762v7", "version": "arXiv:1706.03762v7，2023-08-02",
        "locator": "Scaled Dot-Product Attention, equation (1); Embeddings and Softmax; Why Self-Attention, Table 1",
        "quotes": ["The keys and values are also packed together into matrices", "we share the same weight matrix"],
        "inspection_note": "已讀 QK 的矩陣公式、embedding 與 pre-softmax linear 的共享敘述及自注意力複雜度表。當 Q 與 K 各有 T 列時，QK 轉置乘積為 T×T。這是邏輯注意力表大小，不能推成每個 kernel 都配置完整矩陣，也不能直接宣稱總工作量或記憶體必然省多少。",
    },
    {
        "id": "embedding", "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html", "version": "PyTorch 2.14 官方文件；本機 2.14.1",
        "locator": "Embedding / Parameters / Variables / Shape",
        "quotes": ["the learnable weights of the module of shape (num_embeddings, embedding_dim)"],
        "inspection_note": "已讀字表大小、每列寬度、weight 形狀與整數索引查表的 API 契約。V 種 token、D 個特徵對應 V×D 個權重。",
    },
    {
        "id": "linear", "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html", "version": "PyTorch 2.14 官方文件；本機 2.14.1",
        "locator": "Linear / Parameters / Variables / Shape",
        "quotes": ["the layer will not learn an additive bias."],
        "inspection_note": "已讀 y=xA 轉置+b、weight 的 out_features×in_features 形狀，以及 bias=False 的契約；width→V 的輸出層為每個位置產生 V 個分數。",
    },
]
for source in sources:
    command = ["curl", "-fsSL", "--max-time", "30", source["url"]]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    text = raw.decode()
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.S)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))
    for quote in source["quotes"]:
        assert quote in text, (source["id"], quote)
    source.update(command_argv=command, retrieved_sha256=hashlib.sha256(raw).hexdigest(), retrieved_bytes=len(raw), checked_original=True)
print(json.dumps({"accessed_at_utc": datetime.now(UTC).isoformat(), "environment": {"retriever": "curl via Python subprocess", "parser": "Python standard library", "storage": "第三方全文僅在記憶體處理，不寫入工作樹"}, "sources": sources}, ensure_ascii=False, indent=2))
