"""用 curl 讀官方原文，只保存支持主張的短摘錄與來源收據。"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, text):
        if not self.skip and text.strip():
            self.parts.append(text.strip())


specifications = [
    {
        "id": "nemo",
        "url": "https://docs.nvidia.com/nemo-framework/user-guide/25.07/sft_peft/packed_sequence.html",
        "version": "NeMo Framework User Guide 25.07",
        "locator": "Sequence Packing / introduction and block triangular mask discussion",
        "excerpts": [
            "multiple training sequences (examples) are concatenated into one long sequence (pack).",
            "a block triangular mask",
        ],
        "inspection_note": "已閱讀序列裝填的定義、PAD 浪費、區塊三角遮罩與普通稠密注意力的複雜度討論。支持減少 PAD 與保留文件隔離是兩件事；不將該文件的效能敘述套到本課手寫注意力。",
    },
    {
        "id": "broadcast",
        "url": "https://docs.pytorch.org/docs/2.14/notes/broadcasting.html",
        "version": "PyTorch 2.14 documentation; local execution 2.14.1",
        "locator": "Broadcasting semantics / General semantics",
        "excerpts": [
            "automatically expanded to be of equal sizes (without making copies of the data).",
        ],
        "inspection_note": "已讀 trailing dimensions 的相等／長度 1／不存在規則，以及結果各軸取最大長度的規則；支持 (1,4) 與 (4,1) 逐元素比較得到 (4,4)。擴展不代表複製儲存。",
    },
    {
        "id": "sdpa",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html",
        "version": "PyTorch 2.14 documentation; local execution 2.14.1",
        "locator": "scaled_dot_product_attention signature, equivalent pseudocode, attn_mask and is_causal parameters",
        "excerpts": [
            "Computes scaled dot product attention on query, key and value tensors",
            "upper left causal bias",
        ],
        "inspection_note": "已讀等價偽碼、bool attn_mask 的 True 表示參與注意力、可廣播至 (N,...,Hq,L,S) 的形狀契約與 non-square causal bias。偽碼由 Q/K/V、遮罩決定可讀範圍，沒有 EOS token ID 特判；合併本專案 forward 的程式可核對 EOS 不會自動隔離文件。API 合併多個注意力步驟，但本頁未宣稱固定使用 Flash 或普遍加速。",
    },
    {
        "id": "cross_entropy",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html",
        "version": "PyTorch 2.14 documentation; local execution 2.14.1",
        "locator": "CrossEntropyLoss / input, ignore_index and reduction definitions",
        "excerpts": [
            "The input is expected to contain the unnormalized logits for each class",
        ],
        "inspection_note": "已讀未正規化 logits、ignore_index 的目標不計入代價，以及 reduction='sum'／mean 的公式。支持本課原始候選分數的 logit 定義與 PAD target 可被排除；不表示 PAD 的 embedding、FFN 或注意力分數不再計算。",
    },
    {
        "id": "autograd",
        "url": "https://docs.pytorch.org/docs/2.14/notes/autograd.html",
        "version": "PyTorch 2.14 documentation; local execution 2.14.1",
        "locator": "Autograd mechanics / How autograd encodes the history",
        "excerpts": [
            "automatically compute the gradients using the chain rule.",
        ],
        "inspection_note": "已讀 autograd 對純量目標使用連鎖律計算梯度與 backward graph 的說明。支持參數梯度是答案代價對參數的敏感度；更新實際使用 AdamW 與裁剪，不能把梯度值直接當成最終參數增量。",
    },
]

results = []
for specification in specifications:
    command = ["curl", "-fLsS", "--max-time", "45", specification["url"]]
    completed = subprocess.run(command, check=True, capture_output=True)
    parser = TextOnly()
    parser.feed(completed.stdout.decode())
    text = " ".join(" ".join(parser.parts).split())
    for excerpt in specification["excerpts"]:
        assert excerpt in text, specification["id"]
    results.append({**specification, "command_argv": command, "exit_code": completed.returncode, "accessed_on": datetime.now(timezone.utc).date().isoformat(), "retrieved_html_sha256": hashlib.sha256(completed.stdout).hexdigest(), "retrieved_bytes": len(completed.stdout), "checked_original": True})

print(json.dumps({"method": "curl 原文讀取；全文僅在記憶體，未保存第三方全文。", "sources": results}, ensure_ascii=False, indent=2))
