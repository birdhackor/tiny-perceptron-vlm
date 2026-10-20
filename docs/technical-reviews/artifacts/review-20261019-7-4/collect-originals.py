"""用 curl 讀原始來源；只輸出支持主張的短摘錄與取得收據。"""

import hashlib
import json
import re
import shlex
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


SOURCES = [
    {
        "id": "transformer",
        "url": "https://arxiv.org/html/1706.03762v7",
        "version": "arXiv:1706.03762v7，2023-08-02",
        "locator": "Model Architecture；Encoder and Decoder Stacks／Decoder",
        "quote": "This masking, combined with fact that the output embeddings are offset by one position",
        "inspection_note": "已讀 autoregressive 說明與 Decoder 段落：生成下一個符號時使用已知前文，遮罩禁止讀後續位置，輸入與目標錯開一位。本節把此規則套用到自訂的對話 token 序列；論文不規定本專案的角色 ID、答案遮罩或 byte offset。",
    },
    {
        "id": "nonzero",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nonzero.html",
        "version": "PyTorch 2.14 官方 API 文件；本機 torch 2.14.1",
        "locator": "torch.nonzero；as_tuple=False 的預設行為與結果排序",
        "quote": "returns a 2-D tensor where each row is the index for a nonzero value.",
        "inspection_note": "已讀預設 as_tuple=False、每列為一個非零值的座標、結果依字典序排序，以及維度說明。一維 bool mask 得到 [有效數量, 1] 的 LongTensor，第一列只有一個整數座標。本文只對一維 labels 這樣取 item，不擴充到 batch mask。",
    },
    {
        "id": "item",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.item.html",
        "version": "PyTorch 2.14 官方 API 文件；本機 torch 2.14.1",
        "locator": "Tensor.item 的值與單元素前提",
        "quote": "Returns the value of this tensor as a standard Python number. This only works for tensors with one element.",
        "inspection_note": "已讀回傳 Python number、僅適用單元素 tensor 的契約與例子。本例 nonzero 的整數 dtype 讓 item 回傳 Python int；item 本身不是一般浮點數轉整數函式。",
    },
    {
        "id": "cross_entropy",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html",
        "version": "PyTorch 2.14 官方 API 文件；本機 torch 2.14.1",
        "locator": "ignore_index 參數；Shape；reduction",
        "quote": "Specifies a target value that is ignored and does not contribute to the input gradient.",
        "inspection_note": "已讀 class-index target、ignore_index 與 reduction，以及 input/target 對應的 Shape。ignore_index 只跳過 loss target，不會刪掉模型的 input token；本專案先 flatten 成同序位置，sum 後除有效 target 數，並未在 cross_entropy 內再位移。",
    },
]


def main():
    results = []
    for source in SOURCES:
        argv = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", source["url"]]
        completed = subprocess.run(argv, capture_output=True, check=True)
        parser = VisibleText()
        parser.feed(completed.stdout.decode("utf-8"))
        text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
        assert source["quote"] in text
        assert len(source["quote"].split()) <= 25
        results.append({
            **source,
            "accessed_on": datetime.now(UTC).date().isoformat(),
            "command": shlex.join(argv),
            "exit_code": completed.returncode,
            "stderr": completed.stderr.decode("utf-8"),
            "response_bytes": len(completed.stdout),
            "response_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "checked_original": True,
            "storage_policy": "全文只在記憶體；永久保存不超過 25 個英文詞的短摘錄、取得收據及審閱者支持範圍。",
        })
    print(json.dumps({"result": "各份原文的短摘錄均與 curl 取得內容精確相符。", "sources": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
