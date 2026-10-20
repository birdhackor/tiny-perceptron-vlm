"""以 curl 讀取官方原文，只保留支持主張的小段摘錄與下載收據。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime
from html import unescape


DOCUMENTS = [
    (
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html",
        "Embedding — PyTorch 2.14 documentation",
        "Embedding 的說明與 Variables.weight",
        [
            "A simple lookup table that stores embeddings of a fixed dictionary and size.",
            "the learnable weights of the module",
        ],
        "讀取 Embedding 類別說明、參數、weight 與 Shape：整數索引選取對應向量，weight 是可學習矩陣。此處使用預設未凍結且無 padding_idx 的建構方式。",
    ),
    (
        "https://docs.pytorch.org/docs/2.14/generated/torch.flatten.html",
        "torch.flatten — PyTorch 2.14 documentation",
        "torch.flatten 的說明、start_dim 參數與範例",
        ["The order of elements in input is unchanged."],
        "讀取 flatten 說明與 start_dim=1 範例：保留 batch 軸，其餘軸按原元素順序展平；不表示一定零複製。",
    ),
    (
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html",
        "Linear — PyTorch 2.14 documentation",
        "Linear 的仿射公式、Shape 與 Variables.weight/bias",
        ["Applies an affine linear transformation to the incoming data:"],
        "讀取公式 y=xA^T+b、輸入輸出形狀與可學習 weight/bias。公式支持各輸出加權混合特徵，也容許 A=0 的輸入無關情況；公式本身不保證學會有效特徵。",
    ),
]


def main():
    receipts = []
    for url, title, locator, excerpts, note in DOCUMENTS:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
        raw = subprocess.check_output(command)
        plain = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", raw.decode("utf-8"))))
        assert title in plain, "下載原文的文件版本與預期不符"
        for excerpt in excerpts:
            assert excerpt in plain, "原文中找不到指定摘錄"
        receipts.append(
            {
                "url": url,
                "version": title,
                "accessed_on": datetime.now().astimezone().date().isoformat(),
                "command": command,
                "downloaded_original_sha256": hashlib.sha256(raw).hexdigest(),
                "locator": locator,
                "excerpts": excerpts,
                "inspection_note": note,
            }
        )
    print(
        json.dumps(
            {
                "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-02/fetch_source_excerpts.py",
                "environment": {
                    "python": platform.python_version(),
                    "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0],
                },
                "result": "三份官方文件的版本與短摘錄核對成功；第三方全文只在記憶體中處理。",
                "sources": receipts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
