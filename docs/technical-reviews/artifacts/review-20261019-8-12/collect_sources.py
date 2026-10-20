"""以 curl 讀原始來源，只保存收據、小段摘錄與查核摘要。"""

import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def plain(text):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())


specs = [
    {
        "id": "judge_paper",
        "url": "https://arxiv.org/html/2306.05685v4",
        "version": "arXiv:2306.05685v4",
        "locator": "Limitations of LLM-as-a-Judge / Verbosity bias；HTML S3.SS3.p6.1 與 S3.SS3.p7.1",
        "excerpt": "We then make them unnecessarily verbose by asking GPT-4 to rephrase the list without adding any new information",
        "inspection": "已讀 verbosity bias 的定義與 repetitive list 設計：在不加入新資訊下擴寫並比較偏好；只支持偏差定義與控制內容的方法，不能把本頁字元裁判當作真實模型實驗。",
    },
    {
        "id": "python_functions",
        "url": "https://docs.python.org/3.13/library/functions.html",
        "version": "Python 3.13.16 documentation",
        "locator": "len；max",
        "excerpt": "The key argument specifies a one-argument ordering function",
        "inspection": "已讀 len 的項目數定義及 max 全部 API 說明：依 key 排序、預設比較物件本身，最大值相同時回傳先遇到者；本例長度不同，不涉及平手。",
    },
    {
        "id": "python_types",
        "url": "https://docs.python.org/3.13/library/stdtypes.html",
        "version": "Python 3.13.16 documentation",
        "locator": "Text Sequence Type — str；str.split；Common Sequence Operations；Set Types — set, frozenset",
        "excerpt": "Strings are immutable sequences of Unicode code points.",
        "inspection": "已讀 str 定義、split 指定 sep 保留空字串的規則與尾端逗號例子、序列乘法及切片負索引、set 的不同元素與去重定義。字元在此為 Unicode code point，這批中文與數字都各占一個。",
    },
    {
        "id": "python_comparison",
        "url": "https://docs.python.org/3.13/reference/expressions.html",
        "version": "Python 3.13.16 documentation",
        "locator": "Comparisons / Value comparisons；str 的字典序規則",
        "excerpt": "compare lexicographically using the numerical Unicode code points",
        "inspection": "已讀 Value comparisons 中 str 比較的規則；這是 max 不傳 key 時對字串使用的比較，不會隱含改成字元長度比較。",
    },
    {
        "id": "sentencepiece",
        "url": "https://raw.githubusercontent.com/google/sentencepiece/v0.2.1/README.md",
        "version": "SentencePiece v0.2.1 README.md",
        "locator": "開頭介紹；Technical highlights；Overview / What is SentencePiece?",
        "excerpt": "SentencePiece implements **subword units**",
        "inspection": "已讀專案維護者的 README：輸入是 Unicode 字元序列，但模型可用 BPE 或 unigram 的 subword 單位並映射成 token ID；所以 len(str) 不是一般 tokenizer 的 token 計數。本頁沒有指定 tokenizer，不計算特定模型的 token 數。",
    },
]

receipts = []
for spec in specs:
    command = ["curl", "-fLsS", "--max-time", "30", spec["url"]]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    text = raw.decode("utf-8")
    decoded = plain(text) if "raw.githubusercontent" not in spec["url"] else " ".join(text.split())
    if " ".join(spec["excerpt"].split()) not in decoded:
        raise ValueError("來源摘錄不在 curl 回傳的原文中：" + spec["id"])
    if "docs.python.org" in spec["url"]:
        title = plain(re.search(r"<title>(.*?)</title>", text, re.S)[1])
        if spec["version"] not in title:
            raise ValueError("文件實際版本與預期不符：" + title)
    else:
        title = None
    receipts.append(
        {
            **spec,
            "command_argv": command,
            "exit_code": completed.returncode,
            "accessed_at_utc": datetime.now(UTC).isoformat(),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            "document_title": title,
            "checked_original": True,
        }
    )

destination = Path(__file__).with_name("source-receipts.json")
with destination.open("x", encoding="utf-8") as handle:
    json.dump({"sources": receipts, "storage": "第三方全文僅在記憶體讀取；未保存全文。"}, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"result": "已核對所有摘錄與文件版本", "path": destination.as_posix(), "source_count": len(receipts)}, ensure_ascii=False))
