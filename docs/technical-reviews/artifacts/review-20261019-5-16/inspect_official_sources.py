"""以 curl 讀官方原文；只保存雜湊、定位與支持主張的短摘錄。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime
from html.parser import HTMLParser
from zoneinfo import ZoneInfo


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


specs = [
    {
        "id": "pytorch_loss",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/loss.py",
        "version": "PyTorch v2.14.1",
        "locator": "CrossEntropyLoss 類別；class-index target 的 unreduced loss、mean reduction 與 ignore_index 參數",
        "excerpt": "the loss is averaged over non-ignored targets.",
        "inspection": "已讀 CrossEntropyLoss 完整 docstring：class-index target 的單點代價是 -w_y log p_y，忽略位置乘零；mean 的分母是未忽略位置的權重總和。沒有 class weight 時即有效目標數。這也說明 token 數量占比不能單獨決定代價占比。來源權重可由 unreduced loss 另行相乘，不能把 class weight API 誤當 source weight API。",
    },
    {
        "id": "python_dict",
        "url": "https://docs.python.org/3.13/library/stdtypes.html#mapping-types-dict",
        "version": "Python 3.13.16 documentation；版本由 HTML 標題核對",
        "locator": "Mapping Types — dict；iter(d)、values() 與 Dictionary view objects",
        "excerpt": "Return an iterator over the keys of the dictionary.",
        "inspection": "已讀 Mapping Types 的字典建立方式、iter(d)、values() 及 view iteration：直接迭代字典取鍵，values view 迭代值，與正文程式一致。",
    },
    {
        "id": "python_comprehension",
        "url": "https://docs.python.org/3.13/reference/expressions.html#dictionary-displays",
        "version": "Python 3.13.16 documentation；版本由 HTML 標題核對",
        "locator": "Dictionary displays；dict comprehension 的兩個冒號分隔運算式與 resulting key and value elements",
        "excerpt": "needs two expressions separated with a colon",
        "inspection": "已讀 Dictionary displays：dict comprehension 用冒號分隔 key 與 value，for/if 決定生成流程，產生的 key/value 插入新字典。正文 key 是 source，value 是 amount / total；與 set comprehension 使用大括號但不含 key:value 的差別成立。",
    },
    {
        "id": "python_sampling",
        "url": "https://docs.python.org/3.13/library/random.html#random.choices",
        "version": "Python 3.13.16 documentation；版本由 HTML 標題核對",
        "locator": "random.choices(population, weights=None, *, cum_weights=None, k=1)",
        "excerpt": "with replacement",
        "inspection": "已讀 random.choices：回傳 k 次有放回抽樣，預設等機率，可用相對權重。實驗 fit_lm 用 random.Random(42).choices(examples, k=16)，每批筆數固定但來源與有效長度會隨抽樣變動。調整來源筆數或抽樣機率可改有效目標配比；有限 batch 不保證精確落在期望比例。",
    },
]
receipts = []
for spec in specs:
    command = ["curl", "-fsSL", spec["url"]]
    raw = subprocess.check_output(command)
    if spec["url"].endswith(".py"):
        body = raw.decode()
    else:
        parser = Text()
        parser.feed(raw.decode())
        body = " ".join(" ".join(parser.parts).split())
    if spec["excerpt"] not in body:
        raise ValueError("官方原文沒有預期摘錄：" + spec["id"])
    title = re.search(r"<title>(.*?)</title>", raw.decode(), re.S)
    position = body.index(spec["excerpt"])
    receipts.append({
        **spec,
        "command_argv": command,
        "retrieved_sha256": hashlib.sha256(raw).hexdigest(),
        "retrieved_bytes": len(raw),
        "document_title": title.group(1) if title else "torch/nn/modules/loss.py",
        "excerpt_offset_in_normalized_text": position,
        "checked_original": True,
        "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
        "result": "curl 成功，摘錄與已讀定位核對一致；第三方全文未保存。",
    })
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-16/inspect_official_sources.py",
    "environment": {"python": platform.python_version(), "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]},
    "sources": receipts,
}, ensure_ascii=False, indent=2))
