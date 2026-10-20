"""以 curl 讀取原始來源；全文僅留記憶體，保存定位、短摘錄與收據。"""

import hashlib
import json
import subprocess
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "section", "h1", "h2", "h3", "li", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


specs = [
    {
        "id": "paper_sft",
        "url": "https://arxiv.org/html/2203.02155v1",
        "version": "arXiv:2203.02155v1，2022-03-04，HTML 原文",
        "needles": ["Step 1", "Supervised fine-tuning", "language modeling objective", "next token"],
        "excerpt": "We fine-tune GPT-3 on our labeler demonstrations using supervised learning.",
    },
    {
        "id": "torch_cross_entropy",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/functional.py",
        "version": "PyTorch v2.14.1，torch/nn/functional.py，cross_entropy",
        "needles": ["def cross_entropy(", "ignore_index (int, optional)"],
        "excerpt": "Specifies a target value that is ignored and does not contribute to the input gradient.",
    },
    {
        "id": "trl_sft",
        "url": "https://huggingface.co/docs/trl/v0.23.1/sft_trainer",
        "version": "TRL v0.23.1，SFT Trainer 官方文件",
        "needles": ["Computing the loss", "Train on assistant messages only"],
        "excerpt": "the model is trained to predict the next token given the previous ones.",
    },
]

receipts = []
for spec in specs:
    command = ["curl", "--silent", "--show-error", "--fail", "--location", "--max-time", "30", spec["url"]]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    original = raw.decode("utf-8")
    if spec["id"] != "torch_cross_entropy":
        parser = TextParser()
        parser.feed(original)
        text = " ".join("".join(parser.parts).split())
    else:
        text = original[original.index("def cross_entropy("):original.index("def binary_cross_entropy(")]
    print(json.dumps({"來源": spec["url"], "版本": spec["version"], "全文位元組": len(raw)}, ensure_ascii=False))
    for needle in spec["needles"]:
        start = 0
        for _ in range(3):
            index = text.lower().find(needle.lower(), start)
            if index < 0:
                break
            print(text[max(0, index - 180):index + 1300])
            start = index + len(needle)
    normalized = " ".join(text.split())
    if spec["excerpt"] not in normalized:
        raise ValueError("短摘錄未出現在原文中")
    receipts.append({
        **{key: value for key, value in spec.items() if key != "needles"},
        "command_argv": command,
        "exit_code": completed.returncode,
        "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
        "full_response_sha256": hashlib.sha256(raw).hexdigest(),
        "full_response_bytes": len(raw),
        "checked_original": True,
        "full_response_retained": False,
    })

destination = Path(__file__).with_name("source-receipts.json")
destination.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
