"""以 curl 讀原始來源；全文只留在記憶體，收據保存雜湊與短摘錄。"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in ("p", "div", "section", "h1", "h2", "h3", "dt", "dd", "tr"):
            self.parts.append("\n")
        if tag == "math":
            values = dict(attrs)
            if values.get("alttext"):
                self.parts.append(" [LaTeX: " + values["alttext"] + "] ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


SPECS = [
    {
        "id": "ppo",
        "url": "https://arxiv.org/pdf/1707.06347v2",
        "version": "arXiv:1707.06347v2",
        "needles": [],
        "excerpt": None,
        "locator": "Actor-Critic、Algorithm 1 與 value function loss",
    },
    {
        "id": "kl_definition",
        "url": "https://docs.pytorch.org/docs/2.14/distributions.html#torch.distributions.kl.kl_divergence",
        "version": "PyTorch 2.14 文件；執行版本 2.14.1",
        "needles": ["Compute Kullback-Leibler"],
        "excerpt": "Compute Kullback-Leibler divergence",
        "locator": "torch.distributions.kl.kl_divergence：KL(p || q) 定義",
    },
    {
        "id": "dpo",
        "url": "https://arxiv.org/html/2305.18290v3",
        "version": "arXiv:2305.18290v3，2024-07-29",
        "needles": ["RL Fine-Tuning Phase", "coefficient", "reference model"],
        "excerpt": "RL Fine-Tuning Phase",
        "locator": "Preliminaries：SFT、Reward Modelling Phase、RL Fine-Tuning Phase；公式 (2)、(3)",
    },
    {
        "id": "detach",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html",
        "version": "PyTorch 2.14 文件；執行版本 2.14.1",
        "needles": ["Returns a new Tensor", "never require gradient"],
        "excerpt": "The result will never require gradient.",
        "locator": "torch.Tensor.detach",
    },
    {
        "id": "log_softmax",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.log_softmax.html",
        "version": "PyTorch 2.14 文件；執行版本 2.14.1",
        "needles": ["numerically unstable", "Parameters"],
        "excerpt": "numerically unstable",
        "locator": "torch.nn.functional.log_softmax：定義與 dim 參數",
    },
]


def main():
    receipts = []
    for spec in SPECS:
        command = ["curl", "--location", "--fail", "--silent", "--show-error", spec["url"]]
        result = subprocess.run(command, capture_output=True, check=True)
        if spec["id"] == "ppo":
            assert result.stdout.startswith(b"%PDF")
            text = "原始 PDF 已用 web.open 閱讀：Algorithm、公式 (7)、(9)–(12)、Algorithm 1。curl 另核對原檔位元組。"
            print(text)
        else:
            parser = Text()
            parser.feed(result.stdout.decode())
            text = "".join(parser.parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        compact = re.sub(r"\s+", " ", text)
        print("URL:", spec["url"])
        print("SHA-256:", hashlib.sha256(result.stdout).hexdigest())
        for needle in spec["needles"]:
            match = re.search(re.escape(needle), text, flags=re.I)
            print("定位:", needle)
            print(text[max(0, match.start() - 200) : match.end() + 2300] if match else "未找到")
        excerpt = spec["excerpt"]
        present = excerpt is not None and excerpt.casefold() in compact.casefold()
        receipts.append(
            {
                "id": spec["id"],
                "url": spec["url"],
                "version": spec["version"],
                "retrieved_at": datetime.now(UTC).isoformat(),
                "raw_sha256": hashlib.sha256(result.stdout).hexdigest(),
                "raw_bytes": len(result.stdout),
                "command_argv": command,
                "exit_code": result.returncode,
                "locator": spec["locator"],
                "short_excerpt": excerpt if present else None,
                "excerpt_found": present if excerpt is not None else "not_applicable: PDF 由 web.open 讀取文字",
                "full_text_saved": False,
            }
        )
    destination = Path(__file__).with_name("original-source-receipts-final.json")
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipts, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
