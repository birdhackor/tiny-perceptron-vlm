"""以 curl 讀取官方原文，只保存定位所需的小段摘錄與取得收據。"""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


BASE = Path(__file__).resolve().parent
ITEMS = [
    ("typesafe-introduction", "https://docs.typesafe.ai/introduction.md", "Introduction", "Jev evaluates typed *questions* against a *state* and returns structured results directly. No text generation, no parsing."),
    ("typesafe-choice", "https://docs.typesafe.ai/primitives/choice.md", "Response structure", "The full probability distribution across every option. The sum of all values is 1."),
    ("typesafe-noul", "https://docs.typesafe.ai/primitives/noul.md", "Noul", "return the probability that the answer is yes."),
    ("typesafe-score", "https://docs.typesafe.ai/primitives/score.md", "Request structure / Response structure", "An ordered array of level descriptions"),
    ("typesafe-control", "https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md", "How to build with TypeSafe", "It does not generate code or choose its own next action."),
    ("torch-softmax", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html", "torch.nn.functional.softmax", "It is applied to all slices along"),
    ("torch-tensor-softmax", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.softmax.html", "Tensor.softmax", "Alias for"),
    ("torch-tolist", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.tolist.html", "Tensor.tolist", "Returns the tensor as a (nested) list."),
    ("torch-argmax", "https://docs.pytorch.org/docs/2.14/generated/torch.argmax.html", "torch.argmax", "Returns the indices of the maximum value of all elements in the"),
    ("torch-linear", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html", "Linear / Shape / Variables", "Applies an affine linear transformation to the incoming data:"),
    ("torch-cross-entropy", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html", "torch.nn.functional.cross_entropy / Parameters", "Compute the cross entropy loss between input logits and target."),
]


def main():
    from html.parser import HTMLParser

    class Article(HTMLParser):
        def __init__(self):
            super().__init__()
            self.take = False
            self.article = ""

        def handle_starttag(self, tag, attrs):
            self.take = tag == "script" and ("type", "application/ld+json") in attrs

        def handle_endtag(self, tag):
            if tag == "script":
                self.take = False

        def handle_data(self, data):
            if self.take:
                try:
                    self.article = json.loads(data).get("articleBody", self.article)
                except ValueError:
                    pass

    records = []
    for identifier, url, locator, excerpt in ITEMS:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
        completed = subprocess.run(command, capture_output=True, check=True)
        raw = completed.stdout
        original = raw.decode("utf-8")
        if "docs.pytorch.org" in url:
            parser = Article()
            parser.feed(original)
            inspected = parser.article
        else:
            inspected = original
        assert excerpt in inspected, "官方原文找不到預期的定位文字"
        assert len(excerpt.split()) <= 25, "摘錄超出限定長度"
        records.append({
            "id": identifier,
            "url": url,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "version": "PyTorch 2.14 官方文件" if "pytorch.org" in url else "未標文件修訂號；以讀取日期與回應 SHA-256 識別",
            "command_argv": command,
            "exit_code": completed.returncode,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            "locator": locator,
            "excerpt": excerpt,
            "checked_original": True,
        })
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(Path.cwd())),
        "environment": {"python": platform.python_version(), "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]},
        "sources": records,
        "scope": "讀取官方原文的收據；只保存支持主張的小段摘錄，不保存第三方全文。",
    }
    target = BASE / "source-excerpts.json"
    with target.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"來源數": len(records), "輸出": str(target.relative_to(Path.cwd()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
