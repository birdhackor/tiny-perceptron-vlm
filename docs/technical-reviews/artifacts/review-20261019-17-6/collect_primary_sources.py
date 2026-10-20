"""以 curl 讀取官方原文，只保存來源收據與支持主張的小段摘錄。"""

import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
REQUESTS = [
    ("linspace", "https://docs.pytorch.org/docs/2.14/generated/torch.linspace.html", [
        "Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end , inclusive.",
    ]),
    ("arange", "https://docs.pytorch.org/docs/2.14/generated/torch.arange.html", [
        "with values from the interval [start, end) taken with common difference step beginning from start.",
    ]),
    ("round", "https://docs.pytorch.org/docs/2.14/generated/torch.round.html", [
        "Rounds elements of input to the nearest integer.",
        "round half to even",
    ]),
    ("dtype", "https://docs.pytorch.org/docs/2.14/tensor_attributes.html", [
        "8-bit integer (signed)",
    ]),
    ("mean", "https://docs.pytorch.org/docs/2.14/generated/torch.mean.html", [
        "Returns the mean value of all elements in the input tensor.",
    ]),
    ("max", "https://docs.pytorch.org/docs/2.14/generated/torch.max.html", [
        "Returns the maximum value of all elements in the input tensor.",
    ]),
    ("quantization", "https://docs.pytorch.org/docs/2.14/quantization-support.html", [
        r"x_\text{out} = (Q_\text{input}-z)*s",
        "zero is represented with no quantization error",
    ]),
    ("application_validation", "https://pytorch.org/blog/quantization-in-practice/", [
        "it’s best to empirically verify which scheme works best for your application and architecture",
    ]),
    ("floor_division", "https://docs.python.org/3.13/reference/expressions.html", [
        "floor division of integers results in an integer; the result is that of mathematical division with the ‘floor’ function applied to the result.",
    ]),
]


def main():
    receipts = []
    for name, url, excerpts in REQUESTS:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url]
        completed = subprocess.run(command, capture_output=True, check=True)
        raw = completed.stdout
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw.decode("utf-8")))
        text = " ".join(text.split())
        title_match = re.search(r"<title>(.*?)</title>", raw.decode("utf-8"), re.S)
        title = html.unescape(title_match[1]) if title_match else ""
        selected = []
        for excerpt in excerpts:
            index = text.find(excerpt)
            if index < 0:
                raise ValueError(f"官方原文中找不到預期摘錄：{name} {excerpt}")
            selected.append({"text": excerpt, "normalized_text_offset": index})
        receipts.append({
            "name": name,
            "url": url,
            "command_argv": command,
            "exit_code": completed.returncode,
            "accessed_at_utc": datetime.now(UTC).isoformat(),
            "title": title,
            "response_bytes": len(raw),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "extraction": "Strip HTML tags, decode entities, collapse whitespace; third-party full text is not saved.",
            "excerpts": selected,
        })
        print(json.dumps(receipts[-1], ensure_ascii=False))
    (DIRECTORY / "primary-source-receipts.json").write_text(
        json.dumps({"receipts": receipts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
