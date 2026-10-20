"""以 curl 讀原始來源；只輸出收據與支持主張的短摘錄，不保存全文。"""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from html import unescape


def fetch(url):
    command = ["curl", "-fsSL", "--max-time", "45", url]
    result = subprocess.run(command, check=True, capture_output=True)
    return result.stdout, command


def receipt(url, version, locator, excerpt):
    raw, command = fetch(url)
    text = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", raw.decode())))
    if excerpt not in text:
        raise ValueError("來源中找不到預定短摘錄：" + url)
    return {
        "url": url,
        "version": version,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "exit_code": 0,
        "bytes_read": len(raw),
        "original_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": locator,
        "excerpt": excerpt,
        "full_text_saved": False,
    }


api, api_command = fetch("https://api.github.com/repos/marcotcr/checklist/commits/master")
commit = json.loads(api)["sha"]
sources = [
    receipt(
        "https://huggingface.co/docs/transformers/v4.38.0/en/chat_templating",
        "Transformers v4.38.0",
        "Advanced: Extra credit: How do chat templates work?；並讀取 Introduction、generation prompts、training 與 template choice",
        "Each message is a dictionary with two keys, role and content .",
    ),
    receipt(
        "https://arxiv.org/html/2407.10759v1",
        "arXiv:2407.10759v1，2024-07-15",
        "Introduction；Methodology / Model Architecture（公式 1）、Supervised Fine-tuning",
        "conditioning on audio representations and previous text sequences",
    ),
    receipt(
        f"https://raw.githubusercontent.com/marcotcr/checklist/{commit}/README.md",
        commit,
        "Perturbing data for INVs and DIRs；Creating and running tests；Custom expectation functions",
        "Expectation on pairs of `(orig, new)` examples (for `INV` and `DIR`):",
    ),
]
print(json.dumps({
    "curl": subprocess.run(["curl", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
    "commit_resolution": {"command": api_command, "exit_code": 0, "commit": commit},
    "sources": sources,
}, ensure_ascii=False, indent=2))
