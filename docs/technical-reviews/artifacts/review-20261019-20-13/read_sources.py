"""以 curl 讀取原始文件，只保留核對所需的小段摘錄與收據。"""

import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


SOURCES = [
    {"id": "python_hashlib", "url": "https://docs.python.org/3.13/library/hashlib.html",
     "needles": ["bytes-like objects", "This is a bytes object of size", "returned as a string object of double length"]},
    {"id": "python_literals", "url": "https://docs.python.org/3.13/reference/lexical_analysis.html",
     "needles": ["Bytes literals are always prefixed", "bytes type"]},
    {"id": "peft_checkpoint", "url": "https://raw.githubusercontent.com/huggingface/peft/v0.18.1/docs/source/developer_guides/checkpoint.md",
     "needles": ["you need to have the original model available", "only contains the parameters of the adapter module", "information about the base model"]},
    {"id": "torch_checkpoint", "url": "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
     "needles": ["resuming training", "optimizer’s state_dict", "optimizer's state_dict"]},
    {"id": "fleurs_dataset", "url": "https://huggingface.co/datasets/google/fleurs/raw/main/README.md",
     "needles": ["focus on read-speech", "We collect between one and three recordings", "Mandarin"]},
    {"id": "aishell_dataset", "url": "https://www.openslr.org/33/",
     "needles": ["Chinese Mandarin speech corpus", "400 people"]},
    {"id": "qwen_metadata", "url": "https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct/revision/89644892e4d85e24eaac8bacfd4f463576704203"},
    {"id": "whisper_metadata", "url": "https://huggingface.co/api/models/openai/whisper-large-v3-turbo/revision/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"},
]


def fetch(source):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "35", source["url"]]
    response = subprocess.run(command, capture_output=True)
    result = {"id": source["id"], "url": source["url"], "command": command,
              "accessed_at": datetime.now(UTC).isoformat(), "exit_code": response.returncode,
              "raw_sha256": hashlib.sha256(response.stdout).hexdigest(), "raw_bytes": len(response.stdout)}
    if response.returncode:
        result["error"] = response.stderr.decode("utf-8", errors="replace")
        return result
    if "_metadata" in source["id"]:
        data = json.loads(response.stdout)
        result["excerpt"] = {key: data.get(key) for key in ["id", "sha", "safetensors", "library_name"]}
    else:
        decoded = response.stdout.decode("utf-8")
        if not source["url"].endswith(".md"):
            parser = Plain()
            parser.feed(decoded)
            decoded = " ".join(" ".join(parser.parts).split())
        title = re.search(rb"<title>(.*?)</title>", response.stdout, re.S)
        result["title"] = title[1].decode("utf-8") if title else source["id"]
        result["matches"] = []
        for needle in source["needles"]:
            start = decoded.casefold().find(needle.casefold())
            result["matches"].append({"needle": needle, "found": start >= 0,
                                      "excerpt": decoded[max(0, start - 80):start + 360] if start >= 0 else ""})
    return result


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(fetch, SOURCES))
print(json.dumps({"environment": {"python": sys.version, "device": "cpu; curl HTTPS"}, "sources": results}, ensure_ascii=False, indent=2))
