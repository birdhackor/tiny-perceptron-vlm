"""以 curl 讀原始來源；只保存短摘錄與查核結果，不保存第三方全文。"""

import concurrent.futures
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = [
    ("uv", "https://docs.astral.sh/uv/concepts/projects/sync/", "Locking and syncing are automatic in uv."),
    ("lfs", "https://raw.githubusercontent.com/git-lfs/git-lfs/main/docs/man/git-lfs-config.adoc", "GIT_LFS_SKIP_SMUDGE"),
    ("longpaths", "https://gitforwindows.org/faq", "core.longpaths"),
    ("autograd", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html", "Computes the gradient"),
    ("colab", "https://research.google.com/colaboratory/faq.html", "Google Account"),
    ("kernel", "https://ipython.readthedocs.io/en/stable/install/kernel_install.html", "--sys-prefix"),
    ("site", "https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html", "MathJax"),
    ("site_info", "https://birdhackor.github.io/tiny-perceptron-vlm/build-info.json", ""),
    ("hf_v2", "https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2?recursive=true&expand=false", ""),
]


def obtain(spec):
    name, url, needle = spec
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    process = subprocess.run(command, capture_output=True, timeout=50)
    raw = process.stdout
    record = {"id": name, "url": url, "accessed_on": "2026-10-20", "command": command,
              "returncode": process.returncode, "stderr": process.stderr.decode(errors="replace"),
              "raw_sha256": hashlib.sha256(raw).hexdigest(), "bytes_read": len(raw)}
    text = raw.decode(errors="replace")
    plain = html.unescape(re.sub("<[^>]+>", " ", text))
    plain = re.sub(r"\s+", " ", plain)
    position = plain.lower().find(needle.lower()) if needle else 0
    record["locator"] = needle
    record["found"] = position >= 0
    if needle and position >= 0:
        record["excerpt"] = " ".join(plain[position:position + 600].split()[:23])
    if name == "site":
        record["scripts"] = re.findall(r'<script[^>]+src="([^"]+)"', text)
        record["features"] = {"cpu_output": "CPU 執行結果" in plain or "output-label" in text,
            "colab": "colab.research.google.com" in text, "download": ".ipynb" in text,
            "zoom": "diagram-zoom" in text, "search": "search" in text}
    if name in ("site_info", "hf_v2") and process.returncode == 0:
        record["response"] = json.loads(text)
    return record


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(obtain, SOURCES))
    output = HERE / "network-output.json"
    output.write_text(json.dumps({"sources": records}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "results": [{k:v for k,v in r.items() if k not in ("response", "command")} for r in records]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
