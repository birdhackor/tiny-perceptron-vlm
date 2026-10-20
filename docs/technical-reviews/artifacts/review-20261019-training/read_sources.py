"""以 curl 讀取原始文件；只保存短摘錄及取得收據。"""

import concurrent.futures
import datetime
import hashlib
import html
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(__file__).resolve().parent
SPECS = [
    ("venv", "https://docs.python.org/3.13/library/venv.html", ["Scripts", "bin"]),
    ("autograd", "https://docs.pytorch.org/docs/2.14/notes/autograd.html", ["backward", "gradient"]),
    ("serialization", "https://docs.pytorch.org/docs/2.14/notes/serialization.html", ["weights_only=True", "arbitrary", "denial"]),
    ("sdpa", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html", ["three implementations", "automatically", "FlashAttention"]),
    ("cuda", "https://docs.pytorch.org/docs/2.14/notes/cuda.html", ["memory_allocated", "synchronize", "nvidia-smi"]),
    ("groups", "https://scikit-learn.org/stable/modules/cross_validation.html", ["same group", "test set", "GroupKFold"]),
    ("rms", "https://arxiv.org/html/1910.07467v1", ["re-centering", "RMSNorm", "RMS("]),
    ("moe", "https://arxiv.org/html/1701.06538v1", ["top-k", "gating network", "small number"]),
    ("dpo", "https://arxiv.org/html/2305.18290v3", ["Deriving the DPO Objective", "reference model", "logsigmoid"]),
    ("distill", "https://arxiv.org/html/1503.02531v1", ["soft targets", "temperature", "correct labels"]),
    ("flash", "https://arxiv.org/html/2205.14135v2", ["tiling", "HBM", "SRAM"]),
    ("kl", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.kl_div.html", ["Kullback", "target", "reduction"]),
    ("qat", "https://raw.githubusercontent.com/pytorch/ao/main/torchao/quantization/qat/api.py", ["fake quantization", "convert", "quantization-aware"]),
    ("gsm8k", "https://raw.githubusercontent.com/openai/grade-school-math/master/README.md", ["grade school", "8.5K"]),
    ("pku", "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/raw/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md", ["license:", "safer_response_id", "is_response_0_safe"]),
]


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, value):
        if not self.skip:
            self.text.append(value)


def fetch(spec):
    identifier, url, needles = spec
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    run = subprocess.run(command, capture_output=True)
    receipt = {"id": identifier, "url": url, "accessed_on": datetime.datetime.now(datetime.UTC).date().isoformat(), "command": command, "returncode": run.returncode}
    if run.returncode:
        receipt["error"] = run.stderr.decode()
        return receipt, ""
    receipt["original_sha256"] = hashlib.sha256(run.stdout).hexdigest()
    if run.stdout.startswith(b"%PDF"):
        conversion = subprocess.run(["pdftotext", "-", "-"], input=run.stdout, capture_output=True, check=True)
        text = conversion.stdout.decode()
        receipt["version"] = url.rsplit("/", 1)[-1]
    else:
        source = run.stdout.decode()
        parser = Plain()
        parser.feed(source)
        text = " ".join(parser.text) if "<html" in source.lower() or "<!doctype" in source.lower() else source
        matches = re.findall(r"(?:PyTorch|scikit-learn)[^\n<>]{0,70}", text)
        receipt["version_markers"] = matches[-5:]
        receipt["version"] = "arXiv " + url.rsplit("/", 1)[-1] if "arxiv.org" in url else "PyTorch 2.14" if "/2.14/" in url else "讀取日期及 URL 指定版本；原文 SHA-256 已記錄"
    text = html.unescape(re.sub(r"\s+", " ", text))
    windows = []
    for needle in needles:
        index = text.casefold().find(needle.casefold())
        later = text.casefold().find(needle.casefold(), max(index + 1, 3000))
        if index < 3000 and later >= 0:
            index = later
        if index >= 0:
            windows.append({"needle": needle, "text": text[max(0, index - 260):index + 1100]})
    receipt["located"] = [window["needle"] for window in windows]
    first = text.casefold().find(needles[0].casefold())
    later = text.casefold().find(needles[0].casefold(), max(first + 1, 3000))
    if first < 3000 and later >= 0:
        first = later
    receipt["excerpt"] = " ".join(text[first:first + 500].split()[:24]) if first >= 0 else " ".join(text.split()[:24])
    return receipt, json.dumps({"id": identifier, "windows": windows, "version_markers": receipt.get("version_markers", [])}, ensure_ascii=False)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        collected = list(pool.map(fetch, SPECS))
    (BASE / "original-sources.json").write_text(json.dumps([r for r, _ in collected], ensure_ascii=False, indent=2) + "\n")
    for receipt, windows in collected:
        print(windows or json.dumps(receipt, ensure_ascii=False))
