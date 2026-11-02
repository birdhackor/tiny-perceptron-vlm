"""T.11 fresh reviewer: bounded inference, immutable identities, original source excerpts.

No training, no final evaluation and no publication entry points are called.
"""

import hashlib
import html.parser
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Plain(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def fetch_originals():
    requests = [
        ("python-hashlib", "https://docs.python.org/3.12/library/hashlib.html", ["sha256", "digest", "collision"]),
        ("git-glossary", "https://git-scm.com/docs/gitglossary", ["commit object", "snapshot", "object name"]),
        ("sklearn-cross-validation", "https://scikit-learn.org/1.7/modules/cross_validation.html", ["test set", "validation set", "hyperparameters"]),
        ("torch-checkpoint", "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html", ["general checkpoint", "optimizer", "resuming training"]),
        ("neurips-checklist", "https://neurips.cc/public/guides/PaperChecklist", ["reproducibility", "training", "experimental"]),
    ]
    result = []
    for name, url, terms in requests:
        row = {"id": name, "url": url, "accessed_at": datetime.now(UTC).isoformat()}
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read()
                row.update(status=response.status, final_url=response.url, original_sha256=sha(raw), original_bytes=len(raw))
            parser = Plain()
            parser.feed(raw.decode("utf-8"))
            text = " ".join(parser.parts)
            excerpts = []
            for term in terms:
                start = 0
                for _ in range(2):
                    at = text.lower().find(term.lower(), start)
                    if at < 0:
                        break
                    excerpts.append({"term": term, "excerpt": text[max(0, at - 180):at + 850]})
                    start = at + len(term)
            row.update(excerpts=excerpts)
        except Exception as error:
            row.update(error=f"{type(error).__name__}: {error}")
        result.append(row)
    (OUT / "original-source-excerpts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"original_sources": [{k: v for k, v in row.items() if k in {"id", "status", "error", "original_sha256"}} for row in result]}), flush=True)


def execute():
    import torch

    report = {
        "reviewer_task": "/root/c_v3_technical_coord/numbered_t11",
        "started_at": datetime.now(UTC).isoformat(),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu", "uv": subprocess.check_output(["uv", "--version"], text=True).strip()},
        "scope": "New executions of downloads and single CPU demonstrations only. Preserved final data are read, never re-evaluated. No training or publication.",
        "commands": [],
    }
    operations = [
        ["-m", "scripts.course_product", "download"],
        ["-m", "scripts.course_product", "judge"],
        ["-m", "scripts.omni_revision.public_download", "--output", "checkpoints/omni-products-v1"],
        ["-m", "scripts.omni_revision.public_demo", "av", "--model-root", "checkpoints/omni-products-v1"],
        ["-m", "scripts.omni_revision.public_demo", "av", "--model-root", "outputs/omni-P5-public-anonymous-guide-v1/fresh-model-root"],
    ]
    for operation in operations:
        argv = ["uv", "run", "--extra", "cpu", "--no-sync", "python", *operation]
        row = {"command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 " + " ".join(argv), "started_at": datetime.now(UTC).isoformat()}
        begin = time.monotonic()
        try:
            completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=75, env=os.environ.copy())
            row.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        except subprocess.TimeoutExpired as error:
            row.update(returncode="timeout", stdout=str(error.stdout), stderr=str(error.stderr))
        row.update(elapsed_seconds=time.monotonic() - begin, ended_at=datetime.now(UTC).isoformat())
        report["commands"].append(row)
        (OUT / "cpu-commands.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"command": operation, "returncode": row["returncode"], "stdout": row.get("stdout", "")[:1200], "stderr": row.get("stderr", "")[-600:]}, ensure_ascii=False), flush=True)
    fetch_originals()


if __name__ == "__main__":
    execute()
