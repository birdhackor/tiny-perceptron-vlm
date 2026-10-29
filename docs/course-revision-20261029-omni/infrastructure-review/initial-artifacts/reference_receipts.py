"""Fetch public primary sources only; keep external full texts outside Git."""

import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

TARGET = Path(__file__).resolve().parent
CACHE = Path("/tmp/omni-infra-review-external-20261029")
CACHE.mkdir(exist_ok=True)
URLS = {
    "modal-pypi-1.6.0": "https://pypi.org/pypi/modal/1.6.0/json",
    "hf-pypi-1.33.0": "https://pypi.org/pypi/huggingface_hub/1.33.0/json",
    "modal-timeouts": "https://modal.com/docs/guide/timeouts",
    "modal-preemption": "https://modal.com/docs/guide/preemption",
    "modal-resources": "https://modal.com/docs/guide/resources",
    "modal-billing": "https://modal.com/docs/guide/billing",
    "gh-contents": "https://docs.github.com/en/rest/repos/contents",
    "gh-concurrency": "https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency",
    "gh-dispatch": "https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows",
    "gh-live-main-workflow": "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/main/.github/workflows/omni-revision.yml",
}
receipts = []
for name, url in URLS.items():
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw, code, final_url = response.read(5 * 1024 * 1024), response.status, response.url
    except urllib.error.HTTPError as error:
        raw, code, final_url = error.read(5 * 1024 * 1024), error.code, error.url
    path = CACHE / name
    path.write_bytes(raw)
    receipt = dict(name=name, url=url, final_url=final_url, http_status=code,
                   retrieved_at=datetime.now(UTC).isoformat(), bytes=len(raw),
                   sha256=hashlib.sha256(raw).hexdigest(), local_full_text=str(path))
    if "pypi" in name and code == 200:
        metadata = json.loads(raw)
        receipt["distribution_version"] = metadata["info"]["version"]
        receipt["distributions"] = [
            {"filename": item["filename"], "sha256": item["digests"]["sha256"], "url": item["url"]}
            for item in metadata["urls"]
        ]
    receipts.append(receipt)

MODAL = Path("/tmp/omni-review-modal-1.6.0")
HF = Path("/workspace/tiny-perceptron-vlm/.venv/lib/python3.13/site-packages")
for root, names, version in [
    (MODAL, ["modal/app.py", "modal/volume.py", "modal/_workspace.py", "modal/_image.py",
             "modal/runner.py", "modal/cli/app.py", "modal/_container_entrypoint.py"], "modal==1.6.0"),
    (HF, ["huggingface_hub/hf_api.py", "huggingface_hub/_commit_api.py", "huggingface_hub/lfs.py"],
     "huggingface-hub==1.33.0"),
]:
    for name in names:
        raw = (root / name).read_bytes()
        receipts.append(dict(distribution=version, path=str(root / name), bytes=len(raw),
                             sha256=hashlib.sha256(raw).hexdigest()))
(TARGET / "reference-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps({"receipts": len(receipts), "public_fetches": len(URLS),
                  "statuses": {r["name"]: r["http_status"] for r in receipts if "name" in r}}, indent=2))
