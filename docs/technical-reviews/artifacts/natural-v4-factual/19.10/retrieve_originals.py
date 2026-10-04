"""Retrieve only original papers; full text remains in ignored research."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import urllib.error
import urllib.request

research = Path("outputs/natural-v4/factual-research/19.10")
artifact = Path("docs/technical-reviews/artifacts/natural-v4-factual/19.10")
research.mkdir(parents=True, exist_ok=True)
receipts = []
for version in ["1503.02531v1", "2106.08295v2", "2106.08295v1", "2401.04088v1", "1910.07467v1"]:
    url = "https://arxiv.org/pdf/" + version
    path = research / (version + ".pdf")
    receipt = {"url": url, "version_requested": version,
               "retrieval_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        if path.exists():
            data = path.read_bytes()
            receipt["retrieval"] = "Previously retrieved by this same fresh reviewer in this task; rehashed here."
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "Original-source factual review"})
            with urllib.request.urlopen(request, timeout=40) as response:
                data = response.read()
                receipt["response_url"] = response.url
            path.write_bytes(data)
            receipt["retrieval"] = "Actual HTTPS retrieval in this task."
        subprocess.run(["pdftotext", "-layout", str(path), str(research / (version + ".txt"))], check=True)
        receipt.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                       ignored_original_path=str(path), outcome="success")
    except urllib.error.HTTPError as error:
        receipt.update(outcome="HTTP error", http_status=error.code, details=str(error))
    receipts.append(receipt)
    print(json.dumps(receipt))
artifact.joinpath("original-retrieval-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
