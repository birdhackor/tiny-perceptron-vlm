"""Fetch only the primary sources used for this supplemental review."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import urllib.request

OUT = Path(__file__).resolve().parent
destination = OUT / "primary-sources"
destination.mkdir(exist_ok=True)
targets = [
    ("transformer.pdf", "https://arxiv.org/pdf/1706.03762"),
    ("pytorch-layernorm.html", "https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"),
]
receipts = []
for filename, url in targets:
    record = {"url": url, "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Supplemental educational correctness review"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
            record.update({"http_status": response.status, "final_url": response.url})
        path = destination / filename
        path.write_bytes(data)
        record.update({"downloaded_file": str(path.relative_to(OUT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes":len(data)})
        if filename.endswith(".pdf"):
            text_file = path.with_suffix(".txt")
            process = subprocess.run(["pdftotext", "-layout", str(path), str(text_file)], capture_output=True, text=True)
            record["pdftotext_exit_code"] = process.returncode
            record["pdftotext_stderr"] = process.stderr
            if process.returncode == 0:
                record["text_file"] = str(text_file.relative_to(OUT))
                record["text_sha256"] = hashlib.sha256(text_file.read_bytes()).hexdigest()
        print(filename, "HTTP", record["http_status"],len(data), "bytes")
    except Exception as error:
        record["error"] = repr(error)
        print(filename, repr(error))
    record["ended_at"] = datetime.now(timezone.utc).isoformat()
    receipts.append(record)
(destination / "receipt.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2)+"\n")
