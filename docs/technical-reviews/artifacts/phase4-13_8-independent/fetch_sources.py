from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, UTC
import hashlib
import json

OUT = Path(__file__).resolve().parent
SOURCES = {
    "sycophancy-v4-direct.pdf": "https://arxiv.org/pdf/2310.13548v4",
    "sycophancy-v4-abstract.html": "https://arxiv.org/abs/2310.13548v4",
    "model-spec-2025-12-18.html": "https://model-spec.openai.com/2025-12-18.html",
}
records = []
for name, url in SOURCES.items():
    record = {"url": url, "accessed_at": datetime.now(UTC).isoformat()}
    try:
        with urlopen(Request(url, headers={"User-Agent": "Independent technical review source verification"}), timeout=30) as response:
            data = response.read(12 * 1024 * 1024)
            record.update(status=response.status, final_url=response.url,
                          content_type=response.headers.get("Content-Type"), bytes=len(data))
        destination = OUT / "sources" / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(data)
        record.update(path=str(destination.relative_to(OUT)), sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    records.append(record)
    print(json.dumps(record, ensure_ascii=False))
(OUT / "source-fetch.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
