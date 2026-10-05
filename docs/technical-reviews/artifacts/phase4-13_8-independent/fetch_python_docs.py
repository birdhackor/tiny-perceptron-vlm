from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, UTC
import hashlib
import json

OUT = Path(__file__).resolve().parent
records = []
for name, path in [("functions", "library/functions.html"),
                   ("stdtypes", "library/stdtypes.html"),
                   ("expressions", "reference/expressions.html")]:
    url = "https://docs.python.org/release/3.13.5/" + path
    record = {"url": url, "accessed_at": datetime.now(UTC).isoformat()}
    try:
        with urlopen(Request(url, headers={"User-Agent": "Independent technical review"}), timeout=30) as response:
            data = response.read(2 * 1024 * 1024)
            record.update(status=response.status, final_url=response.url, bytes=len(data))
        destination = OUT / "sources" / ("python-3.13.5-" + name + ".html")
        destination.write_bytes(data)
        record.update(path=str(destination.relative_to(OUT)), sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    records.append(record)
    print(json.dumps(record, ensure_ascii=False))
(OUT / "python-doc-fetch.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
