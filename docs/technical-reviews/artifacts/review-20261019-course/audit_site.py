"""只讀取公開網站與下載工作頁，比對每個程式格的 CPU 輸出區塊。"""

import hashlib
import json
import platform
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = "https://birdhackor.github.io/tiny-perceptron-vlm/"


def fetch(relative):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "35", BASE + relative]
    process = subprocess.run(command, capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.decode())
    return process.stdout


def check(item):
    try:
        raw = fetch(item["id"] + ".html")
        notebook_raw = fetch(item["notebook"])
        notebook = json.loads(notebook_raw)
        expected = sum(c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {}) for c in notebook["cells"])
        observed = len(re.findall("實際執行結果 · CPU", raw.decode()))
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
        return {"id": item["id"], "html_sha256": hashlib.sha256(raw).hexdigest(), "notebook_sha256": hashlib.sha256(notebook_raw).hexdigest(), "code_cells": expected, "cpu_output_blocks": observed, "matches": expected == observed,
                "section_has_output": expected == 0 or observed > 0, "code_when_counts_differ": code if expected != observed else []}
    except Exception as error:
        return {"id": item["id"], "error": str(error), "matches": False}


index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
before = json.loads(fetch("build-info.json"))
with ThreadPoolExecutor(max_workers=12) as executor:
    pages = list(executor.map(check, index))
after = json.loads(fetch("build-info.json"))
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-course/audit_site.py",
    "environment": {"python": platform.python_version(), "device": "remote static HTTPS", "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]},
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "request_pattern": "curl --fail --location --silent --show-error --max-time 35 https://birdhackor.github.io/tiny-perceptron-vlm/<頁面或工作頁相對路徑>",
    "build_info_before": before, "build_info_after": after,
    "same_build": before == after,
    "page_count": len(pages), "all_cell_counts_match": all(x["matches"] for x in pages),
    "all_sections_with_code_have_cpu_output": all(x.get("section_has_output", False) for x in pages),
    "code_cells": sum(x.get("code_cells", 0) for x in pages), "pages": pages,
    "scope": "核對公開網站目前版本及其自己的 Notebook，不把它當作尚未發布凍結教材的執行證據。CPU 執行的來源另核對發布 workflow 與匯出器。沒有 print 或展示值的設定程式不應憑空產生輸出；格數與輸出區塊數不同時保留程式供檢視。未保存 HTML 或工作頁的全文副本。"
}, ensure_ascii=False, indent=2))
