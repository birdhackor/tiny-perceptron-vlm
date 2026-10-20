"""讀原始文件及釘選的語音來源，僅保存少量摘錄與結構統計。"""

import hashlib
import io
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq

ARTIFACT = Path(__file__).resolve().parent


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "180", url]
    result = subprocess.run(command, capture_output=True, check=True)
    raw = result.stdout
    return raw, {"command": command, "url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "accessed_on": datetime.now(UTC).date().isoformat()}


def main():
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-3/read_originals.py", "environment": {"python": platform.python_version(), "pyarrow": pyarrow.__version__, "device": "cpu"}, "documents": []}
    specs = [
        ("cross_validation", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst", "1.7.2", ["When evaluating different settings", "Cross-validation iterators for grouped data"]),
        ("ocr_detection", "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.0.0/docs/version3.x/module_usage/text_detection.en.md", "v3.0.0", ["## 1. Overview"]),
        ("minds_card", "https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md", "40ce77cb32a384e4d50a568e1ec39ac804019d33", ["### Data Fields"]),
    ]
    for identifier, url, version, locators in specs:
        raw, receipt = fetch(url)
        text = raw.decode("utf-8")
        receipt.update(id=identifier, version=version, inspected=[])
        for locator in locators:
            start = text.index(locator)
            excerpt = text[start:start + 2300]
            print(json.dumps({"document": identifier, "locator": locator, "original_read": excerpt}, ensure_ascii=False), flush=True)
            receipt["inspected"].append({"locator": locator, "source_line": text[:start].count("\n") + 1, "short_excerpt": " ".join(excerpt.split()[:12])})
        result["documents"].append(receipt)
    url = "https://huggingface.co/datasets/PolyAI/minds14/resolve/40ce77cb32a384e4d50a568e1ec39ac804019d33/zh-CN/train-00000-of-00001.parquet"
    raw, receipt = fetch(url)
    assert hashlib.sha256(raw).hexdigest() == "cea0246e1a54afe5a3ab9d9542baa53c7cb45fc8f4e28e7872ebafefb25ba99d"
    table = pq.read_table(io.BytesIO(raw))
    source = table.select(["path", "intent_class", "transcription"]).to_pylist()
    result["minds_parquet"] = {**receipt, "columns": table.column_names, "rows": len(table), "schema": str(table.schema), "selected_intent_counts": {str(label): sum(r["intent_class"] == label for r in source) for label in [1, 2, 6]}, "no_speaker_or_session_columns": not any("speaker" in col or "session" in col for col in table.column_names)}
    out = ARTIFACT / "original-source-evidence.json"
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2); handle.write("\n")
    print(json.dumps({"result": "completed", "artifact": out.name, "source_columns": table.column_names, "source_rows": len(table)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
