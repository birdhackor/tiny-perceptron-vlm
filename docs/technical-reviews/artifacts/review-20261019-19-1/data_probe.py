"""在記憶體讀取固定資料包，只保留題型與字卡格式的核對結果。"""

import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]


def main():
    manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_bytes())
    url = "https://github.com/birdhackor/tiny-perceptron-vlm/raw/refs/tags/selftrained-v2/assets/training/selftrained-v2.tar.gz"
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    process = subprocess.run(command, capture_output=True, check=True)
    raw = process.stdout
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == manifest["package"]["sha256"]
    expected = {entry["path"]: entry["sha256"] for entry in manifest["records"]}
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-19-1/data_probe.py",
              "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu"},
              "download": {"command": command, "exit_code": process.returncode, "bytes": len(raw), "sha256": digest}, "records": {}}
    rows = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name not in expected:
                continue
            contents = archive.extractfile(member).read()
            assert hashlib.sha256(contents).hexdigest() == expected[name]
            entries = [json.loads(line) for line in contents.splitlines() if line.strip()]
            rows.extend(entries)
            result["records"][name] = {"sha256": expected[name], "rows": len(entries)}
    assert set(result["records"]) == set(expected)
    result["task_counts"] = dict(Counter(row["task"] for row in rows))
    ocr = [row for row in rows if row["task"] == "ocr"]
    result["ocr"] = {
        "rows": len(ocr), "lengths": dict(Counter(len(row["supervision"]["ocr_text"]) for row in ocr)),
        "target_with_newline": sum("\n" in row["messages"][-1]["content"] for row in ocr),
        "target_differs_from_plain_ocr_text": sum(row["messages"][-1]["content"] != row["supervision"]["ocr_text"] for row in ocr),
        "newline_in_user_request": sum(any("換行" in m["content"] or "每行" in m["content"] for m in row["messages"] if m["role"] == "user") for row in ocr),
        "train_font_families": dict(Counter(row["supervision"]["font_family"] for row in ocr if row["split"] == "train")),
        "example": {k: ocr[0][k] for k in ["id", "split", "messages", "roi"]},
    }
    text = [row for row in rows if row["task"] == "text"]
    result["text"] = {"rows": len(text), "intents": dict(Counter(row["supervision"]["intent"] for row in text)),
                       "matching_user_messages": []}
    for row in text:
        for message in row["messages"]:
            if message["role"] == "user" and any(term in message["content"] for term in ["清單", "列表", "列出", "前兩", "第一個"]):
                result["text"]["matching_user_messages"].append({"id": row["id"], "message": message["content"]})
    result["text"]["matching_user_messages"] = result["text"]["matching_user_messages"][:30]
    path = ART / "data-probe.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ocr": result["ocr"], "text": result["text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
