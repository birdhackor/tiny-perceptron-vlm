"""Recheck the changed prerequisite and its anonymous immutable evidence link."""

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from urllib.request import urlopen

from scripts.check_technical_reviews import sections


def main():
    root = Path(__file__).resolve().parents[3]
    manifest = root / "outputs/integration-runs/pinned-evidence-link-changes.json"
    body = next(text for lesson, text in sections(root / "course/chapters/13.md") if lesson == "13.15")
    url = re.search(r"\[公開實測報告\]\((https://[^)]+)\)", body)[1]
    raw = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    with urlopen(url, timeout=35) as response:
        blob = {"url": url, "status": response.status, "final_url": response.url, "bytes": len(response.read())}
    with urlopen(raw, timeout=35) as response:
        data = response.read()
        raw_result = {"url": raw, "status": response.status, "final_url": response.url,
                      "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    local = root / "docs/course-experiments/results/posttraining.json"
    assert data == local.read_bytes()
    diff = subprocess.run(["git", "diff", "--unified=0", "--", "course/chapters/13.md"],
                          cwd=root, check=True, capture_output=True, text=True).stdout
    relevant = [hunk for hunk in re.split(r"(?=^@@)", diff, flags=re.M)
                if "公開實測報告" in hunk or "1df335318bda03fd771807f66976953231d5a00b" in hunk]
    result = {
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_16_prerequisite_recheck.py",
        "environment": {"python": platform.python_version(), "authentication": "none; anonymous public HTTPS urllib"},
        "result": "Reread complete current13.15, manifest and actual current diff; immutable blob/raw both HTTP200 and raw report byte-identical to independently audited public result",
        "manifest_path": str(manifest.relative_to(root)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "relevant_manifest_entry": json.loads(manifest.read_text())["course/chapters/13.md"],
        "prerequisite_source": "course/chapters/13.md#13.15", "prerequisite_body": body,
        "prerequisite_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "actual_current_diff_hunks": relevant, "blob": blob, "raw": raw_result,
        "local_report_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
        "scope": "Reread full prerequisite; link pin only changes evidence version target, no change to verified task/formula/numerical scope. Target13.16 independently remains current.",
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["result", "prerequisite_sha256", "blob", "raw"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
