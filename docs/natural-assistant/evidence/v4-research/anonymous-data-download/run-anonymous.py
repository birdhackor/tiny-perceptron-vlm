"""Run the actual fixed-pin data list/fetch/verify CLI in a new anonymous destination."""

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = Path(__file__).resolve().parent
REVISION = "9a61ecf524c9518f33f1501c28aa72997d4a82d0"
MANIFEST_SHA = "0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60"
DESTINATION = ROOT / "outputs/natural-v4/anonymous-data-download/data"


def fingerprint(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    if DESTINATION.exists() or DESTINATION.is_symlink():
        raise FileExistsError("Fresh anonymous verification requires a previously absent destination")
    script = ROOT / "scripts/fetch_natural_data.py"
    manifest = ROOT / "docs/natural-assistant/v4/manifest.json"
    before = {str(p.relative_to(ROOT)): fingerprint(p) for p in (script, manifest)}
    assert before[str(manifest.relative_to(ROOT))]["sha256"] == MANIFEST_SHA
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip() == REVISION
    assert subprocess.check_output(["git", "show", f"{REVISION}:scripts/fetch_natural_data.py"], cwd=ROOT) == script.read_bytes()
    environment = dict(os.environ)
    removed = sorted(key for key in environment if "TOKEN" in key.upper() or key.upper().endswith(("SECRET", "PASSWORD")))
    for key in removed:
        environment.pop(key)
    environment["PYTHONPATH"] = str(EVIDENCE)
    base = [sys.executable, str(script), "--manifest", str(manifest), "--revision", REVISION,
            "--manifest-sha256", MANIFEST_SHA, "--output", str(DESTINATION)]
    records = []
    started_at = datetime.now(UTC).isoformat()
    for stage, extra in (("list", ["--list"]), ("fetch", []), ("verify", ["--verify"])):
        environment["NATURAL_ANONYMOUS_HTTP_RECEIPT"] = str(EVIDENCE / (stage + ".http.json"))
        begin = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        result = subprocess.run(base + extra, cwd=ROOT, env=environment, capture_output=True, text=True)
        finish = resource.getrusage(resource.RUSAGE_CHILDREN)
        (EVIDENCE / (stage + ".stdout.json")).write_text(result.stdout)
        (EVIDENCE / (stage + ".stderr.txt")).write_text(result.stderr)
        record = {"stage": stage, "command": base + extra, "exit_code": result.returncode,
                  "elapsed_seconds": time.monotonic() - start, "CPU_user_seconds": finish.ru_utime - begin.ru_utime,
                  "CPU_system_seconds": finish.ru_stime - begin.ru_stime, "child_max_RSS_KiB": finish.ru_maxrss}
        if result.returncode == 0:
            record["result"] = json.loads(result.stdout)
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key not in ("command", "result")}), flush=True)
        if result.returncode:
            break
    after = {str(p.relative_to(ROOT)): fingerprint(p) for p in (script, manifest)}
    report = {"scope": "Actual fresh anonymous public-media v4 CLI list/fetch/verify; no existing image/audio cache reuse",
              "started_at_UTC": started_at, "finished_at_UTC": datetime.now(UTC).isoformat(),
              "revision": REVISION, "manifest_sha256": MANIFEST_SHA, "destination": str(DESTINATION),
              "destination_absent_before_first_command": True, "source_before": before, "source_after": after,
              "script_and_manifest_bytes_stable": before == after, "credential_env_variable_names_removed": removed,
              "credential_values_logged": False, "HTTP_observer": "sitecustomize urllib.Request audit hook only; no mocks/opener substitutions",
              "stages": records, "all_three_commands_passed": len(records) == 3 and all(r["exit_code"] == 0 for r in records),
              "GPU_started": False, "model_generation_or_training": False}
    (EVIDENCE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for path in (script, manifest):
        if path == script:
            (EVIDENCE / "fetch_natural_data.py.source.txt").write_bytes(path.read_bytes())
    if not report["all_three_commands_passed"] or before != after:
        raise RuntimeError("Fresh anonymous CLI verification did not complete; inspect preserved real logs")
    fetch = records[1]["result"]
    assert fetch["status"] == "downloaded_verified" and fetch["files_verified"] == 1513
    assert records[2]["result"]["status"] == "verified"
    assert all(item["authentication"] == "none" for item in [fetch["remote_manifest"], *fetch["transport"]])
    print(json.dumps({"fresh_anonymous_download_success": True, "three_archives_downloaded_bytes": fetch["download_bytes"],
                      "files_verified": fetch["files_verified"], "unpacked_bytes": fetch["unpacked_bytes"], "revision": REVISION,
                      "manifest_sha256": MANIFEST_SHA}, indent=2), flush=True)


if __name__ == "__main__":
    main()
