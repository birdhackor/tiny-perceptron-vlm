"""新正式工作共用的有界匯出；不建立遠端資源。"""

import hashlib
import json
import time
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def export_and_wait(output, summary, job):
    """只有自己的新結果；外部權重／第三方全文不放進輸出封包。"""
    output = Path(output)
    files = [path for path in output.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in files) or sum(path.stat().st_size for path in files) > 256 * 1024**2:
        raise ValueError("結果來源超出匯出界或含 symlink")
    json_file(
        output / "file-manifest.json",
        {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(files)},
    )
    destination = Path("/tmp/omni-result.zip")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    if destination.stat().st_size > 128 * 1024**2:
        raise ValueError("結果封包超出事前128MiB上限")
    json_file(
        "/tmp/omni-result.json",
        {
            "job": job["job"],
            "success": summary["success"],
            "bytes": destination.stat().st_size,
            "sha256": sha(destination),
            "final_opened": summary.get("final_opened", False),
            "config_sha256": sha("job.json"),
            "core_selection_sha256": job["core_selection_sha256"],
            "core_final_opening_sha256": job["core_final_opening_sha256"],
        },
    )
    print("OMNI_JOB_READY", flush=True)
    while True:
        time.sleep(10)
