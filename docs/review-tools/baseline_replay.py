"""在暫存硬連結副本重建舊發布版本，原樣執行舊審閱契約。

不改舊manifest、hash或判定。覆蓋檔案先unlink，絕不寫入硬連結原檔。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replay(root=ROOT):
    root = Path(root).resolve()
    batch = root / "docs/course-revision-20261016"
    previous = root / "docs/course-publish-20261008"
    freeze = json.loads((previous / "current-source-freeze.json").read_bytes())
    overlay = json.loads((batch / "baseline-overlay.json").read_bytes())
    entries = [ref for kind in ("source_files", "criteria", "figures", "notebooks") for ref in freeze[kind].values()]
    entries += overlay["files"]
    cache = root / "outputs/review-baseline-replay"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="immutable-baseline-", dir=cache) as folder:
        target = Path(folder)
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        for name in filter(None, names):
            source, destination = root / name, target / name
            if not source.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                destination.symlink_to(os.readlink(source))
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        # 未提交的新sidecar也必須可在新checkout中使用。
        for ref in entries:
            source = root / ref["path"]
            raw = source.read_bytes()
            if digest(raw) != ref["sha256"]:
                raise ValueError("baseline snapshot changed: " + ref["path"])
            destination = target / ref["live_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.write_bytes(raw)
        spec = importlib.util.spec_from_file_location(
            "preserved_composite", root / "docs/review-tools/composite_review.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.audit_manifest(target, target / "docs/course-publish-20261008/review-manifest.json", "all")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = replay()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "failures")}, ensure_ascii=False))
    raise SystemExit(bool(report["failures"]))
