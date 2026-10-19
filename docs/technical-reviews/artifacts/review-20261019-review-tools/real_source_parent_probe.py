"""在真實 clone 核對追蹤檔的來源父路徑是否也拒絕外部 symlink。"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from independent_replay_probe import canonical, digest, load_module, state


def worker(root):
    module = load_module(root / "docs/review-tools/baseline_replay.py")
    outside = root.parent / "outside-tests"
    original_link = module.os.link
    copied = []

    def link(source, destination, *args, **kwargs):
        source = Path(source)
        if source.resolve().is_relative_to(outside):
            copied.append(source.relative_to(root).as_posix())
        return original_link(source, destination, *args, **kwargs)

    module.os.link = link
    try:
        supplements = json.loads((root / "docs/course-review-20261019/baseline-supplements.json").read_bytes())["files"]
        result = module.replay(root, supplements)
    finally:
        module.os.link = original_link
    print(json.dumps({"result": result, "outside_file_count": len(copied), "outside_paths": copied}, ensure_ascii=False))
    return bool(result["failures"])


def main(root):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    before = state(root, environment)
    tests = root / "tests"
    outside = root.parent / "outside-tests"
    tests.rename(outside)
    tests.symlink_to(outside, target_is_directory=True)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--root", str(root)]
    try:
        done = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    finally:
        tests.unlink()
        outside.rename(tests)
    after = state(root, environment)
    print(json.dumps({"command": command, "exit_code": done.returncode, "stdout": done.stdout,
                      "stderr": done.stderr, "before_sha256": digest(canonical(before)),
                      "after_sha256": digest(canonical(after)), "tracked_files_unchanged": before == after,
                      "setup": "把 clone 的 tests 移到 clone 外的同級目錄，再讓 tests symlink 指向它；結束後還原。"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["worker", "controller"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "worker":
        raise SystemExit(worker(args.root.resolve()))
    main(args.root.resolve())
