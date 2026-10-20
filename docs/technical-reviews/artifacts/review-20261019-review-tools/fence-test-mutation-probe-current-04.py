"""在全新副本中確認圍欄測試能抓到移除行首檢查或移除雙解析狀態的錯法。"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main(args):
    repo, area = args.repo.resolve(), args.area.resolve()
    area.mkdir()
    source = (repo / "docs/review-tools/incremental_revision.py").read_text()
    needle = "line[1] == opener[1] and line[2] == opener[2]"
    union = "return any(inside_fence(text, position, closes) for closes in (closes_superfences, closes_commonmark))"
    assert source.count(needle) == source.count(union) == 1
    variants = {
        "control": source,
        "skip_prefix": source.replace(needle, "line[2] == opener[2]"),
        "super_only": source.replace(union, "return inside_fence(text, position, closes_superfences)"),
    }
    rows = []
    for name, content in variants.items():
        root = area / name
        for path in ("docs/review-tools/incremental_revision.py", "docs/review-tools/baseline_replay.py", "tests/test_incremental_revision_review.py"):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = content.encode() if path.endswith("incremental_revision.py") else (repo / path).read_bytes()
            with target.open("xb") as stream:
                stream.write(raw)
        temp = root / "test-temp"
        temp.mkdir()
        cmd = [str(repo / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider", str(root / "tests/test_incremental_revision_review.py"),
               "-k", "test_new_text_inside_a_fenced_code_block_is_rejected or test_new_text_after_a_closed_fenced_block_is_undone"]
        result = subprocess.run(cmd, cwd=root, text=True, capture_output=True,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(temp)))
        rows.append({"case": name, "command": cmd, "cwd": str(root), "exit_code": result.returncode,
                     "source_sha256": hashlib.sha256(content.encode()).hexdigest(), "stdout": result.stdout, "stderr": result.stderr})
    print(json.dumps({"environment": {"python": sys.version, "platform": platform.platform()}, "cases": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--area", type=Path, required=True)
    main(parser.parse_args())
