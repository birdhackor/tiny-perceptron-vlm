"""在全新副本確認測試能抓到移除倒退後檢查或檔尾檢查的錯法。"""
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
    after = '        fence_blocks(text, f"{label}：倒退這一筆之後的 {where}")\n'
    eof = 'require(opened is None, f"{where} 最後一個程式圍欄沒有收尾'
    assert source.count(after) == source.count(eof) == 1
    variants = {
        "control": source,
        "skip_after": source.replace(after, ""),
        "skip_eof": source.replace(eof, 'require(True, f"{where} 最後一個程式圍欄沒有收尾'),
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
               "-k", "plain_fenced_block or fences_are_not_plain or ending_inside_a_fence or reviewed_version_must_have_plain or closed_plain_block"]
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
