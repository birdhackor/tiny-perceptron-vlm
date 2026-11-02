"""在讀者審閱之前格式化指定 Markdown 的 Python 範例，保持正文與 Notebook 同步。"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = re.compile(r"(^```python[ \t]*\n)(.*?)(^```[ \t]*$)", re.M | re.S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="要格式化的 Markdown 來源")
    args = parser.parse_args()
    ruff = Path(sys.executable).with_name("ruff.exe" if os.name == "nt" else "ruff")
    executable = str(ruff) if ruff.is_file() else shutil.which("ruff")
    if not executable:
        raise RuntimeError("請先安裝專案的 dev 工具，再格式化教材範例")
    changed, blocks = 0, 0
    for source in args.sources:
        original = source.read_text(encoding="utf-8")

        def format_block(match):
            nonlocal blocks
            blocks += 1
            result = subprocess.run(
                [executable, "format", "--stdin-filename", str(ROOT / "lesson.py"), "-"],
                input=match[2],
                text=True,
                capture_output=True,
                check=True,
                cwd=ROOT,
            )
            return match[1] + result.stdout + match[3]

        formatted = BLOCKS.sub(format_block, original)
        if formatted != original:
            source.write_text(formatted, encoding="utf-8")
            changed += 1
    print(f"{blocks} 段 Python；{changed} 份正文格式已更新（更新後需要讀者核對目前版本）")


if __name__ == "__main__":
    main()
