"""文字 PR 跑 Linux；執行程式或共用輸入改動才加跑 Windows 與 macOS。"""

import json
import os
import subprocess
from pathlib import Path

if __package__:
    from .course_execution import code_signature
else:
    from course_execution import code_signature


def runtime_changed(names, notebook_changed):
    for name in names:
        if name.endswith(".ipynb"):
            if notebook_changed(name):
                return True
        elif name.endswith((".py", ".yml", ".yaml", ".lock")) or name in {
            "pyproject.toml",
            "uv.toml",
            ".python-version",
            ".gitattributes",
        }:
            return True
        elif name.startswith(("assets/", "checkpoints/", "tiny_perceptron/", "course/figures/")):
            return True
        elif name.startswith("docs/") and not name.endswith(".md"):
            return True
    return False


def main():
    base = os.environ.get("BASE_SHA") or "HEAD^"
    full = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    try:
        names = subprocess.check_output(["git", "diff", "--name-only", "-z", base, "HEAD"]).decode().split("\0")

        def changed(name):
            old = subprocess.check_output(["git", "show", f"{base}:{name}"], stderr=subprocess.DEVNULL)
            return code_signature(json.loads(old)) != code_signature(json.loads(Path(name).read_bytes()))

        full = full or runtime_changed(names, changed)
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError):
        full = True
    matrix = [{"os": "ubuntu-latest", "extra": "--extra cpu"}]
    if full:
        matrix += [{"os": "windows-latest", "extra": "--extra cpu"}, {"os": "macos-latest", "extra": ""}]
    print(json.dumps(matrix))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write("matrix=" + json.dumps(matrix) + "\n")
        stream.write("full=" + str(full).lower() + "\n")


if __name__ == "__main__":
    main()
