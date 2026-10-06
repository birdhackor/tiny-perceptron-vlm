"""Run publication fixtures with Windows newline and Git autocrlf defaults.

This reproduces byte normalization locally; it is not an actual Windows run.
Explicit newline choices in the tests remain effective, just as on Windows.
"""

import argparse
import hashlib
import os
import platform
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
TEST = ROOT / "tests/test_selftrained_data_publish.py"
PUBLISHER = ROOT / "scripts/selftrained/publish_data.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autocrlf", choices=("true", "false"), default="true")
    parser.add_argument("--native-newline", action="store_true")
    args = parser.parse_args()
    print(f"Actual platform: {platform.platform()}")
    newline_setting = "native" if args.native_newline else "CRLF"
    print(f"Settings: default text newline={newline_setting}; command-scoped core.autocrlf={args.autocrlf}")
    print(f"Python: {sys.version}")
    for path in (TEST, PUBLISHER):
        print(f"Source SHA-256 {path.relative_to(ROOT)}: {hashlib.sha256(path.read_bytes()).hexdigest()}")

    # The subprocesses use a command-scoped config instead of changing the
    # user's global Git configuration. Preserve any existing scoped entries.
    config_index = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    os.environ[f"GIT_CONFIG_KEY_{config_index}"] = "core.autocrlf"
    os.environ[f"GIT_CONFIG_VALUE_{config_index}"] = args.autocrlf
    os.environ["GIT_CONFIG_COUNT"] = str(config_index + 1)

    original_write_text = Path.write_text

    def windows_write_text(path, data, encoding=None, errors=None, newline=None):
        return original_write_text(
            path,
            data,
            encoding=encoding,
            errors=errors,
            newline="\r\n" if newline is None else newline,
        )

    if not args.native_newline:
        Path.write_text = windows_write_text
    try:
        return pytest.main(["-q", "--tb=short", str(TEST)])
    finally:
        Path.write_text = original_write_text


if __name__ == "__main__":
    raise SystemExit(main())
