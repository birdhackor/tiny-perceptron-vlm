"""Bounded figure rendering; keep actual commands and renderer limitations."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "course/figures/rewrite-01-chain.svg"
records = []


def run(name, command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code = error.stdout or b"", error.stderr or b"", 124
        timed_out = True
    (HERE / f"{name}.stdout.txt").write_bytes(stdout)
    (HERE / f"{name}.stderr.txt").write_bytes(stderr)
    record = {"name": name, "command_argv": command, "timeout_seconds": timeout,
              "exit_code": code, "timed_out": timed_out}
    records.append(record)
    return code


browser = HERE / "chain-chromium.png"
run("chromium", ["chromium", "--headless", "--no-sandbox", "--disable-gpu",
    "--disable-dev-shm-usage", "--no-first-run", "--hide-scrollbars",
    "--user-data-dir=/tmp/phase4-1_10-chromium-profile", "--window-size=640,795",
    "--screenshot=" + str(browser), SOURCE.as_uri()], 15)
png = HERE / "chain-inkscape.png"
code = run("inkscape", ["inkscape", str(SOURCE), "--export-type=png",
    "--export-filename=" + str(png), "--export-width=640"], 20)
if code:
    raise SystemExit(code)
version = subprocess.check_output(["inkscape", "--version"], text=True).strip()
receipt = {"source": str(SOURCE.relative_to(ROOT)),
           "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           "inkscape_version": version, "render_commands": records,
           "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
           "browser_png_exists": browser.is_file()}
(HERE / "figure-render.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
