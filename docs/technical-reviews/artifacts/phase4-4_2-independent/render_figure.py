"""Render the currently referenced SVG, recording the actual tools and limits."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
SOURCE = REPO / "course/figures/rewrite-04-residual.svg"
records = []
commands = [
    ["chromium", "--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--no-proxy-server", "--hide-scrollbars", "--window-size=640,735", f"--user-data-dir={OUT / 'chromium-profile'}", f"--screenshot={OUT / 'figure-chromium.png'}", SOURCE.as_uri()],
    ["inkscape", str(SOURCE), "--export-type=png", f"--export-filename={OUT / 'figure-inkscape.png'}"],
]
for tool in ("chromium", "inkscape"):
    completed = subprocess.run([tool, "--version"], capture_output=True, timeout=10)
    (OUT / f"{tool}-version.txt").write_bytes(completed.stdout + completed.stderr)
for command in commands:
    name = command[0]
    started = time.monotonic()
    record = {"command_argv": command, "cwd": str(REPO), "timeout_seconds": 20}
    with (OUT / f"render-{name}.stdout.txt").open("wb") as stdout, (OUT / f"render-{name}.stderr.txt").open("wb") as stderr:
        try:
            completed = subprocess.run(command, cwd=REPO, stdout=stdout, stderr=stderr, timeout=20)
            record["exit_code"] = completed.returncode
        except subprocess.TimeoutExpired:
            record.update(exit_code=124, timed_out=True)
    record["elapsed_seconds"] = time.monotonic() - started
    records.append(record)
receipt = {"source": SOURCE.relative_to(REPO).as_posix(), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "runs": records,
           "renders": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in OUT.glob("figure-*.png")]}
(OUT / "render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
