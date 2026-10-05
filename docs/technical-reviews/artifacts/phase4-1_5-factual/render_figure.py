from datetime import UTC, datetime
from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parent
svg = Path("course/figures/foundations_bigram_row.svg").resolve()
png = root / "figure.png"
command = ["/usr/bin/inkscape", str(svg), "--export-type=png", "--export-filename=" + str(png)]
version = subprocess.check_output(["/usr/bin/inkscape", "--version"], text=True).strip()
result = subprocess.run(command, capture_output=True, timeout=40, check=False)
(root / "figure-render-stdout.txt").write_bytes(result.stdout)
(root / "figure-render-stderr.txt").write_bytes(result.stderr)
receipt = {
    "command_argv": command, "exit_code": result.returncode,
    "completed_at": datetime.now(UTC).isoformat(), "renderer": version,
    "source_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
    "render_sha256": hashlib.sha256(png.read_bytes()).hexdigest() if png.is_file() else None,
    "purpose": "Render unchanged referenced SVG for actual independent visual inspection.",
}
(root / "figure-render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
assert result.returncode == 0 and png.is_file()
