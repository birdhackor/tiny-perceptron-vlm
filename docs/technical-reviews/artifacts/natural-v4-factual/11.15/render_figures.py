"""Render own current SVGs with the available browser; no source changes."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parent

def main():
    records = []
    version = subprocess.check_output(["/usr/bin/inkscape", "--version"], text=True).strip()
    for name, height in [("practical_stroke", 1080), ("natural_shared_chat", 1010)]:
        source = ROOT / "course/figures" / (name + ".svg")
        command = ["/usr/bin/inkscape", str(source), "--export-type=png",
                   "--export-width=720", "--export-height=" + str(height),
                   "--export-filename=" + str(ART / (name + ".png"))]
        completed = subprocess.run(command, capture_output=True, text=True)
        print(completed.stdout)
        print(completed.stderr)
        completed.check_returncode()
        records.append({"source": str(source.relative_to(ROOT)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "render": name + ".png", "size": [720, height],
                        "renderer": version, "command": command,
                        "exit_code": completed.returncode})
    (ART / "figure-renders.json").write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records, indent=2))

main()
