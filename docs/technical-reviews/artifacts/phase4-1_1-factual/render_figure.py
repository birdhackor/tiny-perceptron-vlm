"""Render the unchanged SVG with the installed Inkscape after Chromium timeout."""
from pathlib import Path
import hashlib
import json
import platform
import shlex
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
argv = ["/usr/bin/inkscape", str(OUT / "figure.svg"), "--export-type=png", "--export-filename=" + str(OUT / "figure.png"), "--export-width=640", "--export-height=710"]
run = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=30)
(OUT / "figure-render-stdout.txt").write_bytes(run.stdout)
(OUT / "figure-render-stderr.txt").write_bytes(run.stderr)
version = subprocess.run(["/usr/bin/inkscape", "--version"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
record = {"command":shlex.join(argv),"command_argv":argv,"cwd":str(ROOT),"exit_code":run.returncode,"environment":{"python":sys.version,"device":"CPU; standalone SVG rasterization; no model/training","inkscape":version,"platform":platform.platform()},"source_svg_sha256":hashlib.sha256((OUT / "figure.svg").read_bytes()).hexdigest(),"stdout_sha256":hashlib.sha256(run.stdout).hexdigest(),"stderr_sha256":hashlib.sha256(run.stderr).hexdigest(),"png_sha256":hashlib.sha256((OUT / "figure.png").read_bytes()).hexdigest() if (OUT / "figure.png").exists() else None}
(OUT / "figure-render.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record,ensure_ascii=False,indent=2))
raise SystemExit(run.returncode)
