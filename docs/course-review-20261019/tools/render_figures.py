"""Render SVG figures at their own width/height with headless Chrome; write PNGs named by the SVG's SHA-256."""

import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS_PX = {"": 1, "px": 1, "pt": 4 / 3}
root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
version = subprocess.run([CHROME, "--version"], capture_output=True, text=True).stdout.strip()
records = []
for rel in sys.argv[3:]:
    svg = root / rel
    raw = svg.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    tag = re.search(rb"<svg[^>]*>", raw).group(0).decode()
    sized = re.search(r'\swidth="([\d.]+)([a-z%]*)"', tag), re.search(r'\sheight="([\d.]+)([a-z%]*)"', tag)
    if all(sized):
        # 視窗以 CSS px 計：1pt 是 4/3 px。其他單位不換算，寧可停下也不截掉圖。
        if any(m.group(2) not in CSS_PX for m in sized):
            raise SystemExit(f"unsupported SVG size unit: {rel}: {tag}")
        width, height = (math.ceil(float(m.group(1)) * CSS_PX[m.group(2)]) for m in sized)
    else:
        box = [float(v) for v in re.search(r'viewBox="([^"]+)"', tag).group(1).replace(",", " ").split()]
        width, height = int(box[2]), int(box[3])
    png = out / f"{digest}.png"
    command = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--window-size={width},{height}",
        f"--screenshot={png}",
        svg.as_uri(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode or not png.exists():
        raise SystemExit(f"render failed: {rel}: {result.stderr[-500:]}")
    records.append(
        {
            "svg": rel,
            "svg_sha256": digest,
            "png": png.name,
            "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
            "css_size": [width, height],
            "device_scale_factor": 2,
        }
    )
receipt = {"rendered_at": datetime.now(UTC).isoformat(), "renderer": version, "records": records}
(out / "renders.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"rendered": len(records), "renderer": version}))
