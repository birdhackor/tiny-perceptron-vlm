"""核對位置縮放的轉角、張量索引與圖中箭頭座標。"""

import hashlib
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.modern import rope


def close(actual, expected, tolerance, name):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{name}：實際 {actual}，預期 {expected}，容忍差 {tolerance}")


x = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(1, 1, 1, 4)
frequency = 10000 ** (-torch.arange(0, 4, 2, dtype=torch.float32) / 4)
records = []
for scale in (1, 2, 4):
    positions = torch.tensor([0.0, 1.0]) / scale
    vectors = rope(x.expand(1, 1, 2, 4), positions)
    origin, turned = vectors[0, 0]
    torch.testing.assert_close(origin, x[0, 0, 0], rtol=0, atol=0)
    degrees = torch.rad2deg(torch.atan2(turned[1::2], turned[0::2]))
    expected = [math.degrees(1 / scale), math.degrees(0.01 / scale)]
    for actual, wanted in zip(degrees.tolist(), expected):
        close(actual, wanted, 0.00001, "轉角")
    records.append({
        "scale": scale,
        "positions": positions.tolist(),
        "output_shape": list(vectors.shape),
        "origin": origin.tolist(),
        "turned": turned.tolist(),
        "horizontal": turned[0::2].tolist(),
        "vertical": turned[1::2].tolist(),
        "degrees": degrees.tolist(),
        "rounded": degrees.round(decimals=3).tolist(),
        "expected_degrees": expected,
        "pair_norms": turned.reshape(2, 2).norm(dim=-1).tolist(),
    })

svg_path = ROOT / "course/figures/rewrite-14-9-nearby-angles.svg"
svg_raw = svg_path.read_bytes()
svg = ET.fromstring(svg_raw)
ns = {"s": "http://www.w3.org/2000/svg"}
solid = [line for line in svg.findall(".//s:line", ns) if "marker-end" in line.attrib]
dashed = [line for line in svg.findall(".//s:line", ns) if "stroke-dasharray" in line.attrib]
arcs = svg.findall(".//s:path", ns)[1:]
figure = []
for line, start, arc, expected in zip(solid, dashed, arcs, [60, 30, 15, 7.5], strict=True):
    cx, cy, end_x, end_y = [float(line.attrib[k]) for k in ("x1", "y1", "x2", "y2")]
    angle = math.degrees(math.atan2(cy - end_y, end_x - cx))
    radius = math.hypot(end_x - cx, end_y - cy)
    close(angle, expected, 0.0001, "圖中實線角度")
    close(radius, 64, 0.0001, "圖中實線長度")
    close(float(start.attrib["y2"]), cy, 0, "虛線水平方向")
    close(float(start.attrib["x2"]), cx + 64, 0, "虛線朝右方向")
    close(float(start.attrib["x1"]), cx, 0, "虛線起點")
    close(float(start.attrib["y1"]), cy, 0, "虛線起點")
    values = list(map(float, re.findall(r"[-+]?\d*\.?\d+", arc.attrib["d"])))
    arc_angle = math.degrees(math.atan2(cy - values[-1], values[-2] - cx))
    close(arc_angle, expected, 0.0001, "圖中弧線角度")
    figure.append({"center": [cx, cy], "expected_degrees": expected,
                   "measured_degrees": angle, "arrow_length": radius,
                   "arc_endpoint_degrees": arc_angle})

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-9/verify_angles.py",
    "result": "所有轉角、位置、索引、圖中實線與弧線核對通過",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "dtype": "torch.float32"},
    "input_shape": list(x.shape),
    "input": x.flatten().tolist(),
    "frequency": frequency.tolist(),
    "rotation_records": records,
    "toy": {"rates_degrees": [60, 15], "period_position_increments": [360 / 60, 360 / 15],
            "scale_2_delta_position": 0.5, "scale_2_angles": [60 / 2, 15 / 2],
            "scale_4_delta_position": 0.25, "scale_4_angles": [60 / 4, 15 / 4]},
    "figure_sha256": hashlib.sha256(svg_raw).hexdigest(),
    "figure_coordinates": figure,
    "tolerances": {"degrees": 0.00001, "svg_degrees": 0.0001, "svg_length": 0.0001},
}
print(json.dumps(output, ensure_ascii=False, indent=2))
