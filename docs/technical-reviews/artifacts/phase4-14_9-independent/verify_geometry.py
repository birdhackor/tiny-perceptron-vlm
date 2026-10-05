"""Bounded CPU checks of the section's hand example and original SVG endpoints."""
import hashlib
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent

def rotate(vector, angle_degrees):
    a = math.radians(angle_degrees)
    c, s = math.cos(a), math.sin(a)
    x, y = vector
    return (c * x - s * y, s * x + c * y)

def close_pair(a, b, tol=1e-12):
    return max(abs(x - y) for x, y in zip(a, b)) <= tol

checks = []
for speed, period in ((60.0, 6.0), (15.0, 24.0)):
    assert speed * period == 360.0
    for divisor, expected in ((1, speed), (2, speed / 2), (4, speed / 4)):
        p0, p1 = 0.0, 1.0 / divisor
        delta = (p1 - p0) * speed
        assert delta == expected
        q = rotate((1.0, 0.0), p0 * speed)
        k = rotate((1.0, 0.0), p1 * speed)
        dot = sum(x * y for x, y in zip(q, k))
        assert abs(dot - math.cos(math.radians(expected))) < 1e-12
        assert abs(sum(v * v for v in k) - 1) < 1e-12
        checks.append({"speed_degrees_per_position": speed, "position_divisor": divisor,
                       "position_delta": p1-p0, "angle_degrees": delta, "fixed_content_dot": dot})
    assert close_pair(rotate((1.0, 0.0), period * speed), (1.0, 0.0))

# Small meaningful variants: move both coordinates while preserving their gap;
# a full fast-group turn does not cause the slow group to complete its turn.
for speed in (60.0, 15.0):
    for divisor in (2, 4):
        q = rotate((1.0, 0.0), 10.0 * speed / divisor)
        k = rotate((1.0, 0.0), 11.0 * speed / divisor)
        dot = sum(x * y for x, y in zip(q, k))
        assert abs(dot - math.cos(math.radians(speed/divisor))) < 1e-12
assert close_pair(rotate((1.0, 0.0), 6*60), (1.0, 0.0))
assert close_pair(rotate((1.0, 0.0), 6*15), (0.0, 1.0))
# Content is not merged: applying one rotation to two distinct vectors preserves
# their nonzero Euclidean distance. This does not claim model discrimination.
for angle in (7.5, 30.0):
    a, b = rotate((1.0, 0.0), angle), rotate((0.0, 1.0), angle)
    assert abs(sum((x-y)**2 for x, y in zip(a, b)) - 2.0) < 1e-12

svg = OUT / "figure-original.svg"
tree = ET.fromstring(svg.read_bytes())
ns = "{http://www.w3.org/2000/svg}"
orange = [e for e in tree.iter(ns+"line") if e.attrib.get("stroke") == "#b85119"]
blue = [e for e in tree.iter(ns+"line") if e.attrib.get("stroke-dasharray")]
assert len(orange) == len(blue) == 4
figure_checks = []
for e, target in zip(orange, (60.0, 30.0, 15.0, 7.5)):
    dx = float(e.attrib["x2"])-float(e.attrib["x1"])
    dy_math = float(e.attrib["y1"])-float(e.attrib["y2"])
    angle = math.degrees(math.atan2(dy_math, dx))
    length = math.hypot(dx, dy_math)
    assert abs(angle-target) < 0.0001
    assert abs(length-64.0) < 0.0001
    figure_checks.append({"expected_degrees": target, "svg_angle_degrees": angle,
                          "svg_vector_length_px": length,
                          "angle_error_degrees": angle-target})
for e in blue:
    assert float(e.attrib["y1"]) == float(e.attrib["y2"])
    assert float(e.attrib["x2"])-float(e.attrib["x1"]) == 64.0

result = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-14_9-independent/verify_geometry.py",
    "environment": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "device": "CPU, stdlib only; no model, training, or evaluation"},
    "geometry_checks": checks,
    "figure_checks": figure_checks,
    "meaningful_variants": {"translated_positions": [10, 11], "divisors": [2, 4],
                            "fast_period_slow_disambiguation": "position 6: fast returns to [1,0], slow is [0,1]",
                            "content_distance_squared_preserved": 2.0},
    "numeric_tolerance": {"rotation_or_dot_absolute": 1e-12,
                          "svg_degrees_and_length_px_absolute": 0.0001,
                          "section_degree_arithmetic": "exact arithmetic on representable integers/halves/quarters"},
    "figure_original_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
    "result": "all assertions passed",
    "scope": "Hand-selected 60/15 degrees per position. Mathematical mechanism only; no evidence of language-model quality or long-context ability.",
}
(OUT / "geometry-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
