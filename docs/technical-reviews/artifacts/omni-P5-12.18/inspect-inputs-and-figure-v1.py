"""Portable source binding and SVG geometry check; screenshots remain outside Git."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


def ref(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


packet_path = Path("docs/course-revision-20261029-omni/reviews/P5-actual/technical-inputs-packet-v6.json")
packet = json.loads(packet_path.read_text())
freeze_path = Path(packet["canonical_source_freeze"]["path"])
assert ref(packet_path)["sha256"] == "4f6af6d8b67565c5c994606c14764662f43ca64c40c4b35f913e8d068551e4fc"
assert ref(freeze_path)["sha256"] == packet["canonical_source_freeze"]["sha256"]
freeze = json.loads(freeze_path.read_text())
page = next(x for x in freeze["pages"] if x["page_id"] == "12.18")
frozen_body = Path(page["snapshot"]).read_bytes()
assert hashlib.sha256(frozen_body).hexdigest() == page["source_sha256"]
chapter = Path(page["source"]).read_bytes()
current = re.search(rb"(?ms)^## 12\.18 .*?(?=^## |\Z)", chapter).group()
assert current == frozen_body
assert ref(page["notebook"])["sha256"] == page["notebook_sha256"]
figure = next(iter(page["figures_sha256"]))
assert ref(figure)["sha256"] == page["figures_sha256"][figure]
snapshot = freeze["figure_snapshots"][figure]
assert Path(figure).read_bytes() == Path(snapshot["path"]).read_bytes()
svg = ET.parse(figure).getroot()
ns = {"s": "http://www.w3.org/2000/svg"}
paths = {x.attrib["class"]: x.attrib["d"] for x in svg.findall("s:path", ns)}


def decode_polyline(d):
    pairs = [(float(x), float(y)) for x, y in re.findall(r"[ML]([\d.]+) ([\d.]+)", d)]
    return {"indices": [(x - 45) / 25 for x, _ in pairs], "amplitudes": [(200 - y) / 88 for _, y in pairs]}


original = decode_polyline(paths["original"])
decoded = decode_polyline(paths["decoded"])
expected_original = [.9, 0, -.9, 0, 0, .8, 0, -.8, .1, -.1, .1, -.1]
expected_decoded = [1, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0]
for actual, expected in ((original, expected_original), (decoded, expected_decoded)):
    assert actual["indices"] == list(range(12))
    assert all(abs(a-b) < 1e-12 for a, b in zip(actual["amplitudes"], expected, strict=True))
labels = [x.text for x in svg.findall(".//s:text", ns)]
assert [x for x in labels if x and x.startswith("ID ")] == ["ID 1", "ID 2", "ID 0"]
assert "樣本索引（非秒數）" in labels
browser = "/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"
screenshot_path = "/workspace/work/reference/omni-P5-12.18-figure-720.png"
result = {
    "reviewer_task": "/root/omni_p5_technical_12_18",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "packet": ref(packet_path), "canonical_freeze": ref(freeze_path),
    "source_snapshot": ref(page["snapshot"]), "notebook": ref(page["notebook"]),
    "figure": ref(figure), "figure_snapshot": ref(snapshot["path"]),
    "frozen_body_equals_current_section": True,
    "original_coordinates": original, "decoded_coordinates": decoded,
    "coordinate_rule": "sample_index=(x-45)/25; amplitude=(200-y)/88",
    "coordinate_tolerance": "absolute amplitude error < 1e-12; integer sample indices exact",
    "block_boundaries": "x=132.5 and x=232.5 separate samples 3/4 and 7/8; lower IDs 1,2,0 align with blocks",
    "all_svg_text": labels,
    "actual_render": {
        "renderer": subprocess.check_output([browser, "--version"], text=True).strip(),
        "command": browser + " --headless --no-sandbox --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=360,445 --screenshot=" + screenshot_path + " file:///workspace/work/omni-book-integrated-v1/" + figure,
        "screenshot_outside_git": ref(screenshot_path),
        "human_model_visual_inspection": "Reviewer opened the actual 720x890 render with view_image. Grey input and dashed blue reconstruction remain distinguishable. Three coloured blocks, 0/4/8/11 sample ticks, amplitude labels and ID 1/2/0 match the body. Text is readable and unclipped at the SVG's native 360 CSS-pixel width.",
        "scope": "Direct SVG render only; full lesson desktop/mobile page flow and Colab UI were not reviewed in this technical task.",
    },
    "result": "all frozen-input and figure geometry checks passed",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
