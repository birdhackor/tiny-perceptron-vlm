"""Narrow necessary-context recheck; does not rerun 11.16 metrics or models."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
import torch
from playwright.sync_api import sync_playwright
from tiny_perceptron.natural_concepts import swapped_picture, picture_order_report

out = Path(__file__).resolve().parent
first, second = swapped_picture()
result = picture_order_report()
assert result == {"different_pixels": 128, "same_4_by_4_summary": True, "same_pixel_sequence": False}
assert int((first != second).any(dim=0).sum()) == 64
assert torch.equal(first[:, 8:16, :8].sum(dim=(1, 2)), second[:, 8:16, :8].sum(dim=(1, 2)))
svg_path = ROOT / "course/figures/practical_order.svg"
svg = svg_path.read_text()
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 720, "height": 1120}, device_scale_factor=1)
    page.set_content('<html><head><meta charset="utf-8"><style>html,body{margin:0}</style></head><body>' + svg + '</body></html>')
    page.screenshot(path=str(out / "practical_order.png"), full_page=True)
    version = browser.version
    browser.close()
print(json.dumps({"environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "chromium": version},
                  "new_context_only": "11.14 toy image construction/information-loss interpretation, not a new 11.16 or GPU/data benchmark",
                  "picture_order_report": result, "different_spatial_locations": 64,
                  "channel_sums_in_same_8x8_block": first[:, 8:16, :8].sum(dim=(1, 2)).tolist(),
                  "hand_derivation": "Each image has 32 red and 32 blue unit scalar values in one 64-position block; both channel means are 32/64=0.5. Swapping all 64 locations changes two channel scalars at each location, totaling128. This identical summary admits two different spatial orders, so summary equality cannot recover order; retained sequences are not proof of learned natural-scene comprehension.",
                  "svg_path": str(svg_path.relative_to(ROOT)), "svg_sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(),
                  "render_path": str((out / "practical_order.png").relative_to(ROOT)), "render_method": "Chromium page.set_content on unchanged current SVG; full-page screenshot720x1120"}, indent=2))
