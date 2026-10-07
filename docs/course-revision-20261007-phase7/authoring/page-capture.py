"""Capture a completed page; no viewing or review judgment occurs in this helper."""

import argparse
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("page_id")
parser.add_argument("--owner", required=True)
parser.add_argument("--open-details", action="store_true")
parser.add_argument("--tables", action="store_true")
parser.add_argument("--text", action="append", default=[], help="Locate a substring in an already-read paragraph")
args = parser.parse_args()
assert "/" not in args.page_id and args.page_id not in (".", "..")
assert "/" not in args.owner and args.owner not in (".", "..")
run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
folder = (
    Path("outputs/phase7-render/pages")
    / args.owner
    / args.page_id
    / ("expanded" if args.open_details else "captures")
    / run_id
)
folder.mkdir(parents=True, exist_ok=False)
items = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"]
    )
    for width, height in ((1280, 800), (390, 844)):
        page = browser.new_page(viewport={"width": width, "height": height}, reduced_motion="reduce")
        page.goto(f"http://127.0.0.1:8767/{args.page_id}.html", wait_until="networkidle")
        page.locator(".md-content").wait_for()
        if args.open_details:
            page.locator(".md-content details").evaluate_all("els => els.forEach(e => e.open = true)")
            page.wait_for_timeout(150)
        positions = page.locator(".illustration").evaluate_all(
            """els => els.map((e, i) => {
                const r = e.getBoundingClientRect();
                return {index: i, kind: 'figure', y: r.top + window.scrollY, height: r.height,
                        visible: e.getClientRects().length > 0 && r.height > 0};
            })"""
        )
        hidden = [pos["index"] for pos in positions if not pos["visible"]]
        visible = [pos for pos in positions if pos["visible"]]
        tables = page.locator(".md-content table").evaluate_all(
            """els => els.map((e, i) => {
                const r = e.getBoundingClientRect();
                return {index: i, kind: 'table', y: r.top + window.scrollY, height: r.height,
                        visible: e.getClientRects().length > 0 && r.height > 0};
            })"""
        )
        if args.tables:
            visible += [pos for pos in tables if pos["visible"]]
        text_counts = []
        for text_index, requested in enumerate(args.text):
            matches = page.locator(".md-content p").filter(has_text=requested).evaluate_all(
                """els => els.map((e, i) => {
                    const r = e.getBoundingClientRect();
                    return {index: i, kind: 'text', y: r.top + window.scrollY, height: r.height,
                            visible: e.getClientRects().length > 0 && r.height > 0};
                })"""
            )
            shown = [pos for pos in matches if pos["visible"]]
            if not shown:
                raise ValueError(f"No visible already-read paragraph matches: {requested!r}")
            text_counts.append({"requested": requested, "visible_matches": len(shown)})
            for pos in shown:
                pos["index"] = f"text{text_index}-match{pos['index']}"
                pos["requested_text"] = requested
                visible.append(pos)
        if not visible:
            visible = [{"index": "page", "kind": "page", "y": page.locator(".md-content").evaluate(
                "e => e.getBoundingClientRect().top + window.scrollY"
            ), "height": height}]
        for pos in visible:
            horizontal_positions = [0]
            container_width = None
            if pos["kind"] == "table":
                table = page.locator(".md-content table").nth(pos["index"])
                container_width = table.evaluate("""e => {
                    for (let p = e.parentElement; p && p !== document.body; p = p.parentElement) {
                        if (p.scrollWidth > p.clientWidth &&
                            ['auto','scroll'].includes(getComputedStyle(p).overflowX)) {
                            return {client: p.clientWidth, maximum: p.scrollWidth - p.clientWidth};
                        }
                    }
                    return null;
                }""")
                if container_width:
                    step = max(1, container_width["client"] - 48)
                    maximum = container_width["maximum"]
                    horizontal_positions = list(range(0, maximum, step)) + [maximum]
            for horizontal_index, horizontal_position in enumerate(horizontal_positions):
                actual_horizontal = 0
                if container_width:
                    actual_horizontal = table.evaluate("""(e, x) => {
                        for (let p = e.parentElement; p && p !== document.body; p = p.parentElement) {
                            if (p.scrollWidth > p.clientWidth &&
                                ['auto','scroll'].includes(getComputedStyle(p).overflowX)) {
                                p.scrollLeft = x;
                                return p.scrollLeft;
                            }
                        }
                        return 0;
                    }""", horizontal_position)
                for part in range(max(1, int((pos["height"] + height - 161) // (height - 160)))):
                    page.evaluate("y => window.scrollTo(0, y)", max(0, pos["y"] - 80 + part * (height - 160)))
                    page.wait_for_timeout(120)
                    suffix = f"table-{pos['index']}-h{horizontal_index}" if pos["kind"] == "table" else str(pos["index"])
                    output = folder / f"{width}-{suffix}-{part}.png"
                    page.screenshot(path=str(output))
                    items.append({
                        "path": output.as_posix(),
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "viewport": [width, height],
                        "position": page.evaluate("window.scrollY"),
                        "details_opened": args.open_details,
                        "capture_kind": pos["kind"],
                        "figure_index": pos["index"] if pos["kind"] == "figure" else None,
                        "table_index": pos["index"] if pos["kind"] == "table" else None,
                        "requested_text": pos.get("requested_text"),
                        "horizontal_position": actual_horizontal,
                        "horizontal_container": container_width,
                    })
        overflow = page.evaluate(
            "({scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth})"
        )
        print(json.dumps({
            "page_id": args.page_id, "viewport": [width, height], "overflow": overflow,
            "details_opened": args.open_details, "hidden_figure_indices": hidden,
            "visible_figure_count": len(positions) - len(hidden),
            "table_captures_requested": args.tables,
            "visible_table_count": sum(pos["visible"] for pos in tables),
            "text_captures_requested": args.text,
            "text_matches": text_counts,
        }))
        page.close()
    browser.close()
print(json.dumps({"artifacts": items, "capture_directory": folder.as_posix()}))
