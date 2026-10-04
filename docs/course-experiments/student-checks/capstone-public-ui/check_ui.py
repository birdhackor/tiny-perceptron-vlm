import hashlib
import json
import threading
from pathlib import Path

import torch
from playwright.sync_api import sync_playwright

from tiny_perceptron.capstone import build_dataset, expected_final, load_capstone
from tiny_perceptron.capstone_ui import create_server

out = Path("outputs/integration-runs/ui-anonymous-public")
out.mkdir(parents=True, exist_ok=True)
checkpoint = Path("outputs/integration-runs/anonymous-public-models/joint/model.pt")
torch.set_num_threads(1)
model, saved = load_capstone(checkpoint)
server = create_server(model, port=0, stage=saved["stage"])
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = f"http://127.0.0.1:{server.server_port}/"
splits, _ = build_dataset(42)
row = next(r for r in splits["validation"] if r["task"] == "calculator")
evidence = {
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "split": "validation",
    "row_id": row["id"],
    "expected_answer": expected_final(row),
    "page_errors": [],
    "screens": {},
}
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda e: evidence["page_errors"].append(str(e)))
        page.goto(url)
        page.wait_for_function("document.querySelector('#model-info').textContent.includes('328,128')")
        page.fill("#prompt", row["user"])
        page.click("#submit")
        page.wait_for_function("document.querySelector('#status').textContent==='已完成這次提問。'", timeout=30000)
        record = json.loads(page.locator("#trace").text_content())
        assert record["runtime"]["status"] == "ok"
        assert record["answer"] == expected_final(row), (record["answer"], expected_final(row))
        assert record["final_trace"] is not None
        evidence["actual_record"] = record
        page.screenshot(path=str(out / "desktop.png"), full_page=True)
        evidence["screens"]["desktop"] = {
            "width": 1440,
            "scroll_width": page.evaluate("document.documentElement.scrollWidth"),
        }
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(out / "mobile.png"), full_page=True)
        evidence["screens"]["mobile"] = {
            "width": 390,
            "scroll_width": page.evaluate("document.documentElement.scrollWidth"),
        }
        assert evidence["screens"]["mobile"]["scroll_width"] == 390
        assert not evidence["page_errors"]
        browser.close()
finally:
    server.shutdown()
    server.server_close()
    thread.join()
(out / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "answer": record["answer"],
            "action": record["parsed_action"],
            "screens": evidence["screens"],
            "page_errors": evidence["page_errors"],
        },
        ensure_ascii=False,
    )
)
