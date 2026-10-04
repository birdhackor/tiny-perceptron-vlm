"""Exercise the actual serve CLI in Chromium without updating environment files."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_v2_19_01"


def main():
    checkpoint = ROOT / "outputs/technical-review-replay/fact_v2_19_01-download/joint/model.pt"
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/capstone.py",
        "serve",
        "--checkpoint",
        str(checkpoint),
        "--port",
        "0",
    ]
    server_env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    stderr_path = Path(str(PREFIX) + "_ui_server_stderr.txt")
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=server_env, stdout=subprocess.PIPE, stderr=stderr, text=True)
        try:
            first_line = process.stdout.readline()
            second_line = process.stdout.readline()
            match = re.search(r"http://127\.0\.0\.1:\d+/", first_line)
            assert match, (first_line, second_line)
            base = match[0]
            Path(str(PREFIX) + "_ui_server_stdout.txt").write_text(first_line + second_line)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"]
                )
                page = browser.new_page(viewport={"width": 1280, "height": 1000})
                errors, requests, responses = [], [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on(
                    "request",
                    lambda request: requests.append(
                        {"method": request.method, "url": request.url, "post_data": request.post_data}
                    ),
                )
                page.on("response", lambda response: responses.append({"status": response.status, "url": response.url}))
                page.goto(base, wait_until="networkidle")
                assert "328,128" in page.locator("#model-info").inner_text()
                page.locator('[data-example="calc"]').click()
                page.wait_for_timeout(200)
                example_state = {
                    "prompt": page.locator("#prompt").input_value(),
                    "chat_requests": sum(request["url"].endswith("/api/chat") for request in requests),
                    "raw": page.locator("#raw").inner_text(),
                }
                assert example_state["prompt"] == "1+2等於多少？"
                assert example_state["chat_requests"] == 0
                records = {}

                def submit(name):
                    with page.expect_response(
                        lambda response: response.url.endswith("/api/chat") and response.request.method == "POST",
                        timeout=30000,
                    ) as found:
                        page.locator("#submit").click()
                    response = found.value
                    assert response.status == 200
                    record = response.json()
                    page.wait_for_function(
                        "document.getElementById('status').textContent === '已完成這次提問。'", timeout=30000
                    )
                    records[name] = {
                        "payload": response.request.post_data_json,
                        "record": record,
                        "displayed_raw": page.locator("#raw").inner_text(),
                        "displayed_tool": page.locator("#tool").inner_text(),
                        "displayed_answer": page.locator("#answer").inner_text(),
                    }
                    return record

                calc = submit("calculator_enabled")
                assert calc["action_trace"]["raw"] == "TOOL:calculator:1+2"
                assert calc["runtime"] == {"status": "ok", "result": "3"}
                assert calc["final_trace"]["raw"] == "DIRECT:3" and calc["answer"] == "3"
                page.screenshot(path=str(PREFIX) + "_ui_tool_loop.png", full_page=True)
                page.locator("#calculator").uncheck()
                disabled = submit("calculator_disabled")
                assert disabled["action_trace"]["raw"] == "ASK:計算器未開" and disabled["runtime"] is None
                page.locator("#calculator").check()
                page.locator('[data-example="image"]').click()
                page.locator("#color").select_option("green")
                page.locator("#shape").select_option("square")
                page.locator("#prompt").fill("圖片是什麼形狀？")
                image = submit("known_square_failure")
                assert image["action_trace"]["raw"] == "DIRECT:circle"
                page.screenshot(path=str(PREFIX) + "_ui_square_failure.png", full_page=True)
                page.locator('[data-example="audio"]').click()
                page.locator("#frequency").fill("880")
                page.locator("#frequency").dispatch_event("change")
                audio = submit("high_tone")
                assert audio["action_trace"]["raw"] == "DIRECT:high"
                preview = page.evaluate("async () => await post('/api/preview', {...fields(), prompt: ''})")
                assert len(preview["audio_waveform"]) == 640
                assert preview["sample_rate"] == 16000 and preview["duration_seconds"] == 0.04
                assert any(value < 0 for value in preview["audio_waveform"])
                image_preview = page.evaluate(
                    "async () => await post('/api/preview', {prompt: '', calculator_available: true, style: '短', image: {color: 'green', shape: 'square'}, audio: null})"
                )
                assert len(image_preview["image_pixels"]) == 16 and len(image_preview["image_pixels"][0]) == 16
                assert image_preview["image_pixels"][8][8] == [0, 230, 0]
                assert image_preview["image_pixels"][0][0] == [0, 0, 0]
                assert not errors
                result = {
                    "command": command,
                    "environment": {
                        "python": platform.python_version(),
                        "playwright": importlib.metadata.version("playwright"),
                        "chromium": browser.version,
                        "device": "cpu",
                        "per_process_OMP_NUM_THREADS": "2",
                        "per_process_MKL_NUM_THREADS": "2",
                    },
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "source_sha256": {
                        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                        for p in [
                            "tiny_perceptron/capstone_ui.py",
                            "tiny_perceptron/capstone.py",
                            "scripts/capstone.py",
                        ]
                    },
                    "server_stdout": first_line + second_line,
                    "example_state_before_submission": example_state,
                    "records": records,
                    "preview": preview,
                    "image_preview": image_preview,
                    "requests": requests,
                    "responses": responses,
                    "page_errors": errors,
                    "scope": "Own Chromium run of actual CPU serve CLI; ephemeral loopback port avoids concurrent reviewers; no model training or external web publication.",
                }
                Path(str(PREFIX) + "_ui.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                browser.close()
                print(
                    json.dumps(
                        {
                            "example_clicked_without_chat": example_state["chat_requests"] == 0,
                            "calculator_result": calc["answer"],
                            "disabled_raw": disabled["action_trace"]["raw"],
                            "known_shape_failure": image["action_trace"]["raw"],
                            "audio_raw": audio["action_trace"]["raw"],
                            "page_errors": errors,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
