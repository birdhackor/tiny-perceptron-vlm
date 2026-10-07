"""Verify unlocked W.1 kernel persistence/restart with fresh synthetic names."""
import hashlib
import json
import sys
from pathlib import Path

from jupyter_client import KernelManager

km = KernelManager(kernel_name="tiny-perceptron")
km.start_kernel(cwd="/tmp/p7-technical-a-w1-g4z24kt8/tiny-perceptron-vlm")
client = km.client()
client.start_channels()
client.wait_for_ready(timeout=30)

def execute(code):
    msg_id = client.execute(code)
    observed = []
    while True:
        msg = client.get_iopub_msg(timeout=30)
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        typ, content = msg["msg_type"], msg["content"]
        if typ == "stream":
            observed.append({"type": typ, "text": content["text"]})
        elif typ == "error":
            observed.append({"type": typ, "ename": content["ename"], "evalue": content["evalue"]})
        elif typ == "status" and content.get("execution_state") == "idle":
            break
    return {"code": code, "observed": observed}

try:
    runs = [execute("review_value = 1"), execute("review_value += 1\nprint(review_value)")]
    km.restart_kernel(now=True)
    client.wait_for_ready(timeout=30)
    runs.append(execute("print(review_value)"))
finally:
    client.stop_channels()
    km.shutdown_kernel(now=True)
record = {"command": "/tmp/p7-technical-a-w1-g4z24kt8/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/w1-kernel-state.py", "environment": {"python": sys.version.split()[0], "device": "Linux CPU", "kernel": "tiny-perceptron"}, "runs": runs, "scope": "fresh synthetic cells; no unread notebook source or model training"}
path = Path(__file__).parent / "w1-kernel-state-result.json"
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False))
print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())
