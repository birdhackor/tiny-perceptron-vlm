"""freeze03 局部技術複查：核對未變證據及一次 CPU dry-run，沒有參數更新。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
ORIGINAL = ROOT / "docs/course-repair-20261017/reviews/technical-training-original-01.json"
EXPECTED_ORIGINAL = "0203d0628ddd6e8cb7dd949b2eee1755c94149ca94c203386b470796871928da"
FREEZE = ROOT / "docs/course-repair-20261017/freeze-03.json"
EXPECTED_FREEZE = "c2651ec9c2b63fed22d68733451a2418282bd8ef3d35733b9690619b17627c89"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(ORIGINAL) == EXPECTED_ORIGINAL
assert sha(FREEZE) == EXPECTED_FREEZE
original = json.loads(ORIGINAL.read_text())
body = (ROOT / "course/training.md").read_bytes()
old = (ROOT / "docs/course-repair-20261017/freeze-01/sources/training.md").read_bytes()
old_phrase = "先看報告的幾個欄位："
new_phrase = "先看報告的幾個欄位（若看最後摘要，`loss` 等批次數值在 `history` 第一筆）："
assert hashlib.sha256(old).hexdigest() == original["source_sha256"]
assert old.count(old_phrase.encode()) == 1
assert body == old.replace(old_phrase.encode(), new_phrase.encode())
assert re.findall(rb"```[^\n]*\n.*?```", old, re.S) == re.findall(rb"```[^\n]*\n.*?```", body, re.S)
page = next(p for p in json.loads(FREEZE.read_text())["pages"] if p["page_id"] == "training")
assert page["source_sha256"] == hashlib.sha256(body).hexdigest()
assert (ROOT / page["snapshot"]).read_bytes() == body

reused = []
for group in ("sources", "artifacts"):
    for item in original[group]:
        if "path" not in item or "sha256" not in item:
            continue
        path = ROOT / item["path"]
        actual = sha(path)
        assert actual == item["sha256"], item["path"]
        reused.append({"group": group, "id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})

environment = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "device": "cpu",
    "platform": platform.platform(),
}
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "2"
env["MKL_NUM_THREADS"] = "2"
command = [sys.executable, "scripts/train.py", "--task", "text", "--device", "cpu"]
started = datetime.now(UTC).isoformat()
run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=45, env=env)
assert run.returncode == 0, run.stderr
rows = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
assert len(rows) == 2
batch, summary = rows
assert summary["mode"] == "dry-run-no-weight-update"
assert summary["task"] == "text" and summary["device"] == "cpu"
assert len(summary["history"]) == 1 and summary["history"][0] == batch
assert all(key in summary["history"][0] for key in ("loss", "grad_norm", "effective_tokens"))
assert all(key not in summary for key in ("loss", "grad_norm", "effective_tokens"))

result = {
    "reviewer_task": "/root/repair_tech_training",
    "current_context": "informed_original_owner_source_revision_callback",
    "source": "course/training.md",
    "source_sha256": hashlib.sha256(body).hexdigest(),
    "freeze": str(FREEZE.relative_to(ROOT)),
    "freeze_sha256": sha(FREEZE),
    "original_report": str(ORIGINAL.relative_to(ROOT)),
    "original_report_sha256": sha(ORIGINAL),
    "source_change": {"old": old_phrase, "new": new_phrase, "only_exact_replacement": True, "all_fences_identical": True, "lines": len(body.splitlines())},
    "reused_evidence": reused,
    "reuse_scope": "先逐段重讀完整頁及全部 details；原始 sources/artifacts 全部核 hash 未變，再沿用原核查範圍與限制。沒有重跑42個命令或正式歷史訓練。",
    "command": command,
    "wrapper_command": "uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-training/callback-freeze03.py",
    "environment": environment,
    "started_at": started,
    "completed_at": datetime.now(UTC).isoformat(),
    "exit_code": run.returncode,
    "stdout": run.stdout,
    "stderr": run.stderr,
    "observed": {"json_lines": 2, "history_entries": 1, "history_first_equals_printed_batch": True, "batch_fields": batch, "summary_top_level_fields": list(summary)},
    "technical_conclusion": "T.2 這三條命令未加 --train，程式只走一個 batch；最後摘要的 history 第一筆正是先列印的 batch 數值。新增定位句成立，不改變欄位定義或訓練行為。",
    "limits": ["這次只有 CPU text dry-run，沒有 optimizer.step；不是訓練或品質重現。", "Windows/GPU/Modal/HF操作均未執行。", "原始全部能力、效能與硬體限制沿用，未以這次 dry-run 擴張結論。"],
}
target = OUT / "callback-freeze03-execution.json"
assert not target.exists(), "保留已存在的 callback artifact"
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source_sha256": result["source_sha256"], "reused_hashed_items": len(reused), "observed": result["observed"], "conclusion": result["technical_conclusion"]}, ensure_ascii=False, indent=2))
