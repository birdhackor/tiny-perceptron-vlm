"""Bind A's stage chain and the B/C test code without redoing model runs."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


rows, pins, bests = [], {}, set()
for stage in ["pretrain", "sft", "vision", "ocr", "audio", "joint", "weighted", "native"]:
    folder = ROOT / f"docs/selftrained/results/training-raw/moe-{stage}"
    outer_path, train_path = folder / "receipt.json", folder / "raw/train-receipt.json"
    outer, train = json.loads(outer_path.read_bytes()), json.loads(train_path.read_bytes())
    files = {x["path"]: x for x in outer["files"]}
    assert sha(train_path) == files["train-receipt.json"]["sha256"]
    assert all(x["checkpoint_sha256"] in bests for x in train["stage_history"])
    assert outer["status"] == "completed" and outer["returncode"] == 0
    bests.add(files["best.pt"]["sha256"])
    rows.append({"stage": stage, "run_id": outer["run_id"], "best_sha256": files["best.pt"]["sha256"], "train_receipt_matches": True, "all_parent_checkpoints_in_prior_actual_receipts": True})
    for p in [outer_path, train_path]:
        pins[str(p.relative_to(ROOT))] = sha(p)
settings_path = ROOT / "docs/lesson-products/results/settings-freeze-v1.json"
settings = json.loads(settings_path.read_bytes())
identities = {}
for filename, expected in settings["source_sha256s"].items():
    p = ROOT / "tiny_perceptron/lesson_products" / filename
    identities[str(p.relative_to(ROOT))] = {"expected": expected, "observed": sha(p)}
    assert sha(p) == expected
pins[str(settings_path.relative_to(ROOT))] = sha(settings_path)
official_path = Path("/workspace/work/curriculum-20261016/chapter-19-typesafe-introduction.html")
assert sha(official_path) == "75746b82d1678fc99948b5efdaf9ec43e5a11f0837a8105998f4172375a3a018"
result = {
    "reviewer_task": "/root/backfill_tech_chapter_19",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/chapter-19-chain-original-01.py",
    "environment": {"python": platform.python_version(), "device": "cpu", "library": "Python standard library"},
    "result": "all asserted source/receipt identities match",
    "a_actual_receipt_chain": rows,
    "bc_frozen_source_identities": identities,
    "source_sha256": pins,
    "official_source": {
        "url": "https://docs.typesafe.ai/introduction",
        "version": "live official introduction retrieved 2026-10-16, fixed by SHA-256",
        "raw_sha256": sha(official_path),
        "raw_bytes": official_path.stat().st_size,
        "raw_path_external_to_git": str(official_path),
        "locator": "Introduction lead and TypeSafe primitives table",
        "short_excerpt": "Jev evaluates typed questions against a state and returns structured results directly. No text generation, no parsing. You get typed values and probability distributions that your code can branch on, sort by, and route with.",
        "inspection_note": "Personally read the introduction and primitives table from a fresh anonymous HTTPS GET of the original page, with script/style text excluded using HTMLParser. Supports the typed state-to-structured-judgment interface analogy. The source does not disclose Jev's neural architecture; no architecture, performance, calibrated correctness or exact API reproduction is inferred for this project's C.",
    },
}
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "a_stages": len(rows), "bc_source_files": len(identities), "output": str(out.relative_to(ROOT))}))
