"""Replay all 19.4 validation without needing any ignored original weights."""

import hashlib
import json
import platform
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from tiny_perceptron.capstone import evaluate_rows, load_capstone

ROOT = Path(__file__).resolve().parents[3]
COMMAND = "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_04_public_replay.py"
torch.set_num_threads(2)
audit = json.loads((ROOT / "docs/technical-reviews/artifacts/fact_v2_19_04_audit.json").read_text())
dataset = json.loads((ROOT / "docs/technical-reviews/artifacts/fact_v2_19_04_dataset.json").read_text())
results = []
for stage in audit["stages"]:
    retrieval = stage["public_retrieval"]
    path = hf_hub_download(
        retrieval["repo"], retrieval["file"]["path"],
        revision=retrieval["revision"], token=False,
    )
    file_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert file_sha == retrieval["actual_sha256"]
    model, payload = load_capstone(path)
    assert set(payload["model"]) == set(stage["tensors"])
    for name, tensor in payload["model"].items():
        actual_sha = hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        assert actual_sha == stage["tensors"][name]["sha256"]
    generated = evaluate_rows(model, dataset["splits"]["validation"])
    assert generated == stage["original_validation"]
    results.append({
        "stage": stage["stage"], "anonymous_pin": retrieval,
        "public_file_sha256": file_sha,
        "original_inference_sha256": stage["checkpoint_sha256"],
        "all_original_tensor_fingerprints_match": True,
        "all_84_records_match_original": True,
        "complete_cpu_validation": generated,
    })
    print(stage["stage"], "84/84 records match", flush=True)
output = ROOT / "docs/technical-reviews/artifacts/fact_v2_19_04_public_replay.json"
output.write_text(json.dumps({
    "command": COMMAND,
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "result": "All four public pinned models match every original tensor SHA and all 84 validation records. No original ignored checkpoint files were read.",
    "source_inputs": ["fact_v2_19_04_audit.json", "fact_v2_19_04_dataset.json"],
    "stages": results,
}, ensure_ascii=False, indent=2) + "\n")
