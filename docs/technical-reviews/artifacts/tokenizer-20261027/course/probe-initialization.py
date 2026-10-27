"""Rebuild documented random starts without training or downloading weights."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, Tokenizer
from tiny_perceptron.lesson_products.training import state_digest

torch.set_num_threads(1)
rows = []
for product, seed, cls, width, ff in (
    ("b", 7421, LoopedSolver, 128, 256),
    ("c", 9041, CandidateScorer, 64, 128),
):
    torch.manual_seed(seed)
    model = cls(Config(len(Tokenizer()), width=width, ff=ff))
    path = ROOT / f"docs/lesson-products/results/initial-{product}-execution.json"
    original = json.loads(path.read_bytes())
    expected = original["training"]["initial_weights_sha256"]
    observed = state_digest(model.state_dict())
    assert observed == expected
    rows.append({
        "product": product,
        "seed": seed,
        "original_record": path.relative_to(ROOT).as_posix(),
        "original_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected": expected,
        "observed": observed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "matches_saved_initialization": True,
    })

result = {
    "command": "uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/course/probe-initialization.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU"},
    "result": rows,
    "scope": "Two random model constructions, zero parameter updates, no weights/data downloaded and no GPU call. Historical training is inspected through original saved records.",
}
target = Path(__file__).with_name("initialization-check-reproducible.json")
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
