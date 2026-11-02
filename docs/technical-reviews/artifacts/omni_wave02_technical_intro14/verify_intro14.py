"""CPU-only checks for the chapter-14 introduction's concrete software promises."""

import hashlib
import json
import platform
import re
from pathlib import Path

import torch

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import RMSNorm, rope


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"
EXPECTED_FREEZE_SHA = "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert digest(FREEZE) == EXPECTED_FREEZE_SHA
freeze = json.loads(FREEZE.read_text())
pages = {page["page_id"]: page for page in freeze["pages"]}
intro = pages["chapter-14"]
intro_text = (ROOT / intro["snapshot"]).read_text()
assert digest(ROOT / intro["snapshot"]) == intro["source_sha256"]
assert intro["figures_sha256"] == {}
checked_pages = {}
for page_id in ["4.5", "4.6"] + [f"14.{number}" for number in range(1, 20)]:
    page = pages[page_id]
    snapshot = ROOT / page["snapshot"]
    assert digest(snapshot) == page["source_sha256"]
    text = snapshot.read_text()
    assert text.startswith(f"## {page_id} ")
    checked_pages[page_id] = {"snapshot": page["snapshot"], "sha256": digest(snapshot)}

links = []
for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", intro_text):
    path, _, anchor = target.partition("#")
    destination = (ROOT / intro["source"]).parent / path
    assert destination.is_file(), target
    if anchor:
        assert re.search(r"^## " + re.escape(anchor) + r" ", destination.read_text(), re.M), target
    links.append({"target": target, "exists": True, "numbered_anchor_found": bool(anchor)})

torch.set_num_threads(1)
torch.manual_seed(0)
ids = torch.tensor([[1, 2, 3]], device="cpu")
variants = [
    ("baseline", {}),
    ("rotary", {"rotary": True}),
    ("rms", {"norm": "rms"}),
    ("relu2", {"activation": "relu2"}),
    ("swiglu", {"activation": "swiglu"}),
    ("tied", {"tied": True}),
    ("combined", {"rotary": True, "norm": "rms", "activation": "swiglu", "tied": True}),
]
outputs = []
for name, options in variants:
    config = ModelConfig(vocab_size=20, width=8, layers=1, heads=2, max_length=16, **options)
    model = TinyLM(config).eval()
    with torch.no_grad():
        result = model(ids)
    logits = result["logits"]
    assert list(logits.shape) == [1, 3, 20]
    assert torch.isfinite(logits).all()
    assert (model.position is None) == config.rotary
    assert model.blocks[0].attention.rotary == config.rotary
    assert isinstance(model.blocks[0].norm1, RMSNorm) == (config.norm == "rms")
    assert isinstance(model.blocks[0].norm2, RMSNorm) == (config.norm == "rms")
    assert isinstance(model.final_norm, RMSNorm) == (config.norm == "rms")
    assert model.blocks[0].ffn.activation == config.activation
    assert (model.blocks[0].ffn.gate is not None) == (config.activation == "swiglu")
    assert (model.output.weight is model.embedding.weight) == config.tied
    outputs.append({
        "variant": name, "settings": options, "logits_shape": list(logits.shape),
        "finite": True, "position_table_absent": model.position is None,
        "normalization": type(model.final_norm).__name__, "activation": config.activation,
        "has_gate": model.blocks[0].ffn.gate is not None,
        "same_input_output_parameter": model.output.weight is model.embedding.weight,
        "parameter_count": model.description()["parameters"],
    })

fractional_positions = torch.arange(16, dtype=torch.float32) * 8 / 16
rotated = rope(torch.ones(1, 1, 16, 4), fractional_positions)
assert list(rotated.shape) == [1, 1, 16, 4] and torch.isfinite(rotated).all()

print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "platform": platform.platform()},
    "freeze_sha256": digest(FREEZE),
    "intro_sha256": digest(ROOT / intro["snapshot"]),
    "intro_figures": intro["figures_sha256"], "links": links,
    "checked_frozen_pages": checked_pages, "variants": outputs,
    "fractional_rope_positions": {"first": fractional_positions[0].item(),
                                  "last": fractional_positions[-1].item(),
                                  "output_shape": list(rotated.shape), "finite": True},
    "scope": "Only deterministic CPU forward/configuration/link checks; no backward, optimizer step, training, generation-quality or timing comparison.",
}, ensure_ascii=False, indent=2))
