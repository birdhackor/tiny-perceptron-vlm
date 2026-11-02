"""Independently execute lesson 10.6's frozen code and its stated exercise on CPU."""

import contextlib
import hashlib
import io
import json
import platform
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
COMMAND = (
    "uv run --extra cpu python "
    "docs/technical-reviews/artifacts/omni-technical-technical_10_6-v1/verify_lesson.py"
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_digest(tensor):
    values = tensor.detach().cpu().flatten().tolist()
    return digest(struct.pack("<" + "f" * len(values), *values))


def execute(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "frozen-lesson-10.6", "exec"), namespace)
    return namespace, output.getvalue()


def main():
    torch.set_num_threads(1)
    freeze = json.loads(FREEZE.read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "10.6")
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    blocks = re.findall(r"```python\n(.*?)\n```", raw.decode(), flags=re.S)
    assert len(blocks) == 1
    original = blocks[0]
    namespace, stdout = execute(original)
    vision = namespace["vision"]
    layer = namespace["projector"]
    features = namespace["language_features"]
    assert vision.shape == (1, 16, 8)
    assert features.shape == (1, 16, 12)
    assert layer.weight.shape == (12, 8)
    assert vision.dtype == torch.float32 and vision.device.type == "cpu"
    assert layer.bias is not None and layer.bias.shape == (12,)
    expected_stdout = "圖片特徵 (1, 16, 8)\n接頭輸出 (1, 16, 12)\n接頭矩陣 (12, 8)\n"
    assert stdout == expected_stdout
    per_position = torch.stack(
        [layer.weight @ vision[0, position] + layer.bias for position in range(16)]
    )[None]
    maximum_error = (features - per_position).abs().max().item()
    assert maximum_error < 1e-6

    exercise_code = original.replace("nn.Linear(8, 12)", "nn.Linear(8, 10)")
    assert exercise_code != original
    exercise, exercise_stdout = execute(exercise_code)
    short_features = exercise["language_features"]
    assert short_features.shape == (1, 16, 10)
    assert exercise["projector"].weight.shape == (10, 8)
    assert exercise_stdout == "圖片特徵 (1, 16, 8)\n接頭輸出 (1, 16, 10)\n接頭矩陣 (10, 8)\n"

    toy = nn.Linear(2, 3)
    with torch.no_grad():
        toy.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        toy.bias.zero_()
    toy_input = torch.tensor([1.0, 2.0])
    toy_output = toy(toy_input)
    assert torch.equal(toy_output, torch.tensor([1.0, 2.0, 3.0]))

    from tiny_perceptron.model import ModelConfig, TinyLM
    from tiny_perceptron.multimodal import MultiModalLM

    core = TinyLM(ModelConfig(width=12))
    logits = core(embeddings=features)["logits"]
    assert logits.shape == (1, 16, 264) and torch.isfinite(logits).all()
    try:
        core(embeddings=short_features)
    except RuntimeError as error:
        mismatch_error = str(error)
    else:
        raise AssertionError("Width 10 unexpectedly entered the width-12 core")
    assert "10" in mismatch_error and "12" in mismatch_error

    wrapped = MultiModalLM(core, vision_width=8)
    assert wrapped.image_projector.in_features == 8
    assert wrapped.image_projector.out_features == core.config.width == 12
    page108 = next(p for p in freeze["pages"] if p["page_id"] == "10.8")
    raw108 = (ROOT / page108["snapshot"]).read_bytes()
    assert digest(raw108) == page108["source_sha256"]
    first108 = re.findall(r"```python\n(.*?)\n```", raw108.decode(), flags=re.S)[0]
    inline108 = {}
    exec(compile(first108, "frozen-lesson-10.8-constructor", "exec"), inline108)
    wrapped_inline = inline108["MultiModalLM"](core, vision_width=8)
    assert wrapped_inline.image_projector.weight.shape == (12, 8)

    identical_features = torch.zeros(2, 8)
    collision = layer(identical_features)
    assert torch.equal(collision[0], collision[1])

    result = {
        "reviewer_task": "/root/omni_technical_coordinator/technical_10_6",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND,
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "device": "cpu",
            "dtype": "torch.float32",
            "platform": platform.platform(),
            "threads": str(torch.get_num_threads()),
        },
        "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE.read_bytes())},
        "source": {"path": page["snapshot"], "sha256": digest(raw)},
        "code_block_sha256": digest(original.encode()),
        "original_stdout": stdout,
        "original": {
            "input_shape": list(vision.shape),
            "input_seed": 0,
            "input_float32_le_sha256": tensor_digest(vision),
            "first_position_input": vision[0, 0].tolist(),
            "output_shape": list(features.shape),
            "output_float32_le_sha256": tensor_digest(features),
            "weight_shape": list(layer.weight.shape),
            "bias_shape": list(layer.bias.shape),
            "per_position_affine_formula_max_abs_error": maximum_error,
            "formula_tolerance": "absolute error < 1e-6 for float32 accumulation ordering",
            "all_outputs_finite": bool(torch.isfinite(features).all()),
        },
        "exercise": {
            "only_change": "nn.Linear(8, 12) -> nn.Linear(8, 10)",
            "stdout": exercise_stdout,
            "output_shape": list(short_features.shape),
            "weight_shape": list(exercise["projector"].weight.shape),
            "width_12_core_failure": mismatch_error,
        },
        "toy": {
            "input": toy_input.tolist(),
            "weight": toy.weight.detach().tolist(),
            "bias": toy.bias.detach().tolist(),
            "output": toy_output.detach().tolist(),
            "matches_exact_integer_result": True,
        },
        "core": {
            "config_width": core.config.width,
            "position_embedding_shape": list(core.position.weight.shape),
            "compatible_logits_shape": list(logits.shape),
            "library_projector_weight_shape": list(wrapped.image_projector.weight.shape),
            "inline_10_8_projector_weight_shape": list(wrapped_inline.image_projector.weight.shape),
            "inline_10_8_source_sha256": digest(raw108),
        },
        "collision": {
            "identical_input_shape": list(identical_features.shape),
            "outputs_exactly_equal": True,
            "scope": "A deterministic projector with fixed parameters cannot distinguish equal feature vectors; no training or quality test was run.",
        },
        "result": "All assertions passed. Exact frozen lesson program, stated width-10 exercise, arithmetic example, compatible core and mismatch were verified on CPU.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
