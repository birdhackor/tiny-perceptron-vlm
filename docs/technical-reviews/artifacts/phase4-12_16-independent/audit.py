"""Bounded CPU structural checks. No model, audio, tokenizer, or training downloads.

The Jinja template is the exact return expression of the pinned official processor.
This demonstrates serialization and numeric replacement only, never model behavior.
"""
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import jinja2
import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
processor_path = ART / "processing-qwen2-audio-v4.57.1.py"
tree = ast.parse(processor_path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2AudioProcessor")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "default_chat_template")
template_return = next(n for n in method.body if isinstance(n, ast.Return))
template = jinja2.Template(ast.literal_eval(template_return.value))


def messages(initial="接下來請用兩點回答。"):
    return [
        {"role": "user", "content": initial},
        {"role": "user", "content": [{"type": "audio", "audio": "synthetic-audio-reference"}]},
        {"role": "assistant", "content": "1. 檢查網路。2. 重新啟動 App。"},
        {"role": "user", "content": "改成一句。"},
    ]


def render(history):
    return template.render(messages=history, add_generation_prompt=True)


base = render(messages())
format_variant = render(messages("接下來請用一句回答。"))
assert "接下來請用兩點回答。" in base
assert "<|audio_bos|><|AUDIO|><|audio_eos|>" in base
assert "我的銀行 App 打不開" not in base  # no hand-written transcript was supplied
assert base.index("接下來請用兩點回答。") < base.index("<|audio_bos|>")
assert base.index("<|audio_bos|>") < base.index("1. 檢查網路。2. 重新啟動 App。") < base.index("改成一句。")
assert base.endswith("<|im_start|>assistant\n")
assert base != format_variant
assert render(messages()[1:]) == render(messages("接下來請用一句回答。")[1:])
print("OFFICIAL_TEMPLATE_SERIALIZATION", json.dumps({
    "source_method_lines": [method.lineno, method.end_lineno],
    "source_sha256": digest(processor_path),
    "retained_roles": [m["role"] for m in messages()],
    "serialized": base,
    "initial_format_changes_input": base != format_variant,
    "dropping_initial_turn_erases_format_difference": True,
    "contains_audio_boundary_and_placeholder": True,
    "contains_handwritten_transcript": False,
}, ensure_ascii=False))

# Corresponds to official forward's consecutive-token branch at lines 822-843.
# Use fixed small tensors to test embedding replacement, with no learned weights.
ids = torch.tensor([[10, 11, 99, 99, 12, 13]])
embeds = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
features = torch.tensor([[101., 102., 103., 104.], [201., 202., 203., 204.]])
mask = (ids == 99).unsqueeze(-1).expand_as(embeds)
mixed = embeds.masked_scatter(mask, features)
assert torch.equal(mixed[0, 2:4], features)
assert torch.equal(mixed[~mask], embeds[~mask])
changed_features = features + 1000
variant = embeds.masked_scatter(mask, changed_features)
assert not torch.equal(mixed[0, 2:4], variant[0, 2:4])
assert torch.equal(mixed[~mask], variant[~mask])
print("NUMERIC_FEATURE_REPLACEMENT", json.dumps({
    "input_shape": list(embeds.shape), "audio_shape": list(features.shape),
    "audio_sequence_positions": [2, 3], "mixed_embeddings": mixed.tolist(),
    "audio_positions_equal_supplied_features": True,
    "non_audio_positions_unchanged": True,
    "changing_audio_changes_only_audio_positions": True,
    "scope": "Synthetic numerical contract only; no semantic response, recognition, instruction following, or model performance was tested.",
}, ensure_ascii=False))

# Render the exact source SVG and preserve the actual renderer results.
figure = ROOT / "course/figures/new-12.16-shared-history.svg"
command = ["inkscape", str(figure), "--export-type=png", "--export-filename=" + str(ART / "shared-history.png")]
completed = subprocess.run(command, capture_output=True, text=True, check=False)
assert completed.returncode == 0
version = subprocess.run(["inkscape", "--version"], capture_output=True, text=True, check=True).stdout.strip()
print("FIGURE_RENDER", json.dumps({"command_argv": command, "exit_code": completed.returncode,
    "stdout": completed.stdout, "stderr": completed.stderr, "svg_sha256": digest(figure),
    "png_sha256": digest(ART / "shared-history.png"), "inkscape": version}, ensure_ascii=False))

environment = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
    "torch": str(torch.__version__), "torch_git": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu", "jinja2": importlib.metadata.version("jinja2"), "inkscape": version,
    "model_weights_loaded": "false", "training_performed": "false"}
(ART / "audit-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
print("ENVIRONMENT", json.dumps(environment, ensure_ascii=False))
print("PASS bounded serialization, controlled input variants, numeric substitution, and exact SVG render; no inference or training")
