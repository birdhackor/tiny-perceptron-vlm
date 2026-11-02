"""Execute the frozen 12.13 code verbatim, then inspect narrowly scoped dependencies."""

import ast
import contextlib
import hashlib
import io
import json
import math
from datetime import UTC, datetime
from pathlib import Path
import platform
import re
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("execution.json")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "12.13")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


raw = (ROOT / page["snapshot"]).read_bytes()
assert digest(raw) == page["source_sha256"]
body = raw.decode("utf-8")
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
chapter = (ROOT / page["source"]).read_text()
current = re.search(r"^## 12\.13 .*?(?=^## |\Z)", chapter, re.M | re.S)[0]
assert current == body
notebook = json.loads((ROOT / page["notebook"]).read_bytes())
matches = [i for i, c in enumerate(notebook["cells"])
           if c["cell_type"] == "code" and "".join(c["source"]).rstrip("\n") == blocks[0].rstrip("\n")]
assert len(matches) == 1

torch.set_num_threads(1)
namespace = {"__name__": "__reviewed_12_13__"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(blocks[0], page["snapshot"] + ":python-block-1", "exec"), namespace)
stdout = captured.getvalue()
assert stdout == "時間框與頻帶 21 16\n先後序列相同 False\n平均摘要近似相同 True\n"
sequence = namespace["sequence"]
reordered = namespace["reordered"]
wave = namespace["wave"]
mean_a, mean_b = sequence.mean(0), reordered.mean(0)
mean_error = (mean_a - mean_b).abs()
limit = 1e-8 + 1e-5 * mean_b.abs()
assert torch.equal(sequence, reordered.flip(0))
assert sequence.shape == (21, 16)
assert (mean_error <= limit).all()

from tiny_perceptron.multimodal import log_mel, mel_filter_bank
from tiny_perceptron.selftrained.model import AudioEncoder

repo_sequence = log_mel(wave).T
assert torch.equal(namespace["bank"], mel_filter_bank())
assert torch.equal(sequence, repo_sequence)
frequencies = torch.fft.rfftfreq(1600, d=1 / 16000)
peaks = [float(frequencies[torch.fft.rfft(part).abs().argmax()])
         for part in wave.split(1600)]
assert peaks == [440.0, 880.0]
reverse_wave_features = log_mel(wave.flip(0)).T

# Structural check of the existing A encoder. No checkpoint or training is used.
torch.manual_seed(0)
encoder = AudioEncoder(width=8).eval()
observed = {}
h1 = encoder.sequence.register_forward_hook(
    lambda module, args, output: observed.update(sequence=output[0][0].detach()))
h2 = encoder.projection.register_forward_pre_hook(
    lambda module, args: observed.update(pooled=args[0].detach()))
with torch.no_grad():
    tokens, logits = encoder(torch.linspace(-1, 1, 4000).reshape(100, 40))
h1.remove()
h2.remove()
length = len(observed["sequence"])
intervals = [(i * length // 16, ((i + 1) * length + 15) // 16) for i in range(16)]
manual = torch.stack([observed["sequence"][a:b].mean(0) for a, b in intervals])
assert torch.allclose(observed["pooled"], manual)
assert tokens.shape == (16, 8)
assert encoder.time_position.weight.shape == (16, 8)

# Read and execute only the needed 12.17 imports, constants, bank and local_tokens.
dependency = next(p for p in freeze["pages"] if p["page_id"] == "12.17")
dependency_raw = (ROOT / dependency["snapshot"]).read_bytes()
assert digest(dependency_raw) == dependency["source_sha256"]
dependency_code = re.findall(r"```python\n(.*?)```", dependency_raw.decode(), re.S)[0]
tree = ast.parse(dependency_code)
needed = {"rate", "duration_ms", "chunk_samples", "mel_edges", "hz_edges", "hz",
          "left", "middle", "right", "bank"}
selected = []
for node in tree.body:
    targets = {n.id for target in getattr(node, "targets", [])
               for n in ast.walk(target) if isinstance(n, ast.Name)}
    if isinstance(node, (ast.Import, ast.ImportFrom)) or (targets and targets <= needed):
        selected.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name == "local_tokens":
        selected.append(node)
dep_namespace = {"__name__": "__needed_12_17_slice__"}
exec(compile(ast.Module(body=selected, type_ignores=[]), dependency["snapshot"], "exec"), dep_namespace)
local_tokens = dep_namespace["local_tokens"]
local_wave = torch.zeros(6400, dtype=torch.float64)
local_wave[:3200] = 0.5 * torch.sin(2 * math.pi * 440 * torch.arange(3200, dtype=torch.float64) / 16000)
before = local_tokens(local_wave, 1000)
perturbed = local_wave.clone()
perturbed[3200:] = 0.75
after = local_tokens(perturbed, 1000)
assert torch.equal(before[0]["content"], after[0]["content"])
assert not torch.equal(before[1]["content"], after[1]["content"])
selected_text = "\n\n".join(ast.get_source_segment(dependency_code, n) for n in selected)

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-12-13-v1/verify.py",
    "environment": {"python": sys.version, "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version,
                    "platform": platform.platform(), "device": "cpu", "dtype": str(sequence.dtype),
                    "threads": str(torch.get_num_threads()), "baseline_encoder_seed": "0"},
    "frozen_source": {"path": page["snapshot"], "sha256": digest(raw),
                      "code_block_sha256": digest(blocks[0].encode()),
                      "current_section_matches": True,
                      "notebook_matching_cell": matches[0],
                      "notebook_sha256": digest((ROOT / page["notebook"]).read_bytes())},
    "embedded_code_stdout": stdout,
    "embedded_code_details": {
        "sample_times": len(namespace["time"]), "wave_samples": len(wave),
        "samples_per_tone": 1600, "seconds_per_tone": 1600 / 16000,
        "tone_fft_peak_hz": peaks, "bank_shape": list(namespace["bank"].shape),
        "spectrum_shape": list(namespace["spectrum"].shape),
        "sequence_shape": list(sequence.shape), "all_values_finite": bool(sequence.isfinite().all()),
        "exact_frame_permutation": bool(torch.equal(sequence, reordered.flip(0))),
        "mean_max_abs_error": float(mean_error.max()),
        "allclose_rtol": 1e-5, "allclose_atol": 1e-8,
        "max_error_over_elementwise_tolerance": float((mean_error / limit).max()),
        "inline_bank_matches_current_repository": True,
        "inline_sequence_matches_current_repository": True,
        "reverse_wave_recomputed_equals_flipped_frames": bool(torch.allclose(reverse_wave_features, reordered)),
        "reverse_wave_recomputed_max_abs_difference": float((reverse_wave_features - reordered).abs().max())},
    "existing_A_encoder": {
        "source_path": "tiny_perceptron/selftrained/model.py",
        "source_sha256": digest((ROOT / "tiny_perceptron/selftrained/model.py").read_bytes()),
        "input_shape": [100, 40], "gru_sequence_shape": list(observed["sequence"].shape),
        "pooled_shape": list(observed["pooled"].shape), "tokens_shape": list(tokens.shape),
        "logits_shape": list(logits.shape), "position_weight_shape": list(encoder.time_position.weight.shape),
        "pool_intervals_in_gru_sequence_half_open": intervals,
        "pool_max_difference_vs_explicit_means": float((observed["pooled"] - manual).abs().max()),
        "scope": "Structural ordering and sixteen slots only. Bidirectional GRU mixes context before pooling; no locality or trained quality is established."},
    "needed_12_17_slice": {
        "snapshot": dependency["snapshot"], "snapshot_sha256": digest(dependency_raw),
        "selected_code_sha256": digest(selected_text.encode()),
        "selected_code_block_line_ranges": [[n.lineno, n.end_lineno] for n in selected],
        "selected_code": selected_text,
        "first_chunk_unchanged_by_second_chunk_perturbation": True,
        "second_chunk_changes": True,
        "supports_ms": [[t["start"], t["end"]] for t in before],
        "anchors_ms": [t["anchor"] for t in before],
        "scope": "Only the explicitly linked per-block preprocessing and physical-time formula are checked; no other 12.17 claims or model capability are reviewed."},
    "limits": "Deterministic CPU forward calculations. No GPU, optimization, checkpoint loading, ASR, or generalization test. No evidence of trained task quality."
}
with OUT.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"output": OUT.relative_to(ROOT).as_posix(),
                  "sha256": digest(OUT.read_bytes()), "stdout": stdout,
                  "mean_max_abs_error": result["embedded_code_details"]["mean_max_abs_error"],
                  "existing_A_tokens_shape": list(tokens.shape),
                  "local_slice_perturbation_pass": True}, ensure_ascii=False))
