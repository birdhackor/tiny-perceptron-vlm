"""Independent bounded CPU check; this does not rerun the historical GPU experiment."""
import contextlib
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.training import learning_rate, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source_path = ROOT / "docs/course-repair-20261017/freeze-02/sources/5.7.md"
body = source_path.read_text()
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
(OUT / "original-cell.py").write_text(code)
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, str(source_path) + ":python-cell", "exec"), {})
result = {
    "scope": "Actual execution of the exact frozen cell, plus bounded diagnostic updates. No historical model training, GPU execution, or old checkpoint loading.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": torch.get_num_threads(), "platform": platform.platform()},
    "source_sha256": sha(source_path),
    "original_cell_sha256": sha(OUT / "original-cell.py"),
    "original_cell": {"assertions_passed": True, "stdout": stdout.getvalue()},
    "branches": [],
}

# Two independent tiny runs, each containing one pre-save optimizer update and
# one post-save update in each of the two compared models.
for restore_optimizer in (True, False):
    torch.manual_seed(42)
    random.seed(1234)
    model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss = masked_loss(model(x)["logits"], y)
    loss.backward()
    optimizer.step()
    step1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.pt"
        save_checkpoint(path, model, optimizer, step=1, metadata={"chars": ["。", "狗", "看", "貓", "，"]})
        checkpoint_present_inside = path.exists()
        expected_torch = torch.rand(4)
        expected_python = [random.random() for _ in range(4)]
        loaded, payload = load_checkpoint(path, restore_rng=True)
        saved_optimizer_steps = sorted({float(s["step"]) for s in payload["optimizer"]["state"].values()})
        torch_rng_restored = torch.equal(torch.rand(4), expected_torch)
        python_rng_restored = [random.random() for _ in range(4)] == expected_python
        restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.01)
        if restore_optimizer:
            restored_optimizer.load_state_dict(payload["optimizer"])
        states_before = len(restored_optimizer.state)
        pred_before = torch.equal(model(x)["logits"], loaded(x)["logits"])
        loss_after_first = float(masked_loss(model(x)["logits"], y).detach())
        for candidate, opt in ((model, optimizer), (loaded, restored_optimizer)):
            opt.zero_grad()
            masked_loss(candidate(x)["logits"], y).backward()
            opt.step()
        weight_max = max(float((v - loaded.state_dict()[k]).abs().max()) for k, v in model.state_dict().items())
        logit_max = float((model(x)["logits"] - loaded(x)["logits"]).detach().abs().max())
        parameters = sum(v.numel() for v in step1.values())
        row = {
            "restore_optimizer": restore_optimizer,
            "input_ids": x.tolist(), "target_ids": y.tolist(), "valid_targets": int(y.numel()), "label_dtype": str(y.dtype),
            "logit_shape": list(model(x)["logits"].shape), "parameter_dtype": str(next(model.parameters()).dtype),
            "metadata": payload["metadata"], "step": payload["step"], "checkpoint_keys": sorted(payload),
            "loss_before_first_update": float(loss.detach()), "loss_after_first_update": loss_after_first,
            "weights_actually_updated": any(not torch.equal(before[k], v) for k, v in step1.items()),
            "weight_tensors_compared": len(step1), "weight_elements_compared": parameters,
            "saved_optimizer_states": len(payload["optimizer"]["state"]),
            "saved_optimizer_state_keys": sorted(next(iter(payload["optimizer"]["state"].values()))),
            "saved_optimizer_steps": saved_optimizer_steps,
            "loaded_optimizer_states_before_next_update": states_before,
            "restored_steps_after_next_update": sorted({float(s["step"]) for s in restored_optimizer.state.values()}),
            "equal_logits_before_update": pred_before,
            "equal_logits_after_update": torch.equal(model(x)["logits"], loaded(x)["logits"]),
            "max_logit_difference_after_update": logit_max,
            "max_weight_difference_after_update": weight_max,
            "torch_rng_restored": torch_rng_restored, "python_rng_restored": python_rng_restored,
            "dropout_modules": sum(isinstance(m, torch.nn.Dropout) for m in model.modules()),
            "checkpoint_present_inside_context": checkpoint_present_inside,
        }
    row["temporary_directory_removed"] = not Path(directory).exists()
    assert row["weights_actually_updated"] and pred_before
    assert torch_rng_restored and python_rng_restored
    assert row["temporary_directory_removed"]
    assert row["equal_logits_after_update"] == restore_optimizer
    if restore_optimizer:
        assert weight_max == 0.0 and logit_max == 0.0
    else:
        assert weight_max > 0 and logit_max > 0
    result["branches"].append(row)

# This block is an executed audit of the original report and its exact bound
# implementation. It does not execute that implementation or any old model.
report_path = ROOT / "docs/course-experiments/results/text_foundation.json"
raw = json.loads(report_path.read_text())
revision = raw["revision"]
historical = {}
for rel, first, last in (("scripts/course_experiments/text.py", 205, 259), ("scripts/course_experiments/common.py", 44, 52)):
    command = ["git", "show", revision + ":" + rel]
    data = subprocess.check_output(command, cwd=ROOT)
    digest = hashlib.sha256(data).hexdigest()
    assert digest == raw["code_sha256"][rel]
    historical[rel] = {"command": " ".join(command), "sha256": digest, "matches_original_report_binding": True}
    excerpt = "\n".join(data.decode().splitlines()[first-1:last]) + "\n"
    (OUT / ("historical-" + Path(rel).stem + "-excerpt.txt")).write_text(f"Revision {revision}; {rel}; original lines {first}-{last}; full-file SHA256 {digest}\n\n" + excerpt)

result["archival_audit"] = {
    "method": "JSON parsing, exact git object retrieval, SHA-256 comparison and source reading; not a rerun of the historical GPU experiment",
    "original_report": report_path.relative_to(ROOT).as_posix(), "original_report_sha256": sha(report_path),
    "revision": revision,
    "environment_recorded_by_original_run": {k: raw[k] for k in ("device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status")},
    "original_resume_result": raw["results"]["resume"],
    "original_main_training_steps": raw["results"]["training"]["steps"],
    "original_probe_model": {"vocab_size": 264, "width": 16, "layers": 1, "heads": 1, "max_length": 64, "optimizer": "AdamW", "peak_lr": 0.003, "batch_draws_per_update": 4, "clipping_max_norm": 1.0},
    "comparison_denominators": {"uninterrupted_updates": 80, "saved_after_update": 40, "resumed_updates": 40, "sample_draws_uninterrupted": 320, "sample_draws_resumed_suffix": 160, "seed": 42, "repetitions_recorded": 1, "heldout_quality_claim": False},
    "historical_source_binding": historical,
    "shared_model_and_checkpoint_sources_match_original": {rel: sha(ROOT/rel) == raw["code_sha256"][rel] for rel in ("tiny_perceptron/model.py", "tiny_perceptron/training.py")},
    "checkpoint_hashes_recorded_only_not_reopened": [x for x in raw["artifacts"] if x["path"] in ("resume-middle.pt", "resume-final.pt")],
}
assert result["archival_audit"]["original_resume_result"]["total_steps"] == 80
assert result["archival_audit"]["original_resume_result"]["max_weight_difference"] == 0
assert result["archival_audit"]["original_main_training_steps"] == 600
assert all(result["archival_audit"]["shared_model_and_checkpoint_sources_match_original"].values())
result["schedule_numeric_check"] = {"step_index": 40, "original_total": 80, "changed_total": 120, "original_lr": learning_rate(40, 80, peak=0.003), "changed_lr": learning_rate(40, 120, peak=0.003), "scope": "Function evaluation only; no optimizer update or training in this calculation"}
assert result["schedule_numeric_check"]["original_lr"] != result["schedule_numeric_check"]["changed_lr"]
print(json.dumps(result, ensure_ascii=False, indent=2))
