"""Bounded CPU checks: original exercise variation, state roundtrip, stochastic continuation,
and actual parsing of the original GPU JSON. No original training rerun."""
import ast
import contextlib
import hashlib
import io
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.training import learning_rate, load_checkpoint, save_checkpoint, seed_everything

torch.set_num_threads(1)
assert not torch.cuda.is_available() and torch.version.cuda is None
torch.set_default_device("cpu")

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def tensor_id(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": sha(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())}

raw = (HERE / "original/fence-1.py").read_text()
assert raw.count('    restored_optimizer.load_state_dict(payload["optimizer"])\n') == 1
assert raw.count('    assert torch.equal(model(x)["logits"], loaded(x)["logits"])\n') == 2
changed = raw.replace('    restored_optimizer.load_state_dict(payload["optimizer"])\n',
                      '    # Exercise: omit optimizer-state restoration.\n')
position = changed.rindex('    assert torch.equal(model(x)["logits"], loaded(x)["logits"])\n')
changed = changed[:position] + changed[position:].replace(
    '    assert torch.equal(model(x)["logits"], loaded(x)["logits"])\n',
    '    assert not torch.equal(model(x)["logits"], loaded(x)["logits"])\n'
    '    print("exercise_after_update_max_logit_difference", float((model(x)["logits"] - loaded(x)["logits"]).detach().abs().max()))\n', 1)
(HERE / "exercise-no-optimizer.py").write_text(changed)
namespace = {"__name__": "__main__"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(changed, str(HERE / "exercise-no-optimizer.py"), "exec"), namespace)
exercise_difference = float((namespace["model"](namespace["x"])["logits"] - namespace["loaded"](namespace["x"])["logits"]).detach().abs().max())
assert exercise_difference > 0
print(captured.getvalue(), end="")

choices = [([1, 2, 3], [2, 3, 4]), ([3, 2, 1], [2, 1, 0]), ([4, 1, 2], [1, 2, 3])]

def update(model, optimizer, step, total=4):
    selected = random.randrange(len(choices))
    x, y = (torch.tensor([value]) for value in choices[selected])
    embeddings = F.dropout(model.embedding(x), p=0.25, training=True)
    rate = learning_rate(step, total, peak=0.01, warmup=1)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad()
    loss = masked_loss(model(embeddings=embeddings)["logits"], y)
    loss.backward()
    optimizer.step()
    return {"step": step + 1, "sample_index": selected, "lr": rate,
            "loss": float(loss.detach()), "stochastic_embeddings": tensor_id(embeddings)}

with tempfile.TemporaryDirectory(prefix="factual-5_7-") as directory:
    path = Path(directory) / "micro-state.pt"
    seed_everything(42)
    model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    prefix = [update(model, optimizer, step) for step in range(2)]
    save_checkpoint(path, model, optimizer, step=2, metadata={"chars": ["。", "狗", "看", "貓", "，"], "schedule_steps": 4, "peak_lr": 0.01})
    checkpoint_identity = {"role": "generated tiny CPU roundtrip output and branch input", "bytes": path.stat().st_size, "sha256": sha(path.read_bytes()), "temporary": True, "persistent_weights": False}
    saved_model = {name: tensor_id(value) for name, value in model.state_dict().items()}
    baseline_history = [update(model, optimizer, step) for step in range(2, 4)]
    final = {name: value.detach().clone() for name, value in model.state_dict().items()}
    variants = {}
    for name, restore_rng, restore_optimizer, total in [
        ("full", True, True, 4), ("missing_optimizer", True, False, 4),
        ("missing_rng", False, True, 4), ("changed_schedule", True, True, 8),
    ]:
        loaded, payload = load_checkpoint(path, restore_rng=restore_rng)
        assert saved_model == {key: tensor_id(value) for key, value in loaded.state_dict().items()}
        restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.5)
        if restore_optimizer:
            restored_optimizer.load_state_dict(payload["optimizer"])
            assert restored_optimizer.param_groups[0]["lr"] == payload["optimizer"]["param_groups"][0]["lr"]
        else:
            for group in restored_optimizer.param_groups:
                group["lr"] = payload["optimizer"]["param_groups"][0]["lr"]
        if not restore_rng:
            random.seed(7)
            torch.manual_seed(7)
        history = [update(loaded, restored_optimizer, step, total) for step in range(2, 4)]
        differences = {key: float((value - loaded.state_dict()[key]).abs().max()) for key, value in final.items()}
        variants[name] = {"history": history, "max_parameter_difference": max(differences.values()), "every_tensor_equal": all(torch.equal(value, loaded.state_dict()[key]) for key, value in final.items())}
    assert variants["full"]["every_tensor_equal"]
    assert variants["full"]["history"] == baseline_history
    assert all(not variants[name]["every_tensor_equal"] for name in ("missing_optimizer", "missing_rng", "changed_schedule"))
    # Explicitly test saved CPU Torch/Python random-state restoration after constructor RNG use.
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    expected_random = (torch.rand(4), [random.random() for _ in range(4)])
    _, restored_payload = load_checkpoint(path, restore_rng=True)
    observed_random = (torch.rand(4), [random.random() for _ in range(4)])
    assert torch.equal(expected_random[0], observed_random[0]) and expected_random[1] == observed_random[1]
    # Optimizer.load_state_dict may reuse loaded CPU state tensors. Summarize
    # an unconsumed roundtrip payload, not a branch payload already advanced.
    payload = restored_payload
    assert all(float(value["step"]) == 2.0 for value in payload["optimizer"]["state"].values())
    state_summary = {"payload_keys": sorted(payload), "step": payload["step"], "config": payload["config"], "metadata": payload["metadata"],
                     "optimizer_state_count": len(payload["optimizer"]["state"]), "optimizer_state_keys": sorted(next(iter(payload["optimizer"]["state"].values()))),
                     "optimizer_steps": sorted({float(value["step"]) for value in payload["optimizer"]["state"].values()}),
                     "torch_rng": tensor_id(payload["torch_rng"]), "python_rng_state_sha256": sha(repr(payload["python_rng"]).encode()),
                     "cuda_rng_count": len(payload["cuda_rng"]), "mps_rng_is_none": payload["mps_rng"] is None,
                     "first_optimizer_state_values": {key: value.tolist() for key, value in next(iter(payload["optimizer"]["state"].values())).items()},
                     "cpu_rng_roundtrip_equal": True}
assert not path.exists()

original_path = HERE / "inputs/docs/course-experiments/results/text_foundation.json"
original = json.loads(original_path.read_text())
resume = original["results"]["resume"]
assert resume == {"total_steps": 80, "saved_step": 40, "max_weight_difference": 0.0,
                  "schedule_total_preserved": True, "optimizer_and_rng_restored": True, "tolerance": 1e-5}
assert original["device"] == "cuda" and original["gpu"] == "NVIDIA L4"
assert original["step_scale"] == 1.0 and original["status"] == "completed"
assert original["results"]["training"]["steps"] == 600
assert "resume-middle.pt" in original["hf"]["verified_checkpoints"]
assert "resume-final.pt" in original["hf"]["verified_checkpoints"]
hash_checks = {}
for name in ("scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/training.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/data.py"):
    actual = sha((HERE / "historical" / name).read_bytes())
    assert actual == original["code_sha256"][name]
    hash_checks[name] = actual
# Inspect original toy-input constructor without training or fetching data.
prepare_raw = subprocess.check_output(["git", "show", f"{original['revision']}:scripts/prepare_data.py"], cwd=ROOT)
prepare_snapshot = HERE / "historical/scripts/prepare_data.py"
prepare_snapshot.write_bytes(prepare_raw)
prepare_tree = ast.parse(prepare_raw)
generate_node = next(node for node in prepare_tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_records")
generator_ns = {}
exec(compile(ast.Module(body=[generate_node], type_ignores=[]), "historical:generate_records", "exec"), generator_ns)
records = generator_ns["generate_records"]("toy-text")
from tiny_perceptron.data import ByteTokenizer, shifted
examples = []
tok = ByteTokenizer()
for row in records:
    ids = [tok.bos_id] + tok.encode(row["text"]) + [tok.eos_id]
    for start in range(0, len(ids) - 1, 64):
        examples.append(shifted(ids[start:start + 65]))
assert len(records) == len(examples) == 12
historical_inspection = {"raw_result_sha256": sha(original_path.read_bytes()), "revision": original["revision"], "resume": resume,
    "device": original["device"], "gpu": original["gpu"], "torch_version": original["torch_version"], "seed": original["seed"],
    "verified_code_hashes": hash_checks, "prepare_data_revision": original["revision"], "prepare_data_sha256": sha(prepare_raw),
    "denominators": {"compared_final_trajectories": 2, "planned_steps_per_trajectory": 80, "saved_step": 40, "resumed_updates": 40,
                     "actual_update_calls_in_probe": 120, "batch_examples_per_update": 4, "sample_draws_in_program": 480, "synthetic_source_records": 12,
                     "prepared_examples": len(examples), "example_valid_target_lengths": [len(y) for x, y in examples]},
    "weight_identity_only": [item for item in original["artifacts"] if item["path"].startswith("resume-")],
    "evidence_scope": "Checked recorded result and hash-matched implementation. No original GPU execution or independent GPU final-weight recomputation; no model/data download. Original result max-difference is an observed historical value; its acceptance threshold was 1e-5. CPU microprobes are separate.",
    "separate_foundation_training_steps": original["results"]["training"]["steps"]}

report = {"environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())},
    "original_exercise_without_optimizer": {"first_logit_equality_assertion_passed": True, "second_logit_equality": False, "after_update_max_logit_difference": exercise_difference},
    "checkpoint_identity": checkpoint_identity, "temporary_checkpoint_removed": not path.exists(), "saved_state": state_summary,
    "bounded_probe": {"width": 8, "vocab_size": 5, "planned_steps": 4, "prefix": prefix, "baseline_tail": baseline_history, "variants": variants},
    "historical_original_json": historical_inspection}
(HERE / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"exercise_difference": exercise_difference, "bounded_variant_max_differences": {name: value["max_parameter_difference"] for name, value in variants.items()}, "cpu_rng_roundtrip": True, "original_gpu_resume": resume, "original_gpu_retrained": False, "temporary_checkpoint_removed": True}, ensure_ascii=False, indent=2))
