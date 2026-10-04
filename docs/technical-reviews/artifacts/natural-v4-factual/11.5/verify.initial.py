"""Independent bounded CPU mechanisms and existing GPU result-record audit.

No checkpoint loads, new training schedules, downloads, or GPU calls.
"""
import ast
import hashlib
import inspect
import json
import platform
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.course_experiments.modalities import _freeze, _sequence, _vision_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene

torch.set_num_threads(2)
torch.manual_seed(42)
root = Path.cwd()
tok = ByteTokenizer()
report = {"scope": "Own bounded CPU mechanisms plus independent arithmetic on existing CUDA records; no own GPU replication",
          "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "torch_git_version": str(torch.version.git_version), "device": "cpu",
                          "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())}}

counts = {}
for width, layers in ((8, 1), (8, 2), (8, 4), (64, 2)):
    model = MultiModalLM(TinyLM(ModelConfig(width=width, layers=layers)))
    result = {}
    for mode in ("projector", "partial", "all"):
        model.requires_grad_(mode == "all")
        model.image_projector.requires_grad_(True)
        if mode == "partial":
            model.language.blocks[-1].requires_grad_(True)
        result[mode] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert result["projector"] == width * 16 + width
    assert result["partial"] == result["projector"] + 12 * width**2 + 9 * width
    result["block"] = sum(p.numel() for p in model.language.blocks[-1].parameters())
    result["submodules"] = {name: sum(p.numel() for p in sub.parameters()) for name, sub in model.named_children()}
    counts[f"w{width}-l{layers}"] = result
assert [counts["w8-l1"][x] for x in ("projector", "partial", "all")] == [136, 976, 8296]
assert [counts["w8-l2"][x] for x in ("projector", "partial", "all")] == [136, 976, 9136]
assert [counts["w64-l2"][x] for x in ("projector", "partial", "all")] == [1088, 50816, 145664]
report["parameter_counts"] = counts

# Execute the actual current CLI freeze branch, extracted by AST rather than a reimplementation.
tree = ast.parse((root / "scripts/train.py").read_text())
branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "train_model is not model")
cli_model = MultiModalLM(TinyLM(ModelConfig(width=8, layers=2)))
namespace = {"train_model": cli_model, "model": cli_model.language,
             "args": SimpleNamespace(vision_encoder=None, audio_encoder=None, freeze="partial"), "torch": torch}
exec(compile(ast.Module(body=[branch], type_ignores=[]), "actual scripts/train.py freeze branch", "exec"), namespace)
cli_names = [n for n, p in cli_model.named_parameters() if p.requires_grad]
assert any(n.startswith("language.blocks.0.") for n in cli_names)
assert any(n.startswith("language.blocks.1.") for n in cli_names)
assert not cli_model.language.embedding.weight.requires_grad
assert not cli_model.language.output.weight.requires_grad
fixed_model = MultiModalLM(TinyLM(ModelConfig(width=8, layers=2)))
_freeze(fixed_model, "partial")
fixed_names = [n for n, p in fixed_model.named_parameters() if p.requires_grad]
assert not any(n.startswith("language.blocks.0.") for n in fixed_names)
assert any(n.startswith("language.blocks.1.") for n in fixed_names)
report["freeze_scope_execution"] = {"cli_source_lines": [branch.lineno, branch.end_lineno],
                                    "cli_count": sum(p.numel() for p in cli_model.parameters() if p.requires_grad),
                                    "fixed_count": sum(p.numel() for p in fixed_model.parameters() if p.requires_grad),
                                    "cli_names": cli_names, "fixed_names": fixed_names}

# One actual image-only backward/step verifies unused audio has no gradient or allocated Adam state.
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
_freeze(model, "all")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, foreach=False)
ctx = SimpleNamespace(device="cpu", seed=42)
row = {"modality": "vision", "question": "shape?", "answer": "square"}
ids, labels, count = _sequence(row, ctx)
before_audio = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(("audio.", "audio_projector."))}
output = model(ids, labels, image=scene())
assert int((output["labels"] != -100).sum()) == count == 7
loss = masked_loss(output["logits"], output["labels"])
loss.backward()
gradient_names = [n for n, p in model.named_parameters() if p.grad is not None]
assert not any(n.startswith(("audio.", "audio_projector.")) for n in gradient_names)
assert model.image_projector.weight.grad.norm().item() > 0
optimizer.step()
for n, p in model.named_parameters():
    if n in before_audio:
        assert p.grad is None and p not in optimizer.state and torch.equal(before_audio[n], p)
    elif p.grad is not None:
        assert set(optimizer.state[p]) == {"step", "exp_avg", "exp_avg_sq"}
        assert optimizer.state[p]["exp_avg"].shape == optimizer.state[p]["exp_avg_sq"].shape == p.shape
assert torch.tensor(0., dtype=torch.float32).element_size() == 4
report["actual_image_path"] = {"loss": loss.item(), "effective_targets": count,
                                "gradient_parameter_names": gradient_names,
                                "unused_audio_parameter_elements": sum(p.numel() for n, p in model.named_parameters() if n in before_audio),
                                "audio_gradients_none_state_absent_weights_unchanged": True,
                                "adamw_state_keys_active": ["step", "exp_avg", "exp_avg_sq"], "float32_element_bytes": 4}

# Independent supervised-mask arithmetic, including the section's transparent 3 x 10 example.
logits = torch.zeros(1, 5, 4, requires_grad=True)
labels = torch.tensor([[-100, 1, 2, -100, 3]])
loss = masked_loss(logits, labels)
loss.backward()
assert torch.count_nonzero(logits.grad[0, [0, 3]]).item() == 0
assert int((labels != -100).sum()) == 3
report["transparent_target_example"] = {"targets_per_use": 3, "uses": 10, "total_targets": 30,
                                        "uniform_four_class_mean_loss": loss.item(), "ignored_logit_gradients_zero": True}

# Independently recompute every reported table entry and target budget from the original record.
result_path = root / "docs/course-experiments/results/vqa.json"
original = json.loads(result_path.read_text())
splits = _vision_records(("shape?", "color?"))
audit = {"record_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
         "recorded_runtime": {k: original[k] for k in ("device", "gpu", "python_version", "torch_version", "seed", "step_scale", "elapsed_seconds", "timing_scope")},
         "splits": {}, "variants": {}}
for split, records in splits.items():
    recorded = original["results"]["data"]["splits"][split]
    h = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert records == recorded["records"] and h == recorded["sha256"] and len(records) == recorded["count"]
    audit["splits"][split] = {"count": len(records), "sha256": h, "offsets": sorted({r["offset"] for r in records}),
                             "families": len({r["family"] for r in records})}
for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
    assert not ({r["family"] for r in splits[a]} & {r["family"] for r in splits[b]})
budgets = []
for step in range(160):
    rng = random.Random(42 + step)
    budgets.append(sum(len(tok.encode(rng.choice(splits["train"])["answer"])) + 1 for _ in range(4)))
assert sum(budgets) == 3830
for key, expected in (("projector_only", (1088, 3, 3)), ("partial", (50816, 10, 12)), ("all", (145664, 12, 9))):
    v = original["results"]["variants"][key]
    t = v["training"]
    assert t["steps"] == len(t["history"]) == 160
    assert [h["effective_targets"] for h in t["history"]] == budgets
    assert t["effective_tokens"] == t["effective_targets"] == sum(budgets)
    assert (t["trainable_parameters"], v["validation"]["correct"], v["test"]["correct"]) == expected
    evaluated = {}
    for split in ("validation", "test"):
        samples = v[split]["samples"]
        assert len(samples) == len(splits[split]) == v[split]["examples"] == 12
        correct, groups, errors = 0, {}, []
        for row, sample in zip(splits[split], samples, strict=True):
            ids = sample["generated_ids"]
            raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            match = raw == tok.encode(row["answer"])
            assert sample["target"] == row["answer"] and sample["question"] == row["question"] and sample["family"] == row["family"]
            assert sample["exact_match"] == match and sample["generated"] == tok.decode(raw)
            assert len(ids) <= 16
            correct += match
            g = groups.setdefault(row["question"], {"correct": 0, "count": 0})
            g["correct"] += match
            g["count"] += 1
            if not match: errors.append({"row": sample["row"], "target": row["answer"], "generated": sample["generated"], "family": row["family"]})
        assert correct == v[split]["correct"] and groups == v[split]["groups"]
        evaluated[split] = {"correct": correct, "count": 12, "groups": groups, "errors": errors}
    audit["variants"][key] = {"trainable_parameters": t["trainable_parameters"], "steps": 160,
                              "sample_draws": 640, "effective_targets": 3830, "replay_probability": v["replay_probability_per_example"],
                              "config": t["config"], "freeze_scope": v["freeze_scope"], **evaluated}
audit["code_hash_comparison"] = {}
for file in ("tiny_perceptron/model.py", "tiny_perceptron/multimodal.py", "scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py"):
    current = hashlib.sha256((root / file).read_bytes()).hexdigest()
    recorded = original["code_sha256"][file]
    audit["code_hash_comparison"][file] = {"current": current, "recorded": recorded, "equal": current == recorded}
report["existing_gpu_record_audit"] = audit
print(json.dumps(report, ensure_ascii=False, indent=2))
