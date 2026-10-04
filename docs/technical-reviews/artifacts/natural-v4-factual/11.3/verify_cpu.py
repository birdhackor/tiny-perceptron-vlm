"""Independent bounded factual probes; no training schedule or GPU benchmark."""
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder, expand_modalities, scene
from tiny_perceptron.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parent
torch.set_num_threads(2)
tok = ByteTokenizer()
report = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "torch_git": torch.version.git_version, "device": "cpu",
                          "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads()},
          "scope": "Own CPU graph/record/checkpoint evaluation only; no 300-step retraining or GPU timing reproduction."}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def independent_nll(model, ids, labels, image=None):
    out = model(ids, labels, image=image)
    valid = out["labels"] != -100
    scores = out["logits"][valid].double()
    truth = out["labels"][valid]
    terms = torch.logsumexp(scores, dim=-1) - scores.gather(1, truth[:, None])[:, 0]
    actual = masked_loss(out["logits"], out["labels"])
    assert abs(float(actual) - float(terms.mean())) < 2e-6
    return float(terms.sum()), len(terms), out

torch.manual_seed(0)
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
model.requires_grad_(False)
model.image_projector.requires_grad_(True)
answer = tok.encode("red square") + [tok.eos_id]
prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)
out = model(ids, labels, image=scene())
loss = masked_loss(out["logits"], out["labels"])
before = {n: p.detach().clone() for n, p in model.named_parameters()}
loss.backward()
grad = model.image_projector.weight.grad
assert grad.norm() > 0 and model.language.embedding.weight.grad is None
assert all(p.grad is None for p in model.language.parameters())
assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
assert len(answer) == 11 and len(tok.encode("red square")) == 10
assert out["logits"].shape == (1, 30, 264)
assert (out["labels"] != -100).sum() == 11
assert out["labels"][0, 19:].tolist() == answer
assert (out["labels"][0, :19] == -100).all()
manual_sum, count, _ = independent_nll(model, ids, labels, scene())
report["tiny_demo"] = {"loss": float(loss.detach()), "manual_nll_sum": manual_sum,
                       "targets": count, "logits_shape": list(out["logits"].shape),
                       "labels": out["labels"].tolist(), "answer_ids": answer,
                       "trainable": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
                       "projector_grad_norm": float(grad.norm()),
                       "squared_sum_norm": float(grad.square().sum().sqrt()),
                       "printed_booleans": [True, True], "all_language_grad_none": True,
                       "no_parameter_updated_by_backward": True}
assert torch.allclose(grad.norm(), grad.square().sum().sqrt())

# A local derivative of the actual frozen-language computation, in float64.
double = MultiModalLM(TinyLM(ModelConfig(width=8))).double()
double.load_state_dict(model.state_dict())
double.requires_grad_(False)
double.image_projector.requires_grad_(True)
d_loss = masked_loss(double(ids, labels, image=scene().double())["logits"], out["labels"])
d_loss.backward()
dgrad = double.image_projector.weight.grad
flat_index = int(dgrad.abs().argmax())
index = (flat_index // dgrad.shape[1], flat_index % dgrad.shape[1])
epsilon = 1e-5
with torch.no_grad():
    w = double.image_projector.weight
    base = float(w[index])
    values = []
    for sign in (1, -1):
        w[index] = base + sign * epsilon
        o = double(ids, labels, image=scene().double())
        values.append(float(masked_loss(o["logits"], o["labels"])))
    w[index] = base
finite = (values[0] - values[1]) / (2 * epsilon)
assert math.isclose(finite, float(dgrad[index]), rel_tol=1e-6, abs_tol=1e-8)
report["derivative_probe"] = {"index": index, "epsilon": epsilon,
                              "autograd": float(dgrad[index]), "central_difference": finite,
                              "loss_plus_minus": values, "tolerance": "rel=1e-6, abs=1e-8"}

# Fully disabled learning path must fail rather than silently simulate training.
torch.manual_seed(0)
disabled = MultiModalLM(TinyLM(ModelConfig(width=8)))
disabled.requires_grad_(False)
dout = disabled(ids, labels, image=scene())
dloss = masked_loss(dout["logits"], dout["labels"])
trainable = sum(p.numel() for p in disabled.parameters() if p.requires_grad)
assert trainable == 0 and not dloss.requires_grad
try:
    dloss.backward()
except RuntimeError as e:
    failure = str(e)
else:
    raise AssertionError("all-frozen backward unexpectedly succeeded")
report["all_frozen_exercise"] = {"trainable": trainable, "loss_requires_grad": False,
                                  "observed_error": failure, "expected_failure": True}

# Expansion prerequisite: original example and exercise have exactly one shift.
embedding = nn.Embedding(264, 8)
xp, yp = expand_modalities(torch.tensor([1, 5, 4, 73, 2]),
                          torch.tensor([-100, -100, -100, 73, 2]),
                          embedding, {5: torch.randn(3, 8)}, {5})
assert tuple(xp.shape) == (1, 6, 8) and yp.tolist() == [[-100] * 4 + [73, 2]]
report["prerequisite_expansion"] = {"shape": list(xp.shape), "targets": yp.tolist()}

records = {}
for name in ("projector", "encoders", "sft"):
    p = ROOT / f"docs/course-experiments/results/{name}.json"
    records[name] = json.loads(p.read_text())
report["original_records"] = {n: {"path": f"docs/course-experiments/results/{n}.json",
                                     "sha256": digest(ROOT / f"docs/course-experiments/results/{n}.json"),
                                     "revision": d["revision"], "device": d["device"], "seed": d["seed"],
                                     "python": d["python_version"], "torch": d["torch_version"],
                                     "gpu": d["gpu"], "elapsed_seconds": d["elapsed_seconds"],
                                     "timing_scope": d.get("timing_scope", "not recorded")}
                              for n, d in records.items()}
code_paths = ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py",
              "tiny_perceptron/training.py", "scripts/course_experiments/modalities.py",
              "scripts/course_experiments/text.py", "scripts/course_experiments/common.py"]
report["code_bindings"] = {}
for path in code_paths:
    actual = digest(ROOT / path)
    matches = {n: d["code_sha256"][path] == actual for n, d in records.items()}
    assert all(matches.values())
    report["code_bindings"][path] = {"sha256": actual, "matches_original_records": matches}

projector = records["projector"]["results"]
parts = {k: v["records"] for k, v in projector["data"]["splits"].items()}
for split, offsets in (("train", [-2, -1, 0]), ("validation", [1]), ("test", [2])):
    rows = parts[split]
    assert len(rows) == (18 if split == "train" else 6)
    assert sorted({r["offset"] for r in rows}) == offsets
    assert {(r["color"], r["shape"]) for r in rows} == {(c, s) for c in ["red", "green", "blue"] for s in ["circle", "square"]}
    assert all(r["answer"] == r["color"] + " " + r["shape"] and r["question"] == "describe" for r in rows)
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == projector["data"]["splits"][split]["sha256"]
assert not (set(r["family"] for r in parts["train"]) & set(r["family"] for r in parts["test"]))
assert not (set(r["family"] for r in parts["train"]) & set(r["family"] for r in parts["validation"]))
assert not (set(r["family"] for r in parts["validation"]) & set(r["family"] for r in parts["test"]))
history = projector["training"]["history"]
assert len(history) == 300 and [r["step"] for r in history] == list(range(1, 301))
counts = []
for step in range(300):
    rng = random.Random(42 + step)
    batch = [rng.choice(parts["train"]) for _ in range(4)]
    count = sum(len(r["answer"].encode()) + 1 for r in batch)
    assert count == history[step]["effective_targets"]
    assert math.isfinite(history[step]["loss"]) and math.isfinite(history[step]["grad_norm"])
    counts.append(count)
assert sum(counts) == 14408 == projector["training"]["effective_targets"]
report["sampling_recalculation"] = {"seed": 42, "updates": 300, "batch": 4, "sampled_examples": 1200,
                                    "answer_targets_total": sum(counts), "per_step_min": min(counts),
                                    "per_step_max": max(counts), "all_300_history_counts_equal": True,
                                    "nonzero_gradient_steps": sum(r["grad_norm"] > 0 for r in history),
                                    "split_counts": {k: len(v) for k, v in parts.items()},
                                    "evaluation_targets_per_split": {k: sum(len(r["answer"].encode()) + 1 for r in v) for k, v in parts.items()}}

def record_generations(evaluation, target_key="target", exact_key="exact_match"):
    matches, ended = 0, 0
    for row in evaluation["samples"]:
        gen = row["generated_ids"]
        raw = gen[:gen.index(2)] if 2 in gen else gen
        exact = raw == tok.encode(row[target_key])
        assert exact == row[exact_key] and (2 in gen) == row["eos"]
        assert tok.decode(raw) == row["generated"]
        matches += exact
        ended += 2 in gen
    return {"examples": len(evaluation["samples"]), "correct": matches, "eos": ended,
            "target_tokens": sum(len(tok.encode(row[target_key])) + 1 for row in evaluation["samples"])}

report["record_recalculation"] = {k: record_generations(projector[k]) for k in ("before", "validation", "test")}
for k, calc in report["record_recalculation"].items():
    assert calc["correct"] == projector[k]["correct"] == 0
    assert calc["target_tokens"] == projector[k]["effective_tokens"] == 72
    assert calc["eos"] / calc["examples"] == projector[k]["eos_rate"]
report["test_samples"] = [{"target": r["target"], "generated": r["generated"], "generated_ids": r["generated_ids"], "eos": r["eos"]} for r in projector["test"]["samples"]]

public = ROOT / "outputs/natural-r4-public-checkpoints/public-models"
report["checkpoint_bindings"] = {}
for n, f in (("projector", "model.pt"), ("sft", "model.pt"), ("encoders", "vision.pt"), ("encoders", "audio.pt")):
    release_path = ROOT / f"docs/course-experiments/public-releases/{n}.json"
    release = json.loads(release_path.read_text())["release"]
    entry = next(e for e in release["public_manifest"]["files"] if e["output"] == f)
    p = public / n / f
    assert digest(p) == entry["sha256"] and p.stat().st_size == entry["bytes"]
    report["checkpoint_bindings"][f"{n}/{f}"] = {"path": str(p.relative_to(ROOT)), "sha256": digest(p),
                                                    "release_manifest": str(release_path.relative_to(ROOT)),
                                                    "manifest_sha256": digest(release_path), "public_revision": release["revision"]}

final, final_payload = load_checkpoint(public / "projector/model.pt", "cpu")
sft, sft_payload = load_checkpoint(public / "sft/model.pt", "cpu")
vision_payload = torch.load(public / "encoders/vision.pt", map_location="cpu", weights_only=True)
audio_payload = torch.load(public / "encoders/audio.pt", map_location="cpu", weights_only=True)
assert final.language.config.width == 64 and final.language.config.layers == 2
assert sum(p.numel() for p in final.parameters()) == 145664
assert final.image_projector.weight.numel() == 64 * 16 and final.image_projector.bias.numel() == 64
unchanged = {k: torch.equal(v, final.language.state_dict()[k]) for k, v in sft.state_dict().items()}
assert all(unchanged.values()) and projector["language_weights_unchanged"]
assert all(torch.equal(v, final.vision.state_dict()[k]) for k, v in vision_payload["encoder"].items())
torch.manual_seed(42)
initial_language, _ = load_checkpoint(public / "sft/model.pt", "cpu")
initial = MultiModalLM(initial_language)
initial.vision.load_state_dict(vision_payload["encoder"])
initial.audio.load_state_dict(audio_payload["encoder"])
changed = not torch.equal(initial.image_projector.weight, final.image_projector.weight)
assert changed
report["checkpoint_weights"] = {"parameters": sum(p.numel() for p in final.parameters()),
                               "projector_parameters": 64 * 16 + 64,
                               "language_equal_tensor_count": len(unchanged), "all_language_tensors_equal": True,
                               "vision_equal_original_encoder": True, "projector_changed_from_seed42_initialization": changed,
                               "language_config": sft_payload["config"]}

def sequence(row, include_answer=True):
    prefix = [tok.bos_id, tok.user_id, tok.image_id] + tok.encode(row["question"]) + [tok.eos_id, tok.assistant_id]
    tail = tok.encode(row["answer"]) + [tok.eos_id] if include_answer else []
    return torch.tensor(prefix + tail), torch.tensor([-100] * len(prefix) + tail)

def own_greedy(model, row):
    prefix, _ = sequence(row, False)
    current = prefix.clone()
    image = scene(row["color"], row["shape"], offset=row["offset"])
    generated = []
    for _ in range(16):
        full = torch.cat([current, torch.tensor([tok.pad_id])])
        scores = model(full, image=image)["logits"][0, -1]
        token = int(scores.argmax())
        generated.append(token)
        current = torch.cat([current, torch.tensor([token])])
        if token in [tok.eos_id, tok.image_id, tok.audio_id]:
            break
    return generated

with torch.no_grad():
    for m in (initial, final): m.eval()
    cpu = {}
    for side, m, rows in (("before", initial, parts["test"]), ("validation", final, parts["validation"]), ("test", final, parts["test"])):
        total, targets, generated = 0.0, 0, []
        for i, row in enumerate(rows):
            x, y = sequence(row)
            v, n, o = independent_nll(m, x, y, scene(row["color"], row["shape"], offset=row["offset"]))
            assert o["logits"].shape[1] == len(x) + 14
            total += v; targets += n
            gen = own_greedy(m, row)
            assert gen == projector[side]["samples"][i]["generated_ids"]
            generated.append(gen)
        nll = total / targets
        assert abs(nll - projector[side]["mean_token_nll"]) < 2e-5
        cpu[side] = {"nll_sum": total, "targets": targets, "nll": nll,
                     "original_cuda_nll": projector[side]["mean_token_nll"], "all_greedy_ids_equal_original": True}
    rng = random.Random(41)
    probe_rows = [rng.choice(parts["train"]) for _ in range(4)]
    probe = {}
    for label, m in (("initial", initial), ("final", final)):
        totals = [independent_nll(m, *sequence(row), scene(row["color"], row["shape"], offset=row["offset"]))[:2] for row in probe_rows]
        probe[label] = sum(v for v, _ in totals) / sum(n for _, n in totals)
        assert abs(probe[label] - projector["training"][f"{label}_loss"]) < 2e-5
    probe["families"] = [r["family"] for r in probe_rows]
    probe["targets"] = sum(len(tok.encode(r["answer"])) + 1 for r in probe_rows)
    cpu["fixed_probe"] = probe
    report["cpu_checkpoint_evaluation"] = cpu

    encoder = VisionEncoder(**vision_payload["config"])
    encoder.load_state_dict(vision_payload["encoder"])
    classifier = nn.Linear(16, 6)
    classifier.load_state_dict(vision_payload["classifier"])
    images = torch.stack([scene(r["color"], r["shape"], offset=r["offset"]) for r in parts["test"]])
    logits = classifier(encoder(images).mean(1))
    predictions = logits.argmax(-1).tolist()
    assert predictions == list(range(6))
    report["encoder_cpu"] = {"predicted_classes": predictions, "correct": 6, "examples": 6,
                             "features_shape": list(encoder(images).shape)}

# Original text records: independently verify every generated byte and all relevant denominators.
sft_result = records["sft"]["results"]
sft_eval = sft_result["after"]["test"]
sft_calc = record_generations(sft_eval, "expected", "exact")
assert sft_calc == {"examples": 10, "correct": 5, "eos": 10, "target_tokens": 69}
assert sft_eval["nll_sum"] / 69 == sft_eval["nll"]
assert sft_result["checkpoint"] == "model.pt" and sft_result["training"]["steps"] == 900
assert sft_result["data"]["train"]["records"] == 45
report["sft_original_recalculation"] = sft_calc
report["record_training_facts"] = {"projector_initial_loss": projector["training"]["initial_loss"],
                                   "projector_final_loss": projector["training"]["final_loss"],
                                   "weights_changed": projector["training"]["weights_changed"],
                                   "nonzero_gradient_seen": projector["training"]["nonzero_gradient_seen"],
                                   "training_seconds_recorded": projector["training"]["seconds"],
                                   "training_timing_scope": projector["training"]["timing_scope"],
                                   "encoder_original_test_count": records["encoders"]["results"]["vision"]["test"]["count"],
                                   "encoder_original_test_correct": records["encoders"]["results"]["vision"]["test"]["correct"]}
(ART / "cpu-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
