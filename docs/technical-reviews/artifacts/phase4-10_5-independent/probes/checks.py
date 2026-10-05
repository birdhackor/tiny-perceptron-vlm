"""Independent, bounded CPU checks for current lesson 10.5; no training or checkpoints."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from tiny_perceptron.multimodal import VisionEncoder, patchify, scene

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(name, value):
    (ART / "probes" / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

environment = {
    "python": sys.version,
    "torch": str(torch.__version__),
    "torch_git_version": torch.version.git_version,
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "platform": platform.platform(),
    "threads": str(torch.get_num_threads()),
    "cwd": str(ROOT),
    "probe_source_sha256": sha(Path(__file__).read_bytes()),
    "implementation_sha256": sha((ROOT / "tiny_perceptron/multimodal.py").read_bytes()),
}
save("environment.json", environment)

# Execute the unchanged, self-contained fence as stdin in a new Python process.
# It imports everything it uses and receives no notebook bootstrap namespace.
raw = (ART / "original/fence-1.py").read_bytes()
standalone = subprocess.run([sys.executable, "-"], input=raw, cwd=ROOT,
                           capture_output=True, timeout=30, check=False)
save("standalone.json", {
    "command": ".venv/bin/python - < docs/technical-reviews/artifacts/phase4-10_5-independent/original/fence-1.py",
    "command_argv": [sys.executable, "-"],
    "stdin_file": "original/fence-1.py", "stdin_sha256": sha(raw),
    "cwd": str(ROOT), "exit_code": standalone.returncode,
    "stdout": standalone.stdout.decode(), "stderr": standalone.stderr.decode(),
    "bootstrap_used": False,
})
assert standalone.returncode == 0
assert standalone.stdout.decode() == "特徵 (2, 16, 8) 分數 (2, 2)\n入口收到梯度 True\n"

def forward_backward(labels, dtype=torch.float32, freeze=False):
    torch.manual_seed(0)
    encoder = VisionEncoder(width=8).to(dtype=dtype)
    classifier = nn.Linear(8, 2).to(dtype=dtype)
    if freeze:
        encoder.requires_grad_(False)
    originals = {"encoder." + n: p.detach().clone() for n, p in encoder.named_parameters()}
    originals.update({"classifier." + n: p.detach().clone() for n, p in classifier.named_parameters()})
    images = torch.stack([scene("red", "square"), scene("blue", "circle")]).to(dtype=dtype)
    features = encoder(images)
    if features.requires_grad:
        features.retain_grad()
    pooled = features.mean(1)
    if pooled.requires_grad:
        pooled.retain_grad()
    logits = classifier(pooled)
    logits.retain_grad()
    y = torch.tensor(labels)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    parameters = list(("encoder." + n, p) for n, p in encoder.named_parameters())
    parameters += list(("classifier." + n, p) for n, p in classifier.named_parameters())
    records = [{"name": n, "shape": list(p.shape), "numel": p.numel(),
                "requires_grad": p.requires_grad,
                "grad_is_none": p.grad is None,
                "grad_norm": None if p.grad is None else p.grad.norm().item(),
                "unchanged_after_backward": torch.equal(p.detach(), originals[n])}
               for n, p in parameters]
    assert all(r["unchanged_after_backward"] for r in records)
    return encoder, classifier, images, features, pooled, logits, y, loss, records

normal = forward_backward([0, 1])
flipped = forward_backward([1, 0])
frozen = forward_backward([0, 1], freeze=True)
result = {}
for name, run in [("original_labels", normal), ("flipped_labels", flipped), ("frozen_encoder", frozen)]:
    encoder, classifier, images, features, pooled, logits, y, loss, records = run
    result[name] = {"images_shape": list(images.shape),
                    "patches_shape": list(patchify(images).shape),
                    "features_shape": list(features.shape), "pooled_shape": list(pooled.shape),
                    "logits_shape": list(logits.shape), "labels": y.tolist(),
                    "labels_dtype": str(y.dtype), "logits": logits.detach().tolist(),
                    "loss": loss.item(), "parameters": records,
                    "image_requires_grad": images.requires_grad,
                    "loss_denominator_examples": 2, "pool_denominator_positions": 16}
assert normal[0].projection.weight.grad.norm().item() > 0
assert flipped[0].projection.weight.grad.norm().item() > 0
assert all(p.grad is None for p in frozen[0].parameters())
assert frozen[1].weight.grad.norm().item() > 0
assert all(r["requires_grad"] for r in result["original_labels"]["parameters"])
assert sum(r["numel"] for r in result["original_labels"]["parameters"]) == 626

# The finite difference changes one synthetic parameter temporarily, restores it,
# and performs no optimizer step. Double precision bounds roundoff error.
encoder, classifier, images, features, pooled, logits, y, loss, _ = forward_backward([0, 1], torch.float64)
manual_loss = -logits.log_softmax(-1)[torch.arange(2), y].mean()
logit_grad = (logits.detach().softmax(-1) - F.one_hot(y, num_classes=2)) / 2
loss_error = abs(manual_loss.item() - loss.item())
logit_grad_error = (logit_grad - logits.grad).abs().max().item()
pool_error = (features.grad - pooled.grad[:, None, :] / 16).abs().max().item()
assert loss_error < 1e-12 and logit_grad_error < 1e-12 and pool_error < 1e-12
parameter = encoder.projection.weight
flat_index = parameter.grad.abs().argmax().item()
index = (flat_index // parameter.shape[1], flat_index % parameter.shape[1])
analytic = parameter.grad[index].item()
base = parameter[index].item()
epsilon = 1e-6
with torch.no_grad():
    parameter[index] = base + epsilon
    plus = F.cross_entropy(classifier(encoder(images).mean(1)), y).item()
    parameter[index] = base - epsilon
    minus = F.cross_entropy(classifier(encoder(images).mean(1)), y).item()
    parameter[index] = base
finite_difference = (plus - minus) / (2 * epsilon)
fd_error = abs(finite_difference - analytic)
assert fd_error < 1e-8
result["derivatives"] = {"precision": "float64", "loss_formula": "mean_i(-log_softmax(logits_i)[label_i]), N=2",
                         "manual_loss": manual_loss.item(), "loss_formula_abs_error": loss_error,
                         "logits_gradient_formula": "(softmax(logits)-one_hot(labels))/2",
                         "logits_gradient_max_abs_error": logit_grad_error,
                         "mean_gradient_formula": "features.grad=pooled.grad[:,None,:]/16",
                         "mean_gradient_max_abs_error": pool_error,
                         "finite_difference_index": list(index), "epsilon": epsilon,
                         "analytic_projection_gradient": analytic,
                         "central_finite_difference": finite_difference, "abs_error": fd_error,
                         "tolerance": 1e-8, "parameter_restored": parameter[index].item() == base}

# A pure averaging counterexample concerns a sequence of feature rows. Position
# coding upstream may preserve spatial information; this is not an impossibility
# claim about the actual position-aware VisionEncoder.
sequence = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64)
reversed_sequence = sequence.flip(1)
assert not torch.equal(sequence, reversed_sequence)
assert torch.equal(sequence.mean(1), reversed_sequence.mean(1))
result["pooling_counterexample"] = {"left_right": sequence.tolist(),
                                    "right_left": reversed_sequence.tolist(),
                                    "both_means": sequence.mean(1).tolist(),
                                    "scope": "post-encoding row order alone is removed by mean; upstream positional features may retain spatial cues"}

# A hand-coded color rule fits both displayed examples without understanding
# shape and fails on the recombined examples. This is not trained-model inference.
combinations = [("red", "square", 0), ("blue", "circle", 1),
                ("red", "circle", 1), ("blue", "square", 0)]
color_rule = []
for color, shape, target in combinations:
    image = scene(color, shape)
    mean_rgb = image.mean((1, 2))
    prediction = 0 if mean_rgb[0] > mean_rgb[2] else 1
    color_rule.append({"color": color, "shape": shape, "target_shape": target,
                       "mean_rgb": mean_rgb.tolist(), "color_only_prediction": prediction,
                       "correct": prediction == target})
assert [r["correct"] for r in color_rule] == [True, True, False, False]
result["color_shortcut_counterexample"] = color_rule

# Match the actual SVG raster-cell RGB values to every generated source pixel.
svg = ET.parse(ROOT / "course/figures/rewrite-10-shape-gradient.svg").getroot()
rectangles = list(svg.iter("{http://www.w3.org/2000/svg}rect"))
figure = {"svg_sha256": sha((ROOT / "course/figures/rewrite-10-shape-gradient.svg").read_bytes()),
          "cells": [], "texts": [node.text for node in svg.iter("{http://www.w3.org/2000/svg}text")],
          "paths": [dict(node.attrib) for node in svg.iter("{http://www.w3.org/2000/svg}path")]}
for x_base, color, shape in [(55, "red", "square"), (373, "blue", "circle")]:
    image = scene(color, shape)
    matched = 0
    filled = 0
    for row in range(16):
        for col in range(16):
            candidates = [r for r in rectangles if r.get("x") == str(x_base + 13 * col)
                          and r.get("y") == str(87 + 13 * row)
                          and r.get("width") == "13" and r.get("height") == "13"]
            assert len(candidates) == 1
            expected = "#" + "".join(f"{round(v * 255):02x}" for v in image[:, row, col].tolist())
            assert candidates[0].get("fill") == expected
            matched += 1
            filled += int(expected != "#000000")
    figure["cells"].append({"image": color + " " + shape, "matched_cells": matched, "colored_cells": filled})
assert [r["colored_cells"] for r in figure["cells"]] == [81, 49]
save("figure-check.json", figure)
save("cpu-checks.json", result)

# Read only named raw result/provenance/config/sample pointers. Do not recurse
# into any author summary, notes, review, scope-correction, calibration, or audio.
raw_result = (ROOT / "docs/course-experiments/results/encoders.json").read_bytes()
history_root = json.loads(raw_result)
vision = history_root["results"]["vision"]
pointers = ["/" + k for k in ["schema_version", "experiment_id", "revision", "device", "seed", "torch_version", "python_version", "step_scale"]]
selected = {p: history_root[p[1:]] for p in pointers}
for k in ["tiny_perceptron/multimodal.py", "scripts/course_experiments/modalities.py"]:
    p = "/code_sha256/" + k.replace("~", "~0").replace("/", "~1")
    pointers.append(p)
    selected[p] = history_root["code_sha256"][k]
    assert sha((ROOT / k).read_bytes()) == selected[p]
for k in ["classes", "config"]:
    p = "/results/vision/" + k
    pointers.append(p)
    selected[p] = vision[k]
for k in ["parameters", "trainable_parameters", "initial_loss", "final_loss", "loss_probe", "history", "steps", "effective_tokens", "effective_targets", "weights_changed", "nonzero_gradient_seen", "cpu_smoke", "checkpoint"]:
    p = "/results/vision/training/" + k
    pointers.append(p)
    selected[p] = vision["training"][k]
for split in ["before", "validation", "test"]:
    for k in ["count", "correct", "accuracy", "samples"]:
        p = "/results/vision/" + split + "/" + k
        pointers.append(p)
        selected[p] = vision[split][k]
    values = vision[split]
    assert len(values["samples"]) == values["count"]
    correct = sum(row["target"] == row["predicted"] for row in values["samples"])
    assert all(row["correct"] == (row["target"] == row["predicted"]) for row in values["samples"])
    assert correct == values["correct"] and correct / values["count"] == values["accuracy"]
identities = {}
for split in ["train", "validation", "test"]:
    values = vision["data"]["splits"][split]
    for k in ["count", "sha256", "records"]:
        p = "/results/vision/data/splits/" + split + "/" + k
        pointers.append(p)
        selected[p] = values[k]
    assert len(values["records"]) == values["count"]
    assert sha(json.dumps(values["records"], ensure_ascii=False, sort_keys=True).encode()) == values["sha256"]
    identities[split] = {r["family"] for r in values["records"]}
    assert all(r["answer"] == r["color"] + " " + r["shape"] for r in values["records"])
    assert {r["question"] for r in values["records"]} == {"describe"}
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not identities[a] & identities[b]
training = vision["training"]
assert len(training["history"]) == training["steps"] == 250
assert sum(r["effective_targets"] for r in training["history"]) == training["effective_targets"] == 2000
assert [r["step"] for r in training["history"]] == list(range(1, 251))
assert all(r["effective_targets"] == 8 and math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in training["history"])
assert vision["config"] == {"width": 16, "image_size": 16, "patch_size": 4}
assert len(vision["classes"]) == 6
save("history-selected-pointers.json", {"source": "docs/course-experiments/results/encoders.json",
                                       "source_sha256": sha(raw_result), "inspection_pointers": pointers,
                                       "selected_original_values": selected,
                                       "exclusions": "no scope, summary-status, audio, calibration, notes, review or scope_correction values inspected"})
save("history-check.json", {"source_sha256": sha(raw_result), "revision": history_root["revision"],
                            "config": vision["config"], "classes": vision["classes"],
                            "split_counts": {s: vision["data"]["splits"][s]["count"] for s in identities},
                            "split_family_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
                            "train_offsets": sorted({r["offset"] for r in vision["data"]["splits"]["train"]["records"]}),
                            "validation_offsets": [1], "test_offsets": [2],
                            "sample_accuracy_checks": {s: {k: vision[s][k] for k in ["count", "correct", "accuracy"]} for s in ["before", "validation", "test"]},
                            "steps": training["steps"], "targets_per_step": 8,
                            "total_effective_targets": training["effective_targets"],
                            "reported_parameters": training["parameters"],
                            "reported_trainable_parameters": training["trainable_parameters"],
                            "initial_loss": training["initial_loss"], "final_loss": training["final_loss"],
                            "weights_changed": training["weights_changed"],
                            "execution_scope": "parsed original recorded measurements; no historical model loaded, trained, or inferred"})
print(json.dumps({"standalone_exit": standalone.returncode,
                  "original_projection_gradient_norm": normal[0].projection.weight.grad.norm().item(),
                  "flipped_projection_gradient_norm": flipped[0].projection.weight.grad.norm().item(),
                  "all_original_parameters_unchanged": True,
                  "original_parameters": 626,
                  "frozen_encoder_gradient_absent": True,
                  "finite_difference_abs_error": fd_error,
                  "loss_formula_abs_error": loss_error,
                  "logit_gradient_max_abs_error": logit_grad_error,
                  "pool_gradient_max_abs_error": pool_error,
                  "svg_matched_cells": 512,
                  "historic_classes": 6, "historic_config_width": 16,
                  "historic_steps": 250, "historic_targets": 2000,
                  "historic_test_correct": 6, "historic_test_count": 6,
                  "status": "all bounded checks passed; report verdict must independently assess scope/figure"},
                 ensure_ascii=False, indent=2))
