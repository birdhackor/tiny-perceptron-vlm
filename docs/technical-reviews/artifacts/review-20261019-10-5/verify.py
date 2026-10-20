"""核對圖片分類範例、資料紀錄與所用 PyTorch 的原始契約。"""

import ast
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.multimodal import VisionEncoder, patchify, scene
from scripts.course_experiments.run import experiment_spec


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.set_default_device("cpu")
report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-10-5/verify.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": "float32; finite-difference probe uses float64",
    },
}
source = ROOT / "docs/course-review-20261019/freeze-01/sources/10.5.md"
fences = source.read_text().split("```python\n")[1:]
snippet = fences[1].split("```")[0]
tree = ast.parse((ROOT / "tiny_perceptron/multimodal.py").read_text())
actual = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VisionEncoder")
report["snippet_matches_implementation"] = ast.dump(ast.parse(snippet).body[0]) == ast.dump(actual)
report["source_sha256"] = sha(source)
default_encoder = VisionEncoder()
default_images = torch.stack([scene("red", "square"), scene("blue", "circle")])
report["default_encoder"] = {
    "patch_size": default_encoder.patch_size, "image_size": default_encoder.image_size,
    "position_registered": "position" in dict(default_encoder.named_parameters()),
    "position_requires_grad": default_encoder.position.requires_grad,
    "projection_in_features": default_encoder.projection.in_features,
    "projection_out_features": default_encoder.projection.out_features,
    "feature_layers": [type(layer).__name__ for layer in default_encoder.feature],
    "output_shape": list(default_encoder(default_images).shape),
    "module_call_equals_forward": torch.equal(default_encoder(default_images), default_encoder.forward(default_images)),
}
report["gradient_cases"] = []
for labels in ([0, 1], [1, 0]):
    torch.manual_seed(0)
    encoder = VisionEncoder(width=8)
    classifier = nn.Linear(8, 2)
    before = {k: v.detach().clone() for k, v in encoder.named_parameters()}
    classifier_before = {k: v.detach().clone() for k, v in classifier.named_parameters()}
    images = torch.stack([scene("red", "square"), scene("blue", "circle")])
    patches = patchify(images, 4)
    projected = encoder.projection(patches)
    positioned = projected + encoder.position
    features = encoder.feature(positioned)
    pooled = features.mean(1)
    logits = classifier(pooled)
    for tensor in (projected, positioned, features, pooled, logits):
        tensor.retain_grad()
    loss = F.cross_entropy(logits, torch.tensor(labels))
    loss.backward()
    manual_loss = -logits.log_softmax(-1)[torch.arange(2), torch.tensor(labels)].mean()
    expected_logit_grad = (logits.detach().softmax(-1) - F.one_hot(torch.tensor(labels), 2)) / 2
    report["gradient_cases"].append({
        "labels": labels,
        "shapes": {k: list(v.shape) for k, v in {
            "images": images, "patches": patches, "features": features,
            "pooled": pooled, "logits": logits, "position": encoder.position,
            "projection_weight": encoder.projection.weight,
        }.items()},
        "loss": loss.item(), "manual_loss": manual_loss.item(),
        "logits": logits.detach().tolist(),
        "all_gradients_finite": all(bool(torch.isfinite(p.grad).all()) for p in encoder.parameters()),
        "parameter_gradient_norms": {k: p.grad.norm().item() for k, p in encoder.named_parameters()},
        "intermediate_gradient_norms": {k: v.grad.norm().item() for k, v in {
            "projected": projected, "positioned": positioned, "features": features,
            "pooled": pooled, "logits": logits,
        }.items()},
        "expected_logit_gradient_max_error": (logits.grad - expected_logit_grad).abs().max().item(),
        "pooling_gradient_max_error": (features.grad - pooled.grad[:, None, :] / 16).abs().max().item(),
        "weights_unchanged_after_backward": all(torch.equal(before[k], p) for k, p in encoder.named_parameters()),
        "classifier_weights_unchanged_after_backward": all(torch.equal(classifier_before[k], p) for k, p in classifier.named_parameters()),
        "position_rms": encoder.position.detach().square().mean().sqrt().item(),
        "projection_output_rms": projected.detach().square().mean().sqrt().item(),
    })

torch.manual_seed(0)
encoder = VisionEncoder(width=8).double()
classifier = nn.Linear(8, 2).double()
images = torch.stack([scene("red", "square"), scene("blue", "circle")]).double()
labels = torch.tensor([0, 1])
loss = F.cross_entropy(classifier(encoder(images).mean(1)), labels)
loss.backward()
gradient = encoder.projection.weight.grad
flat_index = int(gradient.abs().argmax())
index = (flat_index // gradient.shape[1], flat_index % gradient.shape[1])
automatic = gradient[index].item()
epsilon = 1e-6
with torch.no_grad():
    original = encoder.projection.weight[index].item()
    encoder.projection.weight[index] = original + epsilon
    plus = F.cross_entropy(classifier(encoder(images).mean(1)), labels).item()
    encoder.projection.weight[index] = original - epsilon
    minus = F.cross_entropy(classifier(encoder(images).mean(1)), labels).item()
    encoder.projection.weight[index] = original
finite_difference = (plus - minus) / (2 * epsilon)
report["finite_difference"] = {"index": index, "epsilon": epsilon, "automatic": automatic,
                               "central_difference": finite_difference, "absolute_error": abs(automatic - finite_difference)}
invalid = {}
for shape in ((2, 1, 16, 16), (2, 3, 15, 16)):
    try:
        encoder(torch.zeros(shape, dtype=torch.float64))
    except ValueError as error:
        invalid[str(shape)] = str(error)
report["invalid_image_shapes"] = invalid
features = torch.arange(2 * 16 * 8, dtype=torch.float64).reshape(2, 16, 8)
permutation = torch.arange(15, -1, -1)
report["pooling_permutation_max_error"] = (features.mean(1) - features[:, permutation].mean(1)).abs().max().item()
images = torch.stack([scene("red", "square"), scene("blue", "circle")])
report["color_shortcut"] = {
    "rgb_means": images.mean((2, 3)).tolist(),
    "rule": "紅色通道平均大於藍色通道平均時回答方塊，否則回答圓形",
    "predicted_labels": (images[:, 2].mean((1, 2)) > images[:, 0].mean((1, 2))).long().tolist(),
}

svg = ET.parse(ROOT / "course/figures/rewrite-10-shape-gradient.svg").getroot()
ns = {"s": "http://www.w3.org/2000/svg"}
pixels = {}
for name, left in (("red square", 55), ("blue circle", 373)):
    grid = [[None] * 16 for _ in range(16)]
    for rect in svg.findall("s:rect", ns):
        if rect.get("width") == "13" and rect.get("height") == "13":
            x, y = int(rect.get("x")), int(rect.get("y"))
            if left <= x < left + 208:
                grid[(y - 87) // 13][(x - left) // 13] = rect.get("fill")
    color, shape = name.split()
    image = scene(color, shape)
    wanted = "#ff0000" if color == "red" else "#0000ff"
    mismatches = sum(grid[y][x] != (wanted if bool(image[:, y, x].any()) else "#000000") for y in range(16) for x in range(16))
    pixels[name] = {"pixels": 256, "mismatches_with_scene": mismatches, "foreground_pixels": int(image.any(0).sum())}
report["figure_pixel_comparison"] = pixels
png = ROOT / "outputs/course-review-20261019/figure-renders-04/8a2ecec6c4af76bef48e803c373ce2bd9d056def991f02f088999b2d5c08847f.png"
report["viewed_png_sha256"] = sha(png)

result_path = ROOT / "docs/course-experiments/results/encoders.json"
record = json.loads(result_path.read_text())
vision = record["results"]["vision"]
empirical = {
    "record_path": result_path.relative_to(ROOT).as_posix(), "record_sha256": sha(result_path),
    "record_environment": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status")},
    "config": vision["config"], "classes": vision["classes"],
    "training": {k: vision["training"][k] for k in ("steps", "effective_targets", "weights_changed", "nonzero_gradient_seen")},
    "history_count": len(vision["training"]["history"]),
    "history_steps_contiguous": [row["step"] for row in vision["training"]["history"]] == list(range(1, 251)),
    "history_effective_targets_sum": sum(row["effective_targets"] for row in vision["training"]["history"]),
    "splits": {}, "evaluation": {},
}
for split, data in vision["data"]["splits"].items():
    records = data["records"]
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    empirical["splits"][split] = {"count": len(records), "recorded_count": data["count"],
        "offsets": sorted({row["offset"] for row in records}),
        "classes": sorted({row["answer"] for row in records}),
        "record_hash_matches": hashlib.sha256(canonical).hexdigest() == data["sha256"]}
for split in ("validation", "test"):
    scores = vision[split]
    calibration = vision["calibration"][split]
    logits = torch.tensor(calibration["logits"])
    selected = logits.argmax(-1).tolist()
    truth = calibration["labels"]
    empirical["evaluation"][split] = {
        "recorded": scores, "recomputed_from_logits": {"count": len(truth), "correct": sum(a == b for a, b in zip(selected, truth, strict=True))},
        "row_flags_match": all(row["correct"] == (row["target"] == row["predicted"]) for row in scores["samples"]),
        "predicted_names_match_logits": [vision["classes"][i] for i in selected] == [row["predicted"] for row in scores["samples"]],
    }
empirical["code_hashes"] = {path: {"recorded": record["code_sha256"][path], "current": sha(ROOT / path)} for path in ("tiny_perceptron/multimodal.py", "scripts/course_experiments/modalities.py")}
empirical["experiment_entry"] = experiment_spec("encoders")
report["experiment_record_audit"] = empirical

commit = str(torch.version.git_version)
base_url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/"
requests = {
    "torch/_tensor.py": [("def backward(", 23), ("def norm(", 22)],
    "torch/nn/modules/linear.py": [("class Linear(Module):", 32)],
    "torch/nn/modules/loss.py": [("class CrossEntropyLoss(_WeightedLoss):", 78)],
    "torch/nn/parameter.py": [("class Parameter(torch.Tensor", 28)],
    "torch/nn/modules/module.py": [("def __init__(self, *args: Any, **kwargs: Any)", 35), ("def _call_impl(", 22), ("def __setattr__(", 44)],
    "torch/nn/modules/container.py": [("class Sequential(Module):", 25), ("def forward(self, input):", 6)],
    "torch/nn/modules/normalization.py": [("class LayerNorm(Module):", 30)],
    "torch/nn/modules/activation.py": [("class GELU(Module):", 24)],
    "torch/_torch_docs.py": [('    torch.mean,', 46), ('    torch.stack,', 25), ('    torch.randn,', 24)],
}
official = []
for path, selectors in requests.items():
    url = base_url + path
    command = ["curl", "--location", "--fail", "--silent", "--show-error", url]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    lines = process.stdout.splitlines()
    excerpts = []
    for marker, count in selectors:
        start = next(i for i, line in enumerate(lines) if marker in line)
        excerpts.append({"symbol_selector": marker, "start_line": start + 1,
                         "end_line": min(len(lines), start + count), "text": "\n".join(lines[start:start + count])})
    official.append({"path": path, "url": url, "version": commit, "accessed_on": datetime.now(UTC).date().isoformat(),
        "command_argv": command, "exit_code": process.returncode,
        "upstream_full_file_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(), "excerpts": excerpts})
report["official_source_excerpts"] = official
output = Path(__file__).with_name("verification.json")
output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "sha256": sha(output),
    "gradient_norms": [case["parameter_gradient_norms"]["projection.weight"] for case in report["gradient_cases"]],
    "finite_difference_absolute_error": report["finite_difference"]["absolute_error"],
    "snippet_matches_implementation": report["snippet_matches_implementation"],
    "figure_pixel_comparison": pixels,
    "official_sources_read": len(official)}, ensure_ascii=False))
