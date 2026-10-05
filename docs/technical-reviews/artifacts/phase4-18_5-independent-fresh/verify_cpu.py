"""Independent bounded CPU checks for current section 18.5; no model weights."""
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.alignment import distillation_kl

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
env = {
    "python": sys.version,
    "torch": torch.__version__,
    "torch_git_version": torch.version.git_version,
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu",
    "platform": platform.platform(),
}
(BASE / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

raw_fence = (BASE / "frozen/fence-1.py").read_bytes()
namespace = {"__name__": "__main__"}
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(raw_fence, "course/chapters/18.md#18.5:fence-1", "exec"), namespace)
(BASE / "original-fence-stdout.txt").write_text(capture.getvalue())
print("ORIGINAL FENCE")
print(capture.getvalue(), end="")
assert namespace["logits"].tolist() == [4.0, 1.0, 0.0]
assert not namespace["logits"].requires_grad

numeric = []
expected_rounds = {
    1: [0.9362, 0.0466, 0.0171],
    2: [0.7361, 0.1643, 0.0996],
    4: [0.5434, 0.2567, 0.1999],
}
for temperature in [1, 2, 4]:
    masses = [math.exp(z / temperature) for z in [4, 1, 0]]
    total = sum(masses)
    probability = [m / total for m in masses]
    tensor = (torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64) / temperature).softmax(0)
    assert max(abs(a - b) for a, b in zip(probability, tensor.tolist())) < 1e-15
    assert [round(p, 4) for p in probability] == expected_rounds[temperature]
    assert abs(sum(probability) - 1) < 1e-15
    assert abs(probability[0] / probability[1] - math.exp(3 / temperature)) < 1e-13
    assert sorted(range(3), key=probability.__getitem__, reverse=True) == [0, 1, 2]
    numeric.append({"T": temperature, "exp_masses": masses, "denominator": total,
                    "probabilities": probability, "ratio_first_second": probability[0] / probability[1],
                    "percentages": [round(100 * p, 2) for p in probability]})

z = torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64)
assert torch.equal((2 * z / 2).softmax(0), z.softmax(0))
assert torch.equal((2 * z / 4).softmax(0), (z / 2).softmax(0))
assert torch.allclose(((z + 100) / 2).softmax(0), (z / 2).softmax(0), atol=1e-15, rtol=0)
assert abs(float((z.softmax(0) / 2).sum()) - 0.5) < 1e-15
high = (z / 1e8).softmax(0)
assert torch.allclose(high, torch.full_like(high, 1 / 3), atol=1e-8, rtol=0)
entropy = [float(-(p * p.log()).sum()) for p in [(z / t).softmax(0) for t in [1, 2, 4]]]
assert entropy[0] < entropy[1] < entropy[2]
print("INDEPENDENT ARITHMETIC", json.dumps(numeric, ensure_ascii=False))
print("VARIANTS doubling logits, shift invariance, probability/T mass=0.5, high-T limit and entropy", entropy)

gradient_rows = []
for temperature in [1.0, 2.0, 4.0]:
    student = torch.tensor([[[1.3, -0.5, 0.8]] * 3], requires_grad=True)
    teacher = torch.tensor([[[4.0, 1.0, 0.0]] * 3], requires_grad=True)
    labels = torch.tensor([[-100, 0, 1]])
    loss = distillation_kl(student, teacher, labels, temperature=temperature)
    p = (teacher.detach() / temperature).softmax(-1)
    q = (student.detach() / temperature).softmax(-1)
    unscaled = (p * (p.log() - q.log())).sum(-1)[labels != -100].mean()
    assert torch.allclose(loss.detach(), unscaled * temperature**2, atol=1e-6, rtol=1e-6)
    loss.backward()
    expected_gradient = temperature * (q - p) / 2
    expected_gradient[:, 0] = 0
    assert torch.allclose(student.grad, expected_gradient, atol=2e-7, rtol=2e-6)
    assert teacher.grad is None
    gradient_rows.append({"T": temperature, "valid_positions": 2, "candidate_axis": -1,
                          "raw_kl": float(unscaled), "scaled_helper_loss": float(loss.detach()),
                          "student_gradient": student.grad.tolist(), "teacher_gradient": None})
print("HELPER T-SQUARED ONCE / MASK / GRADIENT", json.dumps(gradient_rows))

figure = ET.parse(BASE / "frozen/temperature.svg").getroot()
ns = {"s": "http://www.w3.org/2000/svg"}
percent_labels = [n.text for n in figure.findall("s:text", ns) if n.attrib.get("class") == "percent"]
expected_percent_labels = [f"{x:.2f}%" for row in numeric for x in row["percentages"]]
assert percent_labels == expected_percent_labels
figure_rows = []
for group, row in zip(figure.findall("s:g", ns), numeric, strict=True):
    rects = group.findall("s:rect", ns)
    widths = [float(r.attrib["width"]) for r in rects]
    starts = [float(r.attrib["x"]) for r in rects]
    assert [r.attrib["fill"] for r in rects] == ["#4f83c7", "#e8a04b", "#58a285"]
    assert max(abs(w - 520 * p) for w, p in zip(widths, row["probabilities"])) <= 5e-5
    assert abs(sum(widths) - 520) <= 1.1e-4
    assert abs(starts[0] - 40) == 0
    assert max(abs(starts[i+1] - starts[i] - widths[i]) for i in [0,1]) <= 1.1e-4
    figure_rows.append({"T": row["T"], "bar_width_svg_units": sum(widths),
                        "segments_svg_units": widths, "labels": row["percentages"]})
print("FIGURE GEOMETRY", json.dumps(figure_rows))

empirical = []
pointers = {}
for filename in ["distillation", "multimodal_distillation"]:
    src = BASE / "original-results" / (filename + ".json")
    obj = json.loads(src.read_text())
    assert obj["revision"] == "5af615e5d7c9642afee800390fa072257f895d0c"
    original_code_hash = hashlib.sha256((BASE / "code/compression-experiment-revision.py").read_bytes()).hexdigest()
    assert obj["code_sha256"]["scripts/course_experiments/compression.py"] == original_code_hash
    assert obj["code_sha256"]["tiny_perceptron/alignment.py"] == hashlib.sha256((BASE / "code/alignment.py").read_bytes()).hexdigest()
    checked = ["/revision", "/seed", "/torch_version", "/code_sha256/scripts~1course_experiments~1compression.py", "/code_sha256/tiny_perceptron~1alignment.py"]
    for task, value in obj["results"]["tasks"].items():
        for name, run in value["runs"].items():
            if not (name.endswith("_ce_kl") or name == "ce_kl"):
                continue
            prefix = "/results/tasks/" + task + "/runs/" + name
            training = run["training"]
            keys = ["objective", "temperature", "temperature_squared_applied_once", "steps", "optimizer_updates", "training_examples"]
            keys += [k for k in ["alpha", "effective_supervised_tokens", "effective_answer_tokens"] if k in training]
            checked += [prefix + "/training/" + k for k in keys]
            assert training["temperature"] == 2.0
            assert training["temperature_squared_applied_once"] is True
            assert training["steps"] == training["optimizer_updates"] > 0
            if "alpha" in training:
                assert training["alpha"] == 0.5
            else:
                assert training["objective"] == "0.5 CE + 0.5 KL(teacher||student) T²"
            if "generation" in run["test"]:
                checked.append(prefix + "/test/generation")
                assert run["test"]["generation"] == "greedy, full recompute, no KV cache"
            empirical.append({"file": filename, "pointer": prefix, "training": {k: training[k] for k in keys}})
    pointers[filename] = checked
assert len(empirical) == 6
source_tree = ast.parse((BASE / "code/compression-experiment-revision.py").read_text())
for name in ["_fit_text", "_fit_modal"]:
    function = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "distillation_kl"]
    assert len(calls) == 1
    temperature_square = [n for n in ast.walk(function) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow)]
    assert not temperature_square
print("RECORDED CONFIGURATION ONLY", json.dumps(empirical, ensure_ascii=False))
(BASE / "raw-result-pointers.json").write_text(json.dumps(pointers, indent=2) + "\n")
(BASE / "cpu-observations.json").write_text(json.dumps({"numeric": numeric, "gradient": gradient_rows,
    "figure_geometry": figure_rows, "empirical_configuration": empirical}, indent=2, ensure_ascii=False) + "\n")
print("PASS all bounded CPU assertions; no training, model download or model evaluation")
