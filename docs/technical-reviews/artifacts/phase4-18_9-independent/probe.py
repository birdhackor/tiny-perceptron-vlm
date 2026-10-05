"""Bounded CPU verification of 18.9; no model loading, training, or score evaluation."""
import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.alignment import distillation_kl, distillation_loss
from tiny_perceptron.model import masked_loss

torch.set_num_threads(1)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

environment = {"python": sys.version, "torch": torch.__version__, "torch_git": torch.version.git_version,
               "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads())}
results = {"environment": environment, "method": "original fence plus fixed tiny tensors; raw JSON arithmetic only", "checks": []}
for name in ("tiny_perceptron/alignment.py", "tiny_perceptron/model.py"):
    assert sha(ROOT / name) == sha(OUT / "inputs" / name)

# Original fence has no backward or optimizer call and leaves its input untouched.
namespace = {}
exec(compile((OUT / "original-fence/fence-1.py").read_bytes(), "18.9-original-fence", "exec"), namespace)
assert torch.equal(namespace["student"], torch.zeros(1, 2, 2))
assert namespace["student"].grad is None
results["original_no_update"] = True

# Derive natural-log CE/KL independently with Python double arithmetic.
ce_expected = -math.log(0.5)
kl_expected = sum(p * math.log(p / 0.5) for p in (0.8, 0.2))
expected_gradients = {0.0: [[-0.25, 0.25], [0.25, -0.25]],
                      0.25: [[-0.225, 0.225], [0.15, -0.15]],
                      0.5: [[-0.2, 0.2], [0.05, -0.05]],
                      1.0: [[-0.15, 0.15], [-0.15, 0.15]]}
for alpha in (0.0, 0.25, 0.5, 1.0):
    student = torch.zeros(1, 2, 2, requires_grad=True)
    teacher = torch.tensor([[[0.8, 0.2], [0.8, 0.2]]]).log().requires_grad_()
    labels = torch.tensor([[0, 1]])
    ce, kl = masked_loss(student, labels), distillation_kl(student, teacher, labels, temperature=1)
    loss = distillation_loss(student, teacher, labels, alpha=alpha, temperature=1)
    expected = (1-alpha) * ce_expected + alpha * kl_expected
    assert abs(loss.item() - expected) < 1e-7
    loss.backward()
    assert torch.allclose(student.grad, torch.tensor([expected_gradients[alpha]]), atol=1e-7, rtol=0)
    assert teacher.grad is None
    assert torch.equal(student, torch.zeros(1, 2, 2))
    results["checks"].append({"alpha": alpha, "ce": ce.item(), "kl": kl.item(), "loss": loss.item(),
                               "manual_loss": expected, "student_gradient": student.grad.tolist(), "teacher_gradient": None})

# T=2 changes both KL distributions, keeps the hard-label CE at T=1, applies T² once.
student = torch.zeros(1, 2, 2, requires_grad=True)
teacher = torch.tensor([[[0.8, 0.2], [0.8, 0.2]]]).log()
labels = torch.tensor([[0, 1]])
p_t = [math.sqrt(p) / (math.sqrt(0.8) + math.sqrt(0.2)) for p in (0.8, 0.2)]
raw_kl = sum(p * math.log(p / 0.5) for p in p_t)
k = distillation_kl(student, teacher, labels, temperature=2)
loss = distillation_loss(student, teacher, labels, alpha=0.5, temperature=2)
assert abs(k.item() - 4 * raw_kl) < 2e-7
assert abs(loss.item() - (ce_expected + 4*raw_kl)/2) < 2e-7
assert abs(loss.item() - (ce_expected + 16*raw_kl)/2) > 0.3
results["temperature_two"] = {"teacher_probability": p_t, "raw_kl": raw_kl, "scaled_kl": k.item(), "mixed_loss": loss.item()}

# Distinct position probabilities reveal the effective-position denominator and mask.
student = torch.tensor([[[1.0, -0.4], [0.3, 0.7], [-0.5, 1.5]]], requires_grad=True)
teacher = torch.tensor([[[0.7, 0.3], [0.8, 0.2], [0.2, 0.8]]]).log().requires_grad_()
labels = torch.tensor([[-100, 0, 1]])
loss = distillation_loss(student, teacher, labels, alpha=0.25, temperature=1)
q = student.softmax(-1)
manual_ce = -(q[0,1,0].log() + q[0,2,1].log()) / 2
manual_kl = (teacher.detach().exp() * (teacher.detach() - q.log())).sum(-1)[0,1:].mean()
assert torch.allclose(loss, 0.75*manual_ce + 0.25*manual_kl, atol=1e-7, rtol=0)
loss.backward()
assert torch.equal(student.grad[0,0], torch.zeros(2))
assert teacher.grad is None
results["mask"] = {"shape": list(student.shape), "labels": labels.tolist(), "effective_positions": 2,
                   "ce": manual_ce.item(), "kl": manual_kl.item(), "mixed_loss": loss.item(), "gradient": student.grad.tolist()}

# Raw recorded training traces: exact named measurement/provenance fields only.
raw = OUT / "inputs/docs/course-experiments/results/distillation.json"
report = json.loads(raw.read_text())
allowed = ("steps", "optimizer_updates", "batch_size", "initialization_sha256", "batch_plan_sha256",
           "training_examples", "training_sequence_chunks", "effective_supervised_tokens", "objective",
           "alpha", "temperature", "temperature_squared_applied_once", "loss_trace")
results["raw_json_sha256"] = sha(raw)
results["raw_json_top_level_types"] = {k:type(v).__name__ for k,v in report.items()}
results["raw_json_pointers"] = ["/revision", "/device", "/seed", "/torch_version", "/python_version",
                                "/step_scale", "/code_sha256"]
results["recorded_provenance"] = {k:report[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "step_scale")}
historical = OUT / "sources/compression-recorded-5af615e.py"
assert sha(historical) == report["code_sha256"]["scripts/course_experiments/compression.py"]
results["recorded_code_sha_matches"] = True
results["recorded_runs"] = []
for task, case in report["results"]["tasks"].items():
    for name, run in case["runs"].items():
        if not name.endswith("ce_kl"):
            continue
        training = run["training"]
        selected = {k:training[k] for k in allowed}
        assert selected["alpha"] == 0.5 and selected["temperature"] == 2.0
        assert selected["temperature_squared_applied_once"] is True
        errors = [abs(v["loss"] - ((1-training["alpha"])*v["ce"] + training["alpha"]*v["kl"])) for v in selected["loss_trace"]]
        # Trace tensors were FP32 before conversion to Python floats: adding and
        # multiplying rounds once in FP32, unlike the Python-double reconstruction.
        assert all(error < 2e-7 * max(1.0, abs(v["loss"])) for error,v in zip(errors, selected["loss_trace"]))
        baseline = case["runs"][name[:-5]+"ce"]["training"]
        for key in ("steps", "batch_plan_sha256", "initialization_sha256", "effective_supervised_tokens"):
            assert training[key] == baseline[key]
        pointer = f"/results/tasks/{task}/runs/{name}/training"
        results["raw_json_pointers"].extend(pointer+"/"+key for key in allowed)
        selected["task"], selected["run"], selected["max_loss_reconstruction_error"] = task, name, max(errors)
        results["recorded_runs"].append(selected)
(OUT / "probe-results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n")
print("Manual CE/KL:", ce_expected, kl_expected)
print("Alpha 0.25:", results["checks"][1]["loss"])
print("Gradients:", [(v["alpha"],v["student_gradient"]) for v in results["checks"]])
print("T=2:", results["temperature_two"])
print("Mask effective positions:", results["mask"]["effective_positions"])
print("Raw runs/trace samples:", len(results["recorded_runs"]), sum(len(r["loss_trace"]) for r in results["recorded_runs"]))
print("PASS all bounded CPU assertions; no optimization or original-model rescoring")
