import ast
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
torch.set_default_device("cpu")
bundle = json.loads((DESTINATION / "section-execution.json").read_text())
code = bundle["files"]["fence-1.py"]["text"]


def run(text):
    namespace = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(text, "教材程式", "exec"), namespace)
    return namespace, stdout.getvalue()


original, stdout = run(code)
teacher, student = original["teacher"], original["student"]
torch.manual_seed(0)
reference_teacher = nn.Linear(2, 3)
teacher_deltas = {name: float((value - reference_teacher.state_dict()[name]).abs().max())
                  for name, value in teacher.state_dict().items()}
residuals = (original["output"] - original["target"]).detach()
mse_manual = sum(float(value) ** 2 for value in residuals.flatten()) / residuals.numel()
mse_tensor = float(residuals.square().mean())
expected_student = original["student_before"] - 0.1 * student.weight.grad
update_error = float((student.weight.detach() - expected_student).abs().max())
assert all(value == 0 for value in teacher_deltas.values())
assert all(value.grad is None and not value.requires_grad for value in teacher.parameters())
assert update_error < 1e-7
assert abs(mse_manual - mse_tensor) < 1e-7
optimizer_ids = {id(p) for group in original["optimizer"].param_groups for p in group["params"]}
assert optimizer_ids == {id(p) for p in student.parameters()}
assert not optimizer_ids & {id(p) for p in teacher.parameters()}

probe = original["x"].clone().requires_grad_()
teacher(probe).sum().backward()
expected_input_gradient = teacher.weight.sum(dim=0, keepdim=True)
gradient_error = float((probe.grad - expected_input_gradient).abs().max())
assert gradient_error < 1e-7
saved_tensors = {}
for label, input_tensor, enabled in (("固定輸入", original["x"], True),
                                     ("可微輸入", probe, True),
                                     ("可微輸入停用記圖", probe, False)):
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(list(t.shape)) or t), lambda t: t):
        with torch.set_grad_enabled(enabled):
            output = teacher(input_tensor)
    saved_tensors[label] = {"shapes": saved, "requires_grad": output.requires_grad}
assert not saved_tensors["固定輸入"]["shapes"]
assert saved_tensors["可微輸入"]["shapes"]
assert not saved_tensors["可微輸入停用記圖"]["shapes"]

lines = code.splitlines(keepends=True)
assert lines[-1].startswith('print("輸入可微時教師輸出記圖"')
exercise_code = "".join(lines[:-1]) + "with torch.no_grad():\n    " + lines[-1]
exercise, exercise_stdout = run(exercise_code)
assert exercise_stdout.splitlines()[-1].endswith("False")
assert torch.equal(exercise["teacher"].weight, original["teacher"].weight)
assert torch.equal(exercise["student"].weight, original["student"].weight)

alias = student.weight
detached_alias = student.weight.detach()
clone = student.weight.detach().clone()
original["optimizer"].step()
snapshot = {"reference_is_same_object": alias is student.weight,
            "detached_shares_storage": detached_alias.data_ptr() == student.weight.data_ptr(),
            "clone_has_distinct_storage": clone.data_ptr() != student.weight.data_ptr(),
            "reference_comparison_delta": float((student.weight - alias).detach().abs().max()),
            "clone_comparison_delta": float((student.weight.detach() - clone).abs().max())}
assert snapshot["reference_comparison_delta"] == 0
assert snapshot["clone_comparison_delta"] > 0

dropout = nn.Dropout(0.5)
inputs = torch.ones(1000, requires_grad=True)
dropout.train()
train_output = dropout(inputs)
dropout.eval()
eval_output = dropout(inputs)
assert bool((train_output == 0).any())
assert torch.equal(eval_output, inputs) and eval_output.requires_grad

inspection = json.loads((DESTINATION / "source-inspection.json").read_text())
recorded = inspection["recorded_implementation"]
raw = subprocess.check_output(["git", "show", recorded["available_revision"] + ":" + recorded["path"]], cwd=ROOT)
assert hashlib.sha256(raw).hexdigest() == recorded["sha256"]
tree = ast.parse(raw.decode())
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
guard_checks = {}
for name in ("_cache_text", "_hard_targets"):
    node = functions[name]
    source = ast.get_source_segment(raw.decode(), node)
    guard_checks[name] = {
        "no_grad_decorator": any(ast.unparse(x) == "torch.no_grad()" for x in node.decorator_list),
        "eval_and_freeze": "teacher.eval().requires_grad_(False)" in source,
    }
    assert all(guard_checks[name].values())
case_source = ast.get_source_segment(raw.decode(), functions["_distill_case"])
assert "if _parameter_hash(teacher) != frozen_hash:" in case_source
assert case_source.index("if _parameter_hash(teacher) != frozen_hash:") < case_source.index('"teacher_frozen_and_unchanged": True')
parameter_hash_source = ast.get_source_segment(raw.decode(), functions["_parameter_hash"])
assert "model.state_dict().items()" in parameter_hash_source
assert "value.detach().cpu().contiguous().numpy().tobytes()" in parameter_hash_source

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu",
                    "platform": platform.platform()},
    "original": {"stdout": stdout, "teacher_parameter_max_deltas": teacher_deltas,
                 "output_shape": list(original["output"].shape), "dtype": str(residuals.dtype),
                 "teacher_target": original["target"].tolist(), "student_output_before_step": original["output"].detach().tolist(),
                 "residuals": residuals.tolist(), "mse_denominator": residuals.numel(),
                 "mse_manual": mse_manual, "mse_tensor": mse_tensor,
                 "sgd_weight_update_max_error": update_error,
                 "student_weight_max_delta": float((clone - original["student_before"]).abs().max()),
                 "optimizer_only_student": True, "teacher_all_parameters_frozen_and_grad_none": True},
    "input_gradient": {"observed": probe.grad.tolist(), "expected_weight_sum": expected_input_gradient.tolist(),
                       "max_error": gradient_error, "saved_tensors": saved_tensors},
    "exercise": {"modified_program": exercise_code, "stdout": exercise_stdout,
                 "same_teacher_and_student_updates": True},
    "snapshot": snapshot,
    "eval_dropout": {"training_has_zero": True, "evaluation_identity": True, "evaluation_requires_grad": True},
    "recorded_implementation": {"sha256": recorded["sha256"], "guards": guard_checks,
                                "all_state_tensors_hashed": True, "hash_compared_before_true_result": True},
    "result": "所有明列斷言通過；本機只做單步線性層更新，未重跑既有 GPU 訓練。",
}
with (DESTINATION / "verification.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
