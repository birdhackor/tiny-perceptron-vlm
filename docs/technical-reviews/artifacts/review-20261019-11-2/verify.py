import contextlib
import hashlib
import inspect
import io
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def receipt(suffix, excerpts):
    url = "https://docs.pytorch.org/docs/2.14/" + suffix
    command = ["curl", "-L", "--fail", "--silent", "--show-error", url]
    result = subprocess.run(command, capture_output=True, check=True)
    parser = Text()
    parser.feed(result.stdout.decode())
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    for item in excerpts:
        assert item["quote"] in text, "原始來源未找到指定摘錄"
    return {
        "url": url,
        "version": "PyTorch 2.14 版本化官方文件",
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "command_argv": command,
        "exit_code": result.returncode,
        "stderr": result.stderr.decode(),
        "response_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "response_bytes": len(result.stdout),
        "excerpts": excerpts,
    }


torch.set_num_threads(1)
torch.manual_seed(0)
torch.set_default_device("cpu")
source_path = ROOT / "docs/course-review-20261019/freeze-01/sources/11.2.md"
source = source_path.read_text()
code = re.search(r"```python\n(.*?)```", source, re.S).group(1)
namespace = {}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(code, str(source_path), "exec"), namespace)
model = namespace["model"]
optimizer = namespace["optimizer"]
selected = [p for p in model.parameters() if p.requires_grad]
listed = [p for group in optimizer.param_groups for p in group["params"]]
assert [id(p) for p in selected] == [id(p) for p in listed], "名單物件不一致"
assert namespace["trainable"] == [("image_projector.weight", 128), ("image_projector.bias", 8)]
assert all(p.grad is None for p in model.parameters())
assert len(optimizer.state) == 0
baseline = {
    "stdout": printed.getvalue(),
    "trainable": namespace["trainable"],
    "parameter_groups": len(optimizer.param_groups),
    "groups_are_dicts": all(isinstance(group, dict) for group in optimizer.param_groups),
    "params_are_lists": all(isinstance(group["params"], list) for group in optimizer.param_groups),
    "same_parameter_objects": True,
    "all_grads_none": True,
    "optimizer_state_entries": len(optimizer.state),
    "language_width": model.language.config.width,
    "vision_width": model.vision.projection.out_features,
    "image_weight_shape": list(model.image_projector.weight.shape),
    "image_bias_shape": list(model.image_projector.bias.shape),
}

exercise_code = code.replace(
    "model.image_projector.requires_grad_(True)",
    "model.image_projector.requires_grad_(True)\nmodel.language.blocks[-1].requires_grad_(True)",
)
exercise_namespace = {}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(exercise_code, "exercise", "exec"), exercise_namespace)
exercise_model = exercise_namespace["model"]
block_entries = [(name, p.numel()) for name, p in exercise_model.named_parameters() if name.startswith("language.blocks.")]
exercise_total = sum(p.numel() for group in exercise_namespace["optimizer"].param_groups for p in group["params"])
assert len(exercise_model.language.blocks) == 1
assert exercise_model.language.blocks[-1] is exercise_model.language.blocks[0]
assert sum(count for _, count in block_entries) == 840
assert exercise_total == 976
exercise = {
    "stdout": printed.getvalue(),
    "block_parameters": block_entries,
    "block_count": len(exercise_model.language.blocks),
    "last_block_is_first": True,
    "extra_parameter_count": sum(count for _, count in block_entries),
    "optimizer_parameter_count": exercise_total,
    "derivation": "width=8；兩個 LayerNorm 各 8+8，共 32；q/k/v/out 各 8×8，共 256；FFN 為 (8×32+32)+(32×8+8)=552；區塊合計 840；接頭 (16×8)+8=136；總數 840+136=976。",
}

torch.manual_seed(0)
grad_model = MultiModalLM(TinyLM(ModelConfig(width=8)))
grad_model.requires_grad_(False)
grad_model.image_projector.requires_grad_(True)
tok = ByteTokenizer()
answer = tok.encode("red square") + [tok.eos_id]
prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)
out = grad_model(ids, labels, image=scene())
loss = masked_loss(out["logits"], out["labels"])
loss.backward()
gradient_norm = grad_model.image_projector.weight.grad.norm().item()
assert gradient_norm > 0
assert all(p.grad is None for p in grad_model.language.parameters())
frozen_path = {
    "seed": 0,
    "loss": loss.item(),
    "valid_targets": int((out["labels"] != -100).sum()),
    "projector_gradient_norm": gradient_norm,
    "language_grads_all_none": True,
    "parameter_updates": 0,
}
features = grad_model.image_projector(grad_model.vision(scene().unsqueeze(0)))
with torch.no_grad():
    detached_logits = grad_model.language(embeddings=features)["logits"]
assert not detached_logits.requires_grad
frozen_path["no_grad_language_logits_require_grad"] = detached_logits.requires_grad

layer = nn.Linear(2, 1)
before_flags = [p.requires_grad for p in layer.parameters()]
layer.eval()
y = layer(torch.ones(1, 2))
y.sum().backward()
assert before_flags == [p.requires_grad for p in layer.parameters()]
assert y.requires_grad and all(p.grad is not None for p in layer.parameters())
evaluation = {"training": layer.training, "requires_grad_unchanged": True, "backward_completes": True}

stage_optimizer = torch.optim.AdamW(grad_model.image_projector.parameters(), lr=0.001)
stage_optimizer.step()
old_parameters = list(grad_model.image_projector.parameters())
old_states = {id(p): {key: value.clone() for key, value in stage_optimizer.state[p].items()} for p in old_parameters}
grad_model.image_projector.requires_grad_(False)
grad_model.language.blocks[-1].requires_grad_(True)
held = [p for group in stage_optimizer.param_groups for p in group["params"]]
assert [id(p) for p in held] == [id(p) for p in old_parameters]
assert not {id(p) for p in grad_model.language.blocks[-1].parameters()} & {id(p) for p in held}
assert all(torch.equal(value, stage_optimizer.state[p][key]) for p in old_parameters for key, value in old_states[id(p)].items())
assert all(p.grad is not None for p in old_parameters)
fresh_optimizer = torch.optim.AdamW([p for p in grad_model.parameters() if p.requires_grad], lr=0.001)
fresh_parameters = [p for group in fresh_optimizer.param_groups for p in group["params"]]
assert {id(p) for p in fresh_parameters} == {id(p) for p in grad_model.language.blocks[-1].parameters()}
stage = {
    "old_optimizer_still_holds_projector": True,
    "newly_unfrozen_block_absent_from_old_optimizer": True,
    "states_unchanged_by_requires_grad_toggle": True,
    "state_keys": sorted(stage_optimizer.state[old_parameters[0]]),
    "first_moment_norm": stage_optimizer.state[old_parameters[0]]["exp_avg"].norm().item(),
    "second_moment_norm": stage_optimizer.state[old_parameters[0]]["exp_avg_sq"].norm().item(),
    "toggle_preserves_existing_gradients": True,
    "fresh_optimizer_block_only": True,
    "fresh_optimizer_parameter_count": sum(p.numel() for p in fresh_parameters),
    "fresh_optimizer_state_entries": len(fresh_optimizer.state),
}

def frozen_step(grad, weight_decay, seed_momentum):
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    opt = torch.optim.AdamW([parameter], lr=0.1, weight_decay=weight_decay, foreach=False)
    if seed_momentum:
        parameter.grad = torch.ones_like(parameter)
        opt.step()
    parameter.requires_grad_(False)
    parameter.grad = None if grad == "none" else torch.zeros_like(parameter)
    before = parameter.item()
    old_step = opt.state.get(parameter, {}).get("step", torch.tensor(0.0)).item()
    opt.step()
    return {"gradient": grad, "weight_decay": weight_decay, "seed_momentum": seed_momentum, "before": before, "after": parameter.item(), "step_before": old_step, "step_after": opt.state.get(parameter, {}).get("step", torch.tensor(0.0)).item()}

none_case = frozen_step("none", 0.2, True)
zero_decay = frozen_step("zero", 0.2, False)
zero_momentum = frozen_step("zero", 0.0, True)
assert none_case["before"] == none_case["after"] and none_case["step_before"] == none_case["step_after"]
assert abs(zero_decay["after"] - 1.96) < 1e-12
assert zero_momentum["after"] < zero_momentum["before"]

receipts = [
    receipt("notes/autograd.html", [
        {"locator": "Setting requires_grad", "quote": "only leaf tensors with requires_grad=True"},
        {"locator": "No-grad Mode", "quote": "computations in no-grad mode are never recorded in the backward graph"},
        {"locator": "Evaluation Mode (nn.Module.eval())", "quote": "orthogonal to no-grad mode and inference mode"},
    ]),
    receipt("generated/torch.nn.Module.html", [
        {"locator": "Module.named_parameters", "quote": "yielding both the name of the parameter as well as the parameter itself"},
        {"locator": "Module.requires_grad_", "quote": "requires_grad attributes in-place"},
    ]),
    receipt("optim.html", [
        {"locator": "Per-parameter options", "quote": "params key, containing a list of parameters belonging to it"},
    ]),
    receipt("generated/torch.optim.AdamW.html", [
        {"locator": "AdamW.zero_grad", "quote": "skips the step altogether"},
        {"locator": "AdamW.add_param_group", "quote": "frozen layers can be made trainable and added to the Optimizer"},
    ]),
    receipt("generated/torch.numel.html", [
        {"locator": "torch.numel", "quote": "Returns the total number of elements in the input tensor."},
    ]),
]

implementation = []
for obj, needle in [(nn.Module.requires_grad_, "p.requires_grad_(requires_grad)"), (nn.Module.eval, "return self.train(False)"), (torch.optim.AdamW._init_group, "if p.grad is not None:")]:
    inspected = inspect.getsource(obj)
    path = Path(inspect.getsourcefile(obj))
    assert needle in inspected
    implementation.append({"object": obj.__qualname__, "installed_file": str(path.relative_to(ROOT)), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "excerpt": needle})

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-2/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype_model": "torch.float32", "dtype_scalar": "torch.float64", "threads": str(torch.get_num_threads())},
    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "baseline": baseline,
    "exercise": exercise,
    "frozen_gradient_path": frozen_path,
    "evaluation": evaluation,
    "stage_transition": stage,
    "none_gradient": none_case,
    "zero_gradient_decay": zero_decay,
    "zero_gradient_momentum": zero_momentum,
    "original_source_receipts": receipts,
    "installed_source_inspection": implementation,
    "assertions_passed": True,
}, ensure_ascii=False, indent=2))
