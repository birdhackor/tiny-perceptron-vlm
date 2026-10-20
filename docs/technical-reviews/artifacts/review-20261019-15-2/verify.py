"""保存原始來源摘錄，核對獨立參數與教材算例；不訓練模型。"""

import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def fetch(url):
    command = ["curl", "-fLsS", "--max-time", "45", url]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return raw, {"url": url, "command_argv": command, "retrieved_at": datetime.now(UTC).isoformat(), "full_sha256": sha(raw)}


commit = torch.version.git_version
snapshots = []
targets = {
    "torch/nn/modules/container.py": [("class ModuleList(", "    _modules:"), ("    def __getitem__", "    def __setitem__"), ("    def __iter__", "    def __iadd__")],
    "torch/nn/modules/module.py": [("def _forward_unimplemented", "class Module:"), ("    def parameters(", "    def named_parameters("), ("    def named_parameters(", "    def buffers("), ("    def _named_members(", "    def parameters("), ("    def __setattr__(", "    def __delattr__(")],
    "torch/nn/modules/linear.py": [("class Linear(", "    __constants__"), ("    def reset_parameters(", "    def extra_repr(")],
    "torch/autograd/grad_mode.py": [("class no_grad(", "    def __enter__(")],
}
for path, ranges in targets.items():
    raw, receipt = fetch(f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{path}")
    body = raw.decode()
    if path.endswith("/container.py"):
        body = body[body.index("class ModuleList("):body.index("class ModuleDict(")]
    excerpts = []
    for start, stop in ranges:
        begin = body.index(start)
        end = body.index(stop, begin + len(start))
        excerpts.append({"start_identifier": start.strip(), "end_identifier_exclusive": stop.strip(), "text": body[begin:end]})
    snapshots.append({**receipt, "version": f"PyTorch {torch.__version__}, commit {commit}", "excerpts": excerpts})

raw, receipt = fetch("https://arxiv.org/html/2101.03961v3")
parser = PlainText()
parser.feed(raw.decode())
paper_text = re.sub(r"\s+", " ", " ".join(parser.parts))
needles = ["We replace the dense feed forward network", "Mixture of Expert Routing."]
excerpts = []
for needle in needles:
    matches = list(re.finditer(re.escape(needle), paper_text, re.IGNORECASE))
    for match in matches[:1]:
        excerpts.append({"search_term": needle, "text": paper_text[max(0, match.start() - 40):match.end() + 850]})
snapshots.append({**receipt, "version": "arXiv:2101.03961v3 HTML, Switch Transformers", "excerpts": excerpts})
raw, receipt = fetch("https://arxiv.org/html/1706.03762v7")
parser = PlainText()
parser.feed(raw.decode())
paper_text = re.sub(r"\s+", " ", " ".join(parser.parts))
start = paper_text.index("two linear transformations")
snapshots.append({**receipt, "version": "arXiv:1706.03762v7 HTML, Attention Is All You Need", "excerpts": [{"search_term": "two linear transformations", "text": paper_text[start - 160:start + 600]}]})
write("primary-sources.json", {"sources": snapshots})

run = ROOT / "outputs/reviewer-tools/runs/20261020T065442130466Z"
copied = {}
for name in ("section.md", "fence-1.py", "extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"):
    raw = (run / name).read_bytes()
    copied[name] = {"sha256": sha(raw), "content": raw.decode()}
write("section-execution.json", {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/15.md#15.2" --execute', "copied_files": copied})

frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/15.2.md"
body = frozen.read_text()
code = re.search(r"```python\n(.*?)```", body, re.DOTALL)[1]
modified = code.replace("x = torch.tensor([[1.0, 2.0]])", "x = torch.tensor([[1.0, 2.0]])\nwith torch.no_grad():\n    experts[0].weight *= 3")
exercise_namespace = {}
exercise_stdout = io.StringIO()
with redirect_stdout(exercise_stdout):
    exec(compile(modified, "exercise", "exec"), exercise_namespace)
experts = exercise_namespace["experts"]
x = exercise_namespace["x"]
exercise_values = [expert(x).tolist() for expert in experts]
assert exercise_values == [[[3.0, 6.0]], [[2.0, 4.0]], [[2.0, 1.0]]]
assert len({expert.weight.data_ptr() for expert in experts}) == 3
assert [p.numel() for p in experts.parameters()] == [4, 4, 4]
copy_target = nn.Linear(2, 2, bias=False)
identity_before = id(copy_target.weight)
table = torch.eye(2)
with torch.no_grad():
    copied_tensor = copy_target.weight.copy_(table)
copy_behavior = {"same_parameter_object_after_copy": id(copy_target.weight) == identity_before, "copy_returns_destination": copied_tensor is copy_target.weight, "separate_storage_from_table": copy_target.weight.data_ptr() != table.data_ptr(), "grad_fn": str(copy_target.weight.grad_fn), "requires_grad_after_copy": copy_target.weight.requires_grad}
assert all(copy_behavior[key] for key in ("same_parameter_object_after_copy", "copy_returns_destination", "separate_storage_from_table", "requires_grad_after_copy"))
plain_parent = nn.Module()
plain_parent.experts = [nn.Linear(2, 2, bias=False) for _ in range(3)]
registered_parent = nn.Module()
registered_parent.experts = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
try:
    registered_parent.experts(x)
except NotImplementedError as error:
    forward_error = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("ModuleList 呼叫應缺少 forward")
shared = nn.Linear(2, 2, bias=False)
repeated = nn.ModuleList([shared, shared, shared])
list_behavior = {"plain_list_parameter_count": sum(p.numel() for p in plain_parent.parameters()), "module_list_parameter_count": sum(p.numel() for p in registered_parent.parameters()), "integer_index_identity": registered_parent.experts[0] is list(registered_parent.experts)[0], "zip_pairs": [(i, name) for i, name in zip(range(3), ["identity", "double", "swap"])], "distinct_modules": len({id(m) for m in experts}), "repeated_module_unique_count": len({id(m) for m in repeated}), "repeated_parameter_count": sum(p.numel() for p in repeated.parameters()), "module_list_forward": forward_error}
assert list_behavior["plain_list_parameter_count"] == 0
assert list_behavior["module_list_parameter_count"] == 12
assert list_behavior["repeated_parameter_count"] == 4

from tiny_perceptron.model import ModelConfig, TinyLM

record_path = ROOT / "docs/course-experiments/results/moe.json"
record = json.loads(record_path.read_bytes())
variants = {}
for name, variant in record["results"]["variants"].items():
    if not variant["model"]["config"]["experts"]:
        continue
    model = TinyLM(ModelConfig(**variant["model"]["config"]))
    expert_counts = [[sum(p.numel() for p in expert.parameters()) for expert in block.ffn.experts] for block in model.blocks]
    expert_total = sum(sum(counts) for counts in expert_counts)
    router_total = sum(sum(p.numel() for p in block.ffn.router.parameters()) for block in model.blocks)
    total = sum(p.numel() for p in model.parameters())
    expert_weights = [expert.up.weight for block in model.blocks for expert in block.ffn.experts]
    assert len({p.data_ptr() for p in expert_weights}) == 8
    assert expert_counts == [[33088] * 4] * 2
    assert expert_total == 264704
    assert total == variant["model"]["parameters"] == 340608
    assert expert_total == variant["budget"]["expert_parameters"]
    variants[name] = {"config": variant["model"]["config"], "expert_counts": expert_counts, "expert_parameters": expert_total, "router_parameters": router_total, "other_parameters": total - expert_total - router_total, "total_parameters": total, "training_steps": variant["training"]["steps"], "variant_result_fields": sorted(variant), "heldout_fields": {key: sorted(value) for key, value in variant["heldout"].items()}, "validation_routing_fields": sorted(variant["validation_routing"]), "routing_layer_fields": sorted(variant["validation_routing"]["layers"][0]), "effective_routing_inputs": variant["validation_routing"]["effective_input_tokens"]}

result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-2/verify.py", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": commit, "device": "cpu"}, "source_sha256": sha(frozen.read_bytes()), "exercise_stdout": exercise_stdout.getvalue(), "exercise_values": exercise_values, "parameter_elements": [p.numel() for p in experts.parameters()], "copy_behavior": copy_behavior, "list_behavior": list_behavior, "single_ffn_derivation": {"up_weight": 64 * 256, "up_bias": 256, "down_weight": 256 * 64, "down_bias": 64, "sum": 64 * 256 + 256 + 256 * 64 + 64}, "experiment_record": {"path": record_path.relative_to(ROOT).as_posix(), "sha256": sha(record_path.read_bytes()), "revision": record["revision"], "seed": record["seed"], "torch_version": record["torch_version"], "python_version": record["python_version"], "device": record["device"], "gpu": record["gpu"], "dataset": record["results"]["dataset"], "limitations": record["results"]["limitations"]}, "variants": variants, "scope": "CPU 確定性算例與隨機模型建構計數；只讀既有短訓紀錄及其欄位，沒有重跑訓練或核驗任何學科專長。"}
write("verification.json", result)
print(json.dumps(result, ensure_ascii=False, indent=2))
