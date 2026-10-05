"""Fresh 18.2 reviewer: bounded CPU checks, no training or weight files."""
import ast
import copy
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

section = (BASE / "frozen-input/18.2.md").read_bytes()
fence = re.search(rb"```python\n(.*?)```", section, re.S).group(1)
assert fence == (BASE / "execution/original-fence.py").read_bytes()
print("ORIGINAL FENCE BEGIN")
exec(compile(fence, "course/chapters/18.md#18.2:original-fence", "exec"), {})
print("ORIGINAL FENCE END")

rows = []
for width, layers, heads, expected in [(16, 2, 1, 16960), (8, 1, 1, 6104),
                                      (16, 1, 1, 13744), (64, 2, 1, 141568),
                                      (16, 1, 2, 13744), (32, 1, 2, 33632)]:
    torch.manual_seed(42)
    model = TinyLM(ModelConfig(width=width, layers=layers, heads=heads))
    parameters = sum(p.numel() for p in model.parameters())
    # Independent count from explicit shapes: embeddings and final norm,
    # four square attention matrices, two FFN matrices/biases, two norms.
    shared = 264 * width + 128 * width + 264 * width + 2 * width
    block = 4 * width * width + 8 * width * width + 5 * width + 4 * width
    assert parameters == shared + layers * block == expected
    assert all(p.dtype == torch.float32 and p.element_size() == 4 and p.requires_grad
               for p in model.parameters())
    physical_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert physical_bytes == parameters * 4
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert list(model.embedding(ids).shape) == [1, 3, width]
        assert list(model(ids)["logits"].shape) == [1, 3, 264]
    modules = {name: sum(p.numel() for p in child.parameters())
               for name, child in model.named_children()}
    row = dict(width=width, layers=layers, heads=heads, parameters=parameters,
               fp32_bytes=physical_bytes, shared=shared, per_block=block,
               modules=modules, output_shape=list(model.output.weight.shape))
    rows.append(row)
    print("ARCHITECTURE", json.dumps(row, sort_keys=True))

assert rows[1]["parameters"] != rows[0]["parameters"] / 8
assert rows[2]["parameters"] != rows[0]["parameters"] / 2

raw_result = (BASE / "code/docs/course-experiments/results/distillation.json").read_bytes()
result = json.loads(raw_result)
attributes = result["results"]["tasks"]["attributes"]
pointer_values = {}
def select(pointer):
    value = result
    for component in pointer.lstrip("/").split("/"):
        value = value[component]
    pointer_values[pointer] = value
    return value

for field in ["revision", "device", "seed", "torch_version", "python_version"]:
    select("/" + field)
teacher_config = select("/results/tasks/attributes/teacher_provenance/config")
assert teacher_config["width"] == 64 and teacher_config["layers"] == 2
assert select("/results/tasks/attributes/teacher_storage/parameter_count") == 141568
assert select("/results/tasks/attributes/teacher_storage/parameter_tensor_bytes") == 566272
assert select("/results/tasks/attributes/student_layers") == 1

for name in ["tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py"]:
    recorded = select("/code_sha256/" + name.replace("/", "~1")) if False else result["code_sha256"][name]
    pointer_values["/code_sha256/" + name.replace("/", "~1")] = recorded
    assert sha((BASE / "code" / name).read_bytes()) == recorded
historical = BASE / "code/compression-result-version.py"
assert sha(historical.read_bytes()) == result["code_sha256"]["scripts/course_experiments/compression.py"]
pointer_values["/code_sha256/scripts~1course_experiments~1compression.py"] = sha(historical.read_bytes())

for width, expected, byte_count in [(16, 13744, 54976), (32, 33632, 134528)]:
    hashes, modules = [], []
    for branch in ["ce", "teacher_hard", "ce_kl"]:
        prefix = f"/results/tasks/attributes/runs/w{width}_{branch}"
        hashes.append(select(prefix + "/training/initialization_sha256"))
        modules.append(select(prefix + "/storage/retained_float_modules"))
        assert select(prefix + "/storage/parameter_count") == expected
        assert select(prefix + "/storage/parameter_tensor_bytes") == byte_count
        for field in ["steps", "optimizer_updates", "batch_size", "effective_supervised_tokens", "training_examples"]:
            select(prefix + "/training/" + field)
    assert len(set(hashes)) == 1
    assert modules[0] == modules[1] == modules[2]
    torch.manual_seed(result["seed"])
    initial = TinyLM(ModelConfig(width=width, layers=1, heads=2, max_length=teacher_config["max_length"]))
    clones = [copy.deepcopy(initial) for _ in range(3)]
    assert all(parameter_hash(model) == hashes[0] for model in clones)
    assert all(torch.equal(model.output.weight, initial.output.weight) for model in clones)
    print("MATCHED INITIALIZATION", width, hashes[0], "output", list(initial.output.weight.shape))

tree = ast.parse(historical.read_bytes())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_distill_case")
width_loop = next(node for node in function.body if isinstance(node, ast.For))
initialization = next(node for node in width_loop.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "initial" for target in node.targets))
assert "heads=2" in ast.unparse(initialization)
assert "layers=1" in ast.unparse(initialization)
method_loop = next(node for node in width_loop.body if isinstance(node, ast.For))
fit = next(node.value for node in method_loop.body if isinstance(node, ast.Assign)
           and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
           and node.value.func.id == "_fit_text")
assert ast.unparse(fit.args[1]) == "copy.deepcopy(initial)"
assert sum(1 for node in ast.walk(method_loop) if isinstance(node, ast.Call)
           and isinstance(node.func, ast.Name) and node.func.id in {"TinyLM", "ModelConfig"}) == 0

environment = dict(python=sys.version, torch=str(torch.__version__),
                   torch_git_version=str(torch.version.git_version), device="cpu",
                   cuda_build=str(torch.version.cuda), cuda_available=str(torch.cuda.is_available()),
                   platform=platform.platform(), threads=str(torch.get_num_threads()),
                   original_result_sha256=sha(raw_result),
                   raw_section_sha256=sha(section))
(BASE / "execution/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
(BASE / "execution/selected-pointers.json").write_text(json.dumps(pointer_values, indent=2, ensure_ascii=False) + "\n")
(BASE / "execution/architecture-counts.json").write_text(json.dumps(rows, indent=2) + "\n")
print("PASS: original fence, variation, independent counts, FP32 bytes, output axes, raw result architecture and initialization")
