"""核對修訂文字、既有證據與現行程式；所有輸出另存新檔。"""

import ast
import difflib
import hashlib
import inspect
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from tiny_perceptron.modern import DenseFFN
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(relative):
    return (ROOT / relative).read_bytes()


def function(source, name):
    tree = ast.parse(source)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


class NormalizeConfig(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "config":
            return ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="config", ctx=ast.Load())
        return node


torch.set_num_threads(1)
torch.manual_seed(19)
base = Path(__file__).resolve().parent
original_path = "docs/course-review-20261019/reviews/rv1019_technical_19-2/technical-19-2-original-01.json"
original = json.loads(read(original_path))
assert sha(read(original_path)) == "31be47d6293f4e2a21942003637c0308e7ce9fd92beb7cb0989adaafb35cebff"
checks = []
for group in ["sources", "artifacts"]:
    for item in original[group]:
        if "path" in item:
            actual = sha(read(item["path"]))
            assert actual == item["sha256"]
            checks.append({"group": group, "id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})
old = read("docs/course-review-20261019/freeze-01/sources/19.2.md")
new = read("docs/course-review-20261019/freeze-05/sources/19.2.md")
assert sha(old) == original["source_sha256"]
assert sha(new) == "95130a7e3b57ea0c045761eec0fece91a74df49d7cc74422eff9811ad534f8cd"
assert sha(read("docs/course-review-20261019/freeze-05.json")) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
chapter = read("course/chapters/19.md")
start = re.search(rb"(?m)^## 19\.2 ", chapter).start()
following = re.search(rb"(?m)^## ", chapter[start + 1:])
end = start + 1 + following.start() if following else len(chapter)
assert chapter[start:end] == new
old_fences = re.findall(r"```python\n(.*?)```", old.decode(), re.S)
new_fences = re.findall(r"```python\n(.*?)```", new.decode(), re.S)
assert old_fences == new_fences and len(new_fences) == 3
oldlines, newlines = old.decode().splitlines(), new.decode().splitlines()
changes = []
for tag, oldstart, oldend, newstart, newend in difflib.SequenceMatcher(None, oldlines, newlines).get_opcodes():
    if tag != "equal":
        changes.append({"operation": tag, "old_text": oldlines[oldstart:oldend], "new_text": newlines[newstart:newend]})
assert len(changes) == 1 and len(changes[0]["new_text"]) == 1

scope = {}
exec(compile(new_fences[1], "<current-frozen-class>", "exec"), scope)
simple_type = scope["LimitedAssistant"]
simple_init = function(new_fences[1], "__init__")
library_init = function(inspect.getsource(LimitedAssistant), "__init__")
parts = ["lm", "vision_encoder", "ocr_encoder", "audio_encoder"]
constructors = {}
for part in parts:
    def assigned_value(body):
        return next(n.value for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self" and t.attr == part for t in n.targets))
    simple_call = NormalizeConfig().visit(assigned_value(simple_init.body))
    actual_call = assigned_value(library_init.body)
    assert ast.dump(simple_call, include_attributes=False) == ast.dump(actual_call, include_attributes=False)
    constructors[part] = {"simple_normalized": ast.unparse(simple_call), "library": ast.unparse(actual_call), "matches": True}
models = {}
for architecture in ["moe", "dense"]:
    config = SelftrainedConfig(vocab_size=550, width=256, ffn_hidden=512, top_k=2, architecture=architecture)
    library, simple = LimitedAssistant(config), simple_type(config)
    loaded = simple.load_state_dict(library.state_dict())
    names = set(dict(library.named_parameters()))
    assert names == set(dict(simple.named_parameters()))
    assert all(dict(library.named_parameters())[name].shape == dict(simple.named_parameters())[name].shape for name in names)
    assert all(torch.equal(library.state_dict()[name], simple.state_dict()[name]) for name in library.state_dict())
    assert simple.description() == {key: library.description()[key] for key in simple.description()}
    models[architecture] = {"load_result": str(loaded), "same_parameter_names": True, "same_parameter_shapes": True, "all_loaded_tensors_equal": True, "named_parameter_count": len(names), "counts": simple.description()}

source, destination = DenseFFN(2, 4, activation="gelu"), DenseFFN(2, 4, activation="relu2")
with torch.no_grad():
    for parameter in source.parameters():
        parameter.fill_(0.5)
loaded = destination.load_state_dict(source.state_dict())
x = torch.tensor([[1.0, 2.0]])
activation = {"load_result": str(loaded), "source_activation": source.activation, "destination_activation": destination.activation, "source_output": source(x).detach().tolist(), "destination_output": destination(x).detach().tolist(), "outputs_equal": torch.equal(source(x), destination(x)), "gelu_parameters": len(list(nn.GELU().parameters())), "gelu_state_keys": list(nn.GELU().state_dict())}
assert not activation["outputs_equal"] and activation["gelu_parameters"] == 0
rejections = {}
for label, state in [("missing_name", {key: value for key, value in source.state_dict().items() if key != "up.bias"}), ("wrong_shape", {**source.state_dict(), "up.bias": torch.zeros(5)})]:
    try:
        destination.load_state_dict(state)
    except RuntimeError as error:
        rejections[label] = str(error)
    else:
        raise AssertionError("預設嚴格載入未拒絕不相容狀態")

previous_sources = json.loads(read("docs/technical-reviews/artifacts/review-20261019-19-2/original-source-excerpts.json"))
module_raw = Path(inspect.getsourcefile(nn.Module)).read_bytes()
assert sha(module_raw) == previous_sources["sources"][1]["local_file_sha256"]
assert sha(module_raw) == previous_sources["sources"][1]["receipt"]["response_sha256"]
result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-2/verify-current-01.py", "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "seed": "19"}, "original_report": {"path": original_path, "sha256": sha(read(original_path))}, "source_sha256": sha(new), "current_section_matches_frozen": True, "freeze_sha256": sha(read("docs/course-review-20261019/freeze-05.json")), "changes": changes, "python_fences_unchanged": True, "fence_sha256": [sha(code.encode()) for code in new_fences], "reused_evidence_hash_checks": checks, "constructor_comparison": constructors, "strict_loading": models, "incompatible_states_rejected": rejections, "stateless_activation_not_checked": activation, "torch_official_source_unchanged": {"git_commit": torch.version.git_version, "sha256": sha(module_raw), "matches_original_upstream_receipt": True}, "official_source_inspection": {"load_state_dict": inspect.getsource(nn.Module.load_state_dict)[:1000], "shape_check": "if not is_param_lazy and input_param.shape != param.shape:", "scope": "本節使用預設 strict=True、assign=False 與標準非 lazy 模組，沒有自訂載入 hook；此 API 不比較 activation 與 forward。"}, "preserved_extra_artifacts": {"self-check.json": sha((base / "self-check.json").read_bytes())}, "result": "所有斷言通過；改過的三項敘述均成立，原問題已解決；未改主張的全部 repo 來源與既有正式證據均未變。"}
with (base / "verification-current-01.json").open("x", encoding="utf-8") as output:
    json.dump(result, output, ensure_ascii=False, indent=2)
    output.write("\n")
print(json.dumps({"source_sha256": result["source_sha256"], "evidence_hash_checks": len(checks), "constructor_comparison": constructors, "strict_loading": models, "stateless_activation_not_checked": activation, "result": result["result"]}, ensure_ascii=False, indent=2))
