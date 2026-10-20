"""核對凍結程式、轉換語意、抽題與既有實驗紀錄；不執行訓練。"""

import ast
import contextlib
import copy
import hashlib
import inspect
import io
import json
import platform
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.common import split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import QuantizedLinear, replace_linear_layers


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


torch.set_num_threads(1)
raw = (ROOT / "docs/course-review-20261019/freeze-01/sources/17.9.md").read_bytes()
assert sha(raw) == "4f34cf2c4688979885009a4e22eb7e2ce4da081b01cb3536d194ead64294d9f3"
fences = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(fences) == 3
extracted = ast.parse(fences[1]).body[0]
library = ast.parse(inspect.getsource(replace_linear_layers)).body[0]
assert ast.dump(extracted, include_attributes=False) == ast.dump(library, include_attributes=False)
result = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
        "threads": str(torch.get_num_threads()),
    },
    "source_sha256": sha(raw),
    "excerpt_ast_matches_library": True,
    "variants": {},
}
for bits in (4, 8):
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, code in enumerate(fences):
            if index == 0:
                code = code.replace("bits = 4", f"bits = {bits}")
            exec(compile(code, "凍結程式", "exec"), namespace)
    before, after = namespace["before"], namespace["after"]
    original, quantized = namespace["original"], namespace["quantized"]
    diff = (before - after).abs()
    assert tuple(before.shape) == tuple(after.shape) == (1, 3, 264)
    assert torch.equal(before, namespace["original_again"])
    assert torch.equal(namespace["again"](namespace["ids"])["logits"], after)
    assert not before.requires_grad and not after.requires_grad
    linear_names = [name for name, layer in original.named_modules() if isinstance(layer, nn.Linear)]
    quantized_names = [name for name, layer in quantized.named_modules() if isinstance(layer, QuantizedLinear)]
    assert linear_names == quantized_names
    for name in ("embedding", "position", "blocks.0.norm1", "blocks.0.norm2", "final_norm"):
        old, new = original.get_submodule(name), quantized.get_submodule(name)
        assert type(old) is type(new)
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    result["variants"][str(bits)] = {
        "stdout": output.getvalue(),
        "shape": list(after.shape),
        "input_dtype": str(namespace["ids"].dtype),
        "logits_dtype": str(after.dtype),
        "mae_denominator": diff.numel(),
        "absolute_error_sum": diff.sum().item(),
        "mae": diff.mean().item(),
        "original_max_difference": (before - namespace["original_again"]).abs().max().item(),
        "snippet_equals_library": True,
        "original_direct_children": [name for name, _ in original.named_children()],
        "linear_names": linear_names,
        "quantized_names": quantized_names,
        "retained_embedding_and_norms_equal_with_independent_storage": True,
        "packed_value_dtypes": sorted({str(layer.values.dtype) for layer in quantized.modules() if isinstance(layer, QuantizedLinear)}),
        "scale_dtypes": sorted({str(layer.scale.dtype) for layer in quantized.modules() if isinstance(layer, QuantizedLinear)}),
        "no_backward_graph_for_compared_logits": True,
    }
assert result["variants"]["8"]["mae"] < result["variants"]["4"]["mae"]
assert abs(result["variants"]["4"]["mae"] - 0.029) < 0.0005
skipped = {"__name__": "__main__"}
skipped_output = io.StringIO()
with contextlib.redirect_stdout(skipped_output):
    exec(fences[0], skipped)
    exec(fences[2], skipped)
assert skipped["replace_linear_layers"] is replace_linear_layers
assert namespace["replace_linear_layers"] is not replace_linear_layers
result["binding_check"] = {
    "skipping_definition_keeps_library_function": True,
    "executing_definition_binds_excerpt_function": True,
    "skipping_definition_stdout": skipped_output.getvalue(),
}

torch.manual_seed(0)
model = TinyLM(ModelConfig(width=8)).eval()
assert model.embedding.weight is not model.output.weight
assert model.embedding.weight.data_ptr() != model.output.weight.data_ptr()
clone = copy.deepcopy(model)
assert clone is not model
assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), clone.parameters(), strict=True))
old_layer = clone.output
returned = replace_linear_layers(clone, 4)
assert returned is clone and clone.output is not old_layer
ids = torch.tensor([[1, 2, 3]])
outside = model(ids)["logits"]
with torch.no_grad():
    inside = model(ids)["logits"]
assert outside.requires_grad and not inside.requires_grad
tied = TinyLM(ModelConfig(width=8, tied=True)).eval()
assert tied.embedding.weight is tied.output.weight
retained = tied.embedding.weight
replace_linear_layers(tied, 4)
assert tied.embedding.weight is retained and isinstance(tied.output, QuantizedLinear)
assert not hasattr(tied.output, "weight")
result["semantics"] = {
    "deepcopy_parameter_storage_independent": True,
    "conversion_returns_same_object_and_replaces_output": True,
    "eval_alone_preserves_autograd": outside.requires_grad,
    "no_grad_disables_backward_graph": not inside.requires_grad,
    "tied_before_conversion": True,
    "embedding_parameter_retained_but_output_has_buffers_after_conversion": True,
}

quant = read_json("docs/course-experiments/results/quantization.json")
sft = read_json("docs/course-experiments/results/sft.json")
qat = read_json("docs/course-experiments/results/qat.json")
parts = split_records(generate_records("attributes-sft"), seed=42)
counts = {key: len(rows) for key, rows in parts.items()}
families = {key: sorted({row["family"] for row in rows}) for key, rows in parts.items()}
assert counts == quant["results"]["data"]["counts"] == {"train": 45, "validation": 5, "test": 10}
assert {key: len(value) for key, value in families.items()} == quant["results"]["data"]["families"]
assert all(not set(families[a]) & set(families[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
dataset_sha = sha((json.dumps(parts, ensure_ascii=False, indent=2) + "\n").encode())
assert dataset_sha == quant["results"]["data"]["sha256"] == qat["results"]["data"]["sha256"]
examples = text_examples(parts["train"], mode="sft", max_length=128)
effective_per_example = [int((labels != IGNORE).sum()) for _, labels in examples]
rng = random.Random(42)
plan = [[rng.randrange(len(examples)) for _ in range(16)] for _ in range(120)]
effective = sum(effective_per_example[index] for batch in plan for index in batch)
plan_sha = sha(json.dumps(plan).encode())
assert effective == quant["results"]["training"]["effective_supervised_tokens"] == 13610
assert plan_sha == quant["results"]["training"]["batch_plan_sha256"]
assert quant["results"]["teacher_provenance"] == qat["results"]["teacher_provenance"]
assert quant["results"]["teacher_provenance"]["sha256"] == next(a["sha256"] for a in sft["artifacts"] if a["path"] == "model.pt")
tok = ByteTokenizer()
verified_scores = {}
for name, report in [("original_fp32", qat["results"]["runs"]["initial_fp32"]), *quant["results"]["runs"].items()]:
    rows = report["test"]["generated_samples"]
    assert len(rows) == 10
    correct = 0
    for expected_record, row in zip(parts["test"], rows, strict=True):
        assert row["family"] == expected_record["family"]
        assert row["question"] == expected_record["messages"][0]["content"]
        assert row["expected"] == expected_record["messages"][-1]["content"]
        generated_ids = row["generated_ids"]
        answer_ids = generated_ids[:generated_ids.index(tok.eos_id)] if tok.eos_id in generated_ids else generated_ids
        hit = answer_ids == tok.encode(row["expected"])
        assert hit == row["exact"]
        correct += hit
    assert correct == report["test"]["correct"]
    verified_scores[name] = correct
assert verified_scores == {"original_fp32": 5, "fp32": 6, "packed4": 6, "packed8": 6}
assert sft["results"]["after"]["test"]["matches"] == 5
result["experiment_audit"] = {
    "historical_environment": {key: quant[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu")},
    "teacher_provenance": quant["results"]["teacher_provenance"],
    "training": quant["results"]["training"],
    "counts": counts,
    "families": families,
    "dataset_sha256_recomputed": dataset_sha,
    "batch_plan_sha256_recomputed": plan_sha,
    "effective_supervised_tokens_recomputed_without_training": effective,
    "test_correct_recounted_from_raw_token_ids": verified_scores,
    "test_records": 10,
    "test_supervised_tokens": 69,
    "original_sft_test_matches": sft["results"]["after"]["test"]["matches"],
    "same_fp32_source_recorded": quant["results"]["same_fp32_source"],
    "reloaded_before_evaluation_recorded": quant["results"]["reloaded_packed_checkpoints_before_evaluation"],
    "scope": "核對既有紀錄、逐題 token、資料生成與抽題；沒有重跑歷史 CUDA 訓練或重新載入歷史權重。",
}
paths = [
    "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py",
    "tiny_perceptron/quantization.py", "tiny_perceptron/data.py", "scripts/prepare_data.py",
    "scripts/course_experiments/compression.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/text.py", "docs/course-experiments/results/quantization.json",
    "docs/course-experiments/results/sft.json", "docs/course-experiments/results/qat.json",
]
result["inspected_file_sha256"] = {path: sha((ROOT / path).read_bytes()) for path in paths}
result["historical_code_matches_current"] = {
    path: result["inspected_file_sha256"][path] == quant["code_sha256"][path]
    for path in paths if path in quant["code_sha256"]
}
historical = read_json("docs/technical-reviews/artifacts/review-20261019-17-9/historical-code.json")
current = ast.parse((ROOT / historical["path"]).read_text())
current_functions = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
function_matches = {
    name: ast.dump(ast.parse(code).body[0], include_attributes=False)
    == ast.dump(current_functions[name], include_attributes=False)
    for name, code in historical["functions"].items()
}
assert all(function_matches.values())
assert all(value for path, value in result["historical_code_matches_current"].items() if path != historical["path"])
result["historical_quantization_functions_match_current"] = function_matches
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
