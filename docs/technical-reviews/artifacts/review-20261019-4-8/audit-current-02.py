"""核對預設架構的參數、資料流與固定配方紀錄；不執行訓練。"""

import hashlib
import json
import platform
import re
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.data import SPECIALS, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(20261020)
torch.set_default_device("cpu")
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-8/audit-current-02.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": str(torch.get_default_dtype()),
        "seed": 20261020,
        "threads": torch.get_num_threads(),
    },
    "inputs_sha256": {
        path: digest(path)
        for path in (
            "tiny_perceptron/model.py",
            "tiny_perceptron/attention.py",
            "tiny_perceptron/modern.py",
            "tiny_perceptron/data.py",
            "scripts/course_experiments/text.py",
            "scripts/course_experiments/common.py",
            "course/chapters/02.md",
            "course/chapters/03.md",
            "course/chapters/04.md",
            "course/training.md",
            "docs/course-experiments/results/text_foundation.json",
            "docs/course-review-20261019/freeze-01/sources/4.8.md",
            "docs/course-review-20261019/freeze-07.json",
        )
    },
    "models": [],
}
body = (ROOT / "course/chapters/04.md").read_bytes()
section = re.search(rb"(?ms)^## 4\.8 .*?(?=^## |\Z)", body).group()
assert section == (ROOT / "docs/course-review-20261019/freeze-01/sources/4.8.md").read_bytes()
result["current_section_matches_freeze"] = True
training_text = (ROOT / "course/training.md").read_text()
training_section = re.search(r"(?ms)^## T\.4 .*?(?=^## |\Z)", training_text).group()
reproduction_command = ".venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda"
assert reproduction_command in training_section
attention_text = (ROOT / "course/chapters/03.md").read_text()
attention_section = re.search(r"(?ms)^## 3\.7 .*?(?=^## |\Z)", attention_text).group()
assert all(f"self.{name} = nn.Linear(width, width, bias=False)" in attention_section
           for name in ("q", "k", "v", "out"))
linear_text = (ROOT / "course/chapters/02.md").read_text()
linear_section = re.search(r"(?ms)^## 2\.3 .*?(?=^## |\Z)", linear_text).group()
assert "layer = nn.Linear(3, 2)" in linear_section and "layer.bias" in linear_section
result["linked_materials"] = {
    "fixed_recipe_command_present": reproduction_command,
    "attention_four_bias_free_linears_present": True,
    "linear_bias_example_present": True,
    "scope": "核對本節明確連結的材料與命令位置；未執行 CUDA 訓練命令。",
}

for width, depth, vocab, max_length in [
    (8, 1, 20, 128), (8, 2, 20, 128), (8, 3, 20, 128),
    (16, 1, 20, 128), (16, 2, 20, 128), (16, 3, 20, 128),
    (64, 2, 264, 128), (4, 2, 11, 17),
]:
    config = ModelConfig(width=width, layers=depth, vocab_size=vocab, max_length=max_length)
    model = TinyLM(config)
    fixed = 2 * vocab * width + max_length * width + 2 * width
    per_block = 12 * width**2 + 9 * width
    params = dict(model.named_parameters())
    observed_fixed = sum(p.numel() for n, p in params.items() if not n.startswith("blocks."))
    groups = []
    for block in model.blocks:
        observed = {
            "attention_weights": sum(p.numel() for p in block.attention.parameters()),
            "ffn_weights": block.ffn.up.weight.numel() + block.ffn.down.weight.numel(),
            "ffn_biases": block.ffn.up.bias.numel() + block.ffn.down.bias.numel(),
            "layernorm_affine": sum(p.numel() for p in block.norm1.parameters())
            + sum(p.numel() for p in block.norm2.parameters()),
        }
        assert observed == {
            "attention_weights": 4 * width**2,
            "ffn_weights": 8 * width**2,
            "ffn_biases": 5 * width,
            "layernorm_affine": 4 * width,
        }
        assert all(getattr(block.attention, name).bias is None for name in ("q", "k", "v", "out"))
        groups.append(observed)
    assert model.output.bias is None
    assert model.embedding.weight.data_ptr() != model.output.weight.data_ptr()
    assert len({p.data_ptr() for p in model.parameters()}) == len(params)
    block_inputs, block_outputs, handles, linear_calls = [], [], [], []
    for block in model.blocks:
        handles.append(block.register_forward_pre_hook(lambda m, a: block_inputs.append(a[0].detach().clone())))
        handles.append(block.register_forward_hook(lambda m, a, o: block_outputs.append(o[0].detach().clone())))
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(lambda m, a, o, name=name: linear_calls.append(name)))
    saved = []

    def pack(tensor):
        saved.append({"shape": list(tensor.shape), "numel": tensor.numel(), "dtype": str(tensor.dtype)})
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        logits = model(torch.tensor([[1, 2]], dtype=torch.int64))["logits"]
        loss = logits.square().mean()
    assert logits.shape == (1, 2, vocab)
    for previous, current in zip(block_outputs, block_inputs[1:]):
        assert torch.equal(previous, current)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for handle in handles:
        handle.remove()
    assert observed_fixed == fixed
    assert model.description()["parameters"] == fixed + depth * per_block
    result["models"].append({
        "description": model.description(),
        "input": [[1, 2]],
        "input_dtype": "torch.int64",
        "mask": "預設因果遮罩，沒有 padding 或 packing",
        "fixed_parameters": observed_fixed,
        "block_groups": groups,
        "expected_total": fixed + depth * per_block,
        "logits_shape": list(logits.shape),
        "independent_parameter_storage": True,
        "sequential_updated_representations": True,
        "all_parameters_receive_finite_gradients": True,
        "linear_forward_calls": linear_calls,
        "saved_tensor_count": len(saved),
        "saved_numel_sum_including_repeated_references_and_parameters": sum(item["numel"] for item in saved),
        "saved_tensor_scope": "只記 autograd 保存事件，含參數及重複引用，不能當作記憶體峰值或運算時間。",
    })

record = json.loads((ROOT / "docs/course-experiments/results/text_foundation.json").read_text())
training = record["results"]["training"]
assert training["parameters"] == 141568
result["training_receipt"] = {
    "experiment_id": record["experiment_id"],
    "revision": record["revision"],
    "status": record["status"],
    "evidence_status": record["evidence_status"],
    "device": record["device"],
    "gpu": record["gpu"],
    "seed": record["seed"],
    "torch_version": record["torch_version"],
    "python_version": record["python_version"],
    "training": training,
    "data": record["results"]["data"],
    "causal_max_difference": record["results"]["causal_max_difference"],
    "scaling_models": {
        name: {"parameters": item["parameters"], "steps": item["training"]["steps"],
               "unique_training_records": item["unique_training_records"]}
        for name, item in record["results"]["scaling"].items()
    },
    "scope": "讀取既有執行紀錄與現行配方，未下載權重或重跑歷史訓練；縮放組改了寬度與資料量，未提供固定寬度的深度對照。",
}
tok = ByteTokenizer()
encoded = tok.encode("中")
assert len(SPECIALS) == 8 and tok.vocab_size == 256 + len(SPECIALS)
assert len(encoded) == 3 and tok.decode(encoded) == "中"
result["byte_vocabulary"] = {
    "specials": list(SPECIALS), "byte_values": 256, "vocab_size": tok.vocab_size,
    "example": "中", "utf8_bytes": list("中".encode()), "encoded": encoded,
}
result["width_ratios"] = {"square_weights": 16**2 / 8**2, "embedding_width": 16 / 8,
                           "block_parameters": 3216 / 840}
assert list(torch.nn.GELU().parameters()) == []
result["gelu_parameters"] = 0
result["result"] = "全部結構、整數公式、資料流與紀錄欄位核對通過。"


previous_path = "docs/course-review-20261019/reviews/rv1019_technical_4-8/technical-4-8-current-01.json"
previous_hash = "840a775bec9548b59ee7c6ae8f495d101e6fffd50a4ef522e20feae8a231789b"
original_path = "docs/course-review-20261019/reviews/rv1019_technical_4-8/technical-4-8-original-01.json"
original_hash = "c3bdd330fec759cb12a847e3b6d45ecc3ba867cd6b18712e8c3c768ec355b4ee"
assert digest(previous_path) == previous_hash
assert digest(original_path) == original_hash
previous_report = json.loads((ROOT / previous_path).read_bytes())
original_report = json.loads((ROOT / original_path).read_bytes())
assert previous_report["issues"] == []
source_hashes = {}
for source in previous_report["sources"]:
    if source["kind"] != "repository_code":
        continue
    current_hash = digest(source["path"])
    source_hashes[source["id"]] = {
        "path": source["path"], "previous_sha256": source["sha256"],
        "current_sha256": current_hash, "changed": current_hash != source["sha256"],
    }
retained = {}
for report in (previous_report, original_report):
    for artifact in report["artifacts"]:
        assert digest(artifact["path"]) == artifact["sha256"]
        retained[artifact["path"]] = {"path": artifact["path"], "sha256": artifact["sha256"], "unchanged": True}
for name in ("self-check-output.txt", "self-check-output-current-01.txt"):
    path = "docs/technical-reviews/artifacts/review-20261019-4-8/" + name
    assert digest(path) == "4583e10fe0d5a3bff9a8a687dd6623a14e6b685283d3b9e448ff6a8209a24002"
    retained[path] = {"path": path, "sha256": digest(path), "unchanged": True}
expected_current = {
    "course/training.md": "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b",
    "course/chapters/03.md": "5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5",
}
assert {item["path"] for item in source_hashes.values() if item["changed"]} == set(expected_current)
checks = []
for source_id, section_id, affected_claims in (
    ("training-guide", "T.4", ["reproduction-link"]),
    ("course-attention", "3.7", ["attention-output-bias"]),
):
    record = source_hashes[source_id]
    path = record["path"]
    assert record["current_sha256"] == expected_current[path]
    diff_command = ["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", path]
    difference = subprocess.run(diff_command, cwd=ROOT, capture_output=True, check=True)
    versions, sections = [], []
    pattern = rb"(?ms)^## " + re.escape(section_id.encode()) + rb" .*?(?=^## |\Z)"
    for revision in ("ecf6dec71280", "c6b9339d2595"):
        command = ["git", "show", revision + ":" + path]
        shown = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
        raw = shown.stdout
        part = re.search(pattern, raw).group()
        versions.append({"command_argv": command, "exit_code": shown.returncode,
                         "file_sha256": hashlib.sha256(raw).hexdigest(),
                         "used_section_sha256": hashlib.sha256(part).hexdigest()})
        sections.append(part)
    current_section = re.search(pattern, (ROOT / path).read_bytes()).group()
    assert versions[0]["file_sha256"] == record["previous_sha256"]
    assert versions[1]["file_sha256"] == record["current_sha256"]
    assert sections[0] == sections[1] == current_section
    checks.append({"source_id": source_id, "path": path, "used_section": path + "#" + section_id,
                   "affected_claim_ids": affected_claims, "diff_command_argv": diff_command,
                   "diff_exit_code": difference.returncode, "diff_stdout": difference.stdout.decode(),
                   "diff_stderr": difference.stderr.decode(), "versions": versions,
                   "current_file_sha256": record["current_sha256"],
                   "current_used_section_sha256": hashlib.sha256(current_section).hexdigest(),
                   "used_section_unchanged": True,
                   "judgment": "引用的小節逐位元組相同，改動沒有影響該主張；現行小節已重新讀取及執行核對。"})
freeze_path = "docs/course-review-20261019/freeze-07.json"
assert digest(freeze_path) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(item for item in freeze["pages"] if item["page_id"] == "4.8")
assert page["source_sha256"] == hashlib.sha256(section).hexdigest()
assert page["figures_sha256"] == {}
result["recheck"] = {
    "previous_report": {"path": previous_path, "sha256": previous_hash, "unchanged": True},
    "original_report": {"path": original_path, "sha256": original_hash, "unchanged": True},
    "previous_issues": previous_report["issues"], "resolved_issue_ids": [],
    "repository_source_hashes": source_hashes, "retained_artifacts": list(retained.values()),
    "dependency_checks": checks, "page_unchanged": True, "changed_page_sentences": [],
    "source_freeze": {"path": freeze_path, "sha256": digest(freeze_path)},
    "all_claim_ids_rechecked": [item["id"] for item in previous_report["claims"]],
    "reused_original_primary_sources": "原始來源沿用本人先前親讀的固定版本及未變artifact，本次未重新下載。",
    "scope_exceptions": "只讀本節、本人報告指名的證據及工具本次生成的執行目錄；未讀其他角色報告或其他舊審閱紀錄。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
