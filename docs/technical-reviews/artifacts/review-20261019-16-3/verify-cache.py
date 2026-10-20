"""核對快取的數值、位置、遮罩與既有實驗紀錄；不訓練模型。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.utils.rnn import pad_sequence

from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(left, right):
    return float((left - right).abs().max())


def compare(model, ids, reference_ids=None):
    reference_ids = ids if reference_ids is None else reference_ids
    full = model(reference_ids)["logits"][:, -1]
    prefix = model(ids[:, :3])
    step = model(ids[:, 3:], cache=prefix["cache"])
    cached = step["logits"][:, 0]
    return {
        "ids": ids.tolist(),
        "reference_ids": reference_ids.tolist(),
        "full_shape": list(full.shape),
        "cached_shape": list(cached.shape),
        "max_error": difference(full, cached),
        "allclose_atol_1e_6_default_rtol": torch.allclose(full, cached, atol=1e-6),
        "allclose_atol_1e_6_rtol_zero": torch.allclose(full, cached, atol=1e-6, rtol=0),
        "prefix_k_shape": list(prefix["cache"][0][0].shape),
        "step_k_shape": list(step["cache"][0][0].shape),
        "old_k_retained_exactly": torch.equal(prefix["cache"][0][0], step["cache"][0][0][:, :, :3]),
        "old_v_retained_exactly": torch.equal(prefix["cache"][0][1], step["cache"][0][1][:, :, :3]),
    }


torch.set_num_threads(1)
torch.manual_seed(0)
torch.set_default_device("cpu")
model = TinyLM(ModelConfig(width=8)).eval()
ids = torch.tensor([[1, 2, 3, 4]])
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-3/verify-cache.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "dtype": str(next(model.parameters()).dtype), "float32_bits": str(torch.finfo(torch.float32).bits), "threads": str(torch.get_num_threads())},
    "model_config": vars(model.config),
}
with torch.no_grad():
    result["baseline"] = compare(model, ids)
    result["new_last_id"] = compare(model, torch.tensor([[1, 2, 3, 5]]))
    result["changed_reference_prefix"] = compare(model, ids, torch.tensor([[9, 2, 3, 4]]))

    old = model(ids[:, :3])["cache"]
    changed = model(torch.tensor([[9, 2, 3]]))["cache"]
    result["changed_first_id_cache_differences"] = [
        {"layer": layer, "k_max_abs_by_position": (left[0] - right[0]).abs().amax(dim=(0, 1, 3)).tolist(), "v_max_abs_by_position": (left[1] - right[1]).abs().amax(dim=(0, 1, 3)).tolist()}
        for layer, (left, right) in enumerate(zip(old, changed))
    ]
    correct = model(ids[:, 3:], cache=old)["logits"]
    wrong = model(ids[:, 3:], cache=old, positions=torch.tensor([0]))["logits"]
    result["wrong_position"] = {"expected_position": old[0][0].shape[2], "supplied_position": 0, "max_logit_error": difference(correct, wrong)}

    shapes = {}
    handles = []
    for name, module in model.named_modules():
        if name.endswith(("attention.q", "attention.k", "attention.v", "ffn")):
            handles.append(module.register_forward_pre_hook(lambda _, args, name=name: shapes.setdefault(name, []).append(list(args[0].shape))))
    model(ids[:, 3:], cache=old)
    for handle in handles:
        handle.remove()
    result["cached_step_module_inputs"] = shapes

    result["cache_mask_rejections"] = {}
    for name, value in {"valid": torch.ones((1, 4), dtype=torch.bool), "segments": torch.zeros((1, 4), dtype=torch.long)}.items():
        try:
            model(ids[:, 3:], cache=old, **{name: value})
        except ValueError as error:
            result["cache_mask_rejections"][name] = {"exception": type(error).__name__, "message": str(error)}
        else:
            raise AssertionError("帶遮罩的快取呼叫必須報錯")

    torch.manual_seed(0)
    deeper = TinyLM(ModelConfig(width=8, layers=3)).eval()
    full = deeper(ids)
    prefix = deeper(ids[:, :3])
    result["append_invariance_each_layer"] = [
        {"layer": layer, "k_max_error": difference(pair[0][:, :, :3], old_pair[0]), "v_max_error": difference(pair[1][:, :, :3], old_pair[1])}
        for layer, (pair, old_pair) in enumerate(zip(full["cache"], prefix["cache"]))
    ]
    result["all_four_keys_allowed"] = attention_mask(torch.tensor([3]), torch.arange(4)).tolist()
    result["padded_batch"] = pad_sequence([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], batch_first=True, padding_value=ByteTokenizer.pad_id).tolist()
    scores = torch.tensor([1.0, 1.0 - 1e-7])
    perturbed = torch.tensor([1.0, 1.0 + 1e-7])
    result["near_tie_argmax"] = {"original": scores.tolist(), "perturbed": perturbed.tolist(), "max_error": difference(scores, perturbed), "original_id": int(scores.argmax()), "perturbed_id": int(perturbed.argmax())}

record_path = ROOT / "docs/course-experiments/results/efficiency.json"
record = json.loads(record_path.read_text())
tok = ByteTokenizer()
prompt = [tok.bos_id] + tok.encode("Once upon a time, a girl")
result["existing_experiment"] = {
    "path": record_path.relative_to(ROOT).as_posix(),
    "sha256": digest(record_path),
    "environment": {key: record[key] for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu"]},
    "runtime": record["results"]["runtime"],
    "prompt_ids_independently_encoded": prompt,
    "models": {},
    "inspection_scope": "核對保存的結果、配置及產生它的程式；本次沒有重跑 GPU 實驗。",
}
for name, expected_text in [("mha", "eueeuare"), ("gqa", "eredquare")]:
    entry = record["results"]["models"][name]
    cache = entry["cache"]
    full_ids, cached_ids = cache["generated_ids_full"], cache["generated_ids_cached"]
    errors = cache["per_step_logit_max_error"]
    summary = {
        "config": entry["model"]["config"],
        "parameters": entry["model"]["parameters"],
        "bytes_per_parameter": entry["model"]["parameter_bytes"] / entry["model"]["parameters"],
        "training_steps": entry["training"]["steps"],
        "prompt_tokens": cache["prompt_tokens"],
        "generated_tokens": cache["generated_tokens"],
        "generated_ids_full": full_ids,
        "generated_ids_cached": cached_ids,
        "equal_ids_independently_checked": full_ids == cached_ids,
        "generated_text_recorded": cache["generated_text"],
        "generated_text_independently_decoded": tok.decode(full_ids),
        "eos_id": tok.eos_id,
        "eos_count": full_ids.count(tok.eos_id),
        "per_step_logit_max_error": errors,
        "max_error_independently_reduced": max(errors),
        "all_steps_below_absolute_1e_5": all(value <= 1e-5 for value in errors),
    }
    assert cache["prompt_tokens"] == len(prompt) == 25
    assert len(full_ids) == len(cached_ids) == len(errors) == cache["generated_tokens"] == 12
    assert full_ids == cached_ids
    assert tok.decode(full_ids) == cache["generated_text"] == expected_text
    assert max(errors) == 1.9073486328125e-6
    assert summary["all_steps_below_absolute_1e_5"]
    result["existing_experiment"]["models"][name] = summary

result["repository_inputs"] = {
    name: digest(ROOT / name)
    for name in ["tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "tiny_perceptron/data.py", "scripts/course_experiments/architecture.py"]
}
assert result["baseline"]["allclose_atol_1e_6_default_rtol"]
assert result["new_last_id"]["allclose_atol_1e_6_default_rtol"]
assert not result["changed_reference_prefix"]["allclose_atol_1e_6_default_rtol"]
assert abs(result["changed_reference_prefix"]["max_error"] - 0.064) < 0.0005
assert all(item["k_max_error"] < 1e-6 and item["v_max_error"] < 1e-6 for item in result["append_invariance_each_layer"])
print(json.dumps(result, ensure_ascii=False, indent=2))
