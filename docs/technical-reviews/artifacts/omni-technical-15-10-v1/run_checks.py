"""15.10 的有界 CPU 核對；直接執行凍結教材程式，另獨立重算模型參數。"""

import contextlib
import datetime
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.architecture import _parameter_budget
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_inline(code, expected):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "15.10-frozen-inline", "exec"), {})
    lines = out.getvalue().splitlines()
    assert lines == expected, (lines, expected)
    return {
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "stdout": out.getvalue(),
        "exact_expected_lines": expected,
        "match": True,
    }


def main():
    torch.set_num_threads(1)
    freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
    page = next(p for p in json.loads(freeze_path.read_bytes())["pages"] if p["page_id"] == "15.10")
    snapshot = ROOT / page["snapshot"]
    raw = snapshot.read_text()
    assert sha(snapshot) == page["source_sha256"]
    current = re.search(r"^## 15\.10 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S)[0]
    assert current == raw
    blocks = re.findall(r"```python\n(.*?)```", raw, re.S)
    assert len(blocks) == 1
    code = blocks[0]
    assert code.startswith("width, hidden, k = 4, 8, 2\n")
    baseline = run_inline(code, [
        "expert數 4 總權重 272 每token使用估計 144 FP32權重bytes 1088",
        "expert數 8 總權重 544 每token使用估計 160 FP32權重bytes 2176",
    ])
    exercise_code = code.replace("width, hidden, k = 4, 8, 2\n", "width, hidden, k = 4, 8, 1\n", 1)
    exercise = run_inline(exercise_code, [
        "expert數 4 總權重 272 每token使用估計 80 FP32權重bytes 1088",
        "expert數 8 總權重 544 每token使用估計 96 FP32權重bytes 2176",
    ])

    # Optional-text model counts: direct named-parameter classification first,
    # then compare with both the existing record and the helper implementation.
    record_path = ROOT / "docs/course-experiments/results/moe.json"
    record = json.loads(record_path.read_bytes())
    counts = []
    for name in ("top1_aux0", "top1_aux0.01", "top2_aux0", "top2_aux0.01"):
        variant = record["results"]["variants"][name]
        config = ModelConfig(**variant["model"]["config"])
        torch.manual_seed(42)
        model = TinyLM(config).eval()
        total = sum(p.numel() for p in model.parameters())
        experts = sum(p.numel() for n, p in model.named_parameters() if ".ffn.experts." in n)
        router = sum(p.numel() for n, p in model.named_parameters() if ".ffn.router." in n)
        other = total - experts - router
        direct = {
            "total_parameters": total,
            "expert_parameters": experts,
            "router_parameters": router,
            "active_parameter_proxy_per_token": other + router + experts * config.top_k // config.experts,
        }
        assert direct == variant["budget"] == _parameter_budget(model)
        assert (total, experts, router, other) == (340608, 264704, 512, 75392)
        one_expert = sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters())
        assert one_expert == 64 * 256 + 256 + 256 * 64 + 64 == 33088
        assert total == variant["model"]["parameters"]

        # One token, no padding or masking: count selected expert slots and retain
        # all registered parameters. This is a dispatch check, never a speed test.
        chosen = []
        hooks = [block.ffn.register_forward_hook(lambda module, args, output: chosen.append(output[2].tolist()))
                 for block in model.blocks]
        with torch.no_grad():
            ids = torch.tensor([[1]], dtype=torch.long)
            output = model(ids)
            embedding = model.embedding(ids)
            position = model.position(torch.tensor([0]))
        for hook in hooks:
            hook.remove()
        assert all(len(layer) == 1 and len(layer[0]) == config.top_k for layer in chosen)
        assert sum(p.numel() for p in model.parameters()) == total
        assert tuple(output["logits"].shape) == (1, 1, 264)
        assert torch.isfinite(output["logits"]).all()
        assert embedding.numel() == position.numel() == 64
        assert model.embedding.weight.numel() == 16896
        assert model.position.weight.numel() == 8192
        counts.append({
            "variant": name,
            "config": variant["model"]["config"],
            "direct_budget": direct,
            "other_parameters": other,
            "one_expert_including_bias": one_expert,
            "matches_record_and_helper": True,
            "single_token_input_ids": [[1]],
            "input_dtype": str(ids.dtype),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "valid_or_segments": "None; no padding or packing",
            "selected_experts_per_layer": chosen,
            "parameters_after_forward": total,
            "logits_shape": list(output["logits"].shape),
            "logits_finite": True,
            "embedding_lookup_elements": embedding.numel(),
            "position_lookup_elements": position.numel(),
            "embedding_full_table_elements": model.embedding.weight.numel(),
            "position_full_table_elements": model.position.weight.numel(),
            "kv_cache_shapes": [[list(k.shape), list(v.shape)] for k, v in output["cache"]],
            "recorded_training_steps_not_rerun": variant["training"]["steps"],
        })

    element_size = torch.empty(0, dtype=torch.float32).element_size()
    assert element_size == 4
    # A single scalar update checks the existence of gradient and two Adam state
    # tensors; it does not train or evaluate a language model.
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    ((parameter - 3).square().sum()).backward()
    gradient = parameter.grad.tolist()
    assert gradient == [-4.0]
    optimizer.step()
    states = {name: {"shape": list(value.shape), "numel": value.numel(), "dtype": str(value.dtype)}
              for name, value in optimizer.state[parameter].items()}
    assert states["exp_avg"]["numel"] == states["exp_avg_sq"]["numel"] == 1

    dependencies = ["tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/attention.py",
                    "scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py"]
    dependency_receipts = [{"path": path, "sha256": sha(ROOT / path),
                            "recorded_run_sha256": record["code_sha256"].get(path),
                            "same_as_recorded_run": sha(ROOT / path) == record["code_sha256"].get(path)}
                           for path in dependencies]
    result = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-15-10-v1/run_checks.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_git_commit": torch.version.git_version, "device": "cpu",
                        "threads": str(torch.get_num_threads())},
        "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path)},
        "snapshot": {"path": page["snapshot"], "sha256": sha(snapshot), "matches_current": True},
        "figure_map": page["figures_sha256"],
        "inline_code_baseline": baseline,
        "inline_code_exercise_k1": exercise,
        "existing_experiment": {"path": str(record_path.relative_to(ROOT)), "sha256": sha(record_path),
                                "recorded_revision": record["revision"], "status": record["status"],
                                "evidence_status": record["evidence_status"],
                                "all_requested_updates_completed": record["results"]["all_requested_updates_completed"]},
        "model_counts": counts,
        "fp32_element_bytes": element_size,
        "scalar_adam_check": {"gradient": gradient, "state_tensors": states},
        "code_dependencies": dependency_receipts,
        "result": "PASS: exact inline outputs, k=1 exercise, independent model parameter sums, proxy scope, FP32 storage, gradient/Adam state and one-token dispatch/cache checks.",
        "scope": "CPU count/one-token/scalar mechanisms only. No dataset download, language-model training, GPU, throughput, peak-memory or model-quality measurement.",
    }
    target = Path(__file__).with_name("execution.json")
    with target.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": sha(target),
                      "result": result["result"], "dependencies": dependency_receipts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
