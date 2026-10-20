"""核對凍結算例、已保存的量測紀錄與官方來源；不啟動GPU工作。"""

import hashlib
import importlib
import json
import math
import platform
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
PIN = "5c4886908584029761b579af026dcfb627c84070"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def fetch(url):
    argv = ["curl", "-fLsS", "--max-time", "40", url]
    result = subprocess.run(argv, capture_output=True, check=True)
    return result.stdout, {
        "command_argv": argv,
        "exit_code": result.returncode,
        "url": url,
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "bytes": len(result.stdout),
        "sha256": digest(result.stdout),
        "full_text_saved": False,
    }


def main():
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-9/audit.py",
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "device": "cpu",
            "platform": platform.platform(),
            "cuda_available": str(torch.cuda.is_available()),
        },
        "scope": "CPU算例與保存紀錄的再計算、程式閱讀及官方原文查核；未重跑CUDA實驗。",
    }
    local_paths = [
        "docs/course-review-20261019/freeze-01/sources/16.9.md",
        "docs/course-review-20261019/freeze-04.json",
        "docs/course-experiments/results/flash_probe.json",
        "docs/course-experiments/results/efficiency.json",
        "scripts/course_experiments/architecture.py",
        "scripts/course_experiments/run.py",
        "docs/course-experiments/plan.json",
        "course/training.md",
        "course/figures/flash_allocated_memory.svg",
        "course/figures/rewrite-16-online-softmax.svg",
        "docs/course-review-20261019/figure-renders-04.json",
    ]
    result["local_inputs"] = {
        path: digest((ROOT / path).read_bytes()) for path in local_paths
    }
    section = (ROOT / local_paths[0]).read_bytes()
    chapter = (ROOT / "course/chapters/16.md").read_bytes()
    current = chapter[chapter.index(b"## 16.9 ") : chapter.index(b"## 16.10 ")]
    assert section == current
    result["frozen_source_matches_current_section"] = True
    result["square_cells"] = {str(t): t * t for t in (4, 8, 8192)}
    result["log_values"] = {str(x): math.log(x) for x in (1, 2, 4)}
    result["inverse_examples"] = {str(x): math.exp(math.log(x)) for x in (1, 2, 4)}
    result["direct_weighted_average"] = (10 + 2 * 20 + 4 * 30) / 7
    result["shifted_weights"] = [math.exp(math.log(x) - math.log(4)) for x in (1, 2, 4)]
    scores = torch.tensor([0.0, math.log(2), math.log(4)])
    values = torch.tensor([10.0, 20.0, 30.0])
    result["chunk_runs"] = {}
    for size in (1, 2, 3):
        m, den, num = torch.tensor(-math.inf), torch.tensor(0.0), torch.tensor(0.0)
        rows = []
        for start in range(0, 3, size):
            block = scores[start : start + size]
            new_m = torch.maximum(m, block.max())
            factor = torch.exp(m - new_m)
            weight = torch.exp(block - new_m)
            den = den * factor + weight.sum()
            num = num * factor + (weight * values[start : start + size]).sum()
            m = new_m
            rows.append({"start": start, "block_length": len(block), "factor": factor.item(), "denominator": den.item(), "numerator": num.item()})
        actual, reference = (num / den).item(), (scores.softmax(0) * values).sum().item()
        assert abs(actual - reference) < 1e-5
        assert round(actual, 4) == 24.2857
        result["chunk_runs"][str(size)] = {"rows": rows, "result": actual, "full_softmax_reference": reference}

    sys.path.insert(0, str(ROOT))
    architecture = importlib.import_module("scripts.course_experiments.architecture")
    inputs = tuple(torch.randn(1, 1, 3, 2).requires_grad_() for _ in range(3))
    upstream = torch.ones_like(inputs[0])
    grad_checks = {}
    for backward in (False, True):
        observed_grad_modes = []

        def observed_manual(*items):
            observed_grad_modes.append(torch.is_grad_enabled())
            return architecture._flash_manual_attention(*items)

        measured = architecture._flash_measure(observed_manual, inputs, upstream, "cpu", backward)
        assert measured["status"] == "completed"
        assert measured["warmup_calls"] == 3 and measured["measured_calls"] == 9
        assert observed_grad_modes == [backward] * 12
        assert measured["peak_allocated_bytes"] is None
        grad_checks[str(backward)] = {"warmup_calls": measured["warmup_calls"], "measured_calls": measured["measured_calls"], "observed_grad_modes": observed_grad_modes, "status": measured["status"], "memory_scope": measured["memory_scope"]}
    result["measurement_control_flow_cpu_check"] = grad_checks

    flash = read_json("docs/course-experiments/results/flash_probe.json")
    recorded = flash["results"]
    assert flash["code_sha256"]["scripts/course_experiments/architecture.py"] == result["local_inputs"]["scripts/course_experiments/architecture.py"]
    assert recorded["configuration"]["shape_B_H_T_D"] == [2, 4, 512, 32]
    assert recorded["configuration"]["is_causal"] is True
    assert recorded["configuration"]["dropout_p"] == 0
    expected_ms = {
        ("fp16", "forward"): (0.513, 0.076),
        ("fp16", "forward_backward"): (1.689, 0.503),
        ("bf16", "forward"): (0.516, 0.071),
        ("bf16", "forward_backward"): (1.502, 0.458),
    }
    summary = {
        "revision": flash["revision"],
        "runtime": {key: recorded["runtime"][key] for key in ("torch", "torch_git_version", "cuda_runtime_version", "gpu", "compute_capability", "driver_query", "flash_attention_built")},
        "configuration": recorded["configuration"],
        "code_matches_recorded_sha256": True,
        "measurement_rows": [],
        "backend_and_errors": {},
    }
    for dtype in ("fp16", "bf16"):
        route = recorded["routes"][dtype]
        summary["backend_and_errors"][dtype] = {"status": route["status"], "correctness": route["correctness"], "profiles": {}}
        for mode in ("forward", "forward_backward"):
            profile = route["profiles"][mode]
            assert profile["flash_cuda_verified"]
            assert profile["flash_forward_operator_observed"]
            assert profile["cuda_kernel_names"]
            if mode == "forward_backward":
                assert profile["flash_backward_operator_observed"]
            summary["backend_and_errors"][dtype]["profiles"][mode] = {
                key: profile[key] for key in ("includes_backward", "flash_forward_operator_observed", "flash_backward_operator_observed", "flash_cuda_verified")
            }
            summary["backend_and_errors"][dtype]["profiles"][mode]["cuda_kernel_count"] = len(profile["cuda_kernel_names"])
            for index, backend in enumerate(("manual", "forced_flash")):
                item = route["measurements"][mode][backend]
                median = statistics.median(item["samples_seconds"])
                assert median == item["median_seconds"]
                assert item["warmup_calls"] == 3 and item["measured_calls"] == len(item["samples_seconds"]) == 9
                assert item["synchronized"]
                assert round(1000 * median, 3) == expected_ms[dtype, mode][index]
                before, peak, extra = (item[key] for key in ("allocated_before_bytes", "peak_allocated_bytes", "additional_peak_allocated_bytes"))
                assert peak - before == extra
                assert before / 2**20 == 65
                expected_extra = (24.25, 0.267) if mode == "forward" else (32.75, 2.032)
                assert round(extra / 2**20, 3) == expected_extra[index]
                summary["measurement_rows"].append({"dtype": dtype, "mode": mode, "backend": backend, "median_recomputed_ms": median * 1000, "sample_count": len(item["samples_seconds"]), "warmup_calls": item["warmup_calls"], "baseline_MiB": before / 2**20, "additional_peak_MiB": extra / 2**20, "total_peak_MiB": peak / 2**20})
    result["archived_flash_audit"] = summary
    efficiency = read_json("docs/course-experiments/results/efficiency.json")
    sdpa = efficiency["results"]["manual_vs_sdpa"]
    assert sdpa["backend"]["selected"] == "memory_efficient"
    assert sdpa["backend"]["flash_attention_observed"] is False
    medians = {key: statistics.median(value["samples_seconds"]) * 1000 for key, value in sdpa["timings"].items()}
    result["previous_model_measurement"] = {"revision": efficiency["revision"], "backend": sdpa["backend"], "median_ms_recomputed": medians, "difference_ms": medians["manual"] - medians["sdpa"]}
    assert round(medians["manual"] - medians["sdpa"], 3) == 0.644

    svg = ET.parse(ROOT / "course/figures/flash_allocated_memory.svg")
    rectangles = svg.findall("{http://www.w3.org/2000/svg}rect")[1:]
    expected_values = [65, 24.25, 65, 0.267, 65, 32.75, 65, 2.032]
    scales = [float(rect.attrib["width"]) / value for rect, value in zip(rectangles, expected_values, strict=True)]
    assert all(abs(scale - 4.4) < 1e-12 for scale in scales)
    result["memory_figure_pixels_per_MiB"] = scales
    result["MiB_bytes"] = 1024 * 1024
    plan = read_json("docs/course-experiments/plan.json")
    spec = next(item for item in plan["sequence"] + plan.get("supporting_experiments", []) if item["id"] == "flash_probe")
    assert spec["module"] == "architecture" and spec["function"] == "run_flash_probe"
    argv = [sys.executable, "-B", "-m", "scripts.course_experiments.run", "--list-assets", "flash_probe"]
    listing = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
    result["reproduction_entrypoint"] = {"spec": spec, "read_only_command_argv": argv, "exit_code": listing.returncode, "stdout": listing.stdout, "stderr": listing.stderr, "scope": "只驗證命令入口與註冊，不執行CUDA探針。"}

    urls = {
        "flashattention_paper": "https://arxiv.org/pdf/2205.14135v2",
        "online_softmax_paper": "https://arxiv.org/pdf/1805.02867v2",
        "pytorch_selector": f"https://raw.githubusercontent.com/pytorch/pytorch/{PIN}/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp",
        "pytorch_sdpa": f"https://raw.githubusercontent.com/pytorch/pytorch/{PIN}/torch/nn/functional.py",
        "pytorch_memory": f"https://raw.githubusercontent.com/pytorch/pytorch/{PIN}/torch/cuda/memory.py",
        "nvidia_capability": "https://developer.nvidia.com/cuda/gpus.md",
    }
    readings = {}
    for name, url in urls.items():
        raw, receipt = fetch(url)
        if name == "flashattention_paper":
            assert raw.startswith(b"%PDF")
            receipt.update(version="arXiv:2205.14135v2, 23 Jun 2022", inspection="另以web PDF解析閱讀GPU Memory Hierarchy、Standard Attention Implementation、An Efficient Attention Algorithm With Tiling and Recomputation、Algorithm 1、Theorem 1、Theorem 2；下載全文只在記憶體中取雜湊。", excerpt="Algorithm 1 returns O = softmax(QKᵀ)V")
        elif name == "online_softmax_paper":
            assert raw.startswith(b"%PDF")
            receipt.update(version="arXiv:1805.02867v2, 28 Jul 2018", inspection="另以web PDF解析閱讀Original softmax與Online normalizer calculation、Algorithm 3、Theorem 1、Parallel online normalizer calculation；online指依序處理輸入並維護統計。", excerpt='In the remainder of this paper we will call algorithm 3 "Online Softmax".')
        else:
            text = raw.decode("utf-8")
            if name == "pytorch_selector":
                assert "using sm80 = SMVersion<8, 0>;" in text
                assert "check_dtypes_flash_attention" in text
                assert "{at::kHalf, at::kBFloat16}" in text
                receipt.update(version=PIN, inspection="已讀check_flash_attention_hardware_support、check_dtypes_low_precision、check_dtypes_flash_attention及can_use_flash_attention的build、mask、shape、dtype限制。", excerpt="return check_dtypes_low_precision(params, debug);")
            elif name == "pytorch_sdpa":
                assert "attn_weight = torch.softmax(attn_weight, dim=-1)" in text
                assert "Due to the nature of fusing floating point operations" in text
                receipt.update(version=PIN, inspection="已讀scaled_dot_product_attention的縮放、causal mask、softmax、PV參考計算，以及後端選擇與浮點差異說明。", excerpt="attn_weight = torch.softmax(attn_weight, dim=-1)")
            elif name == "pytorch_memory":
                assert 'get("allocated_bytes.all.peak", 0)' in text
                assert 'get("allocated_bytes.all.current", 0)' in text
                receipt.update(version=PIN, inspection="已讀memory_allocated、max_memory_allocated、memory_reserved及reset_peak_memory_stats；allocated統計與保留量、nvidia-smi不同。", excerpt='return memory_stats(device=device).get("allocated_bytes.all.peak", 0)')
            else:
                row = next(line for line in text.splitlines() if line.startswith("| 8.9 |"))
                assert "NVIDIA L4" in row
                receipt.update(version="2026-10-20取得的官方Markdown表", inspection="已讀compute capability定義與8.9資料列；L4列於8.9。", excerpt="| 8.9 | NVIDIA L4 NVIDIA L40 NVIDIA L40S |")
        readings[name] = receipt
    result["primary_readings"] = readings
    result["result"] = "所有數值、保存紀錄與官方程式條件核對通過。"
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
