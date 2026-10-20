"""保存量化算例、既有量測的核對，以及原始來源的小份收據。"""

import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from tiny_perceptron.quantization import QuantizedLinear
import tiny_perceptron.quantization as quantization


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


sources = [
    {
        "id": "gptq",
        "url": "https://arxiv.org/html/2210.17323v2",
        "version": "arXiv:2210.17323v2，2023-03-22",
        "locator": "Practical Speedups；The Full Algorithm；Summary and Limitations",
        "excerpt": "a quantized-matrix full-precision-vector product kernel",
        "anchors": ["dynamically dequantizing weights when needed", "does not require any activation quantization", "almost all of the speedup is due to our kernels"],
        "inspection_note": "原文描述量化矩陣乘完整精度向量的專用 kernel，需要時才還原權重，以較少記憶體存取抵銷還原工作；不要求 activation 量化。加速示範的 batch、模型、GPU 與精度固定，不能當作所有 shape 或後端都加速的保證。另讀演算法的逆 Hessian 更新與誤差補償，與本專案單純 round/clamp 不同。",
    },
    {
        "id": "cuda_memory",
        "url": "https://docs.pytorch.org/docs/2.14/notes/cuda.html",
        "version": "PyTorch 2.14 CUDA semantics，與本機 2.14.1 同系列",
        "locator": "Memory management",
        "excerpt": "PyTorch uses a caching memory allocator to speed up memory allocations.",
        "anchors": ["memory occupied by tensors", "total amount of memory managed by the caching allocator"],
        "inspection_note": "實讀 Memory management：allocated / max allocated 計 tensor 占用，reserved / max reserved 計 caching allocator 管理總量；未使用的 cache 仍可能呈現在 nvidia-smi。這些計數不可混用成整體 GPU 記憶體。",
    },
    {
        "id": "peak_api",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html",
        "version": "PyTorch 2.14 API，對照實驗記錄 2.14.1+cu126",
        "locator": "torch.cuda.memory.max_memory_allocated API 說明",
        "excerpt": "reset_peak_memory_stats() can be used to reset the starting point in tracking this metric.",
        "anchors": ["Return the maximum GPU memory occupied by tensors in bytes"],
        "inspection_note": "實讀 API：回傳 tensor 占用的峰值 byte 數，reset_peak_memory_stats 可重設起點。因此迴圈內重設後的最外層讀數不是全實驗的不可逆總峰值。",
    },
    {
        "id": "allocated_api",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_allocated.html",
        "version": "PyTorch 2.14 API，對照實驗記錄 2.14.1+cu126",
        "locator": "torch.cuda.memory.memory_allocated API 的 Note",
        "excerpt": "some unused memory can be held by the caching allocator and some context needs to be created on GPU.",
        "anchors": ["Return the current GPU memory occupied by tensors in bytes"],
        "inspection_note": "實讀 Note：allocated 可能小於 nvidia-smi，原因含 allocator 未使用的 cache 以及 GPU context。支持正文不把 driver/context 等額外占用算入這組 tensor 計數的限制；不是在本機直接量 CUDA driver byte 數。",
    },
    {
        "id": "benchmark",
        "url": "https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html",
        "version": "PyTorch Tutorials 2.14.0+cu130；讀取日的線上版",
        "locator": "Benchmarking with Blocked Autorange；CUDA synchronization 的示範",
        "excerpt": "which also serves as a warm up",
        "anchors": ["PyTorch’s benchmark module does the synchronization for us", "taking measurements until the target time is reached"],
        "inspection_note": "實讀暖機、重複量測與 CUDA 同步的示範：未同步可能只量 launch 成本；暖機與後續多次測量分開。本頁所述是基本量測控制，幾次暖機或三次測量不保證統計上充分，也沒有由 byte 比例推算速度。",
    },
    {
        "id": "cutlass",
        "url": "https://raw.githubusercontent.com/NVIDIA/cutlass/v4.3.0/media/docs/cpp/functionality.md",
        "version": "NVIDIA CUTLASS v4.3.0",
        "locator": "Device-level functionality；CUTLASS 2.x Kernels 的 GEMM 型別與 compute capability 表",
        "excerpt": "s4 * s4 + s32 => {s32, s4}",
        "anchors": ["75+", "s8 * s8 + s32", "f16 * f16 + f32"],
        "inspection_note": "實讀官方功能表：整數 s4/s8 與混合精度 f16 累加 f32 路徑列出各自的架構、layout、toolkit 要求。只證明專用整數／混合精度 kernel 存在且受條件限制，不保證本教材的形狀會受益。",
    },
    {
        "id": "cuda_kernel",
        "url": "https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html",
        "version": "CUDA C++ Programming Guide 12.6.0 archived edition",
        "locator": "Programming Model / Kernels；VecAdd 範例；CUDA Runtime / Context",
        "excerpt": "C++ functions, called kernels",
        "anchors": ["executed N times in parallel", "VecAdd"],
        "inspection_note": "實讀 Kernels 定義與載入、運算範例：kernel 是在裝置 thread 上執行的計算函式；再讀 runtime 初始化 context、載入 device code 與記憶體管理的職責。支持正文給 kernel / driver 的簡化功能描述，不把每個 kernel 限定為一定有乘加。",
    },
]

for source in sources:
    argv = ["curl", "-fLsS", source["url"]]
    raw = subprocess.check_output(argv)
    text = raw.decode("utf-8")
    if "<html" in text[:500].lower() or "<!doctype" in text[:100].lower():
        parser = TextParser()
        parser.feed(text)
        text = " ".join(parser.parts)
    text = re.sub(r"\s+", " ", text)
    for anchor in [source["excerpt"], *source["anchors"]]:
        assert anchor in text, f"來源定位文字不存在：{source['id']}"
    source.update(
        command_argv=argv,
        accessed_on="2026-10-20",
        original_sha256=sha(raw),
        original_bytes=len(raw),
        curl_exit_code=0,
        checked_original=True,
    )
save("original-source-receipts.json", sources)

run = ROOT / "outputs/reviewer-tools/runs/20261020T064451501106Z"
copied = {}
for name in ("section.md", "fence-1.py", "bootstrap.py", "extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"):
    raw = (run / name).read_bytes()
    copied[name] = {"sha256": sha(raw), "utf8_content": raw.decode("utf-8")}
assert copied["section.md"]["sha256"] == "0aebe45b87c495f20ac4c07e7ad206dcebefecd78f20e0aaae6d46e5de2d34f7"
assert json.loads(copied["execution.json"]["utf8_content"])["exit_code"] == 0
save("section-execution.json", {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/17.md#17.10" --execute',
    "result": "exit_code=0；原始輸出逐字複製於 copied_files",
    "copied_files": copied,
})

torch.manual_seed(42)
torch.set_num_threads(1)
layer = nn.Linear(64, 32)
x = torch.ones(2, 64)
linear = quantization.F.linear
observed = []
for bits, expected_code, expected_total in ((4, 1024, 1280), (8, 2048, 2304)):
    compressed = QuantizedLinear(layer, bits)
    calls = []

    def record_linear(inputs, weight, bias):
        calls.append({
            "input_shape": list(inputs.shape),
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype),
            "weight_bytes": weight.numel() * weight.element_size(),
            "bias_dtype": str(bias.dtype),
        })
        return linear(inputs, weight, bias)

    before = compressed.values.clone()
    with patch.object(quantization.F, "linear", record_linear):
        y = compressed(x)
        again = compressed(x)
    buffers = {name: {"shape": list(b.shape), "dtype": str(b.dtype), "bytes": b.numel() * b.element_size()} for name, b in compressed.named_buffers()}
    assert buffers["values"]["bytes"] == expected_code
    assert buffers["scale"]["bytes"] == buffers["bias"]["bytes"] == 128
    assert compressed.storage_bytes() == expected_total
    assert list(y.shape) == [2, 32] and y.dtype == torch.float32
    assert all(c["weight_bytes"] == 8192 and c["weight_dtype"] == "torch.float32" for c in calls)
    assert torch.equal(before, compressed.values) and torch.equal(y, again)
    observed.append({"bits": bits, "buffers": buffers, "storage_bytes": compressed.storage_bytes(), "output_shape": list(y.shape), "output_dtype": str(y.dtype), "linear_calls": calls})
assert sum(p.numel() * p.element_size() for p in layer.parameters()) == 8320

experiment_path = "docs/course-experiments/results/quantization.json"
experiment = json.loads((ROOT / experiment_path).read_bytes())
history = {}
for path, commit, functions in [
    ("scripts/course_experiments/compression.py", "dc8d1ac3cec62d33ad973e54ba7eaf44732d8c17", ["_sync", "_packed", "_timing", "run_quantization"]),
    ("scripts/course_experiments/run.py", "a5133f444e0dca2347635f74f8ac83a27bb203d4", []),
]:
    argv = ["git", "show", f"{commit}:{path}"]
    raw = subprocess.check_output(argv, cwd=ROOT)
    assert sha(raw) == experiment["code_sha256"][path]
    old_tree, current_tree = ast.parse(raw), ast.parse((ROOT / path).read_bytes())
    comparisons = {}
    for function in functions:
        old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        current = next(n for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        comparisons[function] = ast.dump(old) == ast.dump(current)
        assert comparisons[function]
    history[path] = {"command_argv": argv, "sha256": sha(raw), "matches_experiment_code_sha256": True, "current_function_ast_matches": comparisons}
    if path.endswith("compression.py"):
        timing_node = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_timing")
    else:
        decoded = raw.decode("utf-8")
        assert 'torch.cuda.max_memory_allocated() if device == "cuda" else None' in decoded
        assert '"Branch measurements can reset this counter; this is not necessarily the full-experiment peak. "' in decoded

class Probe:
    config = SimpleNamespace(max_length=128)

    def __init__(self):
        self.lengths = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, ids):
        self.lengths.append(ids.shape[1])
        logits = torch.zeros(1, ids.shape[1], 3)
        logits[..., 2] = 1
        return {"logits": logits}

namespace = {"torch": torch, "time": time, "_sync": lambda device: None, "_prompt": lambda record, length: torch.ones(46, dtype=torch.long)}
exec(compile(ast.Module(body=[timing_node], type_ignores=[]), "saved_timing_function", "exec"), namespace)
probe = Probe()
probe_timing = namespace["_timing"](probe, {}, "cpu")
expected_lengths = [46] + ([46] + list(range(47, 55))) * 3
assert probe.lengths == expected_lengths and probe.eval_called
assert probe_timing["repetitions"] == 3 and probe_timing["decode_tokens"] == 8

summary = {}
for name, expected_prefill, expected_decode, expected_peak in [
    ("fp32", "2.405", "2.371", 335872),
    ("packed4", "4.958", "4.882", 376320),
    ("packed8", "2.768", "2.814", 359936),
]:
    timing = experiment["results"]["runs"][name]["timing"]
    prefill = f"{timing['prefill_seconds'] * 1000:.3f}"
    decode = f"{timing['decode_seconds_per_token'] * 1000:.3f}"
    additional = timing["cuda_peak_allocated"] - timing["cuda_allocated_before"]
    assert prefill == expected_prefill and decode == expected_decode
    assert additional == timing["cuda_additional_peak_bytes"] == expected_peak
    assert timing["cuda_allocated_before"] == 68655616
    assert timing["repetitions"] == 3 and timing["prompt_tokens"] == 46 and timing["decode_tokens"] == 8
    assert abs(timing["decode_seconds"] / 8 - timing["decode_seconds_per_token"]) < 1e-15
    summary[name] = {"recorded_timing": timing, "prefill_milliseconds_rounded": prefill, "decode_milliseconds_per_step_rounded": decode, "peak_subtraction": additional}
assert experiment["peak_allocated_bytes"] == summary["packed8"]["recorded_timing"]["cuda_peak_allocated"]
assert summary["packed4"]["recorded_timing"]["prefill_seconds"] > summary["fp32"]["recorded_timing"]["prefill_seconds"]
assert summary["packed4"]["recorded_timing"]["decode_seconds_per_token"] > summary["fp32"]["recorded_timing"]["decode_seconds_per_token"]

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-10/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "status": "全部斷言通過",
    "quantization": {"original_parameter_bytes": 8320, "weight_elements": layer.weight.numel(), "bias_elements": layer.bias.numel(), "variants": observed},
    "experiment_source": {"path": experiment_path, "sha256": sha((ROOT / experiment_path).read_bytes()), "recorded_revision": experiment["revision"], "recorded_environment": {k: experiment[k] for k in ("python_version", "torch_version", "gpu", "device", "seed")}, "model_config": experiment["results"]["teacher_provenance"]["config"], "timing_summary": summary, "peak_allocated_bytes": experiment["peak_allocated_bytes"], "peak_memory_scope": experiment["peak_memory_scope"]},
    "historical_source_checks": history,
    "timing_control_probe": {"device": "cpu", "purpose": "只核對呼叫次數、輸入長度與固定步數；不作速度比較", "greedy_token_id": 2, "actual_forward_lengths": probe.lengths, "warmup_forward_calls": 1, "measured_repetitions": 3, "decode_steps_per_repetition": 8, "result": "符合；即使每步 argmax 都產生 EOS 的 token ID，仍跑完固定步數"},
    "limitations": "未執行 NVIDIA L4 或隔離載入記憶體實驗。既有實測的數值、單位、分母及程式控制可核對；不獨立重現硬體計時，不建立其他後端的速度或 RAM 保證。",
}
save("verification.json", result)
print(json.dumps({"status": result["status"], "original_parameter_bytes": 8320, "variant_bytes": [v["storage_bytes"] for v in observed], "source_receipts": len(sources), "timing_control_forward_calls": len(probe.lengths)}, ensure_ascii=False))
