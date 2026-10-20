import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import QuantizedLinear, pack_int4, quantize_symmetric, replace_linear_layers, unpack_int4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(42)
source_path = ROOT / "docs/course-review-20261019/freeze-01/sources/17.8.md"
source = source_path.read_text()
blocks = re.findall(r"```python\n(.*?)\n```", source, re.S)
library_path = ROOT / "tiny_perceptron/quantization.py"
library_tree = ast.parse(library_path.read_text())
excerpt_tree = ast.parse(blocks[1])
excerpt_equal = {}
for name in ("pack_int4", "unpack_int4"):
    original = next(node for node in library_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    excerpt = next(node for node in excerpt_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    excerpt_equal[name] = ast.dump(original, include_attributes=False) == ast.dump(excerpt, include_attributes=False)
    assert excerpt_equal[name], "節錄函式必須與函式庫相同"

q = torch.tensor([-8, -1, 0, 1, 7, 2, 3], dtype=torch.int8)
values = (q.to(torch.int16) + 8).to(torch.uint8)
padded = F.pad(values, (0, 1), value=8)
packed = pack_int4(q)
restored = unpack_int4(packed, q.shape)
assert packed.tolist() == [112, 152, 175, 139], "打包值必須符合逐組計算"
assert torch.equal(q, restored), "拆回的整數必須完全相同"
even = torch.cat((q, torch.tensor([4], dtype=torch.int8)))
even_packed = pack_int4(even)
assert even_packed.tolist() == [112, 152, 175, 203], "偶數個輸入的末 byte 必須使用真正的高碼"
assert torch.equal(unpack_int4(even_packed, even.shape), even), "偶數個輸入必須完整拆回"

codes = torch.arange(-8, 8, dtype=torch.int8)
pairs = torch.cartesian_prod(codes, codes)
pair_packed = pack_int4(pairs)
pair_expected = (pairs[:, 0].to(torch.int16) + 8) + 16 * (pairs[:, 1].to(torch.int16) + 8)
assert torch.equal(pair_packed.to(torch.int16), pair_expected), "所有有序碼對的位元與算術結果必須一致"
assert torch.equal(unpack_int4(pair_packed, pairs.shape), pairs), "所有有序碼對必須無損拆回"
assert pair_packed.unique().numel() == 256, "所有 byte 應恰好對應所有有序碼對"
odd_results = []
for value in codes.tolist():
    single = torch.tensor([value], dtype=torch.int8)
    result = pack_int4(single)
    assert result.item() >> 4 == 8, "奇數輸入補位必須表示零"
    assert torch.equal(unpack_int4(result, single.shape), single), "補位不能混入原始輸出"
    odd_results.append({"input": value, "packed": result.item()})
all_bytes = torch.arange(256, dtype=torch.int16).to(torch.uint8)
assert torch.equal(all_bytes & 15, all_bytes % 16), "遮罩必須取得餘數"
assert torch.equal(all_bytes >> 4, all_bytes // 16), "右移必須取得整數商"
matrix = pairs.T
assert torch.equal(unpack_int4(pack_int4(matrix), matrix.shape), matrix), "非連續的多軸輸入必須依原形狀拆回"
rejections = []
for value in (-9, 8):
    try:
        pack_int4(torch.tensor([value], dtype=torch.int8))
    except ValueError as error:
        rejections.append({"input": value, "error": str(error)})
    else:
        raise AssertionError("超出可打包範圍的整數必須拒絕")

namespace = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()) as capture:
    exec(blocks[0], namespace)
    imported = namespace["pack_int4"]
    exec(blocks[1], namespace)
    replaced = namespace["pack_int4"] is not imported
    exec(blocks[2], namespace)
assert replaced and namespace["same"], "依序執行必須真的核對節錄函式"
skipped = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()) as skip_capture:
    exec(blocks[0], skipped)
    imported_skip = skipped["pack_int4"]
    exec(blocks[2], skipped)
assert skipped["pack_int4"] is imported_skip and skipped["same"], "跳過函式定義時確實只會再呼叫函式庫版本"

weights = torch.tensor([-1.0, -0.2, 0.0, 0.2, 1.0], dtype=torch.float32)
quantized, scale = quantize_symmetric(weights, bits=4)
dequantized = quantized * scale
assert quantized.min().item() == -7 and quantized.max().item() == 7, "四位元對稱量化範圍必須排除負八"
assert not torch.equal(weights, dequantized), "浮點轉碼確實可能產生誤差"
layer = nn.Linear(7, 3)
quantized_layer = QuantizedLinear(layer, bits=4)
x = torch.randn(2, 7)
unpacked_weights = unpack_int4(quantized_layer.values, quantized_layer.shape)
reference = F.linear(x, unpacked_weights * quantized_layer.scale, quantized_layer.bias)
assert torch.equal(quantized_layer(x), reference), "參考 forward 必須使用拆碼與刻度恢復的權重"

record_path = ROOT / "docs/course-experiments/results/quantization.json"
record = json.loads(record_path.read_text())
storage = record["results"]["runs"]["packed4"]["storage"]
config = record["results"]["teacher_provenance"]["config"]
model = replace_linear_layers(TinyLM(ModelConfig(**config)).eval(), bits=4)
measured_buffers = {name: value.numel() * value.element_size() for name, value in model.named_buffers()}
assert measured_buffers == storage["buffers"], "同配置實際容器大小必須逐項等於實驗紀錄"
modules = {name: module for name, module in model.named_modules() if isinstance(module, QuantizedLinear)}
measured_parameters = sum(value.numel() * value.element_size() for value in model.parameters())
groups = {suffix: sum(size for name, size in measured_buffers.items() if name.endswith("." + suffix)) for suffix in ("values", "scale", "bias")}
assert list(modules) == storage["packed_linear_modules"] and len(modules) == 13, "量化層清單必須一致"
assert groups == {"values": 57600, "scale": 5664, "bias": 2560}, "各類 buffer 大小必須符合紀錄"
assert sum(groups.values()) == storage["buffer_tensor_bytes"] == 65824, "buffer 合計必須符合紀錄"
assert measured_parameters == storage["parameter_tensor_bytes"] == 102912, "保留浮點參數大小必須符合紀錄"
assert sum(groups.values()) + measured_parameters == storage["tensor_bytes"] == 168736, "tensor 合計必須符合紀錄"
module_details = {name: {"shape": module.shape, "bits": module.bits, "values_dtype": str(module.values.dtype), "scale_shape": list(module.scale.shape), "scale_dtype": str(module.scale.dtype), "bias_dtype": None if module.bias is None else str(module.bias.dtype)} for name, module in modules.items()}

report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-8/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads()), "seed": "42"},
    "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in (source_path, library_path, record_path, ROOT / "scripts/course_experiments/compression.py", ROOT / "tiny_perceptron/model.py")},
    "excerpt_ast_equal": excerpt_equal,
    "example": {"input": q.tolist(), "input_dtype": str(q.dtype), "input_bytes": q.numel() * q.element_size(), "flat_dtype": str(q.flatten().to(torch.int16).dtype), "shifted_codes": values.tolist(), "padded_codes": padded.tolist(), "low_codes": padded[0::2].tolist(), "high_codes": padded[1::2].tolist(), "packed": packed.tolist(), "packed_dtype": str(packed.dtype), "packed_bytes": packed.numel() * packed.element_size(), "unpacked_before_trim": (torch.stack((packed & 15, packed >> 4), -1).flatten().to(torch.int16) - 8).tolist(), "count": math.prod(q.shape), "restored": restored.tolist(), "low_nibbles": (packed & 15).tolist(), "high_nibbles": (packed >> 4).tolist()},
    "even_example": {"input": even.tolist(), "packed": even_packed.tolist(), "packed_bytes": even_packed.numel() * even_packed.element_size()},
    "exhaustive": {"ordered_pairs": len(pairs), "distinct_packed_bytes": pair_packed.unique().numel(), "odd_singletons": odd_results, "byte_mask_and_shift_checks": all_bytes.numel(), "noncontiguous_shape": list(matrix.shape), "all_passed": True},
    "range_rejections": rejections,
    "notebook_sequence": {"excerpt_replaces_import": replaced, "normal_stdout": capture.getvalue(), "skipping_excerpt_retains_import": skipped["pack_int4"] is imported_skip, "skipping_excerpt_comparison": skipped["same"]},
    "quantization": {"input": weights.tolist(), "bits": 4, "integers": quantized.tolist(), "scale": scale.item(), "dequantized": dequantized.tolist(), "max_abs_error": (weights - dequantized).abs().max().item()},
    "quantized_linear": {"shape": quantized_layer.shape, "bits": quantized_layer.bits, "buffers": {name: str(value.dtype) for name, value in quantized_layer.named_buffers()}, "forward_equals_explicit_dequantized_linear": True},
    "storage": {"original_experiment_device": record["device"], "original_experiment_seed": record["seed"], "config": config, "measured_on": "新建相同架構的 CPU FP32 模型後直接轉換；未執行訓練，bytes 不依賴權重內容。", "module_count": len(modules), "group_bytes": groups, "buffer_bytes": sum(groups.values()), "parameter_bytes": measured_parameters, "tensor_bytes": sum(groups.values()) + measured_parameters, "buffers_equal_record": measured_buffers == storage["buffers"], "module_details": module_details},
    "limitations": ["只核對凍結小節、明列輸入與範圍內有號整數；不宣稱任意 dtype 或錯誤 shape 均可接受。", "儲存量只算 tensor 的 numel × element_size，未計 Python 物件、allocator、序列化或執行時反量化暫存。", "未重跑原 GPU 訓練、品質、速度或 GPU 記憶體實驗；本節未以它們作結論。"],
    "result": "所有斷言通過",
}
print(json.dumps(report, ensure_ascii=False, indent=2))
