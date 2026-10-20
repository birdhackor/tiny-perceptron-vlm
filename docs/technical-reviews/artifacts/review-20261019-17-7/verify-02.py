"""核對固定算例、模組狀態、實驗儲存量與原始 PyTorch 契約。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

import tiny_perceptron.quantization as quant
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (ARTIFACTS / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def inspect_originals():
    base = "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/"
    selections = {
        "module": (
            "torch/nn/modules/module.py",
            [
                "Buffers, by\n        default, are persistent and will be saved alongside parameters.",
                'if "_buffers" not in self.__dict__:',
                'self._buffers[name] = tensor',
                'Move and/or cast the parameters and buffers.',
                'for _name, param in self.named_parameters(recurse=recurse):\n            yield param',
                'lambda module: module._parameters.items(),',
                'forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)',
                'return forward_call(*args, **kwargs)',
            ],
        ),
        "weight_only": (
            "torch/ao/quantization/fx/README.md",
            [
                "Weight only quantization:\n```\n                                       input - float_op - output\n      weight - quantize_per_tensor - dequantize /\n```",
            ],
        ),
        "linear": (
            "torch/nn/modules/linear.py",
            [
                'Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.',
                'return F.linear(input, self.weight, self.bias)',
            ],
        ),
        "detach": (
            "torch/_tensor.py",
            [
                'Returns a new Tensor, detached from the current graph.',
                'The result will never require gradient.',
                'Returned Tensor shares the same storage with the original one.',
            ],
        ),
    }
    receipt = []
    for identifier, (relative, snippets) in selections.items():
        url = base + relative
        command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
        result = subprocess.run(command, capture_output=True, check=True)
        raw = result.stdout
        text = raw.decode("utf-8")
        excerpts = []
        for snippet in snippets:
            position = text.index(snippet)
            excerpts.append({"line": text[:position].count("\n") + 1, "text": snippet})
        receipt.append({
            "id": identifier,
            "url": url,
            "version": "v2.14.1",
            "accessed_at": datetime.now(UTC).isoformat(),
            "command_argv": command,
            "exit_code": result.returncode,
            "stderr": result.stderr.decode("utf-8"),
            "original_bytes": len(raw),
            "original_sha256": sha(raw),
            "excerpts": excerpts,
        })
    save("source-excerpts.json", receipt)
    return {item["id"]: item["original_sha256"] for item in receipt}


def run_fences(fences, bits, omit_excerpt=False):
    namespace = {"__name__": "__review__"}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(fences[0].replace("bits = 4", f"bits = {bits}"), "frozen-first-fence", "exec"), namespace)
        library_class = namespace["QuantizedLinear"]
        if not omit_excerpt:
            exec(compile(fences[1], "frozen-class-excerpt", "exec"), namespace)
        exec(compile(fences[2], "frozen-last-fence", "exec"), namespace)
    layer, compressed = namespace["layer"], namespace["compressed"]
    codes, scale = quant.quantize_symmetric(layer.weight.detach(), bits, per_channel=True)
    result = {
        "bits": bits,
        "excerpt_executed": not omit_excerpt,
        "stdout": stream.getvalue(),
        "compressed_class_is_library": type(compressed) is library_class,
        "rebuilt_class_is_library": type(namespace["rebuilt"]) is library_class,
        "same_output": torch.equal(namespace["rebuilt"](namespace["x"]), compressed(namespace["x"])),
        "codes": codes.tolist(),
        "scales": scale.flatten().tolist(),
        "restored_weight": (codes * scale).tolist(),
        "output": compressed(namespace["x"]).tolist(),
        "input_dtype": str(namespace["x"].dtype),
        "output_dtype": str(compressed(namespace["x"]).dtype),
        "values_dtype": str(compressed.values.dtype),
        "shape": compressed.shape,
        "buffers": {name: {"shape": list(t.shape), "dtype": str(t.dtype), "bytes": t.numel() * t.element_size()} for name, t in compressed.named_buffers()},
        "storage_bytes": compressed.storage_bytes(),
    }
    expected_codes = [[7, 2], [-4, 7]] if bits == 4 else [[127, 36], [-74, 127]]
    assert result["codes"] == expected_codes
    assert result["storage_bytes"] == (10 if bits == 4 else 12)
    assert result["same_output"]
    assert result["rebuilt_class_is_library"] == omit_excerpt
    assert torch.allclose(layer(namespace["x"]), torch.tensor([[0.9, 0.5]]), atol=1e-6, rtol=0)
    if bits == 4:
        assert torch.allclose(compressed(namespace["x"]), torch.tensor([[0.9, 0.5142857142857142]]), atol=1e-6, rtol=0)
    return namespace, result


def module_behavior(namespace):
    original, packed = namespace["layer"], namespace["compressed"]
    detached = original.weight.detach()
    assert not detached.requires_grad and detached.grad_fn is None
    assert detached.data_ptr() == original.weight.data_ptr()
    assert not list(packed.parameters())
    assert sorted(packed.state_dict()) == ["scale", "values"]
    state = io.BytesIO()
    torch.save(packed.state_dict(), state)
    state.seek(0)
    restored = torch.load(state, weights_only=True)
    assert all(torch.equal(restored[name], value) for name, value in packed.state_dict().items())
    meta = copy.deepcopy(packed).to("meta")
    assert all(t.device.type == "meta" for t in meta.buffers())
    old_optimizer = torch.optim.SGD(original.parameters(), lr=0.1)
    before = {name: value.clone() for name, value in packed.named_buffers()}
    packed(torch.ones(1, 2, requires_grad=True)).sum().backward()
    assert original.weight.grad is None
    old_optimizer.step()
    assert all(torch.equal(before[name], value) for name, value in packed.named_buffers())
    weights = []
    functional_linear = quant.F.linear

    def capture(x, weight, bias):
        weights.append(weight)
        return functional_linear(x, weight, bias)

    with patch.object(quant.F, "linear", capture):
        packed(torch.ones(1, 2))
        packed(torch.ones(1, 2))
    assert weights[0].data_ptr() != weights[1].data_ptr()
    biased = nn.Linear(2, 2)
    with torch.no_grad():
        biased.weight.copy_(original.weight)
        biased.bias.copy_(torch.tensor([0.25, -0.125]))
    converted = quant.QuantizedLinear(biased)
    assert converted.bias.dtype == torch.float32 and not converted.bias.requires_grad
    assert converted.bias.data_ptr() != biased.bias.data_ptr()
    x = torch.tensor([[1.0, 1.0]])
    q, scale = quant.quantize_symmetric(biased.weight.detach(), 4, True)
    expected = x @ (q * scale).T + biased.bias.detach()
    assert torch.equal(converted(x), expected)
    return {
        "detached_requires_grad": detached.requires_grad,
        "detached_grad_fn": str(detached.grad_fn),
        "detached_shares_storage": detached.data_ptr() == original.weight.data_ptr(),
        "parameter_count": sum(p.numel() for p in packed.parameters()),
        "persistent_state_keys": sorted(packed.state_dict()),
        "state_roundtrip_equal": True,
        "moved_buffer_devices": {name: str(value.device) for name, value in meta.named_buffers()},
        "original_weight_grad_after_packed_backward": str(original.weight.grad),
        "old_optimizer_left_buffers_unchanged": True,
        "distinct_live_forward_weight_tensors": weights[0].data_ptr() != weights[1].data_ptr(),
        "forward_weight_dtypes": [str(value.dtype) for value in weights],
        "biased_output": converted(x).tolist(),
        "bias_is_cloned_float_buffer": True,
        "biased_output_matches_affine_formula": True,
    }


def full_model_storage():
    record_path = ROOT / "docs/course-experiments/results/quantization.json"
    record = json.loads(record_path.read_text())
    config = record["results"]["teacher_provenance"]["config"]
    recorded = record["results"]["runs"]["packed4"]["storage"]
    model = quant.replace_linear_layers(TinyLM(ModelConfig(**config)), bits=4)
    parameters = {name: {"numel": p.numel(), "dtype": str(p.dtype), "bytes": p.numel() * p.element_size()} for name, p in model.named_parameters()}
    buffers = {name: value.numel() * value.element_size() for name, value in model.named_buffers()}
    retained = [name for name, module in model.named_modules() if not isinstance(module, quant.QuantizedLinear) and any(True for _ in module.parameters(recurse=False))]
    total = sum(value["bytes"] for value in parameters.values())
    assert total == recorded["parameter_tensor_bytes"] == 102912
    assert buffers == recorded["buffers"]
    assert retained == recorded["retained_float_modules"]
    assert all(p.dtype == torch.float32 for p in model.parameters())
    biases = {name: str(value.dtype) for name, value in model.named_buffers() if name.endswith(".bias")}
    assert all(dtype == "torch.float32" for dtype in biases.values())
    return {
        "config": config,
        "record_path": record_path.relative_to(ROOT).as_posix(),
        "record_sha256": sha(record_path.read_bytes()),
        "recorded_storage": recorded,
        "observed_parameters": parameters,
        "observed_retained_modules": retained,
        "observed_buffers": buffers,
        "observed_bias_buffer_dtypes": biases,
        "observed_parameter_bytes": total,
        "observed_buffer_bytes": sum(buffers.values()),
        "scope": "以相同配置重建儲存配置；不重跑既有 GPU 訓練、品質與速度量測。",
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/17.7.md"
    raw = frozen.read_bytes()
    assert sha(raw) == "910138fde3d9486c81e8720db72b3d757506b8977941f3b341d952f1a80c8156"
    fences = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
    assert len(fences) == 3
    current_ast = ast.parse((ROOT / "tiny_perceptron/quantization.py").read_text())
    library_class = next(node for node in current_ast.body if isinstance(node, ast.ClassDef) and node.name == "QuantizedLinear")
    excerpt_class = ast.parse(fences[1]).body[0]
    assert ast.dump(library_class) == ast.dump(excerpt_class)
    ns4, four = run_fences(fences, 4)
    _, eight = run_fences(fences, 8)
    _, skipped = run_fences(fences, 4, omit_excerpt=True)
    hand = {
        "original_output": [str(Fraction(7, 10) + Fraction(2, 10)), str(-Fraction(7, 10) + Fraction(12, 10))],
        "four_bit_scales": [str(Fraction(7, 10) / 7), str(Fraction(12, 10) / 7)],
        "four_bit_second_negative_code_before_round": str(-Fraction(7, 10) / (Fraction(12, 10) / 7)),
        "four_bit_second_row_restored": [str(-4 * Fraction(12, 10) / 7), str(7 * Fraction(12, 10) / 7)],
        "four_bit_second_output": str(3 * Fraction(12, 10) / 7),
        "four_bit_storage_formula": "4 * 4 / 8 + 2 * 4 = 10 bytes",
        "eight_bit_code_ratios": [["127", str(Fraction(2, 10) / (Fraction(7, 10) / 127))], [str(-Fraction(7, 10) / (Fraction(12, 10) / 127)), "127"]],
        "eight_bit_storage_formula": "4 * 1 + 2 * 4 = 12 bytes",
        "retained_float_formula": "(264*64 + 128*64 + (2*2+1)*2*64) * 4 = 102912 bytes",
    }
    results = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-7/verify-02.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform()},
        "source_sha256": sha(raw),
        "class_excerpt_ast_equal": True,
        "four_bit": four,
        "eight_bit": eight,
        "skipped_excerpt": skipped,
        "module_behavior": module_behavior(ns4),
        "full_model_storage": full_model_storage(),
        "exact_arithmetic": hand,
        "original_source_sha256": inspect_originals(),
        "result": "全部斷言通過；浮點數值以絕對誤差 1e-6 核對，整數碼、buffer bytes 與同版輸出精確相等。",
    }
    save("verification-output.json", results)
    print(json.dumps({"result": results["result"], "artifacts": ["source-excerpts.json", "verification-output.json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
