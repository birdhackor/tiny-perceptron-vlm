"""Anonymous pinned downloads, real CLI calls, packing and entrypoint guards."""

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.common import Context  # noqa: E402
from scripts.course_experiments.compression import _dataset, _steps  # noqa: E402
from scripts.course_experiments.modalities import (  # noqa: E402
    _steps as modal_source_steps,
)
from tiny_perceptron.quantization import (  # noqa: E402
    QuantizedLinear,
    pack_int4,
    unpack_int4,
)
from tiny_perceptron.training import load_checkpoint  # noqa: E402

torch.set_num_threads(2)
out = ROOT / "docs/technical-reviews/artifacts"
manifest = json.loads((ROOT / "docs/course-experiments/public-models.json").read_text())
receipt = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_t_10-supplement.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "downloads": [],
    "cli": [],
}
with tempfile.TemporaryDirectory(prefix="fact_finish_t_10-") as temporary:
    temp = Path(temporary)
    for experiment, names in [
        ("distillation", ["sft-w32-ce.pt", "sft-w32-ce_kl.pt", "sft-w32-ce_kl-packed4.pt"]),
        ("multimodal_distillation", ["joint-ce_kl.pt"]),
    ]:
        entry = next(item for item in manifest["models"] if item["id"] == experiment)
        for name in names:
            file = next(item for item in entry["files"] if item["output"] == name)
            url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{file['path']}"
            # urllib adds no authentication header; HTTPS certificate verification remains enabled.
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
                status = response.status
            actual_hash = hashlib.sha256(data).hexdigest()
            assert actual_hash == file["sha256"]
            assert len(data) == file["bytes"]
            path = temp / name
            path.write_bytes(data)
            model, _ = load_checkpoint(path, "cpu")
            receipt["downloads"].append(
                {"url": url, "status": status, "authentication": "none", "sha256": actual_hash, "bytes": len(data)}
            )
            if name in ["sft-w32-ce.pt", "sft-w32-ce_kl.pt"]:
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/infer.py",
                    str(path),
                    "--chat",
                    "--prompt",
                    "color=red;shape=square;pitch=high;joint?",
                    "--tokens",
                    "24",
                    "--device",
                    "cpu",
                    "--json",
                ]
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
                generated = json.loads(process.stdout)
                assert generated["answer"] == ("square,low" if name == "sft-w32-ce.pt" else "square,high")
                receipt["cli"].append(
                    {
                        "command": command,
                        "exit_code": process.returncode,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                    }
                )
            if name == "sft-w32-ce_kl-packed4.pt":
                parts = {"packed_values": 0, "scales": 0, "biases": 0, "remaining_float_parameters": 0}
                for layer in model.modules():
                    if isinstance(layer, QuantizedLinear):
                        parts["packed_values"] += layer.values.numel() * layer.values.element_size()
                        parts["scales"] += layer.scale.numel() * layer.scale.element_size()
                        if layer.bias is not None:
                            parts["biases"] += layer.bias.numel() * layer.bias.element_size()
                parts["remaining_float_parameters"] = sum(p.numel() * p.element_size() for p in model.parameters())
                assert sum(parts.values()) == 64160
                receipt["packed_storage"] = {
                    "parts": parts,
                    "tensor_bytes": sum(parts.values()),
                    "public_file_bytes": len(data),
                    "forward_dtype": "FP32 weights restored by QuantizedLinear.forward",
                }
    ctx = Context("cpu", temp, temp, ROOT / "assets/training", seed=42)
    assert _steps(ctx, 350) == 350
    assert modal_source_steps(ctx, 160, 8) == 8
    assert modal_source_steps(ctx, 400, 16) == 16
    errors = []
    try:
        ctx.dependency("missing")
    except FileNotFoundError as error:
        errors.append({"guard": "missing dependency", "exception": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("missing dependency must fail")
    (temp / "leak").mkdir()
    (temp / "leak/dataset.json").write_text(
        json.dumps({"train": [{"family": "same"}], "validation": [{"family": "same"}], "test": [{"family": "other"}]})
    )
    try:
        _dataset(ctx, "leak")
    except ValueError as error:
        errors.append({"guard": "overlapping family", "exception": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("overlapping families must fail")
    receipt["entrypoint_guards"] = errors
    receipt["schedules"] = {
        "compression_cpu_350": _steps(ctx, 350),
        "vqa_source_cpu": modal_source_steps(ctx, 160, 8),
        "joint_source_cpu": modal_source_steps(ctx, 400, 16),
    }
q = torch.tensor([-8, -1, 0, 1, 7, 2, 3], dtype=torch.int8)
packed = pack_int4(q)
assert packed.tolist() == [112, 152, 175, 139]
assert torch.equal(unpack_int4(packed, q.shape), q)
receipt["packing_probe"] = {
    "integers": q.tolist(),
    "packed": packed.tolist(),
    "bytes": packed.numel() * packed.element_size(),
    "restored": unpack_int4(packed, q.shape).tolist(),
}
command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--list-assets", "distillation"]
process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
assert process.stdout.strip() == "assets/training/gsm8k-v1.tar.gz"
receipt["cli"].append(
    {"command": command, "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
)
receipt["result"] = (
    "All assertions passed; four anonymous pinned downloads verified, two text CLI examples reproduced, no long training performed."
)
(out / "fact_finish_t_10-supplement-result.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
