"""有界 GPU／檔案探測；不載入模型、不訓練、不需要服務憑證。"""

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch  # noqa: E402


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("探測必須恰有一張 CUDA GPU")
    device = torch.cuda.get_device_properties(0)
    if device.name not in {"NVIDIA L4", "L4"} or device.total_memory < 22 * 1024**3:
        raise RuntimeError("不是預定的 L4／記憶體形狀")
    identity = torch.eye(32, device="cuda")
    product = identity @ identity
    torch.cuda.synchronize()
    if not torch.isfinite(product).all().item() or not torch.equal(product, identity):
        raise RuntimeError("GPU 計算檢查失敗")
    raw = bytes(range(256)) * 4
    Path("/tmp/omni-probe-payload.bin").write_bytes(raw)
    result = {
        "schema": "omni-runtime-probe-v1",
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "safetensors", "accelerate")
        },
        "gpu": device.name,
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_bytes": device.total_memory,
        "cuda": torch.version.cuda,
        "finite_identity_matmul": True,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "model_loaded": False,
        "training_started": False,
    }
    Path("/tmp/omni-probe.json").write_text(json.dumps(result, allow_nan=False) + "\n")
    print("OMNI_PROBE_READY", flush=True)
    # filesystem helper 需要主程序仍存活；控制端取回後主動終止，伺服器另有限時。
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
