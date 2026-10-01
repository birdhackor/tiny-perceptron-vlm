"""環境檢查：確認這台機器能不能跑小小感知機。

用法：
    uv run python scripts/check_env.py

如果安裝時用了 extra（例如 `uv sync --extra cu128`），執行時也要帶同一個 extra：
    uv run --extra cu128 python scripts/check_env.py

回報問題時，請把這個腳本的輸出一起貼上。
"""

import importlib.metadata
import os
import platform
import shutil
import sys

PACKAGES = ["torch", "numpy", "pillow", "soundfile", "tokenizers", "datasets", "huggingface-hub"]


def show(key, value):
    print(f"{key:<16}{value}")


def pick_device(torch):
    """依序嘗試 CUDA → Apple MPS → CPU，並印出裝置資訊。"""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        show("device", f"CUDA：{p.name}，{p.total_memory / 2**30:.1f} GiB，sm_{p.major}{p.minor}")
        # 不算模擬：T4 等 Ampere 之前的顯卡只能「模擬」bf16，速度很慢
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        show("bf16", "支援" if native_bf16 else "不支援，訓練時改用 fp16（搭配 GradScaler）或 fp32")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        show("device", "Apple Silicon（MPS）")
        return torch.device("mps")
    show("device", f"CPU（{torch.get_num_threads()} 執行緒）")
    if shutil.which("nvidia-smi"):
        # 有 NVIDIA 驅動卻只能用 CPU，幾乎都是裝錯了 torch 版本
        if torch.version.cuda is None:
            show("hint", "有 NVIDIA 驅動，但裝的是 CPU 版 torch：請照 README 改用 --extra cu130 或 cu126")
        else:
            show("hint", f"torch 是 CUDA {torch.version.cuda} 版卻抓不到 GPU，多半是驅動太舊：請改用 --extra cu126")
    return torch.device("cpu")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 把輸出導到檔案時也能印中文

    show("python", f"{platform.python_version()}（{sys.executable}）")
    show("os", platform.platform())
    for name in PACKAGES:
        try:
            show(name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            show(name, "未安裝！請先執行 uv sync")

    import torch

    show("torch.cuda", torch.version.cuda or "無（CPU / MPS 版）")
    device = pick_device(torch)

    # 在選到的裝置上做一次前向 + 反向傳播，確認真的能算
    x = torch.randn(256, 256, device=device, requires_grad=True)
    (x @ x).sum().backward()
    show("compute", f"OK（{device.type}）")

    import soundfile

    show("libsndfile", soundfile.__libsndfile_version__)
    show("ffmpeg", shutil.which("ffmpeg") or "未找到（影片階段才需要）")
    show("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "未設定（使用 huggingface.co）"))


if __name__ == "__main__":
    main()
