"""獨立核算 log 功率，並檢查固定前處理的形狀與振幅縮放。"""

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.multimodal import AudioEncoder, log_mel, mel_filter_bank


torch.set_num_threads(1)
power = torch.tensor([0.0, 1e-10, 1e-4, 1.0])
cases = []
for floor in (1e-8, 1e-12):
    expected = [math.log(max(float(p), floor)) for p in (0.0, 1e-10, 1e-4, 1.0)]
    expected_difference = [
        math.log(max(4 * p, floor)) - math.log(max(p, floor)) for p in (0.0, 1e-10, 1e-4, 1.0)
    ]
    actual = power.clamp(min=floor).log()
    difference = (4 * power).clamp(min=floor).log() - actual
    maximum_error = max(abs(x - y) for x, y in zip(expected, actual.tolist(), strict=True))
    difference_error = max(abs(x - y) for x, y in zip(expected_difference, difference.tolist(), strict=True))
    assert maximum_error < 2e-6, "log 功率與 math.log 的差超出 float32 容忍差"
    assert difference_error < 2e-6, "縮放後差值與手算的差超出 float32 容忍差"
    assert bool(actual.isfinite().all()), "正下限處理後應皆為有限值"
    cases.append(
        {
            "下限": floor,
            "手算公式": "ln(max(p, floor))；差值為 ln(max(4p, floor))−ln(max(p, floor))",
            "math_log預期": expected,
            "torch_log觀察": actual.tolist(),
            "顯示至兩位": [round(x, 2) for x in actual.tolist()],
            "預期差": expected_difference,
            "觀察差": difference.tolist(),
            "差值顯示至兩位": [round(x, 2) for x in difference.tolist()],
            "最大絕對誤差": maximum_error,
            "差值最大絕對誤差": difference_error,
            "全為有限值": bool(actual.isfinite().all()),
        }
    )

times = torch.arange(1600, dtype=torch.float64) / 16000
wave = 0.25 * torch.sin(2 * math.pi * 440 * times)
window = torch.hann_window(400, dtype=wave.dtype)
spectrum = torch.stft(wave, 400, 160, window=window, return_complex=True)
doubled_spectrum = torch.stft(2 * wave, 400, 160, window=window, return_complex=True)
spectral_power = spectrum.abs().square()
double_power = doubled_spectrum.abs().square()
amplitude_error = float((double_power - 4 * spectral_power).abs().max())
assert amplitude_error < 1e-10, "固定 STFT 設定下，振幅加倍的功率應為四倍"
bank = mel_filter_bank()
mel_power = bank.to(wave.dtype) @ spectral_power
features = log_mel(wave)
assert features.shape == (16, spectrum.shape[-1]), "mel 合成與 log 應保留時間軸"
assert torch.equal(features, mel_power.clamp(min=1e-8).log()), "固定前處理應符合功率合成後取 log"
assert bool(features.isfinite().all()), "聲音特徵應為有限值"

source_path = ROOT / "docs/course-review-20261019/freeze-01/sources/12.7.md"
chapter = (ROOT / "course/chapters/12.md").read_bytes()
section = chapter[chapter.index(b"## 12.7 ") : chapter.index(b"## 12.8 ")]
assert section == source_path.read_bytes(), "執行小節應與凍結小節逐位元組相同"
repo_code = ROOT / "tiny_perceptron/multimodal.py"
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-7/verify_log_power.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "device": "cpu",
        "scalar_dtype": str(power.dtype),
        "waveform_dtype": str(wave.dtype),
        "platform": platform.platform(),
    },
    "原文SHA256": hashlib.sha256(section).hexdigest(),
    "freeze_manifest_SHA256": hashlib.sha256((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes()).hexdigest(),
    "repository_HEAD": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    "implementation_SHA256": hashlib.sha256(repo_code.read_bytes()).hexdigest(),
    "線性亮度示例": {
        "條件": "僅示範功率直接映射至 0–255 並四捨五入；課文未宣稱指定顯示器的可見閾值",
        "相對最大功率": power.tolist(),
        "八位元亮度": [round(255 * p) for p in (0, 1e-10, 1e-4, 1)],
    },
    "下限算例": cases,
    "未處理零的log": str(torch.tensor(0.0).log().item()),
    "ln4": math.log(4),
    "不同下限的零功率差": math.log(1e-8) - math.log(1e-12),
    "振幅與功率": {
        "STFT輸入": "1600 個 float64 樣本；16000 Hz；440 Hz 正弦；振幅 0.25",
        "STFT配置": "n_fft=400，hop_length=160，Hann window，center=True，pad_mode=reflect",
        "振幅倍數": 2,
        "功率倍數": 4,
        "功率最大絕對誤差": amplitude_error,
    },
    "前處理形狀": {
        "STFT功率": list(spectral_power.shape),
        "mel權重": list(bank.shape),
        "mel功率": list(mel_power.shape),
        "log_mel": list(features.shape),
        "各框皆保留": True,
        "逐元素固定公式完全相等": True,
        "log_mel有可學參數": False,
        "入口的可學模組": str(AudioEncoder().projection),
    },
    "result": "所有數值與形狀核對通過；未執行訓練，不能支持辨識品質或其他後端的效能結論。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
