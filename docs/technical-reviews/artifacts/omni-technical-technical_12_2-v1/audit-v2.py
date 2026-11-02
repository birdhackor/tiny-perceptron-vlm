"""12.2 獨立審閱：直接執行凍結教材程式，無訓練、無 GPU。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.course_experiments.modalities import _resample_8_to_16
from tiny_perceptron.multimodal import tone

ROOT = Path(__file__).resolve().parents[4]
RAW = Path("/workspace/work/reference/omni-technical-technical_12_2-v1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


snapshot = ROOT / "docs/course-review-20261019/freeze-05/sources/12.2.md"
freeze = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
page = next(p for p in json.loads(freeze.read_text())["pages"] if p["page_id"] == "12.2")
assert sha(snapshot) == page["source_sha256"]
assert page["figures_sha256"] == {}
body = snapshot.read_text()
current = re.search(r"^## 12\.2 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S).group()
assert current.encode() == snapshot.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 2
execution = []
for index, block in enumerate(blocks, 1):
    namespace = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(block, f"{snapshot}:python-block-{index}", "exec"), namespace)
    output = stream.getvalue()
    execution.append({"block": index, "code_sha256": hashlib.sha256(block.encode()).hexdigest(), "stdout": output})
    if index == 1:
        assert output == (
            "取樣率 16000 時長 0.1 奈奎斯特界線 8000.0 440Hz每週期樣本 36.36\n"
            "取樣率 8000 時長 0.2 奈奎斯特界線 4000.0 440Hz每週期樣本 18.18\n"
        )
    else:
        assert output == (
            "樣本數 16\n"
            "9000Hz前四點 [0.0, -0.191, 0.354, -0.462]\n"
            "7000Hz前四點 [0.0, 0.191, -0.354, 0.462]\n"
            "每一點都只差正負號 True\n"
        )

alias_pairs = []
for high_hz, low_hz in [(9000.0, 7000.0), (9500.0, 6500.0)]:
    high, low = tone(high_hz, seconds=0.001), tone(low_hz, seconds=0.001)
    difference = (high + low).abs()
    tolerance = 1e-5 + 1e-5 * low.abs()
    assert high.shape == low.shape == (16,)
    assert high.dtype == low.dtype == torch.float32
    assert torch.allclose(high, -low, atol=1e-5)
    assert bool((difference <= tolerance).all())
    n = np.arange(16, dtype=np.float64)
    direct = np.sin(2 * np.pi * high_hz * n / 16000) + np.sin(2 * np.pi * low_hz * n / 16000)
    assert np.max(np.abs(direct)) < 3e-14
    alias_pairs.append({
        "high_hz": high_hz, "low_hz": low_hz, "dtype": str(high.dtype), "device": str(high.device),
        "samples_each": len(high), "max_abs_high_plus_low": float(difference.max()),
        "min_elementwise_tolerance_margin": float((tolerance - difference).min()),
        "allclose": True, "independent_float64_identity_error": float(np.max(np.abs(direct))),
    })
exercise = blocks[1].replace("9000.0", "9500.0").replace("7000.0", "6500.0")
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(compile(exercise, "12.2-exercise-substitution", "exec"), {})
assert stream.getvalue().splitlines()[-1] == "每一點都只差正負號 True"

results_path = ROOT / "docs/course-experiments/results/real_modal.json"
record = json.loads(results_path.read_text())["results"]["fsdd"]["resampling"][0]
source = ROOT / "data/training/fsdd-initial/recordings/0_jackson_5.wav"
upstream = RAW / "0_jackson_5-upstream.wav"
assert sha(source) == sha(upstream) == record["source_sha256"]
values, rate = sf.read(source, dtype="float32")
assert values.shape == (4591,) and rate == 8000
converted = _resample_8_to_16(values)
assert converted.shape == (9182,) and np.isfinite(converted).all()
converted_path = RAW / "0_jackson_5-reproduced-v2-16000.wav"
sf.write(converted_path, converted, 16000, subtype="FLOAT")
roundtrip, new_rate = sf.read(converted_path, dtype="float32")
assert np.array_equal(converted, roundtrip) and new_rate == 16000
# FLOAT WAV 的 PEAK chunk 含寫入時間；全檔 SHA 不等不能推出數值不等。
first_raw = (RAW / "0_jackson_5-reproduced-16000.wav").read_bytes()
repeat_raw = (RAW / "0_jackson_5-reproduced-repeat-16000.wav").read_bytes()
changed_offsets = [i for i, (a, b) in enumerate(zip(first_raw, repeat_raw)) if a != b]
assert first_raw[48:52] == repeat_raw[48:52] == b"PEAK"
assert first_raw[80:] == repeat_raw[80:]
assert changed_offsets and all(60 <= i < 64 for i in changed_offsets)
assert len(values) / rate == len(converted) / new_rate == 0.573875

# 標籤改動的時間伸縮公式；同一個 n 序列沒有重新取樣。
old_rate, playback_rate, original_hz = 16000, 8000, 440
relabel_frequency = original_hz * playback_rate / old_rate
assert relabel_frequency == 220
assert 8000 / 8000 == 16000 / 16000 == 1

notebook = ROOT / page["notebook"]
assert sha(notebook) == page["notebook_sha256"]
cells = ["".join(c["source"]) for c in json.loads(notebook.read_text())["cells"] if c["cell_type"] == "code"]
assert all(any(block.strip() in cell for cell in cells) for block in blocks)

receipt = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_12_2",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_12_2-v1/audit-v2.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "torch_git": torch.version.git_version,
                    "numpy": np.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                    "platform": platform.platform(), "device": "cpu", "torch_default_dtype": str(torch.get_default_dtype())},
    "source_freeze": str(freeze.relative_to(ROOT)), "source_freeze_sha256": sha(freeze),
    "snapshot": str(snapshot.relative_to(ROOT)), "snapshot_sha256": sha(snapshot),
    "current_section_equals_snapshot": True, "figure_map": {},
    "notebook_sha256": sha(notebook), "notebook_contains_both_exact_source_blocks": True,
    "hashes": {str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT / "tiny_perceptron/multimodal.py", ROOT / "scripts/course_experiments/modalities.py", results_path,
        ROOT / "assets/training/manifest.json", ROOT / "assets/training/sources/fsdd.json"]},
    "embedded_execution": execution, "alias_pairs": alias_pairs,
    "exercise_stdout": stream.getvalue(),
    "relabel": {"same_samples": True, "rate_before": old_rate, "rate_playback": playback_rate,
                "duration_multiplier": old_rate / playback_rate, "original_frequency": original_hz, "played_frequency": relabel_frequency},
    "recording": {"source": str(source), "upstream": str(upstream), "source_sha256": sha(source),
                  "rate_before": rate, "samples_before": len(values), "rate_after": new_rate, "samples_after": len(converted),
                  "duration_before": len(values) / rate, "duration_after": len(converted) / new_rate,
                  "source_amplitude_min": float(values.min()), "source_amplitude_max": float(values.max()),
                  "recorded_method": record["method"], "reproduced_wav": str(converted_path),
                  "reproduced_wav_sha256": sha(converted_path), "recorded_derived_sha256": record["derived_sha256"],
                  "matches_recorded_derived_sha256": sha(converted_path) == record["derived_sha256"],
                  "peak_timestamp_check": {"first_wav_sha256": hashlib.sha256(first_raw).hexdigest(),
                    "repeat_wav_sha256": hashlib.sha256(repeat_raw).hexdigest(),
                    "differing_byte_offsets": changed_offsets, "equal_data_chunks": True,
                    "explanation": "FLOAT WAV PEAK chunk offsets 60..63 store a write timestamp; same samples can have different file SHA. No historical derived-file byte reproduction is claimed."}},
    "scope": "只核兩個教材短程式、練習數值與單一 WAV 重取樣；未訓練模型，未使用 GPU，未測模型品質。",
    "result": "PASS: every explicit assertion completed",
}
print(json.dumps(receipt, ensure_ascii=False, indent=2))
