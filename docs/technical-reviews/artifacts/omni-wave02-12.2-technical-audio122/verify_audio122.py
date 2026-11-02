"""12.2 的離線 CPU 核對；不訓練、不下載，也不執行整本 Notebook。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import struct
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.course_experiments.modalities import _resample_8_to_16
from tiny_perceptron.multimodal import tone


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
EXTERNAL = Path("/workspace/technical-audio122-validation")
EXTERNAL.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "12.2")
snapshot = ROOT / page["snapshot"]
assert sha(freeze_path) == "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
assert sha(snapshot) == page["source_sha256"]
assert page["figures_sha256"] == {}
blocks = re.findall(r"```python\n(.*?)```", snapshot.read_text(), re.S)
assert len(blocks) == 2

executions = []
for label, code in [
    ("第一段原碼", blocks[0]),
    ("第二段原碼", blocks[1]),
    ("9500/6500 練習", blocks[1].replace("9000", "9500").replace("7000", "6500")),
]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, page["snapshot"] + ":" + label, "exec"), {})
    stdout = stream.getvalue()
    print(label)
    print(stdout, end="")
    executions.append({"label": label, "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "stdout": stdout})

assert executions[0]["stdout"] == (
    "取樣率 16000 時長 0.1 奈奎斯特界線 8000.0 440Hz每週期樣本 36.36\n"
    "取樣率 8000 時長 0.2 奈奎斯特界線 4000.0 440Hz每週期樣本 18.18\n"
)
assert executions[1]["stdout"] == (
    "樣本數 16\n"
    "9000Hz前四點 [0.0, -0.191, 0.354, -0.462]\n"
    "7000Hz前四點 [0.0, 0.191, -0.354, 0.462]\n"
    "每一點都只差正負號 True\n"
)
assert executions[2]["stdout"].endswith("每一點都只差正負號 True\n")

aliases = []
for high_hz, low_hz in [(9000, 7000), (9500, 6500)]:
    high = tone(high_hz, seconds=0.001)
    low = tone(low_hz, seconds=0.001)
    difference = (high + low).abs()
    threshold = 1e-5 + 1e-5 * low.abs()
    assert bool((difference <= threshold).all())
    aliases.append({
        "high_hz": high_hz,
        "low_hz": low_hz,
        "sample_rate_hz": 16000,
        "sample_count": len(high),
        "dtype": str(high.dtype),
        "allclose": bool(torch.allclose(high, -low, atol=1e-5)),
        "max_absolute_error": float(difference.max()),
        "min_comparison_margin": float((threshold - difference).min()),
    })
print("alias_checks", json.dumps(aliases))

record_path = ROOT / "docs/course-experiments/results/real_modal.json"
record = json.loads(record_path.read_text())
conversion = record["results"]["fsdd"]["resampling"][0]
source_path = ROOT / "data/training/fsdd-initial/recordings/0_jackson_5.wav"
assert sha(source_path) == conversion["source_sha256"]
values, source_rate = sf.read(source_path, dtype="float32")
assert values.ndim == 1 and len(values) == 4591 and source_rate == 8000
derived = _resample_8_to_16(values)
assert len(derived) == 9182 and bool(np.isfinite(derived).all())
output_path = EXTERNAL / "0_jackson_5-resampled.wav"
sf.write(output_path, derived, 16000, subtype="FLOAT")
# 浮點 WAV 的 PEAK chunk 含寫入時間，整檔 hash 不是跨次重算的數值判準。
first_bytes = output_path.read_bytes()
time.sleep(1.1)
second_path = EXTERNAL / "same-samples-second-write.wav"
sf.write(second_path, derived, 16000, subtype="FLOAT")
second_bytes = second_path.read_bytes()
chunks = []
offset = 12
while offset + 8 <= len(first_bytes):
    tag, count = struct.unpack_from("<4sI", first_bytes, offset)
    chunks.append({"tag": tag.decode(), "offset": offset + 8, "bytes": count})
    offset += 8 + count + count % 2
peak = next(c for c in chunks if c["tag"] == "PEAK")
data = next(c for c in chunks if c["tag"] == "data")
differing_offsets = [i for i, (a, b) in enumerate(zip(first_bytes, second_bytes)) if a != b]
assert differing_offsets and all(peak["offset"] + 4 <= i < peak["offset"] + 8 for i in differing_offsets)
assert len(first_bytes) == len(second_bytes) == 36808
data_hash = hashlib.sha256(first_bytes[data["offset"] : data["offset"] + data["bytes"]]).hexdigest()
source_duration, target_duration = len(values) / source_rate, len(derived) / 16000
assert source_duration == target_duration == 0.573875
resampling = {
    "source": str(source_path.relative_to(ROOT)),
    "source_sha256": sha(source_path),
    "source_rate": source_rate,
    "source_samples": len(values),
    "output": str(output_path),
    "output_sha256": sha(output_path),
    "target_rate": 16000,
    "target_samples": len(derived),
    "source_duration_seconds": source_duration,
    "target_duration_seconds": target_duration,
    "recorded_derived_sha256": conversion["derived_sha256"],
    "matches_recorded_derived_sha256": sha(output_path) == conversion["derived_sha256"],
    "waveform_data_sha256": data_hash,
    "repeated_write_check": {
        "second_path": str(second_path),
        "second_sha256": sha(second_path),
        "differing_offsets": differing_offsets,
        "peak_chunk": peak,
        "interpretation": "相同 derived 數值寫成 WAV，差異只在 PEAK chunk 的 4-byte timestamp；因此不要求重播檔與舊紀錄整檔 hash 相同。",
    },
    "method": "實際呼叫既有 _resample_8_to_16；沒有重跑 real_modal 訓練",
}
print("resampling_check", json.dumps(resampling))

torch_docs = Path(torch.__file__).with_name("_torch_docs.py")
source_text = torch_docs.read_text()
allclose_offset = source_text.index("    torch.allclose,")
allclose_line = source_text[:allclose_offset].count("\n") + 1
allclose_excerpt = source_text[allclose_offset : source_text.index("Example::", allclose_offset)]
assert "rtol: float = 1e-05" in allclose_excerpt
assert "atol: float = 1e-08" in allclose_excerpt
assert "\\lvert \\text{other}_i \\rvert" in allclose_excerpt

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-wave02-12.2-technical-audio122/verify_audio122.py",
    "cwd": str(ROOT),
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "device": str(torch.get_default_device()),
        "platform": platform.platform(),
    },
    "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path)},
    "snapshot": {"path": page["snapshot"], "sha256": sha(snapshot)},
    "embedded_executions": executions,
    "alias_checks": aliases,
    "resampling": resampling,
    "torch_allclose_source": {
        "path": str(torch_docs),
        "sha256": sha(torch_docs),
        "line": allclose_line,
        "excerpt": allclose_excerpt,
    },
    "source_hashes": {
        path: sha(ROOT / path)
        for path in [
            "tiny_perceptron/multimodal.py",
            "scripts/course_experiments/modalities.py",
            "docs/course-experiments/results/real_modal.json",
            "tiny_perceptron/modal_data.py",
            "pyproject.toml",
        ]
    },
    "initial_probe": {
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-wave02-12.2-technical-audio122/verify_audio122.py > /workspace/technical-audio122-validation/run.stdout.txt 2> /workspace/technical-audio122-validation/run.stderr.txt",
        "code": str(EXTERNAL / "initial-verify-audio122.py"),
        "code_sha256": sha(EXTERNAL / "initial-verify-audio122.py"),
        "stdout": str(EXTERNAL / "initial-run.stdout.txt"),
        "stdout_sha256": sha(EXTERNAL / "initial-run.stdout.txt"),
        "stderr": str(EXTERNAL / "initial-run.stderr.txt"),
        "stderr_sha256": sha(EXTERNAL / "initial-run.stderr.txt"),
        "exit_code": 1,
        "result": "正文輸出通過；審閱自加的跨次 WAV 整檔 hash 相等斷言失敗。後續重複寫入證實同數值也會因 PEAK timestamp 改變整檔 hash。未將此視為課文數量／時長主張的反例。",
    },
    "result": "全部正文數值 assertion 通過；兩段原碼及修改練習輸出符合；原錄音 SHA 與既有紀錄相同，重取樣點數與時長相同。未取得舊 derived WAV 作逐點重播比對；整檔 hash 因 PEAK timestamp 不可作跨次相等判準。",
}
with (ARTIFACTS / "execution-results.json").open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print("result", result["result"])
