"""12.2 的有界 CPU 核對：算術、AST、合成訊號與現存 WAV 標頭。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import tarfile
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14/sources/12.2.md"
EXPECTED_SOURCE_SHA = "5ed63e8c806b1082c3e8349b90bd736e768620a56d0a98d774ad9e39ea4d566e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isolated_function(path, name):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "lines": [node.lineno, node.end_lineno]}


def run_block(code, tone):
    tree = ast.parse(code)
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "tiny_perceptron.multimodal")
    ]
    namespace = {"tone": tone}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(tree, "frozen_12.2_code_block", "exec"), namespace)
    return output.getvalue(), namespace


def main():
    assert sha(SNAPSHOT) == EXPECTED_SOURCE_SHA
    chapter = (ROOT / "course/chapters/12.md").read_text()
    body = re.search(r"^## 12\.2 .*?(?=^## |\Z)", chapter, re.M | re.S).group()
    assert hashlib.sha256(body.encode()).hexdigest() == EXPECTED_SOURCE_SHA
    assert body.encode() == SNAPSHOT.read_bytes()
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(blocks) == 2
    tone, tone_source = isolated_function(ROOT / "tiny_perceptron/multimodal.py", "tone")
    rate_stdout, _ = run_block(blocks[0], tone)
    alias_stdout, values = run_block(blocks[1], tone)
    assert rate_stdout.splitlines() == [
        "取樣率 16000 時長 0.1 奈奎斯特界線 8000.0 440Hz每週期樣本 36.36",
        "取樣率 8000 時長 0.2 奈奎斯特界線 4000.0 440Hz每週期樣本 18.18",
    ]
    assert alias_stdout.splitlines() == [
        "樣本數 16",
        "9000Hz前四點 [0.0, -0.191, 0.354, -0.462]",
        "7000Hz前四點 [0.0, 0.191, -0.354, 0.462]",
        "每一點都只差正負號 True",
    ]
    high, low = values["high"], values["low"]
    residual = (high + low).abs()
    tolerance = 1e-5 + 1e-5 * low.abs()
    assert bool((residual <= tolerance).all())
    exercise = blocks[1].replace("9000.0", "9500.0").replace("7000.0", "6500.0")
    exercise = exercise.replace("9000Hz", "9500Hz").replace("7000Hz", "6500Hz")
    exercise_stdout, exercise_values = run_block(exercise, tone)
    assert exercise_stdout.splitlines()[-1] == "每一點都只差正負號 True"
    experiment = ROOT / "docs/course-experiments/results/real_modal.json"
    record = json.loads(experiment.read_text())["results"]["fsdd"]["resampling"][0]
    assert record["source"] == "0_jackson_5.wav"
    assert (record["source_rate"], record["target_rate"], record["samples_before"], record["samples_after"]) == (8000, 16000, 4591, 9182)
    with tarfile.open(ROOT / "assets/training/fsdd-v1.tar.gz", "r:gz") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith("/0_jackson_5.wav"))
        raw = archive.extractfile(member).read()
    with wave.open(io.BytesIO(raw), "rb") as recording:
        header = {"channels": recording.getnchannels(), "sample_rate": recording.getframerate(), "frames": recording.getnframes(), "sample_width_bytes": recording.getsampwidth()}
    assert hashlib.sha256(raw).hexdigest() == record["source_sha256"]
    assert header == {"channels": 1, "sample_rate": 8000, "frames": 4591, "sample_width_bytes": 2}
    duration_before = record["samples_before"] / record["source_rate"]
    duration_after = record["samples_after"] / record["target_rate"]
    assert duration_before == duration_after == 0.573875
    resample, resample_source = isolated_function(ROOT / "scripts/course_experiments/modalities.py", "_resample_8_to_16")
    synthetic = np.arange(8, dtype=np.float32) / 8
    resampled = resample(synthetic)
    assert len(resampled) == 16
    assert len(synthetic) / 8000 == len(resampled) / 16000
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "default_dtype": str(torch.get_default_dtype())},
        "source_sha256": sha(SNAPSHOT),
        "tone_source": tone_source,
        "rate_block_stdout": rate_stdout,
        "alias_block_stdout": alias_stdout,
        "alias_arrays": {"shape": list(high.shape), "dtype": str(high.dtype), "max_abs_high_plus_low": residual.max().item(), "min_allclose_margin": (tolerance - residual).min().item(), "allclose_atol": 1e-5, "allclose_default_rtol": 1e-5},
        "exercise_block_stdout": exercise_stdout,
        "exercise_max_abs_high_plus_low": (exercise_values["high"] + exercise_values["low"]).abs().max().item(),
        "allclose_installed_signature": next(line for line in torch.allclose.__doc__.splitlines() if line.strip()),
        "archival_experiment": {"path": experiment.relative_to(ROOT).as_posix(), "sha256": sha(experiment), "json_pointer": "/results/fsdd/resampling/0", "record": record, "original_wav_header": header, "duration_before": duration_before, "duration_after": duration_after},
        "resample_synthetic": {"source": resample_source, "input_samples": len(synthetic), "output_samples": len(resampled), "output_dtype": str(resampled.dtype), "input_rate": 8000, "output_rate": 16000},
        "exercise_sample_count": 8000 / 8000 * 16000,
        "playback_rate_arithmetic": {"new_duration_factor": 16000 / 8000, "new_frequency": 440 * 8000 / 16000},
        "scope": "Only two frozen code blocks, their stated frequency substitution, an 8-sample synthetic resampler check, JSON arithmetic, AST extraction, and an existing WAV header. No neural forward, teacher, training, kernel, full-notebook or full-book execution. Archival real-audio resampling and its training experiment were not rerun.",
        "assertions_passed": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
