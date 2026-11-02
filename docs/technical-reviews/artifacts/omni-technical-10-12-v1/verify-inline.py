"""10.12 fix08：執行凍結正文的實際程式與有界邊界核對；不訓練模型。"""
import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix08.json"
page = next(p for p in json.loads(FREEZE.read_bytes())["pages"] if p["page_id"] == "10.12")
snapshot = ROOT / page["snapshot"]
raw = snapshot.read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(blocks) == 1
code = blocks[0]
torch.set_num_threads(1)
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, str(snapshot) + ":python", "exec"), namespace)
assert stdout.getvalue().splitlines() == [
    "A 1500 ms 採影格 1",
    "B 1500 ms 採影格 3",
    "A 在切換前／切換時的影格 [1, 2]",
    "記錄項數 17",
    "兩個片段的開始位置 [(0, 'A'), (8, 'B')]",
]
frames = namespace["frames"]
times_a, times_b = namespace["times_a"], namespace["times_b"]
at, validate = namespace["at"], namespace["validate"]
assert tuple(frames.shape) == (4, 1, 32, 32)
assert frames.device.type == "cpu" and frames.dtype == torch.float32
assert frames.unique().tolist() == [0.0, 1.0]
assert torch.equal(frames[0], frames[1]) and torch.equal(frames[2], frames[3])
assert not torch.equal(frames[1], frames[2])
assert all(torch.equal(at(frames, times, 4000.0, query)[1], frames[idx])
           for times, query, idx in [(times_a, 1500.0, 1), (times_b, 1500.0, 3)])
queries = [0.0, 499.0, 500.0, 999.0, 1000.0, 1499.0, 1500.0, 1999.0, 2000.0, 2999.0, 3000.0, 3999.0]
expected_a = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]
expected_b = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3]
observed_a = [at(frames, times_a, 4000.0, t)[0] for t in queries]
observed_b = [at(frames, times_b, 4000.0, t)[0] for t in queries]
assert observed_a == expected_a and observed_b == expected_b
exercise_times = torch.tensor([0.0, 500.0, 1750.0, 3000.0])
exercise_index, exercise_image = at(frames, exercise_times, 4000.0, 1500.0)
assert exercise_index == 1 and torch.equal(exercise_image, frames[0])
rejections = {}

def rejects(name, fn):
    try:
        fn()
    except ValueError as error:
        rejections[name] = str(error)
    else:
        raise AssertionError(name + " was accepted")

rejects("query_at_end", lambda: at(frames, times_a, 4000.0, 4000.0))
rejects("query_before_start", lambda: at(frames, times_a, 4000.0, -1.0))
rejects("query_nan", lambda: at(frames, times_a, 4000.0, float("nan")))
rejects("exercise_first_time_100", lambda: validate(frames, torch.tensor([100.0, 500.0, 1750.0, 3000.0]), 4000.0))
rejects("zero_frames", lambda: validate(frames[:0], times_a[:0], 4000.0))
rejects("seventeen_frames", lambda: validate(frames[:1].repeat(17, 1, 1, 1), torch.arange(17.0), 4000.0))
rejects("mismatched_time_count", lambda: validate(frames, times_a[:3], 4000.0))
rejects("wrong_image_dimensions", lambda: validate(frames[:, :, :31], times_a, 4000.0))
rejects("duplicate_times", lambda: validate(frames, torch.tensor([0.0, 1000.0, 1000.0, 3000.0]), 4000.0))
rejects("descending_times", lambda: validate(frames, torch.tensor([0.0, 2000.0, 1000.0, 3000.0]), 4000.0))
rejects("frame_at_end", lambda: validate(frames, torch.tensor([0.0, 1000.0, 2000.0, 4000.0]), 4000.0))
rejects("duration_zero", lambda: validate(frames, times_a, 0.0))
rejects("duration_over_four_seconds", lambda: validate(frames, times_a, 4001.0))
bad_pixels = frames.clone()
bad_pixels[0, 0, 0, 0] = float("nan")
rejects("nonfinite_pixels", lambda: validate(bad_pixels, times_a, 4000.0))
rejects("nonfinite_times", lambda: validate(frames, torch.tensor([0.0, 1000.0, 2000.0, float("inf")]), 4000.0))
for count in [1, 16]:
    validate(frames[:1].repeat(count, 1, 1, 1), torch.arange(count, dtype=torch.float32), 4000.0)
records = namespace["records"]
assert len(records) == 17
assert records[-1] == {"kind": "text", "text": "A 在 1500 ms 是什麼形狀？"}
for offset, clip, times in [(0, "A", times_a), (8, "B", times_b)]:
    assert records[offset] == {"kind": "clip_begin", "clip": clip}
    assert records[offset + 1] == {"kind": "video_begin"}
    for j in range(4):
        row = records[offset + 2 + j]
        assert row["kind"] == "frame" and row["time_ms"] == float(times[j])
        assert tuple(row["pixels"].shape) == (1, 32, 32)
        assert torch.equal(row["pixels"], frames[j])
        assert not any(k in row for k in ["shape", "label", "answer"])
    assert records[offset + 6] == {"kind": "video_end"}
    assert records[offset + 7] == {"kind": "clip_end"}
assert torch.equal(frames.mean(dim=0), frames.mean(dim=0))
assert at(frames, times_a, 4000.0, 1500.0)[0] != at(frames, times_b, 4000.0, 1500.0)[0]
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "source_snapshot": str(snapshot.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "inline_python_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-10-12-v1/verify-inline.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "threads": "1"},
    "inline_stdout": stdout.getvalue().splitlines(),
    "frames": {"shape": list(frames.shape), "dtype": str(frames.dtype), "unique_values": frames.unique().tolist(), "white_pixel_counts": frames.sum((1, 2, 3)).tolist()},
    "sample_and_hold": {"queries_ms": queries, "indices_a": observed_a, "indices_b": observed_b},
    "exercise": {"times_b": exercise_times.tolist(), "query_ms": 1500.0, "index": exercise_index, "is_circle": bool(torch.equal(exercise_image, frames[0]))},
    "validation_rejections": rejections,
    "validation_accepted_frame_counts": [1, 16],
    "records": {"count": len(records), "clip_starts": [[i, r["clip"]] for i, r in enumerate(records) if r["kind"] == "clip_begin"], "pixel_records": 8, "no_shape_answer_fields": True, "times_reset_per_clip": True},
    "result": "All assertions passed. Deterministic inline code and bounded exercises only; no model inference/training or quality measurement.",
}
target = OUT / "execution-summary.json"
with target.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
