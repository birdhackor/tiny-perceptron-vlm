"""10.12 獨立 CPU 探針：執行凍結內嵌碼，不讀作者或其他審閱結果。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.omni_products.data import AVInput  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject(call):
    try:
        call()
    except ValueError as error:
        return str(error)
    raise AssertionError("應拒絕的輸入卻被接受")


def main():
    torch.set_num_threads(1)
    freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"
    freeze = json.loads(freeze_path.read_text())
    page = next(item for item in freeze["pages"] if item["page_id"] == "10.12")
    body = (ROOT / page["snapshot"]).read_text()
    current = re.search(r"^## 10\.12 .+?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S)[0]
    assert current == body
    assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"]
    assert digest(ROOT / page["notebook"]) == page["notebook_sha256"]
    for path, sha in page["figures_sha256"].items():
        assert digest(ROOT / path) == sha

    code_blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(code_blocks) == 1
    code = code_blocks[0]
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, page["snapshot"], "exec"), namespace)
    expected = (
        "A 1500 ms 採影格 1\nB 1500 ms 採影格 3\n"
        "A 在切換前／切換時的影格 [1, 2]\n記錄項數 17\n"
        "兩個片段的開始位置 [(0, 'A'), (8, 'B')]\n"
    )
    assert output.getvalue() == expected
    frames, times_a, times_b = (namespace[k] for k in ("frames", "times_a", "times_b"))
    at, validate = namespace["at"], namespace["validate"]
    circle, square = namespace["circle"], namespace["square"]
    assert frames.shape == (4, 1, 32, 32)
    assert namespace["y"][2, 3].item() == -13.5
    assert namespace["x"][2, 3].item() == -12.5
    assert torch.equal(frames[0, 0], circle) and torch.equal(frames[3, 0], square)
    assert torch.equal(frames.mean(0), frames.mean(0))

    queries = [0, 499, 500, 999, 1000, 1499, 1500, 1999, 2000, 2999, 3000, 3999]
    index_rows = {}
    for name, times in [("A", times_a), ("B", times_b)]:
        index_rows[name] = [{"query_ms": q, "index": at(frames, times, 4000, q)[0]} for q in queries]
        for row in index_rows[name]:
            manual = max(i for i, t in enumerate(times.tolist()) if t <= row["query_ms"])
            assert row["index"] == manual
    assert torch.equal(at(frames, times_a, 4000, 1500)[1][0], circle)
    assert torch.equal(at(frames, times_b, 4000, 1500)[1][0], square)
    query_rejections = {str(q): reject(lambda q=q: at(frames, times_a, 4000, q)) for q in [-1, 4000, float("nan"), float("inf")]}

    invalid = {
        "rank": (frames[:, 0], times_a, 4000),
        "channel": (frames.expand(-1, 3, -1, -1), times_a, 4000),
        "empty": (frames[:0], times_a[:0], 4000),
        "17_frames": (frames[:1].repeat(17, 1, 1, 1), torch.arange(17) * 100, 4000),
        "timestamp_count": (frames, times_a[:3], 4000),
        "nonfinite_pixels": (torch.full_like(frames, float("nan")), times_a, 4000),
        "nonfinite_times": (frames, torch.tensor([0., 1000., float("inf"), 3000.]), 4000),
        "zero_duration": (frames, times_a, 0),
        "over_duration": (frames, times_a, 4001),
        "first_100": (frames, torch.tensor([100., 1000., 2000., 3000.]), 4000),
        "duplicate": (frames, torch.tensor([0., 1000., 1000., 3000.]), 4000),
        "decreasing": (frames, torch.tensor([0., 2000., 1000., 3000.]), 4000),
        "frame_at_end": (frames, torch.tensor([0., 1000., 2000., 4000.]), 4000),
    }
    validation_rejections = {name: reject(lambda args=args: validate(*args)) for name, args in invalid.items()}
    exercise = torch.tensor([0., 500., 1750., 3000.])
    exercise_index, exercise_image = at(frames, exercise, 4000, 1500)
    assert exercise_index == 1 and torch.equal(exercise_image[0], circle)
    first_100_exercise = reject(lambda: at(frames, torch.tensor([100., 500., 1750., 3000.]), 4000, 1500))

    records = namespace["records"]
    record_skeleton = [{k: v for k, v in row.items() if k != "pixels"} for row in records]
    assert [i for i, row in enumerate(records) if row["kind"] == "clip_begin"] == [0, 8]
    pixel_payloads = [row["pixels"] for row in records if row["kind"] == "frame"]
    assert len(pixel_payloads) == 8 and all(x.shape == (1, 32, 32) for x in pixel_payloads)
    assert not any("圓" in str(row) or "方" in str(row) for row in record_skeleton if row["kind"] != "text")

    # 只用人工資料呼叫輸入驗證；不建立或推論任何 NN，不讀 final family。
    max_frames = frames[:1].repeat(16, 1, 1, 1)
    max_times = torch.arange(16, dtype=torch.float32) * 250
    product_input = AVInput(max_frames, max_times, torch.zeros(1600), 0., 4000)
    assert product_input.validate() is product_input
    product_rejections = {
        "17_frames": reject(lambda: replace(product_input, frames=frames[:1].repeat(17, 1, 1, 1), frame_ms=torch.arange(17) * 200).validate()),
        "4001_ms": reject(lambda: replace(product_input, duration_ms=4001).validate()),
        "3_channels": reject(lambda: replace(product_input, frames=max_frames.expand(-1, 3, -1, -1)).validate()),
        "first_100": reject(lambda: replace(product_input, frame_ms=max_times + 100).validate()),
    }

    notebook = json.loads((ROOT / page["notebook"]).read_text())
    assert "".join(notebook["cells"][5]["source"]) == code.rstrip("\n")
    notebook_ns = {}
    notebook_outputs = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                exec(compile("".join(cell["source"]), f"{page['notebook']}:cell{index}", "exec"), notebook_ns)
            notebook_outputs.append({"cell_index": index, "stdout": captured.getvalue()})
    assert notebook_outputs[-1]["stdout"] == expected

    data_path = ROOT / "tiny_perceptron/omni_products/data.py"
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_p5_technical_10_12",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU", "platform": platform.platform()},
        "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": digest(freeze_path), "source_sha256": page["source_sha256"], "notebook_sha256": page["notebook_sha256"], "figure_sha256": page["figures_sha256"]},
        "input_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "inline_stdout": output.getvalue(),
        "image_geometry": {"shape": list(frames.shape), "dtype": str(frames.dtype), "axis_min": namespace["axis"].min().item(), "axis_max": namespace["axis"].max().item(), "y_2_3": namespace["y"][2, 3].item(), "x_2_3": namespace["x"][2, 3].item(), "circle_white_pixels": int(circle.sum()), "square_white_pixels": int(square.sum()), "pixel_values": torch.unique(frames).tolist()},
        "query_rows": index_rows,
        "query_rejections": query_rejections,
        "validation_rejections": validation_rejections,
        "exercise": {"times_ms": exercise.tolist(), "query_ms": 1500, "index": exercise_index, "is_circle": True, "first_100_rejection": first_100_exercise},
        "records": {"count": len(records), "skeleton": record_skeleton, "pixel_payload_count": len(pixel_payloads), "payload_shape": [1, 32, 32], "contains_shape_answer_metadata": False},
        "product_A_input": {"path": str(data_path.relative_to(ROOT)), "sha256": digest(data_path), "valid_max_frames": 16, "valid_duration_ms": 4000, "valid_shape": list(max_frames.shape), "rejections": product_rejections, "neural_inference_executed": False},
        "notebook_code_cells": notebook_outputs,
        "limitations": ["只在 Python 3.13.5、PyTorch 2.14.1+cpu 人工 CPU 材料上執行。", "未啟動 Colab、安裝分支、訓練、模型推論或任何 preserved final family。", "未驗證完整網站頁面或學生閱讀；原始資料／權重未改動。"],
        "all_assertions_passed": True,
    }
    target = Path(__file__).with_name("probe-result-v1.json")
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "sha256": digest(target), "all_assertions_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
