"""16.9 獨立技術核對：原文程式、固定算例、既有 CUDA 紀錄統計與圖收據。

不執行 CUDA、不建立或訓練模型。第三方全文與 PNG 保留在 Git 外。
"""

import contextlib
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/reference/omni-technical-technical_16_9-v1")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(path, identifier):
    text = path.read_text()
    headings = list(re.finditer(r"^## (\S+).*$", text, re.M))
    for i, heading in enumerate(headings):
        if heading[1] == identifier:
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            return text[heading.start() : end]
    raise ValueError(identifier)


def run_inline(code):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "freeze-16.9-inline.py", "exec"), namespace)
    return {
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "stdout": stdout.getvalue(),
        "scores_dtype": str(namespace["scores"].dtype),
        "device": str(namespace["scores"].device),
        "denominator": namespace["denominator"].item(),
        "numerator": namespace["numerator"].item(),
        "result": (namespace["numerator"] / namespace["denominator"]).item(),
    }


report = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_16_9",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_16_9-v1/audit.py docs/technical-reviews/artifacts/omni-technical-technical_16_9-v1/evidence-v1.json",
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "device": "cpu",
        "cuda_runtime": str(torch.version.cuda),
    },
    "scope": "本次親跑 CPU 課文／練習與離線測試、重算歷史紀錄；未重跑 GPU、未訓練模型。",
}
freeze = json.loads(FREEZE.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "16.9")
snapshot = ROOT / page["snapshot"]
raw = snapshot.read_text()
assert digest(snapshot) == page["source_sha256"]
assert raw == section(ROOT / "course/chapters/16.md", "16.9")
report["source"] = {
    "freeze": str(FREEZE.relative_to(ROOT)),
    "freeze_sha256": digest(FREEZE),
    "snapshot": page["snapshot"],
    "source_sha256": digest(snapshot),
    "current_section_equals_snapshot": True,
    "prerequisites": [],
}
for identifier in ("16.8", "16.10"):
    item = next(p for p in freeze["pages"] if p["page_id"] == identifier)
    current = section(ROOT / item["source"], identifier)
    assert current == (ROOT / item["snapshot"]).read_text()
    report["source"]["prerequisites"].append({
        "lesson_id": identifier, "snapshot": item["snapshot"],
        "current_sha256": hashlib.sha256(current.encode()).hexdigest(),
        "scope_read": "16.8 完整；16.10 重算機制與範圍，非另審其品質數字。",
    })

code = re.search(r"```python\n(.*?)```", raw, re.S)[1]
report["inline_original"] = run_inline(code)
assert report["inline_original"]["stdout"].splitlines() == [
    "分母/分子 1.5 25.0", "分母/分子 1.75 42.5", "分塊結果 24.2857", "完整結果 24.2857"
]
assert report["inline_original"]["denominator"] == 1.75
assert report["inline_original"]["numerator"] == 42.5
assert abs(report["inline_original"]["result"] - 170 / 7) < 2e-6
assert code.count("range(0, 3, 2)") == 1 and code.count("start : start + 2") == 2
exercise = code.replace("range(0, 3, 2)", "range(0, 3, 1)").replace("start : start + 2", "start : start + 1")
report["inline_exercise_block_1"] = run_inline(exercise)
assert report["inline_exercise_block_1"]["stdout"].splitlines() == [
    "分母/分子 1.0 10.0", "分母/分子 1.5 25.0", "分母/分子 1.75 42.5", "分塊結果 24.2857", "完整結果 24.2857"
]
report["numeric"] = {
    "T_squared": {str(t): t * t for t in (4, 8, 8192)},
    "log_values": [math.log(i) for i in (1, 2, 4)],
    "exp_log_values": [math.exp(math.log(i)) for i in (1, 2, 4)],
    "shifted_exp_weights": [math.exp(math.log(i) - math.log(4)) for i in (1, 2, 4)],
    "unshifted_numerator": 10 + 2 * 20 + 4 * 30,
    "unshifted_denominator": 1 + 2 + 4,
    "shifted_numerator": 0.25 * 10 + 0.5 * 20 + 30,
    "shifted_denominator": 0.25 + 0.5 + 1,
    "first_rescale_factor": torch.exp(torch.tensor(float("-inf")) - torch.tensor(math.log(2))).item(),
    "second_rescale_factor": math.exp(math.log(2) - math.log(4)),
    "MiB_in_bytes": 1024 * 1024,
}

probe_path = ROOT / "docs/course-experiments/results/flash_probe.json"
probe = json.loads(probe_path.read_text())
record = probe["results"]
implementation = "scripts/course_experiments/architecture.py"
implementation_digest = digest(ROOT / implementation)
assert implementation_digest == probe["code_sha256"][implementation]
old_code = subprocess.run(["git", "show", f"{probe['revision']}:{implementation}"], cwd=ROOT, capture_output=True, check=True)
assert hashlib.sha256(old_code.stdout).hexdigest() == implementation_digest
report["historical_flash_probe"] = {
    "path": str(probe_path.relative_to(ROOT)), "sha256": digest(probe_path),
    "revision": probe["revision"], "modal_run_id": probe["modal"]["run_id"],
    "implementation_sha256": implementation_digest,
    "recorded_revision_and_current_implementation_equal": True,
    "runtime": {k: record["runtime"][k] for k in ("gpu", "torch", "torch_git_version", "cuda_runtime_version", "compute_capability", "flash_attention_built", "driver_query")},
    "configuration": record["configuration"],
    "routes": {},
    "scope": "重算保存的原始九次時間與配置器 bytes；閱讀當次 profiler operator/kernel 名稱和誤差紀錄，未在本次重跑 CUDA。",
}
assert record["configuration"]["shape_B_H_T_D"] == [2, 4, 512, 32]
for dtype, route in record["routes"].items():
    route_output = {"status": route["status"], "profiles": {}, "measurements": {}, "correctness": route["correctness"]}
    assert route["status"] == "completed"
    for mode, profile in route["profiles"].items():
        ops = profile["operator_names"]
        kernels = profile["cuda_kernel_names"]
        assert "aten::_scaled_dot_product_flash_attention" in ops
        assert any("pytorch_flash::flash_fwd_kernel" in k for k in kernels)
        if mode == "forward_backward":
            assert "aten::_scaled_dot_product_flash_attention_backward" in ops
            assert any("pytorch_flash::flash_bwd" in k for k in kernels)
        route_output["profiles"][mode] = {
            "flash_operator_names": [o for o in ops if o.startswith("aten::") and "flash" in o],
            "cuda_kernel_count": len(kernels), "cuda_flash_kernel_names_checked": True,
            "includes_backward": profile["includes_backward"],
        }
    for mode, comparison in route["measurements"].items():
        route_output["measurements"][mode] = {}
        for backend, measurement in comparison.items():
            median = statistics.median(measurement["samples_seconds"])
            assert median == measurement["median_seconds"]
            assert measurement["warmup_calls"] == 3
            assert measurement["measured_calls"] == len(measurement["samples_seconds"]) == 9
            assert measurement["synchronized"] is True
            baseline = measurement["allocated_before_bytes"]
            peak = measurement["peak_allocated_bytes"]
            assert peak - baseline == measurement["additional_peak_allocated_bytes"]
            assert baseline == 65 * 2**20
            route_output["measurements"][mode][backend] = {
                "warmup_calls": 3, "measured_calls": 9,
                "median_ms_recomputed": 1000 * median,
                "median_ms_rounded_3": round(1000 * median, 3),
                "baseline_MiB": baseline / 2**20,
                "additional_peak_MiB": (peak - baseline) / 2**20,
                "total_peak_MiB": peak / 2**20,
            }
    for item in [route["correctness"]["output"], *route["correctness"]["gradients"].values()]:
        assert item["finite"] and item["within_declared_tolerance"]
    report["historical_flash_probe"]["routes"][dtype] = route_output

efficiency_path = ROOT / "docs/course-experiments/results/efficiency.json"
efficiency = json.loads(efficiency_path.read_text())["results"]["manual_vs_sdpa"]
assert efficiency["backend"]["selected"] == "memory_efficient"
assert efficiency["backend"]["flash_attention_observed"] is False
report["prior_sdpa_record"] = {
    "path": str(efficiency_path.relative_to(ROOT)), "sha256": digest(efficiency_path),
    "dtype": efficiency["dtype"], "shape": efficiency["shape"], "backend": efficiency["backend"],
    "timings_ms_recomputed": {},
}
for backend, timing in efficiency["timings"].items():
    median = statistics.median(timing["samples_seconds"])
    assert median == timing["median_seconds"]
    report["prior_sdpa_record"]["timings_ms_recomputed"][backend] = 1000 * median
report["prior_sdpa_record"]["difference_of_rounded_ms"] = 2.990 - 2.346

test_cmd = ["uv", "run", "--extra", "cpu", "python", "-m", "pytest", "tests/test_course_flash_probe.py", "-q", "-k", "manual_same_dtype_reference or manual_reference_masks or measurement_completes or cpu_sdpa_profiler"]
test = subprocess.run(test_cmd, cwd=ROOT, text=True, capture_output=True)
report["offline_tests"] = {"command": test_cmd, "exit_code": test.returncode, "stdout": test.stdout, "stderr": test.stderr, "test_source_sha256": digest(ROOT / "tests/test_course_flash_probe.py")}
assert test.returncode == 0

report["figures"] = []
figures = [
    ("course/figures/rewrite-16-online-softmax.svg", "online-softmax-inkscape.png", "三框依序向下；舊分子與分母同乘0.5，再加入新分母1與分子30；42.5/1.75=24.2857。未見裁切或遮蓋。"),
    ("course/figures/flash_allocated_memory.svg", "flash-memory-inkscape.png", "四柱共用起點65MiB與比例尺；新增量24.25/0.267/32.75/2.032，總峰值89.25/65.27/97.75/67.03，圖註限制為注意力單一步驟與allocated bytes。未見裁切或遮蓋。"),
]
for source, image_name, note in figures:
    svg, png = ROOT / source, CACHE / image_name
    assert digest(svg) == page["figures_sha256"][source]
    with Image.open(png) as picture:
        dimensions = list(picture.size)
    report["figures"].append({
        "source": source, "source_sha256": digest(svg),
        "render_command": f"inkscape {source} --export-type=png --export-filename={png}",
        "renderer": "Inkscape 1.4 (e7c3feb100, 2024-10-09)",
        "png_path_outside_git": str(png), "png_sha256": digest(png),
        "dimensions": dimensions, "viewed_with": "functions.exec tools.view_image",
        "inspection_note": note,
    })
tree = ET.parse(ROOT / "course/figures/flash_allocated_memory.svg")
rectangles = tree.findall("{http://www.w3.org/2000/svg}rect")[1:]
widths = [float(r.attrib["width"]) for r in rectangles]
assert widths == [286, 106.7, 286, 1.1748, 286, 144.1, 286, 8.9408]
report["memory_figure_scale"] = {
    "pixels_per_MiB": 4.4,
    "baseline_widths": widths[0::2], "additional_widths": widths[1::2],
    "additional_label_values": [24.25, 0.267, 32.75, 2.032],
    "max_width_error_pixels_vs_rounded_labels": max(abs(w - v * 4.4) for w, v in zip(widths[1::2], [24.25, 0.267, 32.75, 2.032], strict=True)),
}

report["original_retrieval_receipts"] = json.loads((CACHE / "retrieval.json").read_text())
for receipt in report["original_retrieval_receipts"]:
    assert digest(Path(receipt["file"])) == receipt["sha256"]
report["original_inspection"] = [
    {"file": "flashattention-2205.14135v2.pdf", "version": "arXiv:2205.14135v2, 23 Jun 2022", "locators": "§2.1–2.2 pp.3–4; §3.1 pp.4–5, Algorithm 1/Theorem 1; §3.2 p.6 Theorem 2; Appendix B.2 pp.18–19; Appendix C pp.22–23 induction", "quote": "by storing the output O and the softmax normalization statistics (m, ℓ), we can recompute the attention matrix S and P", "note": "親讀原PDF的pdftotext抽取內容；核對tiling、m/l重縮放、分子合併、exact完整注意力與反傳重算；未拿論文的A100速度當L4結果。"},
    {"file": "pytorch-sdp-utils.cpp", "locators": "L413–468 hardware; L894–915 dtype; L1047–1096 build/input checks", "quote": "using sm80 = SMVersion<8, 0>; using sm121 = SMVersion<12, 1>;", "note": "親讀CUDA條件，不沿用可能落後的註解；SM8.9在[sm80,sm121]，SM8.x透過low_precision只接受Half/BFloat16；build與其他輸入條件仍必要。"},
    {"file": "pytorch-functional.py", "locators": "L6367–6403 SDPA equivalent formula; L6452–6471 selection/limitations/floating point", "quote": "Due to the nature of fusing floating point operations, the output of this function may be different", "note": "親讀scale、softmax、V加權與backend條件；CPU本次執行不證明CUDA選擇。"},
    {"file": "pytorch-cuda-memory.py", "locators": "L381–398 reset_peak_memory_stats; L525–560 memory_allocated/max_memory_allocated", "quote": "Return the maximum GPU memory occupied by tensors in bytes for a given device.", "note": "核對allocated而非reserved或nvidia-smi整卡用量；reset後的max是該區間追蹤的配置器峰值。"},
    {"file": "pytorch-numerical-accuracy.md", "locators": "L5–17", "quote": "PyTorch is not guaranteed to produce bitwise identical results for floating point computations that are mathematically identical.", "note": "原始官方文件明示運算順序與平台可能改浮點尾數。"},
    {"file": "nvidia-cuda-gpus.html", "locators": "CUDA GPU Compute Capability table, 8.9 row, NVIDIA L4", "quote": "8.9 / NVIDIA L4", "note": "親自解析當次官方HTML表格；確認L4列於8.9，不把型號當完整backend核驗。"},
]
report["completed_at"] = datetime.now(UTC).isoformat()
report["result"] = "全部CPU算例、block=1練習、7項離線測試、歷史量測統計／原始實作SHA、圖尺度核對通過；GPU結果限既有紀錄審查。"
output = Path(sys.argv[1])
with output.open("x") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps({"path": str(output), "sha256": digest(output), "result": report["result"]}, ensure_ascii=False))
