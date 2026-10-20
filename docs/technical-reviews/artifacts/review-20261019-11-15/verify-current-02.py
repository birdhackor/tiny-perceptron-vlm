"""核對目前凍結原文、亮線索引與可沿用證據，保存當次實跑輸出。"""

import ast
import difflib
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.natural_concepts import thin_stroke_report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_11-15/technical-11-15-original-01.json"
original = json.loads(original_path.read_text())
old_source = ROOT / "docs/course-review-20261019/freeze-01/sources/11.15.md"
new_source = ROOT / "docs/course-review-20261019/freeze-05/sources/11.15.md"
freeze = ROOT / "docs/course-review-20261019/freeze-05.json"
old_raw = old_source.read_bytes()
new_raw = new_source.read_bytes()
chapter = (ROOT / "course/chapters/11.md").read_bytes()
start = chapter.index(b"## 11.15 ")
end = chapter.index(b"\n## ", start + 1) + 1
current_raw = chapter[start:end]

assert digest(original_path) == "046b64a08731e3892f3287fb90397f1aa0dcf7a9cea1cde761b1a4666a245d1e", "原始報告已變更"
assert hashlib.sha256(new_raw).hexdigest() == "78a9270ab0c4425cc5445cad50ba4a54aa07b075f00d45121fa7acb9ac2c9d84", "目前原文雜湊不符"
assert digest(freeze) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba", "凍結清單雜湊不符"
assert new_raw == current_raw, "現行章檔與凍結小節不同"
assert old_raw.replace("把第 15 欄整欄".encode(), "把欄索引 15（從 0 起算）整欄".encode()) == new_raw, "變動超過預期的索引措辭"

reuse = []
for kind, values in [("artifact", original["artifacts"]), ("repository_code", [s for s in original["sources"] if s["kind"] == "repository_code"])]:
    for item in values:
        actual = digest(ROOT / item["path"])
        reuse.append({"kind": kind, "id": item["id"], "path": item["path"], "expected_sha256": item["sha256"], "current_sha256": actual, "unchanged": actual == item["sha256"]})
assert all(item["unchanged"] for item in reuse), "舊證據與現行來源出現變動"

function_source = inspect.getsource(thin_stroke_report)
function_tree = ast.parse(function_source)
assign = next(n for n in ast.walk(function_tree) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript))
column = ast.literal_eval(assign.targets[0].slice.elts[-1])
image = torch.zeros(1, 1, 32, 32)
image[:, :, :, column] = 1
small = F.interpolate(image, (4, 4), mode="area")
manual = image.reshape(1, 1, 4, 8, 4, 8).mean((3, 5))
other = torch.zeros_like(image)
other[:, :, :, 14] = 1
assert column == 15, "亮線欄索引與正文不同"
assert torch.equal(small, manual), "面積縮圖與逐區平均不同"
assert torch.equal(small, F.interpolate(other, (4, 4), mode="area")), "縮圖碰撞沒有重現"
report = thin_stroke_report()
assert report == {"original_shape": [32, 32], "small_shape": [4, 4], "original_brightest": 1.0, "small_brightest": 0.125, "pixels_at_least_half_bright_before": 32, "pixels_at_least_half_bright_after": 0}, "函式數值與預期不同"

run = Path(sys.argv[1]).resolve()
assert run.parent == ROOT / "outputs/reviewer-tools/runs", "必須明確指名本次工具執行目錄"
files = {name: {"sha256": digest(run / name), "content": (run / name).read_text()} for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json")}
execution = json.loads(files["execution.json"]["content"])
assert execution["exit_code"] == 0, "原文程式實跑失敗"
assert execution["source_sha256"] == hashlib.sha256(new_raw).hexdigest(), "執行的原文版本不同"
figure = ROOT / "course/figures/practical_stroke.svg"
png = ROOT / "outputs/course-review-20261019/figure-renders-04/3e4a4ddedd9558254a44c3a87f09a45f0f6abba45d565f6f67a910869a75901f.png"
assert digest(figure) == original["figure_sha256"]["course/figures/practical_stroke.svg"], "圖檔版本不同"
assert digest(png) == "36ce5f578cde88537d5543e4772c362f42b19e0abf694afd7055fb6693cb2a31", "圖的算繪版本不同"
old_probe = json.loads((HERE / "verification.json").read_text())
functional_hash = digest(Path(F.__file__))
assert functional_hash == old_probe["installed_functional_file_sha256"], "安裝的 PyTorch 程式來源版本不同"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + str(run.relative_to(ROOT)),
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(image.dtype), "platform": platform.platform()},
    "source_sha256": hashlib.sha256(new_raw).hexdigest(),
    "source_freeze_sha256": digest(freeze),
    "current_chapter_matches_frozen_section": new_raw == current_raw,
    "only_change_is_explicit_zero_based_column_index": True,
    "diff": "".join(difflib.unified_diff(old_raw.decode().splitlines(keepends=True), new_raw.decode().splitlines(keepends=True), fromfile=str(old_source.relative_to(ROOT)), tofile=str(new_source.relative_to(ROOT)))),
    "original_report_sha256": digest(original_path),
    "reused_evidence": reuse,
    "function_source": function_source,
    "column_assignment_index_from_ast": column,
    "bright_column_indices_zero_based": image[0, 0].sum(0).nonzero().flatten().tolist(),
    "bright_column_ordinals_one_based": (image[0, 0].sum(0).nonzero().flatten() + 1).tolist(),
    "report": report,
    "small_matrix": small[0, 0].tolist(),
    "area_matches_block_mean_exactly": True,
    "different_column_same_small_image": True,
    "installed_functional_file_sha256": functional_hash,
    "figure_sha256": digest(figure),
    "viewed_png": {"path": str(png.relative_to(ROOT)), "sha256": digest(png)},
    "section_execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/11.md#11.15\" --execute", "files": files},
}
with (HERE / "verification-current-02.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"source_sha256": result["source_sha256"], "only_change_is_explicit_zero_based_column_index": True, "all_reused_evidence_unchanged": True, "column_assignment_index_from_ast": column, "bright_column_indices_zero_based": result["bright_column_indices_zero_based"], "report": report, "section_execution_exit_code": execution["exit_code"]}, ensure_ascii=False))
