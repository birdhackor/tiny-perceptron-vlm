"""核對章檔變動、沿用證據與修改後的具體句子。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import ModelConfig, TinyLM

ARTIFACT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_16-13/technical-16-13-original-01.json"
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T084436017225Z"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run_git(*arguments):
    argv = ["git", *arguments]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    return result.stdout


def sections(raw):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    return {
        header.group().decode(): raw[header.start():headers[i + 1].start() if i + 1 < len(headers) else len(raw)]
        for i, header in enumerate(headers)
    }


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def write_exclusive(name, value):
    path = ARTIFACT / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path.read_bytes())}


original = json.loads(REPORT.read_bytes())
check(digest(REPORT.read_bytes()) == "f94f062bd1cd2074cc9161dd166946394c295ae05740b690cf762c9e0e33905d", "原始報告雜湊不符")
reused = []
for item in original["artifacts"]:
    actual = digest((ROOT / item["path"]).read_bytes())
    check(actual == item["sha256"], "原始 artifact 內容已變動")
    reused.append({"path": item["path"], "sha256": actual, "matches": True})
check(original["issues"] == [], "原始報告的問題清單與預期不同")

chapter = ROOT / "course/chapters/16.md"
current_raw = chapter.read_bytes()
base_raw = run_git("show", "5ef56bc756f3:course/chapters/16.md")
revised_raw = run_git("show", "3af7f13ef959:course/chapters/16.md")
check(digest(base_raw) == "36df4c1312242eb13942e4b653a6762da66d37b275b7cbf2c0b2929efdf4f2c7", "差異起點與原始報告章檔不一致")
check(current_raw == revised_raw, "現行章檔與指定差異終點不一致")
check(digest(current_raw) == "5c53a5678eaecab7ec75a683807da1bf4aafc511fde465b334b8e134cda391e3", "現行章檔雜湊不符")
base_sections, current_sections = sections(base_raw), sections(current_raw)
changed_titles = [title for title, raw in current_sections.items() if base_sections.get(title) != raw]
unchanged = []
for title in current_sections:
    if any(title.startswith("## " + lesson + " ") for lesson in ["16.5", "16.9", "16.12", "16.13"]):
        same = base_sections[title] == current_sections[title]
        check(same, "受審內容或必要前置與原版本不同")
        unchanged.append({"title": title, "sha256": digest(current_sections[title]), "byte_identical": same})
review_title = next(title for title in current_sections if title.startswith("## 16.13 "))
frozen_path = ROOT / "docs/course-review-20261019/freeze-01/sources/16.13.md"
check(current_sections[review_title] == frozen_path.read_bytes(), "現行小節與凍結原文不同")

helper_files = {}
for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]:
    raw = (RUN / name).read_bytes()
    helper_files[name] = {"sha256": digest(raw), "bytes": len(raw), "content": raw.decode()}
execution = json.loads(helper_files["execution.json"]["content"])
extraction = json.loads(helper_files["extraction.json"]["content"])
check(execution["exit_code"] == 0, "原始小節執行失敗")
check(extraction["source_sha256"] == original["source_sha256"], "執行的小節版本不符")
check(extraction["source_file_sha256"] == digest(current_raw), "執行的章檔版本不符")
old_run = json.loads((ROOT / original["artifacts"][0]["path"]).read_bytes())
check(helper_files["stdout.txt"]["content"] == old_run["files"]["stdout.txt"]["content"], "原始程式輸出與前次不一致")

attention = ROOT / "tiny_perceptron/attention.py"
old_attention = next(s for s in original["sources"] if s["id"] == "repo-attention")
check(digest(attention.read_bytes()) == old_attention["sha256"], "注意力實作已變動")

# 使用課文相同 seed 與配置，直接比較輸入改動前後的快取。
torch.manual_seed(0)
model = TinyLM(ModelConfig(width=8)).eval()
with torch.no_grad():
    before = model(torch.tensor([[1, 2, 3]]))
    after = model(torch.tensor([[9, 2, 3]]))
    old_step = model(torch.tensor([[4]]), cache=before["cache"])
    full = model(torch.tensor([[9, 2, 3, 4]]))
kv_deltas = {
    name: (after["cache"][0][i] - before["cache"][0][i]).abs().amax(dim=(0, 1, 3)).tolist()
    for i, name in enumerate(["key", "value"])
}
check(len(model.blocks) == 1, "模型不是單層配置")
for values in kv_deltas.values():
    check(values[0] > 0 and values[1:] == [0.0, 0.0], "K/V 的變動位置與修訂句子不一致")
logit_error = (full["logits"][:, -1] - old_step["logits"][:, 0]).abs().max().item()
check(not torch.allclose(full["logits"][:, -1], old_step["logits"][:, 0], atol=1e-6), "更改前文後舊快取仍通過一致性檢查")

# 核對原始 BF16 算例的顯示，以及恰在中點時的 Python round。
a = torch.tensor([[1.001, 2.002], [3.003, 4.004]])
b = torch.tensor([[0.5, 1.0], [1.5, -1.0]])
full_precision = a @ b
with torch.autocast("cpu", dtype=torch.bfloat16):
    low_precision = a @ b
error = (full_precision - low_precision.float()).abs().max().item()
rounded = round(error, 4)
check(rounded == 0.0075, "精度算例的顯示數值不符")
ties = {str(value): round(value, 4) for value in [0.03125, 0.09375]}
check(ties == {"0.03125": 0.0312, "0.09375": 0.0938}, "Python round 中點行為不符")

sources = [chapter, attention, ROOT / "tiny_perceptron/model.py", ROOT / "docs/review-tools/section_facts.py"]
section_receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py 'course/chapters/16.md#16.13' --execute",
    "exit_code": 0,
    "copy_policy": "files.content 是本次工具輸出完整原文；雜湊依原始 bytes 計算。",
    "files": helper_files,
}
section_record = write_exclusive("section-run-current-01.json", section_receipt)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-13/recheck-current-01.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "status": "pass",
    "original_report": {"path": REPORT.relative_to(ROOT).as_posix(), "sha256": digest(REPORT.read_bytes()), "issues": original["issues"]},
    "current_commit": run_git("rev-parse", "HEAD").decode().strip(),
    "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in sources},
    "git_diff": {"command": "git diff 5ef56bc756f3 3af7f13ef959 -- course/chapters/16.md", "stdout": run_git("diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/16.md").decode(), "exit_code": 0},
    "changed_titles": changed_titles,
    "unchanged_sections": unchanged,
    "section_matches_freeze": True,
    "reused_artifacts": reused,
    "section_execution": section_record,
    "cache_sentence": {"layers": len(model.blocks), "seed": 0, "prefix_before": [1, 2, 3], "prefix_after": [9, 2, 3], "key_value_max_abs_delta_per_position": kv_deltas, "stale_cache_logit_max_abs_error": logit_error, "stale_cache_allclose_atol_1e_6": False, "status": "verified"},
    "round_sentence": {"python_api": "round(number, ndigits=4)", "input_dtype": str(a.dtype), "output_dtype": str(low_precision.dtype), "fp32": full_precision.tolist(), "bf16": low_precision.float().tolist(), "unrounded_max_error": error, "displayed_max_error": rounded, "ties_to_even_examples": ties, "status": "verified"},
    "limitations": "兩處改句在受審小節之外，只補驗指定差異；沒有完整重審它們所屬頁面。GPU 與模型品質未量測。",
}
evidence_record = write_exclusive("recheck-evidence-current-01.json", result)
print(json.dumps({"status": result["status"], "artifacts": [section_record, evidence_record], "cache_sentence": result["cache_sentence"], "round_sentence": result["round_sentence"]}, ensure_ascii=False))
