"""重新執行現行教材與原補驗，保存差異及證據延用的核對結果。"""

import contextlib
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-16-5"
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_16-5/technical-16-5-original-01.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(command):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return {"command_argv": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def simple_class(text):
    return next(code for code in re.findall(r"```python\n(.*?)```", text, flags=re.S) if "class CausalAttention" in code)


assert digest(ORIGINAL.read_bytes()) == "f6f9c2d8d05f74ef90568c74404a0014e73b5b2fc07dcfb5b81cd3dad3ade582"
original = json.loads(ORIGINAL.read_bytes())
assert original["issues"] == []
old_chapter_receipt = run(["git", "show", "5ef56bc756f3:course/chapters/03.md"])
new_chapter_receipt = run(["git", "show", "3af7f13ef959:course/chapters/03.md"])
old_chapter = old_chapter_receipt["stdout"]
new_chapter = new_chapter_receipt["stdout"]
current_chapter = (ROOT / "course/chapters/03.md").read_text()
assert new_chapter == current_chapter
assert digest(old_chapter.encode()) == "df0709c102dba36fa89b3c88904f55172339a61c15c4a0583484b9b919bd2ee7"
assert digest(new_chapter.encode()) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
old_sentence = "d=4 時，未縮放標準差應靠近 2；d=64 時靠近 8。縮放後兩者都靠近 1，樣本有限所以不會恰好等於理論值。這次是檢查尺度推算，沒有做完整匹配表或訓練。"
new_sentence = "d=4 時，未縮放標準差應靠近 2；d=64 時靠近 8。縮放後兩者都靠近 1；有限樣本通常有抽樣誤差，所以不必恰好等於理論值。這次是檢查尺度推算，沒有做完整匹配表或訓練。"
assert old_chapter.count(old_sentence) == 1
assert old_chapter.replace(old_sentence, new_sentence) == current_chapter
old_class = simple_class(old_chapter)
current_class = simple_class(current_chapter)
assert old_class == current_class
assert "torch.ones(t, t, dtype=torch.bool).tril()[None, None]" in current_class
assert "attention_mask(" not in current_class

source_hashes = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    current_hash = digest((ROOT / source["path"]).read_bytes())
    changed = current_hash != source["sha256"]
    assert changed == (source["path"] == "course/chapters/03.md")
    source_hashes.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current_hash, "changed": changed})
artifact_hashes = []
for artifact in original["artifacts"]:
    current_hash = digest((ROOT / artifact["path"]).read_bytes())
    assert current_hash == artifact["sha256"]
    artifact_hashes.append({"path": artifact["path"], "sha256": current_hash, "unchanged": True})

section_command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/16.md#16.5", "--execute"]
section_receipt = run(section_command)
section_facts = json.loads(section_receipt["stdout"])
assert section_facts["source_sha256"] == original["source_sha256"]
assert section_facts["execution_exit_code"] == 0
section_directory = Path(section_facts["artifact_directory"])
assert section_directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
files = {}
for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py", "fence-2.py"]:
    raw = (section_directory / name).read_bytes()
    files[name] = {"sha256": digest(raw), "content": raw.decode()}
assert json.loads(files["execution.json"]["content"])["exit_code"] == 0
assert json.loads(files["extraction.json"]["content"])["figure_sha256"] == original["figure_sha256"]

probe_receipt = run([str(ROOT / ".venv/bin/python"), str(BASE / "verify.py")])
probe = json.loads(probe_receipt["stdout"])
assert probe["attention_routes"]["simple_class_uses_tril_only"] is True
assert probe["attention_routes"]["simple_class_source_sha256"] == digest(current_chapter.encode())
previous_probe = json.loads((BASE / "verification.json").read_bytes())
assert probe["empirical_record"] == previous_probe["empirical_record"]
assert probe["mask"] == previous_probe["mask"]
assert probe["cache"] == previous_probe["cache"]
assert probe["packing_targets"] == previous_probe["packing_targets"]

import torch
scaling_code = next(code for code in re.findall(r"```python\n(.*?)```", current_chapter, flags=re.S) if "for d in [4, 64]:" in code)
scaling_stdout = io.StringIO()
with contextlib.redirect_stdout(scaling_stdout):
    exec(compile(scaling_code, "course/chapters/03.md:scaled-dot-product", "exec"), {})
scaling_rows = []
for row in scaling_stdout.getvalue().splitlines():
    dimension, raw_std, scaled_std = row.split()
    dimension, raw_std, scaled_std = int(dimension), float(raw_std), float(scaled_std)
    scaling_rows.append({"dimension": dimension, "sample_size": 1000, "unscaled_sample_std": raw_std, "scaled_sample_std": scaled_std, "theoretical_unscaled_std": math.sqrt(dimension), "theoretical_scaled_std": 1.0, "unscaled_relative_error": abs(raw_std / math.sqrt(dimension) - 1), "scaled_absolute_error": abs(scaled_std - 1)})
assert [row["dimension"] for row in scaling_rows] == [4, 64]

figure_path = ROOT / "course/figures/rewrite-16-packed-mask.svg"
assert digest(figure_path.read_bytes()) == original["figure_sha256"][figure_path.relative_to(ROOT).as_posix()]
png = ROOT / "outputs/course-review-20261019/figure-renders-04/4285aeae3f78a1aeba22140c6e3bf828e25b0a45b696eadeaf88994054488b59.png"
assert digest(png.read_bytes()) == "d21876240412c93803ab16c84956a46c7ae9b0e55a9e17d1be13e6863a5aa53a"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-5/recheck-current-01.py",
    "environment": probe["environment"],
    "result": "全部斷言通過；現行小節原始程式與原補驗重新實跑成功。",
    "original_report": {"path": ORIGINAL.relative_to(ROOT).as_posix(), "sha256": digest(ORIGINAL.read_bytes()), "issues": original["issues"]},
    "repository_source_hashes": source_hashes,
    "reused_artifact_hashes": artifact_hashes,
    "dependency_change": {
        "diff": run(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/03.md"]),
        "old_source_command": old_chapter_receipt["command_argv"],
        "new_source_command": new_chapter_receipt["command_argv"],
        "old_source_sha256": digest(old_chapter.encode()),
        "new_source_sha256": digest(new_chapter.encode()),
        "single_sentence_replacement_only": True,
        "old_sentence": old_sentence,
        "new_sentence": new_sentence,
        "changed_sentence_status": "verified",
        "changed_sentence_derivation": "每格 q_i*k_i 為獨立、等機會 ±1，平均 0、變異數 1；D 格加總變異數為 D，所以母體標準差為 sqrt(D)，除 sqrt(D) 後母體標準差為 1。有限抽樣的 std 是隨機統計量，沒有必定等於母體標準差的約束；改句使用通常有抽樣誤差／不必恰好，符合這個條件。實跑顯示本次 1000 組樣本的實際差異，不能用此單次數字斷言所有有限樣本必不相等。",
        "changed_sentence_execution": {"source_code": scaling_code, "source_sha256": digest(scaling_code.encode()), "stdout": scaling_stdout.getvalue(), "rows": scaling_rows},
        "causal_attention_class_unchanged": True,
        "class_sha256": digest(current_class.encode()),
        "current_class_source": current_class,
        "impact": "變動位在縮放點積的有限抽樣說明；本頁所引用的精簡 CausalAttention 完全未變，直接使用 tril 並省略 attention_mask 的主張仍成立。",
    },
    "section_execution": {"receipt": section_receipt, "facts": section_facts, "files": files},
    "probe_execution": {"receipt": {key: value for key, value in probe_receipt.items() if key != "stdout"}, "result": probe},
    "figure": {"svg_sha256": digest(figure_path.read_bytes()), "png_sha256": digest(png.read_bytes()), "inspection": "已用 view_image 重看指定 PNG，Query 列、Key 欄、四個位置與隔離矩陣一致。PNG 不列 artifact。"},
    "limitations": ["未重跑 L4 訓練或私有 checkpoint；沿用並重核公開實驗紀錄與相符程式。", "公開 packing 紀錄無逐步 latency，無法獨立重算原始中位數。"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
