"""核對依賴教材的矩陣算例，重新執行遮罩與資料邊界驗證。"""

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-16-5"
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_16-5/technical-16-5-current-01.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(command):
    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, check=True)
    return {"command_argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def find_class(text):
    return next(code for code in re.findall(r"```python\n(.*?)```", text, flags=re.S) if "class CausalAttention" in code)


previous = json.loads(PREVIOUS.read_bytes())
assert digest(PREVIOUS.read_bytes()) == "7b269d9f3a6b19c9d4f2b948b9141db3d588943f82ca07e9c0c4c9e75363db49"
assert previous["issues"] == []
original_path = ROOT / previous["original_report"]["path"]
assert digest(original_path.read_bytes()) == previous["original_report"]["sha256"]
old_command = ["git", "show", "ecf6dec71280:course/chapters/03.md"]
new_command = ["git", "show", "c6b9339d2595:course/chapters/03.md"]
old = run(old_command)["stdout"]
new = run(new_command)["stdout"]
current = (ROOT / "course/chapters/03.md").read_text()
assert new == current
assert digest(old.encode()) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
assert digest(new.encode()) == "5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5"
old_clause = "各位置向量長度相同時，跟自己的分數也最高（或並列最高）"
new_clause = "本例 A、B 跟自己都得 1、彼此得 0，所以最高分都落在自己"
assert old.count(old_clause) == 1
assert old.replace(old_clause, new_clause) == current
old_class, current_class = find_class(old), find_class(current)
assert old_class == current_class
assert "torch.ones(t, t, dtype=torch.bool).tril()[None, None]" in current_class
assert "attention_mask(" not in current_class

source_checks = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        current_sha = digest((ROOT / source["path"]).read_bytes())
        changed = current_sha != source["sha256"]
        assert changed == (source["path"] == "course/chapters/03.md")
        source_checks.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current_sha, "changed": changed})
artifact_checks = []
for artifact in previous["artifacts"]:
    current_sha = digest((ROOT / artifact["path"]).read_bytes())
    assert current_sha == artifact["sha256"]
    artifact_checks.append({"path": artifact["path"], "sha256": current_sha, "unchanged": True})

import torch
matrix_code = next(code for code in re.findall(r"```python\n(.*?)```", current, flags=re.S) if 'print("共用一份"' in code)
namespace = {"torch": torch}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(matrix_code, "course/chapters/03.md:query-key-example", "exec"), namespace)
has, need = namespace["has"], namespace["need"]
shared, separate = has @ has.T, need @ has.T
assert shared.tolist() == [[1.0, 0.0], [0.0, 1.0]]
assert separate.tolist() == [[0.0, 1.0], [1.0, 0.0]]
assert shared.argmax(dim=-1).tolist() == [0, 1]
assert torch.equal(shared, shared.T)

section_receipt = run([str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/16.md#16.5", "--execute"])
section_facts = json.loads(section_receipt["stdout"])
assert section_facts["source_sha256"] == previous["source_sha256"]
assert section_facts["execution_exit_code"] == 0
run_directory = Path(section_facts["artifact_directory"])
assert run_directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
section_files = {}
for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py", "fence-2.py"]:
    raw = (run_directory / name).read_bytes()
    section_files[name] = {"sha256": digest(raw), "content": raw.decode()}
assert json.loads(section_files["extraction.json"]["content"])["figure_sha256"] == previous["figure_sha256"]
assert json.loads(section_files["execution.json"]["content"])["exit_code"] == 0

probe_receipt = run([str(ROOT / ".venv/bin/python"), str(BASE / "verify.py")])
probe = json.loads(probe_receipt["stdout"])
assert probe["attention_routes"]["simple_class_source_sha256"] == digest(current.encode())
assert probe["attention_routes"]["simple_class_uses_tril_only"] is True
previous_execution = json.loads((BASE / "recheck-current-01.json").read_bytes())
previous_probe = previous_execution["probe_execution"]["result"]
for field in ["mask", "cache", "eos_and_positions", "packing_targets", "empirical_record", "original_experiment_code"]:
    assert probe[field] == previous_probe[field]
svg = ROOT / "course/figures/rewrite-16-packed-mask.svg"
assert digest(svg.read_bytes()) == previous["figure_sha256"][svg.relative_to(ROOT).as_posix()]
png = ROOT / "outputs/course-review-20261019/figure-renders-04/4285aeae3f78a1aeba22140c6e3bf828e25b0a45b696eadeaf88994054488b59.png"
assert digest(png.read_bytes()) == "d21876240412c93803ab16c84956a46c7ae9b0e55a9e17d1be13e6863a5aa53a"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-5/recheck-current-02.py",
    "environment": probe["environment"],
    "result": "全部斷言通過；現行遮罩程式、精簡類別、資料目標與公開實驗數字核對均通過。",
    "previous_report": {"path": PREVIOUS.relative_to(ROOT).as_posix(), "sha256": digest(PREVIOUS.read_bytes()), "issues": previous["issues"]},
    "original_report": {"path": original_path.relative_to(ROOT).as_posix(), "sha256": digest(original_path.read_bytes())},
    "repository_source_hashes": source_checks,
    "reused_artifact_hashes": artifact_checks,
    "dependency_change": {
        "diff": run(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/chapters/03.md"]),
        "old_source_command": old_command,
        "new_source_command": new_command,
        "old_source_sha256": digest(old.encode()),
        "new_source_sha256": digest(new.encode()),
        "single_clause_replacement_only": True,
        "old_clause": old_clause,
        "new_clause": new_clause,
        "status": "verified",
        "derivation": "本例 a=(1,0)、b=(0,1)。a·a=1²+0²=1，b·b=0²+1²=1，a·b=b·a=1×0+0×1=0。因此共用 Q/K 的 Gram 矩陣為 [[1,0],[0,1]]，各列的最高分都落在自己。使用 need 作 Query 時矩陣改為 [[0,1],[1,0]]。實數點積的交換律給出一般 Gram 矩陣的對稱性；改句對自己最高的敘述明確限於本例，不再依賴向量長度一詞的解讀。",
        "matrix_execution": {"source_code": matrix_code, "source_sha256": digest(matrix_code.encode()), "stdout": printed.getvalue(), "has": has.tolist(), "need": need.tolist(), "shared_scores": shared.tolist(), "separate_scores": separate.tolist(), "shared_row_argmax": shared.argmax(dim=-1).tolist(), "dtype": str(has.dtype), "device": str(has.device)},
        "causal_attention_class_unchanged": True,
        "class_sha256": digest(current_class.encode()),
        "current_class_source": current_class,
        "impact": "只改動 Query／Key 算例的一個敘述。本頁所引用的精簡 CausalAttention 完全未變，本次執行仍直接使用 tril 而沒有呼叫 attention_mask。",
    },
    "section_execution": {"receipt": section_receipt, "facts": section_facts, "files": section_files},
    "probe_execution": {"receipt": {key: value for key, value in probe_receipt.items() if key != "stdout"}, "result": probe},
    "figure": {"svg_sha256": digest(svg.read_bytes()), "png_sha256": digest(png.read_bytes()), "inspection": "本次已用 view_image 看指定 PNG；Query 列、Key 欄及 [1000;1100;0010;0011] 與執行結果一致。PNG 不列 artifact。"},
    "limitations": ["未重跑 L4 訓練或查驗私有 checkpoint；核對公開結果與相符程式。", "原公開結果無逐步 latency 樣本，無法獨立重算歷史中位數。"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
