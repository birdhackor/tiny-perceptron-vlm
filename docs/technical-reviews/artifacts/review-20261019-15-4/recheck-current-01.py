"""複查凍結小節、沿用證據、MoE 欄位與 Dense 空值，並保存本次執行收據。"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from scripts.course_experiments.architecture import _routing
from tiny_perceptron.model import ModelConfig, TinyLM

BASE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_15-4/technical-15-4-original-01.json"
OLD_SOURCE = ROOT / "docs/course-review-20261019/freeze-01/sources/15.4.md"
NEW_SOURCE = ROOT / "docs/course-review-20261019/freeze-05/sources/15.4.md"
FREEZE = ROOT / "docs/course-review-20261019/freeze-05.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hash(path, expected):
    observed = sha(path)
    assert observed == expected, "檔案 SHA 與預期證據不符"
    return {"path": path.relative_to(ROOT).as_posix(), "expected": expected, "observed": observed, "equal": True}


original = json.loads(ORIGINAL.read_bytes())
old_evidence = json.loads((BASE / "verification.json").read_bytes())
original_checks = [checked_hash(ORIGINAL, "694f2cd590b4517a4ecdab5f3cd70686d7d4e6bd25cb5f37fb0f2d6b64a0a178")]
for artifact in original["artifacts"]:
    original_checks.append(checked_hash(ROOT / artifact["path"], artifact["sha256"]))
original_checks.append(checked_hash(BASE / "self-check.txt", "90c5029f8ec49da124a3497c59c821667e20aacd10fb1a8e5b1282995f17865f"))
repository_checks = [checked_hash(ROOT / source["path"], source["sha256"]) for source in original["sources"] if source["kind"] == "repository_code"]
for path, expected in old_evidence["repository_files"].items():
    repository_checks.append(checked_hash(ROOT / path, expected))
freeze_checks = [
    checked_hash(OLD_SOURCE, "0f3270661b7233909b938b556384c99d91a1c2d3496a5fb1aba1bc330a9dfac9"),
    checked_hash(NEW_SOURCE, "3ba0675a3bea0b0be372c1be1c96b64b5c5d920b9348708e16985f29845e7143"),
    checked_hash(FREEZE, "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"),
]
old_phrase = "`results.variants` 各組的 `validation_routing.layers` 裡，"
new_phrase = "`results.variants` 各個 MoE 組的 `validation_routing.layers` 裡（Dense 對照組這一欄是 null），"
old_raw = OLD_SOURCE.read_bytes()
new_raw = NEW_SOURCE.read_bytes()
assert old_raw.count(old_phrase.encode()) == 1, "應恰有一處待複查片語"
assert old_raw.replace(old_phrase.encode(), new_phrase.encode()) == new_raw, "尚有其他來源差異待審"
chapter = (ROOT / "course/chapters/15.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
index = next(i for i, header in enumerate(headers) if header[0].startswith(b"## 15.4 "))
end = headers[index + 1].start() if index + 1 < len(headers) else len(chapter)
assert chapter[headers[index].start():end] == new_raw, "現行小節與指定凍結不同"
diff_command = ["git", "diff", "--no-index", str(OLD_SOURCE.relative_to(ROOT)), str(NEW_SOURCE.relative_to(ROOT))]
diff = subprocess.run(diff_command, cwd=ROOT, text=True, capture_output=True)
assert diff.returncode == 1, "預期 git diff 回報來源有差異"

command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/15.md#15.4", "--execute"]
environment = os.environ.copy()
environment["PYTHONDONTWRITEBYTECODE"] = "1"
completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=True)
summary = json.loads(completed.stdout)
assert summary["source_sha256"] == freeze_checks[1]["observed"], "工具執行來源與凍結不同"
assert summary["python_fences"] == 1 and summary["svg_references"] == 0, "程式或圖的數目改變"
assert summary["execution_exit_code"] == 0, "原小節程式執行失敗"
run = Path(summary["artifact_directory"])
section = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/15.md#15.4\" --execute", "command_stdout": completed.stdout, "command_stderr": completed.stderr, "summary": summary}
for name in ("extraction.json", "execution.json", "environment.json"):
    section[name] = json.loads((run / name).read_bytes())
for name in ("stdout.txt", "stderr.txt", "fence-1.py"):
    section[name] = (run / name).read_text()
old_section = json.loads((BASE / "section-execution.json").read_bytes())
assert section["fence-1.py"] == old_section["fence-1.py"], "程式區塊已有變動"
assert section["stdout.txt"] == old_section["stdout.txt"], "本次輸出與原審不同"
assert section["stderr.txt"] == "", "原程式有 stderr 待核對"
assert str(torch.__version__) == old_evidence["environment"]["torch"], "torch 版本改變，需重新審查 API 與算例"
assert str(torch.version.git_version) == old_evidence["environment"]["torch_git_version"], "torch commit 改變"

experiment_path = ROOT / "docs/course-experiments/results/moe.json"
experiment = json.loads(experiment_path.read_bytes())
variants = []
for name, variant in experiment["results"]["variants"].items():
    config = variant["model"]["config"]
    assert "validation_routing" in variant, "每組應有 validation_routing 欄位"
    routing = variant["validation_routing"]
    record = {"name": name, "experts": config["experts"], "top_k": config["top_k"], "validation_routing": routing}
    if config["experts"] == 0:
        assert routing is None, "Dense 對照組的 validation_routing 應是 null"
        record["layers_field_exists"] = False
    else:
        assert isinstance(routing, dict) and len(routing["layers"]) == config["layers"] == 2, "MoE 應有兩層統計"
        assert config["experts"] == 4 and routing["effective_input_tokens"] == 39256, "MoE 配置或有效位置數不符"
        for layer in routing["layers"]:
            assert len(layer["dispatch_counts"]) == 4, "每層應有四位 expert 的完整次數"
            denominator = layer["dispatch_denominator"]
            assert denominator == sum(layer["dispatch_counts"]) == 39256 * config["top_k"], "每層分派分母不符"
            assert max(abs(count / denominator - fraction) for count, fraction in zip(layer["dispatch_counts"], layer["load_fraction"], strict=True)) < 1e-15, "每層分派份額不符"
        record["layers_field_exists"] = True
    variants.append(record)
assert len([entry for entry in variants if entry["experts"] == 0]) == 3, "Dense 對照組數與本次讀取內容不符"
assert len([entry for entry in variants if entry["experts"] > 0]) == 4, "MoE 組數與本次讀取內容不符"

torch.set_num_threads(1)
torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8, layers=2, heads=2, max_length=8, experts=0))
dense_result = _routing(model, [], SimpleNamespace(device="cpu"))
assert dense_result is None, "無 expert 的模型不應產生分派統計"
source_text = (ROOT / "scripts/course_experiments/architecture.py").read_text()
start = source_text.index("def _routing(")
stop = source_text.index("    counts =", start)
dense_probe = {"model": {"width": 8, "layers": 2, "heads": 2, "experts": 0}, "result": dense_result, "json_serialization": json.dumps(dense_result), "source_path": "scripts/course_experiments/architecture.py", "source_identifier": "_routing", "source_excerpt": source_text[start:stop]}

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-4/recheck-current-01.py",
    "checked_at": datetime.now(UTC).isoformat(),
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "platform": sys.platform, "device": "cpu"},
    "original_report_issues": original["issues"],
    "existing_evidence_integrity": original_checks,
    "repository_source_integrity": repository_checks,
    "freeze_integrity": freeze_checks,
    "diff": {"command": diff_command, "returncode": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr, "changed_phrase_before": old_phrase, "changed_phrase_after": new_phrase, "other_source_bytes_unchanged": True},
    "section_execution": section,
    "fence_and_stdout_equal_original": True,
    "current_json_check": {"path": experiment_path.relative_to(ROOT).as_posix(), "sha256": sha(experiment_path), "variants": variants, "dense_return_probe": dense_probe, "interpretation": "MoE 組的 validation_routing 是含 layers 的物件；Dense 組的 validation_routing 本身為 null，沒有 layers 子欄位。"},
    "carried_forward_evidence": {"path": "docs/technical-reviews/artifacts/review-20261019-15-4/verification.json", "sha256": sha(BASE / "verification.json"), "scope": "未變動主張沿用原始來源、練習、梯度、稀疏 dispatch、PAD 與資料重建證據；本次逐一確認既有 artifact 與依賴原檔 SHA 相同，原 fence 並重新實跑。"},
    "result": "MoE 路由欄位、Dense 空值、原小節執行與既有證據完整性全部核對通過",
}
for saved in original_checks:
    checked_hash(ROOT / saved["path"], saved["expected"])
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
