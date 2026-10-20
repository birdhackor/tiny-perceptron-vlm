"""複查來源變動、工具預設裝置與既有證據；不訓練模型。"""

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    return sha((ROOT / path).read_bytes())


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    return {"argv": [str(value) for value in argv], "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def section(raw, identifier):
    matches = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, match in enumerate(matches) if match[0].startswith(b"## " + identifier.encode() + b" "))
    return raw[matches[index].start():matches[index + 1].start() if index + 1 < len(matches) else len(raw)]


original_path = "docs/course-review-20261019/reviews/rv1019_technical_9-2/technical-9-2-current-01.json"
original_raw = (ROOT / original_path).read_bytes()
assert sha(original_raw) == "ebac1e6fd200bd18daeb8a2492e7f87d59a174117f2a965e18c0fbd1ac6f55fb", "前份報告內容改變"
original = json.loads(original_raw)
old_checks = []
for artifact in original["artifacts"]:
    actual = digest(artifact["path"])
    assert actual == artifact["sha256"], "既有證據內容改變"
    old_checks.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
source_checks = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        actual = digest(source["path"])
        source_checks.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
        if source["path"] != "course/training.md":
            assert actual == source["sha256"], "預期不變的程式來源發生變動"

frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/9.2.md").read_bytes()
current = section((ROOT / "course/chapters/09.md").read_bytes(), "9.2")
assert current == frozen, "現行小節與凍結原文不同"
assert sha(frozen) == "458a7282ac8230f51fbca22f3e0f8d73d18d5bdc4e8a5c0ea237756d28966e71", "小節凍結雜湊不符"
freeze_path = "docs/course-review-20261019/freeze-07.json"
assert digest(freeze_path) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753", "複查凍結檔雜湊不符"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(page for page in freeze["pages"] if page["page_id"] == "9.2")
assert page["source_sha256"] == sha(frozen) and page["figures_sha256"] == {}, "凍結頁面記錄不符"

training_diff = command(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
assert training_diff["exit_code"] == 0, "無法讀取指定差異"
before = subprocess.check_output(["git", "show", "ecf6dec71280:course/training.md"], cwd=ROOT)
after = subprocess.check_output(["git", "show", "c6b9339d2595:course/training.md"], cwd=ROOT)
training = (ROOT / "course/training.md").read_bytes()
assert sha(before) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94", "指定舊版本與前份報告不一致"
assert after == training, "指定新版本與現行訓練頁不一致"
assert sha(training) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b", "現行訓練頁雜湊不符"
old_recipe = section(before, "T.5")
new_recipe = section(training, "T.5")
assert old_recipe == new_recipe, "引用的固定實驗配方發生變動"
safety_command = ".venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda"
assert safety_command in new_recipe.decode(), "固定安全實驗命令缺失"

from scripts.course_experiments import run as runner

parsed_devices = []
for arguments in (["--experiment", "safety"], ["--experiment", "safety", "--device", "cuda"]):
    def fake_execute(experiment_id, device, *args, **kwargs):
        parsed_devices.append({"arguments": arguments, "experiment_id": experiment_id, "device": device})
        return {"experiment_id": experiment_id, "elapsed_seconds": 0, "evidence_status": "parser_probe", "artifacts": []}
    with patch.object(sys, "argv", ["scripts.course_experiments.run", *arguments]), patch.object(runner, "execute", side_effect=fake_execute), contextlib.redirect_stdout(io.StringIO()):
        runner.main()
assert parsed_devices[0]["device"] == "cpu", "實驗工具的預設裝置不是CPU"
assert parsed_devices[1]["device"] == "cuda", "明示CUDA參數沒有傳入執行入口"

section_result = command([sys.executable, "docs/review-tools/section_facts.py", "course/chapters/09.md#9.2", "--execute"])
assert section_result["exit_code"] == 0, "小節程式執行失敗"
summary = json.loads(section_result["stdout"])
run_directory = Path(summary["artifact_directory"])
assert run_directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs"), "工具輸出位置不符"
copied = {}
for name in ("execution.json", "environment.json", "extraction.json", "fence-1.py", "stdout.txt", "stderr.txt"):
    raw = (run_directory / name).read_bytes()
    copied[name] = {"sha256": sha(raw), "content": raw.decode()}
assert copied["stdout.txt"]["content"] == "可見 3 → 3\n可見 None → 看不到球數，請提供數量或圖片。\n", "小節輸出不符"
assert copied["stderr.txt"]["content"] == "", "小節執行出現錯誤輸出"

verification = command([sys.executable, "docs/technical-reviews/artifacts/review-20261019-9-2/verify.py"])
assert verification["exit_code"] == 0, "既有獨立驗證程式失敗"
verification_data = json.loads(verification["stdout"])
previous_data = json.loads((BASE / "verification.json").read_bytes())
for key in ("rule", "saved_observation", "reconstructed_data", "mixed_training_records", "flow_probe", "historical_source", "cli"):
    assert verification_data[key] == previous_data[key], "既有主張的核對結果發生變動"
for path, expected in previous_data["input_sha256"].items():
    if path != "course/training.md":
        assert digest(path) == expected, "原實驗輸入或程式內容改變"
assert verification_data["input_sha256"]["course/training.md"] == sha(training), "複查執行未記下現行來源雜湊"
for artifact in original["artifacts"]:
    assert digest(artifact["path"]) == artifact["sha256"], "複查執行改變了既有證據"
assert digest(original_path) == sha(original_raw), "複查執行改變了前份報告"

assert digest(original["original_report"]["path"]) == original["original_report"]["sha256"], "初始報告內容改變"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-2/recheck-current-02.py",
    "result": "全部斷言通過；前份報告、既有證據與未改主張的核對結果均維持一致。",
    "environment": verification_data["environment"],
    "previous_report": {"path": original_path, "sha256": sha(original_raw), "issues": original["issues"]},
    "source_checks": source_checks,
    "artifact_checks": old_checks,
    "source_freeze": {"path": freeze_path, "sha256": digest(freeze_path)},
    "page_sha256": sha(current),
    "changed_page_sentences": [],
    "training_diff": training_diff,
    "training_recipe": {"old_sha256": sha(old_recipe), "current_sha256": sha(new_recipe), "unchanged": True, "safety_command": safety_command, "text": new_recipe.decode()},
    "unchanged_relevant_source_sentence": {"text": "`scripts.course_experiments.run` 沒指定時用 CPU。", "status": "verified", "support": "原CLI的main經實際argparse解析，execute以替身接收參數；沒有執行訓練。", "observed": parsed_devices},
    "out_of_scope_diff": "差異只有多模態練習的抽查、驗證題與文字能力評估說明；不是本頁所引安全配方的依據，未將這些段落當成本頁的新主張。",
    "section_execution": {"command_result": section_result, "summary": summary, "copied_files": copied},
    "independent_verification": verification,
    "unchanged_verification_fields": ["rule", "saved_observation", "reconstructed_data", "mixed_training_records", "flow_probe", "historical_source", "cli"],
    "read_exception": "只讀取本次section_facts所產生、由其stdout明確指名的執行目錄；執行結果全文複製於本檔。沒有讀取其他outputs或其他角色報告。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
