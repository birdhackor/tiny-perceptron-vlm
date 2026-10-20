import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition):
    if not condition:
        raise AssertionError("核對結果與預期不符")


def execute(argv):
    process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    require(process.returncode == 0)
    return {"command_argv": argv, "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}


def fixed_recipe(text):
    start = text.index("### 重跑第11章的固定實驗")
    end = text.index("這套 `vqa`", start)
    return text[start:end]


previous_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_9-9/technical-9-9-current-01.json"
require(digest(previous_path.read_bytes()) == "919b9412f119311119867e4bee92ed2c24783e5ab47d79fbe3dff33530e6b280")
previous = json.loads(previous_path.read_text())
require(previous["issues"] == [])
sources = []
artifacts = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        actual = digest((ROOT / source["path"]).read_bytes())
        sources.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
require([source["id"] for source in sources if not source["unchanged"]] == ["s_training"])
for artifact in previous["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    require(actual == artifact["sha256"])
    artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})

frozen = ROOT / "docs/course-review-20261019/freeze-05/sources/9.9.md"
require(digest(frozen.read_bytes()) == previous["source_sha256"] == "25112c3910c3199a4935f0d9df8124e7274e8dff38b08241d0c86fd614a21c92")
freeze = ROOT / "docs/course-review-20261019/freeze-07.json"
require(digest(freeze.read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753")
run = (ROOT / sys.argv[1]).resolve()
require(run.is_relative_to(ROOT / "outputs/reviewer-tools/runs"))
section = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/09.md#9.9" --execute'}
for name in ["environment.json", "extraction.json", "execution.json"]:
    section[name] = json.loads((run / name).read_text())
for name in ["stdout.txt", "stderr.txt", "fence-1.py"]:
    section[name] = (run / name).read_text()
require(section["extraction.json"]["source_sha256"] == previous["source_sha256"])
require(section["extraction.json"]["figure_sha256"] == {})
require(section["execution.json"]["exit_code"] == 0)
require(section["stdout.txt"] == "倍率 1.0 信心 0.6652 選擇 0 正確 False\n倍率 10.0 信心 1.0 選擇 0 正確 False\n")
old_section = json.loads((BASE / "section-execution.json").read_text())
require(section["fence-1.py"] == old_section["code"])

training_diff = execute(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
old_training = execute(["git", "show", "ecf6dec71280:course/training.md"])
new_training = execute(["git", "show", "c6b9339d2595:course/training.md"])
training = (ROOT / "course/training.md").read_text()
require(digest(old_training["stdout"].encode()) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94")
require(new_training["stdout"] == training)
require(digest(training.encode()) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b")
require(fixed_recipe(old_training["stdout"]) == fixed_recipe(training))
linked_command = ".venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda"
require(linked_command in fixed_recipe(training))
cli = execute([".venv/bin/python", "-m", "scripts.course_experiments.run", "--help"])
from scripts.course_experiments.run import experiment_spec
spec = experiment_spec("encoders")
require(spec["module"] == "modalities" and spec["function"] == "run_encoders")
numeric = execute([".venv/bin/python", str((BASE / "verify.py").relative_to(ROOT))])
numeric_result = json.loads(numeric.pop("stdout"))
require(numeric_result["chosen_temperature"] == 0.5)
round_cases = [{"number": n, "digits": d, "result": round(n, d)} for n, d in [(0.6652409434318542, 4), (0.9999545812606812, 4), (1.25, 1), (1.75, 1)]]
require([item["result"] for item in round_cases] == [0.6652, 1.0, 1.2, 1.8])
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + sys.argv[1],
    "result": "全部核對通過；頁面與原碼未變、舊證據完整、數值重算相符，現行文件的固定實驗段落與命令未變。",
    "environment": numeric_result["environment"],
    "previous_report": {"path": str(previous_path.relative_to(ROOT)), "sha256": digest(previous_path.read_bytes()), "issues": previous["issues"]},
    "source_sha256": digest(frozen.read_bytes()),
    "freeze_sha256": digest(freeze.read_bytes()),
    "technical_guide_sha256": digest((ROOT / "docs/technical-review-guide.md").read_bytes()),
    "repository_sources": sources,
    "previous_artifacts": artifacts,
    "section_execution": section,
    "training_diff": training_diff,
    "training_versions": {"old": {"command_argv": old_training["command_argv"], "sha256": digest(old_training["stdout"].encode())}, "current": {"command_argv": new_training["command_argv"], "sha256": digest(new_training["stdout"].encode())}, "fixed_recipe_identical": True, "linked_command": linked_command, "fixed_recipe_excerpt": fixed_recipe(training)},
    "cli": cli,
    "experiment_spec": spec,
    "unchanged_numeric_rerun": {**numeric, "result": numeric_result, "scope": "唯讀執行原 verify.py，核對未改的數字與歷史 logits；該程式保存的舊 source_sha256 是原控制輸入，不作為本次頁面版本證據。本次頁面版本以 section_execution 核對。"},
    "round_cases": round_cases,
    "claim_recheck": [{"claim_id": claim["id"], "status": "verified", "basis": "現行被引用段落及命令重新核對。" if claim["id"] == "c27" else "頁面主張未變，原證據 SHA-256 不變且仍適用；程式與數值另實跑重驗。"} for claim in previous["claims"]],
    "evidence_access": "只讀自己的報告及其明確指名證據、指定版本差異與本次 section_facts 執行目錄。未讀其他角色或其他舊輪次報告。",
    "limitations": "只核對本頁引用的固定實驗命令；相依文件其他本機路線的新說明未擴大為整節審閱。未重演歷史 GPU 訓練、載入歷史權重或測試自然音訊、生成品質、安全與讀者理解。",
}
output = BASE / "recheck-current-02.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": digest(output.read_bytes()), "result": result["result"]}, ensure_ascii=False))
