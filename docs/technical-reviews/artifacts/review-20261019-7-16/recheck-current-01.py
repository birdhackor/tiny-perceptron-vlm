"""核對引用來源的變動、保存的新執行結果與既有證據的完整性。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
ARTIFACTS = Path(__file__).parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_7-16/technical-7-16-original-01.json"
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T102739097204Z"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run_git(*arguments):
    command = ["git", *arguments]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return {"command_argv": command, "exit_code": result.returncode,
            "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


original_raw = ORIGINAL.read_bytes()
assert digest(original_raw) == "8a4159cb669dc937c9a021fcd54fa1c508808c89536c51db59f88c40bcfeb516"
original = json.loads(original_raw)
source_checks = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = digest((ROOT / source["path"]).read_bytes())
    entry = {"source_id": source["id"], "path": source["path"],
             "previous_sha256": source["sha256"], "current_sha256": actual,
             "unchanged": actual == source["sha256"]}
    if source["id"] != "s_training":
        assert entry["unchanged"], "先前引用的其他來源已有變動，需另外檢視"
    source_checks.append(entry)
artifact_checks = []
for artifact in original["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], "先前保存的證據指紋不同"
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"],
                            "sha256": actual, "unchanged": True})

before = run_git("show", "5ef56bc756f3:course/training.md")
after = run_git("show", "6a7aa1de5227:course/training.md")
current = (ROOT / "course/training.md").read_bytes()
assert digest(before["stdout"].encode()) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
assert current == after["stdout"].encode()
assert digest(current) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
pattern = rb'<details id="fixed-sft-recipe"[^>]*>.*?</details>'
old_recipe = re.findall(pattern, before["stdout"].encode(), flags=re.S)
new_recipe = re.findall(pattern, current, flags=re.S)
assert len(old_recipe) == len(new_recipe) == 1
assert old_recipe[0] == new_recipe[0], "引用的固定對話配方已有變動"
source_diff = run_git("diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md")

copied = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (RUN / name).read_bytes()
    copied[name] = {"sha256": digest(raw), "utf8_content": raw.decode()}
execution = json.loads(copied["execution.json"]["utf8_content"])
extraction = json.loads(copied["extraction.json"]["utf8_content"])
environment = json.loads(copied["environment.json"]["utf8_content"])
assert execution["exit_code"] == 0
assert extraction["source_sha256"] == original["source_sha256"]
assert extraction["figure_sha256"] == {}
assert copied["stderr.txt"]["utf8_content"] == ""
assert copied["stdout.txt"]["utf8_content"] == "每批來源 ['A', 'A', 'B', 'B']\n有效目標數 8\n待測B-only與回放各自的A/B成績\n"

from scripts.course_experiments import run
from scripts.course_experiments.common import Context

cli_checks = []
for extra, expected in (([], "cpu"), (["--device", "cuda"], "cuda")):
    calls = []

    def capture_execute(*args, **kwargs):
        calls.append({"experiment": args[0], "device": args[1],
                      "output": str(args[2]), "dependencies": str(args[3]),
                      "step_scale": kwargs["step_scale"]})
        return {"experiment_id": args[0], "elapsed_seconds": 0,
                "evidence_status": "cli_capture_only", "artifacts": []}

    stdout = io.StringIO()
    with patch.object(sys, "argv", ["scripts.course_experiments.run", "--experiment", "sft_ablation", *extra]):
        with patch.object(run, "execute", capture_execute):
            with contextlib.redirect_stdout(stdout):
                run.main()
    assert len(calls) == 1 and calls[0]["device"] == expected
    cli_checks.append({"argv": ["scripts.course_experiments.run", "--experiment", "sft_ablation", *extra],
                       "selected": calls[0], "stdout": stdout.getvalue(),
                       "scope": "只執行真實CLI參數解析，execute以擷取函式替換，不載入模型、不訓練。"})

import inspect

common_source = inspect.getsource(Context.dependency)
runner_source = inspect.getsource(run.main)
text_source = (ROOT / "scripts/course_experiments/text.py").read_text()
assert 'base = load_lm(ctx.dependency("sft"), ctx.device)' in text_source
assert 'filename="model.pt"' in common_source
claim_checks = []
for claim in original["claims"]:
    source_ids = [ref["source_id"] for ref in claim["evidence"]]
    checks = "頁面、來源及引用證據均未變動，沿用原審閱的原始核對。"
    if "s_training" in source_ids:
        checks = "固定對話配方逐位元組相同；重新核對CLI的CPU預設、明示CUDA及sft/model.pt依賴，原主張成立。"
    elif "a_section" in claim.get("artifact_ids", []):
        checks = "重新實跑相同小節原始程式；stdout及原始碼與先前一致，原主張成立。"
    claim_checks.append({"claim_id": claim["id"], "status": "verified", "reason": checks})

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-16/recheck-current-01.py",
    "result": "全部斷言通過；本節未變；固定對話配方未變；全部既有證據指紋相同；CLI的CPU預設與明示CUDA正確。",
    "environment": {"python": platform.python_version(), "torch": str(environment["torch"]), "device": "cpu"},
    "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": digest(original_raw)},
    "original_issues": original["issues"], "resolved_issue_ids": [],
    "source_checks": source_checks, "artifact_checks": artifact_checks,
    "source_diff": source_diff,
    "source_versions": {"before": {"command_argv": before["command_argv"], "sha256": digest(before["stdout"].encode())},
                        "after": {"command_argv": after["command_argv"], "sha256": digest(current)}},
    "fixed_recipe": {"old_sha256": digest(old_recipe[0]), "current_sha256": digest(new_recipe[0]),
                     "byte_identical": True, "current_utf8_content": new_recipe[0].decode()},
    "section_execution": {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/07.md#7.16" --execute',
        "launcher_stdout": '{"artifact_directory": "/Users/yclin/project/tiny-perceptron-vlm/outputs/reviewer-tools/runs/20261020T102739097204Z", "source_sha256": "d9c74585a360e3a4f41e67902b8c033ac128d568f9713a72e7f01cc9bc47878e", "python_fences": 1, "svg_references": 0, "execution_exit_code": 0}\n',
        "copied_files": copied,
    },
    "cli_device_checks": cli_checks,
    "source_excerpts": {"scripts/course_experiments/run.py:main": runner_source,
                        "scripts/course_experiments/common.py:Context.dependency": common_source,
                        "scripts/course_experiments/text.py:run_sft_ablation": 'base = load_lm(ctx.dependency("sft"), ctx.device)'},
    "changed_source_regions": [
        {"location": "course/training.md：Python 怎麼寫", "relevant_sentence": "scripts.course_experiments.run 沒指定時用 CPU。",
         "status": "verified", "evidence": "cli_device_checks與main的預設值；本節重跑命令明寫--device cuda，仍選CUDA。",
         "impact_on_lesson": "不改動固定SFT前置或本節命令的明示裝置。其他工具的裝置說明不屬於本節依據。"},
        {"location": "course/training.md：多模態推論示例後的跨電腦結果提醒", "status": "outside_reviewed_dependency",
         "impact_on_lesson": "位於fixed-sft-recipe以外，未被原報告的任一主張用作依據，本次未另作多模態結果審閱。"},
        {"location": "course/training.md：量化示例後的blue答案提醒", "status": "outside_reviewed_dependency",
         "impact_on_lesson": "位於fixed-sft-recipe以外，未被原報告的任一主張用作依據，本次未另作量化能力審閱。"},
    ],
    "claim_checks": claim_checks,
    "reused_evidence_scope": "只讀原報告與其明確指名的證據檔；沒有讀取其他角色報告。新增執行結果只從本次section_facts建立的目錄複製。",
    "limitations": "沿用先前從保存生成ID與確定性抽樣得到的實驗核對；本次沒有重跑模型訓練、重新生成回答或量測GPU耗時。",
}
path = ARTIFACTS / "evidence-current-01.json"
with path.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes()),
                  "claims_verified": len(claim_checks), "result": receipt["result"]}, ensure_ascii=False))
