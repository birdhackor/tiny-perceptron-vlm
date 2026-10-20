"""核對引用版本、教材差異及命令列解析；攔截訓練入口。"""

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
from scripts.course_experiments import run as runner
from scripts.course_experiments.common import Context

AD = Path(__file__).resolve().parent

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def section(raw, name):
    match = re.search(rb"(?m)^## " + re.escape(name.encode()) + rb" [^\r\n]+", raw)
    remaining = raw[match.start():]
    next_header = re.search(rb"(?m)^## ", remaining[3:])
    return remaining[:next_header.start() + 3] if next_header else remaining

original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_5-16/technical-5-16-original-01.json"
original = json.loads(original_path.read_bytes())
assert digest(original_path.read_bytes()) == "5a5eb784e9211f0b7739d03265032d3b7331088fa1a089735c005f2a5b7f5c8d"
source_checks = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        actual = digest((ROOT / source["path"]).read_bytes())
        source_checks.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
assert [s["path"] for s in source_checks if not s["unchanged"]] == ["course/training.md"]
artifact_checks = []
for artifact in original["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})

before = subprocess.check_output(["git", "show", "5ef56bc756f3:course/training.md"], cwd=ROOT)
current = (ROOT / "course/training.md").read_bytes()
target = subprocess.check_output(["git", "show", "6a7aa1de5227:course/training.md"], cwd=ROOT)
assert current == target
assert digest(before) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
assert digest(current) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
prior_t4, current_t4 = section(before, "T.4"), section(current, "T.4")
assert prior_t4 == current_t4
for phrase in [b"--asset ultrachat-sft", b"--experiment sft --device cuda", b"--experiment sft_ablation", b"sft/model.pt"]:
    assert phrase in current_t4

diff_args = ["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"]
diff = subprocess.check_output(diff_args, text=True, cwd=ROOT)
cli_checks = []
for device in (None, "cuda"):
    argv = ["scripts.course_experiments.run", "--experiment", "sft_ablation"]
    if device:
        argv += ["--device", device]
    output = io.StringIO()
    with patch.object(sys, "argv", argv), patch.object(runner, "execute", return_value={"experiment_id": "sft_ablation", "elapsed_seconds": 0, "evidence_status": "intercepted", "artifacts": []}) as intercepted, contextlib.redirect_stdout(output):
        runner.main()
    args, kwargs = intercepted.call_args
    assert args[0] == "sft_ablation"
    assert args[1] == (device or "cpu")
    assert args[3] == ROOT / "outputs/course-experiments/course-v1"
    cli_checks.append({"argv": argv, "dispatched_device": args[1], "output": str(args[2].relative_to(ROOT)), "dependencies": str(args[3].relative_to(ROOT)), "step_scale": kwargs["step_scale"], "stdout": output.getvalue(), "training_executed": False})
ctx = Context("cpu", ROOT, AD / "uncreated-dependencies", ROOT / "assets/training")
try:
    ctx.dependency("sft")
except FileNotFoundError as error:
    missing_dependency = str(error)
    assert "uncreated-dependencies/sft/model.pt" in missing_dependency
else:
    raise AssertionError("預期缺少依賴時應中止")

verification = json.loads((AD / "verification-current-01.json").read_bytes())
section_execution = json.loads((AD / "section-execution-current-01.json").read_bytes())
assert section_execution["worker_execution"]["exit_code"] == 0
assert section_execution["extraction"]["source_sha256"] == original["source_sha256"]
assert section_execution["extraction"]["figure_sha256"] == {}
assert verification["replayed_sampling"]["b-only"]["effective_tokens"] == 18453
assert verification["replayed_sampling"]["replay"]["effective_tokens"] == 36384
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/5.16.md").read_bytes()
assert section((ROOT / "course/chapters/05.md").read_bytes(), "5.16") == frozen

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-16/recheck-current-01.py",
    "result": "原報告與既有證據未變；引用來源僅training.md改變；固定SFT小節完整原文相同；CLI裝置/依賴核對通過；已保存的本次數量驗證及小節執行結果通過。",
    "environment": {"python": platform.python_version(), "torch": verification["environment"]["torch"], "device": "cpu", "git": subprocess.check_output(["git", "--version"], text=True).strip()},
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": digest(original_path.read_bytes()), "issues": original["issues"]},
    "repo_source_checks": source_checks,
    "prior_artifact_checks": artifact_checks,
    "source_sha256": digest(frozen),
    "current_section_matches_freeze": True,
    "freeze": {"path": "docs/course-review-20261019/freeze-06.json", "sha256": digest((ROOT / "docs/course-review-20261019/freeze-06.json").read_bytes())},
    "diff_command_argv": diff_args,
    "diff_stdout": diff,
    "referenced_training_section": {"heading": current_t4.decode().splitlines()[0], "previous_sha256": digest(prior_t4), "current_sha256": digest(current_t4), "byte_identical": True, "current_excerpt": current_t4.decode()},
    "changed_sentence_checks": [
        {"location": "training.md 的 Python 怎麼寫", "sentence": "scripts.course_experiments.run 沒指定時用 CPU。", "status": "verified", "evidence": "讀取main的default=cpu，並攔截execute實跑main；未指定裝置時得到cpu，明寫cuda時得到cuda。固定配方明寫cuda。", "scope": "與本頁重做命令相關的裝置敘述。"},
        {"location": "training.md 的多模態推論答案說明", "status": "outside_lesson_scope", "reason": "本頁及原報告未引用此段；不將整檔SHA更新當成這段已審。"},
        {"location": "training.md 的量化前後答案說明", "status": "outside_lesson_scope", "reason": "本頁及原報告未引用此段；不將整檔SHA更新當成這段已審。"}
    ],
    "cli_parse_checks": cli_checks,
    "missing_dependency_observed": missing_dependency,
    "new_execution_artifacts": [{"path": str((AD / name).relative_to(ROOT)), "sha256": digest((AD / name).read_bytes())} for name in ["verification-current-01.json", "section-execution-current-01.json"]],
    "superseded_receipt": {"path": str((AD / "recheck-evidence-current-01.json").relative_to(ROOT)), "sha256": digest((AD / "recheck-evidence-current-01.json").read_bytes()), "reason": "前份收據的title誤寫為轉述；此份heading取自實際原文，並重跑相同核對。前份收據未改寫。"},
    "claim_recheck": [{"claim_id": c["id"], "status": "verified", "basis": "本節原文未改；既有證據版本一致，且本次重跑算例、遮罩與抽樣核對。" if c["id"] != "c18" else "重讀現行固定SFT區塊；實跑配方解析，並攔截CLI確認裝置與依賴位置。"} for c in original["claims"]],
    "permitted_evidence_reads": ["僅讀原報告明確列出的artifact。", "僅讀指定section_facts本次新建run的原文/執行/環境紀錄，不掃描outputs。"],
    "scope": "不執行訓練、不下載資料包、不讀其他角色報告。只複查本頁及其實際依據；training.md未被本頁引用的其他段落不在此頁複查範圍。"
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
