"""核對凍結原文、引用版本、固定配方與實際程式輸出。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.common import Context
from scripts.course_experiments.run import experiment_spec


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    for index, heading in enumerate(headings):
        if heading[0].startswith(b"## " + identifier.encode() + b" "):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
            return raw[heading.start():end]
    raise ValueError("找不到指定原文")


prior_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_5-16/technical-5-16-current-01.json"
prior = json.loads(prior_path.read_bytes())
assert digest(prior_path.read_bytes()) == "1c7c2db46f370eb594ca2e228a337b545bcef18f88e71c0262d4110dbb69bbb6"
assert prior["issues"] == []
original_path = ROOT / prior["original_report"]["path"]
assert digest(original_path.read_bytes()) == prior["original_report"]["sha256"]
source_audit = []
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        actual = digest((ROOT / source["path"]).read_bytes())
        source_audit.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
assert [source["path"] for source in source_audit if not source["unchanged"]] == ["course/training.md"]
artifact_audit = []
for artifact in prior["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    artifact_audit.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})

before = subprocess.check_output(["git", "show", "ecf6dec71280:course/training.md"], cwd=ROOT)
after = subprocess.check_output(["git", "show", "c6b9339d2595:course/training.md"], cwd=ROOT)
current = (ROOT / "course/training.md").read_bytes()
assert after == current
assert digest(before) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert digest(current) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
prior_recipe, current_recipe = section(before, "T.4"), section(current, "T.4")
assert prior_recipe == current_recipe
for phrase in [b"--asset ultrachat-sft", b"--experiment sft --device cuda", b"--experiment sft_ablation", b"sft/model.pt"]:
    assert phrase in current_recipe
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/5.16.md").read_bytes()
assert digest(frozen) == prior["source_sha256"]
assert section((ROOT / "course/chapters/05.md").read_bytes(), "5.16") == frozen
freeze_path = ROOT / "docs/course-review-20261019/freeze-07.json"
assert digest(freeze_path.read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"

run = ROOT / "outputs/reviewer-tools/runs/20261020T152143199402Z"
execution = json.loads((run / "execution.json").read_bytes())
extraction = json.loads((run / "extraction.json").read_bytes())
environment = json.loads((run / "environment.json").read_bytes())
assert execution["exit_code"] == 0
assert extraction["source_sha256"] == digest(frozen)
assert extraction["figure_sha256"] == {}
assert (run / "section.md").read_bytes() == frozen
section_output = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.16" --execute',
    "stdout": (run / "stdout.txt").read_text(), "stderr": (run / "stderr.txt").read_text(),
    "environment": environment, "extraction": extraction, "execution": execution,
    "original_code": (run / "fence-1.py").read_text(), "bootstrap": (run / "bootstrap.py").read_text(),
}
probe_command = [str(ROOT / ".venv/bin/python"), str(ARTIFACTS / "verify.py")]
probe = subprocess.run(probe_command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, check=True)
verified = json.loads(probe.stdout)
assert verified["replayed_sampling"]["b-only"]["effective_tokens"] == 18453
assert verified["replayed_sampling"]["replay"]["effective_tokens"] == 36384
assert all(item["matches_record"] for item in verified["record_split_checks"].values())
assert experiment_spec("sft_ablation")["dependencies"] == ["sft"]
assert experiment_spec("sft")["assets"] == ["ultrachat-sft"]
context = Context("cpu", ROOT, ARTIFACTS / "uncreated-dependencies", ROOT / "assets/training")
try:
    context.dependency("sft")
except FileNotFoundError as error:
    dependency_error = str(error)
    assert "uncreated-dependencies/sft/model.pt" in dependency_error
else:
    raise AssertionError("缺少固定配方檢查點時應停止")

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-16/recheck-current-02.py",
    "result": "原文與凍結一致；八個既有artifact雜湊未變；九個repo來源僅training.md變動，固定SFT完整小節未改；指定小節實跑與獨立數量/遮罩/抽樣驗證通過，配方依賴仍成立。",
    "environment": {"python": platform.python_version(), "torch": str(environment["torch"]), "device": "cpu"},
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "previous_report": {"path": str(prior_path.relative_to(ROOT)), "sha256": digest(prior_path.read_bytes()), "issues": prior["issues"]},
    "original_report": prior["original_report"],
    "source_audit": source_audit, "prior_artifact_audit": artifact_audit,
    "source_sha256": digest(frozen), "current_section_matches_freeze": True,
    "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": digest(freeze_path.read_bytes())},
    "training_diff": {"command": "git diff ecf6dec71280 c6b9339d2595 -- course/training.md", "stdout": subprocess.check_output(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"], cwd=ROOT, text=True)},
    "referenced_recipe": {"heading": current_recipe.decode().splitlines()[0], "previous_sha256": digest(prior_recipe), "current_sha256": digest(current_recipe), "byte_identical": True, "current_excerpt": current_recipe.decode()},
    "changed_lesson_sentences": [],
    "changed_dependency_sentences": [],
    "outside_lesson_scope": ["指定diff只改多模態推論抽查、隨機生成例子與圖文訓練前後文字對照說明。本頁及其固定SFT重做依據未改，不以本頁pass替這些新敘述背書。"],
    "section_execution": section_output,
    "independent_probe": {"command": verified["command"], "exit_code": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr},
    "dependency_probe": {"sft_ablation_dependencies": experiment_spec("sft_ablation")["dependencies"], "sft_assets": experiment_spec("sft")["assets"], "missing_checkpoint": dependency_error},
    "claim_recheck": [{"claim_id": claim["id"], "status": "verified", "basis": "本節原文、原依據與舊artifact版本已核對；本次再次執行小節及獨立算例/遮罩/抽樣驗證。" if claim["id"] != "c18" else "重讀training.md現行固定SFT區塊及指定git diff；固定配方小節完整原文相同，實際配方/檢查點依賴再次核對。"} for claim in prior["claims"]],
    "permitted_evidence_reads": ["僅讀上一份報告及它明確列出的artifact各自路徑。", "僅讀本次指定section_facts新建run的執行與抽取紀錄，不掃描其他outputs或審閱報告。"],
    "scope": "核對本頁全部主張與實際引用依據；不啟動訓練、不下載資料包、不重新量測品質/速度/記憶體；外部官方來源沿用原審閱且雜湊不變的收據。",
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
