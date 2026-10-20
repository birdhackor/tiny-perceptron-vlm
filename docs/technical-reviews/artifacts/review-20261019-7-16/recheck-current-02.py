"""核對必要前置是否改動，並保存小節的原始執行結果。"""

import copy
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).parent
REPORTS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_7-16"
PREVIOUS = REPORTS / "technical-7-16-current-01.json"
REPORT = REPORTS / "technical-7-16-current-02.json"
EVIDENCE = ARTIFACTS / "evidence-current-02.json"
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T152457242116Z"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    return digest(path.read_bytes())


def save_new(path, data):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def git(*args):
    command = ["git", *args]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return {"command_argv": command, "exit_code": completed.returncode,
            "stdout": completed.stdout.decode(), "stderr": completed.stderr.decode()}


preserved = {str(p.relative_to(ROOT)): file_digest(p)
             for directory in (REPORTS, ARTIFACTS) for p in directory.iterdir()
             if p.is_file() and p.resolve() != Path(__file__).resolve()}
assert file_digest(PREVIOUS) == "2ab20525d25ffe7bea03ee99e1e649d6acc917b4d13232a36dfb2b82ab8473ab"
previous = json.loads(PREVIOUS.read_bytes())
assert previous["issues"] == [], "先前仍有問題，不能直接沿用無待解決問題的判斷"
source_checks = []
for source in previous["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = file_digest(ROOT / source["path"])
    unchanged = actual == source["sha256"]
    if source["id"] != "s_training":
        assert unchanged, "其他引用來源已變動，需逐項重新核對"
    source_checks.append({"source_id": source["id"], "path": source["path"],
                          "previous_sha256": source["sha256"], "current_sha256": actual,
                          "unchanged": unchanged})
artifact_checks = []
for artifact in previous["artifacts"]:
    actual = file_digest(ROOT / artifact["path"])
    assert actual == artifact["sha256"], "既有證據指紋不同"
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"],
                            "sha256": actual, "unchanged": True})

before = git("show", "ecf6dec71280:course/training.md")
after = git("show", "c6b9339d2595:course/training.md")
current = (ROOT / "course/training.md").read_bytes()
assert digest(before["stdout"].encode()) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert current == after["stdout"].encode()
assert digest(current) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
pattern = rb'<details id="fixed-sft-recipe"[^>]*>.*?</details>'
old_recipe = re.findall(pattern, before["stdout"].encode(), re.S)
new_recipe = re.findall(pattern, current, re.S)
assert len(old_recipe) == len(new_recipe) == 1 and old_recipe == new_recipe, "固定對話配方已變動"
source_diff = git("diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md")

copies = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (RUN / name).read_bytes()
    copies[name] = {"sha256": digest(raw), "utf8_content": raw.decode()}
extraction = json.loads(copies["extraction.json"]["utf8_content"])
execution = json.loads(copies["execution.json"]["utf8_content"])
environment = json.loads(copies["environment.json"]["utf8_content"])
assert extraction["source_sha256"] == previous["source_sha256"]
assert extraction["figure_sha256"] == {}
assert execution["exit_code"] == 0 and copies["stderr.txt"]["utf8_content"] == ""
assert copies["stdout.txt"]["utf8_content"] == "每批來源 ['A', 'A', 'B', 'B']\n有效目標數 8\n待測B-only與回放各自的A/B成績\n"
prior_section = json.loads((ROOT / next(a["path"] for a in previous["artifacts"] if a["id"] == "a_section")).read_bytes())
assert copies["fence-1.py"]["sha256"] == prior_section["copied_files"]["fence-1.py"]["sha256"]

freeze_path = ROOT / "docs/course-review-20261019/freeze-07.json"
assert file_digest(freeze_path) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
freeze_page = next(p for p in json.loads(freeze_path.read_bytes())["pages"] if p["page_id"] == previous["lesson_id"])
assert freeze_page["source_sha256"] == extraction["source_sha256"]
assert freeze_page["figures_sha256"] == {}
assert file_digest(ROOT / freeze_page["snapshot"]) == extraction["source_sha256"]

claim_checks = []
for claim in previous["claims"]:
    method = "主張原文未變，相關來源及執行證據指紋相同，原始核對仍適用。"
    if claim["id"] == "c_rerun_command":
        method = "逐位元組比較fixed-sft-recipe；固定SFT、ultrachat-sft前置與sft/model.pt依賴未變。run.main與Context.dependency亦未變，沿用既有CLI解析證據。"
    elif "a_section" in claim.get("artifact_ids", []):
        method = "重新執行相同小節原始程式，核對stdout、stderr、來源SHA與原始程式SHA相同。"
    claim_checks.append({"claim_id": claim["id"], "status": "verified", "details": method})

changes = [
    {"location": "course/training.md：多模態練習的抽查與固定驗證題說明",
     "status": "outside_reviewed_dependency", "affects_reviewed_claims": False,
     "details": "差異位於fixed-sft-recipe之外；本節未以這些多模態train.py命令作為回放實驗的依據。未將新增敘述宣告為本節已核實主張。"},
    {"location": "course/training.md：多模態檢查點與文字能力退步對照說明",
     "status": "outside_reviewed_dependency", "affects_reviewed_claims": False,
     "details": "差異涉及多模態檢查點與vqa，不影響本節固定sft與sft_ablation命令、模型依賴或其成績。未另作多模態小節的完整技術審閱。"},
]
evidence = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-16/recheck-current-02.py",
    "result": "全部斷言通過；必要前置逐位元組未變；小節原始程式實跑成功；全部沿用證據指紋相同。",
    "environment": {"python": environment["python"], "torch": environment["torch"], "device": "cpu"},
    "previous_report": {"path": str(PREVIOUS.relative_to(ROOT)), "sha256": file_digest(PREVIOUS)},
    "previous_issues": previous["issues"], "resolved_issue_ids": [],
    "source_checks": source_checks, "artifact_checks": artifact_checks,
    "source_diff": source_diff,
    "source_versions": {"before_command": before["command_argv"], "before_sha256": digest(before["stdout"].encode()),
                        "after_command": after["command_argv"], "current_sha256": digest(current)},
    "fixed_recipe": {"old_sha256": digest(old_recipe[0]), "current_sha256": digest(new_recipe[0]), "byte_identical": True},
    "section_execution": {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/07.md#7.16" --execute',
        "launcher_stdout": '{"artifact_directory": "/Users/yclin/project/tiny-perceptron-vlm/outputs/reviewer-tools/runs/20261020T152457242116Z", "source_sha256": "d9c74585a360e3a4f41e67902b8c033ac128d568f9713a72e7f01cc9bc47878e", "python_fences": 1, "svg_references": 0, "execution_exit_code": 0}\n',
        "copied_files": copies,
    },
    "changed_source_regions": changes, "claim_checks": claim_checks,
    "preserved_files": preserved,
    "limitations": "未重跑訓練、重新生成回答或量測GPU時間；多模態來源改句不在本節所引用的固定對話配方內，未另作它們的完整技術審閱。",
}
save_new(EVIDENCE, evidence)

report = copy.deepcopy(previous)
report["original_report"] = {"path": "docs/course-review-20261019/reviews/rv1019_technical_7-16/technical-7-16-original-01.json", "sha256": "8a4159cb669dc937c9a021fcd54fa1c508808c89536c51db59f88c40bcfeb516"}
report["previous_report"] = evidence["previous_report"]
report["initial_context"] = "recheck by the original reviewer"
report["resolved_issue_ids"] = []
report["source_freeze"] = {"path": str(freeze_path.relative_to(ROOT)), "sha256": file_digest(freeze_path)}
report["reviewer_task"] = "/root/rv1019_technical_7-16"
report["reviewer_context"] = "fresh"
report["verdict"] = "pass"
report["issues"] = []
source = next(s for s in report["sources"] if s["id"] == "s_training")
source["sha256"] = digest(current)
source["inspection_note"] = "讀取指定git diff與現行fixed-sft-recipe，並與上次來源逐位元組比較。兩處差異位於多模態練習，未改本節引用的固定對話配方。固定SFT與ultrachat-sft前置、sft/model.pt依賴仍成立；執行器、依賴實作與既有CLI解析證據指紋相同。相關比較保存於a_latest；沒有把新hash當作內容審閱。"
report["artifacts"].extend([
    {"id": "a_latest_code", "kind": "code", "path": str(Path(__file__).relative_to(ROOT)), "sha256": file_digest(Path(__file__)),
     "description": "本次複查的唯讀核對程式；新證據及報告採不覆寫模式保存。"},
    {"id": "a_latest", "kind": "execution", "path": str(EVIDENCE.relative_to(ROOT)), "sha256": file_digest(EVIDENCE),
     "description": "來源差異與固定前置的位元組比較、全部沿用證據指紋、小節新執行結果及逐主張複查記錄。",
     "command": evidence["command"], "result": evidence["result"], "environment": evidence["environment"]},
])
report["sources"].append({"id": "s_latest", "kind": "execution", "title": "必要前置變動的複查收據", "verified": True, "artifact_id": "a_latest"})
for claim, check in zip(report["claims"], claim_checks, strict=True):
    claim["artifact_ids"].append("a_latest")
    claim["recheck"] = {"status": "verified", "details": check["details"]}
    if claim["id"] == "c_rerun_command":
        claim["evidence"].append({"source_id": "s_latest", "locator": "source_diff；fixed_recipe；source_checks", "supports": "多模態說明的修改未涉及固定SFT前置，原重跑主張仍成立。"})
        claim["verification"]["observed"] = "本次指定diff不涉及fixed-sft-recipe，該區塊逐位元組相同。run.py與common.py未變，沿用上次已核實且SHA相同的CLI解析與sft/model.pt依賴證據。"
report["checks"]["factual_accuracy"]["details"] = "逐項複查全部39條主張。頁面與實驗來源未變；唯一變動的training.md只改多模態說明，必要前置逐位元組相同，原判斷仍成立。"
report["checks"]["numeric_verification"]["details"] = "重新實跑原始小節程式，來源A、A、B、B，有效目標8，stderr為空且exit_code=0。其餘透明算例、90筆生成ID、資料切分、抽樣及歷史計時核對沿用SHA未變的原始證據。"
report["checks"]["figure_consistency"]["details"] = "凍結原文與本次抽取均沒有圖，figure_sha256維持空字典。"
report["checks"]["source_verification"]["details"] = "重算全部引用repo檔與沿用artifact的SHA；除training.md外均相同。讀取指定diff與固定前置，確認只有未用作本節依據的多模態說明改動。沿用原審閱已讀取的原始論文與實驗證據，明確區分重新實跑與歷史核對。"
report["checks"]["limitations"]["details"] = "保留單seed小題集、B題皆錯及監督／輸入量未匹配的限制。這次沒有重跑訓練、重新生成回答或獨立驗證GPU時間，亦不對其他小節新增的多模態說明作全面判斷。"
report["source_change_review"] = {"lesson_text_changed": False, "changed_repo_sources": ["course/training.md"], "fixed_recipe_byte_identical": True,
                                  "changed_regions": changes, "reviewed_claim_ids": ["c_rerun_command"],
                                  "details": "本節完整主張均列於claims；來源改句位於未引用的多模態小節，未轉成回放效果的證據。"}
report["unverified_scope"] = [
    "未重跑完整SFT與回放訓練、重新生成回答或量測NVIDIA L4耗時；沿用來源與artifact的SHA均未變的原始實驗核對。",
    "未驗證其他seed、模型、資料與平台的品質或速度，本頁未作此類外推。",
    "未另作training.md改動的多模態隨機訓練、固定驗證題與文字能力對照敘述的完整審閱；它們不在本節引用的fixed-sft-recipe之內。",
    "原實驗revision無法直接讀取指定檔案的追溯限制仍維持；原審閱已保存相同實作SHA與歷史執行器來源證據。",
]
report["independence_note"] = "由原審閱者複查，只讀自己的指定報告與明確指名證據、現行原稿及必要實作。未讀其他角色或先前輪次報告。新執行紀錄只從本次section_facts建立的目錄複製；未重新執行會覆寫舊artifact的程式，所有新檔採open(path, x)。"
report["escalation_signals"] = {"A": "無；指定來源SHA、diff及小節執行與派工內容一致。", "B": "無；本次派工未附已駁回清單。", "C": "無；只新增指定報告與artifact，小節工具依既定方式建立自有執行目錄。", "D": "無；未修改教材或既有程式。"}
for source in report["sources"]:
    if source["kind"] == "repository_code":
        assert file_digest(ROOT / source["path"]) == source["sha256"], "報告的來源指紋不是現行檔"
spec = importlib.util.spec_from_file_location("technical_validator", ROOT / "scripts/check_technical_reviews.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
errors = validator._validate(ROOT, ROOT / "course/chapters/07.md", report["lesson_id"],
                             (ROOT / freeze_page["snapshot"]).read_text(), report,
                             {report["reviewer_task"]: [report["lesson_id"]]}, {"/root/rv1019_reader_ch07"})
assert not errors, errors
save_new(REPORT, report)
for path, expected in preserved.items():
    assert file_digest(ROOT / path) == expected, "既有檔案內容有變動"
print(json.dumps({"verdict": report["verdict"], "claims": len(report["claims"]),
                  "report": str(REPORT.relative_to(ROOT)), "sha256": file_digest(REPORT),
                  "evidence": str(EVIDENCE.relative_to(ROOT)), "evidence_sha256": file_digest(EVIDENCE)}, ensure_ascii=False))
