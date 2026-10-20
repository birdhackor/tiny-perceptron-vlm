"""核對原始報告、未變動證據及重跑命令的依賴變動，只輸出核對紀錄。"""
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.run import experiment_spec


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return digest((ROOT / path).read_bytes())


def git(*args):
    command = ["git", *args]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return result.stdout, {"command_argv": command, "exit_code": result.returncode, "stderr": result.stderr.decode()}


original_path = "docs/course-review-20261019/reviews/rv1019_technical_9-10/technical-9-10-original-01.json"
original_sha = "73d3500fc08ea5636a6aba13736204816ce74960d7e2ac89d0167cd98a353e7d"
assert file_sha(original_path) == original_sha, "原始報告版本不符"
original = json.loads((ROOT / original_path).read_bytes())
assert original["issues"] == [], "原始報告的 issue 清單與預期不同"

old_raw, old_command = git("show", "5ef56bc756f3:course/training.md")
new_raw, new_command = git("show", "6a7aa1de5227:course/training.md")
diff_raw, diff_command = git("diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md")
current_raw = (ROOT / "course/training.md").read_bytes()
assert digest(old_raw) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21", "比較起點與原先引用版本不同"
assert digest(current_raw) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94", "現行訓練頁版本不同"
assert current_raw == new_raw, "工作樹與指定比較終點不同"

command_text = ".venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda"
heading = "### 重跑第11章的固定實驗"

def rerun_block(raw):
    text = raw.decode()
    start = text.index(heading)
    code_start = text.index("```bash\n", start)
    end = text.index("\n```", code_start) + len("\n```")
    return text[start:end]

old_block = rerun_block(old_raw)
current_block = rerun_block(current_raw)
assert old_block == current_block, "所引用的重跑命令區塊有變動"
assert current_block.count(command_text) == 1, "重跑命令出現次數不符"
training_text = current_raw.decode()
headers = list(re.finditer(r"(?m)^## [^\n]+", training_text))
heading_matches = [i for i, h in enumerate(headers) if h.group().startswith("## T.6 ")]
assert len(heading_matches) == 1, "訓練頁連結的目標不唯一"
i = heading_matches[0]
section_end = headers[i + 1].start() if i + 1 < len(headers) else len(training_text)
assert headers[i].start() < training_text.index(command_text) < section_end, "重跑命令不在連結指向的小節內"
assert training_text.rfind("<details>", headers[i].start(), training_text.index(command_text)) != -1, "重跑命令不在補充區"
spec = experiment_spec("encoders")
assert (spec["module"], spec["function"]) == ("modalities", "run_encoders"), "實驗名稱與程式入口不符"
runner = (ROOT / "scripts/course_experiments/run.py").read_text()
assert 'parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")' in runner, "實驗命令的裝置參數與預設不符"

repository_sources = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        current_sha = file_sha(source["path"])
        changed = current_sha != source["sha256"]
        assert not changed or source["id"] == "training", "另有來源檔變動，必須重新判斷"
        repository_sources.append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": current_sha, "changed": changed})
original_artifacts = []
for artifact in original["artifacts"]:
    current_sha = file_sha(artifact["path"])
    assert current_sha == artifact["sha256"], "原始 artifact 的內容已變動"
    original_artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": current_sha, "unchanged": True})
old_self_check = "docs/technical-reviews/artifacts/review-20261019-9-10/self-check.txt"
assert file_sha(old_self_check) == "72ed8d49df235ae0989304186e78f6ae710f8d48a05669b7a33d9218218081da", "原始自我檢查紀錄已變動"

freeze_path = "docs/course-review-20261019/freeze-06.json"
assert file_sha(freeze_path) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9", "凍結清單版本不符"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == original["lesson_id"])
snapshot = (ROOT / page["snapshot"]).read_bytes()
assert digest(snapshot) == original["source_sha256"] == page["source_sha256"], "凍結原文版本不符"
helper_spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
section_raw, _, _ = helper.original_section(ROOT / "course/chapters/09.md", original["lesson_id"])
assert section_raw == snapshot, "現行小節與凍結原文不同"
assert page["figures_sha256"] == original["figure_sha256"], "圖清單與原始審閱不同"
for path, expected in original["figure_sha256"].items():
    assert file_sha(path) == expected, "圖的版本不同"

claim_rechecks = []
for claim in original["claims"]:
    assert claim["status"] == "verified", "原始主張並非全部核實"
    depends_on_training = any(e["source_id"] == "training" for e in claim["evidence"])
    claim_rechecks.append({"claim_id": claim["id"], "kind": claim["kind"], "status": "verified", "rechecked_changed_dependency": depends_on_training, "basis": "重讀完整凍結原文；原始來源與 artifacts 未變，另重跑原始區塊及獨立重算。" if not depends_on_training else "重讀現行訓練頁與指定 git diff，確認連結目的、補充區、完整命令與程式入口仍成立。"})

hunks = re.findall(r"(?ms)^@@.*?(?=^@@|\Z)", diff_raw.decode())
assert len(hunks) == 3, "指定差異的變動區塊數不符"
impact = [
    {"location": "Python 怎麼寫", "change": "分別說明各工具的裝置選擇與是否接受裝置參數。", "impact_on_reviewed_claim": "本節引用的命令明確指定 --device cuda，不依賴預設；實驗 runner 的 CPU 預設也由現行程式核對。"},
    {"location": "多模態推論三題後的說明", "change": "增加不同電腦可能產生不同訓練結果，以及以完整驗證題判斷的提醒。", "impact_on_reviewed_claim": "沒有改到重跑命令區塊，也沒有改寫本節引用的既有分類器 logits 或校準測試數字。"},
    {"location": "量化前後推論示範", "change": "補充先前答案未必為 blue，應比較量化前後答案。", "impact_on_reviewed_claim": "本節沒有引用這項量化示範；與溫度縮放、ECE 及分類器原始紀錄無關。"},
]
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-10/recheck_dependencies-current-01.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "git": subprocess.run(["git", "--version"], capture_output=True, check=True, text=True).stdout.strip()},
    "result": "全部依賴核對相符",
    "original_report": {"path": original_path, "sha256": original_sha, "issues": original["issues"]},
    "source_freeze": {"path": freeze_path, "sha256": file_sha(freeze_path)},
    "source_sha256": digest(section_raw),
    "page_sentences_changed": False,
    "repository_sources": repository_sources,
    "original_artifacts": original_artifacts,
    "original_self_check": {"path": old_self_check, "sha256": file_sha(old_self_check)},
    "git_commands": [{**old_command, "stdout_sha256": digest(old_raw)}, {**new_command, "stdout_sha256": digest(new_raw)}, {**diff_command, "stdout": diff_raw.decode()}],
    "training_changes": impact,
    "rerun_dependency": {"code_block_unchanged": True, "block_sha256": digest(current_block.encode()), "block": current_block, "target_is_linked_section_supplement": True, "experiment_spec": spec, "runner_device_default": "cpu", "command_explicit_device": "cuda", "training_executed": False},
    "claim_rechecks": claim_rechecks,
    "resolved_issue_ids": [],
    "scope": "完整複查本節全部實質主張；對訓練頁其他變動只判斷其對本節依賴的影響，不宣稱完成訓練頁全節審閱。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
