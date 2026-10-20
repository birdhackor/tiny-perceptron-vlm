"""核對文字實驗入口、引用版本與沿用證據的完整性。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    return sha((ROOT / path).read_bytes())


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


previous_path = "docs/course-review-20261019/reviews/rv1019_technical_5-11/technical-5-11-current-01.json"
previous_sha = "223e8c503f2ac957dac561d6e922605f0b0f19e0dc4f650116b226da664d3fef"
assert digest(previous_path) == previous_sha
previous = json.loads((ROOT / previous_path).read_bytes())
assert not previous["issues"]
assert digest(previous["original_report"]["path"]) == previous["original_report"]["sha256"]
freeze_path = "docs/course-review-20261019/freeze-07.json"
assert digest(freeze_path) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == previous["lesson_id"])
snapshot = (ROOT / page["snapshot"]).read_bytes()
assert sha(snapshot) == previous["source_sha256"] == page["source_sha256"]
assert page["figures_sha256"] == previous["figure_sha256"] == {}
chapter = (ROOT / page["source"]).read_bytes()
pattern = rb"(?ms)^## " + re.escape(previous["lesson_id"].encode()) + rb" .*?(?=^## |\Z)"
assert re.search(pattern, chapter).group() == snapshot

artifacts = []
for item in previous["artifacts"]:
    actual = digest(item["path"])
    assert actual == item["sha256"]
    artifacts.append({"path": item["path"], "sha256": actual, "matches_previous": True})
sources = []
for item in previous["sources"]:
    if item["kind"] != "repository_code":
        continue
    actual = digest(item["path"])
    same = actual == item["sha256"]
    if item["path"] == "course/training.md":
        assert actual == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
    else:
        assert same
    sources.append({"path": item["path"], "previous_sha256": item["sha256"], "current_sha256": actual, "unchanged": same})

before_command = ["git", "show", "ecf6dec71280:course/training.md"]
after_command = ["git", "show", "c6b9339d2595:course/training.md"]
before = subprocess.check_output(before_command, cwd=ROOT)
after = subprocess.check_output(after_command, cwd=ROOT)
current = (ROOT / "course/training.md").read_bytes()
assert sha(before) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert after == current

def supporting_section(raw):
    text = raw.decode("utf-8")
    return re.search(r"(?ms)^## T\.4 .*?(?=^## |\Z)", text).group()

support_before = supporting_section(before)
support_current = supporting_section(current)
assert support_before == support_current
recipe = ".venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda"
assert recipe in support_current
cli = command([sys.executable, "-m", "scripts.course_experiments.run", "--list-assets", "real_text"])
assert cli["exit_code"] == 0
assert set(cli["stdout"].splitlines()) == {"assets/training/tinystories-v1.tar.gz", "assets/training/chinese-poetry-v1.tar.gz"}
diff = command(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
assert diff["exit_code"] == 0

run = ROOT / "outputs/reviewer-tools/runs/20261020T152104248892Z"
execution = json.loads((run / "execution.json").read_bytes())
extraction = json.loads((run / "extraction.json").read_bytes())
environment = json.loads((run / "environment.json").read_bytes())
stdout = (run / "stdout.txt").read_text()
stderr = (run / "stderr.txt").read_text()
assert execution["exit_code"] == 0
assert extraction["source_sha256"] == previous["source_sha256"]
assert len(extraction["python_fences"]) == 3
assert not extraction["svg_references"]
old_section = next(a for a in previous["artifacts"] if a["id"] == "a-section")
assert stdout == json.loads((ROOT / old_section["path"]).read_bytes())["stdout"]
assert stderr == ""
section = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "' + previous["source"] + '" --execute',
    "original_run_directory": str(run.relative_to(ROOT)),
    "extraction": extraction,
    "execution": execution,
    "environment": environment,
    "stdout": stdout,
    "stderr": stderr,
    "original_output_sha256": {name: sha((run / name).read_bytes()) for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]}
}
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-11/recheck-current-02.py",
    "environment": {"python": platform.python_version(), "torch": str(environment["torch"]), "device": "cpu", "platform": platform.platform()},
    "repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "previous_report": {"path": previous_path, "sha256": previous_sha},
    "source_sha256": sha(snapshot),
    "page_source_unchanged": True,
    "source_freeze": {"path": freeze_path, "sha256": digest(freeze_path)},
    "retained_artifacts": artifacts,
    "repository_sources": sources,
    "training_dependency": {
        "path": "course/training.md", "before_command_argv": before_command, "after_command_argv": after_command,
        "previous_sha256": sha(before), "current_sha256": sha(current), "current_equals_specified_after_commit": True,
        "supporting_section_identical": True, "supporting_section_sha256": sha(support_current.encode()),
        "recipe": recipe, "diff": diff,
        "inspection_note": "差異只在多模態手動練習的抽查與文字能力對照敘述；文字訓練與對話練習的完整小節未變，真實文字實驗重做命令仍在。這裡核對的是本頁引用的重做入口，沒有對多模態段落另作全節裁定。"
    },
    "cli": cli,
    "section_execution": section,
    "claim_dispositions": [{"claim_id": c["id"], "status": "verified", "basis": "已核對引用補充區全文相同並重新執行只讀實驗入口；引用現行檔案 hash。" if c["id"] == "c27" else "頁面沒有修改，原始來源與沿用證據 hash 已確認；既有支持範圍仍適用。"} for c in previous["claims"]],
    "result": "引用補充區全文未變，真實文字實驗名稱與資料需求核對成功；頁面原文、來源及沿用證據版本符合預期。小節三段程式重新跑通，輸出相同。"
}
with (DEST / "recheck-current-02.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source_sha256": result["source_sha256"], "training_dependency": result["training_dependency"], "cli": cli, "section_stdout": stdout, "claim_count": len(result["claim_dispositions"]), "result": result["result"]}, ensure_ascii=False, indent=2))
