"""核對引用配方、頁面凍結與既有證據；僅以不覆寫方式新增收據。"""

import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

ROOT = Path.cwd()
ART = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_6-5/technical-6-5-current-01.json"
SOURCE = ROOT / "docs/course-review-20261019/freeze-05/sources/6.5.md"
FREEZE = ROOT / "docs/course-review-20261019/freeze-07.json"
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T152401253725Z"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def fixed_recipe(text):
    start = text.index('<details id="fixed-sft-recipe" markdown="1">')
    end = text.index("</details>", start) + len("</details>")
    return text[start:end]


assert sha(PREVIOUS.read_bytes()) == "b1934460f98eb7ef5f57d58d651c2c211d0497c7c7863edbcdca5a4879b23a7f"
previous = read_json(PREVIOUS)
assert sha(SOURCE.read_bytes()) == previous["source_sha256"] == "9bff6436d575412635d1ea4bba877f131a08131222974e9023a513cfbe9a90d5"
assert sha(FREEZE.read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
page = next(p for p in read_json(FREEZE)["pages"] if p["page_id"] == previous["lesson_id"])
assert page["source_sha256"] == previous["source_sha256"]
assert page["figures_sha256"] == previous["figure_sha256"]

source_receipts = []
for source in previous["sources"]:
    if source["kind"] != "repository_code":
        continue
    current = sha((ROOT / source["path"]).read_bytes())
    unchanged = current == source["sha256"]
    if source["path"] == "course/training.md":
        assert current == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
    else:
        assert unchanged
    source_receipts.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current, "unchanged": unchanged})

artifact_receipts = []
for artifact in previous["artifacts"]:
    current = sha((ROOT / artifact["path"]).read_bytes())
    assert current == artifact["sha256"]
    artifact_receipts.append({"path": artifact["path"], "sha256": current, "unchanged": True})

history = {}
for revision in ["ecf6dec71280", "c6b9339d2595"]:
    result = command(["git", "show", revision + ":course/training.md"])
    text = result["stdout"]
    history[revision] = {"command_argv": result["command_argv"], "exit_code": 0, "file_sha256": sha(text.encode()), "recipe_sha256": sha(fixed_recipe(text).encode())}
    history[revision]["recipe_matches_current"] = fixed_recipe(text) == fixed_recipe((ROOT / "course/training.md").read_text())
assert history["ecf6dec71280"]["file_sha256"] == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert history["c6b9339d2595"]["file_sha256"] == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
assert all(row["recipe_matches_current"] for row in history.values())
diff = command(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
recipe_command = ".venv/bin/python -m scripts.course_experiments.run --experiment tokenizer --device cuda"
assert recipe_command in fixed_recipe((ROOT / "course/training.md").read_text())
cli = [command([".venv/bin/python", "-m", "scripts.course_experiments.run", "--help"]), command([".venv/bin/python", "-m", "scripts.course_experiments.run", "--list-assets", "tokenizer"])]

names = ["stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py", "fence-2.py"]
copied = {name: (RUN / name).read_text() for name in names}
environment, execution, extraction = [json.loads(copied[name]) for name in ["environment.json", "execution.json", "extraction.json"]]
assert execution["exit_code"] == 0 and copied["stderr.txt"] == ""
assert extraction["source_sha256"] == previous["source_sha256"]
assert len(extraction["python_fences"]) == 2
prior_section = read_json(ART / "recheck-evidence-current-02.json")["section_execution"]["copied_files"]
assert all(copied[name] == prior_section[name] for name in ["stdout.txt", "fence-1.py", "fence-2.py"])

figures = []
old_figures = read_json(ART / "recheck-evidence-current-02.json")["figure_inspection"]
for figure in old_figures:
    assert sha((ROOT / figure["path"]).read_bytes()) == figure["sha256"]
    assert sha((ROOT / figure["viewed_png"]["path"]).read_bytes()) == figure["viewed_png"]["sha256"]
    figures.append({"path": figure["path"], "sha256": figure["sha256"], "viewed_png": figure["viewed_png"], "inspection_note": "本次再次用view_image觀看指定PNG並讀SVG；數字、方向、共同分母與長條尺度皆相符。"})

initial = read_json(ART / "verification.json")
record_path = ROOT / "docs/course-experiments/results/tokenizer.json"
assert sha(record_path.read_bytes()) == initial["input_sha256"]["docs/course-experiments/results/tokenizer.json"]
record = read_json(record_path)
metrics = {}
for name, run in record["results"]["runs"].items():
    metrics[name] = {}
    for phase in ["before", "after"]:
        for split in ["test", "validation"]:
            entry = run[phase][split]
            total = entry["mean_token_nll"] * entry["effective_tokens"]
            bpb = total / (entry["raw_utf8_bytes"] * math.log(2))
            assert math.isclose(bpb, entry["bpb_including_eos_boundary_targets"], abs_tol=2e-15)
            metrics[name][f"{phase}.{split}"] = {"mean_nll": entry["mean_token_nll"], "targets": entry["effective_tokens"], "raw_bytes": entry["raw_utf8_bytes"], "total_nll": total, "recomputed_bpb": bpb}

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-5/verify-training-current-01.py",
    "environment": {"python": platform.python_version(), "torch": environment["torch"], "device": "cpu"},
    "result": "核對完成",
    "previous_report": {"path": str(PREVIOUS.relative_to(ROOT)), "sha256": sha(PREVIOUS.read_bytes())},
    "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE.read_bytes()), "unchanged": True},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE.read_bytes())},
    "frozen_page": page,
    "repository_sources": source_receipts,
    "reused_artifacts": artifact_receipts,
    "training_history": history,
    "training_diff": diff,
    "training_recipe": {"path": "course/training.md", "sha256": sha((ROOT / "course/training.md").read_bytes()), "locator": "固定課程配方", "recipe_sha256": sha(fixed_recipe((ROOT / "course/training.md").read_text()).encode()), "command_excerpt": recipe_command, "matches_both_requested_revisions": True},
    "change_scope": "頁面正文、圖與程式未改。training.md變動在模態訓練的抽查與文字能力對照說明；本頁引用的固定tokenizer配方整個摺疊區與指定兩版Git內容逐字相同。本次只判斷它是否仍支持本頁重做命令，不擴張為對模態說明的獨立審閱。",
    "read_only_commands": cli,
    "section_execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/06.md#6.5\" --execute", "copied_files": copied, "copied_file_sha256": {name: sha((RUN / name).read_bytes()) for name in names}, "unchanged_fences_and_stdout": True},
    "figure_inspection": figures,
    "record_sha256": sha(record_path.read_bytes()),
    "record_metrics": metrics,
    "reuse_scope": "原始來源與全部沿用artifact已核對hash；相關核心程式與實驗JSON未變。既有artifact中的training.md舊hash只作歷史收據，本次的現行來源依training_recipe另核對。",
    "limitations": "未重訓或重評權重；資料重建與tokenizer長度的舊執行證據沿用，不宣稱本次再次重建資料或量測跨平台效果。",
}
output = ART / "training-recheck-current-01.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"result": "核對完成", "path": str(output.relative_to(ROOT)), "sha256": sha(output.read_bytes()), "recipe_unchanged": True, "source_sha256": receipt["source"]["sha256"]}, ensure_ascii=False))
