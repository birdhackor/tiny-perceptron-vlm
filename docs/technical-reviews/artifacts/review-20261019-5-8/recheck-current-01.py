"""核對凍結原文、來源變動、舊證據完整性與現行重做入口。"""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def command(args):
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return {"command_argv": args, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


original_path = "docs/course-review-20261019/reviews/rv1019_technical_5-8/technical-5-8-original-01.json"
raw = (ROOT / original_path).read_bytes()
assert sha(raw) == "2bb6dd48224c6ea9db9058aad7cf208389ef9ccc7fb8907cf8fb50939814516e"
original = json.loads(raw)
assert original["issues"] == []
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-8/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "original_report": {"path": original_path, "sha256": sha(raw)},
    "original_issues": original["issues"], "resolved_issue_ids": [],
}
freeze_path = ROOT / "docs/course-review-20261019/freeze-06.json"
assert sha(freeze_path.read_bytes()) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
freeze = json.loads(freeze_path.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == original["lesson_id"])
snapshot = (ROOT / page["snapshot"]).read_bytes()
assert sha(snapshot) == original["source_sha256"] == page["source_sha256"]
assert page["figures_sha256"] == {}
chapter = (ROOT / "course/chapters/05.md").read_bytes()
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
index = next(i for i, h in enumerate(headings) if h[0].startswith(b"## 5.8 "))
end = headings[index + 1].start() if index + 1 < len(headings) else len(chapter)
assert chapter[headings[index].start():end] == snapshot
receipt["source_freeze"] = {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path.read_bytes())}
receipt["page"] = {"snapshot": page["snapshot"], "source_sha256": sha(snapshot), "current_section_identical": True, "figure_sha256": {}}
receipt["repository_sources"] = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = sha((ROOT / source["path"]).read_bytes())
    receipt["repository_sources"].append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": actual, "unchanged": source["sha256"] == actual})
assert [s["path"] for s in receipt["repository_sources"] if not s["unchanged"]] == ["course/training.md"]
assert sha((ROOT / "course/training.md").read_bytes()) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
receipt["reused_artifacts"] = []
for artifact in original["artifacts"]:
    actual = sha((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    receipt["reused_artifacts"].append({"path": artifact["path"], "sha256": actual, "unchanged": True})
old_self = ARTIFACT / "self-check.json"
assert sha(old_self.read_bytes()) == "3f830a455b05eb5a8577392b33e92ebc6502476e56d91d41e6b7376d9c5011ed"
receipt["previous_self_check"] = {"path": str(old_self.relative_to(ROOT)), "sha256": sha(old_self.read_bytes()), "unchanged": True}
receipt["training_diff"] = command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
old = command(["git", "show", "5ef56bc756f3:course/training.md"])
new = command(["git", "show", "6a7aa1de5227:course/training.md"])
assert sha(old["stdout"].encode()) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
assert new["stdout"].encode() == (ROOT / "course/training.md").read_bytes()


def recipe(text):
    start = text.index('<details id="fixed-sft-recipe" markdown="1">')
    end = text.index("</details>", start) + len("</details>")
    return text[start:end]


old_recipe = recipe(old["stdout"])
new_recipe = recipe(new["stdout"])
assert old_recipe == new_recipe
required = [
    ".venv/bin/python scripts/fetch_training_assets.py --asset tinystories --asset chinese-poetry",
    ".venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda",
]
assert all(item in new_recipe for item in required)
receipt["training_dependency"] = {
    "old_revision": "5ef56bc756f3", "current_revision": "6a7aa1de5227",
    "old_sha256": sha(old["stdout"].encode()), "current_sha256": sha(new["stdout"].encode()),
    "inspected_locator": "故事、切詞與固定對話實驗", "recipe_unchanged": True,
    "old_recipe_sha256": sha(old_recipe.encode()), "current_recipe_sha256": sha(new_recipe.encode()),
    "current_recipe": new_recipe, "referenced_commands": required,
    "conclusion": "原文引用的重做命令與前置條件完整保留，來源變動沒有改變該主張。",
    "other_diff_scope": "裝置說明、多模態示例的變異提醒、量化前後比較提醒均不在本頁引用的固定配方區塊；本次不對那些頁面的完整主張另作正式裁決。",
}
python = str(ROOT / ".venv/bin/python")
receipt["read_only_commands"] = [
    command([python, "scripts/fetch_training_assets.py", "--help"]),
    command([python, "scripts/fetch_training_assets.py", "--list"]),
    command([python, "-m", "scripts.course_experiments.run", "--help"]),
    command([python, "-m", "scripts.course_experiments.run", "--list-assets", "real_text"]),
]
listed = receipt["read_only_commands"][-1]["stdout"].splitlines()
assert listed == ["assets/training/tinystories-v1.tar.gz", "assets/training/chinese-poetry-v1.tar.gz"]
receipt["read_code"] = {}
for path, names in [
    ("scripts/fetch_training_assets.py", ["main"]),
    ("scripts/course_experiments/run.py", ["experiment_spec", "list_assets", "main"]),
    ("scripts/course_experiments/text.py", ["run_real_text"]),
]:
    import ast
    text = (ROOT / path).read_text()
    tree = ast.parse(text)
    snippets = {node.name: ast.get_source_segment(text, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    receipt["read_code"][path] = {"sha256": sha((ROOT / path).read_bytes()), "definitions": snippets}
truth = [1, 2, 3, 4]
receipt["numeric_recheck"] = []
for prediction in [[1, 2, 0, 4], [1, 2, 0, 0], [1, 2, 3, 4]]:
    count = sum(a == b for a, b in zip(truth, prediction, strict=True))
    receipt["numeric_recheck"].append({"prediction": prediction, "correct": count, "accuracy": count / len(truth), "exact_match": prediction == truth})
receipt["negative_log_recheck"] = {str(p): -math.log(p) for p in [0.6, 0.9]}
receipt["nll_recheck"] = {}
for name in ["text_foundation", "real_text"]:
    record = json.loads((ROOT / "docs/course-experiments/results" / (name + ".json")).read_bytes())
    body = record["results"] if name == "text_foundation" else record["results"]["runs"]["tinystories"]
    for when in ["before", "after"]:
        validation = body[when]["validation"]
        computed = validation["nll_sum"] / validation["effective_tokens"]
        assert math.isclose(computed, validation["nll"], rel_tol=0, abs_tol=1e-12)
        receipt["nll_recheck"][name + "_" + when] = {"recorded": validation["nll"], "recomputed": computed, "effective_tokens": validation["effective_tokens"], "rounded": f"{computed:.5f}"}
receipt["claim_continuation"] = [{
    "claim_id": claim["id"], "status": "verified",
    "basis": "原文未變；原始來源版本及引用 artifact 未變，原判斷仍成立。" if claim["id"] != "c34" else "全文讀取來源差異，固定配方區塊逐 byte 相同；現行 CLI 列出的資產及參數與命令一致。",
} for claim in original["claims"]]
receipt["artifact_access_exception"] = "僅讀原報告明確列名的 artifact、自己的舊自我檢查，以及本次 section_facts 指定產生的執行紀錄；未讀其他角色報告或其他 outputs。"
manifest = json.loads((ROOT / "assets/training/manifest.json").read_bytes())
receipt["asset_availability"] = [{"id": asset["id"], "path": asset["archive"], "lfs_pointer": (ROOT / asset["archive"]).read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1\n")} for asset in manifest["assets"] if asset["id"] in ["tinystories", "chinese-poetry"]]
receipt["limitations"] = ["未重新訓練、未由原檢查點重新生成。", "未下載 LFS 資料包或逐篇重數故事 token；既有實驗的證據與適用範圍沿用初審核對。", "未擴大審閱 training.md 其他小節；僅判斷其變動是否影響本頁引用的重做配方。", "原始論文及官方文件採用初審實讀版本與雜湊收據，本次未重新擷取。"]
write_new(ARTIFACT / "recheck-current-01.json", receipt)
run = ROOT / "outputs/reviewer-tools/runs/20261020T102237296425Z"
names = ["section.md", "fence-1.py", "bootstrap.py", "extraction.json", "environment.json", "execution.json", "stdout.txt", "stderr.txt"]
bundle = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.8" --execute', "run_directory": str(run.relative_to(ROOT)), "files": {}}
for name in names:
    raw = (run / name).read_bytes()
    bundle["files"][name] = {"sha256": sha(raw), "text": raw.decode()}
assert json.loads(bundle["files"]["execution.json"]["text"])["exit_code"] == 0
assert sha(bundle["files"]["section.md"]["text"].encode()) == original["source_sha256"]
write_new(ARTIFACT / "section-execution-current-01.json", bundle)
print(json.dumps({"verdict": "pass", "continued_claims": len(receipt["claim_continuation"]), "changed_sources": [s["path"] for s in receipt["repository_sources"] if not s["unchanged"]], "recipe_unchanged": True, "section_stdout": bundle["files"]["stdout.txt"]["text"]}, ensure_ascii=False))
