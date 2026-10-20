"""核對入口頁的來源差異、舊證據完整性，並重跑原有短程式。"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-course"
REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_course/technical-course-current-01.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


previous = load(REPORT)
prior_output = load(BASE / "audit_course-output.json")
prior_site = load(BASE / "audit_site-output.json")
differences = load(BASE / "source-diffs-current-02.json")
notes = {'course/chapters/03.md': '把最高分落在自己的說法收窄到指定正交單位向量例子；匹配及混合資訊的主題仍成立。當次執行兩個矩陣，確認共用表的對角分數1、非對角0，分開表剛好相反。', 'course/chapters/0A.md': '新增單元結語，總結輸入是否包含資料、回答是否有來源支持、條件改動時是否走樣；原有外部資料及長文導覽保留，補充結果與重跑命令確實存在。', 'course/chapters/14.md': '逐 token 生成的回讀連結改指向實際接字的例子；當次解析連結並執行其生成程式，另執行手工循環示意，兩個軸保持區分。', 'course/training.md': '收窄三條短訓路線的驗收範圍；modal_loss 在未提供資料檔時逐次隨機生成，evaluate 拒絕完整多模態模型。固定 vqa 另以同一份文字留出題評估前後，頁面所說操作入口仍成立。'}

integrity = []
for item in [dict(path=str(REPORT.relative_to(ROOT)), sha256="144a62fd6b6a102abfe2eea1e08b4bbb8ede55d6056a45598fdc0b40f7f0709b"), *previous["artifacts"]]:
    current = sha(ROOT / item["path"])
    integrity.append({"path": item["path"], "expected_sha256": item["sha256"], "observed_sha256": current, "unchanged": current == item["sha256"]})

source_states = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        current = sha(ROOT / source["path"])
        source_states.append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": current, "changed": current != source["sha256"]})

reviewed = []
for row in differences["sources"]:
    actual = subprocess.check_output(row["command"], cwd=ROOT, text=True)
    assert actual == row["diff"]
    source_id = next(x["id"] for x in source_states if x["path"] == row["path"])
    affected = [c["id"] for c in previous["claims"] if any(e["source_id"] == source_id for e in c["evidence"])]
    reviewed.append({"path": row["path"], "source_id": source_id, "current_sha256": sha(ROOT / row["path"]), "diff_sha256": hashlib.sha256(actual.encode()).hexdigest(), "base_matches_original": row["base_matches_original"], "target_matches_current": row["target_matches_current"], "affected_claim_ids": affected, "status": "verified", "inspection_note": notes[row["path"]]})

command = [sys.executable, str(BASE / "audit_course.py")]
process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
current_audit = json.loads(process.stdout)

raw_receipts = []
for key in ("product_training", "moe_training"):
    for row in prior_output[key]:
        observed = sha(ROOT / row["path"])
        raw_receipts.append({"path": row["path"], "expected_sha256": row["sha256"], "observed_sha256": observed, "unchanged": row["sha256"] == observed})
for key in ("product_final", "moe_final"):
    row = prior_output[key]
    observed = sha(ROOT / row["path"])
    raw_receipts.append({"path": row["path"], "expected_sha256": row["sha256"], "observed_sha256": observed, "unchanged": row["sha256"] == observed})

site_url = "https://birdhackor.github.io/tiny-perceptron-vlm/build-info.json"
site_command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "35", site_url]
site_process = subprocess.run(site_command, text=True, capture_output=True, check=True)
site_info = json.loads(site_process.stdout)
site_check = {"url": site_url, "command": site_command, "returncode": site_process.returncode,
              "observed": site_info, "matches_preserved_site_receipt": site_info == prior_site["build_info_after"],
              "response_sha256": hashlib.sha256(site_process.stdout.encode()).hexdigest()}

old_chapters = {x["path"]: x for x in prior_output["chapter_contents"]}
headings = [{"path": x["path"], "title_unchanged": x["title"] == old_chapters[x["path"]]["title"],
             "headings_unchanged": x["headings"] == old_chapters[x["path"]]["headings"]} for x in current_audit["chapter_contents"]]
freeze = ROOT / "docs/course-review-20261019/freeze-07.json"
freeze_record = load(freeze)
page = next(x for x in freeze_record["pages"] if x["page_id"] == "course")


import ast
import contextlib
import io
import re
from unittest import mock

sys.path.insert(0, str(ROOT))
import torch
from scripts import evaluate as text_evaluation
from scripts import train as train_entry
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM


def section(path, section_id):
    text = (ROOT / path).read_text()
    matches = list(re.finditer(r"^## ([A-Z\d]+\.\d+) (.+)$", text, re.M))
    index = next(i for i, item in enumerate(matches) if item[1] == section_id)
    return text[matches[index].end():matches[index+1].start() if index+1 < len(matches) else len(text)]


def execute_blocks(path, section_id):
    content = section(path, section_id)
    namespace = {"torch": torch}
    buffer = io.StringIO()
    blocks = re.findall(r"```python\n(.*?)```", content, re.S)
    with contextlib.redirect_stdout(buffer):
        for block in blocks:
            exec(compile(block, path, "exec"), namespace)
    return namespace, {"path": path, "section_id": section_id, "python_block_count": len(blocks), "stdout": buffer.getvalue()}


torch.set_num_threads(1)
torch.manual_seed(42)
matching, matching_receipt = execute_blocks("course/chapters/03.md", "3.3")
matching_receipt["shared"] = (matching["has"] @ matching["has"].T).tolist()
matching_receipt["separate"] = (matching["need"] @ matching["has"].T).tolist()
loop, loop_receipt = execute_blocks("course/chapters/14.md", "14.12")
loop_text = section("course/chapters/14.md", "14.12")
link = re.search(r"\[([^\]]+)\]\((01\.md)#([^\)]+)\)", loop_text)
assert link is not None
_, generation_receipt = execute_blocks("course/chapters/01.md", link[3])
generation_receipt["linked_from"] = "course/chapters/14.md"

external_text = (ROOT / "course/chapters/0A.md").read_text()
external_sections = re.findall(r"^## (A\.\d+) (.+)$", external_text, re.M)
external_checks = []
for identifier, title in external_sections:
    part = section("course/chapters/0A.md", identifier)
    external_checks.append({"id": identifier, "title": title, "has_concrete_python_example": "```python\n" in part,
        "supplement_titles": re.findall(r"<summary>(.*?)</summary>", part),
        "rag_receipt_linked": "../../docs/course-experiments/results/rag.json" in part})
rerun = re.search(r"python scripts/course_experiments/run.py --experiment rag --device cpu", external_text)

multimodal = MultiModalLM(TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=128))).eval()
modal_calls = []
original_choice = train_entry.random.choice
for task in ("vision", "audio", "joint"):
    for _ in range(2):
        draws = []
        def choose(items):
            selected = original_choice(items)
            draws.append({"population": list(items), "selected": selected})
            return selected
        with mock.patch.object(train_entry.random, "choice", side_effect=choose), torch.no_grad():
            value = train_entry.modal_loss(multimodal, task, ByteTokenizer(), "cpu")
        modal_calls.append({"task": task, "random_choices": draws, "loss": float(value), "finite_loss": bool(torch.isfinite(value))})
try:
    text_evaluation.evaluate(multimodal, [])
    rejection = {"rejected": False}
except ValueError as error:
    rejection = {"rejected": True, "exception": type(error).__name__, "message": str(error)}

vqa_path = ROOT / "docs/course-experiments/results/vqa.json"
vqa = load(vqa_path)
vqa_variants = {}
for name, value in vqa["results"]["variants"].items():
    if "text_before" not in value:
        continue
    vqa_variants[name] = {key: {k: v for k,v in value[key].items() if not isinstance(v, (dict, list))} for key in ("text_before", "text_after")}
    vqa_variants[name]["text_probe_is_training_diagnostic"] = value["text_probe_is_training_diagnostic"]

extra_paths = ["scripts/train.py", "scripts/evaluate.py", "tiny_perceptron/multimodal.py", "scripts/course_experiments/modalities.py"]
main_ast = next(x for x in ast.parse((ROOT / "scripts/train.py").read_text()).body if isinstance(x, ast.FunctionDef) and x.name == "main")
called_names = sorted({x.func.id if isinstance(x.func, ast.Name) else x.func.attr for x in ast.walk(main_ast) if isinstance(x, ast.Call) and isinstance(x.func, (ast.Name, ast.Attribute))})
current_audit["dependency_checks"] = {
    "matching_example": matching_receipt, "loop_example": loop_receipt, "generation_link": generation_receipt,
    "external_material_sections": external_checks, "rag_rerun_command_present": rerun is not None,
    "modal_loss_calls": modal_calls, "evaluate_rejection": rejection,
    "training_main_called_function_names": called_names,
    "training_main_calls_evaluate": "evaluate" in called_names,
    "fixed_vqa": {"path": str(vqa_path.relative_to(ROOT)), "sha256": sha(vqa_path), "text_data": vqa["results"]["text_data"], "variants": vqa_variants, "scope": vqa["results"]["scope"]},
    "extra_repository_sources": [{"path": p, "sha256": sha(ROOT / p)} for p in extra_paths],
    "scope": "執行短算例及未更新權重的模態 forward；只讀取固定實驗紀錄，沒有重新訓練或載入正式成品。"
}

current_audit["recheck"] = {
    "executed_at": datetime.now(timezone.utc).isoformat(),
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-course/recheck_course-current-02.py",
    "child_audit_command": command, "child_audit_returncode": process.returncode,
    "previous_issues": previous["issues"], "immutable_evidence": integrity,
    "repository_sources": source_states, "source_changes": reviewed,
    "raw_training_and_test_receipts": raw_receipts, "site": site_check,
    "chapter_headings": headings,
    "source_diffs": {"path": str((BASE / "source-diffs-current-02.json").relative_to(ROOT)), "sha256": sha(BASE / "source-diffs-current-02.json")},
    "freeze": {"path": str(freeze.relative_to(ROOT)), "sha256": sha(freeze), "page": page},
    "page_text_unchanged": (ROOT / "course/README.md").read_bytes() == (ROOT / page["snapshot"]).read_bytes(),
    "scope": "逐項判斷來源變動對入口頁全部主張的影響；不是替各來源章節新增的每一句話另作完整技術審閱。原始程式重跑仍不包含長時間訓練或雲端寫入。",
}
print(json.dumps(current_audit, ensure_ascii=False, indent=2))
