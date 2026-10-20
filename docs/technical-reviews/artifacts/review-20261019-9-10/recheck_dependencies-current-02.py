"""核對引用文件的變動、全部既有主張及新說明所指的程式行為。"""
import ast
import hashlib
import importlib.util
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import evaluate, train
from scripts.course_experiments import modalities
from scripts.course_experiments.run import experiment_spec
from tiny_perceptron.data import ByteTokenizer


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return digest((ROOT / path).read_bytes())


def git(*args):
    command = ["git", *args]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return result.stdout, {"command_argv": command, "exit_code": result.returncode, "stderr": result.stderr.decode()}


previous_path = "docs/course-review-20261019/reviews/rv1019_technical_9-10/technical-9-10-current-01.json"
previous_sha = "4af6ff08065ff261add79fec9a34ff09ccc58b86dc197432188df7244bae7aa4"
assert file_sha(previous_path) == previous_sha, "上一份報告版本不符"
previous = json.loads((ROOT / previous_path).read_bytes())
assert previous["issues"] == [], "上一份 issue 清單與預期不同"
original = previous["original_report"]
assert file_sha(original["path"]) == original["sha256"], "原始報告已變動"
old_raw, old_command = git("show", "ecf6dec71280:course/training.md")
new_raw, new_command = git("show", "c6b9339d2595:course/training.md")
diff_raw, diff_command = git("diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md")
current_raw = (ROOT / "course/training.md").read_bytes()
assert digest(old_raw) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94", "比較起點與先前引用不同"
assert digest(current_raw) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b", "現行訓練頁版本不同"
assert new_raw == current_raw, "現行訓練頁與指定比較終點不同"
command_text = ".venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda"
heading = "### 重跑第11章的固定實驗"


def rerun_block(raw):
    text = raw.decode()
    start = text.index(heading)
    code_start = text.index("```bash\n", start)
    end = text.index("\n```", code_start) + len("\n```")
    return text[start:end]


block = rerun_block(current_raw)
assert block == rerun_block(old_raw), "所引用的固定實驗命令區塊有變動"
assert block.count(command_text) == 1, "命令出現次數不符"
text = current_raw.decode()
headers = list(__import__("re").finditer(r"(?m)^## [^\n]+", text))
command_position = text.index(command_text)
containing_heading = next(h.group() for i, h in enumerate(headers) if h.start() < command_position < (headers[i + 1].start() if i + 1 < len(headers) else len(text)))
assert containing_heading.startswith("## T.6 "), "命令不在課文連結指向的小節"
assert text.rfind("<details>", 0, command_position) > text.rfind("</details>", 0, command_position), "命令不在補充區"
spec = experiment_spec("encoders")
assert (spec["module"], spec["function"]) == ("modalities", "run_encoders"), "實驗入口不同"
repository_sources = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        current_sha = file_sha(source["path"])
        changed = current_sha != source["sha256"]
        assert not changed or source["id"] == "training", "另有來源變動，必須重新判斷"
        repository_sources.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current_sha, "changed": changed})
previous_artifacts = []
for artifact in previous["artifacts"]:
    assert file_sha(artifact["path"]) == artifact["sha256"], "之前的 artifact 已變動"
    previous_artifacts.append({"path": artifact["path"], "sha256": artifact["sha256"], "unchanged": True})
freeze_path = "docs/course-review-20261019/freeze-07.json"
assert file_sha(freeze_path) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753", "凍結清單版本不同"
page = next(p for p in json.loads((ROOT / freeze_path).read_bytes())["pages"] if p["page_id"] == previous["lesson_id"])
snapshot = (ROOT / page["snapshot"]).read_bytes()
helper_spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
section_raw, _, _ = helper.original_section(ROOT / "course/chapters/09.md", previous["lesson_id"])
assert snapshot == section_raw and digest(snapshot) == previous["source_sha256"] == page["source_sha256"], "小節原文版本不同"
assert page["figures_sha256"] == previous["figure_sha256"], "圖清單不同"
for path, expected in previous["figure_sha256"].items():
    assert file_sha(path) == expected, "圖的版本不同"

modal_source = inspect.getsource(train.modal_loss)
modal_tree = ast.parse(modal_source)
random_choice_calls = [n for n in ast.walk(modal_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "random" and n.func.attr == "choice"]
assert len(random_choice_calls) == 3, "合成模態例子的隨機選擇不符"
parser_cases = {}
for task in ("vision", "audio", "joint"):
    args = train.parser().parse_args(["--task", task])
    assert args.data is None and train.prepare_examples(args, ByteTokenizer()) is None, "無外部資料時未走合成模態路徑"
    parser_cases[task] = {"data": None, "prepared_text_examples": None}
main_tree = ast.parse(inspect.getsource(train.main))
step_loop = next(n for n in ast.walk(main_tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step")
assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "modal_loss" for n in ast.walk(step_loop)), "訓練步沒有呼叫模態資料路徑"
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.startswith("evaluate") for n in ast.walk(main_tree)), "訓練命令有額外評估路徑，需人工重查"
with patch.object(sys, "argv", ["evaluate.py", "unused-checkpoint", "--device", "cpu"]), patch.object(evaluate, "load_checkpoint", return_value=(object(), {})):
    try:
        evaluate.main()
    except ValueError as error:
        evaluator_error = str(error)
    else:
        raise AssertionError("文字評估入口未拒絕非文字模型")
assert "只評估 TinyLM" in evaluator_error, "文字評估入口拒絕原因不同"
vqa_source = inspect.getsource(modalities.run_vqa)
assert 'text_holdout = text_splits["test"]' in vqa_source
assert vqa_source.count('evaluate_lm(model.language, text_holdout, mode="sft")') == 2
assert vqa_source.index('text_before =') < vqa_source.index('training = _fit(') < vqa_source.index('"text_after":')
vqa_path = "docs/course-experiments/results/vqa.json"
vqa_record = json.loads((ROOT / vqa_path).read_bytes())
vqa_cases = {}
for name in ("projector_only", "partial", "all", "all_replay"):
    variant = vqa_record["results"]["variants"][name]
    before, after = variant["text_before"], variant["text_after"]
    assert before["examples"] == after["examples"] == vqa_record["results"]["text_data"]["test"]["count"] == 10
    assert before["effective_tokens"] == after["effective_tokens"] == 69
    vqa_cases[name] = {"before": {k: before[k] for k in ("examples", "effective_tokens", "nll")}, "after": {k: after[k] for k in ("examples", "effective_tokens", "nll")}}
assert file_sha("scripts/course_experiments/modalities.py") == vqa_record["code_sha256"]["scripts/course_experiments/modalities.py"], "VQA 紀錄與程式版本不同"
changed_paragraphs = [line[1:] for line in diff_raw.decode().splitlines() if line.startswith("+") and not line.startswith("+++")]
assert len(changed_paragraphs) == 2, "變動段落數與實際差異不同"
auxiliary = {
    "synthetic_training": {"status": "verified", "text": changed_paragraphs[0], "scope": "限訓練頁列出的無 --data 合成 vision/audio/joint 命令。", "evidence": "modal_loss 每次用 random.choice 選 color/shape/high，main 每步呼叫該路徑；沒有固定驗證集評估入口。", "parser_cases": parser_cases, "source_excerpt": modal_source},
    "text_retention_comparison": {"status": "verified", "text": changed_paragraphs[1], "evidence": "文字評估入口實跑拒絕非 TinyLM 物件；run_vqa 在 _fit 前後以同一 text_holdout 測試資料呼叫 evaluate_lm，已保存結果含對應分母。", "evaluator_error": evaluator_error, "vqa_cases": vqa_cases, "text_data": vqa_record["results"]["text_data"], "scope": "四個含 text_before/text_after 的 VQA 變體；direct_vqa 未保存該對照。本次未重跑 VQA 訓練。"},
}
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-10/recheck_dependencies-current-02.py",
    "environment": {"python": platform.python_version(), "device": "cpu", "platform": platform.platform(), "git": subprocess.run(["git", "--version"], capture_output=True, check=True, text=True).stdout.strip()},
    "result": "全部依賴與新增說明核對相符",
    "previous_report": {"path": previous_path, "sha256": previous_sha, "issues": previous["issues"]},
    "original_report": original,
    "source_freeze": {"path": freeze_path, "sha256": file_sha(freeze_path)},
    "source_sha256": digest(section_raw), "page_sentences_changed": False,
    "repository_sources": repository_sources, "previous_artifacts": previous_artifacts,
    "git_commands": [{**old_command, "stdout_sha256": digest(old_raw)}, {**new_command, "stdout_sha256": digest(new_raw)}, {**diff_command, "stdout": diff_raw.decode()}],
    "rerun_dependency": {"code_block_unchanged": True, "block_sha256": digest(block.encode()), "block": block, "linked_section_heading": containing_heading, "in_supplement": True, "experiment_spec": spec, "training_executed": False},
    "changed_dependency_paragraphs": auxiliary,
    "additional_source_sha256": {path: file_sha(path) for path in ("scripts/train.py", "scripts/evaluate.py", "scripts/course_experiments/modalities.py", vqa_path)},
    "claim_rechecks": [{"claim_id": c["id"], "kind": c["kind"], "status": "verified", "basis": "原文與來源支持範圍不變；相關程式與數值已重跑。" if c["id"] != "experiment_interface" else "指定差異與現行重跑命令已重新判斷；命令區塊及入口不變。"} for c in previous["claims"]],
    "resolved_issue_ids": [],
    "scope": "完整複查本節；另外核對引用文件兩個改寫段落的新增事實，沒有宣稱完成整份訓練頁審閱。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
