"""核對指定教材相依差異、小節執行結果與既有實驗證據。"""
import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from scripts import evaluate as text_evaluation
from scripts import train
from scripts.course_experiments.behavior import _safety_records
from scripts.course_experiments.common import records_sha256, split_records
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(relative):
    return (ROOT / relative).read_bytes()


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    return {"command_argv": argv, "exit_code": result.returncode,
            "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def section(raw, name):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headings) if h[0].startswith(("## " + name + " ").encode()))
    end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
    return raw[headings[index].start():end]


previous_path = "docs/course-review-20261019/reviews/rv1019_technical_9-4/technical-9-4-current-01.json"
previous_raw = read(previous_path)
assert digest(previous_raw) == "e1e2e93d06713aca88c05310ecf17c6c9da2ecb2c393f7f8bb5fe4a2224b3376"
previous = json.loads(previous_raw)
assert previous["issues"] == []
assert digest(read(previous["original_report"]["path"])) == previous["original_report"]["sha256"]
source_hashes = []
for item in previous["sources"]:
    if item["kind"] == "repository_code":
        actual = digest(read(item["path"]))
        source_hashes.append({"id": item["id"], "path": item["path"],
                              "previous_sha256": item["sha256"], "current_sha256": actual,
                              "unchanged": actual == item["sha256"]})
assert {x["path"] for x in source_hashes if not x["unchanged"]} == {"course/training.md"}
old_artifacts = []
for item in previous["artifacts"]:
    actual = digest(read(item["path"]))
    assert actual == item["sha256"]
    old_artifacts.append({"id": item["id"], "path": item["path"], "sha256": actual})
old_verification = json.loads(read(next(a["path"] for a in previous["artifacts"] if a["id"] == "a-verification")))
old_execution = json.loads(read(next(a["path"] for a in previous["artifacts"] if a["id"] == "a-section")))

frozen = read("docs/course-review-20261019/freeze-01/sources/9.4.md")
assert section(read("course/chapters/09.md"), "9.4") == frozen
assert digest(frozen) == previous["source_sha256"]
freeze_path = "docs/course-review-20261019/freeze-07.json"
assert digest(read(freeze_path)) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
page = next(p for p in json.loads(read(freeze_path))["pages"] if p["page_id"] == "9.4")
assert page["source_sha256"] == digest(frozen) and page["figures_sha256"] == {}
run = ROOT / "outputs/reviewer-tools/runs/20261020T152732923653Z"
helper_files = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    content = (run / name).read_bytes()
    helper_files[name] = {"sha256": digest(content), "content": content.decode()}
assert json.loads(helper_files["execution.json"]["content"])["exit_code"] == 0
assert json.loads(helper_files["execution.json"]["content"])["source_sha256"] == digest(frozen)
assert json.loads(helper_files["extraction.json"]["content"])["svg_references"] == []
assert json.loads(helper_files["environment.json"]["content"])["guard_events"] == []
assert helper_files["stdout.txt"]["content"] == old_execution["files"]["stdout.txt"]["content"]
assert helper_files["stderr.txt"]["content"] == ""

old_training = command(["git", "show", "ecf6dec71280:course/training.md"])["stdout"].encode()
new_training = command(["git", "show", "c6b9339d2595:course/training.md"])["stdout"].encode()
assert digest(old_training) == next(s["sha256"] for s in previous["sources"] if s["id"] == "s-training")
assert new_training == read("course/training.md")
assert digest(new_training) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
training_diff = command(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
assert section(old_training, "T.5") == section(new_training, "T.5")
safety_command = ".venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda"
assert safety_command in section(new_training, "T.5").decode()
help_result = command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"])
spec = experiment_spec("safety")
assert (spec["module"], spec["function"]) == ("behavior", "run_safety")
assert spec["dependencies"] == ["style"] and spec["assets"] == ["pku-safe-rlhf"]

record_path = "docs/course-experiments/results/safety.json"
assert digest(read(record_path)) == old_verification["current_input_sha256"][record_path]
record = json.loads(read(record_path))
parts = split_records(_safety_records(), seed=record["seed"])
arithmetic = split_records(arithmetic_records(), seed=record["seed"])
split_checks = {}
for name, rows in parts.items():
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    assert digest(serialized) == record["results"]["data"][name]["sha256"]
    split_checks[name] = {"records": len(rows), "sha256": digest(serialized)}
tok = ByteTokenizer()
permission_counts = {}
for name in ("safety-only", "model"):
    training_rows = parts["train"] + (arithmetic["train"] if name == "model" else [])
    assert records_sha256(training_rows) == record["results"]["runs"][name]["training"]["records_sha256"]
    rows = [s for s in record["results"]["runs"][name]["safety"]["test"]["samples"] if s["kind"] == "permission"]
    assert rows == old_verification["runs"][name]["samples"] and len(rows) == 6
    assert all(s["exact"] and s["generated"] == s["expected"]
               and s["generated_ids"] == tok.encode(s["expected"]) + [tok.eos_id] for s in rows)
    permission_counts[name] = {"records": len(rows), "matches": sum(s["exact"] for s in rows)}

class CaptureModal:
    def __init__(self):
        self.examples = []
    def __call__(self, ids, labels, image=None, waveform=None):
        self.examples.append({"target": tok.decode(labels[labels != -100].tolist()),
                              "has_image": image is not None, "has_waveform": waveform is not None})
        return {"logits": torch.zeros(1), "labels": labels, "auxiliary": torch.tensor(0.0)}

modal_probes = {}
for task, expected in (("vision", ["circle", "square"]), ("audio", ["low", "high"]),
                       ("joint", ["circle,low", "square,high"])):
    args = train.parser().parse_args(["--task", task])
    assert args.data is None and train.prepare_examples(args, tok) is None
    capture = CaptureModal()
    choices = ["red", "circle", False, "blue", "square", True]
    with patch.object(train.random, "choice", side_effect=choices) as chooser, patch.object(train, "masked_loss", return_value=torch.tensor(0.0)):
        train.modal_loss(capture, task, tok, "cpu")
        train.modal_loss(capture, task, tok, "cpu")
        assert chooser.call_count == len(choices)
    assert [x["target"] for x in capture.examples] == expected
    modal_probes[task] = capture.examples
train_tree = ast.parse(read("scripts/train.py"))
main = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
main_source = ast.get_source_segment(read("scripts/train.py").decode(), main)
assert 'for step in range(start, start + steps)' in main_source and 'modal_loss(' in main_source
assert not any(isinstance(n, ast.Name) and n.id.startswith("evaluate") for n in ast.walk(main))
parser_options = [option for action in train.parser()._actions for option in action.option_strings]
assert not any("validation" in option or "test" in option for option in parser_options)
modal_model = MultiModalLM(TinyLM(ModelConfig(width=16, layers=1, heads=1, max_length=64)))
assert hasattr(modal_model, "language") and hasattr(modal_model, "vision") and hasattr(modal_model, "audio")
with patch.object(text_evaluation, "load_checkpoint", return_value=(modal_model, {"format_version": "multimodal-v1"})), patch.object(sys, "argv", ["evaluate.py", "unused.pt", "--device", "cpu"]):
    try:
        text_evaluation.main()
    except ValueError as error:
        rejection = str(error)
    else:
        raise AssertionError("多模態模型應由文字評估入口拒絕")
assert "TinyLM" in rejection
T6 = section(new_training, "T.6").decode()
assert "### 重跑第11章的固定實驗" in T6
assert ".venv/bin/python -m scripts.course_experiments.run --experiment vqa --device cuda" in T6
assert not re.search(r"^\.venv/bin/python scripts/evaluate\.py", T6, re.M)

vqa_path = "docs/course-experiments/results/vqa.json"
vqa = json.loads(read(vqa_path))
vqa_checks = {}
for name in ("projector_only", "partial", "all", "all_replay"):
    variant = vqa["results"]["variants"][name]
    before, after = variant["text_before"], variant["text_after"]
    before_prompts = [(x["messages"], x["expected"]) for x in before["samples"]]
    after_prompts = [(x["messages"], x["expected"]) for x in after["samples"]]
    assert before_prompts == after_prompts and len(before_prompts) == 10
    for scores in (before, after):
        assert sum(s["exact"] for s in scores["samples"]) == scores["matches"]
    vqa_checks[name] = {"paired_test_records": len(before_prompts), "before_matches": before["matches"],
                        "after_matches": after["matches"], "prompts_and_expected_unchanged": True}
modalities_tree = ast.parse(read("scripts/course_experiments/modalities.py"))
vqa_node = next(n for n in modalities_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_vqa")
vqa_source = ast.get_source_segment(read("scripts/course_experiments/modalities.py").decode(), vqa_node)
assert 'text_holdout = text_splits["test"]' in vqa_source
assert vqa_source.count('evaluate_lm(model.language, text_holdout, mode="sft")') == 2
assert 'replay = text_splits["train"]' in vqa_source

changed_sentences = [
    {"statement": "答案不同時先確認命令與檢查點，三個輸出只是抽查，不能直接代表整體能力。",
     "status": "verified", "evidence": "該段只列三道infer_modal命令；刪除對不存在固定驗證題的指引後，沒有把抽查當成完整評估。"},
    {"statement": "train.py所列vision/audio/joint路線的訓練例子逐步隨機產生，沒有另備固定驗證題。",
     "status": "verified", "evidence": "三條命令沒有--data；prepare_examples回None，訓練迴圈對每筆呼叫modal_loss；受控抽樣探針每次重新抽color/shape/high；main沒有評估路線或驗證切分。",
     "scope": "限定教材展示的無--data CLI命令，使用自己的--data時會改走檔案資料。"},
    {"statement": "固定測試題的比較做法在摺疊區的固定實驗。",
     "status": "verified", "evidence": "該區提供vqa固定命令；run_vqa及既有結果均含固定測試題比較。"},
    {"statement": "evaluate.py拒絕多模態checkpoint，所以所列CLI命令沒有檢查文字能力退步。",
     "status": "verified", "evidence": "以真正MultiModalLM物件、mock檔案載入執行evaluate.main，得到TinyLM限定錯誤；所列命令沒有文字前後測。",
     "scope": "只驗證型別邊界與指令範圍，不宣稱多模態模型不能抽出language另做評估。"},
    {"statement": "文字能力保留要在圖文訓練前後，用同一批文字測試題各核對一次；固定vqa實驗做了此對照。",
     "status": "verified", "evidence": "run_vqa兩次以相同text_holdout評估model.language；四支線text_before/text_after各10題的messages與expected逐筆相同，回放只取train。",
     "scope": "核對既有紀錄及現行原始碼；本次沒有重跑VQA訓練，亦未完整審閱被連結章頁。"},
]
input_paths = ["scripts/train.py", "scripts/evaluate.py", "tiny_perceptron/multimodal.py",
               "scripts/course_experiments/modalities.py", vqa_path, record_path]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-4/recheck-current-02.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "previous_report": {"path": previous_path, "sha256": digest(previous_raw)}, "previous_issues": previous["issues"],
    "source_hash_comparison": source_hashes, "unchanged_artifacts": old_artifacts,
    "source_sha256": digest(frozen),
    "section_execution": {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/09.md#9.4" --execute', "files": helper_files},
    "training_diff": training_diff, "training_T5_unchanged": True,
    "training_T5_sha256": digest(section(new_training, "T.5")), "safety_command": safety_command,
    "cli_help": help_result, "experiment_spec": spec, "split_checks": split_checks,
    "permission_counts": permission_counts, "modal_sampling_probes": modal_probes,
    "train_cli_options": parser_options, "multimodal_text_evaluation_rejection": rejection,
    "vqa_pair_checks": vqa_checks,
    "vqa_environment": {k: vqa[k] for k in ("device", "seed", "torch_version", "python_version", "revision")},
    "vqa_source_excerpt": vqa_source,
    "additional_input_sha256": {p: digest(read(p)) for p in input_paths},
    "modified_dependency_statements": changed_sentences,
    "claim_continuity": [{"claim_id": c["id"], "status": "verified", "basis": "原文未變；引用證據雜湊未變；T.5指令另核對；小節程式、資料切分與逐題token重新核對。"} for c in previous["claims"]],
    "result": "所有複查斷言通過；舊報告與artifact未改寫。",
}
output = DESTINATION / "recheck-current-02.json"
with output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": digest(output.read_bytes()), "result": result["result"]}, ensure_ascii=False))
