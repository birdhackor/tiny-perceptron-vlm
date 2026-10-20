"""核對訓練配方依賴、隨機模態例子與固定文字能力對照。"""

import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts import train
from scripts.evaluate import evaluate
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(argv):
    completed = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, check=True)
    return {"command_argv": argv, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


class CaptureInputs:
    def __init__(self):
        self.rows = []

    def __call__(self, ids, labels, *, image=None, waveform=None):
        self.rows.append({"answer": ByteTokenizer().decode(labels[labels >= 0].tolist()), "has_image": image is not None, "has_audio": waveform is not None})
        return {"labels": labels, "logits": torch.zeros(len(ids), ByteTokenizer.vocab_size), "auxiliary": torch.tensor(0.0)}


def main():
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    previous_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_18-1/technical-18-1-current-01.json"
    previous = json.loads(previous_path.read_bytes())
    assert digest(previous_path.read_bytes()) == "1245e95c42ff7da6558b368fc66ed3f9b41b2e4bbc8134005b289f07082063cd"
    source_checks = []
    for source in previous["sources"]:
        if source["kind"] == "repository_code":
            current = digest((ROOT / source["path"]).read_bytes())
            source_checks.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": current, "unchanged": source["sha256"] == current})
    assert [s["path"] for s in source_checks if not s["unchanged"]] == ["course/training.md"], "引用的來源有未預期變動"
    artifact_checks = [{"path": a["path"], "sha256": a["sha256"], "unchanged": digest((ROOT / a["path"]).read_bytes()) == a["sha256"]} for a in previous["artifacts"]]
    assert all(a["unchanged"] for a in artifact_checks), "先前證據內容已變動"
    old = subprocess.check_output(["git", "show", "ecf6dec71280:course/training.md"], cwd=ROOT)
    current = (ROOT / "course/training.md").read_bytes()
    assert digest(old) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    assert digest(current) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
    assert current == subprocess.check_output(["git", "show", "c6b9339d2595:course/training.md"], cwd=ROOT)
    diff = run(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
    fixed_sections = []
    for section in ("T.4", "T.5", "T.8"):
        pattern = rb"(?ms)^## " + re.escape(section.encode()) + rb" .*?(?=^## |\Z)"
        old_body = re.search(pattern, old).group()
        body = re.search(pattern, current).group()
        assert old_body == body, "固定教師配方有實質變動"
        fixed_sections.append({"heading": body.splitlines()[0].decode(), "previous_sha256": digest(old_body), "current_sha256": digest(body), "identical_bytes": True})
    section_run = (ROOT / sys.argv[1]).resolve()
    section_output = {name: json.loads((section_run / name).read_bytes()) for name in ("extraction.json", "execution.json", "environment.json")}
    section_output["stdout"] = (section_run / "stdout.txt").read_text()
    section_output["stderr"] = (section_run / "stderr.txt").read_text()
    assert (section_run / "section.md").read_bytes() == (ROOT / "docs/course-review-20261019/freeze-01/sources/18.1.md").read_bytes()
    assert section_output["execution.json"]["exit_code"] == 0
    for a in section_output["execution.json"]["artifacts"]:
        assert digest((section_run / a["path"]).read_bytes()) == a["sha256"], "執行證據的雜湊不符"
    probe = run([str(ROOT / ".venv/bin/python"), str(Path(__file__).parent / "verify.py")])
    observations = json.loads(probe["stdout"])
    expected = json.loads((Path(__file__).parent / "verification.json").read_bytes())
    expected["input_sha256"]["course/training.md"] = digest(current)
    assert observations == expected, "補充算例或教師紀錄出現其他差異"
    recipe_commands = re.findall(r"(?m)^\.venv/bin/python scripts/train\.py --task (?:vision|audio|joint) .*", current.decode())
    assert len(recipe_commands) == 3, "無法唯一對應三條模態訓練路線"
    sampler_checks = []
    for command in recipe_commands:
        args = train.parser().parse_args(shlex.split(command)[2:])
        assert args.data is None and train.prepare_examples(args, ByteTokenizer()) is None
        capture = CaptureInputs()
        with patch("scripts.train.random.choice", side_effect=["red", "circle", False, "blue", "square", True]) as choice:
            losses = [float(train.modal_loss(capture, args.task, ByteTokenizer(), "cpu")) for _ in range(2)]
        assert choice.call_count == 6, "模態生成器未在每次呼叫抽取新屬性"
        sampler_checks.append({"command": command, "task": args.task, "data": None, "prepared_examples": None, "random_choice_calls": choice.call_count, "rows": capture.rows, "losses": losses, "scope": "固定兩組mock抽樣值以驗證無資料檔分支，每次呼叫重建答案與輸入；沒有更新模型權重。"})
    main_node = next(n for n in ast.parse((ROOT / "scripts/train.py").read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = sorted({n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(main_node) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))})
    assert not any("evaluat" in name for name in calls), "CLI流程含有未核對的評估呼叫"
    torch.manual_seed(42)
    modal = MultiModalLM(TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=128)))
    try:
        evaluate(modal, [])
    except ValueError as error:
        rejection = {"type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("文字評估入口沒有拒絕多模態模型")
    text_result = evaluate(modal.language, [{"messages": [{"role": "user", "content": "a?"}, {"role": "assistant", "content": "b"}]}], mode="sft", max_new_tokens=1)
    vqa_path = ROOT / "docs/course-experiments/results/vqa.json"
    vqa = json.loads(vqa_path.read_bytes())
    paired = []
    reference = None
    for name, variant in vqa["results"]["variants"].items():
        if "text_before" not in variant:
            continue
        before, after = variant["text_before"], variant["text_after"]
        before_inputs = [{"messages": row["messages"], "expected": row["expected"]} for row in before["samples"]]
        after_inputs = [{"messages": row["messages"], "expected": row["expected"]} for row in after["samples"]]
        assert before_inputs == after_inputs and len(before_inputs) == 10
        if reference is None:
            reference = before_inputs
        assert before_inputs == reference, "固定對照使用的文字測試題不同"
        paired.append({"variant": name, "paired_input_sha256": digest(json.dumps(before_inputs, ensure_ascii=False, sort_keys=True).encode()), "records_before": before["records"], "records_after": after["records"], "effective_tokens_before": before["effective_tokens"], "effective_tokens_after": after["effective_tokens"], "matches_before": before["matches"], "matches_after": after["matches"], "same_questions_and_targets": True})
    assert {v["variant"] for v in paired} == {"projector_only", "partial", "all", "all_replay"}
    inputs = ["course/training.md", "scripts/train.py", "scripts/evaluate.py", "scripts/infer_modal.py", "scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py", "tiny_perceptron/multimodal.py", "docs/course-experiments/results/vqa.json", "course/chapters/11.md", "docs/technical-review-guide.md", "docs/course-review-20261019/freeze-07.json"]
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + str(section_run.relative_to(ROOT)),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
        "result": "全部斷言通過；未執行訓練或下載權重。",
        "previous_report": {"path": str(previous_path.relative_to(ROOT)), "sha256": digest(previous_path.read_bytes()), "issues": previous["issues"]},
        "source_checks": source_checks,
        "artifact_checks": artifact_checks,
        "training_diff": diff,
        "fixed_teacher_sections": fixed_sections,
        "section_execution": section_output,
        "probe_execution": probe,
        "probe_equal_except_training_input_sha256": True,
        "modal_sampler_checks": sampler_checks,
        "train_main_call_names": calls,
        "cli_evaluation_scope": "三條命令都不帶data；prepare_examples回傳None，訓練迴圈呼叫modal_loss。main沒有建立固定validation或評估呼叫，infer_modal每條只生成當前問題。",
        "text_evaluation_rejection": rejection,
        "language_submodel_evaluation": {"accepted": True, "samples": len(text_result["samples"]), "scope": "未訓練的小模型只確認文字子模型可交給文字評估入口；不作能力或品質結論。"},
        "vqa_text_before_after": {"original_environment": {k: vqa[k] for k in ("revision", "device", "seed", "torch_version", "python_version")}, "text_data": vqa["results"]["text_data"], "paired_variants": paired, "reference_questions_and_targets": reference, "not_covered": "direct_vqa分支沒有text_before/text_after；本次不聲稱它有這項對照。"},
        "input_sha256": {name: digest((ROOT / name).read_bytes()) for name in inputs},
        "scope": "只讀原始程式與實驗紀錄；既有算繪與自行產生的小節工具輸出按指定路徑讀取，未讀其他角色報告。",
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
