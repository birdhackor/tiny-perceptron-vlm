"""核對引用教材的變動、多模態命令範圍與未變的小節主張。"""

import ast
import hashlib
import importlib.util
import json
import math
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM
from scripts import train, evaluate

BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-14-1"
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_14-1/technical-14-1-current-01.json"
OUTPUT = BASE / "audit-current-02.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
    return {"argv": argv, "exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr}


def section(raw, name):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headers) if h.group().startswith(("## " + name + " ").encode()))
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def main():
    torch.set_num_threads(1)
    previous_raw = PREVIOUS.read_bytes()
    assert sha(previous_raw) == "fceb74e7ada73b1cc92b7655db510799d6d6de51bd20eceb7e9f3efdf2549322"
    previous = json.loads(previous_raw)
    assert previous["issues"] == []
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-1/audit-current-02.py",
              "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
              "previous_report": {"path": PREVIOUS.relative_to(ROOT).as_posix(), "sha256": sha(previous_raw)},
              "resolved_issue_ids": []}
    result["previous_artifacts"] = []
    for item in previous["artifacts"]:
        observed = sha((ROOT / item["path"]).read_bytes())
        assert observed == item["sha256"]
        result["previous_artifacts"].append({"path": item["path"], "sha256": observed, "unchanged": True})
    result["repository_sources"] = []
    for item in previous["sources"]:
        if "path" not in item:
            continue
        current = sha((ROOT / item["path"]).read_bytes())
        result["repository_sources"].append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"],
                                            "current_sha256": current, "unchanged": current == item["sha256"]})
        assert item["path"] == "course/training.md" or current == item["sha256"]
    old = subprocess.run(["git", "show", "ecf6dec71280:course/training.md"], cwd=ROOT, capture_output=True, check=True).stdout
    new = subprocess.run(["git", "show", "c6b9339d2595:course/training.md"], cwd=ROOT, capture_output=True, check=True).stdout
    current = (ROOT / "course/training.md").read_bytes()
    assert sha(old) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    assert sha(current) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b" and new == current
    old_reference, new_reference = section(old, "T.8"), section(current, "T.8")
    assert old_reference == new_reference
    result["training_change"] = {"diff": command(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"]),
                                 "previous_sha256": sha(old), "current_sha256": sha(current),
                                 "referenced_title": "架構比較一次只換一個條件", "referenced_section_unchanged": True,
                                 "referenced_section_sha256": sha(new_reference),
                                 "modern_command": ".venv/bin/python -m scripts.course_experiments.run --experiment modern --device cuda"}
    assert result["training_change"]["modern_command"].encode() in new_reference
    parsed_commands = []
    for line in current.decode().splitlines():
        if line.startswith(".venv/bin/python scripts/train.py --task "):
            argv = shlex.split(line)
            parsed = train.parser().parse_args(argv[2:])
            if parsed.task in ("vision", "audio", "joint"):
                assert parsed.data is None and train.prepare_examples(parsed, ByteTokenizer()) is None
                parsed_commands.append({"command": line, "task": parsed.task, "steps": parsed.steps,
                                        "batch_size": parsed.batch_size, "data": None, "examples": None})
    assert sorted(c["task"] for c in parsed_commands) == ["audio", "joint", "vision"]
    train_text = (ROOT / "scripts/train.py").read_text()
    tree = ast.parse(train_text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = sorted({ast.unparse(n.func) for n in ast.walk(functions["main"]) if isinstance(n, ast.Call)})
    assert not any("evaluate" in name or "validation" in name for name in calls)
    sampled = []
    chosen = ["red", "circle", False, "blue", "square", True] * 3
    draw_index = 0

    def draw(values):
        nonlocal draw_index
        value = chosen[draw_index]
        draw_index += 1
        assert value in values
        return value

    class ProbeModel:
        def __call__(self, ids, labels, image=None, waveform=None):
            sampled.append({"target": ByteTokenizer().decode(labels[labels >= 0].tolist()),
                            "has_image": image is not None, "has_audio": waveform is not None})
            return {"logits": torch.zeros(len(labels), 264), "labels": labels, "auxiliary": torch.tensor(0.0)}

    with patch.object(train.random, "choice", side_effect=draw), \
         patch.object(train, "scene", side_effect=lambda *args: torch.zeros(3, 16, 16)), \
         patch.object(train, "tone", side_effect=lambda *args: torch.zeros(64)):
        losses = [float(train.modal_loss(ProbeModel(), task, ByteTokenizer(), "cpu"))
                  for task in ("vision", "audio", "joint") for _ in range(2)]
    assert draw_index == 18 and all(math.isfinite(value) for value in losses)
    assert [s["target"] for s in sampled] == ["circle", "square", "low", "high", "circle,low", "square,high"]
    result["random_modal_routes"] = {"published_commands": parsed_commands, "main_call_names": calls,
                                     "modal_loss_current_source": ast.get_source_segment(train_text, functions["modal_loss"]),
                                     "controlled_draw_calls": draw_index, "observed_examples": sampled, "losses": losses,
                                     "inspection_note": "完整讀main：每次更新在batch迴圈呼叫modal_loss，無data時其內重新抽color/shape/high；沒有固定驗證題建立或驗證評估流程。探針只做無梯度的替身模型前向，不更新參數。"}
    multimodal = MultiModalLM(TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=64)))
    rejected = None
    with patch.object(sys, "argv", ["evaluate", "unread.pt", "--device", "cpu"]), \
         patch.object(evaluate, "load_checkpoint", return_value=(multimodal, {"format_version": "multimodal-v1"})), \
         patch.object(evaluate, "evaluate") as evaluator:
        try:
            evaluate.main()
        except ValueError as error:
            rejected = str(error)
        assert not evaluator.called
    assert rejected is not None and "TinyLM" in rejected
    result["evaluate_multimodal_guard"] = {"input_type": type(multimodal).__name__, "exception_type": "ValueError",
                                            "message": rejected, "evaluation_called": False}
    modal_path = ROOT / "scripts/course_experiments/modalities.py"
    modal_text = modal_path.read_text()
    node = next(n for n in ast.parse(modal_text).body if isinstance(n, ast.FunctionDef) and n.name == "run_vqa")
    segment = ast.get_source_segment(modal_text, node)
    assert 'text_holdout = text_splits["test"]' in segment
    assert segment.count('evaluate_lm(model.language, text_holdout, mode="sft")') == 2
    vqa_path = ROOT / "docs/course-experiments/results/vqa.json"
    vqa = json.loads(vqa_path.read_bytes())
    assert sha(modal_path.read_bytes()) == vqa["code_sha256"]["scripts/course_experiments/modalities.py"]
    comparisons = {}
    for name in ("projector_only", "partial", "all", "all_replay"):
        variant = vqa["results"]["variants"][name]
        before, after = variant["text_before"], variant["text_after"]
        def probes(value):
            return [{"messages": s["messages"], "expected": s["expected"]} for s in value["samples"]]
        assert probes(before) == probes(after)
        assert len(before["samples"]) == len(after["samples"]) == before["records"] == after["records"] == 10
        for value in (before, after):
            assert value["nll"] == value["nll_sum"] / value["effective_tokens"]
        comparisons[name] = {"same_prompts_and_targets": True, "records": 10,
                             "effective_tokens_before": before["effective_tokens"], "effective_tokens_after": after["effective_tokens"],
                             "nll_before": before["nll"], "nll_after": after["nll"],
                             "probe_sha256": sha(json.dumps(probes(before), sort_keys=True, ensure_ascii=False).encode())}
    result["vqa_retention_comparison"] = {"record_path": vqa_path.relative_to(ROOT).as_posix(), "sha256": sha(vqa_path.read_bytes()),
                                          "recorded_revision": vqa["revision"], "text_data": vqa["results"]["text_data"],
                                          "run_vqa_source": segment, "comparisons": comparisons,
                                          "scope": "核對保存紀錄的比較設計及同題目/分母，未重新訓練或載入保存權重推論。四種two-stage變體均有對照；direct_vqa沒有此欄，沒有把此主張外推到每個變體。"}
    from scripts.course_experiments.run import experiment_spec
    assert experiment_spec("modern")["function"] == "run_modern"
    result["modern_route"] = experiment_spec("modern")
    result["source_hashes"] = {name: sha((ROOT / name).read_bytes()) for name in
                                ("scripts/train.py", "scripts/evaluate.py", "tiny_perceptron/multimodal.py",
                                 "scripts/course_experiments/modalities.py", "docs/course-experiments/results/vqa.json")}
    run = command([str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/14.md#14.1", "--execute"])
    metadata = json.loads(run["stdout"])
    assert metadata["source_sha256"] == previous["source_sha256"] and metadata["execution_exit_code"] == 0
    directory = Path(metadata["artifact_directory"]).resolve()
    assert directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    files = {}
    for name in ("stdout.txt", "stderr.txt", "execution.json", "environment.json", "extraction.json"):
        raw = (directory / name).read_bytes()
        files[name] = {"sha256": sha(raw), "text": raw.decode()}
    expected = "-0.7071\n-0.7071\n0.0\n2對5 3.9298 12對15 3.9298\n3對5 5.1274\n與函式庫版本相同： True\n"
    assert files["stdout.txt"]["text"] == expected and files["stderr.txt"]["text"] == ""
    result["section_execution"] = {"driver": run, "files": files}
    raw = section((ROOT / "course/chapters/14.md").read_bytes(), "14.1")
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/14.1.md").read_bytes()
    assert raw == frozen and sha(raw) == previous["source_sha256"]
    result["page"] = {"source_sha256": sha(raw), "page_sentences_changed": [],
                       "original_claim_ids": [c["id"] for c in previous["claims"]], "claim_count": len(previous["claims"])}
    receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
    result["figures"] = []
    for item in receipt["records"]:
        if item.get("page_id") == "14.1":
            assert sha((ROOT / item["figure"]).read_bytes()) == previous["figure_sha256"][item["figure"]]
            assert sha((ROOT / item["png"]["path"]).read_bytes()) == item["png"]["sha256"]
            result["figures"].append(item)
    result["visual_inspection"] = "本回合以view_image親看兩張指定PNG；45/135度分量正負、同尺度、共同轉450度後的135度夾角與點積，均與原先核對一致。沒有複製PNG。"
    result["result"] = "所有斷言通過；引用區未變，小節執行相符，多模態練習與固定VQA對照的新增說明有直接依據。"
    with OUTPUT.open("x", encoding="utf-8") as target:
        target.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifact": OUTPUT.relative_to(ROOT).as_posix(), "result": result["result"],
                      "claim_count": result["page"]["claim_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
