"""核對固定題、相依教材修訂，以及多模態練習的資料與評估範圍。"""

import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch
from scripts import evaluate as evaluate_cli
from scripts import train as train_cli
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True)
    return {"argv": args, "exit_code": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, header in enumerate(headers) if header[0].startswith(b"## " + identifier.encode() + b" "))
    return raw[headers[index].start():headers[index + 1].start() if index + 1 < len(headers) else len(raw)]


previous_path = "docs/course-review-20261019/reviews/rv1019_technical_5-1/technical-5-1-current-01.json"
previous_raw = (ROOT / previous_path).read_bytes()
assert sha(previous_raw) == "513068c7c46e51c866fd57a6672a870055d654242aac28ccd2923db1a55858e8"
previous = json.loads(previous_raw)
assert previous["issues"] == []
source_integrity = []
for item in previous["sources"]:
    if "path" not in item:
        continue
    actual = sha((ROOT / item["path"]).read_bytes())
    source_integrity.append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"], "current_sha256": actual, "unchanged": actual == item["sha256"]})
assert [item["path"] for item in source_integrity if not item["unchanged"]] == ["course/training.md"]
artifact_integrity = []
for item in previous["artifacts"]:
    actual = sha((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"]
    artifact_integrity.append({"id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})

baseline = run(["git", "show", "ecf6dec71280:course/training.md"])
assert baseline["exit_code"] == 0
old_training = baseline["stdout"].encode()
training = (ROOT / "course/training.md").read_bytes()
assert sha(old_training) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert sha(training) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
diff = run(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
assert diff["exit_code"] == 0
assert section(old_training, "T.4") == section(training, "T.4")
local_section = section(training, "T.6").decode()
bash = re.findall(r"```bash\n(.*?)```", local_section, flags=re.S)
commands = [line for block in bash for line in block.splitlines() if line.strip()]
modal_training_commands = [line for line in commands if "scripts/train.py --task" in line]
assert len(modal_training_commands) == 3
assert all("--data" not in line for line in modal_training_commands)
assert not any("scripts/evaluate.py" in line or "scripts/evaluate_modal.py" in line for line in commands)
assert "### 重跑第11章的固定實驗" in local_section
assert "--experiment vqa --device cuda" in local_section

frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/5.1.md").read_bytes()
assert frozen == section((ROOT / "course/chapters/05.md").read_bytes(), "5.1")
assert sha(frozen) == "3747cc321cd731cc37ae063196474a4e79e9e59dac24726cdee5628b3d073662"
run_dir = ROOT / "outputs/reviewer-tools/runs/20261020T151918946133Z"
section_files = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_dir / name).read_bytes()
    section_files[name] = {"sha256": sha(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
assert section_files["execution.json"]["content"]["exit_code"] == 0
assert section_files["extraction.json"]["content"]["source_sha256"] == sha(frozen)
stdout = section_files["stdout.txt"]["content"]
values = re.search(r"固定例子代價 ([0-9.eE+-]+) → ([0-9.eE+-]+)", stdout)
before, after = map(float, values.groups())
assert after < before and abs(before - 2.27) < 0.005 and abs(after - 0.0004) < 0.00005
assert stdout.count("第一步embedding梯度大小") == 1
fence = section_files["fence-1.py"]["content"]
modified = fence.replace("y = torch.tensor([[2, 3, 4]])", "y = torch.tensor([[-100, -100, -100]])")
assert modified != fence
torch.set_num_threads(1)
torch.set_default_device("cpu")
captured = io.StringIO()
try:
    with contextlib.redirect_stdout(captured):
        exec(compile(modified, "all-ignored-labels", "exec"), {"__name__": "__main__"})
except ValueError as error:
    empty_labels = {"error_type": type(error).__name__, "message": str(error), "stdout": captured.getvalue()}
else:
    raise AssertionError("全忽略目標應拒絕計算")
assert "所有 labels 都被忽略" in empty_labels["message"] and empty_labels["stdout"] == ""

class CaptureModel:
    def __init__(self):
        self.records = []
    def __call__(self, ids, labels, *, image, waveform):
        self.records.append({"ids": ids.tolist(), "labels": labels.tolist(), "answer": tok.decode(labels[labels != -100].tolist()),
                             "image_shape": None if image is None else list(image.shape),
                             "waveform_shape": None if waveform is None else list(waveform.shape)})
        return {"logits": torch.zeros(1, len(labels), tok.vocab_size, requires_grad=True),
                "labels": labels[None], "auxiliary": torch.tensor(0.0)}

tok = ByteTokenizer()
sampling = {}
for task, expected in (("vision", ["circle", "square"]), ("audio", ["low", "high"]), ("joint", ["circle,low", "square,high"])):
    args = train_cli.parser().parse_args(["--task", task])
    assert args.data is None and train_cli.prepare_examples(args, tok) is None
    model = CaptureModel()
    with patch("scripts.train.random.choice", side_effect=["red", "circle", False, "blue", "square", True]) as chooser:
        losses = [train_cli.modal_loss(model, task, tok, torch.device("cpu")) for _ in range(2)]
        assert chooser.call_count == 6
        draws = [repr(call.args[0]) for call in chooser.call_args_list]
    answers = [record["answer"] for record in model.records]
    assert answers == expected and all(torch.isfinite(loss) for loss in losses)
    sampling[task] = {"data_default": args.data, "prepared_examples": None, "random_choice_calls": 6,
                      "draw_domains": draws, "sampled_records": model.records,
                      "scope": "只替換隨機選擇以核對生成路徑；沒有更新模型，也不表示每步題目必然不同。"}

multimodal = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=64)))
keys = list(multimodal.state_dict())
assert all(any(key.startswith(prefix) for key in keys) for prefix in ("language.", "vision.", "audio.", "image_projector.", "audio_projector."))
with patch.object(evaluate_cli, "load_checkpoint", return_value=(multimodal, {"format_version": "multimodal-v1"})) as loader:
    with patch.object(sys, "argv", ["scripts/evaluate.py", "unused-checkpoint.pt", "--device", "cpu"]):
        try:
            evaluate_cli.main()
        except ValueError as error:
            guard = {"error_type": type(error).__name__, "message": str(error), "load_checkpoint_calls": loader.call_count}
        else:
            raise AssertionError("文字評估入口應拒絕多模態模型")
assert "只評估 TinyLM" in guard["message"]

vqa_path = "docs/course-experiments/results/vqa.json"
vqa_raw = (ROOT / vqa_path).read_bytes()
vqa = json.loads(vqa_raw)
modalities_path = "scripts/course_experiments/modalities.py"
assert sha((ROOT / modalities_path).read_bytes()) == vqa["code_sha256"][modalities_path]
vqa_checks = {}
for name in ("projector_only", "partial", "all", "all_replay"):
    variant = vqa["results"]["variants"][name]
    first, last = variant["text_before"], variant["text_after"]
    first_questions = [{"messages": row["messages"], "expected": row["expected"]} for row in first["samples"]]
    last_questions = [{"messages": row["messages"], "expected": row["expected"]} for row in last["samples"]]
    assert first_questions == last_questions and len(first_questions) == 10
    assert first["effective_tokens"] == last["effective_tokens"] == 69
    assert variant["text_probe_is_training_diagnostic"] is False
    vqa_checks[name] = {"same_questions_and_targets": True, "questions_sha256": sha(json.dumps(first_questions, ensure_ascii=False, sort_keys=True).encode()),
                        "records": 10, "effective_tokens": 69, "text_before_nll": first["nll"], "text_after_nll": last["nll"],
                        "text_probe_is_training_diagnostic": False}
for split, item in vqa["results"]["data"]["splits"].items():
    assert sha(json.dumps(item["records"], ensure_ascii=False, sort_keys=True).encode()) == item["sha256"]

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-1/recheck-current-02.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "head": run(["git", "rev-parse", "HEAD"]),
    "previous_report": {"path": previous_path, "sha256": sha(previous_raw)}, "previous_issues": previous["issues"],
    "repository_source_integrity": source_integrity, "previous_artifact_integrity": artifact_integrity,
    "training_change": {"diff_execution": diff, "baseline_command": baseline["argv"], "baseline_sha256": sha(old_training),
                        "current_sha256": sha(training), "t4_unchanged": True, "t4_sha256": sha(section(training, "T.4")),
                        "t4_text": section(training, "T.4").decode(), "modal_training_commands": modal_training_commands,
                        "no_text_evaluation_command": True, "local_section_bash_commands": commands},
    "section_execution": {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.1" --execute',
                          "copied_from": str(run_dir.relative_to(ROOT)), "files": section_files},
    "fresh_fixed_example": {"before": before, "after": after}, "all_ignored_labels": empty_labels,
    "modal_sampling": sampling, "evaluate_multimodal_rejection": guard,
    "multimodal_state_prefixes_present": ["language.", "vision.", "audio.", "image_projector.", "audio_projector."],
    "vqa_record_inspection": {"path": vqa_path, "sha256": sha(vqa_raw), "status": vqa["status"], "evidence_status": vqa["evidence_status"],
                              "implementation_path": modalities_path, "implementation_sha256": sha((ROOT / modalities_path).read_bytes()),
                              "text_data": vqa["results"]["text_data"], "variant_checks": vqa_checks,
                              "scope": "核對保存紀錄的同題比較與原始碼，不是重新訓練或評估歷史權重；direct_vqa 分支未提供這個文字前後對照。"},
    "result": "所有核對通過；頁面與被引用的文字配方未變，相依文件的新敘述有程式和紀錄支持。",
}
output = Path(__file__).with_name("recheck-evidence-current-02.json")
with output.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": sha(output.read_bytes()), "result": result["result"],
                  "fixed_example": result["fresh_fixed_example"]}, ensure_ascii=False))
