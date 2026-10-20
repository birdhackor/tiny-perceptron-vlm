"""核對教材相依文件變更，沿用內容未變的證據，並保存本次小節執行結果。"""
import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch

from scripts.course_experiments.behavior import _safety_records
from scripts.course_experiments.common import records_sha256, split_records
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.training import choose_device


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
    i = next(i for i, h in enumerate(headings) if h[0].startswith(("## " + name + " ").encode()))
    end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
    return raw[headings[i].start():end]


report_path = "docs/course-review-20261019/reviews/rv1019_technical_9-4/technical-9-4-original-01.json"
original_bytes = read(report_path)
assert digest(original_bytes) == "5b83bbe10d442ebed5bdea9fbb7f8757be489e732d74285db6b22bc28d1ddb70"
original = json.loads(original_bytes)
assert original["issues"] == []
source_hashes = []
for item in original["sources"]:
    if item["kind"] == "repository_code":
        actual = digest(read(item["path"]))
        source_hashes.append({"id": item["id"], "path": item["path"],
                              "previous_sha256": item["sha256"], "current_sha256": actual,
                              "unchanged": actual == item["sha256"]})
assert [x["path"] for x in source_hashes if not x["unchanged"]] == ["course/training.md"]
old_artifacts = []
for item in original["artifacts"]:
    actual = digest(read(item["path"]))
    assert actual == item["sha256"]
    old_artifacts.append({"id": item["id"], "path": item["path"], "sha256": actual})
old_verification = json.loads(read(next(x["path"] for x in original["artifacts"] if x["id"] == "a-verification")))
old_execution = json.loads(read(next(x["path"] for x in original["artifacts"] if x["id"] == "a-section")))
assert digest(read("docs/course-experiments/results/safety.json")) == old_verification["current_input_sha256"]["docs/course-experiments/results/safety.json"]

frozen = read("docs/course-review-20261019/freeze-01/sources/9.4.md")
assert section(read("course/chapters/09.md"), "9.4") == frozen
assert digest(frozen) == original["source_sha256"]
freeze_path = "docs/course-review-20261019/freeze-06.json"
assert digest(read(freeze_path)) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
page = next(p for p in json.loads(read(freeze_path))["pages"] if p["page_id"] == "9.4")
assert page["source_sha256"] == digest(frozen)
assert page["figures_sha256"] == {}
run = ROOT / "outputs/reviewer-tools/runs/20261020T103231648251Z"
helper_files = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    data = (run / name).read_bytes()
    helper_files[name] = {"sha256": digest(data), "content": data.decode()}
helper_execution = json.loads(helper_files["execution.json"]["content"])
helper_environment = json.loads(helper_files["environment.json"]["content"])
helper_extraction = json.loads(helper_files["extraction.json"]["content"])
assert helper_execution["exit_code"] == 0
assert helper_execution["source_sha256"] == digest(frozen)
assert helper_environment["attempted_fences"] == [1]
assert helper_environment["guard_events"] == []
assert helper_extraction["svg_references"] == []
assert helper_files["stdout.txt"]["content"] == old_execution["files"]["stdout.txt"]["content"]
assert helper_files["stderr.txt"]["content"] == ""

old_training = command(["git", "show", "5ef56bc756f3:course/training.md"])["stdout"].encode()
new_training = command(["git", "show", "6a7aa1de5227:course/training.md"])["stdout"].encode()
assert digest(old_training) == next(x["sha256"] for x in original["sources"] if x["id"] == "s-training")
assert new_training == read("course/training.md")
assert digest(new_training) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
training_diff = command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
assert section(old_training, "T.5") == section(new_training, "T.5")
safety_command = ".venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda"
assert safety_command in section(new_training, "T.5").decode()
help_result = command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"])
assert "--experiment" in help_result["stdout"] and "--device {cpu,cuda}" in help_result["stdout"]
spec = experiment_spec("safety")
assert (spec["module"], spec["function"]) == ("behavior", "run_safety")
assert spec["dependencies"] == ["style"]
assert spec["assets"] == ["pku-safe-rlhf"]

cli_paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/course_experiments/run.py",
             "scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py"]
cli_checks = []
for relative in cli_paths:
    raw = read(relative)
    tree = ast.parse(raw)
    declarations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                    and any(isinstance(arg, ast.Constant) and arg.value == "--device" for arg in node.args)]
    if relative in ("scripts/train.py", "scripts/evaluate.py", "scripts/infer.py"):
        expected = "auto"
    elif relative == "scripts/course_experiments/run.py":
        expected = "cpu"
    else:
        expected = None
    observed = ast.literal_eval(next(k.value for k in declarations[0].keywords if k.arg == "default")) if declarations else None
    assert observed == expected
    cli_checks.append({"path": relative, "sha256": digest(raw), "device_default": observed,
                       "declarations": [ast.get_source_segment(raw.decode(), node) for node in declarations]})
device_cases = []
for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
    with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
        observed = str(choose_device())
        assert observed == expected
        device_cases.append({"cuda_available": cuda, "mps_available": mps, "observed": observed})
assert str(choose_device("cpu")) == "cpu"
assert 'load_checkpoint(args.checkpoint)' in read("scripts/quantize.py").decode()
assert 'def load_checkpoint(path, device="cpu", restore_rng=False):' in read("tiny_perceptron/training.py").decode()

record = json.loads(read("docs/course-experiments/results/safety.json"))
parts = split_records(_safety_records(), seed=record["seed"])
arithmetic = split_records(arithmetic_records(), seed=record["seed"])
split_checks = {}
for name, rows in parts.items():
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    assert digest(serialized) == record["results"]["data"][name]["sha256"]
    split_checks[name] = {"records": len(rows), "sha256": digest(serialized)}
permission_counts = {}
tok = ByteTokenizer()
for name in ("safety-only", "model"):
    training_rows = parts["train"] + (arithmetic["train"] if name == "model" else [])
    assert records_sha256(training_rows) == record["results"]["runs"][name]["training"]["records_sha256"]
    rows = [s for s in record["results"]["runs"][name]["safety"]["test"]["samples"] if s["kind"] == "permission"]
    assert rows == old_verification["runs"][name]["samples"]
    assert len(rows) == 6
    assert all(s["exact"] and s["generated"] == s["expected"]
               and s["generated_ids"] == tok.encode(s["expected"]) + [tok.eos_id] for s in rows)
    permission_counts[name] = {"records": len(rows), "matches": sum(s["exact"] for s in rows)}

class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)

url = "https://docs.pytorch.org/docs/2.14/notes/randomness.html"
curl_argv = ["curl", "-fsSL", url]
curl_result = subprocess.run(curl_argv, capture_output=True, check=True)
parser = VisibleText()
parser.feed(curl_result.stdout.decode())
normalized = " ".join(" ".join(parser.parts).split())
excerpt = "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
assert excerpt in normalized
reproducibility_source = {"url": url, "version": "PyTorch 2.14 documentation",
                          "accessed_on": datetime.now(UTC).date().isoformat(), "command_argv": curl_argv,
                          "exit_code": curl_result.returncode, "original_sha256": digest(curl_result.stdout),
                          "locator": "Reproducibility; opening paragraph", "excerpt": excerpt,
                          "inspection_note": "實讀開頭兩段與CPU/GPU即使同seed仍可能不同的限制；只保存支持主張的短摘錄。"}

modified_statements = [
    {"location": "course/training.md：Python 怎麼寫", "status": "verified",
     "statement": "訓練、評估與推論工具預設auto；固定實驗入口預設CPU；quantize固定CPU；資料準備與下載工具沒有--device。",
     "evidence": "AST核對七個CLI的--device宣告與default；mock可用性實跑choose_device；quantize呼叫load_checkpoint且後者預設cpu。",
     "scope": "未實跑CUDA/MPS訓練；Windows替換路徑與MPS未整套驗證的舊句未變。"},
    {"location": "course/training.md：圖片與聲音練習的預期答案提醒", "status": "verified",
     "statement": "同一固定配方在不同電腦上可能有不同結果；應先查命令與checkpoint，再看整份驗證題。",
     "evidence": "PyTorch 2.14 Reproducibility不保證跨平台或CPU/GPU同seed完全重現；判斷整份驗證資料沿用已核對的保留評估原理。",
     "scope": "這是可能性與診斷建議，不是量化跨機器差異；沒有聲稱短訓或小模型一定造成特定差異。"},
    {"location": "course/training.md：量化前後的blue示例", "status": "verified",
     "statement": "原模型若未答blue，量化後不必答blue；此示例比較前後答案是否一致。",
     "evidence": "透明推導：設前後答案為a與q，保持單題輸出的條件為a=q，而不是另加q=blue；a=red且q=red即為前後相同卻都不是blue的例子。",
     "scope": "一致性不等於答案正確；原段落仍要求同一整份驗證題檢查能力，沒有保證量化一定保持輸出。"},
]

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-4/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": report_path, "sha256": digest(original_bytes)},
    "previous_issues": original["issues"], "source_hash_comparison": source_hashes,
    "unchanged_artifacts": old_artifacts, "source_sha256": digest(frozen),
    "section_execution": {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/09.md#9.4" --execute', "files": helper_files},
    "training_diff": training_diff, "training_T5_unchanged": True,
    "training_T5_sha256": digest(section(new_training, "T.5")),
    "safety_command": safety_command, "cli_help": help_result, "experiment_spec": spec,
    "cli_device_checks": cli_checks, "choose_device_mock_cases": device_cases,
    "additional_input_sha256": {p: digest(read(p)) for p in cli_paths + ["tiny_perceptron/training.py"]},
    "split_checks": split_checks, "permission_counts": permission_counts,
    "reproducibility_source": reproducibility_source, "modified_dependency_statements": modified_statements,
    "claim_continuity": [{"claim_id": c["id"], "status": "verified",
                          "basis": "來源與證據未變且雜湊重核；T.5指令另外核對相同；原始fence及資料切分與逐題token亦重新檢查。"}
                         for c in original["claims"]],
    "result": "所有複查斷言通過；原始報告與artifact未改寫。",
}
output = DESTINATION / "recheck-current-01.json"
with output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": digest(output.read_bytes()), "result": result["result"]}, ensure_ascii=False))
