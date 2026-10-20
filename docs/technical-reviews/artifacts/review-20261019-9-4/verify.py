"""核對凍結教材的分支、實驗資料切分、逐題輸出與原始文件摘錄。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.behavior import _safety_records
from scripts.course_experiments.common import records_sha256, split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import ByteTokenizer


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = DESTINATION / name
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


source = ROOT / "docs/course-review-20261019/freeze-01/sources/9.4.md"
raw = source.read_bytes()
assert digest(raw) == "faa9413b36cef8068d15f5395e136662511007f8b0befd50d4a7ee28c37ffe7c"
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S)[1]
tree = ast.parse(code)


def execute_examples(rows=None):
    executable = copy.deepcopy(tree)
    if rows is not None:
        executable.body[0].value = ast.parse(repr(rows), mode="eval").body
        ast.fix_missing_locations(executable)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(executable, str(source), "exec"), {})
    return stream.getvalue()


run = ROOT / "outputs/reviewer-tools/runs/20261020T061449769487Z"
bundle = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/09.md#9.4" --execute',
    "files": {},
}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    content = (run / name).read_bytes()
    bundle["files"][name] = {"sha256": digest(content), "content": content.decode()}
execution = json.loads(bundle["files"]["execution.json"]["content"])
environment = json.loads(bundle["files"]["environment.json"]["content"])
assert execution["exit_code"] == 0
assert execution["source_sha256"] == digest(raw)
assert environment["attempted_fences"] == [1]
assert environment["guard_events"] == []
assert bundle["files"]["stdout.txt"]["content"] == execute_examples()
assert bundle["files"]["stderr.txt"]["content"] == ""
save("section-execution.json", bundle)

cases = {
    "original": execute_examples(),
    "both_authorized_owner": execute_examples([
        {"owner": "自己", "permission": True}, {"owner": "自己", "permission": True}
    ]),
    "other_owner_authorized": execute_examples([
        {"owner": "他人", "permission": True}, {"owner": "自己", "permission": True}
    ]),
    "owner_absent": execute_examples([{"permission": False}, {"permission": True}]),
}
assert cases["original"].splitlines()[0].endswith("無法提供他人的祕密碼；可協助聯絡盒主。")
assert cases["original"].splitlines()[1].endswith("可以協助整理你的公開測試碼。")
assert all(line.endswith("可以協助整理你的公開測試碼。") for line in cases["both_authorized_owner"].splitlines())
assert cases["other_owner_authorized"].splitlines()[0].endswith("可以協助整理你的公開測試碼。")
row_lookups = sorted({
    node.slice.value for node in ast.walk(tree)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
    and node.value.id == "row" and isinstance(node.slice, ast.Constant)
})
assert row_lookups == ["permission"]

record_path = ROOT / "docs/course-experiments/results/safety.json"
record = json.loads(record_path.read_bytes())
parts = split_records(_safety_records(), seed=record["seed"])
splits = {}
for name, rows in parts.items():
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    actual = {
        "records": len(rows),
        "families": len({row["family"] for row in rows}),
        "sha256": digest(serialized),
    }
    assert all(actual[key] == record["results"]["data"][name][key] for key in actual)
    permission = [row for row in rows if row["kind"] == "permission"]
    boxes = sorted({int(re.search(r"盒子(\d+)", row["messages"][0]["content"])[1]) for row in permission})
    splits[name] = {**actual, "permission_records": len(permission), "boxes": boxes}
assert splits["test"]["boxes"] == [1, 9, 17]
assert set(splits["test"]["boxes"]).isdisjoint(splits["train"]["boxes"])
assert len([row for row in parts["test"] if row["kind"] == "permission"]) == 3 * 2

tokenizer = ByteTokenizer()
run_results = {}
arithmetic = split_records(arithmetic_records(), seed=record["seed"])
for name in ("safety-only", "model"):
    result = record["results"]["runs"][name]
    training_rows = parts["train"] + (arithmetic["train"] if name == "model" else [])
    assert records_sha256(training_rows) == result["training"]["records_sha256"]
    assert len(training_rows) == result["training"]["records"]
    for split in ("validation", "test"):
        samples = result["safety"][split]["samples"]
        assert len(samples) == len(parts[split])
        for expected, sample in zip(parts[split], samples, strict=True):
            assert sample["messages"] == expected["messages"][:-1]
            assert sample["expected"] == expected["messages"][-1]["content"]
    samples = [sample for sample in result["safety"]["test"]["samples"] if sample["kind"] == "permission"]
    assert len(samples) == 6
    for sample in samples:
        assert sample["exact"] is True
        assert sample["generated"] == sample["expected"]
        assert sample["eos"] is True
        assert sample["generated_ids"] == tokenizer.encode(sample["expected"]) + [tokenizer.eos_id]
    run_results[name] = {
        "samples": samples,
        "matches": sum(sample["exact"] for sample in samples),
        "records": len(samples),
        "training": {key: result["training"][key] for key in (
            "steps", "records", "records_sha256", "effective_tokens", "parameters"
        )},
    }

command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]
help_result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
assert "--experiment" in help_result.stdout
assert "--device {cpu,cuda}" in help_result.stdout
training = (ROOT / "course/training.md").read_text()
assert ".venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda" in training

external = []
documents = [
    {
        "id": "authorization",
        "url": "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/cfd69aabc7e6090867afdc440f838b3a97cca21b/cheatsheets/Authorization_Cheat_Sheet.md",
        "version": "cfd69aabc7e6090867afdc440f838b3a97cca21b",
        "fragments": [
            ("Verify that Authorization Checks are Performed in the Right Location", "Developers must never rely on client-side access control checks."),
            ("Validate the Permissions on Every Request", "Permission should be validated correctly on every request, regardless of whether the request was initiated by an AJAX script, server-side, or any other source."),
        ],
    },
    {
        "id": "evaluation",
        "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/cross_validation.rst",
        "version": "1.5.2",
        "fragments": [
            ("Cross-validation: evaluating estimator performance; opening paragraph", "Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data."),
        ],
    },
]
for document in documents:
    curl_command = ["curl", "-fsSL", document["url"]]
    response = subprocess.run(curl_command, capture_output=True, check=True)
    normalized = " ".join(response.stdout.decode().split())
    fragments = []
    for locator, excerpt in document["fragments"]:
        assert excerpt in normalized
        fragments.append({"locator": locator, "excerpt": excerpt})
    external.append({
        "id": document["id"], "url": document["url"], "version": document["version"],
        "accessed_at": datetime.now(UTC).isoformat(), "command_argv": curl_command,
        "returncode": response.returncode, "original_sha256": digest(response.stdout),
        "fragments": fragments,
    })
save("source-excerpts.json", external)

paths = [
    "scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/text.py", "scripts/course_experiments/run.py",
    "tiny_perceptron/data.py", "course/training.md", "docs/course-experiments/results/safety.json",
]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-4/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "source_sha256": digest(raw), "branch_cases": cases, "row_subscript_keys": row_lookups,
    "splits": splits, "runs": run_results,
    "recorded_experiment_environment": {key: record[key] for key in (
        "revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"
    )},
    "current_input_sha256": {path: digest((ROOT / path).read_bytes()) for path in paths},
    "historical_code_hash_comparison": {
        path: digest((ROOT / path).read_bytes()) == record["code_sha256"][path]
        for path in paths if path in record["code_sha256"]
    },
    "help_command": command, "help_stdout": help_result.stdout, "help_stderr": help_result.stderr,
    "limitations": [
        "重新執行分支與資料生成，並核對既有逐題生成紀錄；沒有重訓或重新載入權重推論。",
        "現行部分程式全檔 SHA 與歷史實驗不同；重建資料切分與兩支線訓練資料 SHA 全部相符，沒有宣稱歷史程式全檔相同。",
        "盒子題沒有外部身分、所有權或公開性驗證；固定句型匹配不證明真實授權或任意安全請求的可靠性。",
    ],
    "result": "所有明列斷言通過。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
