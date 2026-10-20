import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = ARTIFACT / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def score(text):
    error = None
    value = None
    try:
        value = json.loads(text)
        valid = isinstance(value, dict) and set(value) == {"answer"}
        correct = valid and value["answer"] == 4
    except json.JSONDecodeError as exception:
        valid = correct = False
        error = type(exception).__name__
    return {
        "input": text,
        "parsed_type": type(value).__name__,
        "parsed_repr": repr(value),
        "format": valid,
        "content": correct,
        "error": error,
    }


cases = [
    ('{"answer":4}', True, True),
    ('答案是：{"answer":4}', False, False),
    ('{"answer":5}', True, False),
    ('{"answer":4,"note":"完成"}', False, False),
    ('{}', False, False),
    ('{"note":4}', False, False),
    ('[4]', False, False),
    ('4', False, False),
    ('{"answer":"4"}', True, False),
    ('{"answer":4.0}', True, True),
    ('{"answer":NaN}', True, False),
    ('{"answer":Infinity}', True, False),
    ('{"answer":5,"answer":4}', True, True),
]
observed = []
for text, expected_format, expected_content in cases:
    row = score(text)
    assert (row["format"], row["content"]) == (expected_format, expected_content), "判定與預期不同"
    observed.append(row)

source = ROOT / "docs/course-review-20261019/freeze-01/sources/8.5.md"
body = source.read_text(encoding="utf-8")
fence = re.search(r"```python\n(.*?)```", body, re.S).group(1)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(fence, str(source), "exec"), {})
expected_stdout = (
    '{"answer":4} 格式 True 內容 True\n'
    '答案是：{"answer":4} 格式 False 內容 False\n'
    '{"answer":5} 格式 True 內容 False\n'
)
assert captured.getvalue() == expected_stdout, "原文執行輸出與預期不同"

run = ROOT / "outputs/reviewer-tools/runs/20261020T061415953455Z"
section_execution = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/08.md#8.5" --execute',
    "tool_result": {
        "artifact_directory": str(run),
        "source_sha256": sha(source.read_bytes()),
        "python_fences": 1,
        "svg_references": 0,
        "execution_exit_code": 0,
    },
    "copied_files": {},
}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run / name).read_bytes()
    section_execution["copied_files"][name] = {"sha256": sha(raw), "content": raw.decode("utf-8")}
assert json.loads(section_execution["copied_files"]["execution.json"]["content"])["exit_code"] == 0
assert section_execution["copied_files"]["stdout.txt"]["content"] == expected_stdout
save("section-execution.json", section_execution)

record_path = ROOT / "docs/course-experiments/results/style.json"
record = json.loads(record_path.read_text(encoding="utf-8"))
results = record["results"]
samples = [sample for sample in results["after"]["test"]["samples"] if sample.get("style") == "json"]
checked = []
for sample in samples:
    question = sample["messages"][0]["content"]
    match = re.fullmatch(r"style=json; (\d+)\+(\d+)=\?", question)
    assert match is not None, "題目格式不符"
    a, b = map(int, match.groups())
    value = json.loads(sample["generated"])
    expected = json.loads(sample["expected"])
    shape = isinstance(value, dict) and set(value) == {"answer"}
    integer = shape and type(value["answer"]) is int
    correct = integer and value["answer"] == a + b
    assert expected == {"answer": a + b}, "保存的標準答案與手算不同"
    encoded = [byte + 8 for byte in sample["generated"].encode("utf-8")] + [2]
    raw_ids_match = encoded == sample["generated_ids"]
    assert raw_ids_match, "可見回答與原始生成 ID 不同"
    assert sample["style_correct"] == integer, "保存的格式標記與重算不同"
    assert sample["content_correct"] == correct, "保存的內容標記與重算不同"
    checked.append({
        "saved_sample": sample,
        "sum_from_question": a + b,
        "parsed": value,
        "only_answer": shape,
        "exact_int": integer,
        "content_correct": correct,
        "raw_ids_match_visible_answer_and_eos": raw_ids_match,
    })
counts = {
    "records": len(checked),
    "format_and_integer": sum(row["exact_int"] for row in checked),
    "content_correct": sum(row["content_correct"] for row in checked),
}
assert counts == {"records": 7, "format_and_integer": 7, "content_correct": 0}, "逐題統計與課文不同"

official_requests = [
    ("https://docs.python.org/3.13/library/json.html", "Python 3.13.16", [
        ("json.loads", "to a Python object"),
        ("Standard Compliance and Interoperability", "Infinite and NaN number values are accepted and output;"),
    ]),
    ("https://docs.python.org/3.13/builtins/functions.html", "Python 3.13.16", [
        ("isinstance", "Return True if the object argument is an instance of the classinfo argument"),
    ]),
    ("https://docs.python.org/3.13/builtins/stdtypes.html", "Python 3.13.16", [
        ("Mapping Types — dict / iter(d)", "Return an iterator over the keys of the dictionary."),
        ("Set Types — set, frozenset", "whose elements are taken from iterable."),
    ]),
    ("https://docs.python.org/3.13/tutorial/errors.html", "Python 3.13.16", [
        ("Handling Exceptions", "continues after the try/except block."),
    ]),
    ("https://www.rfc-editor.org/rfc/rfc8259.txt", "RFC 8259, December 2017", [
        ("JSON Grammar", "JSON-text = ws value ws"),
        ("Numbers", "Numeric values that cannot be represented in the grammar below (such as Infinity and NaN) are not permitted."),
    ]),
]
receipts = []
for url, version, excerpts in official_requests:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    response = subprocess.run(command, capture_output=True, check=True)
    raw = response.stdout
    decoded = raw.decode("utf-8")
    parser = PlainText()
    parser.feed(decoded)
    plain = " ".join("".join(parser.parts).split()) if url.endswith("html") else " ".join(decoded.split())
    retained = []
    for locator, excerpt in excerpts:
        assert excerpt in plain, "指定原始引文不在回傳內容"
        retained.append({"locator": locator, "excerpt": excerpt})
    receipts.append({
        "url": url,
        "version": version,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "command_argv": command,
        "exit_code": response.returncode,
        "stderr": response.stderr.decode("utf-8"),
        "response_bytes": len(raw),
        "response_sha256": sha(raw),
        "excerpts": retained,
    })
save("official-source-receipts.json", receipts)

historical_command = ["git", "show", record["revision"] + ":scripts/course_experiments/behavior.py"]
historical = subprocess.run(historical_command, cwd=ROOT, capture_output=True)
repo_sources = {}
for path in (
    "scripts/course_experiments/behavior.py",
    "scripts/course_experiments/common.py",
    "tiny_perceptron/data.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/tokenization.py",
):
    repo_sources[path] = {"sha256": sha((ROOT / path).read_bytes()), "recorded_training_code_sha256": record["code_sha256"][path]}

report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-5/verify_json.py",
    "result": "全部小節輸出、額外解析案例與保存實驗逐題重算均符合預期；已讀取官方原文。",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "parser": "Python stdlib json"},
    "source_sha256": sha(source.read_bytes()),
    "original_fence_stdout": captured.getvalue(),
    "numeric_derivation": {"expression": "2 + 2", "expected": 4, "observed": 2 + 2},
    "cases": observed,
    "experiment": {
        "path": str(record_path.relative_to(ROOT)),
        "sha256": sha(record_path.read_bytes()),
        "saved_environment": {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status")},
        "training": results["training"],
        "arithmetic_splits": results["arithmetic_data"],
        "saved_json_rubric": results["after"]["test"]["rubric"]["json"],
        "recomputed_counts": counts,
        "checked_samples": checked,
        "scope": "重算既有單次實驗保存的七題原始生成，未重新訓練或從權重重新推論。",
    },
    "current_repository_sources": repo_sources,
    "historical_code_availability": {
        "command_argv": historical_command,
        "exit_code": historical.returncode,
        "stderr": historical.stderr.decode("utf-8"),
        "stdout_bytes": len(historical.stdout),
    },
}
save("verification.json", report)
print(json.dumps({"result": report["result"], "cases": len(observed), "experiment_counts": counts}, ensure_ascii=False))
