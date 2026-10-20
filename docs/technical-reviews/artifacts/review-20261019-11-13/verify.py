import contextlib
import hashlib
import html
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.natural_concepts import edit_distance, text_error_report
from scripts.course_experiments.run import experiment_spec


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, payload):
    with (ARTIFACT / name).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


@lru_cache(None)
def distance(left, right):
    if not left or not right:
        return len(left) + len(right)
    return min(
        distance(left[1:], right) + 1,
        distance(left, right[1:]) + 1,
        distance(left[1:], right[1:]) + (left[0] != right[0]),
    )


frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/11.13.md"
body = frozen.read_text()
code = re.search(r"```python\n(.*?)```", body, re.S).group(1)
assert sha(frozen) == "8c2be753e3742e1aadfa429765d03579ae022988352a71870dcb39310a582dba"

variants = []
for prediction, correct_cer in [("1000", False), ("1010", False), ("1010", True), ("101", False)]:
    candidate = code.replace('predicted = "1000"', f"predicted = {prediction!r}")
    if correct_cer:
        candidate = candidate.replace("1 / len(truth)", "0 / len(truth)")
    output, namespace, error = io.StringIO(), {}, None
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(candidate, str(frozen), "exec"), namespace)
    except AssertionError:
        error = "AssertionError"
    variants.append({
        "prediction": prediction,
        "zero_edit_line": correct_cer,
        "stdout": output.getvalue(),
        "exception": error,
        "matched": namespace.get("matched"),
    })
assert variants[0]["matched"] == 3
assert variants[0]["stdout"].splitlines() == [
    "等長逐位正確率 0.75", "整串正確 False", "本例一次替換的CER 0.25", "獨立錯誤假設下五位全對 0.5905"
]
assert variants[1]["matched"] == 4 and "本例一次替換的CER 0.25" in variants[1]["stdout"]
assert "本例一次替換的CER 0.0" in variants[2]["stdout"]
assert variants[3]["exception"] == "AssertionError" and variants[3]["stdout"] == ""

examples = []
for prediction, expected in [("1000", 1), ("10100", 1), ("010", 1), ("1010", 0), ("101", 1)]:
    edits = distance(prediction, "1010")
    report = text_error_report("1010", prediction)
    assert edits == expected == edit_distance("1010", prediction) == report["edits"]
    assert report["cer"] == edits / 4
    examples.append({"reference": "1010", "prediction": prediction, "edits": edits, "cer": edits / 4})

record_path = ROOT / "docs/course-experiments/results/ocr.json"
record = json.loads(record_path.read_text())
test = record["results"]["test"]
tok = ByteTokenizer()
rows = []
for sample in test["samples"]:
    ids = sample["generated_ids"]
    assert ids[-1] == tok.eos_id and ids.count(tok.eos_id) == 1
    assert tok.decode(ids[:-1]) == sample["generated"]
    assert sample["eos"] and sample["generation_error"] is None
    assert sample["invalid_special_tokens"] == 0
    exact = ids[:-1] == tok.encode(sample["target"])
    assert exact == sample["exact_match"] == (sample["generated"] == sample["target"])
    edits = distance(sample["generated"], sample["target"])
    assert edits == edit_distance(sample["target"], sample["generated"])
    rows.append({
        "row": sample["row"], "reference": sample["target"], "prediction": sample["generated"],
        "edits": edits, "reference_characters": len(sample["target"]), "exact": exact, "eos": True,
    })
totals = {
    "examples": len(rows), "eos_count": sum(row["eos"] for row in rows),
    "correct": sum(row["exact"] for row in rows), "character_edits": sum(row["edits"] for row in rows),
    "reference_characters": sum(row["reference_characters"] for row in rows),
}
assert totals == {"examples": 30, "eos_count": 30, "correct": 2, "character_edits": 40, "reference_characters": 57}
for key in ["examples", "correct", "character_edits", "reference_characters"]:
    assert test[key] == totals[key]
assert test["character_error_rate"] == totals["character_edits"] / totals["reference_characters"]
assert test["eos_rate"] == 1 and test["exact_match"] == 2 / 30
assert len(record["results"]["data"]["splits"]["test"]["records"]) == 30
assert [row["reference"] for row in rows] == [
    row["answer"] for row in record["results"]["data"]["splits"]["test"]["records"]
]

probability = Fraction(9, 10) ** 5
assert probability == Fraction(59049, 100000)
assert round(float(probability), 4) == 0.5905
correlated = {"all_correct": "9/10", "all_wrong": "1/10", "each_position_correct": "9/10"}
spec = experiment_spec("ocr")
assert spec["module"] == "modalities" and spec["function"] == "run_ocr"
cli = subprocess.run(
    [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--list-assets", "ocr"],
    cwd=ROOT, capture_output=True, text=True, check=True,
)
assert cli.stdout == "" and cli.stderr == ""

run_path = ROOT / "outputs/reviewer-tools/runs/20261020T054649415953Z"
copied = {name: (run_path / name).read_text() for name in [
    "stdout.txt", "stderr.txt", "execution.json", "environment.json", "extraction.json", "fence-1.py"
]}
assert copied["stdout.txt"] == variants[0]["stdout"]
assert json.loads(copied["execution.json"])["exit_code"] == 0
assert json.loads(copied["extraction.json"])["source_sha256"] == sha(frozen)
save("section-execution.json", {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/11.md#11.13" --execute',
    "origin": str(run_path.relative_to(ROOT)),
    "copied_files": copied,
    "copied_sha256": {name: sha(run_path / name) for name in copied},
})

requests = [
    ("cer", "https://raw.githubusercontent.com/Lightning-AI/torchmetrics/v1.8.2/src/torchmetrics/functional/text/cer.py", "v1.8.2", [
        "errors += _edit_distance(list(pred_tokens), list(tgt_tokens))", "total += len(tgt_tokens)", "return errors / total"
    ]),
    ("distance", "https://raw.githubusercontent.com/Lightning-AI/torchmetrics/v1.8.2/src/torchmetrics/functional/text/helper.py", "v1.8.2", [
        "dp[i][0] = i", "dp[0][j] = j", "dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1", "return dp[-1][-1]"
    ]),
    ("binomial", "https://www.itl.nist.gov/div898/handbook/eda/section3/eda366i.htm", "NIST/SEMATECH e-Handbook，讀取日的網頁版本", [
        "The binomial distribution assumes that p is fixed for all trials.", "(p)^{x}(1 - p)^{(n-x)}"
    ]),
    ("independence", "https://csrc.nist.gov/glossary/term/Binomial_Distribution", "NIST CSRC Glossary，引用 NIST SP 800-22 Rev. 1a", [
        "n independent Bernoulli experiments"
    ]),
    ("eos", "https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html", "PyTorch Tutorials 2.14.0+cu130 documentation", [
        "if it predicts the EOS token we stop there."
    ]),
]
sources = []
for identifier, url, version, excerpts in requests:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url]
    response = subprocess.run(command, capture_output=True, check=True)
    raw = response.stdout
    text = raw.decode()
    cleaned = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", text))) if not url.endswith(".py") else text
    for excerpt in excerpts:
        assert excerpt in cleaned
    sources.append({
        "id": identifier, "url": url, "version": version,
        "accessed_at": datetime.now(timezone.utc).isoformat(), "command_argv": command,
        "exit_code": response.returncode, "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw), "excerpts": excerpts,
    })
save("source-excerpts.json", {"sources": sources, "storage": "只保存支持主張的短摘錄與原始回應收據；未保存第三方全文。"})

paths = [
    "docs/course-experiments/results/ocr.json", "scripts/course_experiments/modalities.py",
    "tiny_perceptron/data.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/natural_concepts.py",
    "scripts/course_experiments/run.py", "docs/course-experiments/plan.json", "docs/course-experiments/README.md",
    "docs/review-tools/section_facts.py",
]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-13/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "source_sha256": sha(frozen), "repository_inputs": {path: sha(ROOT / path) for path in paths},
    "variants": variants, "edit_examples": examples,
    "probability": {"exact": str(probability), "decimal": float(probability), "rounded": round(float(probability), 4)},
    "correlated_counterexample": correlated,
    "ocr": {
        "rows": rows, "totals": totals, "cer": test["character_error_rate"], "cer_percent_rounded": round(test["character_error_rate"] * 100, 1),
        "run_environment": {key: record[key] for key in ["revision", "seed", "device", "gpu", "python_version", "torch_version", "step_scale", "evidence_status"]},
        "training": {key: record["results"]["training"][key] for key in ["config", "modal_config", "steps", "effective_tokens", "effective_targets", "cpu_smoke"]},
        "data": {"split_policy": record["results"]["data"]["split_policy"], "test_families": len({sample["family"] for sample in test["samples"]})},
        "scope": record["results"]["scope"],
    },
    "cli": {"command_argv": cli.args, "exit_code": cli.returncode, "stdout": cli.stdout, "stderr": cli.stderr, "experiment_spec": spec},
    "result": "所有算例、練習變體與逐筆 OCR 紀錄的斷言通過；沒有重跑訓練。",
}
save("verification-output.json", result)
print(json.dumps(result, ensure_ascii=False, indent=2))
