"""Same-reviewer fix13 callback; no training, GPU, external model or API call."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.lesson_products.data import CANDIDATES
from tiny_perceptron.lesson_products.inference import Product


ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
CACHE = Path("/workspace/work/reference/omni-technical-9-15-v1")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
ORIGINAL = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_9_15/technical-original-v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(code, label):
    namespace, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, label, "exec"), namespace)
    return namespace, output.getvalue().splitlines()


torch.set_num_threads(1)
original = json.loads(ORIGINAL.read_bytes())
assert sha(ORIGINAL) == "53fa5f0b6b8b5d1720cea5d3e66d00f6e37e6b878542a09510e3555172b4935f"
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "9.15")
snapshot = ROOT / page["snapshot"]
assert sha(snapshot) == page["source_sha256"] and page["figures_sha256"] == {}
old_sentence = "截至2026-10-20的官方文件，對 n 個 Choice 候選，使用"
new_sentence = "本節於2026-10-29核對的官方文件，對 n 個 Choice 候選，使用"
body = snapshot.read_text()
assert body == (ROOT / original["snapshot"]).read_text().replace(old_sentence, new_sentence, 1)
code = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(code) == 1
code = code[0]
namespace, output = execute(code, page["snapshot"] + ":16-19")
_, original_index = execute(code + "\nprint(p.index(max(p)))\n", "fix13-original-index")
exercise = code.replace("p = [0.6, 0.2, 0.2]", "p = [0.2, 0.6, 0.2]", 1)
_, exercise_output = execute(exercise + "\nprint(p.index(max(p)))\n", "fix13-exercise")
assert output == ["最大機率 0.6 集中度 0.4"]
assert original_index == ["最大機率 0.6 集中度 0.4", "0"]
assert exercise_output == ["最大機率 0.6 集中度 0.4", "1"]
cases = []
for p, expected in [([1 / 3] * 3, 0), ([1, 0, 0], 1), ([0.6, 0.4], 0.2), ([0.6, 0.2, 0.2], 0.4), ([0.99, 0.005, 0.005], 0.985)]:
    changed = code.replace("p = [0.6, 0.2, 0.2]", "p = " + repr(p), 1)
    values, _ = execute(changed, "fix13-boundary")
    assert math.isclose(values["confidence"], expected, rel_tol=0, abs_tol=1e-12)
    cases.append({"p": p, "concentration": values["confidence"], "expected": expected})
try:
    execute(code.replace("p = [0.6, 0.2, 0.2]", "p = [1.0]", 1), "fix13-singleton")
except ZeroDivisionError as error:
    singleton = type(error).__name__
else:
    raise AssertionError("Singleton must be undefined in the inline formula")

repository_pins = {}
for source in original["sources"]:
    if source["id"] in {"s-inference", "s-model", "s-data"}:
        digest = sha(ROOT / source["path"])
        assert digest == source["sha256"]
        repository_pins[source["path"]] = digest
product = Product(CACHE / "zero-weight-cpu-fixture", "c", device="cpu")
api = {}
for question, candidates in CANDIDATES.items():
    answer = product.judge("卡片進度;申請編號=?", question)
    assert list(answer["probabilities"]) == list(candidates) and "confidence" not in answer
    assert math.isclose(sum(answer["probabilities"].values()), 1, rel_tol=0, abs_tol=1e-6)
    n = len(candidates)
    expected = (max(answer["probabilities"].values()) - 1 / n) / (1 - 1 / n)
    assert math.isclose(answer["concentration"], expected, rel_tol=0, abs_tol=1e-12)
    api[question] = {key: value for key, value in answer.items() if key != "seconds"}
try:
    product.judge("卡片進度;申請編號=?", "choice", ["none"])
except ValueError as error:
    api_singleton = str(error)
else:
    raise AssertionError("Actual Product API must reject a singleton")

class FixedScores(torch.nn.Module):
    def forward(self, tokens):
        assert tokens.shape[0] == 3
        return torch.tensor([math.log(0.5), -math.inf, math.log(0.5)])

product.model = FixedScores()
rubric = product.judge("卡片進度;申請編號=?", "rubric")
assert list(rubric["probabilities"].values()) == [0.5, 0, 0.5]
assert rubric["concentration"] == 0.25
official_score = max(0, 1 - 1 / (2 / 3))
assert official_score == 0

source_receipt_path = BASE / "source-inspection-v1.json"
source_receipt = json.loads(source_receipt_path.read_bytes())
for entry in source_receipt["originals"]:
    assert sha(Path(entry["cache_path"])) == entry["sha256"]
headers = Path(source_receipt["historical_date_check"]["current_response_headers_path"])
assert sha(headers) == source_receipt["historical_date_check"]["current_response_headers_sha256"]
assert "date: Thu, 29 Oct 2026 14:43:18 GMT" in headers.read_text()
assert r"\text{confidence} = \frac{p_{\max} - \frac{1}{n}}{1 - \frac{1}{n}}" in (CACHE / "typesafe-confidence.md").read_text()

context = []
for lesson in ["9.9", "9.10", "9.13", "9.14", "9.15", "9.18", "19.16", "19.23"]:
    entry = next(p for p in freeze["pages"] if p["page_id"] == lesson)
    assert sha(ROOT / entry["snapshot"]) == entry["source_sha256"]
    current = (ROOT / entry["source"]).read_text()
    start = re.search(r"^## " + re.escape(lesson) + r" ", current, re.M).start()
    following = re.search(r"^## ", current[start + 1:], re.M)
    current = current[start:start + 1 + following.start()] if following else current[start:]
    assert hashlib.sha256(current.encode()).hexdigest() == entry["source_sha256"]
    context.append({"lesson_id": lesson, "snapshot": entry["snapshot"], "source_sha256": entry["source_sha256"], "current_section_matches": True})
policy_page = next(p for p in freeze["pages"] if p["page_id"] == "9.18")
policy_code = re.findall(r"```python\n(.*?)```", (ROOT / policy_page["snapshot"]).read_text(), re.S)[0]
policy_namespace, policy_output = execute(policy_code, policy_page["snapshot"])
assert policy_namespace["T"] == 1.0 and policy_namespace["tau"] == 0.9
assert 'take = p[best] >= tau and ids[best] != "none" and r["info_pred"]' in policy_code

notebook_path = ROOT / page["notebook"]
assert sha(notebook_path) == page["notebook_sha256"]
notebook = json.loads(notebook_path.read_bytes())
notebook_cells = ["".join(cell["source"]) for cell in notebook["cells"]]
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
assert "".join(code_cells[1]["source"]).strip() == code.strip()
cumulative_path = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix13-cumulative.json"
cumulative = json.loads(cumulative_path.read_bytes())
record = next(row for row in cumulative["records"] if row["lesson"] == "9.15")
executed_path = Path(record["source"])
assert record["notebook_sha256"] == sha(notebook_path)
assert sha(executed_path) == record["executed_sha256"]
executed = json.loads(executed_path.read_bytes())
assert ["".join(cell["source"]) for cell in executed["cells"]] == notebook_cells
assert not any(output["output_type"] == "error" for cell in executed["cells"] if cell["cell_type"] == "code" for output in cell["outputs"])
kernel_output = ["".join(out.get("text", [])) for cell in executed["cells"] if cell["cell_type"] == "code" and "".join(cell["source"]).strip() == code.strip() for out in cell["outputs"]]
assert kernel_output == ["最大機率 0.6 集中度 0.4\n"]

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-15-v1/verify_current_lesson.py > docs/technical-reviews/artifacts/omni-technical-9-15-v1/execution-current-v1.json",
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_git_version": torch.version.git_version, "platform": platform.platform(), "device": "cpu", "threads": "1"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE)},
    "current_snapshot": {"path": page["snapshot"], "sha256": sha(snapshot), "figures_sha256": {}, "complete_text_read": True},
    "change": {"old_quote": old_sentence, "current_quote": new_sentence, "only_changed_text": True},
    "inline": {"code_sha256": hashlib.sha256(code.encode()).hexdigest(), "output": output, "original_plus_index_output": original_index, "exercise_output": exercise_output, "numeric_cases": cases, "singleton_exception": singleton},
    "api": {"scope": "Actual current Product constructor/judge with the original untrained zero-weight CPU fixture; controlled rubric scorer explicitly mocked. No trained quality claim.", "all_three_results": api, "singleton_rejection": api_singleton, "rubric_mock": {"probabilities": rubric["probabilities"], "local_concentration": 0.25, "official_score_formula": official_score}},
    "repository_sources": repository_pins,
    "context_pins": context,
    "linked_9_18": {"scope": "Reexecuted unchanged stipulated-score inline demonstration", "output": policy_output},
    "official_original_reuse": {"receipt_path": str(source_receipt_path.relative_to(ROOT)), "receipt_sha256": sha(source_receipt_path), "all_original_hashes_match": True, "headers_sha256": sha(headers), "response_date": "2026-10-29T14:43:18Z", "inspection": "Re-read original formula, Noul/Score definitions and response headers. The current 2026-10-29 inspection wording is supported; no assertion about 2026-10-20 is made."},
    "notebook": {"path": page["notebook"], "sha256": sha(notebook_path), "lesson_code_matches": True, "provided_cumulative_receipt": {"path": str(cumulative_path.relative_to(ROOT)), "sha256": sha(cumulative_path), "record_pointer": "/records/118"}, "provided_executed_copy": {"path": str(executed_path), "sha256": sha(executed_path), "current_cells_match": True, "lesson_output": kernel_output}, "scope": "Reviewed only the matching 9.15 kernel copy. Kernel execution belongs to the supplied author receipt; reviewer separately executed inline code/exercise/CPU API above. No certification of the other 330 records."},
    "result": "All assertions passed: full fix13 lesson differs only in inspection-date wording; that wording matches original 2026-10-29 official receipt; exact inline code/exercise, numerical and singleton boundaries, all three API fields, ordinal-summary difference, and p_max policy semantics are reverified. No new issue found."
}
print(json.dumps(result, ensure_ascii=False, indent=2))
