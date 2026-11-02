"""9.15 reviewer execution: frozen inline code and synthetic CPU API contract.

No training, external model, GPU, or API call. The Product fixture has zero
weights and is only evidence for return fields and the concentration formula.
"""

import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_perceptron.lesson_products.data import CANDIDATES
from tiny_perceptron.lesson_products.inference import Product
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, Tokenizer


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
CACHE = Path("/workspace/work/reference/omni-technical-9-15-v1")
COMMAND = (
    "uv run --extra cpu python "
    "docs/technical-reviews/artifacts/omni-technical-9-15-v1/verify_lesson.py"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(code, name):
    namespace, capture = {}, io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, name, "exec"), namespace)
    return namespace, capture.getvalue().splitlines()


torch.set_num_threads(1)
torch.manual_seed(0)
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "9.15")
snapshot = ROOT / page["snapshot"]
assert sha(snapshot) == page["source_sha256"]
assert page["figures_sha256"] == {}
body = snapshot.read_text()
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
inline = blocks[0]
namespace, original_output = execute(inline, page["snapshot"] + ":16-19")
assert namespace["p"] == [0.6, 0.2, 0.2]
assert namespace["n"] == 3
assert original_output == ["最大機率 0.6 集中度 0.4"]
_, original_index_output = execute(inline + "\nprint(p.index(max(p)))\n", "original-plus-index")
exercise = inline.replace("p = [0.6, 0.2, 0.2]", "p = [0.2, 0.6, 0.2]", 1)
exercise_namespace, exercise_output = execute(exercise + "\nprint(p.index(max(p)))\n", "exercise")
assert original_index_output == ["最大機率 0.6 集中度 0.4", "0"]
assert exercise_output == ["最大機率 0.6 集中度 0.4", "1"]
assert namespace["confidence"] == exercise_namespace["confidence"]

notebook_path = ROOT / page["notebook"]
assert sha(notebook_path) == page["notebook_sha256"]
notebook = json.loads(notebook_path.read_bytes())
cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
assert len(cells) == 2 and cells[1].strip() == inline.strip()

def concentration(p):
    n = len(p)
    return (max(p) - 1 / n) / (1 - 1 / n)

numeric_cases = []
for name, p, expected in [
    ("uniform-three", [1 / 3, 1 / 3, 1 / 3], 0.0),
    ("point-mass-three", [1.0, 0.0, 0.0], 1.0),
    ("same-max-two", [0.6, 0.4], 0.2),
    ("same-max-three", [0.6, 0.2, 0.2], 0.4),
    ("wrong-high-max-three", [0.99, 0.005, 0.005], 0.985),
]:
    c = concentration(p)
    assert math.isclose(c, expected, rel_tol=0, abs_tol=1e-12)
    numeric_cases.append({"name": name, "p": p, "concentration": c, "expected": expected})
try:
    concentration([1.0])
except ZeroDivisionError as error:
    single_candidate = type(error).__name__
else:
    raise AssertionError("Single-candidate inline formula should divide by zero")

# Actual Product initialization and judge method; fixture weights are synthetic.
fixture = CACHE / "zero-weight-cpu-fixture"
fixture.mkdir(parents=True, exist_ok=True)
tokenizer = Tokenizer()
tokenizer.save(fixture / "tokenizer.json")
config = Config(len(tokenizer), width=16, heads=2, ff=32, c_layers=1)
config.save(fixture / "config.json")
model = CandidateScorer(config)
with torch.no_grad():
    for parameter in model.parameters():
        parameter.zero_()
torch.save(model.state_dict(), fixture / "model.pt")
(fixture / "calibration.json").write_text(json.dumps({"temperatures": {q: 1.0 for q in CANDIDATES}}))
files = ("model.pt", "config.json", "tokenizer.json", "calibration.json")
manifest = {"version": 1, "product": "c", "files": {name: {"sha256": sha(fixture / name), "bytes": (fixture / name).stat().st_size} for name in files}}
(fixture / "artifact-manifest.json").write_text(json.dumps(manifest))
product = Product(fixture, "c", device="cpu")
api_results = {}
for question, expected_candidates in CANDIDATES.items():
    answer = product.judge("卡片進度;申請編號=?", question)
    assert list(answer["probabilities"]) == list(expected_candidates)
    assert "confidence" not in answer
    assert math.isclose(sum(answer["probabilities"].values()), 1, rel_tol=0, abs_tol=1e-6)
    n = len(expected_candidates)
    expected_c = (max(answer["probabilities"].values()) - 1 / n) / (1 - 1 / n)
    assert math.isclose(answer["concentration"], expected_c, rel_tol=0, abs_tol=1e-12)
    api_results[question] = {key: value for key, value in answer.items() if key != "seconds"}
try:
    product.judge("卡片進度;申請編號=?", "choice", ["none"])
except ValueError as error:
    single_candidate_api = str(error)
else:
    raise AssertionError("Product must reject one candidate")

class FixedScores(torch.nn.Module):
    def forward(self, tokens):
        assert tokens.shape[0] == 3
        return torch.tensor([math.log(0.5), -math.inf, math.log(0.5)], dtype=torch.float32)

product.model = FixedScores()
rubric = product.judge("卡片進度;申請編號=?", "rubric")
assert list(rubric["probabilities"].values()) == [0.5, 0.0, 0.5]
assert rubric["concentration"] == 0.25
# Official Score formula, read in confidence.md lines 150-160 and 190-195.
p = list(rubric["probabilities"].values())
m = p.index(max(p))
spread = sum(value * abs(i - m) for i, value in enumerate(p))
even_spread = sum(abs(i - (len(p) - 1) / 2) for i in range(len(p))) / len(p)
official_score_confidence = max(0.0, 1 - spread / even_spread)
assert official_score_confidence == 0.0

# Execute only the linked 9.18 policy demonstration to verify the present
# threshold semantics. It contains stipulated scores, not a trained model.
policy_page = next(p for p in freeze["pages"] if p["page_id"] == "9.18")
policy_path = ROOT / policy_page["snapshot"]
assert sha(policy_path) == policy_page["source_sha256"]
policy_code = re.findall(r"```python\n(.*?)```", policy_path.read_text(), re.S)[0]
policy_namespace, policy_output = execute(policy_code, policy_page["snapshot"])
assert policy_namespace["tau"] == 0.9 and policy_namespace["T"] == 1.0
assert 'take = p[best] >= tau and ids[best] != "none" and r["info_pred"]' in policy_code
assert policy_output[-2:] == ["全部final 接受 4 / 6 錯誤 2 coverage 0.6667 risk 0.5", "零接受 (0, 0, 0.0, None)"]

context = []
for lesson in ["9.9", "9.10", "9.13", "9.14", "9.15", "9.18", "19.16"]:
    entry = next(p for p in freeze["pages"] if p["page_id"] == lesson)
    frozen = ROOT / entry["snapshot"]
    assert sha(frozen) == entry["source_sha256"]
    text = (ROOT / entry["source"]).read_text()
    start = re.search(r"^## " + re.escape(lesson) + r" ", text, re.M).start()
    following = re.search(r"^## ", text[start + 1:], re.M)
    current = text[start:start + 1 + following.start()] if following else text[start:]
    current_sha = hashlib.sha256(current.encode()).hexdigest()
    assert current_sha == entry["source_sha256"]
    context.append({"lesson_id": lesson, "snapshot": entry["snapshot"], "source_sha256": entry["source_sha256"], "current_section_matches": True, "inspection_scope": "9.15 only, or explicitly linked textual context; no independent claim of full technical review of context lesson"})

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": COMMAND,
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "cpu", "platform": platform.platform(), "threads": "1", "seed": "0"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE)},
    "inline": {"snapshot": page["snapshot"], "source_sha256": sha(snapshot), "lines": "16-19", "code_sha256": hashlib.sha256(inline.encode()).hexdigest(), "original_output": original_output, "original_plus_index_output": original_index_output, "exercise_output": exercise_output, "notebook_sha256": sha(notebook_path), "notebook_lesson_cell_matches": True, "colab_bootstrap_executed": False},
    "numeric_cases": numeric_cases,
    "one_candidate_inline_exception": single_candidate,
    "one_candidate_product_rejection": single_candidate_api,
    "api_fixture": {"scope": "Synthetic zero weights; actual Product constructor, tokenizer, CandidateScorer and judge on CPU. No fitted model quality, official Jev request, training or formal GPU result was tested.", "config": {"width": 16, "heads": 2, "ff": 32, "c_layers": 1}, "results": api_results},
    "rubric_controlled_scores": {"scope": "Mock scorer only; actual Product.judge. This is formula/field evidence, not learned ordinal capability.", "probabilities": rubric["probabilities"], "local_concentration": rubric["concentration"], "official_score_formula_on_same_probabilities": official_score_confidence},
    "linked_9_18": {"snapshot": policy_page["snapshot"], "source_sha256": sha(policy_path), "scope": "Stipulated-score policy demonstration only", "output": policy_output},
    "context_pins": context,
    "repository_sources": {p: sha(ROOT / p) for p in ["tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/data.py"]},
    "torch_softmax": {"path": inspect.getsourcefile(F.softmax), "file_sha256": sha(Path(inspect.getsourcefile(F.softmax))), "lines": "2176-2216", "inspection": "Formula exp(x_i)/sum(exp(x_j)); slices along dim sum to 1; calls input.softmax(dim)."},
    "result": "All assertions passed. Exact inline lesson and exercise outputs match; candidate counts and singleton boundary match; three typed interfaces return complete probabilities and local Choice concentration, distinct from official Score confidence; linked acceptance rule uses p_max and predicted information sufficiency.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
