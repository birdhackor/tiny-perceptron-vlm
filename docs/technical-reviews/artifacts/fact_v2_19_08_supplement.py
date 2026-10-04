"""Bounded independent scoring, finite-action and pinned-link verification."""

import hashlib
import json
import platform
from pathlib import Path
from urllib.request import urlopen

import torch

from tiny_perceptron.alignment import sequence_log_probability
from tiny_perceptron.capstone import CapstoneModel, build_dataset, preference_pairs, prepare_batch
from tiny_perceptron.data import IGNORE
from tiny_perceptron.posttraining import FiniteResponsePolicy, bandit_advantage

torch.set_num_threads(2)
torch.manual_seed(42)
rows, _ = build_dataset()
pair = preference_pairs(rows["train"])[0]
model = CapstoneModel()
scores = []
for answer in (pair["chosen"], pair["rejected"]):
    batch, labels = prepare_batch([pair["row"]], answers=[answer])
    logits = model(**batch)["logits"]
    manual = 0.0
    for index, label in enumerate(labels[0].tolist()):
        if label != IGNORE:
            manual += float((logits[0, index, label] - torch.logsumexp(logits[0, index], 0)).detach())
    actual = float(sequence_log_probability(logits, labels).detach())
    assert abs(manual - actual) < 2e-5
    scores.append({"answer": answer, "manual_position_logit_minus_logsumexp_sum": manual, "library_sum": actual, "absolute_error": abs(manual - actual), "tolerance": 2e-5})
finite = FiniteResponsePolicy()
features = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
finite_shape = list(finite(features).shape)
assert finite_shape == [2, 4]
advantage = bandit_advantage(torch.tensor([1., 0.]), torch.tensor([.4, .4], requires_grad=True))
assert not advantage.requires_grad
checks = []
pin = "1df335318bda03fd771807f66976953231d5a00b"
paths = ("scripts/course_experiments/capstone.py", "docs/course-experiments/results/capstone_joint.json", "docs/course-experiments/results/capstone_preference.json", "docs/course-experiments/results/capstone_student.json", "docs/course-experiments/capstone-evidence/dpo/train-report.json", "docs/course-experiments/capstone-evidence/joint/validation.json", "docs/course-experiments/capstone-evidence/dpo/validation.json")
for path in paths:
    url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{pin}/{path}"
    with urlopen(url, timeout=30) as response:
        data = response.read()
        status = response.status
    local = Path(path).read_bytes()
    assert data == local
    checks.append({"url": url, "status": status, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "all_bytes_equal_local_reviewed_source": True})
result = {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_08_supplement.py", "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}, "result": "Independent per-position sum matches to 2e-5; finite policy has four action logits, detached bandit advantage; seven public GitHub pinned links match reviewed local bytes.", "sequence_scores": scores, "finite_response_shape": finite_shape, "advantage": advantage.tolist(), "advantage_requires_grad": advantage.requires_grad, "pinned_original_links": checks}
Path("docs/technical-reviews/artifacts/fact_v2_19_08_supplement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(result["result"])
