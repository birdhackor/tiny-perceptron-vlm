"""Read-only bounded provenance checks and untrained CPU forward shape check for 19.20."""

from datetime import UTC, datetime
import hashlib
import inspect
import json
from pathlib import Path
import platform
import re

import torch

from tiny_perceptron.lesson_products.model import Config, LoopedSolver


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[4]
folder = Path(__file__).resolve().parent
page = root / "docs/course-revision-20261029-omni/reviews/wave-01/source-freeze/sources/19.20.md"
freeze = json.loads((root / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json").read_bytes())
item = next(x for x in freeze["pages"] if x["page_id"] == "19.20")
body = page.read_text()
assert sha(page) == item["source_sha256"]
live = (root / "course/chapters/19.md").read_text()
live_slice = re.search(r"^## 19\.20 .*?(?=^## |\Z)", live, re.M | re.S)[0]
assert live_slice == body
assert item["figures_sha256"] == {} and not re.search(r"!\[.*?\]\(", body)
assert "```" not in body
notebook = json.loads((root / item["notebook"]).read_bytes())
assert not any(x["cell_type"] == "code" for x in notebook["cells"])

torch.manual_seed(1920)
model = LoopedSolver(Config(vocab_size=11, width=8, heads=2, ff=16, max_length=8)).eval()
tokens = torch.tensor([[1, 3, 4], [1, 5, 0]])
with torch.no_grad():
    depths = model(tokens, loops=5, all_depths=True)
    beyond = model(tokens, loops=6)
assert list(depths.shape) == [2, 5, 19]
assert list(beyond.shape) == [2, 19]
train_source = root / "tiny_perceptron/lesson_products/training.py"
assert 'loops=5, all_depths=True' in train_source.read_text()

prior = folder / "execution-v1.json"
independent = json.loads(prior.read_bytes())
metadata_path = folder / "historical-final-review-metadata-v1.json"
metadata = json.loads(metadata_path.read_bytes())
assert datetime.fromisoformat(independent["created_at"]) < datetime.fromisoformat(metadata["created_at"])
assert metadata["prior_independent_execution"]["sha256"] == sha(prior)
tasks = set()
for entry in metadata["records"]:
    p = root / entry["path"]
    assert sha(p) == entry["sha256"]
    # Only allow-listed identity/command/artifact metadata is compared here.
    data = json.loads(p.read_bytes())
    tasks.add(data["reviewer_task"])
    assert data["reviewer_task"] == entry["read_metadata"]["reviewer_task"]
    references = data.get("execution_artifacts", [data["execution"]] if "execution" in data else [])
    for ref in references:
        assert sha(root / ref["path"]) == ref["sha256"]
assert tasks == {"/root/tech_20261016_19_15", "/root/tech_20261016_19_16"}

receipt_env = {}
for product in ("b", "c"):
    p = root / "outputs/lesson-products" / product / "training-receipt.json"
    data = json.loads(p.read_bytes())
    receipt_env[product] = {"sha256": sha(p), "seed": data["seed"], "mode": data["mode"], "reference_depth": data["reference_depth"], "environment": data["environment"], "selected_step": data["selected_step"]}
    assert "L4" in data["environment"]["gpu"]

loss_file = Path(inspect.getsourcefile(torch.nn.CrossEntropyLoss))
_, loss_line = inspect.getsourcelines(torch.nn.CrossEntropyLoss)
value = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-20-v1/verify_scope_and_metadata.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "result": "Frozen source and live exact slice match; no page figures or embedded code and notebook has no code cells; untrained CPU forward returns five supervised depth slots and accepts sixth inference depth; two distinct historical audit task identities and their execution artifact hashes match; preserved training receipts report L4.",
    "scope": "No training updates, final-model inference, formal heldout rerun or historical audit verdict/issues/results read. Tiny random model forward establishes API shape only, with no quality claim.",
    "source": {"path": str(page.relative_to(root)), "sha256": sha(page), "figure_map": {}, "notebook_sha256": sha(root / item["notebook"])},
    "tiny_forward": {"seed": 1920, "input_tokens": tokens.tolist(), "depths5_shape": list(depths.shape), "depth6_shape": list(beyond.shape), "weights": "untrained random", "optimizer_updates": 0},
    "prior_independent_execution": {"path": str(prior.relative_to(root)), "sha256": sha(prior), "created_at": independent["created_at"]},
    "historical_metadata": {"path": str(metadata_path.relative_to(root)), "sha256": sha(metadata_path), "created_at": metadata["created_at"], "distinct_tasks": sorted(tasks), "inspection_scope": metadata["allowed_scope"]},
    "preserved_training_receipts": receipt_env,
    "installed_pytorch_official_source": {"path": str(loss_file), "sha256": sha(loss_file), "class": "torch.nn.CrossEntropyLoss", "start_line": loss_line, "version": str(torch.__version__), "git_version": torch.version.git_version},
    "harness_development": "Before execution-v1.json was created, two harness assertions were corrected: artifact manifest entries contain nested sha256/bytes objects; JSON serialization converts arithmetic-label dictionary keys to strings. Neither failure concerned a textbook claim or altered repository evidence. Successful execution-v1.json was then sealed before any historical reviewer metadata inspection.",
}
with (folder / "scope-execution-v1.json").open("x") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"result": value["result"], "tiny_forward": value["tiny_forward"], "historical_audit_tasks": sorted(tasks)}, ensure_ascii=False))
