"""Inspect common SFT provenance and the bounded QAT training-state contract."""

import copy
import hashlib
import json
import platform
from pathlib import Path

import torch

from scripts.course_experiments.compression import _parameter_hash, _qat_layers
from tiny_perceptron.training import load_checkpoint

ROOT = Path.cwd()
PREFIX = "fact_v2_17_14"
ART = ROOT / "docs/technical-reviews/artifacts"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


qat, _ = load_checkpoint("checkpoints/course/qat/initial_fp32.pt")
sft, _ = load_checkpoint("checkpoints/course/sft/model.pt")
baseline_equal = {}
for name, value in sft.state_dict().items():
    baseline_equal[name] = torch.equal(value, qat.state_dict()[name])
assert all(baseline_equal.values())
formal = json.loads(Path("docs/course-experiments/results/qat.json").read_text())
formal_sft = json.loads(Path("docs/course-experiments/results/sft.json").read_text())
teacher_sha = formal["results"]["teacher_provenance"]["sha256"]
assert teacher_sha == next(x["sha256"] for x in formal_sft["artifacts"] if x["path"] == "model.pt")
assert _parameter_hash(sft) == formal["results"]["initialization_sha256"]

checkpoint_path = ROOT / "outputs/technical-checks" / PREFIX / "cli/qat/qat_fake-training.pt"
payload = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
assert payload["optimizer"] is not None and payload["step"] == 1
assert len(payload["optimizer"]["state"]) > 0
plain, _ = load_checkpoint(checkpoint_path)
assert not any(type(layer).__name__ == "_QATLinear" for layer in plain.modules())
fake = _qat_layers(copy.deepcopy(plain))
optimizer = torch.optim.AdamW(fake.parameters(), lr=0.003, weight_decay=0.01)
optimizer.load_state_dict(payload["optimizer"])
assert len(optimizer.state) == len(payload["optimizer"]["state"])
steps = [float(state["step"]) for state in optimizer.state.values()]
assert set(steps) == {1.0}

receipt = {
    "command": f"PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/{PREFIX}_supplement.py",
    "result": "all assertions passed: every tensor in SFT public model equals initial_fp32; the one-update QAT training checkpoint contains AdamW state and it can be loaded after wrappers are restored",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "sources": {p: sha(p) for p in ["tiny_perceptron/training.py", "scripts/course_experiments/compression.py"]},
    "baseline": {
        "sft_public_file_sha256": sha("checkpoints/course/sft/model.pt"),
        "qat_initial_public_file_sha256": sha("checkpoints/course/qat/initial_fp32.pt"),
        "state_sha256": _parameter_hash(sft), "every_tensor_equal": baseline_equal,
        "formal_teacher_sha256": teacher_sha,
        "formal_sft_report_sha256": sha("docs/course-experiments/results/sft.json"),
        "formal_teacher_config": formal["results"]["teacher_provenance"]["config"],
    },
    "one_update_training_checkpoint": {
        "ignored_path": str(checkpoint_path.relative_to(ROOT)),
        "file_sha256": sha(checkpoint_path), "step": payload["step"],
        "optimizer_state_entries": len(optimizer.state), "optimizer_steps": steps,
        "plain_native_load_contains_wrapper": False,
        "wrapper_restored_optimizer_load_succeeded": True,
    },
    "limitations": "This verifies the code's training-state save/load behavior on the new one-update CPU smoke. It does not claim to have read the private original 350-update optimizer files; pinned private reads failed with existing credentials, as recorded separately. No exact-resume or fresh 350-update training claim is made.",
}
output = ART / f"{PREFIX}_supplement_execution.json"
output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": receipt["result"], "output": str(output.relative_to(ROOT))}, ensure_ascii=False))
