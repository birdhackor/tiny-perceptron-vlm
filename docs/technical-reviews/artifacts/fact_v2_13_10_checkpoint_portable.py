"""Export the personally inspected RM checkpoint and verify it without binary weights.

Export reads the original local reward.pt. Verification reads only the durable JSON,
the existing complete experiment result, and repository model implementation.
"""

import argparse
import hashlib
import json
import platform
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.posttraining import FiniteRewardModel

ROOT = Path.cwd()
BASE = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_10"
DEFAULT_CHECKPOINT = BASE / f"{PREFIX}_cpu_run/reward.pt"
DEFAULT_EXPORT = BASE / f"{PREFIX}_reward_checkpoint.json"
RESULT = BASE / f"{PREFIX}_cpu_run/result.json"
AUDIT = BASE / f"{PREFIX}_audit_output.json"
PRIOR_REVIEW = BASE / f"{PREFIX}_before_portable_review.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def state_digest(state):
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor_bytes(tensor))
    return hasher.hexdigest()


def restore(snapshot):
    assert snapshot["schema_version"] == 1
    assert snapshot["byte_order"] == sys.byteorder == "little"
    restored = {}
    for item in snapshot["tensors"]:
        name = item["name"]
        assert name not in restored
        assert item["dtype"] == "torch.float32"
        tensor = torch.tensor(item["values"], dtype=torch.float32).reshape(item["shape"])
        payload = tensor_bytes(tensor)
        assert tensor.numel() == item["numel"]
        assert len(payload) == item["nbytes"]
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
        restored[name] = tensor
    assert state_digest(restored) == snapshot["state_sha256"]
    assert len(restored) == snapshot["tensor_count"]
    assert sum(t.numel() for t in restored.values()) == snapshot["state_elements"]
    return restored


def export_checkpoint(checkpoint_path, export_path):
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    prior_review = json.loads(PRIOR_REVIEW.read_text(encoding="utf-8"))
    old_artifact = next(item for item in prior_review["artifacts"] if item["id"] == "a_reward_checkpoint")
    assert set(original) == {"model", "metadata"}
    state = original["model"]
    assert len(state) == 5 and all(t.dtype == torch.float32 for t in state.values())
    assert all(torch.isfinite(t).all() for t in state.values())
    checkpoint_hash = digest(checkpoint_path)
    state_hash = state_digest(state)
    saved_checkpoint = next(item for item in report["results"]["checkpoints"] if item["file"] == "reward.pt")
    audited_checkpoint = next(item for item in audit["checkpoints"] if item["file"] == "reward.pt")
    assert checkpoint_hash == saved_checkpoint["sha256"] == audited_checkpoint["fresh_file_sha256"]
    assert checkpoint_hash == old_artifact["sha256"]
    assert state_hash == saved_checkpoint["state_sha256"] == audited_checkpoint["state_sha256"]
    assert checkpoint_path.stat().st_size == saved_checkpoint["bytes"]
    assert original["metadata"]["stage"] == "reward"
    assert original["metadata"]["config"] == report["results"]["config"]
    assert original["metadata"]["scale_from_training_candidates"] == report["results"]["reward"]["fixed_normalization_scale_from_train"]
    tensor_records = []
    for name, tensor in sorted(state.items()):
        payload = tensor_bytes(tensor)
        tensor_records.append({
            "name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "numel": tensor.numel(), "nbytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "role": "registered_buffer" if name == "candidate_identity" else "trainable_parameter",
            "values": tensor.detach().cpu().reshape(-1).tolist(),
        })
    snapshot = {
        "schema_version": 1, "reviewer_task": "/root/integration_technical_coordinator/fact_v2_13_10",
        "source_checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)), "bytes": checkpoint_path.stat().st_size,
                              "sha256": checkpoint_hash, "inspection": "Personally loaded original local file with torch.load(weights_only=True,map_location='cpu'); original binary retained locally."},
        "metadata": original["metadata"], "byte_order": sys.byteorder, "tensor_count": len(state),
        "state_elements": sum(t.numel() for t in state.values()), "parameter_elements": 241, "buffer_elements": 16,
        "state_sha256": state_hash, "state_hash_algorithm": "SHA256 of concatenated sorted UTF8 tensor names then their contiguous CPU C-order NumPy bytes; same method as original experiment _state_sha256.",
        "tensor_encoding": "Flattened finite float32 values encoded as exact Python float JSON numbers; restore in dtype float32 and original shape, verify raw byte hashes.",
        "tensors": tensor_records,
        "original_evidence": {"result_path": str(RESULT.relative_to(ROOT)), "result_sha256": digest(RESULT),
                              "audit_path": str(AUDIT.relative_to(ROOT)), "audit_sha256": digest(AUDIT),
                              "prior_review_path": str(PRIOR_REVIEW.relative_to(ROOT)), "prior_review_sha256": digest(PRIOR_REVIEW),
                              "prior_checkpoint_artifact_record": old_artifact,
                              "original_checkpoint_record": saved_checkpoint, "original_audit_checkpoint_record": audited_checkpoint},
        "inspection_environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    }
    reconstructed = restore(snapshot)
    for name in state:
        assert tensor_bytes(state[name]) == tensor_bytes(reconstructed[name])
    export_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return snapshot, {"source_checkpoint_sha256": checkpoint_hash, "all_tensor_bytes_roundtrip_exact": True,
                      "checkpoint_metadata_matches_original_experiment": True}


def verify_export(export_path):
    snapshot = json.loads(export_path.read_text(encoding="utf-8"))
    state = restore(snapshot)
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    assert digest(RESULT) == snapshot["original_evidence"]["result_sha256"]
    assert digest(AUDIT) == snapshot["original_evidence"]["audit_sha256"]
    assert digest(PRIOR_REVIEW) == snapshot["original_evidence"]["prior_review_sha256"]
    model = FiniteRewardModel().eval()
    model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == snapshot["parameter_elements"] == 241
    assert sum(b.numel() for b in model.buffers()) == snapshot["buffer_elements"] == 16
    checked = {}
    with torch.no_grad():
        for split, evaluation in report["results"]["evaluations"].items():
            rows = evaluation["rows"]
            features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
            expected = torch.tensor([row["reward_model_raw_scores"] for row in rows], dtype=torch.float32)
            observed = model(features)
            assert torch.equal(observed, expected), split
            checked[split] = {"contexts": len(rows), "raw_score_elements": observed.numel(),
                              "max_absolute_difference": float((observed - expected).abs().max()),
                              "all_raw_scores_exactly_equal": True}
    model.requires_grad_(False)
    features = torch.tensor([row["features"] for row in report["results"]["evaluations"]["test"]["rows"]])
    before = [parameter.detach().clone() for parameter in model.parameters()]
    policy_logits = torch.zeros((len(features), 4), requires_grad=True)
    optimizer = torch.optim.SGD([policy_logits], lr=0.1)
    loss = -(policy_logits.softmax(-1) * model(features)).sum(-1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert not torch.equal(policy_logits, torch.zeros_like(policy_logits))
    return snapshot, {"binary_checkpoint_read": False, "restored_state_sha256": state_digest(state),
                      "all_tensor_shape_dtype_hash_checks_passed": True, "evaluations": checked,
                      "frozen_rm_has_no_gradients": True, "frozen_rm_state_unchanged": True,
                      "downstream_policy_updated": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("export", "verify"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "export":
        snapshot, checks = export_checkpoint(args.checkpoint, args.export)
        _, portable_checks = verify_export(args.export)
        checks["portable_verification"] = portable_checks
    else:
        snapshot, checks = verify_export(args.export)
    command = "PYTHONPATH=. .venv/bin/python " + shlex.join(sys.argv)
    receipt = {
        "schema_version": 1, "reviewer_task": "/root/integration_technical_coordinator/fact_v2_13_10",
        "accessed_at": datetime.now(UTC).isoformat(), "command": command,
        "result": "Exit0; all checkpoint provenance, tensor and full score/frozen downstream checks passed.",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(),
                        "device": "cpu", "dtype": "torch.float32", "cpu_threads": "2", "byte_order": sys.byteorder},
        "export_path": str(args.export.relative_to(ROOT)), "export_sha256": digest(args.export),
        "source_checkpoint_sha256": snapshot["source_checkpoint"]["sha256"], "state_sha256": snapshot["state_sha256"],
        "verification_code_sha256": digest(Path(__file__)), "checks": checks,
        "checkpoint_argument_exists": args.checkpoint.exists(),
        "scope": "No training. Export inspects the original local binary. Verify reconstructs float32 tensors solely from persistent JSON and checks all165 contexts/660 scalar scores plus frozen downstream backward; original ZIP serialization bytes are identified by hash, not regenerated.",
    }
    path = BASE / f"{PREFIX}_checkpoint_{args.mode}_receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": args.mode, "source_checkpoint_sha256": receipt["source_checkpoint_sha256"],
                      "state_sha256": receipt["state_sha256"], "export_sha256": receipt["export_sha256"],
                      "result": receipt["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
