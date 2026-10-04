"""Git-checkout replay using nine anonymously downloadable pinned checkpoints."""

import hashlib
import json
import platform
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from tiny_perceptron.capstone import build_dataset, evaluate_rows, load_capstone
from tiny_perceptron.capstone_quantization import load_quantized_capstone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_10_"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    torch.set_num_threads(2)
    public = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())
    historical = json.loads((OUT / f"{PREFIX}weight_inventory.json").read_text())
    test, _ = build_dataset(42)
    aliases = {
        "joint": "joint",
        "joint-int4": "joint4",
        "joint-int8": "joint8",
        "dpo": "dpo",
        "dpo-int4": "dpo4",
        "dpo-int8": "dpo8",
        "student-ce": "ce",
        "student-kd": "kd",
        "student-kd-int4": "kd4",
    }
    result = {
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "repo": public["repo"],
        "revision": public["revision"],
        "token": False,
        "models": {},
    }
    for name, alias in aliases.items():
        entry = next(m for m in public["models"] if m["id"] == name)
        file = next(f for f in entry["files"] if f["output"] == "model.pt")
        path = hf_hub_download(
            public["repo"],
            filename=file["path"],
            revision=public["revision"],
            token=False,
            local_dir=ROOT / "outputs/technical-sources/fact_v2_19_10_public",
        )
        raw = Path(path).read_bytes()
        assert sha(raw) == file["sha256"] and len(raw) == file["bytes"]
        model, payload = (
            load_quantized_capstone(path) if entry["format_version"] == "capstone-ptq-v1" else load_capstone(path)
        )
        state = {}
        for key, tensor in model.state_dict().items():
            state[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": tensor.numel() * tensor.element_size(),
                "sha256": sha(tensor.detach().cpu().contiguous().numpy().tobytes()),
            }
            previous = historical[alias]["restored_tensors"][key]
            assert all(state[key][k] == previous[k] for k in ("shape", "dtype", "bytes", "sha256"))
        folder, filename = (
            ("student", {"ce": "test-ce", "kd": "test-kd", "kd4": "test-kd-ptq4"}[alias])
            if alias in ("ce", "kd", "kd4")
            else (
                "deployment",
                {
                    "joint": "test-joint",
                    "joint4": "test-joint-ptq4",
                    "joint8": "test-joint-ptq8",
                    "dpo": "test-dpo",
                    "dpo4": "test-ptq4",
                    "dpo8": "test-ptq8",
                }[alias],
            )
        )
        saved = json.loads((ROOT / f"docs/course-experiments/capstone-evidence/{folder}/{filename}.json").read_text())
        evaluated = evaluate_rows(model, test["test"])
        for a, b in zip(evaluated["records"], saved["records"], strict=True):
            assert a["id"] == b["id"]
            for key in ("action_trace", "final_trace"):
                assert (a[key] is None) == (b[key] is None)
                if a[key] is not None:
                    assert a[key]["generated_ids"] == b[key]["generated_ids"]
        result["models"][name] = {
            "filename": file["path"],
            "bytes": len(raw),
            "sha256": sha(raw),
            "config": payload["config"],
            "metadata": payload["metadata"],
            "restored_tensors": state,
            "historical_tensors_exactly_equal": True,
            "cpu_test_count": evaluated["count"],
            "cpu_end_to_end_correct": evaluated["end_to_end_correct"],
            "cpu_all_generation_ids_equal_saved_gpu": True,
        }
        print(name, len(raw), evaluated["end_to_end_correct"], "all tensor and generation IDs equal", flush=True)
    (OUT / f"{PREFIX}public_audit_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
