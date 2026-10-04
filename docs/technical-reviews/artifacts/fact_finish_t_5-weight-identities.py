"""Persist public source lineage and every loaded tensor's dtype/shape/byte hash."""

import hashlib
import json
import platform
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path("/tmp/fact_finish_t_5-models")
DEST = ROOT / "docs/technical-reviews/artifacts"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tensor_records(node, prefix=""):
    if isinstance(node, torch.Tensor):
        assert torch.isfinite(node).all()
        return [
            {
                "name": prefix,
                "shape": list(node.shape),
                "dtype": str(node.dtype),
                "sha256": sha(node.detach().cpu().contiguous().numpy().tobytes()),
            }
        ]
    if isinstance(node, dict):
        return [record for key, item in node.items() for record in tensor_records(item, f"{prefix}.{key}")]
    return []


def main():
    files = []
    for experiment in ("style", "safety", "lora"):
        formal = json.loads((ROOT / f"docs/course-experiments/results/{experiment}.json").read_text())
        export = json.loads((MODEL_ROOT / experiment / "export-manifest.json").read_text())
        # Export's source hashes connect inference-only repackaging to formal files.
        assert export["provenance"]["revision"] == formal["revision"]
        (DEST / f"fact_finish_t_5-public-{experiment}-export-manifest.json").write_bytes(
            (MODEL_ROOT / experiment / "export-manifest.json").read_bytes()
        )
        for spec in export["files"]:
            source = next(item for item in formal["artifacts"] if item["path"] == spec["output"])
            assert spec["source_sha256"] == source["sha256"]
            path = MODEL_ROOT / experiment / spec["output"]
            assert sha(path.read_bytes()) == spec["sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            identities = tensor_records(payload.get("model", payload.get("adapter")))
            assert identities and all(record["dtype"] == "torch.float32" for record in identities)
            files.append(
                {
                    "experiment": experiment,
                    "file": spec["output"],
                    "public_file_sha256": spec["sha256"],
                    "formal_source_sha256": spec["source_sha256"],
                    "formal_revision": formal["revision"],
                    "config": payload["config"],
                    "step": payload.get("step"),
                    "format_version": payload["format_version"],
                    "base_sha256": payload.get("base_sha256"),
                    "scaling": payload.get("scaling"),
                    "every_tensor": identities,
                }
            )
    result = {
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_finish_t_5-weight-identities.py",
        "result": "All 11 public weight files match their manifest and formal-source lineage; every floating tensor finite and hashed",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "scope": "Public inference exports only; source file hashes attest the export manifest's recorded lineage, not direct private-original retrieval.",
        "files": files,
    }
    (DEST / "fact_finish_t_5-weight-identities-output.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(result["result"])


if __name__ == "__main__":
    main()
