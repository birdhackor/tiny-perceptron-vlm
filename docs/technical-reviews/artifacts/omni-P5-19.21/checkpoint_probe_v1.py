"""Inspect preserved serialized training state; does not instantiate a model or execute NN inference."""

import hashlib
import io
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import torch


def main():
    proofs = Path(__file__).resolve().parent
    extracted = json.loads((proofs / "primary-extract-v1.json").read_bytes())
    archive = Path("/workspace/work/omni-core-completed-40988311100/core-checkpoints.zip")
    rows = {}
    with ZipFile(archive) as zipped:
        for job, record in extracted["training"].items():
            row = {}
            for name, expected in (("latest", 6000), ("best", record["best"]["update"])):
                entry = f"core/{job}/training/{name}-continuation.pt"
                raw = zipped.read(entry)
                checkpoint = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
                steps = sorted({float(value["step"]) for value in checkpoint["optimizer"]["state"].values()})
                parameter_count = sum(value.numel() for value in checkpoint["model"].values())
                # model state also contains the two non-trainable 8-frequency buffers.
                assert checkpoint["step"] == expected and steps == [float(expected)]
                row[name] = {"archive_entry": entry, "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest(), "checkpoint_step": checkpoint["step"],
                             "optimizer_state_steps": steps, "state_dict_tensor_elements": parameter_count,
                             "optimizer_parameter_states": len(checkpoint["optimizer"]["state"])}
            rows[job] = row
    result = {"created_at": datetime.now(UTC).isoformat(),
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "jobs": rows,
              "scope": "Read raw ZIP serialized state dictionaries with weights_only=True; inspect scalar checkpoint/optimizer steps and tensor element counts. No model instantiated, no trained/final neural inference or optimizer step."}
    output = proofs / "checkpoint-probe-result-v1.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"jobs": len(rows), "all_latest_steps": [row["latest"]["checkpoint_step"] for row in rows.values()],
                      "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
