"""Anonymous pinned downloads of two representative R.4 models; short CPU inference."""

import hashlib
import io
import json
import platform
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.capstone_release import public_stage_id, validate_capstone_payload  # noqa: E402
from tiny_perceptron.capstone import build_dataset, load_capstone, run_assistant  # noqa: E402
from tiny_perceptron.capstone_quantization import load_quantized_capstone  # noqa: E402

OUT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())
    splits, _ = build_dataset()
    models = {}
    with tempfile.TemporaryDirectory(prefix="r4-public-cpu-") as directory:
        for stage in ("joint", "student-kd-int4"):
            spec = next(model for model in manifest["models"] if model["id"] == stage)
            target = Path(directory) / stage
            target.mkdir()
            downloads = []
            for item in spec["files"]:
                url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{item['path']}"
                with urlopen(url, timeout=30) as response:
                    raw = response.read()
                digest = hashlib.sha256(raw).hexdigest()
                assert len(raw) == item["bytes"] and digest == item["sha256"]
                (target / item["output"]).write_bytes(raw)
                downloads.append({"url": url, "output": item["output"], "sha256": digest, "bytes": len(raw)})
                if item["output"] in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
                    (OUT / "originals" / f"{stage}-{item['output']}").write_bytes(raw)
            loader = load_capstone if spec["format_version"] == "capstone-v1" else load_quantized_capstone
            model, payload = loader(target / spec["checkpoint"], "cpu")
            validate_capstone_payload(payload)
            assert public_stage_id(payload) == stage
            assert payload["inference_only"] and not set(payload) & {
                "optimizer",
                "training_state",
                "reference",
                "torch_rng",
                "python_rng",
                "cuda_rng",
            }
            tensor_records = [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                    "finite": bool(torch.isfinite(tensor).all()),
                    "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
                }
                for name, tensor in model.state_dict().items()
            ]
            selected = []
            for task in (
                ("calculator", "image_color", "image_shape", "audio", "joint") if stage == "joint" else ("calculator",)
            ):
                row = next(
                    row
                    for row in splits["test"]
                    if row["task"] == task and (task != "calculator" or row["user"] == "1+2等於多少？")
                )
                selected.append(
                    {
                        "id": row["id"],
                        "task": task,
                        "user": row["user"],
                        "expected": row["answer"],
                        "result": run_assistant(model, row),
                    }
                )
            models[stage] = {
                "downloads": downloads,
                "payload_keys": sorted(payload),
                "config": payload["config"],
                "metadata": payload["metadata"],
                "tensors": tensor_records,
                "samples": selected,
            }
    examples = {}
    for lesson in ("3.1", "15.4", "2.1"):
        import re

        snapshot = OUT / f"prerequisite-{lesson}.md"
        code = re.findall(r"```python\n(.*?)```", snapshot.read_text(), re.S)[0]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exec(compile(code, str(snapshot), "exec"), {"__name__": "__main__"})
        examples[lesson] = {
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "stdout": stdout.getvalue(),
            "default_device": str(torch.empty(0).device),
        }
    result = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_r_4/check_cpu_inference.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "scope": "Two anonymously downloaded fixed-revision files; same joint core for text/image/audio/tool samples. No retraining, no full 90-question CPU rescore and no GPU speed measurement.",
        "models": models,
        "linked_cpu_examples": examples,
    }
    (OUT / "cpu-inference-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "models": {
                    stage: {
                        "files": len(item["downloads"]),
                        "tensor_count": len(item["tensors"]),
                        "samples": [
                            {"task": sample["task"], "answer": sample["result"]["answer"]} for sample in item["samples"]
                        ],
                    }
                    for stage, item in models.items()
                },
                "examples": examples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
