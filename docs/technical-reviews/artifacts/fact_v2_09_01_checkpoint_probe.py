"""CPU verification of the fixed private PKU checkpoint, emitting no text/weights."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from scripts.course_experiments.common import _nll, evaluate_lm, records_sha256, text_examples
from scripts.course_experiments.text import _digest
from tiny_perceptron.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
# Independently inspected original GPU IDs, hashed by the reviewer's local audit.
GPU_GENERATION_HASHES = {
    "validation": [
        "eace0750e5099c690d7a9b5767f1e41d590a833e6cc2c28410ac25e2890141ac",
        "90b44379a602fe4ebba8374147daf500ba4d14e69aae3fa09b235d58f0989440",
        "c985194ebdf9612befde07947fec808b3d0ce16579035e6b89d5dac89cad9172",
        "30cb8098e1719cb5f7abd8a0455d712b6ff25799917d6adcacdb6dbef771861a",
        "57cd977403d253f2f3d5bb4e4fd9ab1c88be93023f665108344bdd4fd9b1b237",
        "863a6c218fc550c969cb6f7bda2d9fa02081252aa165780e421f42d5bee94d2e",
    ],
    "test": [
        "7034759cfb5df6628650b83e181a5388f3b90251ae81173589b2bd5fbed915a0",
        "ac83d4d4b644ff2341c2c1186bdef55138af8169cfefd1248469af008f8bf2dc",
        "b565435c1291bb8575e5eb8bd0dcd7d3a74e16bdf700f397eb889e31692d09f8",
        "c16a61895bd7d9e9339559b124f24051c038db248636e464149f6abfa298b65c",
        "7034759cfb5df6628650b83e181a5388f3b90251ae81173589b2bd5fbed915a0",
        "de0b5ae3571c7cee876d23f053aafb99c3444632d6a8414f459f1e07c6f37db8",
    ],
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--formal-result", type=Path, default=ROOT / "docs/course-experiments/results/safety.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    formal = json.loads(args.formal_result.read_text())
    pilot = formal["results"]["pku_pilot"]
    checkpoint = args.input / "pku-pilot.pt"
    checkpoint_spec = next(item for item in formal["artifacts"] if item["path"] == "pku-pilot.pt")
    assert checkpoint.stat().st_size == checkpoint_spec["bytes"] == 488461
    assert (
        sha256(checkpoint)
        == checkpoint_spec["sha256"]
        == "8f5e7857c1d5422bbf3adcef95a9ecba8f598b8a33ad2e0d931e4f08a8155c3b"
    )
    code_versions = {}
    for name in [
        "scripts/course_experiments/common.py",
        "scripts/course_experiments/text.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/model.py",
        "tiny_perceptron/training.py",
    ]:
        code_versions[name] = sha256(ROOT / name)
        assert code_versions[name] == formal["code_sha256"][name]
    # Local fresh audit has already reconstructed these three exact file hashes
    # from the public 100-row source archive. Verify the provided private files
    # before parsing; no additional source archive is required on this runner.
    parts, splits, seen_families = {}, {}, set()
    for name in ["train", "validation", "test"]:
        path = args.input / "pku-excerpts" / f"{name}.jsonl"
        specification = next(item for item in formal["artifacts"] if item["path"] == f"pku-excerpts/{name}.jsonl")
        assert sha256(path) == specification["sha256"] == pilot["data"][name]["sha256"]
        assert path.stat().st_size == specification["bytes"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == pilot["data"][name]["records"]
        families = {row["family"] for row in rows}
        assert not seen_families & families
        seen_families.update(families)
        parts[name] = rows
        splits[name] = {"records": len(rows), "jsonl_sha256": sha256(path), "bytes": path.stat().st_size}
    model, saved = load_checkpoint(checkpoint, "cpu")
    assert saved["step"] == saved["training_state"]["planned_steps"] == 80
    assert saved["training_state"]["batch_size"] == 4
    assert saved["metadata"]["effective_tokens"] == 37709
    assert saved["metadata"]["records_sha256"] == records_sha256(parts["train"])
    assert model.config.width == 32 and model.config.layers == 1 and model.config.max_length == 256
    assert model.config.vocab_size == 264
    assert {str(parameter.dtype) for parameter in model.parameters()} == {"torch.float32"}
    train_nll = _nll(model, text_examples(parts["train"], "sft", 256))
    assert train_nll["effective_tokens"] == 5599 and train_nll["examples"] == 48
    tolerance = 2e-6
    assert abs(train_nll["nll"] - pilot["training"]["final_loss"]) < tolerance
    complete_evaluations = {}
    for split in ["validation", "test"]:
        observed = evaluate_lm(model, parts[split], mode="sft", tokens=128)
        expected = pilot["evaluation"][split]
        assert observed["records"] == observed["examples"] == 6
        assert observed["effective_tokens"] == 726
        assert observed["matches"] == 0 and observed["eos_rate"] == 0
        assert abs(observed["nll"] - expected["nll"]) < tolerance
        items = []
        for row, sample, original_hash in zip(
            parts[split], observed["samples"], GPU_GENERATION_HASHES[split], strict=True
        ):
            ids_hash = _digest(sample["generated_ids"])
            assert len(sample["generated_ids"]) == 128 and not sample["exact"] and not sample["eos"]
            items.append(
                {
                    "source_row": row["source_row"],
                    "exact": sample["exact"],
                    "eos": sample["eos"],
                    "generated_tokens": len(sample["generated_ids"]),
                    "generated_ids_sha256": ids_hash,
                    "matches_original_gpu_ids": ids_hash == original_hash,
                }
            )
        complete_evaluations[split] = {key: value for key, value in observed.items() if key != "samples"}
        complete_evaluations[split]["all_samples"] = items
    output = {
        "command": "PYTHONPATH=. " + sys.executable + " " + " ".join(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "platform": platform.platform(),
            "threads": str(torch.get_num_threads()),
        },
        "result": "All fixed-input hash, step/config/target, NLL and complete 12-sample assertions passed",
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "formal_result_sha256": sha256(args.formal_result),
        "probe_sha256": sha256(Path(__file__)),
        "code_sha256": code_versions,
        "splits": splits,
        "config": saved["config"],
        "step": saved["step"],
        "effective_update_targets": 37709,
        "optimizer_step_values": sorted({float(state["step"]) for state in saved["optimizer"]["state"].values()}),
        "final_train_nll": train_nll,
        "formal_final_loss": pilot["training"]["final_loss"],
        "nll_tolerance": tolerance,
        "complete_evaluations": complete_evaluations,
        "scope": "Existing checkpoint read on CPU; no update, new training, GPU execution or private text output",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
