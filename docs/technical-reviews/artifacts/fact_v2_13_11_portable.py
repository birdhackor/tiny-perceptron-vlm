"""Export inspected originals once; recheck toy states from portable JSON only."""

import argparse
import builtins
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

import torch

from scripts.course_experiments.posttraining import _evaluate, _features, _state_sha256
from tiny_perceptron.posttraining import FiniteResponsePolicy, FiniteRewardModel, FiniteValueModel

ROOT = Path.cwd()
BASE = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_11_"
CLASSES = {"sft": FiniteResponsePolicy, "reward": FiniteRewardModel, "ppo": FiniteResponsePolicy,
           "value": FiniteValueModel, "dpo": FiniteResponsePolicy}
ENVIRONMENT = {"python": platform.python_version(), "torch": str(torch.__version__),
               "device": "cpu", "cpu_threads": "2", "platform": platform.platform()}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(name, value):
    path = BASE / (PREFIX + name)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_portable_state(name):
    path = BASE / (PREFIX + name + "_portable.json")
    data = json.loads(path.read_text())
    assert data["model_class"] == CLASSES[name].__name__
    state = {}
    for key, entry in data["tensors"].items():
        assert entry["dtype"] == "torch.float32", (name, key, entry["dtype"])
        tensor = torch.tensor(entry["values"], dtype=torch.float32).reshape(entry["shape"])
        assert tensor_sha(tensor) == entry["tensor_sha256"], (name, key)
        state[key] = tensor
    model = CLASSES[name]()
    assert model.state_dict().keys() == state.keys()
    model.load_state_dict(state, strict=True)
    assert _state_sha256(model) == data["state_sha256"]
    return {"model": state, "metadata": data["metadata"], "provenance": data["original_checkpoint"]}


@torch.no_grad()
def raw_predictions(records, models):
    return {
        split: {
            "families_modes": [[row["family"], row["mode"]] for row in rows],
            "outputs": {name: model(_features(rows, "cpu")).tolist() for name, model in models.items()},
        }
        for split, rows in records.items()
    }


def export_originals():
    old = json.loads((BASE / (PREFIX + "before_closure_13.11.json")).read_text())
    expected = {item["id"]: item for item in old["artifacts"]}
    report = json.loads((ROOT / "docs/course-experiments/results/posttraining.json").read_text())
    native_checkpoints = {Path(item["file"]).stem: item for item in report["results"]["checkpoints"]}
    copies = []
    for identifier, suffix in [("a_ppo_pdf", "ppo-original.pdf"),
                               ("a_instructgpt_pdf", "instructgpt-original.pdf"),
                               ("a_dpo_pdf", "dpo-original.pdf"), ("a_records", "records.json")]:
        item = expected[identifier]
        original = ROOT / item["path"]
        data = original.read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        if suffix.endswith(".pdf"):
            assert data.startswith(b"%PDF-")
        else:
            records = json.loads(data)
            assert sum(len(rows) for rows in records.values()) == 165
        target = BASE / (PREFIX + suffix)
        target.write_bytes(data)
        assert sha(target) == sha(original)
        copies.append({"original_path": item["path"], "copied_path": target.relative_to(ROOT).as_posix(),
                       "sha256": sha(target), "bytes": len(data), "identical_bytes": True})
    models, exports = {}, []
    for name, cls in CLASSES.items():
        original = ROOT / expected["a_checkpoint_" + name]["path"]
        data = original.read_bytes()
        original_sha = hashlib.sha256(data).hexdigest()
        assert original_sha == expected["a_checkpoint_" + name]["sha256"] == native_checkpoints[name]["sha256"]
        assert len(data) == native_checkpoints[name]["bytes"]
        state = torch.load(original, map_location="cpu", weights_only=True)
        assert state.keys() == {"model", "metadata"}
        model = cls().eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)
        assert _state_sha256(model) == native_checkpoints[name]["state_sha256"]
        tensors = {}
        for key, tensor in state["model"].items():
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
            tensors[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "tensor_sha256": tensor_sha(tensor), "values": tensor.tolist()}
        original_metadata = {"path": original.relative_to(ROOT).as_posix(), "sha256": original_sha,
                             "bytes": len(data), "read_with": "torch.load(weights_only=True,map_location='cpu')"}
        portable = {"schema_version": 1, "model_class": cls.__name__,
                    "original_checkpoint": original_metadata, "metadata": state["metadata"],
                    "state_sha256": _state_sha256(model),
                    "tensor_hash_encoding": "C-contiguous row-major IEEE-754 float32 bytes on little-endian CPU",
                    "tensors": tensors}
        target = write(name + "_portable.json", portable)
        restored = load_portable_state(name)
        assert restored["metadata"] == state["metadata"]
        for key, tensor in state["model"].items():
            assert torch.equal(tensor, restored["model"][key])
            assert tensor_sha(tensor) == tensor_sha(restored["model"][key])
        models[name] = model
        exports.append({"name": name, "original_checkpoint": original_metadata,
                        "portable_path": target.relative_to(ROOT).as_posix(), "portable_sha256": sha(target),
                        "state_sha256": portable["state_sha256"], "metadata": state["metadata"],
                        "tensors": {key: {k: v for k, v in entry.items() if k != "values"}
                                    for key, entry in tensors.items()},
                        "metadata_equal": True, "all_tensors_byte_identical_after_json_roundtrip": True})
    native = write("native_five_model_reference.json", raw_predictions(records, models))
    receipt = {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_11_portable.py --export",
               "result": "Three PDFs and records copied byte-for-byte; all five original checkpoints personally loaded; metadata and all 21 tensors preserved exactly in JSON; native model outputs saved for all 165 contexts.",
               "environment": ENVIRONMENT, "copies": copies, "checkpoints": exports,
               "native_reference_path": native.relative_to(ROOT).as_posix(), "native_reference_sha256": sha(native),
               "history_manifest": PREFIX + "closure_history.json", "original_files_retained": True,
               "training_performed": False}
    write("portable_export_receipt.json", receipt)
    print(receipt["result"])
    for item in exports:
        print(item["name"], item["original_checkpoint"]["sha256"], "tensors", len(item["tensors"]),
              "metadata", json.dumps(item["metadata"], ensure_ascii=False, sort_keys=True))


def install_original_read_guard():
    read_paths = []
    original_builtin, original_io = builtins.open, io.open

    def guard(fn, file, *args, **kwargs):
        if isinstance(file, (str, bytes, Path)):
            path = Path(file).resolve()
            assert not path.is_relative_to(ROOT / "outputs"), f"Ignored original read prohibited: {path}"
            if path.is_relative_to(BASE):
                read_paths.append(path.relative_to(ROOT).as_posix())
        return fn(file, *args, **kwargs)

    builtins.open = lambda file, *args, **kwargs: guard(original_builtin, file, *args, **kwargs)
    io.open = lambda file, *args, **kwargs: guard(original_io, file, *args, **kwargs)

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("torch.load is forbidden during portable-only verification")

    torch.load = forbidden_load
    return read_paths


def verify_portable():
    read_paths = install_original_read_guard()
    export = json.loads((BASE / (PREFIX + "portable_export_receipt.json")).read_text())
    records = json.loads((BASE / (PREFIX + "records.json")).read_text())
    report = json.loads((ROOT / "docs/course-experiments/results/posttraining.json").read_text())
    models, tensor_counts = {}, {}
    for name, cls in CLASSES.items():
        state = load_portable_state(name)
        model = cls().eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)
        models[name] = model
        tensor_counts[name] = len(state["model"])
        item = next(item for item in export["checkpoints"] if item["name"] == name)
        assert sha(ROOT / item["portable_path"]) == item["portable_sha256"]
        assert state["metadata"] == item["metadata"]
        assert _state_sha256(model) == item["state_sha256"]
    native_path = ROOT / export["native_reference_path"]
    assert sha(native_path) == export["native_reference_sha256"]
    native = json.loads(native_path.read_text())
    observed = raw_predictions(records, models)
    assert observed == native, "All five models' full-context outputs must match native reference exactly"
    policies = {name: models[name] for name in ("sft", "ppo", "dpo")}
    scale = report["results"]["reward"]["fixed_normalization_scale_from_train"]
    evaluations = {name: _evaluate(rows, policies, models["reward"], models["sft"], scale, "cpu")
                   for name, rows in records.items()}
    assert evaluations == report["results"]["evaluations"]
    assert evaluations == json.loads((BASE / (PREFIX + "frozen_evaluations.json")).read_text())
    for item in export["copies"]:
        assert sha(ROOT / item["copied_path"]) == item["sha256"]
    outputs = write("portable_five_model_outputs.json", observed)
    receipt = {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_11_portable.py",
               "result": "Portable-only CPU reconstruction passed: all 21 tensors' byte hashes; all five models' 165-context outputs exactly equal inspected native originals; all 165 complete evaluation rows equal historical results.",
               "environment": ENVIRONMENT, "torch_load_forbidden": True, "outputs_directory_read_forbidden": True,
               "checkpoint_format_used_for_reconstruction": "portable JSON only",
               "contexts_by_split": {name: len(rows) for name, rows in records.items()},
               "tensor_counts": tensor_counts, "maximum_absolute_output_error": 0.0,
               "portable_outputs_path": outputs.relative_to(ROOT).as_posix(), "portable_outputs_sha256": sha(outputs),
               "original_native_reference_sha256": export["native_reference_sha256"],
               "evidence_paths_opened": sorted(set(read_paths)), "training_performed": False}
    write("portable_verify_receipt.json", receipt)
    print(receipt["result"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Read originals once to export byte-preserving snapshots")
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert sys.byteorder == "little"
    if args.export:
        export_originals()
    else:
        verify_portable()


if __name__ == "__main__":
    main()
