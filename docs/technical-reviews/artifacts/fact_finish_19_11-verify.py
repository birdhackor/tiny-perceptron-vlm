"""CPU mechanism and public-file audit; never reproduce a GPU quality claim."""

import copy
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import huggingface_hub
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import scripts.course_experiments.capstone as trainer  # noqa: E402
from tiny_perceptron.capstone import (  # noqa: E402
    CapstoneModel,
    build_dataset,
    load_capstone,
    prepare_batch,
    save_capstone,
)
from tiny_perceptron.model import masked_loss  # noqa: E402

ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_19_11"
PUBLIC = Path("/tmp/fact_finish_19_11-public")
RESULT = {
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "device": "CPU",
        "threads": "2",
    }
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_records(state):
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest(),
        }
        for name, value in state.items()
    }


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(value, b[key]) for key, value in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def run_command(args, expected_exit=0):
    command = [sys.executable, *args]
    execution = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert (execution.returncode == 0) == (expected_exit == 0), execution.stderr
    return {
        "command": command,
        "exit_code": execution.returncode,
        "stdout": execution.stdout,
        "stderr": execution.stderr,
    }


def run_stage(stage, output, **kwargs):
    observed_batches = []
    original = trainer._balanced_sample

    def traced_sample(rows, sampler, size):
        selected = original(rows, sampler, size)
        observed_batches.append([row["id"] for row in selected])
        return selected

    with patch.object(trainer, "_balanced_sample", traced_sample):
        report = trainer.train_stage(stage, output, device="cpu", validation=False, batch_size=2, **kwargs)
    return report, observed_batches


def record_report(report):
    return {
        k: report[k]
        for k in [
            "stage",
            "steps",
            "requested_steps",
            "new_steps",
            "schedule_completed",
            "budget_exhausted",
            "effective_tokens",
            "parent_checkpoint_sha256",
            "code_sha256",
        ]
    }


def resume_trial(stage, parent, directory):
    baseline_report, baseline_batches = run_stage(stage, directory / "baseline", input_checkpoint=parent, steps=4)
    ticks = count()
    with patch.object(trainer, "time", SimpleNamespace(perf_counter=lambda: float(next(ticks)))):
        pause_report, pause_batches = run_stage(
            stage, directory / "pause", input_checkpoint=parent, steps=4, seconds=2.5
        )
    assert pause_report["steps"] == 2 and not pause_report["schedule_completed"]
    partial = torch.load(directory / "pause/model-training.pt", weights_only=True)
    resumed_report, resume_batches = run_stage(
        stage, directory / "resumed", resume=directory / "pause/model-training.pt", steps=4
    )
    baseline = torch.load(directory / "baseline/model-training.pt", weights_only=True)
    resumed = torch.load(directory / "resumed/model-training.pt", weights_only=True)
    same_batches = baseline_batches == pause_batches + resume_batches
    checks = {
        key: equal(baseline[key], resumed[key])
        for key in ["model", "optimizer", "torch_rng", "python_rng", "reference"]
    }
    checks["sampler_rng"] = equal(baseline["training_state"]["sampler_rng"], resumed["training_state"]["sampler_rng"])
    max_diff = max((value - resumed["model"][name]).abs().max().item() for name, value in baseline["model"].items())
    assert same_batches and all(checks.values()) and max_diff == 0
    result = {
        "baseline": record_report(baseline_report),
        "paused": record_report(pause_report),
        "resumed": record_report(resumed_report),
        "baseline_batches": baseline_batches,
        "paused_batches": pause_batches,
        "resumed_batches": resume_batches,
        "batch_sequences_equal": same_batches,
        "state_equality": checks,
        "max_weight_difference": max_diff,
        "baseline_tensors": tensor_records(baseline["model"]),
        "resumed_tensors": tensor_records(resumed["model"]),
        "timing_note": "Pause used an isolated synthetic loop clock to stop exactly after two updates; no speed claim.",
        "parameters": baseline_report["parameters"],
        "batch_size": 2,
        "seed": 42,
        "data_manifest": baseline_report["data_manifest"],
    }
    if stage == "dpo":
        result["reference_preserved_from_partial"] = equal(partial["reference"], resumed["reference"])
        result["reference_differs_from_partial_policy"] = not equal(partial["reference"], partial["model"])
        assert result["reference_preserved_from_partial"] and result["reference_differs_from_partial_policy"]
    return result, directory / "pause/model-training.pt"


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def main():
    torch.set_num_threads(2)
    for name in ["torch-randomness", "torch-saving", "hf-download-guide"]:
        parser = TextOnly()
        parser.feed((ART / f"{PREFIX}-{name}.html").read_text())
        (ART / f"{PREFIX}-{name}.txt").write_text("\n".join(parser.parts) + "\n")
    headers = Path(huggingface_hub.__file__).parent / "utils/_headers.py"
    shutil.copyfile(headers, ART / f"{PREFIX}-hf-headers.py.txt")
    with TemporaryDirectory(prefix=PREFIX) as temporary:
        directory = Path(temporary)
        model = CapstoneModel()
        samples = []
        for step in [0, 1]:
            path = directory / f"example-{step}.pt"
            save_capstone(path, model, stage="sft", step=step, inference_only=True)
            loaded, metadata = load_capstone(path)
            row = build_dataset()[0]["train"][0]
            batch, _ = prepare_batch([row])
            with torch.no_grad():
                logits_equal = torch.equal(model(**batch)["logits"], loaded(**batch)["logits"])
            samples.append(
                {
                    "format": metadata["format_version"],
                    "step": metadata["step"],
                    "inference_only": metadata["inference_only"],
                    "description_equal": model.description() == loaded.description(),
                    "all_tensors_equal": equal(model.state_dict(), loaded.state_dict()),
                    "same_input_logits_equal": logits_equal,
                    "optimizer_updates": 0,
                }
            )
        RESULT["short_program_and_step_exercise"] = samples
        pre, _ = run_stage("pretrain", directory / "pre", steps=1, dense=True)
        sft, _ = run_stage("sft", directory / "sft", steps=1, input_checkpoint=directory / "pre/model.pt")
        joint, joint_partial = resume_trial("joint", directory / "sft/model.pt", directory / "joint")
        dpo, dpo_partial = resume_trial("dpo", directory / "joint/baseline/model.pt", directory / "dpo")
        RESULT["cpu_training"] = {
            "pretrain": record_report(pre),
            "sft": record_report(sft),
            "joint": joint,
            "dpo": dpo,
            "total_updates": 18,
            "scope": "Short Dense mechanism trial, not the 328128-parameter GPU recipe.",
        }
        guards = []

        def expect_error(name, expected, stage="joint", source=joint_partial, mutation=None, **kwargs):
            checkpoint = torch.load(source, weights_only=True)
            if mutation:
                mutation(checkpoint)
            path = directory / f"guard-{name}.pt"
            torch.save(checkpoint, path)
            options = {"steps": 4, "batch_size": 2, "validation": False, "resume": path}
            options.update(kwargs)
            try:
                trainer.train_stage(stage, directory / f"invalid-{name}", **options)
            except ValueError as error:
                assert expected in str(error), str(error)
                guards.append({"case": name, "error": str(error), "expected": expected})
            else:
                raise AssertionError(name)

        expect_error("steps", "fixed update schedule", steps=5)
        expect_error("batch", "batch size", batch_size=3)
        expect_error("stage", "same stage", stage="sft")
        expect_error("seed-data", "frozen training data", seed=43)
        expect_error(
            "code",
            "code hashes changed",
            mutation=lambda p: p["metadata"]["code_sha256"].update({"tiny_perceptron/capstone.py": "0" * 64}),
        )
        expect_error("optimizer", "no optimizer", mutation=lambda p: p.update(optimizer=None))
        expect_error(
            "reference",
            "original frozen reference",
            stage="dpo",
            source=dpo_partial,
            mutation=lambda p: p.update(reference=None),
        )
        expect_error("inference", "training checkpoint", mutation=lambda p: p.update(inference_only=True))
        expect_error(
            "unfinished-parent",
            "completed immediate predecessor",
            stage="dpo",
            source=joint_partial,
            resume=None,
            input_checkpoint=joint_partial,
        )
        RESULT["resume_and_parent_guards"] = guards
        commands = []
        for identity in ["joint", "joint-int4", "student-kd-int4"]:
            run = run_command(
                [
                    "scripts/capstone.py",
                    "infer",
                    "--checkpoint",
                    str(PUBLIC / identity / "model.pt"),
                    "--prompt",
                    "1+2等於多少？",
                    "--device",
                    "cpu",
                ]
            )
            run["parsed"] = json.loads(run["stdout"])
            commands.append(run)
        RESULT["public_inference_commands"] = commands
        errors = []
        for kind in ["--resume", "--input-checkpoint"]:
            stage = "joint" if kind == "--resume" else "dpo"
            errors.append(
                run_command(
                    [
                        "scripts/capstone.py",
                        "train",
                        "--stage",
                        stage,
                        kind,
                        str(PUBLIC / "joint/model.pt"),
                        "--output",
                        str(directory / kind[2:]),
                        "--steps",
                        "600" if stage == "joint" else "100",
                        "--device",
                        "cpu",
                    ],
                    expected_exit=1,
                )
            )
            assert "Checkpoint and frozen training data differ" in errors[-1]["stderr"]
        errors.append(
            run_command(
                ["scripts/capstone.py", "serve", "--checkpoint", str(PUBLIC / "joint-int4/model.pt")], expected_exit=1
            )
        )
        assert "Unsupported capstone checkpoint/data format" in errors[-1]["stderr"]
        errors.append(
            run_command(["scripts/fetch_capstone.py", "--stage", "joint", "--output", str(PUBLIC)], expected_exit=1)
        )
        assert "already exists" in errors[-1]["stderr"]
        RESULT["public_cli_rejections"] = errors
        model, _ = load_capstone(PUBLIC / "joint/model.pt")
        initial = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
        rows, _ = build_dataset()
        batch, labels = prepare_batch([rows["train"][0]])
        loss = masked_loss(model(**batch)["logits"], labels)
        loss.backward()
        optimizer.step()
        changed = [name for name, value in model.state_dict().items() if not torch.equal(value, initial[name])]
        assert changed and torch.isfinite(loss)
        RESULT["public_weights_new_experiment"] = {
            "updates": 1,
            "loss": float(loss.detach()),
            "changed_tensor_count": len(changed),
            "batch_size": 1,
            "note": "Independent new AdamW optimizer; no course resume assertion.",
        }
    raw_path = ROOT / "outputs/integration-runs/v2-joint/capstone-review/capstone_joint/gha-37168518451-1/model.pt"
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    public = torch.load(PUBLIC / "joint/model.pt", map_location="cpu", weights_only=True)
    checks = {name: torch.equal(value, public["model"][name]) for name, value in raw["model"].items()}
    assert set(raw["model"]) == set(public["model"]) and all(checks.values())
    RESULT["joint_repackaging"] = {
        "raw_observed_local_path": str(raw_path.relative_to(ROOT)),
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": sha(raw_path),
        "public_bytes": (PUBLIC / "joint/model.pt").stat().st_size,
        "public_sha256": sha(PUBLIC / "joint/model.pt"),
        "tensor_equality": checks,
        "raw_tensor_hashes": tensor_records(raw["model"]),
        "public_tensor_hashes": tensor_records(public["model"]),
        "raw_metadata_keys": sorted(raw["metadata"]),
        "public_metadata_keys": sorted(public["metadata"]),
        "binary_persistence": "Original binaries remain local/anonymous HF; no binary artifact in Git.",
    }
    (ART / f"{PREFIX}-verification.json").write_text(json.dumps(RESULT, ensure_ascii=False, indent=2) + "\n")
    print("Verified short save/load, 18 short CPU updates, exact joint/DPO resume, nine guards, three inference calls,")
    print("public CLI rejection, new training from public weights, and original joint/public tensor equality.")


if __name__ == "__main__":
    main()
