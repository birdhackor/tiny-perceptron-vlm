"""Read-only CPU verification of W.7; no setup, training, or saved weights."""

import contextlib
import hashlib
import io
import json
import platform
import re
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
PYTHON = ROOT / ".venv/bin/python"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv, cwd=ROOT, stdin=None):
    result = subprocess.run(argv, cwd=cwd, input=stdin, text=True, capture_output=True, check=False)
    return {
        "command": argv,
        "cwd": str(cwd),
        "stdin": stdin,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def dry_run(relative, arguments):
    original = torch.optim.AdamW
    instances = []

    class ObservedAdamW(original):
        def __init__(self, parameters, *args, **kwargs):
            parameters = list(parameters)
            super().__init__(parameters, *args, **kwargs)
            self.references = parameters
            self.before = [p.detach().clone() for p in parameters]
            self.step_calls = 0
            instances.append(self)

        def step(self, *args, **kwargs):
            self.step_calls += 1
            return super().step(*args, **kwargs)

    stream = io.StringIO()
    with (
        patch.object(sys, "argv", [relative, *arguments]),
        patch.object(torch.optim, "AdamW", ObservedAdamW),
        patch.object(torch, "save", side_effect=AssertionError("dry-run tried to save weights")) as save,
        contextlib.redirect_stdout(stream),
    ):
        runpy.run_path(str(ROOT / relative), run_name="__main__")
    optimizer = instances[0]
    comparisons = [torch.equal(before, after.detach()) for before, after in zip(optimizer.before, optimizer.references)]
    assert optimizer.step_calls == 0
    assert all(comparisons)
    assert save.call_count == 0
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records[-1]["mode"] == "dry-run-no-weight-update"
    return {
        "command": [str(PYTHON), relative, *arguments],
        "invocation_note": "runpy in this reviewer process; AdamW subclass observes constructor and step; torch.save would raise",
        "exit_code": 0,
        "stdout": stream.getvalue(),
        "optimizer_step_calls": optimizer.step_calls,
        "save_calls": save.call_count,
        "parameter_tensor_count": len(comparisons),
        "each_parameter_equal_before_after": comparisons,
        "all_parameters_exactly_unchanged": all(comparisons),
        "parameters_with_nonzero_gradient": sum(
            p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in optimizer.references
        ),
    }


def main():
    results = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}}
    path_block = 'from pathlib import Path\nprint(Path.cwd())\nprint(Path("pyproject.toml").exists())\nprint(Path("tiny_perceptron").exists())\n'
    results["path_root"] = command([str(PYTHON), "-B", "-c", path_block])
    results["path_notebooks"] = command([str(PYTHON), "-B", "-c", path_block], ROOT / "notebooks")
    assert results["path_root"]["stdout"].splitlines()[-2:] == ["True", "True"]
    assert results["path_notebooks"]["stdout"].splitlines()[-2:] == ["False", "False"]
    results["interactive"] = command([str(PYTHON), "-B", "-i"], stdin=path_block + "exit()\n")
    assert results["interactive"]["exit_code"] == 0
    assert ">>>" in results["interactive"]["stderr"]
    cases = {
        "ModuleNotFoundError": ["-S", "-c", "import torch"],
        "FileNotFoundError": ["-c", 'from pathlib import Path; Path("fact_finish_w_7_does_not_exist").read_text()'],
        "KeyError": ["-c", 'record = {"animal": "貓", "caption": "貓在睡覺"}; print(record["missing"])'],
        "IndexError": ["-c", 'animals = ["貓", "狗"]; print(animals[2])'],
        "SyntaxError": ["-c", "print("],
        "SyntaxError_colon": ["-c", "if True\n    pass"],
        "IndentationError": ["-c", "if True:\npass"],
    }
    results["exceptions"] = {}
    for expected, arguments in cases.items():
        observation = command([str(PYTHON), "-B", *arguments])
        name = expected.split("_")[0]
        assert observation["exit_code"] != 0
        assert name + ":" in observation["stderr"]
        results["exceptions"][expected] = observation
    results["git_ignore"] = command(
        ["git", "check-ignore", "data/reviewer.json", "checkpoints/reviewer.pt", "outputs/reviewer.json"]
    )
    assert results["git_ignore"]["exit_code"] == 0
    results["tracked_inventory"] = {}
    for folder in ("course/chapters", "notebooks", "tiny_perceptron", "scripts"):
        observation = command(["git", "ls-files", folder])
        files = observation["stdout"].splitlines()
        results["tracked_inventory"][folder] = {"command": observation["command"], "count": len(files), "files": files}
    results["chapters_numbered_md"] = all(
        re.fullmatch(r"(?:\d\d|0[A-Z])\.md", Path(p).name)
        for p in results["tracked_inventory"]["course/chapters"]["files"]
    )
    assert results["chapters_numbered_md"]
    notebook = json.loads((ROOT / "notebooks/01/1.1.ipynb").read_text(encoding="utf-8"))
    notebook_outputs = []
    namespace = {}
    for cell in notebook["cells"]:
        if (
            cell["cell_type"] == "code"
            and not cell.get("metadata", {}).get("course_setup")
            and not cell.get("metadata", {}).get("course_figure")
        ):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile("".join(cell["source"]), "notebooks/01/1.1.ipynb", "exec"), namespace)
            notebook_outputs.append(stream.getvalue())
    results["notebook"] = {
        "path": "notebooks/01/1.1.ipynb",
        "nbformat": notebook["nbformat"],
        "cell_types": [cell["cell_type"] for cell in notebook["cells"]],
        "computation_cells_executed": len(notebook_outputs),
        "stdout": notebook_outputs,
        "setup_and_figure_cells": "read in original; omitted from execution to avoid setup mutations or irrelevant display",
        "restored": namespace["restored"],
    }
    assert namespace["restored"] == "貓看狗，狗看貓。"
    results["shared_component"] = {
        "source": "tiny_perceptron/simple.py",
        "classes": ["BigramLM", "ContextMLP"],
    }
    from tiny_perceptron.simple import BigramLM, ContextMLP

    results["shared_component"]["BigramLM_output_shape"] = list(BigramLM(5)(torch.tensor([[0], [1]])).shape)
    results["shared_component"]["ContextMLP_output_shape"] = list(
        ContextMLP(5, context=3)(torch.tensor([[0, 1, 2]])).shape
    )
    results["entry_help"] = {}
    for name in ("check_env", "prepare_data", "train", "evaluate", "infer", "export_course"):
        observation = command([str(PYTHON), "-B", f"scripts/{name}.py", "--help"])
        # check_env has no parser and performs its short read-only environment check.
        assert observation["exit_code"] == 0
        results["entry_help"][name] = observation
    results["dry_runs"] = [
        dry_run("scripts/train.py", ["--task", "text", "--device", "cpu"]),
        dry_run("scripts/train.py", ["--task", "sft", "--device", "cpu"]),
        dry_run("scripts/train.py", ["--task", "vision", "--device", "cpu"]),
        dry_run("scripts/train_simple.py", ["--device", "cpu"]),
    ]
    from scripts.train import parser

    results["train_flag"] = {
        "default": parser().parse_args([]).train,
        "explicit": parser().parse_args(["--train"]).train,
    }
    assert results["train_flag"] == {"default": False, "explicit": True}
    from scripts.prepare_data import generate_records
    from tiny_perceptron.model import ModelConfig, TinyLM, generate
    from tiny_perceptron.training import load_checkpoint, save_checkpoint

    records = generate_records("toy-text")
    results["data_generator"] = {"kind": "toy-text", "count": len(records), "first_record": records[0]}
    torch.manual_seed(42)
    model = TinyLM(ModelConfig(width=8, max_length=8))
    optimizer = torch.optim.AdamW(model.parameters())
    captured = {}

    def capture_save(payload, destination):
        captured["payload"] = payload
        captured["destination"] = str(destination)

    # Exercise payload construction and strict reconstruction without any disk operation.
    with (
        patch.object(Path, "mkdir") as mkdir,
        patch.object(Path, "replace") as replace,
        patch.object(torch, "save", side_effect=capture_save),
    ):
        save_checkpoint("checkpoints/fact_finish_w_7_memory_only.pt", model, optimizer)
    payload = captured["payload"]
    with patch.object(torch, "load", return_value=payload):
        restored_model, restored_payload = load_checkpoint("checkpoints/fact_finish_w_7_memory_only.pt", "cpu")
    equal = {key: torch.equal(value, restored_model.state_dict()[key]) for key, value in model.state_dict().items()}
    assert all(equal.values())
    generated = generate(restored_model, torch.tensor([[1, 4]]), max_new_tokens=1)
    results["checkpoint_in_memory"] = {
        "note": "new random model, not a trained artifact; torch.save/load and Path.mkdir/replace intercepted; no weight file created",
        "format_version": payload["format_version"],
        "payload_fields": sorted(payload),
        "tensor_shapes": {key: list(value.shape) for key, value in payload["model"].items()},
        "all_tensors_exactly_equal_after_reconstruction": all(equal.values()),
        "per_tensor_equal": equal,
        "optimizer_present": restored_payload["optimizer"] is not None,
        "optimizer_populated": bool(restored_payload["optimizer"]["state"]),
        "step": payload["step"],
        "intercepted_mkdir_calls": mkdir.call_count,
        "intercepted_replace_calls": replace.call_count,
        "one_token_generation": generated.tolist(),
        "scope": "generic saved model state and inference wiring only; no quality claim or empirical exact-resume claim",
    }
    sources = [
        ".gitignore",
        "pyproject.toml",
        "zensical.toml",
        "scripts/check_env.py",
        "scripts/prepare_data.py",
        "scripts/train.py",
        "scripts/train_simple.py",
        "scripts/evaluate.py",
        "scripts/infer.py",
        "scripts/export_course.py",
        "tiny_perceptron/simple.py",
        "tiny_perceptron/training.py",
        "tiny_perceptron/model.py",
        "notebooks/01/1.1.ipynb",
    ]
    results["source_sha256"] = {p: sha(ROOT / p) for p in sources}
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
