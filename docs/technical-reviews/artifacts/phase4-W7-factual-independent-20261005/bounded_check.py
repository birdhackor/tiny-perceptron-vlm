"""W.7 independent checks: read-only paths/errors and one original dry-run."""
import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert (REPO / "pyproject.toml").is_file(), REPO
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
sys.path.insert(0, str(REPO))
import torch

assert torch.version.cuda is None
torch.set_num_threads(1)
facts = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_w_7",
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_git_version": torch.version.git_version,
    "cuda_build": str(torch.version.cuda),
    "device": "cpu",
    "scope": "Path/error observations and exactly one no-update pass; no training recipe or model downloads.",
}
fence = HERE / "original-fence/fence-1.py"
facts["original_fence_sha256"] = hashlib.sha256(fence.read_bytes()).hexdigest()
results = {}
for label, cwd in [("root", REPO), ("notebooks", REPO / "notebooks")]:
    result = subprocess.run(
        [sys.executable, "-B", str(fence)], cwd=cwd,
        capture_output=True, text=True, timeout=10, check=True,
    )
    lines = result.stdout.splitlines()
    expected = [str(cwd), "True", "True"] if label == "root" else [str(cwd), "False", "False"]
    assert lines == expected, (label, lines)
    assert not result.stderr
    results[label] = {"cwd": str(cwd), "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
facts["original_fence_runs"] = results
facts["layout"] = {p: (REPO / p).is_dir() for p in ["course/chapters", "notebooks", "tiny_perceptron", "scripts", "data", "checkpoints", "outputs"]}
assert all(facts["layout"].values())
facts["layout_examples"] = {p: (REPO / p).is_file() for p in ["course/chapters/01.md", "notebooks/01/1.1.ipynb", "scripts/check_env.py", "scripts/prepare_data.py", "scripts/train.py", "scripts/evaluate.py"]}
assert all(facts["layout_examples"].values())

errors = {}
for name, code, expected in [
    ("list", 'animals = ["貓", "狗"]; animals[2]', IndexError),
    ("dictionary", '{"animal": "貓"}["caption"]', KeyError),
    ("file", 'open("__W7_nonexistent_input_7194__.txt")', FileNotFoundError),
    ("module", 'import __W7_nonexistent_module_7194__', ModuleNotFoundError),
    ("missing_colon", 'if True\n    pass', SyntaxError),
    ("indentation", 'if True:\npass', IndentationError),
]:
    try:
        exec(compile(code, "W7-bounded-errors.py", "exec"), {})
    except expected as error:
        formatted = traceback.format_exc()
        assert "W7-bounded-errors.py" in formatted and "line" in formatted
        errors[name] = {"class": type(error).__name__, "message": str(error), "traceback": formatted}
    else:
        raise AssertionError((name, "expected exception did not happen"))
facts["errors"] = errors
assert issubclass(IndentationError, SyntaxError)

# -i gives the same interactive interpreter prompts when driven by bounded input.
result = subprocess.run([sys.executable, "-B", "-i"], cwd=REPO,
                        input=fence.read_text() + '\nexit()\n', capture_output=True, text=True, timeout=10)
assert result.returncode == 0 and ">>>" in result.stderr
assert result.stdout.splitlines() == [str(REPO), "True", "True"]
facts["interactive"] = {"command_argv": [sys.executable, "-B", "-i"], "input": fence.read_text() + '\nexit()\n',
                        "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode,
                        "scope": "Bounded piped interactive mode; visual terminal UI and Notebook kernel not inspected."}

step_calls = []
captured = []
original_adamw = torch.optim.AdamW
original_save = torch.save
class ObserveDryRun(original_adamw):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.param_groups:
            for parameter in group["params"]:
                captured.append((parameter, parameter.detach().clone()))
    def step(self, *args, **kwargs):
        step_calls.append(True)
        raise AssertionError("Dry-run must not call optimizer.step()")
def forbid_weight_save(*args, **kwargs):
    raise AssertionError("Dry-run must not save weights")
module = importlib.import_module("scripts.train")
torch.optim.AdamW = ObserveDryRun
torch.save = forbid_weight_save
previous_argv = sys.argv
raw_stdout = io.StringIO()
sys.argv = [str(REPO / "scripts/train.py"), "--task", "text", "--device", "cpu", "--width", "8", "--max-length", "32", "--batch-size", "1", "--output", str(HERE / "must-not-be-written.pt")]
try:
    with contextlib.redirect_stdout(raw_stdout):
        module.main()
finally:
    sys.argv = previous_argv
    torch.optim.AdamW = original_adamw
    torch.save = original_save
assert captured and not step_calls
assert all(torch.equal(parameter.detach(), before) for parameter, before in captured)
assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter, _ in captured)
report = json.loads(raw_stdout.getvalue().splitlines()[-1])
assert report["mode"] == "dry-run-no-weight-update" and len(report["history"]) == 1
assert not (HERE / "must-not-be-written.pt").exists()
assert module.parser().parse_args([]).train is False
assert module.parser().parse_args(["--train"]).train is True
(HERE / "dry-run-original-stdout.txt").write_text(raw_stdout.getvalue())
facts["dry_run"] = {"mode": report["mode"], "history_entries": len(report["history"]), "optimizer_step_calls": len(step_calls),
                    "captured_parameter_tensors": len(captured), "all_parameters_unchanged": True,
                    "nonzero_gradient_observed": True, "weight_file_created": False,
                    "default_train": False, "explicit_train_flag": True,
                    "bounded_overrides": {"width": 8, "max_length": 32, "batch_size": 1},
                    "scope": "Original main with AdamW observation guard and save prohibition. No --train run and no capability score."}
snapshots = {}
for name, module_obj in sorted(sys.modules.items()):
    path = getattr(module_obj, "__file__", None)
    if path and name.startswith(("tiny_perceptron", "scripts.")):
        path = Path(path).resolve()
        if path.is_relative_to(REPO):
            relative = path.relative_to(REPO)
            raw = path.read_bytes()
            target = HERE / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            snapshots[str(relative)] = hashlib.sha256(raw).hexdigest()
facts["loaded_repository_source_sha256"] = snapshots
(HERE / "bounded-result.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(facts, ensure_ascii=False, indent=2))
