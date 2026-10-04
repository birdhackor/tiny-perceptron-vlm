"""Fresh bounded CPU audit of W.7; no training updates, downloads, or site rebuild."""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "docs/technical-reviews/artifacts"
sys.path.insert(0, str(ROOT))
import torch

torch.set_num_threads(1)
assert torch.version.cuda is None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exporter():
    spec = importlib.util.spec_from_file_location("w7_exporter", ROOT / "scripts/export_course.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_code(cwd, code):
    previous = Path.cwd()
    stream = io.StringIO()
    try:
        os.chdir(cwd)
        with contextlib.redirect_stdout(stream):
            exec(compile(code, "W.7 original fence", "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(previous)
    return stream.getvalue().splitlines()


code_path = ARTIFACTS / "tool-choice-dep-w7-original-code.py"
protected = [ROOT / "pyproject.toml", ROOT / "tiny_perceptron/__init__.py", ROOT / "notebooks/01/1.1.ipynb"]
before = {str(path.relative_to(ROOT)): sha(path) for path in protected}
root_result = run_code(ROOT, code_path.read_bytes())
notebooks_result = run_code(ROOT / "notebooks", code_path.read_bytes())
assert root_result == [str(ROOT), "True", "True"]
assert notebooks_result == [str(ROOT / "notebooks"), "False", "False"]
assert {str(path.relative_to(ROOT)): sha(path) for path in protected} == before
paths = {}
for name in ("course/chapters", "notebooks", "tiny_perceptron", "scripts", "data", "checkpoints", "outputs"):
    path = ROOT / name
    assert path.is_dir(), name
    paths[name] = {"exists": True, "representative_entries": sorted(p.name for p in path.iterdir())[:5]}
assert (ROOT / "scripts/check_env.py").is_file()
assert not (ROOT / "notebooks/scripts/check_env.py").exists()
ignored = subprocess.run(
    ["git", "check-ignore", "--no-index", "-v", "data/w7-review-probe.json", "checkpoints/w7-review-probe.pt", "outputs/w7-review-probe.json"],
    cwd=ROOT, capture_output=True, text=True, check=True, timeout=10,
)
assert "/data/" in ignored.stdout and "/checkpoints/" in ignored.stdout and "/outputs/" in ignored.stdout

errors = {}
for expected, operation in (
    ("FileNotFoundError", lambda: (ROOT / "outputs/reviewer-tools/tool-choice-dep-w7-no-such-file").read_text()),
    ("KeyError", lambda: {"animal": "貓"}["caption"]),
    ("IndexError", lambda: ["貓", "狗"][2]),
    ("SyntaxError", lambda: compile("for animal in ['貓', '狗']\n    print(animal)\n", "missing-colon", "exec")),
):
    try:
        operation()
        raise AssertionError(f"{expected} did not occur")
    except Exception as error:
        assert type(error).__name__ == expected
        errors[expected] = str(error)
missing_torch = subprocess.run(
    [sys.executable, "-S", "-c", "import torch"], cwd=ROOT, capture_output=True, text=True, timeout=10,
)
assert missing_torch.returncode != 0
assert missing_torch.stderr.splitlines()[-1] == "ModuleNotFoundError: No module named 'torch'"
errors["ModuleNotFoundError"] = {"command": [sys.executable, "-S", "-c", "import torch"], "exit_code": missing_torch.returncode, "last_line": missing_torch.stderr.splitlines()[-1], "scope": "-S deliberately hides installed site-packages; this is not evidence torch is absent from the prepared venv."}
repl = subprocess.run(
    [sys.executable, "-i", "-q"], cwd=ROOT / "notebooks", input=code_path.read_text() + "\nexit()\n",
    capture_output=True, text=True, timeout=10,
)
assert repl.returncode == 0 and ">>>" in repl.stderr
assert repl.stdout.splitlines() == notebooks_result

# The unmodified CLI runs once without --train. Record exact parameter equality,
# and fail if the implementation attempts either an optimizer update or weight save.
parameters = []
original_init = torch.optim.AdamW.__init__
original_step = torch.optim.AdamW.step
original_save = torch.save


def capture_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    for group in self.param_groups:
        parameters.extend((p, p.detach().clone()) for p in group["params"])


def forbid_update(*args, **kwargs):
    raise AssertionError("dry-run attempted a weight update/save")


weight_path = ROOT / "outputs/reviewer-tools/tool-choice-dep-w7-audit-dryrun.pt"
assert not weight_path.exists()
old_argv = sys.argv
old_cwd = Path.cwd()
dry_stream = io.StringIO()
try:
    os.chdir(ROOT)
    sys.argv = ["scripts/train_simple.py", "--model", "bigram", "--device", "cpu", "--output", str(weight_path)]
    torch.optim.AdamW.__init__ = capture_init
    torch.optim.AdamW.step = forbid_update
    torch.save = forbid_update
    with contextlib.redirect_stdout(dry_stream):
        runpy.run_path(str(ROOT / "scripts/train_simple.py"), run_name="__main__")
finally:
    torch.optim.AdamW.__init__ = original_init
    torch.optim.AdamW.step = original_step
    torch.save = original_save
    sys.argv = old_argv
    os.chdir(old_cwd)
dry_report = json.loads(dry_stream.getvalue())
assert dry_report["mode"] == "dry-run-no-weight-update"
assert math.isfinite(dry_report["train_loss"]) and math.isfinite(dry_report["validation_loss"])
assert parameters and all(torch.equal(p.detach(), initial) for p, initial in parameters)
assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p, _ in parameters)
assert not weight_path.exists()

exporter = load_exporter()
index = json.loads((ROOT / "course/lesson-index.json").read_text())
executed_root = ROOT / "outputs/tool-choice-site-kernels"
assert len(index) == 226
assert len(list(executed_root.rglob("*.ipynb"))) == 226
checked = 0
for item in index:
    original = json.loads((ROOT / item["notebook"]).read_text())
    executed = exporter.checked_notebook(original, executed_root / Path(item["notebook"]).relative_to("notebooks"))
    assert all(cell.get("execution_count") is not None for cell in executed["cells"] if cell["cell_type"] == "code")
    checked += 1
settings = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
docs = ROOT / settings["docs_dir"]


def nav_paths(items):
    for entry in items:
        if isinstance(entry, str):
            yield entry
        else:
            for value in entry.values():
                yield from nav_paths(value) if isinstance(value, list) else [value]


nav = list(nav_paths(settings["nav"]))
assert set(nav) == {path.name for path in docs.glob("*.md")}
assert len(nav) == len(set(nav))
assert "first-steps.md" in nav and "training.md" in nav
site_info = json.loads((ROOT / "outputs/site/build-info.json").read_text())
assert site_info["lessons"] == 226 and site_info["executed_cpu_outputs"] is True
from bs4 import BeautifulSoup
site = BeautifulSoup((ROOT / "outputs/site/first-steps.html").read_text(), "html.parser")
assert site.find(id="W.7") is not None
assert site.find(id="W.1") is not None and site.find(id="W.2") is not None
main = site.select_one("article")
links = [(a.get_text(" ", strip=True), a.get("href")) for a in main.select("a[href]")]
assert any(href == "#W.1" and "W.1" in text for text, href in links)
assert any(href == "#W.2" and "W.2" in text for text, href in links)
assert any(href == "training.html" and "操作配方" in text for text, href in links)
assert (ROOT / "outputs/site/training.html").is_file()
assert (ROOT / "outputs/site/notebooks/01/1.1.ipynb").is_file()
assert "226 個小節" in (ROOT / "outputs/site/index.html").read_text()

result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "cwd": str(ROOT), "zensical": site_info["builder_version"]},
    "original_fence_sha256": sha(code_path),
    "actual_root_output": root_result,
    "actual_notebooks_exercise_output": notebooks_result,
    "protected_files_before_and_after_sha256": before,
    "directory_table": paths,
    "git_check_ignore_output": ignored.stdout,
    "exceptions": errors,
    "interactive_notebooks_run": {"argv": [sys.executable, "-i", "-q"], "input": code_path.read_text() + "\nexit()\n", "stdout": repl.stdout, "stderr": repl.stderr, "exit_code": repl.returncode},
    "dry_run": {"argv": ["scripts/train_simple.py", "--model", "bigram", "--device", "cpu", "--output", str(weight_path)], "report": dry_report, "parameters_compared": sum(p.numel() for p, _ in parameters), "parameters_bitwise_unchanged": True, "finite_nonzero_gradient_present": True, "optimizer_step_and_torch_save_forbidden": True, "weight_file_exists": weight_path.exists()},
    "export_navigation": {"executed_input": str(executed_root.relative_to(ROOT)), "executed_copies_checked_against_current_source": checked, "chapters_lesson_index_count": len(index), "nav_entries": len(nav), "nav_matches_generated_markdown_inventory": True, "site_info": site_info, "site_output": "outputs/site", "old_outputs_notebooks_used": False, "first_steps_anchors": ["W.1", "W.2", "W.7"], "relevant_links": [[text, href] for text, href in links if href in ("#W.1", "#W.2", "training.html")]},
    "limits": "Only W.7 original and exercise were freshly run. The 226 CPU copies were read and source/count/error checked, not rerun. Site files were read and links inspected, not rebuilt or browser-tested. Dry-run was instrumented to forbid updates/saves; --train branch was inspected, never executed. No GPU or training. Public Colab/Jupyter service availability and every possible Python path/error situation are outside scope."
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
