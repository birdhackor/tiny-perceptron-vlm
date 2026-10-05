"""Independent bounded W.7 probes; product files are only read."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import pty
import select
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parent
WORK = ROOT / "outputs/natural-v4/factual-research/W.7"
PYTHON = ROOT / ".venv/bin/python"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_HISTORY": str(WORK / "python-history"), "IPYTHONDIR": str(WORK / "ipython")}
WORK.mkdir(parents=True, exist_ok=True)
raw = (ROOT / "course/first-steps.md").read_bytes()
section = raw.decode("utf-8").split("## W.7 ", 1)[1]
section = "## W.7 " + section
fence = section.split("```python\n", 1)[1].split("```", 1)[0]
(ART / "lesson_fence.py").write_bytes(fence.encode("utf-8"))

def run(label, args, cwd=ROOT, expected_exit=0):
    completed = subprocess.run([str(a) for a in args], cwd=cwd, env=ENV, capture_output=True, text=True, timeout=40)
    record = {"label": label, "argv": [str(a) for a in args], "cwd": str(cwd), "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "expected_exit": expected_exit}
    assert completed.returncode == expected_exit, record
    return record

records = []
for directory, expected in [(ROOT, ["True", "True"]), (ROOT / "notebooks", ["False", "False"])]:
    rec = run("unchanged lesson fence", [PYTHON, "-B", "-c", fence], directory)
    assert rec["stdout"].splitlines() == [str(directory), *expected], rec
    records.append(rec)
records.append(run("relative scripts pathname from notebooks", [PYTHON, "-B", "scripts/check_env.py"], ROOT / "notebooks", 2))
rec = run("same files still exist from another origin", [PYTHON, "-B", "-c", 'from pathlib import Path; print(Path("../pyproject.toml").exists()); print(Path("../tiny_perceptron").exists()); print(Path("scripts/check_env.py").exists())'], ROOT / "notebooks")
assert rec["stdout"].splitlines() == ["True", "True", "False"]
records.append(rec)

# Audit only the unchanged fence, after its stdlib import is available.
import pathlib
events = []
def audit(event, args):
    if event.startswith("os.") or event == "open":
        events.append([event, repr(args)])
sys.addaudithook(audit)
before = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
events.clear()
exec(compile(fence, "W.7 unchanged fence", "exec"), {})
fence_events = list(events)
after = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
assert before == after
assert not fence_events, fence_events

# A real PTY runs the stated activate/python/exit/cd exercise.
master, slave = pty.openpty()
shell_env = {**ENV, "TERM": "dumb", "PS1": "W7SHELL> ", "PS2": "W7MORE> ", "PROMPT_COMMAND": ""}
shell = subprocess.Popen(["bash", "--noprofile", "--norc", "-i"], cwd=ROOT, env=shell_env, stdin=slave, stdout=slave, stderr=slave)
os.close(slave)
transcript = bytearray()
steps = []
def await_prompt(prompt):
    chunk = bytearray()
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            data = os.read(master, 16384)
            chunk.extend(data)
            transcript.extend(data)
            if prompt.encode() in chunk:
                return chunk.decode("utf-8", errors="replace")
    raise RuntimeError("PTY prompt timeout: " + chunk.decode("utf-8", errors="replace"))
try:
    await_prompt("W7SHELL> ")
    for command, prompt in [
        ("source .venv/bin/activate", "W7SHELL> "),
        ("python", ">>> "),
        ("from pathlib import Path", ">>> "),
        ("print(Path.cwd())", ">>> "),
        ('print(Path("pyproject.toml").exists())', ">>> "),
        ('print(Path("tiny_perceptron").exists())', ">>> "),
        ("exit()", "W7SHELL> "),
        ("cd notebooks", "W7SHELL> "),
        ("python", ">>> "),
        ("from pathlib import Path", ">>> "),
        ("print(Path.cwd())", ">>> "),
        ('print(Path("pyproject.toml").exists())', ">>> "),
        ('print(Path("tiny_perceptron").exists())', ">>> "),
        ("exit()", "W7SHELL> "),
        ("cd ..", "W7SHELL> "),
        ("pwd", "W7SHELL> "),
    ]:
        os.write(master, (command + "\n").encode())
        output = await_prompt(prompt)
        steps.append({"command": command, "expected_prompt": prompt, "observed": output})
    os.write(master, b"exit\n")
    shell.wait(timeout=5)
finally:
    if shell.poll() is None:
        shell.kill()
        shell.wait()
    os.close(master)
(ART / "repl-transcript.txt").write_bytes(transcript)
assert "\r\nTrue\r\n" in steps[4]["observed"]
assert "\r\nTrue\r\n" in steps[5]["observed"]
assert "\r\nFalse\r\n" in steps[11]["observed"]
assert "\r\nFalse\r\n" in steps[12]["observed"]
assert str(ROOT) in steps[-1]["observed"]

exception_cases = [
    ("missing torch in deliberately isolated interpreter", ["-S", "-c", "import torch"], "ModuleNotFoundError", "No module named 'torch'"),
    ("requested file missing from notebooks origin", ["-c", 'from pathlib import Path; Path("pyproject.toml").read_text()'], "FileNotFoundError", "pyproject.toml"),
    ("missing dictionary key", ["-c", 'record={"animal":"貓"}; print(record["caption"])'], "KeyError", "caption"),
    ("third item in two-item list", ["-c", 'animals=["貓","狗"]; print(animals[2])'], "IndexError", "list index out of range"),
    ("unclosed parenthesis", ["-c", 'print("貓"'], "SyntaxError", "was never closed"),
    ("missing colon", ["-c", 'if True\n    print("貓")'], "SyntaxError", "expected ':'"),
    ("incorrect indentation", ["-c", 'if True:\nprint("貓")'], "IndentationError", "expected an indented block"),
]
for label, args, exception, message in exception_cases:
    rec = run(label, [PYTHON, "-B", *args], ROOT / "notebooks", 1)
    assert exception + ":" in rec["stderr"] and message in rec["stderr"], rec
    rec["expected_exception"] = exception
    records.append(rec)
rec = run("valid indices and out-of-range slice control", [PYTHON, "-B", "-c", 'animals=["貓","狗"]; print(animals[0],animals[1],len(animals)); print(animals[2:]); print(issubclass(IndentationError,SyntaxError))'])
assert rec["stdout"].splitlines() == ["貓 狗 2", "[]", "True"]
records.append(rec)

# Kernel display name and argv bind to the currently usable venv.
kernel_rec = run("registered kernel", [PYTHON, "-m", "jupyter", "kernelspec", "list", "--json"])
spec = json.loads(kernel_rec["stdout"])["kernelspecs"]["tiny-perceptron"]["spec"]
assert spec["display_name"] == "Tiny Perceptron"
assert spec["argv"][0] == str(PYTHON)
records.append(kernel_rec)
ipython_rec = run("IPython cwd may change", [PYTHON, "-m", "IPython", "--no-banner", "--colors=NoColor", "--HistoryManager.hist_file=" + str(WORK / "ipython-history.sqlite"), "-c", 'from pathlib import Path\nprint(Path.cwd())\n%cd notebooks\nprint(Path("pyproject.toml").exists())\n%cd ..\nprint(Path("pyproject.toml").exists())'])
assert "False" in ipython_rec["stdout"] and ipython_rec["stdout"].rstrip().endswith("True")
records.append(ipython_rec)

import nbformat
from nbclient import NotebookClient
nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(fence), nbformat.v4.new_code_cell('import sys, torch\nprint(sys.executable)\nprint(torch.__version__)')])
client = NotebookClient(nb, timeout=35, kernel_name="tiny-perceptron", resources={"metadata": {"path": str(ROOT / "notebooks" / "01")}})
client.execute(env=ENV)
notebook_outputs = [[dict(output) for output in cell["outputs"]] for cell in nb["cells"]]
assert notebook_outputs[0][0]["text"].splitlines() == [str(ROOT / "notebooks" / "01"), "False", "False"]
assert str(PYTHON) in notebook_outputs[1][0]["text"]

git_rec = run("actual repository ignore rules", ["git", "check-ignore", "-v", "--no-index", "data/w7-review-probe", "checkpoints/w7-review-probe.pt", "outputs/w7-review-probe.json", "notebooks/01/1.1.ipynb"])
assert len(git_rec["stdout"].splitlines()) == 3
records.append(git_rec)
directory_inventory = {p: (ROOT / p).is_dir() for p in ["course/chapters", "notebooks", "tiny_perceptron", "scripts", "data", "checkpoints", "outputs"]}
table_examples = {p: (ROOT / p).is_file() for p in ["course/chapters/01.md", "notebooks/01/1.1.ipynb", "pyproject.toml", "scripts/check_env.py", "scripts/prepare_data.py", "scripts/train.py", "scripts/evaluate.py"]}
assert all(directory_inventory.values()) and all(table_examples.values())
report = {"source_sha256": hashlib.sha256(section.encode()).hexdigest(), "environment": {"python": platform.python_version(), "python_executable": sys.executable, "torch": importlib.metadata.version("torch"), "IPython": importlib.metadata.version("IPython"), "ipykernel": importlib.metadata.version("ipykernel"), "jupyter_server": importlib.metadata.version("jupyter_server"), "device": "cpu"}, "records": records, "read_only_fence": {"audited_filesystem_events": fence_events, "pyproject_sha256_before": before, "pyproject_sha256_after": after}, "repl": {"process_exit": shell.returncode, "steps": steps}, "notebook": {"cwd": str(ROOT / "notebooks" / "01"), "kernel": "tiny-perceptron", "outputs": notebook_outputs}, "directories": directory_inventory, "table_examples": table_examples, "case_counts": {"raw_fence_cwds": 2, "exception_cases": len(exception_cases), "list_control": 1, "repl_shell_commands": len(steps), "notebook_cells": 2, "ignore_matches": 3}}
(ART / "paths-errors-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source_sha256": report["source_sha256"], "environment": report["environment"], "case_counts": report["case_counts"], "result": "all bounded path/REPL/exception/kernel/ignore assertions passed"}, ensure_ascii=False, indent=2))
