"""Reviewer-owned, offline structural and behavior checks for frozen W.7 only.

No training, final-family inference, package installation, or source mutation.
Prints portable JSON; run from the repository root with the prepared CPU venv.
"""

import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import runpy
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-numbered-source-freeze-v3.json"
page = next(x for x in json.loads((ROOT / FREEZE).read_bytes())["pages"] if x["page_id"] == "W.7")
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
body = raw.decode()


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def section(path, sid):
    text = (ROOT / path).read_text()
    match = re.search(r"^## " + re.escape(sid) + r" [^\n]*\n.*?(?=^## |\Z)", text, re.M | re.S)
    assert match, (path, sid)
    return match.group(), text[: match.start()].count("\n") + 1


def nodes(code):
    return {n.name: n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


mapping = []
chapter_paths = set()
for line in body.splitlines():
    row = re.fullmatch(r"\| `(tiny_perceptron/[^`]+)` \| (.+) \|", line)
    if not row or "chapters/" not in row[2]:
        continue
    package, links = row.groups()
    target = ROOT / package
    assert target.exists(), package
    sources = sorted(p.relative_to(ROOT).as_posix() for p in target.rglob("*.py")) if target.is_dir() else [package]
    source_nodes = {p: nodes((ROOT / p).read_text()) for p in sources}
    entry = {"package": package, "exists": True, "sections": [], "python_sources": []}
    for source, defs in source_nodes.items():
        entry["python_sources"].append({"path": source, "sha256": sha(source), "top_level_definitions": list(defs)})
    for _, chapter, sid, end in re.findall(
        r"\[([^]]+)\]\((chapters/[^)#]+\.md)#([^)]*)\)(?:–([A-Z0-9]+\.[0-9]+))?", links
    ):
        ids = [sid]
        if end:
            prefix, start = sid.rsplit(".", 1)
            ids = [prefix + "." + str(i) for i in range(int(start), int(end.rsplit(".", 1)[1]) + 1)]
        path = "course/" + chapter
        chapter_paths.add(path)
        for sid in ids:
            text, lineno = section(path, sid)
            blocks = re.findall(r"```python\n(.*?)```", text, re.S)
            local_nodes = {}
            imports = []
            for block in blocks:
                parsed = ast.parse(block)
                local_nodes.update(nodes(block))
                imports.extend(ast.unparse(n) for n in parsed.body if isinstance(n, (ast.Import, ast.ImportFrom)))
            evidence = []
            for i, value in enumerate(text.splitlines()):
                if package in value:
                    pos = value.index(package)
                    evidence.append({"line": lineno + i, "excerpt": value[max(0, pos - 140) : pos + 420]})
            shared = []
            for source, defs in source_nodes.items():
                for name in sorted(set(defs) & set(local_nodes)):
                    same = ast.dump(defs[name], include_attributes=False) == ast.dump(local_nodes[name], include_attributes=False)
                    shared.append({"symbol": name, "source": source, "identical_ast": same})
            entry["sections"].append({
                "section": sid, "path": path, "start_line": lineno,
                "section_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "explicit_path_evidence": evidence, "inline_top_level_definitions": list(local_nodes),
                "inline_source_symbol_matches": shared, "imports": imports,
            })
    mapping.append(entry)

# The excerpt-marker inventory is independent of the table's listed links.
# It detects a quoted/simplified package excerpt whose section was omitted.
marker_inventory = []
uncovered_markers = []
for chapter in sorted((ROOT / "course/chapters").glob("*.md")):
    sid = None
    for line_number, line in enumerate(chapter.read_text().splitlines(), 1):
        heading = re.match(r"^## ([A-Z0-9]+\.\d+) ", line)
        if heading:
            sid = heading[1]
        marker = re.search(r"# (?:節錄自|精簡自) (tiny_perceptron/[^：\s]+)", line)
        if marker:
            path = marker[1]
            covered = any(path.startswith(m["package"]) and sid in [s["section"] for s in m["sections"]] for m in mapping)
            item = {"chapter": chapter.relative_to(ROOT).as_posix(), "line": line_number,
                    "section": sid, "package": path, "covered": covered}
            marker_inventory.append(item)
            if not covered:
                uncovered_markers.append(item)
assert not uncovered_markers, uncovered_markers

snippet = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(snippet) == 1
snippet = snippet[0]
path_checks = []
for directory in (ROOT, ROOT / "notebooks"):
    result = subprocess.run([sys.executable, "-c", snippet], cwd=directory, capture_output=True, text=True, check=True)
    expected = [str(directory), "True", "True"] if directory == ROOT else [str(directory), "False", "False"]
    assert result.stdout.splitlines() == expected
    interactive = subprocess.run([str(ROOT / ".venv/bin/python"), "-q", "-i"], input=snippet + "\nexit()\n",
                                 cwd=directory, capture_output=True, text=True, check=True)
    assert interactive.stdout.splitlines() == expected
    path_checks.append({"cwd": str(directory), "command": "python -c <exact frozen Python block>",
                        "stdout": result.stdout, "interactive_command": ".venv/bin/python -q -i (stdin: frozen block; exit())",
                        "interactive_stdout": interactive.stdout, "interactive_prompts": interactive.stderr})

errors = []
for expected, options, code in [
    ("ModuleNotFoundError", ["-I", "-S"], "import torch"),
    ("FileNotFoundError", [], "from pathlib import Path\nPath('w7-deliberately-absent-file').read_text()"),
    ("KeyError", [], "record = {'animal': '貓'}\nrecord['caption']"),
    ("IndexError", [], "animals = ['貓', '狗']\nanimals[2]"),
    ("SyntaxError", [], "if True\n    print('missing colon')"),
]:
    result = subprocess.run([sys.executable, *options, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and result.stderr.splitlines()[-1].startswith(expected + ":")
    errors.append({"expected": expected, "options": options, "code": code, "returncode": result.returncode,
                   "stderr": result.stderr})

animals = ["貓", "狗"]
index_check = {"length": len(animals), "index_0": animals[0], "index_1": animals[1],
               "index_2_requests_ordinal": 3, "index_2_exception": errors[3]["stderr"].splitlines()[-1]}
assert len(animals) == 2 and animals[0] == "貓" and animals[1] == "狗"

directory_inventory = {}
for path in ["course/chapters", "notebooks", "tiny_perceptron", "scripts", "data", "checkpoints", "outputs", ".venv"]:
    p = ROOT / path
    ignored = subprocess.run(["git", "check-ignore", path + "/reviewer-placeholder"], cwd=ROOT,
                             capture_output=True, text=True)
    directory_inventory[path] = {"exists_now": p.is_dir(), "ignored_by_git": ignored.returncode == 0,
                                 "ignore_rule": ignored.stdout.strip()}
directory_inventory["notebooks/01/1.1.ipynb"] = {"exists_now": (ROOT / "notebooks/01/1.1.ipynb").is_file()}
for path in ["scripts/check_env.py", "scripts/prepare_data.py", "scripts/train.py", "scripts/evaluate.py"]:
    directory_inventory[path] = {"exists_now": (ROOT / path).is_file(), "sha256": sha(path)}
assert all(directory_inventory[p]["ignored_by_git"] for p in ["data", "checkpoints", "outputs", ".venv"])

windows_paths = {
    "root_python": str(PureWindowsPath(r".\.venv\Scripts\python.exe")),
    "from_notebooks_python": str(PureWindowsPath(r"..\.venv\Scripts\python.exe")),
    "same_target": PureWindowsPath("C:/project/notebooks/../.venv/Scripts/python.exe").parts,
    "scope": "PureWindowsPath lexical inspection only; no Windows or PowerShell execution.",
}
linux_paths = {"root_python_exists": (ROOT / ".venv/bin/python").is_file(),
               "from_notebooks_same_python": (ROOT / "notebooks/../.venv/bin/python").resolve() == (ROOT / ".venv/bin/python").resolve()}
assert linux_paths["root_python_exists"] and linux_paths["from_notebooks_same_python"]

# uv's actual offline plan confirms the documented default location, without
# creating a venv or resolving/downloading this repository's dependencies.
with tempfile.TemporaryDirectory(prefix="omni-P5-W7-uv-") as temporary:
    temporary_root = Path(temporary)
    (temporary_root / "pyproject.toml").write_text(
        '[project]\nname = "w7-review-venv-probe"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
    )
    probe_env = os.environ.copy()
    probe_env.pop("UV_PROJECT_ENVIRONMENT", None)
    probe_env.pop("VIRTUAL_ENV", None)
    argv = ["uv", "sync", "--dry-run", "--offline", "--no-managed-python", "--python", sys.executable]
    planned = subprocess.run(argv, cwd=temporary_root, env=probe_env, capture_output=True, text=True, check=True)
    assert "Would create project environment at: .venv" in planned.stderr
    assert not (temporary_root / ".venv").exists() and not (temporary_root / "uv.lock").exists()
    uv_plan = {"argv": argv, "cwd": "temporary minimal Python project", "stdout": planned.stdout,
               "stderr": planned.stderr, "created_venv": False, "created_lockfile": False,
               "uv_version": subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True).stdout.strip()}

help_run = subprocess.run([sys.executable, "scripts/train.py", "--help"], cwd=ROOT,
                          capture_output=True, text=True, check=True)
assert "--train" in help_run.stdout and "預設只檢查梯度" in help_run.stdout

# A fresh tiny synthetic dry run, never a fixed final-family inference/training run.
# Count attempted AdamW.step and checkpoint calls without allowing either operation.
import torch
import tiny_perceptron.training as training

step_calls = []
save_calls = []


def forbidden_step(*args, **kwargs):
    step_calls.append(True)
    raise AssertionError("Dry run attempted optimizer.step")


def forbidden_save(*args, **kwargs):
    save_calls.append(True)
    raise AssertionError("Dry run attempted checkpoint saving")


original_step, original_save = torch.optim.AdamW.step, training.save_checkpoint
old_argv = sys.argv
output = io.StringIO()
with tempfile.TemporaryDirectory(prefix="omni-P5-W7-") as temporary:
    checkpoint = Path(temporary) / "must-not-exist.pt"
    argv = ["scripts/train.py", "--task", "text", "--device", "cpu", "--steps", "1", "--batch-size", "1",
            "--width", "8", "--layers", "1", "--heads", "1", "--max-length", "16", "--output", str(checkpoint)]
    try:
        torch.optim.AdamW.step = forbidden_step
        training.save_checkpoint = forbidden_save
        sys.argv = argv
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")
    finally:
        torch.optim.AdamW.step, training.save_checkpoint = original_step, original_save
        sys.argv = old_argv
    printed = output.getvalue()
    report = json.loads(printed.splitlines()[-1])
    assert report["mode"] == "dry-run-no-weight-update" and len(report["history"]) == 1
    assert not step_calls and not save_calls and not checkpoint.exists()
    dry_run = {"argv": argv[:-1] + ["<temporary>/must-not-exist.pt"], "stdout": printed,
               "optimizer_step_attempts": len(step_calls), "checkpoint_save_attempts": len(save_calls),
               "checkpoint_created": checkpoint.exists()}

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_w_7", "scope": "W.7 exact frozen numbered unit only",
    "freeze": {"path": FREEZE, "sha256": sha(FREEZE)},
    "snapshot": {"path": page["snapshot"], "sha256": page["source_sha256"]},
    "environment": {"python": platform.python_version(), "executable": sys.executable,
                    "torch": torch.__version__, "device": "CPU", "platform": platform.platform()},
    "mapping": mapping, "mapping_row_count": len(mapping),
    "mapped_section_reference_count": sum(len(x["sections"]) for x in mapping),
    "chapter_sha256": {p: sha(p) for p in sorted(chapter_paths)},
    "excerpt_marker_inventory": marker_inventory, "uncovered_excerpt_markers": uncovered_markers,
    "frozen_python_block": snippet, "path_checks": path_checks, "exception_checks": errors,
    "index_check": index_check, "directory_inventory": directory_inventory,
    "windows_paths": windows_paths, "linux_paths": linux_paths, "uv_offline_plan": uv_plan,
    "train_help": {"argv": [sys.executable, "scripts/train.py", "--help"], "stdout": help_run.stdout},
    "dry_run": dry_run,
    "limitations": ["No Windows/PowerShell, macOS, GPU, MPS, browser/visual, or Colab execution.",
                    "Static comparison checks where an implementation is taught; it does not re-review every linked lesson's numerical/research conclusions.",
                    "No training, final-family model inference, tuning, Modal, workflow dispatch, publish, or network request."],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
