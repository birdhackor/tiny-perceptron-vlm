"""W.7 reviewer-owned, offline CPU evidence; never trains or changes course/code."""

import ast
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4.json"
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4/sections/W.7.md"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    path = OUT / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def section(path, identifier):
    raw = (ROOT / path).read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## (\S+) .+$", raw, re.M))
    start = next(h for h in headings if h[1] == identifier)
    end = next((h.start() for h in headings if h.start() > start.start()), len(raw))
    return raw[start.start():end], raw[:start.start()].count("\n") + 1


def process(command, cwd=ROOT, input_text=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(command, cwd=cwd, input=input_text, text=True,
                               capture_output=True, env=env, timeout=30)
    return {"command": command, "cwd": str(cwd), "input": input_text,
            "returncode": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr}


def main():
    import torch

    began = datetime.datetime.now(datetime.UTC).isoformat()
    body = SNAPSHOT.read_text(encoding="utf-8")
    frozen = next(p for p in json.loads(FREEZE.read_text())["pages"] if p["page_id"] == "W.7")
    current, _ = section("course/first-steps.md", "W.7")
    assert sha(SNAPSHOT.read_bytes()) == frozen["source_sha256"]
    assert current == body and frozen["figures_sha256"] == {}
    environment = {"python": sys.version, "torch": str(torch.__version__), "device": "cpu",
                   "platform": platform.platform(), "uv": process(["uv", "--version"])["stdout"].strip(),
                   "head": process(["git", "rev-parse", "HEAD"])["stdout"].strip()}

    # Current route destinations: only paragraphs naming/explaining the component
    # plus code-origin markers and relevant primary definitions. No historical
    # lesson snapshots or other reviewers' reports enter this evidence.
    rows = []
    code_files = {}
    comparisons = []
    for line in body.splitlines():
        if not line.startswith("| `tiny_perceptron/"):
            continue
        path = line.split("`")[1]
        if path == "tiny_perceptron/":
            continue
        row = {"path": path, "source_quote": line, "targets": []}
        for match in re.finditer(r"\[([^]]+)\]\((chapters/[^#]+)#([^)]*)\)(?:[–—](\d+\.\d+))?", line):
            identifiers = [match[3]]
            if match[4]:
                prefix, first = match[3].rsplit(".", 1)
                identifiers += [f"{prefix}.{n}" for n in range(int(first) + 1, int(match[4].rsplit(".", 1)[1]) + 1)]
            for identifier in identifiers:
                target_path = "course/" + match[2]
                target, base = section(target_path, identifier)
                lines = target.splitlines()
                excerpts = [{"start_line": base + i, "end_line": base + i, "text": value}
                            for i, value in enumerate(lines)
                            if path in value or (Path(path).is_file() and Path(path).name in value)]
                # Folder routes also name components through dotted Python imports.
                if not excerpts:
                    dotted = path.rstrip("/").replace("/", ".")
                    excerpts = [{"start_line": base + i, "end_line": base + i, "text": value}
                                for i, value in enumerate(lines) if dotted in value]
                assert excerpts, (path, identifier)
                item = {"lesson_id": identifier, "path": target_path,
                        "heading": lines[0], "section_start_line": base,
                        "section_sha256": sha(target.encode()), "excerpts": excerpts,
                        "primary_symbols": []}
                for block in re.finditer(r"```python\n(.*?)```", target, re.S):
                    marker = re.search(r"# (節錄自|精簡自) (tiny_perceptron/[^：\s]+)：", block[1])
                    if not marker:
                        continue
                    primary_path = marker[2]
                    primary = (ROOT / primary_path).read_text(encoding="utf-8")
                    primary_nodes = {n.name: n for n in ast.parse(primary).body
                                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                    code_files[primary_path] = {"sha256": sha(primary.encode()),
                                                "symbols": [{"name": n.name, "line": n.lineno,
                                                             "end_line": n.end_lineno}
                                                            for n in primary_nodes.values()]}
                    for node in ast.parse(block[1]).body:
                        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            continue
                        main_node = primary_nodes.get(node.name)
                        assert main_node is not None, (primary_path, identifier, node.name)
                        record = {"path": primary_path, "symbol": node.name,
                                  "line": main_node.lineno, "end_line": main_node.end_lineno,
                                  "excerpt_kind": marker[1]}
                        if marker[1] == "節錄自":
                            record["ast_equal"] = ast.dump(node) == ast.dump(main_node)
                            assert record["ast_equal"], record
                            comparisons.append({"lesson_id": identifier, **record})
                        item["primary_symbols"].append(record)
                row["targets"].append(item)
        assert row["targets"], path
        rows.append(row)

    # Primary files discussed in prose instead of an inline definition.
    extra = {
        "tiny_perceptron/simple.py": ["ContextMLP"],
        "tiny_perceptron/capstone.py": ["save_capstone", "load_capstone"],
        "tiny_perceptron/tokenization.py": ["ByteLevelBPE"],
        "tiny_perceptron/natural_concepts.py": ["picture_order_report", "thin_stroke_report", "speech_stages"],
        "tiny_perceptron/selftrained/model.py": ["AudioEncoder", "LimitedAssistant"],
        "tiny_perceptron/selftrained/tokenizer.py": ["CharacterTokenizer"],
        "tiny_perceptron/selftrained/dataset.py": ["read_records", "RecordEncoder"],
        "tiny_perceptron/selftrained/tools.py": ["parse_tool_call", "execute_tool_call"],
        "tiny_perceptron/lesson_products/data.py": ["family", "split_of", "write_dataset"],
        "tiny_perceptron/lesson_products/inference.py": ["rule_depth", "auto_solve"],
        "scripts/train.py": ["parser", "main"],
    }
    for path, names in extra.items():
        code = (ROOT / path).read_text(encoding="utf-8")
        nodes = {n.name: n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert all(name in nodes for name in names)
        code_files[path] = {"sha256": sha(code.encode()),
                            "symbols": [{"name": name, "line": nodes[name].lineno,
                                         "end_line": nodes[name].end_lineno} for name in names]}
    route_result = {"created_at": began, "environment": environment, "source_sha256": sha(body.encode()),
                    "current_source_matches_freeze": True, "rows": rows, "primary_code": code_files,
                    "exact_excerpt_comparisons": comparisons,
                    "summary": {"route_rows": len(rows), "target_sections": sum(len(r["targets"]) for r in rows),
                                "exact_ast_matches": len(comparisons)}}
    save("route-inspection.json", route_result)

    inline = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(inline) == 1
    inline = inline[0]
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(inline)) if isinstance(n, ast.Call)]
    assert set(calls) <= {"print", "Path", "Path.cwd", "Path('pyproject.toml').exists", "Path('tiny_perceptron').exists"}
    direct = process([".venv/bin/python", "-c", inline])
    changed = process(["../.venv/bin/python", "-c", inline], ROOT / "notebooks")
    assert direct["returncode"] == changed["returncode"] == 0
    assert direct["stdout"].splitlines() == [str(ROOT), "True", "True"]
    assert changed["stdout"].splitlines() == [str(ROOT / "notebooks"), "False", "False"]
    interactive = process([".venv/bin/python", "-i"], input_text=inline + "\nexit()\n")
    assert interactive["returncode"] == 0 and ">>>" in interactive["stderr"]
    activated = process(["bash", "--noprofile", "--norc", "-c",
                         "source .venv/bin/activate\npython -c 'from pathlib import Path; print(Path.cwd()); print(Path(\"pyproject.toml\").exists()); print(Path(\"tiny_perceptron\").exists())'\ncd notebooks\npython -c 'from pathlib import Path; print(Path.cwd()); print(Path(\"pyproject.toml\").exists()); print(Path(\"tiny_perceptron\").exists())'\ncd ..\n"])
    assert activated["returncode"] == 0
    assert activated["stdout"].splitlines() == [str(ROOT), "True", "True", str(ROOT / "notebooks"), "False", "False"]

    examples = {
        "ModuleNotFoundError": ([str(ROOT / ".venv/bin/python"), "-I", "-S", "-c", "import torch"], "ModuleNotFoundError"),
        "FileNotFoundError": ([sys.executable, "-c", "from pathlib import Path; Path('__w7_missing_input_20261029__').read_text()"], "FileNotFoundError"),
        "KeyError": ([sys.executable, "-c", "record = {'animal': '貓'}; print(record['caption'])"], "KeyError"),
        "IndexError": ([sys.executable, "-c", "animals = ['貓', '狗']; print(animals[0], animals[1]); print(animals[2])"], "IndexError"),
        "SyntaxError_parenthesis": ([sys.executable, "-c", "print('hello'"], "SyntaxError"),
        "SyntaxError_colon": ([sys.executable, "-c", "if True\n    pass"], "SyntaxError"),
        "SyntaxError_indentation": ([sys.executable, "-c", "if True:\npass"], "IndentationError"),
    }
    errors = {}
    for name, (command, expected) in examples.items():
        result = process(command)
        assert result["returncode"] != 0 and expected in result["stderr"]
        errors[name] = result

    inventory = {}
    for path in ["course/chapters", "notebooks", "tiny_perceptron", "scripts", "data", "checkpoints", "outputs", ".venv"]:
        target = ROOT / path
        inventory[path] = {"exists": target.exists(), "is_directory": target.is_dir()}
    files = ["course/chapters/01.md", "notebooks/01/1.1.ipynb", "scripts/check_env.py", "scripts/prepare_data.py",
             "scripts/train.py", "scripts/evaluate.py", "scripts/build_course.py", "pyproject.toml"]
    assert all((ROOT / p).is_file() for p in files)
    ignored = process(["git", "check-ignore", "--no-index", "data/__w7_probe__", "checkpoints/__w7_probe__", "outputs/__w7_probe__", ".venv/__w7_probe__"])
    assert ignored["returncode"] == 0 and len(ignored["stdout"].splitlines()) == 4

    # Run the real default dry-run with observer patches only. The optimizer
    # implementation is not replaced; observe step calls and compare every
    # trainable parameter against its value at optimizer construction.
    import tiny_perceptron.training as training
    observations = {"optimizer_step_calls": 0, "checkpoint_save_calls": 0}
    stored = {}
    real_init = torch.optim.AdamW.__init__
    real_step = torch.optim.AdamW.step
    real_save = training.save_checkpoint

    def observe_init(self, params, *args, **kwargs):
        params = list(params)
        real_init(self, params, *args, **kwargs)
        stored["parameters"] = params
        stored["before"] = [p.detach().clone() for p in params]

    def observe_step(self, *args, **kwargs):
        observations["optimizer_step_calls"] += 1
        return real_step(self, *args, **kwargs)

    def observe_save(*args, **kwargs):
        observations["checkpoint_save_calls"] += 1
        return real_save(*args, **kwargs)

    capture = io.StringIO()
    previous_argv = sys.argv
    sys.path.insert(0, str(ROOT))
    try:
        sys.argv = ["scripts/train.py", "--task", "text", "--device", "cpu"]
        with patch.object(torch.optim.AdamW, "__init__", observe_init), patch.object(torch.optim.AdamW, "step", observe_step), patch.object(training, "save_checkpoint", observe_save), contextlib.redirect_stdout(capture):
            runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")
    finally:
        sys.argv = previous_argv
    raw_stdout = capture.getvalue()
    report = json.loads(raw_stdout.splitlines()[-1])
    observations.update({"all_parameters_bitwise_unchanged": all(torch.equal(p, b) for p, b in zip(stored["parameters"], stored["before"], strict=True)),
                         "parameter_tensors": len(stored["parameters"]), "report": report,
                         "raw_stdout": raw_stdout, "effective_argv": ["scripts/train.py", "--task", "text", "--device", "cpu"]})
    assert report["mode"] == "dry-run-no-weight-update"
    assert len(report["history"]) == 1 and observations["optimizer_step_calls"] == observations["checkpoint_save_calls"] == 0
    assert observations["all_parameters_bitwise_unchanged"]
    save("cpu-verification.json", {"created_at": began, "environment": environment, "inline_code": inline,
                                   "inline_ast_calls": calls, "project_root": direct, "notebooks_directory": changed,
                                   "interactive": interactive, "activated_shell": activated, "exceptions": errors,
                                   "directory_inventory": inventory, "example_files": files, "git_check_ignore": ignored,
                                   "dry_run": observations,
                                   "limits": ["Linux CPU execution only; Windows/macOS path spellings checked against CPython documentation, not executed here.",
                                              "-I -S isolates package search to reproduce missing torch without changing the installed environment.",
                                              "No --train, no optimizer step, no GPU allocation, no long training.",
                                              "Route checks establish current locations and excerpts, not the general validity of the linked sections' theories or empirical results."]})
    print(json.dumps({"created_at": began, "environment": environment, "routes": route_result["summary"],
                      "pathlib_root": direct["stdout"].splitlines(), "pathlib_notebooks": changed["stdout"].splitlines(),
                      "dry_run": {k: v for k, v in observations.items() if k not in ("report", "raw_stdout")},
                      "result": "all bounded checks passed"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
