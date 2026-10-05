"""Bounded, independent checks of the original W.2 fence and inline examples."""
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
RAW = (BASE / "original-execution/fence-1.py").read_bytes()
assert hashlib.sha256(RAW).hexdigest() == "8aa7677c129754bc9d3c104748bc9f4e6f76d9f5866e11cf01cd30828f436d8b"
CODE = RAW.decode("utf-8")
RESULTS = {}

def run(label, code, expected):
    namespace = {}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, label, "exec"), namespace)
    actual = capture.getvalue()
    assert actual == expected, (label, actual, expected)
    (BASE / f"{label}.py").write_text(code, encoding="utf-8")
    (BASE / f"{label}.stdout.txt").write_text(actual, encoding="utf-8")
    RESULTS[label] = {"stdout": actual, "expected": expected, "matched": True,
                      "animals": namespace["animals"], "record": namespace["record"]}
    return namespace

original = run("plain-original", CODE, "貓 2\n貓在睡覺\n看到 貓\n看到 狗\n")
assert original["animals"][0] == "貓" and original["animals"][1] == "狗"
assert isinstance(original["animals"], list) and isinstance(original["record"], dict)
bird_code = CODE.replace('animals = ["貓", "狗"]', 'animals = ["鳥", "貓", "狗"]')
bird_code = bird_code.replace("assert len(animals) == 2", "assert len(animals) == 3")
bird = run("exercise-three-animals", bird_code, "鳥 3\n貓在睡覺\n看到 鳥\n看到 貓\n看到 狗\n")
caption_code = bird_code.replace('"caption": "貓在睡覺"', '"caption": "貓在玩耍"')
caption = run("exercise-caption-only", caption_code, "鳥 3\n貓在玩耍\n看到 鳥\n看到 貓\n看到 狗\n")
assert bird["animals"] == caption["animals"]
assert RESULTS["exercise-three-animals"]["stdout"].splitlines()[0:1] + RESULTS["exercise-three-animals"]["stdout"].splitlines()[2:] == RESULTS["exercise-caption-only"]["stdout"].splitlines()[0:1] + RESULTS["exercise-caption-only"]["stdout"].splitlines()[2:]
RESULTS["counts"] = {"original_item_count": len(original["animals"]),
                     "original_loop_line_count": sum(s.startswith("看到 ") for s in RESULTS["plain-original"]["stdout"].splitlines()),
                     "exercise_item_count": len(bird["animals"]),
                     "exercise_loop_line_count": sum(s.startswith("看到 ") for s in RESULTS["exercise-three-animals"]["stdout"].splitlines()),
                     "units": "list items and printed loop lines; no tensor axes or statistical denominator"}

inline_code = 'def first(items):\n    return items[0]\n'
(BASE / "inline-first.py").write_text(inline_code)
inline_namespace = {}
exec(inline_code, inline_namespace)
assert inline_namespace["first"](original["animals"]) == "貓"
assert inline_namespace["first"](bird["animals"]) == "鳥"
RESULTS["inline_function"] = {"original_return": "貓", "changed_return": "鳥", "definition_stdout": ""}

animals = original["animals"]
another_name = animals
assert another_name is animals
numbers_and_strings = {1: "貓", "caption": 3}
assert numbers_and_strings[1] == "貓" and numbers_and_strings["caption"] == 3
duplicate = {"caption": "舊", "caption": "新"}
assert len(duplicate) == 1 and duplicate["caption"] == "新"
RESULTS["container_contract"] = {"assignment_binds_same_object": another_name is animals,
                                  "numeric_and_text_keys_values": [[1, "貓"], ["caption", 3]],
                                  "duplicate_final_key_count": len(duplicate),
                                  "duplicate_retained_value": duplicate["caption"]}

false_assert = 'assert len(["貓", "狗"]) == 3\nprint("continued")\n'
(BASE / "false-assert.py").write_text(false_assert)
normal = subprocess.run([sys.executable, str(BASE / "false-assert.py")], capture_output=True, text=True, timeout=10)
optimized = subprocess.run([sys.executable, "-O", str(BASE / "false-assert.py")], capture_output=True, text=True, timeout=10)
assert normal.returncode != 0 and "AssertionError" in normal.stderr and normal.stdout == ""
assert optimized.returncode == 0 and optimized.stdout == "continued\n"
(BASE / "false-assert.stderr.txt").write_text(normal.stderr)
RESULTS["assert_contract"] = {"normal_returncode": normal.returncode, "normal_stdout": normal.stdout,
                              "normal_error": "AssertionError", "normal_optimize_flag": sys.flags.optimize,
                              "optimized_returncode": optimized.returncode, "optimized_stdout": optimized.stdout,
                              "equal_comparison": len(animals) == 2, "unequal_comparison": len(animals) == 3}
assert sys.flags.optimize == 0
invalid_code = 'for animal in ["貓", "狗"]:\nprint(animal)\n'
try:
    compile(invalid_code, "missing-indentation", "exec")
except IndentationError:
    RESULTS["block_contract"] = {"unindented_body_error": "IndentationError", "four_space_original_success": True}
else:
    raise AssertionError("unindented loop body unexpectedly compiled")

import torch
assert torch.version.cuda is None and not torch.cuda.is_available()
assert torch.tensor([1]).device.type == "cpu"
RESULTS["import_contract"] = {"torch": str(torch.__version__), "module_bound": torch.__name__ == "torch",
                              "tensor_api_available": callable(torch.tensor), "device": "cpu"}

from nbclient import NotebookClient
import nbformat
nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(CODE),
    nbformat.v4.new_code_cell('import json, sys\nprint(json.dumps({"python": sys.version, "executable": sys.executable, "optimize": sys.flags.optimize}))')])
nb = NotebookClient(nb, timeout=30, kernel_name="python3").execute()
output = "".join(o.get("text", "") for o in nb.cells[0].outputs if o.output_type == "stream")
assert output == RESULTS["plain-original"]["stdout"]
kernel_text = "".join(o.get("text", "") for o in nb.cells[1].outputs if o.output_type == "stream")
kernel_environment = json.loads(kernel_text)
assert Path(kernel_environment["executable"]).samefile(sys.executable)
assert kernel_environment["optimize"] == 0
nbformat.write(nb, BASE / "executed-small-notebook.ipynb")
tracker = json.loads((BASE / "inputs/installed-jupyterlab-tracker.json").read_text())
shortcut = tracker["jupyter.lab.shortcuts"][47]
assert shortcut["command"] == "notebook:run-cell-and-select-next" and shortcut["keys"] == ["Shift Enter"]
RESULTS["notebook_contract"] = {"cell_output": output, "execution_count": nb.cells[0].execution_count,
                                "shortcut_pointer": "/jupyter.lab.shortcuts/47", "shortcut": shortcut,
                                "kernel_environment": kernel_environment,
                                "scope": "Local kernel execution and default shortcut schema; browser keypress and Colab service were not exercised."}
environment = {"python": sys.version, "executable": sys.executable, "device": "CPU",
               "sys_flags_optimize": sys.flags.optimize,
               "packages": {name: importlib.metadata.version(name) for name in ["torch", "nbclient", "nbformat", "jupyterlab", "ipykernel"]}}
(BASE / "variation-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
(BASE / "variation-results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"environment": environment, "checks": RESULTS}, ensure_ascii=False, indent=2))
