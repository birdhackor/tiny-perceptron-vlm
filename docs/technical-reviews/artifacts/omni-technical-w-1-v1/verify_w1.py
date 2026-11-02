"""W.1 reviewer execution and dependency checks; no model training or source edits."""

import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
SCRATCH = Path("/workspace/work/omni-technical-w-1-v1")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/assembly-reader-01/dependency-w-numbered-freeze.json"
SCRATCH.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from scripts.build_course import BOOTSTRAP, notebook
from scripts.export_course import checked_notebook, output_html, section_footer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, *, cwd=ROOT, env=None, timeout=90):
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    record = {
        "command": shlex.join(command),
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if env is not None:
        record["task_specific_overrides"] = {
            key: value for key, value in env.items()
            if key in {"UV_PROJECT_ENVIRONMENT", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "GIT_LFS_SKIP_SMUDGE"}
        }
    assert completed.returncode == 0, record
    return record


result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_w_1"}
result["environment"] = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "platform": platform.platform(),
    **{p: importlib.metadata.version(p) for p in ("torch", "jupyterlab", "ipykernel", "nbclient", "jupyter-server")},
}
import torch

result["environment"].update(device="cpu", torch_cuda_build=str(torch.version.cuda), cuda_available=str(torch.cuda.is_available()))
assert torch.version.cuda is None
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "W.1")
snapshot = ROOT / page["snapshot"]
body = snapshot.read_text(encoding="utf-8")
current = (ROOT / page["source"]).read_text(encoding="utf-8")
current_slice = re.search(r"^## W\.1 .*?(?=^## |\Z)", current, re.M | re.S)[0]
assert digest(snapshot) == page["source_sha256"]
assert current_slice.encode() == snapshot.read_bytes()
assert page["figures_sha256"] == {}
result["freeze"] = {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE), "snapshot": page["snapshot"], "source_sha256": digest(snapshot), "current_slice_identical": True, "figures": {}}
result["w1_fenced_commands"] = re.findall(r"```(bash|powershell)\n(.*?)```", body, re.S)
result["source_dependencies"] = {
    path: digest(ROOT / path) for path in (
        "scripts/build_course.py", "scripts/export_course.py", "scripts/check_env.py", "scripts/check_notebooks.py",
        "pyproject.toml", "uv.lock", "notebooks/01/1.1.ipynb", "course/chapters/01.md", ".github/workflows/pages.yml", ".gitattributes"
    )
}
result["cli_versions"] = [run(["uv", "--version"]), run(["git", "--version"])]

# Execute the exact uv-sync tokens from W.1, isolating its environment so it cannot
# remove packages used by other concurrent reviewers in the shared .venv.
sync_command = next(code for language, code in result["w1_fenced_commands"] if "uv sync" in code).strip().splitlines()[-1]
sync_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(SCRATCH / "venv")}
result["uv_sync"] = run(shlex.split(sync_command), env=sync_env, timeout=120)
isolated_python = SCRATCH / "venv/bin/python"
result["isolated_cpu_versions"] = run([str(isolated_python), "-c", 'import sys, torch, importlib.metadata as m; print(sys.version); print(torch.__version__, torch.version.cuda); print(m.version("jupyterlab")); assert torch.version.cuda is None'])
result["check_env"] = run([".venv/bin/python", "scripts/check_env.py"])
assert "compute         OK（cpu）" in result["check_env"]["stdout"]
assert "未安裝" not in result["check_env"]["stdout"]
result["activation"] = run(["bash", "-c", 'source .venv/bin/activate\npython -c \'import sys; print(sys.executable)\''])
assert result["activation"]["stdout"].strip() == str(ROOT / ".venv/bin/python")
result["kernel_install"] = run([".venv/bin/python", "-m", "ipykernel", "install", "--sys-prefix", "--name", "tiny-perceptron", "--display-name", "Tiny Perceptron"])
kernel_path = ROOT / ".venv/share/jupyter/kernels/tiny-perceptron/kernel.json"
kernel = json.loads(kernel_path.read_bytes())
assert kernel["display_name"] == "Tiny Perceptron"
assert Path(kernel["argv"][0]) == ROOT / ".venv/bin/python"
result["registered_kernel"] = kernel
result["jupyter_directory_parse"] = run([".venv/bin/python", "-m", "jupyterlab", "notebooks", "--show-config-json"], timeout=60)
lab_config = json.loads(result["jupyter_directory_parse"]["stdout"])
assert lab_config["ServerApp"]["root_dir"] == str(ROOT / "notebooks")

# W.1's exercise uses the actual inline 1.1 block and current generated Notebook.
chapter = (ROOT / "course/chapters/01.md").read_text(encoding="utf-8")
section11 = re.search(r"^## 1\.1 .*?(?=^## |\Z)", chapter, re.M | re.S)[0]
inline = re.search(r"```python\n(.*?)```", section11, re.S)[1].strip()
nb = json.loads((ROOT / "notebooks/01/1.1.ipynb").read_bytes())
setup = next(c for c in nb["cells"] if c.get("metadata", {}).get("course_setup"))
actual_code = next(c for c in nb["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {}))
assert "".join(setup["source"]) == BOOTSTRAP.rstrip("\n")
assert "".join(actual_code["source"]).strip() == inline
result["exercise"] = {"inline_code": inline, "notebook_cell_identical": True, "runs": []}
for text in ("貓看狗，狗看貓。", "鳥看狗，狗看鳥。"):
    code = inline.replace('text = "貓看狗，狗看貓。"', f'text = "{text}"')
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "course/chapters/01.md#1.1:actual-inline", "exec"), namespace)
    assert namespace["restored"] == text
    result["exercise"]["runs"].append({"input": text, "stdout": output.getvalue(), "restored": namespace["restored"]})

# Independent new kernel, launched with cwd at the actual local Notebook folder.
import nbformat
from nbclient import NotebookClient

probe = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_code_cell("".join(setup["source"])),
    nbformat.v4.new_code_cell(inline.replace("貓看狗，狗看貓。", "鳥看狗，狗看鳥。")),
    nbformat.v4.new_code_cell('print(root); print(torch.empty(0).device); remembered = "old"'),
    nbformat.v4.new_code_cell('print(remembered); print(restored)'),
])
NotebookClient(probe, timeout=45, kernel_name="tiny-perceptron", resources={"metadata": {"path": str(ROOT / "notebooks/01")}}).execute()
probe_records = []
for cell in probe.cells:
    probe_records.append({"source_sha256": hashlib.sha256(cell.source.encode()).hexdigest(), "execution_count": cell.execution_count, "text_outputs": [o.get("text", "") for o in cell.outputs if o.output_type == "stream"]})
assert "鳥看狗，狗看鳥。" in "".join(probe_records[1]["text_outputs"])
assert str(ROOT) in "".join(probe_records[2]["text_outputs"])
assert "cpu" in "".join(probe_records[2]["text_outputs"])
assert "old" in "".join(probe_records[3]["text_outputs"])
new_kernel = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print("remembered" in globals())')])
NotebookClient(new_kernel, timeout=30, kernel_name="tiny-perceptron", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
assert new_kernel.cells[0].outputs[0].text.strip() == "False"
result["fresh_kernel_exercise"] = {"cells": probe_records, "separate_kernel_previous_state_present": False, "notebook_cwd": str(ROOT / "notebooks/01")}

# Every indexed lesson has a Notebook, and an exact-source executed copy exists
# in these raw kernel-run directories. This is an independent byte/schema/output
# check, not fresh technical review of any other lesson's subject matter.
index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
executed_roots = [ROOT / p for p in (
    "outputs/omni-notebooks-review-v1", "outputs/omni-notebooks-reader-fix01",
    "outputs/omni-notebooks-reader-fix01-20.16", "outputs/omni-notebooks-reader-fix04-runs"
)]
receipts = []
no_match = []
for item in index:
    original_path = ROOT / item["notebook"]
    original = json.loads(original_path.read_bytes())
    matches = []
    for execution_root in executed_roots:
        execution_path = execution_root / Path(item["notebook"]).relative_to("notebooks")
        if not execution_path.exists():
            continue
        try:
            executed = checked_notebook(original, execution_path)
        except ValueError:
            continue
        matches.append((execution_path, executed))
    if not matches:
        no_match.append(item["id"])
        continue
    execution_path, executed = matches[-1]
    code_cells = [c for c in executed["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
    output_count = sum(bool(output_html(c.get("outputs", []))) for c in code_cells)
    footer = section_footer(item, original, "W1-review", True)
    assert f'href="{item["notebook"]}" download' in footer
    assert f'/blob/W1-review/{item["notebook"]}' in footer
    assert "下載本節 .ipynb" in footer and "在 Colab 開啟" in footer
    if code_cells:
        assert "CPU 實際輸出" in footer
        assert "notebooks/" in footer
        assert "雲端環境未經本課實測" in footer
    else:
        assert "本節沒有要在 Notebook 執行的 Python 程式" in footer
        assert "終端機" in footer
        assert "CPU 實際輸出" not in footer
    copied = ROOT / "outputs/omni-review-site-current" / item["notebook"]
    copied_identical = copied.exists() and copied.read_bytes() == original_path.read_bytes()
    receipts.append({"lesson_id": item["id"], "notebook": item["notebook"], "notebook_sha256": digest(original_path), "executed_copy": str(execution_path.relative_to(ROOT)), "executed_sha256": digest(execution_path), "exact_cells_and_types": True, "all_code_executed_without_error": True, "teaching_code_cells": len(code_cells), "code_cells_with_displayed_output": output_count, "site_download_identical": copied_identical})
result["cpu_output_dependency"] = {"indexed_notebooks": len(index), "matched_executed_notebooks": len(receipts), "unmatched_lesson_ids": no_match, "total_teaching_code_cells": sum(x["teaching_code_cells"] for x in receipts), "cells_with_displayed_output": sum(x["code_cells_with_displayed_output"] for x in receipts), "site_downloads_identical": sum(x["site_download_identical"] for x in receipts), "scope": "Existing raw execution copies checked against current generated source; this reviewer independently ran 1.1 only and did not technically review W.2-W.7 or every chapter."}
assert not no_match, no_match
result["footer_examples"] = {kind: section_footer(item, nb_dict, "W1-review", flag) for kind, item, nb_dict, flag in (
    ("python_with_outputs", index[0], nb, True),
    ("python_without_outputs", index[0], nb, False),
    ("terminal_only", {"notebook": "notebooks/example.ipynb"}, notebook("X.1", "terminal", "```bash\npython scripts/check_env.py\n```", ROOT / "course/first-steps.md"), True),
    ("prose_only", {"notebook": "notebooks/example.ipynb"}, notebook("X.1", "prose", "Plain text", ROOT / "course/first-steps.md"), True),
)}
assert "CPU 實際輸出" not in result["footer_examples"]["python_without_outputs"]
assert "本節沒有要在 Notebook 執行的 Python 程式" in result["footer_examples"]["terminal_only"]
assert "本節沒有要在 Notebook 執行的 Python 程式" in result["footer_examples"]["prose_only"]

# Ensure the export gate actually rejects mismatched or unexecuted results.
bad_path = SCRATCH / "bad-executed.ipynb"
mutations = []
negative_original = json.loads((ROOT / receipts[0]["notebook"]).read_bytes())
negative_executed = json.loads((ROOT / receipts[0]["executed_copy"]).read_bytes())
for reason in ("changed_source", "unexecuted", "error_output"):
    changed = copy.deepcopy(negative_executed)
    cell = next(c for c in changed["cells"] if c["cell_type"] == "code")
    if reason == "changed_source":
        cell["source"] = ["print('changed')"]
    elif reason == "unexecuted":
        cell["execution_count"] = None
    else:
        cell["outputs"] = [{"output_type": "error", "ename": "Synthetic", "evalue": "test", "traceback": []}]
    bad_path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        checked_notebook(negative_original, bad_path)
    except ValueError as error:
        mutations.append({"case": reason, "rejected": True, "error": str(error)})
    else:
        raise AssertionError(reason)
result["exporter_negative_checks"] = mutations

# A concise manifest binds raw outputs without copying notebooks or HTML into Git.
(ARTIFACTS / "cpu-output-dependency-manifest.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
result["limitations"] = ["No Windows or macOS execution; platform syntax and execution-policy claims checked against original official documentation.", "No Colab runtime, cloud account or live paid service execution.", "No full fresh clone of the large remote repository; Git clone/LFS behavior checked in official docs and current repository inputs.", "Historical raw notebook execution copies matched structurally; no claim that every lesson was freshly executed by this reviewer.", "No model-quality, GPU, throughput or training conclusions."]
result["completed_at"] = datetime.now(UTC).isoformat()
(ARTIFACTS / "execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"result": "completed", "cpu_output_dependency": result["cpu_output_dependency"], "exercise_outputs": result["exercise"]["runs"], "environment": result["environment"]}, ensure_ascii=False, indent=2))
