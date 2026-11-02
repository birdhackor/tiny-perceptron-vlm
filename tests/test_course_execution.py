"""文字更新重用成功輸出；程式、資料與環境變動必須重跑。"""

import copy
import hashlib
import json
import subprocess

import pytest

from scripts import ci_scope, course_execution
from scripts.export_course import checked_notebook


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def course(tmp_path, monkeypatch):
    # 一般 CPU 單元測試不安裝可選的 Notebook 群組；這裡只測版本如何參與快取鍵。
    monkeypatch.setattr(course_execution, "version", lambda _: "test-1.0")
    notebook = {
        "metadata": {"kernelspec": {"name": "tiny-perceptron"}},
        "cells": [
            {"cell_type": "markdown", "source": ["舊說明"], "metadata": {}},
            {"cell_type": "code", "source": ["print(1)"], "metadata": {}, "execution_count": None, "outputs": []},
        ],
    }
    source = tmp_path / "notebooks/01/1.1.ipynb"
    write_json(source, notebook)
    output = tmp_path / "outputs/notebooks"
    executed = copy.deepcopy(notebook)
    executed["cells"][1].update(execution_count=1, outputs=[{"output_type": "stream", "text": "1\n"}])
    write_json(output / "01/1.1.ipynb", executed)
    write_json(output / "validation-kernel.json", {"total": 1, "passed": 1, "failed": 0})
    write_json(output / "execution.json", {"key": course_execution.execution_key(tmp_path), "executed_revision": "old"})
    return tmp_path, source, notebook, output


def test_text_and_markdown_cell_reordering_reuse_outputs_with_current_text(course, monkeypatch):
    root, source, notebook, output = course
    key = course_execution.execution_key(root)
    notebook["cells"][0]["source"] = ["新說明"]
    notebook["cells"].append(notebook["cells"].pop(0))
    write_json(source, notebook)
    assert course_execution.execution_key(root) == key
    monkeypatch.setattr(course_execution.subprocess, "run", lambda *a, **k: pytest.fail("文字不應重跑"))
    receipt = course_execution.ensure(root, output, revision="new")
    assert receipt["executed_revision"] == "old"
    checked = checked_notebook(notebook, output / "01/1.1.ipynb")
    assert checked["cells"][-1]["source"] == ["新說明"]
    assert checked["cells"][0]["outputs"][0]["text"] == "1\n"


@pytest.mark.parametrize("change", ["code", "kernel", "script", "data", "dependency", "threads", "runtime-package"])
def test_execution_inputs_invalidate_cache(course, monkeypatch, change):
    root, source, notebook, output = course
    before = course_execution.execution_key(root)
    if change in {"code", "kernel"}:
        if change == "code":
            notebook["cells"][1]["source"] = ["print(2)"]
        else:
            notebook["metadata"]["kernelspec"]["name"] = "other-kernel"
        write_json(source, notebook)
    elif change == "threads":
        monkeypatch.setenv("OMP_NUM_THREADS", "999")
    elif change == "runtime-package":
        monkeypatch.setattr(course_execution, "version", lambda _: "test-2.0")
    else:
        path = (
            root / {"script": "scripts/helper.py", "data": "docs/experiment/data.json", "dependency": "uv.lock"}[change]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed")
    assert course_execution.execution_key(root) != before
    with pytest.raises(ValueError):
        course_execution.validate_bundle(root, output, course_execution.execution_key(root))


@pytest.mark.parametrize("damage", ["missing", "error", "partial", "receipt"])
def test_damaged_cache_is_never_reused_and_failed_rerun_has_no_success_receipt(course, monkeypatch, damage):
    root, _, _, output = course
    path = output / "01/1.1.ipynb"
    if damage == "missing":
        path.unlink()
    elif damage == "error":
        notebook = json.loads(path.read_bytes())
        notebook["cells"][1]["outputs"] = [{"output_type": "error"}]
        write_json(path, notebook)
    elif damage == "partial":
        write_json(output / "validation-kernel.json", {"total": 1, "passed": 0, "failed": 1})
    else:
        (output / "execution.json").write_text("corrupt")

    def failed_run(command, **kwargs):
        assert "check_notebooks.py" in command[1]
        assert not (output / "execution.json").exists()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(course_execution.subprocess, "run", failed_run)
    with pytest.raises(subprocess.CalledProcessError):
        course_execution.ensure(root, output)
    assert not (output / "execution.json").exists()


def test_successful_cache_miss_executes_and_then_reuses(course, monkeypatch):
    root, _, notebook, output = course
    (output / "execution.json").unlink()
    calls = []

    def successful_run(command, **kwargs):
        calls.append(command)
        notebook["cells"][1].update(execution_count=1, outputs=[])
        write_json(output / "01/1.1.ipynb", notebook)
        write_json(output / "validation-kernel.json", {"total": 1, "passed": 1, "failed": 0})

    monkeypatch.setattr(course_execution.subprocess, "run", successful_run)
    assert course_execution.ensure(root, output, "current")["executed_revision"] == "current"
    assert course_execution.ensure(root, output, "next")["executed_revision"] == "current"
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["README.md", "course/chapters/01.md", "docs/publishing.md"])
def test_text_changes_need_linux_only(name):
    assert not ci_scope.runtime_changed([name], lambda _: False)


@pytest.mark.parametrize(
    "name",
    [
        "scripts/helper.py",
        "uv.lock",
        ".python-version",
        "docs/data/a.txt",
        "course/figures/a.svg",
        ".github/workflows/ci.yml",
    ],
)
def test_runtime_changes_need_all_platforms(name):
    assert ci_scope.runtime_changed([name], lambda _: False)


def test_notebook_scope_uses_code_instead_of_markdown():
    assert not ci_scope.runtime_changed(["notebooks/01/1.1.ipynb"], lambda _: False)
    assert ci_scope.runtime_changed(["notebooks/01/1.1.ipynb"], lambda _: True)


def test_missing_diff_base_conservatively_runs_all_platforms(tmp_path, monkeypatch):
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("BASE_SHA", "missing")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(ci_scope.subprocess, "check_output", unavailable)
    ci_scope.main()
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert len(json.loads(values["matrix"])) == 3
    assert values["full"] == "true"


def test_optional_research_download_does_not_change_lesson_execution_key(course):
    root, _, _, _ = course
    name = "docs/natural-assistant/v4/data/fixture.json"
    raw = b'{"research":"fixed"}'
    write_json(
        root / "assets/research/catalog.json",
        {"groups": {"natural-v4": {"files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}}}},
    )
    before = course_execution.execution_key(root)
    target = root / name
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    assert course_execution.execution_key(root) == before
    target.write_bytes(b"modified research input")
    with pytest.raises(ValueError, match="固定研究輸入已變動"):
        course_execution.execution_key(root)
