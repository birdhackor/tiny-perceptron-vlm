"""程式輸出補充工具（docs/course-review-20261019/tools/executed_outputs.py）的離線測試。

驗證：`show` 只印讀者游標已到的單元裡的程式輸出，程式段落與輸出按順序對應；折疊選讀裡的輸出要整頁
讀完並補讀過選讀才印；不是自己的 session 不能看；`build` 拒收和課文程式對不上的 Notebook。
不涵蓋：真正的 Notebook 執行（測試用寫好的輸出）。
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load("executed_outputs", ROOT / "docs/course-review-20261019/tools/executed_outputs.py")
READER = load("reader_tool", ROOT / "docs/course-revision-20261016/tools/reader.py")

BODY = """## 9.1 範例

第一段說明。

```python
print("甲")
```

中間的說明。

```python
print("乙")
```

<details><summary>選讀</summary>

```python
print("丙")
```

</details>

最後的說明。
"""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).encode()
    path.write_bytes(raw)
    return {"path": relative, "sha256": sha(raw)}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(TOOL, "ROOT", tmp_path)
    monkeypatch.setattr(TOOL, "BATCH", tmp_path / "batch")
    monkeypatch.setattr(TOOL, "RECEIPTS", tmp_path / "receipts")
    monkeypatch.setattr(TOOL, "reader_units", READER.units)
    snapshot = save(tmp_path, "freeze/sources/9.1.md", BODY)
    page = {"page_id": "9.1", "snapshot": snapshot["path"], "source_sha256": snapshot["sha256"], "figures_sha256": {}}
    freeze = save(tmp_path, "freeze.json", {"pages": [page]})
    blocks = [
        {
            "index": i,
            "in_details": i == 2,
            "code_sha256": sha(code.encode()),
            "outputs": [{"kind": "stdout", "text": text}],
        }
        for i, (code, text) in enumerate([('print("甲")', "甲\n"), ('print("乙")', "乙\n"), ('print("丙")', "丙\n")])
    ]
    bundle = save(
        tmp_path,
        "bundle.json",
        {"source_freeze": freeze, "pages": {"9.1": {"source_sha256": snapshot["sha256"], "blocks": blocks}}},
    )
    plan = save(tmp_path, "plan.json", {"reviewer_task": "/root/reader", "executed_outputs": bundle})
    state = {
        "session": "a" * 32,
        "page_id": "9.1",
        "source_sha256": snapshot["sha256"],
        "reviewer_task": "/root/reader",
        "next_index": 0,
        "reading_plan": plan,
        "source_freeze": freeze,
    }
    units = READER.units(BODY)
    return tmp_path, state, plan, units


def show(tmp_path, state, capsys, reviewer="/root/reader"):
    save(tmp_path, f"batch/reader-sessions/{state['session']}/state.json", state)
    TOOL.show(type("Args", (), {"session": state["session"], "reviewer": reviewer})())
    return capsys.readouterr().out


def unit_with(units, text):
    return next(i for i, unit in enumerate(units) if text in unit)


def test_only_outputs_of_units_already_reached_are_shown(world, capsys):
    tmp_path, state, _, units = world
    first, second = unit_with(units, '"甲"'), unit_with(units, '"乙"')
    assert first < second
    state["next_index"] = first - 1
    assert "甲" not in show(tmp_path, state, capsys)
    state["next_index"] = first
    shown = show(tmp_path, state, capsys)
    assert "甲\n" in shown and "乙\n" not in shown
    state["next_index"] = second
    shown = show(tmp_path, state, capsys)
    assert shown.index("甲\n") < shown.index("乙\n")
    receipts = sorted((tmp_path / "receipts").glob("*.json"))
    assert len(receipts) == 3


def test_folded_outputs_need_a_finished_page_and_a_supplement_read(world, capsys):
    tmp_path, state, plan, units = world
    state["next_index"] = len(units)
    assert "丙\n" not in show(tmp_path, state, capsys)
    save(
        tmp_path,
        "batch/supplement-access/x.json",
        {"reviewer_task": "/root/reader", "reading_plan": plan, "page_id": "9.1"},
    )
    assert "丙\n" in show(tmp_path, state, capsys)


def test_another_reviewers_session_is_refused(world, capsys):
    tmp_path, state, _, _ = world
    with pytest.raises(ValueError):
        show(tmp_path, state, capsys, reviewer="/root/someone_else")


def test_build_refuses_a_notebook_that_does_not_match_the_text(world, monkeypatch):
    tmp_path, _, _, _ = world
    setup = {"cell_type": "code", "metadata": {"course_setup": True}, "source": ["x = 1"], "outputs": []}
    cells = [setup] + [
        {"cell_type": "code", "metadata": {}, "source": [code], "outputs": [], "execution_count": 1}
        for code in ('print("甲")', 'print("乙")', 'print("丁")')
    ]
    notebook = {"cells": cells}
    save(tmp_path, "notebooks/09/9.1.ipynb", notebook)
    save(tmp_path, "executed/09/9.1.ipynb", notebook)
    save(tmp_path, "course/lesson-index.json", [{"id": "9.1", "notebook": "notebooks/09/9.1.ipynb"}])
    arguments = {
        "freeze": Path("freeze.json"),
        "pages": ["9.1"],
        "executed": tmp_path / "executed",
        "executed_with": "test",
        "output": Path("out.json"),
    }
    with pytest.raises(ValueError, match="對不上"):
        TOOL.build(type("Args", (), arguments)())
    cells[-1]["source"] = ['print("丙")']
    save(tmp_path, "notebooks/09/9.1.ipynb", notebook)
    save(tmp_path, "executed/09/9.1.ipynb", notebook)
    TOOL.build(type("Args", (), arguments)())
    built = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert [block["in_details"] for block in built["pages"]["9.1"]["blocks"]] == [False, False, True]
