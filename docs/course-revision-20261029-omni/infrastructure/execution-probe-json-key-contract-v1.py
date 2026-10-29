"""真 pinned SDK CLI JSON rendering；對照原 commit 的 stopped_app，完全離線。"""

import ast
import importlib.metadata
import json
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import patch


def forbidden(*args, **kwargs):
    raise AssertionError("client or network forbidden")


def main():
    from modal.cli import utils
    from modal.client import _Client
    from rich.table import Column
    from rich.text import Text

    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    _Client.__init__ = forbidden
    captured = []
    with patch.object(utils.OutputManager, "get", lambda: SimpleNamespace(print_json=captured.append)):
        utils.display_table(
            [Column("App ID"), Column("Description"), Column("State"), Column("Tasks")],
            [["ap-S6YMhO8bLq0D4qVHhFKFDM", "omni-probe-40957487215-1", Text("stopped"), "0"]],
            json=True,
        )
    row = json.loads(captured[0])[0]
    assert set(row) == {"app_id", "description", "state", "tasks"}
    raw = subprocess.check_output(
        ["git", "show", "90f9a75611e91bf927c9116b94e025fe8d854947:scripts/omni_revision/preflight.py"], text=True
    )
    node = next(
        value for value in ast.parse(raw).body if isinstance(value, ast.FunctionDef) and value.name == "stopped_app"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<original-stopped-app>", "exec"), namespace)
    original = namespace["stopped_app"](row)
    assert original is False
    assert row["state"] == "stopped" and row["tasks"] == "0"
    print(
        json.dumps(
            {
                "modal_version": importlib.metadata.version("modal"),
                "rendered_row": row,
                "original_stopped_app_result": original,
                "correct_lowercase_interpretation": True,
                "client_constructed": False,
                "network_called": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
