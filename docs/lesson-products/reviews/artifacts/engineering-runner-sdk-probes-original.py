"""只讀 Modal1.6 SDK，使用真正exception型別的離線terminal反例。"""

import functools
import hashlib
import inspect
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal
import modal._functions as functions
from modal._utils.function_utils import _process_result

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.lesson_products.modal_runner import confirm_stopped


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    results = []
    terminal = (modal.exception.FunctionTimeoutError, modal.exception.RemoteError, modal.exception.OutputExpiredError, ValueError)
    for name, error, expected in (
        ("completed", None, "confirmed"),
        ("poll_timeout", modal.exception.TimeoutError("poll only"), "propagated"),
        ("runtime_timeout", modal.exception.FunctionTimeoutError("function timeout"), "confirmed"),
        ("remote_terminal", modal.exception.RemoteError("terminal result"), "confirmed"),
        ("expired_terminal", modal.exception.OutputExpiredError(), "confirmed"),
    ):
        events = []

        class Call:
            def cancel(self, terminate_containers):
                events.append(["cancel", terminate_containers])

            def get(self, timeout):
                events.append(["get", timeout])
                if error:
                    raise error
                return {}

        try:
            confirm_stopped(Call(), terminal)
        except Exception as caught:
            actual, error_type = "propagated", type(caught).__module__ + "." + type(caught).__name__
        else:
            actual, error_type = "confirmed", None
        assert actual == expected
        results.append({"id": name, "actual": actual, "error_type": error_type, "events": events})
    methods = []
    for name, obj in (("FunctionCall.get", modal.FunctionCall.get), ("FunctionCall.cancel", modal.FunctionCall.cancel), ("Function.with_options", modal.Function.with_options), ("Invocation.poll_function", functions._Invocation.poll_function), ("_process_result", _process_result)):
        target = inspect.unwrap(obj)
        while isinstance(target, functools.partial):
            target = inspect.unwrap(target.func)
        source = inspect.getsource(obj)
        file = inspect.getsourcefile(target)
        methods.append({"object": name, "source_file": file, "file_sha256": digest(file), "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
    report = {
        "reviewer_task": "/root/tech_20261016_review_tools", "created_at": datetime.now(UTC).isoformat(), "sdk_version": modal.__version__,
        "environment": {"python": sys.version, "platform": platform.platform()}, "inspected_methods": methods,
        "api_signatures": {name: str(inspect.signature(obj)) for name, obj in (("Image.uv_sync", modal.Image.uv_sync), ("Image.add_local_dir", modal.Image.add_local_dir), ("Function.with_options", modal.Function.with_options))},
        "terminal_probes": results,
        "observations": ["cancel sends FunctionCallCancelRequest; it does not poll result", "poll_function with no outputs and unfinished inputs raises modal.exception.TimeoutError; no unfinished inputs raises OutputExpiredError", "get waits indefinitely by default; get(timeout=30) is bounded poll, not container runtime", "with_options supports timeout; dynamic configurations use distinct autoscaling pools"],
        "limits": ["Only SDK source and fake calls; no authenticated Modal/HF API, no GPU. No third-party source copied into Git."],
    }
    path = Path(__file__).with_name("engineering-runner-sdk-inspection-original.json")
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"sdk_version": modal.__version__, "terminal_probes": len(results), "all_matched": True, "receipt_sha256": digest(path)}))


if __name__ == "__main__":
    main()
