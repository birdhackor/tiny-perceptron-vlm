"""獨立 runner fail-closed probes；所有 audit／git／遠端均為本地 fixture。"""

import asyncio
import hashlib
import json
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.omni_revision import execution as runner
from scripts.omni_revision import finance


def forbidden(*args, **kwargs):
    raise AssertionError("network forbidden")


def audit_probes():
    sources = {name: (ROOT / name).read_bytes() for name in runner.SOURCE_FILES}
    results = {}
    with tempfile.TemporaryDirectory(prefix="omni-execution-review-") as temporary:
        temp = Path(temporary)
        for name, raw in sources.items():
            path = temp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        fixture_path = "fixture-audit.json"
        valid = {
            "verdict": "pass", "author_task": "/root", "reviewer_task": "/root/offline_fixture_only",
            "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
        }
        committed = dict(sources)

        def git(arguments, **kwargs):
            if arguments == ["git", "rev-parse", "HEAD"]:
                return ("a" * 40).encode()
            return committed[arguments[-1].split(":", 1)[1]]

        with patch.object(runner, "ROOT", temp), patch.object(runner.subprocess, "check_output", git), patch.object(
            runner.preflight, "source_revision", lambda *args: "a" * 40
        ):
            for label in ("valid_fixture", "self_authored", "missing_file", "wrong_hash", "dirty_source", "symlink"):
                document = json.loads(json.dumps(valid))
                target = temp / runner.SOURCE_FILES[0]
                target.write_bytes(sources[runner.SOURCE_FILES[0]])
                if label == "self_authored":
                    document["reviewer_task"] = "/root"
                elif label == "missing_file":
                    document["files"].pop(runner.SOURCE_FILES[0])
                elif label == "wrong_hash":
                    document["files"][runner.SOURCE_FILES[0]] = "0" * 64
                elif label == "dirty_source":
                    target.write_bytes(b"dirty fixture")
                elif label == "symlink":
                    target.unlink()
                    target.symlink_to(temp / runner.SOURCE_FILES[1])
                raw = json.dumps(document).encode()
                (temp / fixture_path).write_bytes(raw)
                committed[fixture_path] = raw
                try:
                    runner.checked_sources(fixture_path)
                    results[label] = "accepted"
                except ValueError:
                    results[label] = "rejected"
            assert results == {"valid_fixture": "accepted", **{x: "rejected" for x in results if x != "valid_fixture"}}
    return results


class App:
    app_id = "ap-owned"

    def __init__(self, name):
        pass

    @asynccontextmanager
    async def run(self, **kwargs):
        yield self


async def ambiguous_empty_list(output):
    async def image(request):
        return SimpleNamespace(image_id=request.image_id, metadata=None)

    async def create(*args, **kwargs):
        raise TimeoutError("offline response-lost fixture")

    async def listing(**kwargs):
        if False:
            yield None

    client = SimpleNamespace(_stub=SimpleNamespace(ImageFromId=image))
    api = SimpleNamespace(ImageFromIdRequest=SimpleNamespace)
    image_type = SimpleNamespace(_new_hydrated=lambda identifier, *args: identifier)
    sandbox_type = SimpleNamespace(create=create, list=listing)
    names = runner.identifiers("probe-123-1")
    receipt = await runner.run_owned_probe(client, api, App, image_type, sandbox_type, names, output, [])
    rows = [{"App ID": "ap-owned", "Description": names["app_name"], "State": "stopped", "Tasks": "0"}]
    final = runner.reconcile_stop(receipt, rows, names)
    attempt = {"attempt_id": "probe-123-1", "status": "reserved", "reserved_usd": "0.40"}
    terminal = finance.finish({"attempts": [attempt]}, "probe-123-1", final)
    return {"receipt": final, "ledger_status": terminal["attempts"][0]["status"]}


if __name__ == "__main__":
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    with tempfile.TemporaryDirectory(prefix="omni-ambiguous-review-") as temporary:
        ambiguous = asyncio.run(ambiguous_empty_list(Path(temporary)))
    print(json.dumps({"audit_probes": audit_probes(), "ambiguous_create_with_empty_lists": ambiguous}, indent=2))
