"""唯讀回查耐久帳綁定的本次 App 與全部 Sandbox；沒有 stop／create／寫帳。"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scripts.omni_revision import preflight
from scripts.omni_revision.finance import POLICY, GitHubLedger

ROOT = Path(__file__).resolve().parents[2]


async def inspect(attempt):
    from google.protobuf.empty_pb2 import Empty
    from google.protobuf.json_format import MessageToDict
    from modal.client import _Client
    from modal_proto import api_pb2

    app_id = attempt["receipt"].get("app_id")
    names = json.loads(attempt["purpose"])["ownership"]
    if not re.fullmatch(r"ap-[A-Za-z0-9]{10,}", app_id or ""):
        raise ValueError("帳內沒有本次已觀察 App ID")
    if names["tags"]["omni-attempt"] != attempt["attempt_id"] or names["tags"]["omni-round"] != POLICY["round_id"]:
        raise ValueError("持久 reservation ownership 不符")
    client = await _Client.from_credentials(os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"])
    try:
        workspace = await client._stub.WorkspaceNameLookup(Empty(), retry=None, timeout=10)
        if workspace.username != POLICY["workspace"]:
            raise ValueError("不是已授權 workspace")
        response = await client._stub.AppGetInfo(api_pb2.AppGetInfoRequest(app_id=app_id), retry=None, timeout=10)
        info = MessageToDict(response, preserving_proto_field_name=True)
        if response.info.app_id != app_id or response.info.description != names["app_name"]:
            raise ValueError("官方 exact App 身分／名稱不符")
        sandboxes, before = [], None
        for _ in range(8):
            # 只限本次唯一 App，include_finished=True，並非 active-only 的空 list。
            reply = await client._stub.SandboxList(
                api_pb2.SandboxListRequest(
                    app_id=app_id, environment_name="main", include_finished=True, before_timestamp=before
                ),
                retry=None,
                timeout=10,
            )
            if not reply.sandboxes:
                break
            sandboxes.extend(MessageToDict(row, preserving_proto_field_name=True) for row in reply.sandboxes)
            before = reply.sandboxes[-1].created_at
        else:
            raise ValueError("本次 App 的 Sandbox 超出查核上限")
        return {
            "observed_at": datetime.now(UTC).isoformat(),
            "workspace": workspace.username,
            "environment": "main",
            "attempt_id": attempt["attempt_id"],
            "reservation_sha256": attempt["reservation_sha256"],
            "ownership": names,
            "app_info": info,
            "all_sandboxes_include_finished": sandboxes,
            "rpc_allowlist": ["WorkspaceNameLookup", "AppGetInfo", "SandboxList"],
            "remote_mutation": False,
        }
    finally:
        await asyncio.wait_for(client._close(), 10)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {}
    for name in (
        "scripts/omni_revision/attempt_inspection.py",
        ".github/workflows/omni-execution.yml",
        "scripts/omni_revision/finance.py",
        "scripts/omni_revision/preflight.py",
    ):
        raw = (ROOT / name).read_bytes()
        if subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT) != raw:
            raise ValueError("唯讀 inspection source 必須等於 selected commit")
        sources[name] = hashlib.sha256(raw).hexdigest()
    state, blob = GitHubLedger().read()
    matches = [row for row in state["attempts"] if row["attempt_id"] == args.attempt_id]
    if len(matches) != 1 or matches[0]["status"] not in {"unresolved", "completed", "failed-stopped"}:
        raise ValueError("僅回查帳內已有 final receipt 的唯一 attempt")
    result = asyncio.run(inspect(matches[0]))
    args.output.mkdir(parents=True, exist_ok=True)
    preflight.write_json(args.output / "attempt-inspection.json", result)
    preflight.write_json(
        args.output / "inspection-source.json", {"revision": revision, "files": sources, "budget_blob": blob}
    )


if __name__ == "__main__":
    main()
