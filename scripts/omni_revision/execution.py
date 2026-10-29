"""P4 首次付費邊界：先 CAS 完整預約，只沿用既有映像，取回後核實停止。

SDK 固定 1.6.0，raw async objects 共用一個獨立 client；不混用同步單例。
本入口只支援 runtime-probe；不建映像、Volume、Secret 或長期服務。
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from scripts.omni_revision import preflight
from scripts.omni_revision.finance import POLICY, GitHubLedger, digest, finish, reservation_identity, reserve, usd, utc

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ID = "im-1b8GW96gjEVRiK1vg5Jqa0"
AUDIT_PATH = "docs/course-revision-20261029-omni/infrastructure/execution-probe-audit-v1.json"
SOURCE_FILES = (
    ".github/workflows/omni-execution.yml",
    "scripts/omni_revision/execution.py",
    "scripts/omni_revision/probe_worker.py",
    "scripts/omni_revision/finance.py",
    "scripts/omni_revision/preflight.py",
    "scripts/omni_revision/legacy.py",
    "scripts/selftrained/finance.py",
    "docs/selftrained/v2-gross-quota-20261006.json",
    "tests/test_omni_execution.py",
)
RATE_CEILINGS = {
    "gpu_hour_cost_l4": "0.8",
    "cpu_hour_cost_sandbox": "0.1419",
    "mem_gib_hour_cost_sandbox": "0.024",
    "egress_gib_cost": "0.04",
}
COMPONENTS = {"compute": "0.22", "startup_idle": "0.11", "build": "0", "egress": "0.01", "storage_cleanup": "0.06"}
LIFETIME_SECONDS = 600
CREATE_SECONDS = 180
CLEANUP_SECONDS = 120
MAX_RESULT_BYTES = 64 * 1024


def now():
    return datetime.now(UTC).isoformat()


def cost_proof(rates):
    """費率上升／缺欄一律在付費前阻擋；兩個 CPU 與 8GiB 都用 Sandbox 費率。"""
    for key, ceiling in RATE_CEILINGS.items():
        if key not in rates or usd(rates[key]) > usd(ceiling):
            raise ValueError("官方 Sandbox 費率超出已審上界或缺少欄位")
    hourly = usd(rates["gpu_hour_cost_l4"]) + 2 * usd(rates["cpu_hour_cost_sandbox"])
    hourly += 8 * usd(rates["mem_gib_hour_cost_sandbox"])
    for key, seconds in (("compute", LIFETIME_SECONDS), ("startup_idle", CREATE_SECONDS + CLEANUP_SECONDS)):
        amount = (hourly * seconds / 3600).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        if amount > usd(COMPONENTS[key]):
            raise ValueError("啟動、最長存活及停止的資源費用未完全預留")
    return {
        "observed_at": now(),
        "rates": rates,
        "rate_ceilings": RATE_CEILINGS,
        "shape": {"gpu": "L4:1", "cpu_request_limit": [2, 2], "memory_mib_request_limit": [8192, 8192]},
        "lifetime_seconds": LIFETIME_SECONDS,
        "create_deadline_seconds": CREATE_SECONDS,
        "cleanup_deadline_seconds": CLEANUP_SECONDS,
        "resource_wall_bound_seconds": LIFETIME_SECONDS + CREATE_SECONDS + CLEANUP_SECONDS,
        "hourly_resource_bound_usd": str(hourly),
        "egress_bound_bytes": 1024 * 1024,
        "new_persistent_storage": False,
        "image_build_permitted": False,
        "components_usd": COMPONENTS,
        "reserved_usd": str(sum(map(usd, COMPONENTS.values()))),
    }


def checked_sources(audit_path=AUDIT_PATH):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    audit = json.loads((ROOT / audit_path).read_bytes())
    if audit.get("verdict") != "pass" or not re.fullmatch(r"/root/[A-Za-z0-9_/.-]+", audit.get("reviewer_task", "")):
        raise ValueError("需要獨立 reviewer 的實際通過紀錄")
    if audit.get("author_task") != "/root" or audit["reviewer_task"] == "/root":
        raise ValueError("執行器作者不能替自己簽獨立審閱")
    if set(audit.get("files", {})) != set(SOURCE_FILES):
        raise ValueError("獨立稽核範圍不完整")
    hashes = {}
    for name in (*SOURCE_FILES, audit_path):
        path = ROOT / name
        raw = path.read_bytes()
        if path.is_symlink() or subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT) != raw:
            raise ValueError("付費來源與稽核必須等於所選 commit")
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if name != audit_path and audit["files"][name] != hashes[name]:
            raise ValueError("原始碼已偏離獨立稽核")
    # P1 自己核對其其餘傳輸／reconciliation／停止證據，不能只信這份摘要。
    preflight.source_revision(preflight.DEFAULT_RECONCILIATION)
    return {"revision": revision, "files": hashes, "audit_sha256": hashes[audit_path]}


def identifiers(attempt_id):
    if not re.fullmatch(r"probe-[0-9]+-[0-9]+", attempt_id):
        raise ValueError("探測 ID 必須使用唯一 GitHub run／attempt；禁止重用")
    return {
        "app_name": f"omni-{attempt_id}",
        "sandbox_name": f"omni-{attempt_id}",
        "tags": {"omni-round": POLICY["round_id"], "omni-attempt": attempt_id},
    }


def check_fresh(document):
    age = (datetime.now(UTC) - utc(document["observed_at"])).total_seconds()
    if not 0 <= age <= POLICY["snapshot_max_age_seconds"]:
        raise ValueError("付費呼叫前 snapshot／inventory／rates 已過期")


def check_probe(raw):
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError("探測結果超限")
    value = json.loads(raw)
    expected = {
        "torch": "2.8.0+cu128",
        "transformers": "4.57.6",
        "numpy": "2.2.6",
        "safetensors": "0.7.0",
        "accelerate": "1.12.0",
    }
    if (
        value.get("schema") != "omni-runtime-probe-v1"
        or value.get("gpu") not in {"NVIDIA L4", "L4"}
        or value.get("gpu_count") != 1
        or value.get("gpu_memory_bytes", 0) < 22 * 1024**3
        or value.get("finite_identity_matmul") is not True
        or value.get("model_loaded") is not False
        or value.get("training_started") is not False
        or value.get("packages") != expected
        or not value.get("python", "").startswith("3.12.")
    ):
        raise ValueError("實際 GPU／套件／運算結果不符合預定探測")
    return value


async def read_ready(stream):
    raw = ""
    async for part in stream:
        raw += part
        if len(raw.encode()) > MAX_RESULT_BYTES:
            raise ValueError("探測 stdout 超限")
        if "OMNI_PROBE_READY\n" in raw:
            return raw
    raise RuntimeError("探測未完成即退出")


async def hard_wait(coroutine, seconds):
    """不等候被取消工作的 shield；到點保留未知狀態，不能宣稱完成。"""
    task = asyncio.create_task(coroutine)
    done, _ = await asyncio.wait({task}, timeout=max(0, seconds))
    if not done:
        task.cancel()
        # SDK loop 仍存活，讓已取消的 RPC 自行收尾；不重送付費呼叫。
        task.add_done_callback(lambda value: None if value.cancelled() else value.exception())
        raise TimeoutError("hard controller deadline")
    return task.result()


async def run_owned_probe(client, api, app_type, sandbox_type, names, output, freshness, resources, network_access):
    """只有一個明示 SandboxCreate RPC；不使用 loader 的自動重試／取消 shield。"""
    receipt = {"started_at": now(), "success": False, "paid_call_started": False, "cleanup_checks": {}}
    app = app_type(names["app_name"])
    context = app.run(client=client, detach=False, environment_name=POLICY["environment"])
    entered, sandbox, app_id = False, None, None
    known_ids = set()
    tags = [api.SandboxTag(tag_name=key, tag_value=value) for key, value in names["tags"].items()]

    async def active_owned():
        ids, before = [], None
        for _ in range(4):
            response = await client._stub.SandboxList(
                api.SandboxListRequest(
                    app_id=app_id, environment_name="main", tags=tags, include_finished=False, before_timestamp=before
                ),
                retry=None,
                timeout=10,
            )
            if not response.sandboxes:
                return ids
            ids.extend(row.id for row in response.sandboxes)
            before = response.sandboxes[-1].created_at
        raise RuntimeError("本次唯一 App 的 Sandbox 數量異常")

    try:
        image = await hard_wait(
            client._stub.ImageFromId(api.ImageFromIdRequest(image_id=IMAGE_ID), retry=None, timeout=15), 15
        )
        if image.image_id != IMAGE_ID:
            raise ValueError("既有 Image ID 不符")
        await hard_wait(context.__aenter__(), 30)
        entered, app_id = True, app.app_id
        receipt["app_id"] = app_id
        preflight.write_json(output / "owned-resource.json", {"app_id": app_id, **names})
        for document in freshness:
            check_fresh(document)
        definition = api.Sandbox(
            entrypoint_args=["python", "-u", "-c", (ROOT / "scripts/omni_revision/probe_worker.py").read_text()],
            image_id=IMAGE_ID,
            timeout_secs=LIFETIME_SECONDS,
            name=names["sandbox_name"],
            resources=resources,
            runtime="runc",
            network_access=network_access,
            enable_snapshot=False,
            include_oidc_identity_token=False,
        )
        request = api.SandboxCreateRequest(app_id=app_id, definition=definition, tags=tags)
        receipt.update(paid_call_started=True, paid_call_at=now())
        preflight.write_json(output / "runtime-progress.json", receipt)
        creation_deadline = time.monotonic() + CREATE_SECONDS
        # retry=None 同時關掉 transport retry 與 server-driven throttle retry。
        response = await hard_wait(
            client._stub.SandboxCreate(request, retry=None, timeout=CREATE_SECONDS), CREATE_SECONDS
        )
        known_ids.add(response.sandbox_id)
        receipt["sandbox_id"] = response.sandbox_id
        preflight.write_json(
            output / "owned-resource.json", {"app_id": app_id, "sandbox_id": response.sandbox_id, **names}
        )
        sandbox = sandbox_type._new_hydrated(response.sandbox_id, client, response.metadata)
        await hard_wait(
            sandbox._get_task_id(
                raise_if_task_complete=True,
                timeout=max(0.1, creation_deadline - time.monotonic()),
                retry_transient=False,
            ),
            creation_deadline - time.monotonic(),
        )
        (output / "stdout.txt").write_text(await hard_wait(read_ready(sandbox.stdout), 120))
        raw = await hard_wait(sandbox.filesystem.read_bytes("/tmp/omni-probe.json"), 30)
        result = check_probe(raw)
        (output / "probe-result.json").write_bytes(raw)
        payload = await hard_wait(sandbox.filesystem.read_bytes("/tmp/omni-probe-payload.bin"), 30)
        if payload != bytes(range(256)) * 4 or hashlib.sha256(payload).hexdigest() != result["payload_sha256"]:
            raise ValueError("GPU 端輸出檔案逐位元組取回失敗")
        challenge = b"omni-controller-filesystem-roundtrip-v1\x00" + bytes(range(256))
        await hard_wait(sandbox.filesystem.write_bytes(challenge, "/tmp/omni-controller.bin"), 30)
        recovered = await hard_wait(sandbox.filesystem.read_bytes("/tmp/omni-controller.bin"), 30)
        if recovered != challenge:
            raise ValueError("控制端檔案往返不符")
        receipt.update(
            success=True,
            result_sha256=hashlib.sha256(raw).hexdigest(),
            filesystem_roundtrip_sha256=hashlib.sha256(challenge).hexdigest(),
        )
    except BaseException as error:
        receipt.update(success=False, error_type=type(error).__name__)
    finally:
        deadline = time.monotonic() + CLEANUP_SECONDS
        receipt["cleanup_started_at"] = now()
        receipt["cleanup_deadline_at"] = (datetime.now(UTC) + timedelta(seconds=CLEANUP_SECONDS)).isoformat()

        async def cleanup():
            try:
                if entered:
                    try:
                        known_ids.update(await active_owned())
                        receipt["observed_owned_sandbox_ids"] = sorted(known_ids)
                        receipt["cleanup_checks"]["create_identity_observed"] = not receipt[
                            "paid_call_started"
                        ] or bool(known_ids)
                        stopped = {}
                        for identifier in sorted(known_ids):
                            await client._stub.SandboxTerminate(
                                api.SandboxTerminateRequest(sandbox_id=identifier), retry=None, timeout=10
                            )
                            # 只輪詢已觀察且屬本次 App 的 ID，不以空 list 推定未知 create 已停止。
                            while True:
                                state = await client._stub.SandboxWait(
                                    api.SandboxWaitRequest(sandbox_id=identifier, timeout=0), retry=None, timeout=10
                                )
                                if state.result.status:
                                    stopped[identifier] = state.result.status
                                    break
                                await asyncio.sleep(0.5)
                        receipt["stopped_sandboxes"] = stopped
                        remaining = await active_owned()
                        receipt["remaining_owned_sandboxes"] = remaining
                        receipt["cleanup_checks"]["owned_sandboxes_stopped"] = not remaining and (
                            not receipt["paid_call_started"] or bool(stopped)
                        )
                    finally:
                        # App context exit 亦在同一 cleanup deadline 內。
                        await context.__aexit__(None, None, None)
                        receipt["cleanup_checks"]["ephemeral_app_context_disconnected"] = True
            finally:
                await client._close()
                receipt["cleanup_checks"]["client_closed"] = True

        try:
            await hard_wait(cleanup(), deadline - time.monotonic())
        except BaseException as error:
            receipt.update(success=False, cleanup_error_type=type(error).__name__)
            receipt["cleanup_checks"]["sdk_cleanup_completed_within_deadline"] = False
        else:
            receipt["cleanup_checks"]["sdk_cleanup_completed_within_deadline"] = True
        receipt.update(finished_at=now(), app_id=app_id)
        preflight.write_json(output / "runtime-receipt.json", receipt)
    return receipt


async def execute_async(names, output, freshness):
    from modal._resources import convert_fn_config_to_resources_config
    from modal.app import _App
    from modal.client import _Client
    from modal.config import config
    from modal.sandbox import _build_outbound_network_access, _Sandbox
    from modal_proto import api_pb2

    if config.get("sandbox_v2") is not False:
        raise ValueError("未審閱的 Sandbox v2 路徑禁止啟用")
    client = await _Client.from_credentials(os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"])
    return await run_owned_probe(
        client,
        api_pb2,
        _App,
        _Sandbox,
        names,
        output,
        freshness,
        convert_fn_config_to_resources_config(cpu=(2, 2), memory=(8192, 8192), gpu="L4"),
        _build_outbound_network_access(True, None, None),
    )


def reconcile_stop(receipt, rows, names):
    """缺 App ID 的已派工、模糊／仍執行的回應均保留 unresolved。"""
    own = [row for row in rows if row.get("App ID") == receipt.get("app_id")]
    same_name = [row for row in rows if row.get("Description") == names["app_name"]]
    if not receipt["paid_call_started"]:
        stopped = not own or all(preflight.stopped_app(row) for row in own)
    else:
        stopped = bool(receipt.get("app_id")) and len(own) == 1 and preflight.stopped_app(own[0])
    receipt["cleanup_checks"]["official_app_stopped"] = stopped
    receipt["cleanup_checks"]["no_ambiguous_named_app"] = all(row in own for row in same_name)
    receipt["cleanup_checks"]["no_new_persistent_resources_requested"] = True
    receipt["compute_stopped"] = stopped and (
        not receipt["paid_call_started"] or receipt["cleanup_checks"].get("owned_sandboxes_stopped") is True
    )
    receipt["resources_verified"] = receipt["compute_stopped"] and all(receipt["cleanup_checks"].values())
    return receipt


def read_stopped_apps(receipt):
    """官方查核亦使用同一 cleanup deadline；暫態 stopping 只重讀，不重派。"""
    deadline = utc(receipt.get("cleanup_deadline_at", (datetime.now(UTC) + timedelta(seconds=40)).isoformat()))
    while True:
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("官方停止查核已超過 cleanup deadline")
        completed = subprocess.run(
            ["python", "-m", "modal", "app", "list", "--env", "main", "--json"],
            check=True,
            capture_output=True,
            timeout=min(10, remaining),
        )
        rows = json.loads(completed.stdout)
        if not isinstance(rows, list):
            raise ValueError("官方 App inventory 非 list")
        own = [row for row in rows if row.get("App ID") == receipt.get("app_id")]
        if not receipt["paid_call_started"] or len(own) == 1 and preflight.stopped_app(own[0]):
            receipt["cleanup_checks"]["cleanup_completed_within_deadline"] = datetime.now(UTC) <= deadline
            return rows
        time.sleep(min(1, max(0, (deadline - datetime.now(UTC)).total_seconds())))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=["runtime-probe"])
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("MODAL_SANDBOX_V2") != "0":
        raise ValueError("付費前必須明示 MODAL_SANDBOX_V2=0")
    if importlib.metadata.version("modal") != "1.6.0" or os.environ.get("MODAL_ENVIRONMENT", "main") != "main":
        raise ValueError("Modal SDK／environment 偏離已審配置")
    import modal

    args.output.mkdir(parents=True, exist_ok=True)
    sources, names = checked_sources(), identifiers(args.attempt_id)
    preflight.write_json(args.output / "execution-sources.json", sources)
    store = GitHubLedger()
    state, observed_blob = store.read()
    if state is None:
        raise ValueError("既有 round 未初始化；不得自動建立或重設")
    reconciliation = preflight.load_reconciliation(preflight.DEFAULT_RECONCILIATION)
    inventory = preflight.inventory(modal, args.output, reconciliation)
    workspace = modal.Workspace.from_context()
    first = state["origin"]["baseline"]["summaries"][0]["cycle"] + "-01T00:00:00+00:00"
    snapshot = preflight.official_snapshot(workspace, first)
    proof = cost_proof({key: str(value) for key, value in workspace.billing.rates().items()})
    preflight.write_json(args.output / "billing-before.json", snapshot)
    preflight.write_json(args.output / "cost-proof.json", proof)
    purpose = json.dumps(
        {
            "operation": args.operation,
            "ownership": names,
            "source_sha256": digest(sources),
            "cost_proof_sha256": digest(proof),
        },
        sort_keys=True,
    )
    following = reserve(state, snapshot, inventory, args.attempt_id, COMPONENTS, sources["revision"], purpose)
    transaction = store.write(following, observed_blob)  # 成功回應後才可進入 paid create。
    reserved = following["attempts"][-1]
    preflight.write_json(args.output / "reservation.json", {"transaction": transaction, "attempt": reserved})
    receipt = {"success": False, "paid_call_started": False, "cleanup_checks": {}}
    try:
        from modal._utils.async_utils import synchronizer

        receipt = synchronizer.create_blocking(execute_async)(names, args.output, [snapshot, inventory, proof])
    except BaseException as error:
        receipt.update(error_type=type(error).__name__)
    try:
        rows = read_stopped_apps(receipt)
        preflight.write_json(args.output / "apps-after.json", rows)
        receipt = reconcile_stop(receipt, rows, names)
    except Exception as error:
        receipt.update(
            resources_verified=False, compute_stopped=False, stop_verification_error_type=type(error).__name__
        )
    preflight.write_json(args.output / "final-receipt.json", receipt)
    latest, blob = store.read()
    matches = [x for x in latest["attempts"] if x["attempt_id"] == args.attempt_id]
    if len(matches) != 1 or reservation_identity(matches[0]) != reservation_identity(reserved):
        raise ValueError("finish 前 reservation 身分已改；保留原額，不可重試派工")
    terminal = finish(latest, args.attempt_id, receipt)
    finished = store.write(terminal, blob)
    preflight.write_json(
        args.output / "finish-transaction.json", {"transaction": finished, "attempt": terminal["attempts"][-1]}
    )
    if not receipt.get("success") or not receipt.get("resources_verified"):
        raise SystemExit("探測失敗或停止證據不完整；保留預約，不自動重跑")


if __name__ == "__main__":
    main()
