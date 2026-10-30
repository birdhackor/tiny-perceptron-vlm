"""受獨立稽核與耐久預算守門的 development 執行器；單次 create，無自動重跑。"""

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import time
import urllib.request
import zipfile
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from html.parser import HTMLParser
from pathlib import Path

from scripts.omni_revision import execution, preflight
from scripts.omni_revision.execution import check_fresh, hard_wait, now, redacted_create_error
from scripts.omni_revision.finance import POLICY, GitHubLedger, digest, finish, reservation_identity, reserve, usd

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ID = execution.IMAGE_ID
CREATE_SECONDS, CLEANUP_SECONDS = 480, 120
MAX_RESULT_BYTES = 128 * 1024**2
MAX_CONTROL_BYTES = 16 * 1024**2
MAX_ARTIFACT_BYTES = 256 * 1024**2
ARTIFACT_RETENTION_DAYS = 7
STORAGE_DAILY_CEILING = Decimal("0.02")
STORAGE_URL = "https://docs.github.com/en/billing/concepts/product-billing/github-actions"
COMPONENTS = {"compute": "0.32", "startup_idle": "0.22", "build": "0", "egress": "0.01", "storage_cleanup": "0.10"}
JOBS = ("dev-av", "dev-choice", "dev-sound")
AUDIT = "docs/course-revision-20261029-omni/infrastructure/experiment-execution-audit-v2.json"
MODEL_FILES = tuple(
    f"tiny_perceptron/omni_products/{name}.py"
    for name in (
        "__init__",
        "data",
        "av",
        "choice",
        "questions",
        "sound",
        "signal_metrics",
        "reliability",
        "routing",
        "training_runtime",
    )
)
WORKER_FILES = (
    "tiny_perceptron/__init__.py",
    "scripts/omni_revision/__init__.py",
    "scripts/omni_revision/product_worker.py",
    *MODEL_FILES,
)
SOURCES = (
    *WORKER_FILES,
    "scripts/omni_revision/experiment_execution.py",
    ".github/workflows/omni-products.yml",
    "tests/test_omni_training_runtime.py",
    "tests/test_omni_experiment_execution.py",
)


def checked_sources():
    old = execution.checked_sources()
    revision = old["revision"]
    audit = json.loads((ROOT / AUDIT).read_bytes())
    if (
        audit.get("verdict") != "pass"
        or audit.get("author_task") != "/root"
        or not re.fullmatch(r"/root/[A-Za-z0-9_/.-]+", audit.get("reviewer_task", ""))
        or set(audit.get("files", {})) != set(SOURCES)
    ):
        raise ValueError("development 執行器需要完整獨立稽核")
    hashes = {}
    for name in (*SOURCES, AUDIT):
        raw = (ROOT / name).read_bytes()
        if (ROOT / name).is_symlink() or subprocess.check_output(
            ["git", "show", f"{revision}:{name}"], cwd=ROOT
        ) != raw:
            raise ValueError("來源必須與 selected commit 逐位元組相同")
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if name != AUDIT and audit["files"][name] != hashes[name]:
            raise ValueError("原始碼偏離獨立稽核")
    return {"revision": revision, "files": hashes, "prior_execution_audit": old}


def job_config(name):
    if name not in JOBS:
        raise ValueError("未批准／未凍結的工作")
    return {
        "schema": "omni-development-job-v1",
        "job": name,
        "sandbox_seconds": 900,
        "gpu": "L4",
        "cpu": 2,
        "memory_mib": 8192,
        "network_blocked": True,
        "max_result_bytes": MAX_RESULT_BYTES,
        "final_opened": False,
    }


def job_bundle(job):
    raw_config = (json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    job = {**job, "config_sha256": hashlib.sha256(raw_config).hexdigest()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in WORKER_FILES:
            archive.writestr(name, (ROOT / name).read_bytes())
        archive.writestr("job.json", raw_config)
    raw = buffer.getvalue()
    if len(raw) > 512 * 1024:
        raise ValueError("來源封包超出 request 上限")
    return job, raw


def bootstrap(bundle):
    return (
        "import base64,io,zipfile,os,sys,runpy; "
        + "zipfile.ZipFile(io.BytesIO(base64.b64decode("
        + repr(base64.b64encode(bundle).decode())
        + "))).extractall('/tmp/omni-job'); os.chdir('/tmp/omni-job'); sys.path.insert(0,os.getcwd()); "
        + "runpy.run_module('scripts.omni_revision.product_worker',run_name='__main__')"
    )


class PricingText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip, self.parts = 0, []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def storage_quote(raw):
    parser = PricingText()
    parser.feed(raw.decode())
    text = " ".join(parser.parts)
    match = re.search(r"Price per GB/month Shared storage \(artifacts and GitHub Packages\) \$([0-9.]+) USD", text)
    if not match:
        raise ValueError("無法確認官方 Actions artifact 儲存費率")
    price = usd(match.group(1))
    # 官方GB用十進位，保留額以GiB計；以最短28天月份取較高日費。
    daily = price * Decimal(1024**3) / Decimal(10**9) / 28
    if daily > STORAGE_DAILY_CEILING:
        raise ValueError("官方 artifact 費率超過已審保守 ceiling")
    return {
        "observed_at": now(),
        "source_url": STORAGE_URL,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "excerpt": match.group(0),
        "actual_usd_per_decimal_gb_month": str(price),
        "conservative_daily_usd_per_gib": str(STORAGE_DAILY_CEILING),
        "hypothetical_exhausted_free_quota": True,
    }


def official_storage_quote():
    with urllib.request.urlopen(STORAGE_URL, timeout=30) as response:
        if response.url != STORAGE_URL:
            raise ValueError("官方 artifact 費率轉址需重新核對")
        raw = response.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise ValueError("官方費率頁超出來源上界")
    return storage_quote(raw)


def check_artifact_limits(output):
    files = [path for path in Path(output).rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in files):
        raise ValueError("artifact 不可包含 symlink")
    control = sum(path.stat().st_size for path in files if path.name != "result.zip")
    total = sum(path.stat().st_size for path in files)
    results = [path for path in files if path.name == "result.zip"]
    if (
        control > MAX_CONTROL_BYTES
        or total > MAX_ARTIFACT_BYTES
        or len(results) > 1
        or any(path.stat().st_size > MAX_RESULT_BYTES for path in results)
    ):
        raise ValueError("artifact／control側檔超出已預留保存上界")
    return {"total_bytes": total, "control_bytes": control, "retention_days": ARTIFACT_RETENTION_DAYS}


def cost_proof(rates, job, storage=None):
    if storage is None:
        raise ValueError("必須核對官方 artifact 儲存費率及保存上界")
    for key, ceiling in execution.RATE_CEILINGS.items():
        if key not in rates or usd(rates[key]) > usd(ceiling):
            raise ValueError("官方費率超出已審上界或缺欄")
    hourly = (
        usd(rates["gpu_hour_cost_l4"])
        + 2 * usd(rates["cpu_hour_cost_sandbox"])
        + 8 * usd(rates["mem_gib_hour_cost_sandbox"])
    )
    for key, seconds in (("compute", job["sandbox_seconds"]), ("startup_idle", CREATE_SECONDS + CLEANUP_SECONDS)):
        cost = (hourly * seconds / 3600).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        if cost > usd(COMPONENTS[key]):
            raise ValueError("運算／啟停上界未完全預留")
    if usd(rates["egress_gib_cost"]) * Decimal(MAX_RESULT_BYTES) / 1024**3 > usd(COMPONENTS["egress"]):
        raise ValueError("結果輸出上界未完全預留")
    if usd(storage["conservative_daily_usd_per_gib"]) > STORAGE_DAILY_CEILING:
        raise ValueError("artifact 費率超過保守 ceiling")
    storage_bound = (Decimal(MAX_ARTIFACT_BYTES) / 1024**3 * ARTIFACT_RETENTION_DAYS * STORAGE_DAILY_CEILING).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    if storage_bound + Decimal("0.06") > usd(COMPONENTS["storage_cleanup"]):
        raise ValueError("artifact 保存與其他收尾未完整保留")
    return {
        "observed_at": now(),
        "rates": rates,
        "hourly_resource_bound_usd": str(hourly),
        "components_usd": COMPONENTS,
        "job": job,
        "reserved_usd": "0.65",
        "new_persistent_storage": False,
        "image_build_permitted": False,
        "resource_wall_bound_seconds": job["sandbox_seconds"] + CREATE_SECONDS + CLEANUP_SECONDS,
        "storage": {
            "official_quote": storage,
            "artifact_max_bytes": MAX_ARTIFACT_BYTES,
            "control_max_bytes": MAX_CONTROL_BYTES,
            "result_max_bytes": MAX_RESULT_BYTES,
            "artifact_retention_days": ARTIFACT_RETENTION_DAYS,
            "artifact_storage_bound_usd": str(storage_bound),
            "other_cleanup_bound_usd": "0.06",
            "modal_persistent_bytes": 0,
            "modal_ephemeral_fs_retention": "Only this Sandbox lifetime; destroyed when exact owned Sandbox stops. No new Volume/image/Secret or persistent cache.",
            "accounting": "Actions artifact storage is external to Modal gross; this retained component also covers it assuming no free storage quota. No refund or free-tier assumption.",
        },
    }


def protect_core(following):
    """本次 profile 不得花掉尚未派工的核心、成熟研究與 final 預留。"""
    done = {
        json.loads(row["purpose"]).get("job")
        for row in following["attempts"]
        if row["status"] == "completed" and row["purpose"].startswith("{")
    }
    # 三個 development 各0.65；6A、2C、codec、Talker、2成熟研究、2final合計22.80。
    current = json.loads(following["attempts"][-1]["purpose"])["job"]
    future = Decimal("22.80") + Decimal("0.65") * len(set(JOBS) - done - {current})
    guard = following["attempts"][-1]["guard"]
    # 同時保留未來可能成為官方 gross 的一份，不依賴 billing 延遲或退款。
    protected = (
        usd(guard["retained_official_increment_usd"])
        + 2 * (usd(following["reserved_total_usd"]) + future)
        + usd(guard["exact_legacy_carry_usd"])
    )
    if protected > 60:
        raise ValueError("此 profile 將侵占已批准工作的保守保留額")
    return {"future_bounds_usd": str(future), "guarded_total_usd": str(protected), "cap_usd": "60"}


async def read_ready(stream):
    text = ""
    async for part in stream:
        text += part
        if len(text.encode()) > 4 * 1024**2:
            raise ValueError("worker stdout 超限")
        print(part, end="", flush=True)  # worker沒有憑證；讓實際訓練進度同步留在Actions。
        if "OMNI_JOB_READY\n" in text:
            return text
    raise RuntimeError("worker 未匯出結果即結束")


async def run_owned_job(
    client, api, app_type, sandbox_type, names, output, freshness, resources, network_access, job, bundle
):
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
            entrypoint_args=["python", "-u", "-c", bootstrap(bundle)],
            image_id=IMAGE_ID,
            timeout_secs=job["sandbox_seconds"],
            resources=resources,
            network_access=network_access,
            enable_snapshot=False,
            include_oidc_identity_token=False,
        )
        request = api.SandboxCreateRequest(app_id=app_id, definition=definition, tags=tags)
        receipt.update(paid_call_started=True, paid_call_at=now(), phase="SandboxCreate")
        preflight.write_json(output / "runtime-progress.json", receipt)
        creation_deadline = time.monotonic() + CREATE_SECONDS
        # retry=None 同時關掉 transport retry 與 server-driven throttle retry。
        try:
            response = await hard_wait(
                client._stub.SandboxCreate(request, retry=None, timeout=CREATE_SECONDS), CREATE_SECONDS
            )
        except Exception as error:
            receipt["create_error"] = redacted_create_error(error)
            raise
        known_ids.add(response.sandbox_id)
        receipt["sandbox_id"] = response.sandbox_id
        preflight.write_json(
            output / "owned-resource.json", {"app_id": app_id, "sandbox_id": response.sandbox_id, **names}
        )
        sandbox = sandbox_type._new_hydrated(response.sandbox_id, client, response.metadata)
        receipt["phase"] = "waiting_for_task_id"
        preflight.write_json(output / "runtime-progress.json", receipt)
        await hard_wait(
            sandbox._get_task_id(
                raise_if_task_complete=True,
                timeout=max(0.1, creation_deadline - time.monotonic()),
                retry_transient=False,
            ),
            creation_deadline - time.monotonic(),
        )
        receipt["phase"] = "worker_stdout_until_export_ready"
        preflight.write_json(output / "runtime-progress.json", receipt)
        (output / "stdout.txt").write_text(await hard_wait(read_ready(sandbox.stdout), job["sandbox_seconds"] - 60))
        raw = await hard_wait(sandbox.filesystem.read_bytes("/tmp/omni-result.json"), 30)
        receipt["phase"] = "exporting_result"
        result = json.loads(raw)
        if (
            len(raw) > 64 * 1024
            or result.get("job") != job["job"]
            or result.get("config_sha256") != job["config_sha256"]
            or result.get("final_opened") is not False
            or type(result.get("bytes")) is not int
            or not 0 < result["bytes"] <= MAX_RESULT_BYTES
        ):
            raise ValueError("worker identity／結果上界不符")
        payload = await hard_wait(sandbox.filesystem.read_bytes("/tmp/omni-result.zip"), 90)
        if len(payload) != result["bytes"] or hashlib.sha256(payload).hexdigest() != result["sha256"]:
            raise ValueError("GPU 結果封包取回 SHA／長度不符")
        (output / "result.zip").write_bytes(payload)
        (output / "result-summary.json").write_bytes(raw)
        receipt.update(
            success=result["success"] is True,
            result_sha256=result["sha256"],
            result_bytes=len(payload),
            result_exported_before_stop=True,
            job=job["job"],
            config_sha256=job["config_sha256"],
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
                        stopped, stop_errors = {}, {}

                        async def stop_ids(identifiers):
                            for identifier in sorted(identifiers):
                                if identifier in stopped:
                                    continue
                                try:
                                    await client._stub.SandboxTerminate(
                                        api.SandboxTerminateRequest(sandbox_id=identifier), retry=None, timeout=10
                                    )
                                    while True:
                                        state = await client._stub.SandboxWait(
                                            api.SandboxWaitRequest(sandbox_id=identifier, timeout=0),
                                            retry=None,
                                            timeout=10,
                                        )
                                        if state.result.status:
                                            stopped[identifier] = state.result.status
                                            break
                                        await asyncio.sleep(0.5)
                                except Exception as error:
                                    stop_errors[identifier] = type(error).__name__

                        # 已取得 create 回應的 ID 先停；list 的失敗不能阻擋這一步。
                        await stop_ids(known_ids)
                        discovery_ok = True
                        try:
                            known_ids.update(await active_owned())
                        except Exception as error:
                            discovery_ok = False
                            receipt["discovery_error_type"] = type(error).__name__
                        await stop_ids(known_ids - stopped.keys() - stop_errors.keys())
                        receipt["observed_owned_sandbox_ids"] = sorted(known_ids)
                        receipt["stopped_sandboxes"] = stopped
                        receipt["stop_errors"] = stop_errors
                        receipt["cleanup_checks"]["create_identity_observed"] = not receipt[
                            "paid_call_started"
                        ] or bool(known_ids)
                        remaining = None
                        try:
                            remaining = await active_owned()
                        except Exception as error:
                            discovery_ok = False
                            receipt["remaining_discovery_error_type"] = type(error).__name__
                        receipt["remaining_owned_sandboxes"] = remaining
                        receipt["cleanup_checks"]["owned_resource_discovery_completed"] = discovery_ok
                        if not discovery_ok or stop_errors:
                            receipt["success"] = False
                        receipt["cleanup_checks"]["owned_sandboxes_stopped"] = (
                            discovery_ok
                            and remaining == []
                            and not stop_errors
                            and (not receipt["paid_call_started"] or bool(stopped))
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


async def execute_async(names, output, freshness, job, bundle):
    from modal._resources import convert_fn_config_to_resources_config
    from modal.app import _App
    from modal.client import _Client
    from modal.config import config
    from modal.sandbox import _build_outbound_network_access, _Sandbox
    from modal_proto import api_pb2

    if config.get("sandbox_v2") is not False:
        raise ValueError("必須使用已稽核的 Sandbox v1")
    client = await _Client.from_credentials(os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"])
    return await run_owned_job(
        client,
        api_pb2,
        _App,
        _Sandbox,
        names,
        output,
        freshness,
        convert_fn_config_to_resources_config(cpu=(2, 2), memory=(8192, 8192), gpu="L4"),
        _build_outbound_network_access(True, None, None),
        job,
        bundle,
    )


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("job", choices=JOBS)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("MODAL_SANDBOX_V2") != "0"
        or importlib.metadata.version("modal") != "1.6.0"
        or os.environ.get("MODAL_ENVIRONMENT", "main") != "main"
        or not re.fullmatch(r"job-[0-9]+-[0-9]+", args.attempt_id)
    ):
        raise ValueError("SDK／environment／唯一 attempt ID 不符")
    import modal
    from modal._utils.async_utils import synchronizer

    args.output.mkdir(parents=True, exist_ok=True)
    sources = checked_sources()
    names = {
        "app_name": "omni-" + args.attempt_id,
        "tags": {"omni-round": POLICY["round_id"], "omni-attempt": args.attempt_id},
    }
    job, bundle = job_bundle(job_config(args.job))
    preflight.write_json(args.output / "execution-sources.json", sources)
    preflight.write_json(args.output / "job-config.json", job)
    (args.output / "source-bundle.zip").write_bytes(bundle)
    store = GitHubLedger()
    state, blob = store.read()
    if state is None:
        raise ValueError("既有 round 不得重設")
    reconciliation = preflight.load_reconciliation(preflight.DEFAULT_RECONCILIATION)
    inventory = preflight.inventory(modal, args.output, reconciliation)
    workspace = modal.Workspace.from_context()
    first = state["origin"]["baseline"]["summaries"][0]["cycle"] + "-01T00:00:00+00:00"
    snapshot = preflight.official_snapshot(workspace, first)
    storage = official_storage_quote()
    proof = cost_proof({key: str(value) for key, value in workspace.billing.rates().items()}, job, storage)
    preflight.write_json(args.output / "billing-before.json", snapshot)
    preflight.write_json(args.output / "cost-proof.json", proof)
    purpose = json.dumps(
        {
            "job": args.job,
            "ownership": names,
            "sources_sha256": digest(sources),
            "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
            "cost_proof_sha256": digest(proof),
        },
        sort_keys=True,
    )
    following = reserve(state, snapshot, inventory, args.attempt_id, COMPONENTS, sources["revision"], purpose)
    preflight.write_json(args.output / "core-protection.json", protect_core(following))
    check_artifact_limits(args.output)
    transaction = store.write(following, blob)
    reserved = following["attempts"][-1]
    preflight.write_json(args.output / "reservation.json", {"transaction": transaction, "attempt": reserved})
    # 一旦進 executor，外層不能從缺失的 receipt 推定「沒有付費」。
    # 例如回寫 receipt 的磁碟錯誤會越過內層 finally；未知就保留 unresolved。
    receipt = {
        "success": False,
        "paid_call_started": True,
        "paid_call_state": "unknown_after_executor_entry",
        "cleanup_checks": {},
    }
    try:
        receipt = synchronizer.create_blocking(execute_async)(
            names, args.output, [snapshot, inventory, proof, storage], job, bundle
        )
    except BaseException as error:
        receipt["error_type"] = type(error).__name__
    try:
        rows = execution.read_stopped_apps(receipt)
        preflight.write_json(args.output / "apps-after.json", rows)
        receipt = execution.reconcile_stop(receipt, rows, names)
    except Exception as error:
        receipt.update(
            resources_verified=False, compute_stopped=False, stop_verification_error_type=type(error).__name__
        )
    preflight.write_json(args.output / "final-receipt.json", receipt)
    latest, blob = store.read()
    matches = [row for row in latest["attempts"] if row["attempt_id"] == args.attempt_id]
    if len(matches) != 1 or reservation_identity(matches[0]) != reservation_identity(reserved):
        raise ValueError("final reservation 身分已改，不可重試派工")
    terminal = finish(latest, args.attempt_id, receipt)
    transaction = store.write(terminal, blob)
    preflight.write_json(
        args.output / "finish-transaction.json", {"transaction": transaction, "attempt": terminal["attempts"][-1]}
    )
    check_artifact_limits(args.output)
    if not receipt.get("success") or not receipt.get("resources_verified"):
        raise SystemExit("工作失敗或停止未完全核實；保留原額，沒有自動重跑")


if __name__ == "__main__":
    main()
