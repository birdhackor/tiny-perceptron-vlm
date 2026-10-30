"""兩階段素材傳輸；cleanup沿用已審 first-party development runner 的逐ID停止邏輯。

此新入口尚須自己的來源／費用稽核，舊development audit不替它背書。
"""

import asyncio
import hashlib
import json
import textwrap
import time
from datetime import UTC, datetime, timedelta

from scripts.omni_revision import preflight
from scripts.omni_revision.execution import check_fresh, hard_wait, now, redacted_create_error
from scripts.omni_revision.experiment_execution import (
    CLEANUP_SECONDS,
    CREATE_SECONDS,
    IMAGE_ID,
    MAX_RESULT_BYTES,
    read_ready,
)
from scripts.omni_revision.finance import POLICY

MAX_INPUT_BYTES = 64 * 1024**2


def staged_bootstrap(payload):
    if not 0 < len(payload) <= MAX_INPUT_BYTES:
        raise ValueError("兩階段輸入封包超出64MiB上界")
    expected = hashlib.sha256(payload).hexdigest()
    code = textwrap.dedent("""\
        import hashlib,io,os,runpy,sys,time,zipfile
        from pathlib import Path,PurePosixPath
        deadline=time.monotonic()+150
        archive=Path('/tmp/omni-input.zip')
        marker=Path('/tmp/omni-input.sha256')
        while not marker.exists():
            if time.monotonic()>deadline: raise TimeoutError('bounded input transfer')
            time.sleep(.1)
        payload=archive.read_bytes()
        if len(payload)!=EXPECTED_BYTES or hashlib.sha256(payload).hexdigest()!=EXPECTED_SHA or marker.read_text()!=EXPECTED_SHA:
            raise ValueError('fixed input identity mismatch')
        root=Path('/tmp/omni-job'); root.mkdir(exist_ok=False)
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            files=bundle.infolist()
            if len(files)>512 or len({x.filename for x in files})!=len(files) or sum(x.file_size for x in files)>67108864: raise ValueError('expanded input bound')
            for info in files:
                p=PurePosixPath(info.filename)
                if p.is_absolute() or '..' in p.parts or chr(92) in info.filename or (info.external_attr>>16)&61440==40960:
                    raise ValueError('unsafe input entry')
            bundle.extractall(root)
        os.chdir(root);sys.path.insert(0,str(root))
        runpy.run_module('scripts.omni_revision.research_worker',run_name='__main__')
        """)
    code = code.replace("EXPECTED_BYTES", str(len(payload))).replace("EXPECTED_SHA", repr(expected))
    if len(code.encode()) > 4096:
        raise ValueError("bootstrap超出固定短命令上界")
    return code


async def run_staged_job(
    client, api, app_type, sandbox_type, names, output, freshness, resources, network_access, job, payload
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
            entrypoint_args=["python", "-u", "-c", staged_bootstrap(payload)],
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
        receipt["phase"] = "uploading_fixed_input_payload"
        preflight.write_json(output / "runtime-progress.json", receipt)
        await hard_wait(sandbox.filesystem.write_bytes("/tmp/omni-input.zip", payload), 90)
        await hard_wait(
            sandbox.filesystem.write_bytes("/tmp/omni-input.sha256", hashlib.sha256(payload).hexdigest().encode()), 15
        )
        receipt["input_payload_uploaded"] = True
        receipt["input_payload_bytes"] = len(payload)
        receipt["input_payload_sha256"] = hashlib.sha256(payload).hexdigest()
        receipt["phase"] = "worker_stdout_until_export_ready"
        preflight.write_json(output / "runtime-progress.json", receipt)
        (output / "stdout.txt").write_text(await hard_wait(read_ready(sandbox.stdout), job["sandbox_seconds"] - 60))
        raw = await hard_wait(sandbox.filesystem.read_bytes("/tmp/omni-result.json"), 30)
        receipt["phase"] = "exporting_result"
        if len(raw) > 64 * 1024:
            raise ValueError("worker控制結果超出64KiB")
        result = json.loads(raw)
        if (
            result.get("job") != job["job"]
            or result.get("config_sha256") != job["config_sha256"]
            or type(result.get("success")) is not bool
            or type(result.get("final_opened")) is not bool
            or (result["success"] and result["final_opened"] is not True)
            or result.get("core_selection_sha256") != job["core_selection_sha256"]
            or result.get("core_final_opening_sha256") != job["core_final_opening_sha256"]
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
            research_final_opened=result["final_opened"],
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
