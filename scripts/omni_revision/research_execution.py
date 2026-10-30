"""兩個固定成熟模型的有限研究：完整預約、一次create、先匯出再核實停止。"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path, PurePosixPath

from scripts.omni_revision import execution, preflight, research_recovery
from scripts.omni_revision.core_final import evaluation_sources
from scripts.omni_revision.cpu_execution import upload_owned
from scripts.omni_revision.experiment_execution import (
    ARTIFACT_RETENTION_DAYS,
    CLEANUP_SECONDS,
    CREATE_SECONDS,
    IMAGE_ID,
    MAX_ARTIFACT_BYTES,
    MAX_CONTROL_BYTES,
    MAX_RESULT_BYTES,
    STORAGE_DAILY_CEILING,
    official_storage_quote,
)
from scripts.omni_revision.finance import POLICY, GitHubLedger, digest, finish, reservation_identity, reserve, usd
from scripts.omni_revision.formal_local import FROZEN, ROOT
from scripts.omni_revision.github_transport_probe import REPOSITORY, api, find_draft
from scripts.omni_revision.staged_execution import MAX_INPUT_BYTES, run_staged_job
from tiny_perceptron.omni_products.training_runtime import file_sha, json_file

AUDIT = ROOT / "docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v3.json"
JOBS = {
    "kev": {
        "gpu": "L40S",
        "components": {
            "compute": "3.00",
            "startup_idle": "1.00",
            "build": "0",
            "egress": "0.10",
            "storage_cleanup": "0.20",
        },
    },
    "qwen25": {
        "gpu": "L4",
        "components": {
            "compute": "2.00",
            "startup_idle": "0.80",
            "build": "0",
            "egress": "0.10",
            "storage_cleanup": "0.20",
        },
    },
}
RESEARCH_FILES = (
    *[
        f"scripts/omni_revision/{name}.py"
        for name in (
            "research_execution",
            "research_recovery",
            "research_core_inputs",
            "research_measurements",
            "research_model_files",
            "research_worker",
            "research_upstream",
            "research_cases",
            "staged_execution",
            "worker_outputs",
            "kev_research",
            "qwen_research",
            "qwen_inputs",
        )
    ],
    "scripts/omni_revision/research_requirements/kev.txt",
    "scripts/omni_revision/research_requirements/qwen25.txt",
    "tests/test_omni_mature_research.py",
    "tests/test_omni_mature_modal_sdk.py",
    ".github/workflows/omni-mature-research.yml",
    "docs/omni-products/v1/mature-research-protocol.md",
    research_recovery.PERMIT_NAME,
    research_recovery.ORIGINAL_NAME,
)


def source_paths():
    recipe = json.loads((FROZEN / "formal-recipes.json").read_bytes())
    paths = set(recipe["execution_sources_sha256"]) | set(evaluation_sources()) | set(RESEARCH_FILES)
    paths.update(
        (FROZEN / name).relative_to(ROOT).as_posix()
        for name in ("data-definition.json", "metrics.json", "formal-recipes.json")
    )
    return sorted(paths)


def identity(job, attempt):
    if (
        job not in JOBS
        or os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or attempt != f"mature-{job}-{os.environ.get('GITHUB_RUN_ID')}-{os.environ.get('GITHUB_RUN_ATTEMPT')}"
        or re.fullmatch(r"mature-(kev|qwen25)-[0-9]+-[0-9]+", attempt) is None
        or os.environ.get("MODAL_SANDBOX_V2") != "0"
        or os.environ.get("MODAL_ENVIRONMENT") != "main"
    ):
        raise ValueError("成熟研究只限授權repo/main手動單次工作及原main環境")


def release_identity(attempt, revision):
    return (
        "omni-mature-candidate-" + attempt,
        "Owned unpublished limited mature-model research; " + attempt + "; source " + revision,
    )


def check_receipt_identity(job, attempt, sources, reserved, receipt):
    """先綁本次預約與收據，再讀provider；真foreign draft也不能代本次執行。"""
    purpose = research_recovery.purpose(reserved)
    if (
        reserved.get("attempt_id") != attempt
        or reserved.get("revision") != sources["revision"]
        or reserved.get("status") != "reserved"
        or purpose.get("job") != "mature-" + job
        or purpose.get("sources_sha256") != digest(sources)
        or purpose.get("controller_audit_sha256") != sources["audit_sha256"]
        or purpose.get("ownership")
        != {
            "app_name": "omni-" + attempt,
            "tags": {"omni-round": POLICY["round_id"], "omni-attempt": attempt},
        }
    ):
        raise ValueError("成熟研究預約不是本次已審來源／工作")
    present = set(purpose) & set(research_recovery.LINEAGE_KEYS)
    if present and present != set(research_recovery.LINEAGE_KEYS):
        raise ValueError("準備修復lineage不能缺欄位")
    expected_lineage = None
    if present:
        expected_lineage = research_recovery.lineage(file_sha(ROOT / research_recovery.PERMIT_NAME))
        if job != "kev" or any(purpose[key] != value for key, value in expected_lineage.items()):
            raise ValueError("準備修復的預約／permit身分不同")
    tag, body = release_identity(attempt, sources["revision"])
    expected = {
        "schema": "omni-mature-controller-receipt-v2",
        "attempt": attempt,
        "job": job,
        "source_revision": sources["revision"],
        "controller_audit_sha256": sources["audit_sha256"],
        "release_tag": tag,
        "release_body": body,
        "input_payload_sha256": purpose["input_sha256"],
        "config_sha256": purpose["job_config_sha256"],
        "core_selection_sha256": purpose["core_selection_sha256"],
        "core_final_opening_sha256": purpose["core_final_opening_sha256"],
        "preparation_recovery": expected_lineage,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("成熟研究收據來自另一attempt／來源／輸入／草稿")


def committed(path, revision):
    path = Path(path)
    relative = path.relative_to(ROOT).as_posix()
    if (
        path.is_symlink()
        or subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT) != path.read_bytes()
    ):
        raise ValueError("研究來源與已提交revision不同")


def checked_sources():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    if revision != os.environ.get("GITHUB_SHA"):
        raise ValueError("研究checkout不是本次固定Actions SHA")
    hashes = {name: file_sha(ROOT / name) for name in source_paths()}
    audit = json.loads(AUDIT.read_bytes())
    if (
        audit.get("status") != "pass_for_reviewed_scope"
        or not re.fullmatch(r"/root/[A-Za-z0-9_/.-]+", audit.get("reviewer_task", ""))
        or audit.get("files") != hashes
    ):
        raise ValueError("成熟研究執行／費用／輸入／輸出須先有完整獨立稽核")
    for path in [*(ROOT / name for name in hashes), AUDIT]:
        committed(path, revision)
    return {"revision": revision, "files": hashes, "audit_sha256": file_sha(AUDIT)}


def checked_core(revision, state):
    from scripts.omni_revision.research_core_inputs import checked_core as strict_core

    return strict_core(revision, state)


def job_config(name, core):
    if name not in JOBS:
        raise ValueError("不在兩個成熟研究之內")
    return {
        "schema": "omni-mature-research-job-v1",
        "job": name,
        "gpu": JOBS[name]["gpu"],
        "cpu": 4,
        "memory_mib": 49152,
        "sandbox_seconds": 2100,
        "worker_seconds": 1800,
        "image_id": IMAGE_ID,
        "network_blocked": False,
        "own_training_permitted": False,
        **core,
    }


def worker_files():
    paths = {
        "tiny_perceptron/__init__.py",
        "scripts/omni_revision/__init__.py",
        "scripts/omni_revision/core_evaluation.py",
    }
    paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "tiny_perceptron").glob("*.py"))
    paths.update(
        path.relative_to(ROOT).as_posix()
        for folder in ("omni_products", "selftrained", "lesson_products")
        for path in (ROOT / "tiny_perceptron" / folder).glob("*.py")
    )
    paths.update(
        name
        for name in RESEARCH_FILES
        if name.startswith("scripts/")
        and not name.endswith(
            (
                "research_execution.py",
                "research_recovery.py",
                "research_core_inputs.py",
                "research_measurements.py",
                "staged_execution.py",
                "research_cases.py",
            )
        )
    )
    return sorted(paths)


def prepare(job, output):
    with tempfile.TemporaryDirectory(prefix="omni-fixed-research-input-") as temporary:
        root = Path(temporary)
        for name in worker_files():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        for filename in ("data-definition.json", "model-selection.json", "final-opening.json"):
            shutil.copyfile(FROZEN / filename, root / filename)
        if job["job"] == "kev":
            from scripts.omni_revision.research_upstream import download

            preparation = download(root / "thirdparty/kev")
        else:
            from scripts.omni_revision.research_cases import build

            preparation = build(root / "cases")
        files = [path for path in root.rglob("*") if path.is_file()]
        if len(files) >= 512 or sum(path.stat().st_size for path in files) > MAX_INPUT_BYTES:
            raise ValueError("研究輸入超出64MiB／512檔界")
        job["input_files"] = {
            path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": file_sha(path)}
            for path in sorted(files)
        }
        raw_config = (json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(files):
                archive.write(path, path.relative_to(root))
            archive.writestr("job.json", raw_config)
        payload = buffer.getvalue()
        if not 0 < len(payload) <= MAX_INPUT_BYTES:
            raise ValueError("研究輸入zip超出事前上界")
    json_file(
        output / "input-preparation.json",
        {
            "prepared_before_paid_create": True,
            "model_outputs_used": False,
            "input_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "input_payload_bytes": len(payload),
            "preparation": preparation,
        },
    )
    (output / "input.zip").write_bytes(payload)
    (output / "job.json").write_bytes(raw_config)
    return {**job, "config_sha256": hashlib.sha256(raw_config).hexdigest()}, payload


def cost_proof(rates, job, storage):
    components = JOBS[job["job"]]["components"]
    ceilings = {**execution.RATE_CEILINGS, "gpu_hour_cost_l40s": "1.95"}
    for key in (
        "gpu_hour_cost_" + job["gpu"].lower(),
        "cpu_hour_cost_sandbox",
        "mem_gib_hour_cost_sandbox",
        "egress_gib_cost",
    ):
        if key not in rates or usd(rates[key]) > usd(ceilings[key]):
            raise ValueError("成熟研究官方費率超出事前上界")
    hourly = (
        usd(rates["gpu_hour_cost_" + job["gpu"].lower()])
        + 4 * usd(rates["cpu_hour_cost_sandbox"])
        + 48 * usd(rates["mem_gib_hour_cost_sandbox"])
    )
    for key, seconds in (("compute", 2100), ("startup_idle", CREATE_SECONDS + CLEANUP_SECONDS)):
        if (hourly * seconds / 3600).quantize(Decimal("0.01"), rounding=ROUND_CEILING) > usd(components[key]):
            raise ValueError("成熟研究整個資源存活／啟停時間未完整預留")
    if usd(rates["egress_gib_cost"]) * Decimal(MAX_RESULT_BYTES) / 1024**3 > usd(components["egress"]):
        raise ValueError("成熟研究egress預留不足")
    if usd(storage["conservative_daily_usd_per_gib"]) > STORAGE_DAILY_CEILING:
        raise ValueError("Actions artifact官方費率超界")
    storage_bound = (Decimal(MAX_ARTIFACT_BYTES) / 1024**3 * ARTIFACT_RETENTION_DAYS * STORAGE_DAILY_CEILING).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    if storage_bound + Decimal("0.06") > usd(components["storage_cleanup"]):
        raise ValueError("成熟研究artifact保存／收尾未預留")
    return {
        "observed_at": execution.now(),
        "rates": rates,
        "shape": {
            key: job[key] for key in ("gpu", "cpu", "memory_mib", "sandbox_seconds", "worker_seconds", "image_id")
        },
        "hourly_resource_bound_usd": str(hourly),
        "components_usd": components,
        "reserved_usd": str(sum(map(usd, components.values()))),
        "resource_wall_bound_seconds": 2100 + CREATE_SECONDS + CLEANUP_SECONDS,
        "storage": {
            "official_quote": storage,
            "artifact_max_bytes": MAX_ARTIFACT_BYTES,
            "artifact_retention_days": 7,
            "artifact_storage_bound_usd": str(storage_bound),
            "other_cleanup_bound_usd": "0.06",
        },
        "new_persistent_storage": False,
        "image_build_permitted": False,
        "automatic_retry_or_gpu_upgrade": False,
    }


def protect_remaining(state, current):
    purposes = [json.loads(row["purpose"]) for row in state["attempts"] if row["purpose"].startswith("{")]
    completed_core = [
        row
        for row in state["attempts"]
        if row["status"] == "completed"
        and row["purpose"].startswith("{")
        and json.loads(row["purpose"]).get("job") == "cpu-core-formal"
    ]
    if (
        len(completed_core) != 1
        or not completed_core[0]["receipt"].get("all_training_processes_exited")
        or not completed_core[0]["receipt"].get("success")
    ):
        raise ValueError("十個核心正式工作須先完整完成；不能先花掉核心預算")
    attempted = {purpose.get("job") for purpose in purposes}
    future = Decimal("2.00") + sum(
        sum(map(usd, value["components"].values()))
        for name, value in JOBS.items()
        if "mature-" + name not in attempted and name != current
    )
    guard = state["attempts"][-1]["guard"]
    protected = (
        usd(guard["retained_official_increment_usd"])
        + 2 * (usd(state["reserved_total_usd"]) + future)
        + usd(guard["exact_legacy_carry_usd"])
    )
    if protected > 60:
        raise ValueError("成熟研究侵占完整gross60與尚未執行工作的保留額")
    return {
        "future_bounds_usd": str(future),
        "guarded_total_usd": str(protected),
        "cap_usd": "60",
        "core_completed_as_standard_CPU": True,
        "final_and_publication_contingency_usd": "2.00",
        "no_refunds_assumed": True,
    }


async def execute(names, output, freshness, job, payload):
    from modal._resources import convert_fn_config_to_resources_config
    from modal.app import _App
    from modal.client import _Client
    from modal.config import config
    from modal.sandbox import _build_outbound_network_access, _Sandbox
    from modal_proto import api_pb2

    if config.get("sandbox_v2") is not False:
        raise ValueError("須沿用已核對的Sandbox v1")
    client = await _Client.from_credentials(os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"])
    return await run_staged_job(
        client,
        api_pb2,
        _App,
        _Sandbox,
        names,
        output,
        freshness,
        convert_fn_config_to_resources_config(cpu=(4, 4), memory=(49152, 49152), gpu=job["gpu"]),
        _build_outbound_network_access(False, None, None),
        job,
        payload,
    )


def verify_results(path, job):
    """取回原始列；外部權重或第三方全文不能混入自製研究結果。"""
    with zipfile.ZipFile(path) as archive:
        files = archive.infolist()
        names = {row.filename for row in files}
        if len(files) != len(names) or len(files) > 1024 or sum(row.file_size for row in files) > MAX_ARTIFACT_BYTES:
            raise ValueError("研究結果zip展開超界或重複路徑")
        for row in files:
            relative = PurePosixPath(row.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in row.filename
                or (row.external_attr >> 16) & 0o170000 == 0o120000
                or relative.suffix not in (".json", ".jsonl", ".wav", ".log")
            ):
                raise ValueError("研究結果含不允許的檔案／權重／全文")
        manifest = json.loads(archive.read("file-manifest.json"))
        if set(manifest) != names - {"file-manifest.json"}:
            raise ValueError("研究結果逐檔manifest不完整")
        for name, binding in manifest.items():
            raw = archive.read(name)
            if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
                raise ValueError("研究結果逐檔SHA／大小不符")
        summary = json.loads(archive.read("worker-summary.json"))
        if (
            summary.get("schema") != "omni-mature-worker-summary-v1"
            or summary["job"] != job["job"]
            or summary["core_selection_sha256"] != job["core_selection_sha256"]
            or summary.get("own_training_permitted") is not False
            or type(summary.get("success")) is not bool
            or type(summary.get("final_opened")) is not bool
        ):
            raise ValueError("研究結果工作／核心候選不符")
        if summary["success"]:
            from scripts.omni_revision.research_measurements import check_kev, check_qwen, finite, same

            result = json.loads(archive.read("research-result.json"))
            if (
                result.get("completed") is not True
                or result.get("final_opened") is not True
                or summary["final_opened"] is not True
                or hashlib.sha256(archive.read("research-result.json")).hexdigest() != summary["research_result_sha256"]
                or not finite(summary["elapsed_seconds"])
                or summary["elapsed_seconds"] > job["worker_seconds"] + 30
            ):
                raise ValueError("研究結果宣告不完整")
            runtime = json.loads(archive.read("runtime-before.json"))
            isolated = json.loads(archive.read("isolated-runtime.json"))
            if (
                not same(summary["runtime"], runtime)
                or not same(summary["isolated_runtime"], isolated)
                or runtime["python"] != "3.12.10"
                or runtime["torch"] != "2.8.0+cu128"
                or runtime["cuda"] != "12.8"
                or runtime["gpu"] != "NVIDIA " + job["gpu"]
                or isolated["torch"] != runtime["torch"]
                or isolated["cuda"] != runtime["cuda"]
            ):
                raise ValueError("成熟研究實際runtime收據不同")
            lock = ROOT / f"scripts/omni_revision/research_requirements/{job['job']}.txt"
            locked = dict(line.split("==") for line in lock.read_text().splitlines())
            if set(isolated["packages"]) != {
                "transformers",
                "huggingface-hub",
                "tokenizers",
                "numpy",
                "accelerate",
                "safetensors",
            } or any(version != locked[name] for name, version in isolated["packages"].items()):
                raise ValueError("成熟研究實際隔離版本不同")
            opened = json.loads(archive.read("research-opening.json"))
            if (
                opened["job"] != job["job"]
                or opened["core_selection_sha256"] != job["core_selection_sha256"]
                or opened["core_final_opening_sha256"] != job["core_final_opening_sha256"]
                or opened["job_config_sha256"] != job["config_sha256"]
                or opened["data_definition_sha256"] != job["input_files"]["data-definition.json"]["sha256"]
                or opened["input_files"] != job["input_files"]
                or opened["core_models_immutable"] is not True
            ):
                raise ValueError("成熟研究開封／輸入／核心固定來源不同")
            if job["job"] == "kev":
                check_kev(archive, result, job)
            else:
                check_qwen(archive, result, job, path.parent / "input.zip")
    return {
        "all_entries_verified": True,
        "files": len(names),
        "worker_success": summary["success"],
        "research_final_opened": summary["final_opened"],
    }


def bound(output):
    files = [path for path in output.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    controls = sum(path.stat().st_size for path in files if path.suffix != ".zip")
    if (
        any(path.is_symlink() for path in files)
        or controls > MAX_CONTROL_BYTES
        or total > MAX_ARTIFACT_BYTES
        or any(path.stat().st_size > MAX_RESULT_BYTES for path in files if path.suffix == ".zip")
    ):
        raise ValueError("成熟研究artifact超出16MiB control／256MiB aggregate／128MiB單包")
    return {"total_bytes": total, "control_bytes": controls, "retention_days": 7}


def run(job_name, attempt, output):
    identity(job_name, attempt)
    if importlib.metadata.version("modal") != "1.6.0":
        raise ValueError("固定Modal SDK不符")
    import modal
    from modal._utils.async_utils import synchronizer

    sources = checked_sources()
    output.mkdir(parents=True, exist_ok=False)
    json_file(output / "sources.json", sources)
    store = GitHubLedger()
    state, blob = store.read()
    if state is None:
        raise ValueError("沿用既有round，不可重設預算")
    lineage = research_recovery.eligibility(state, job_name)
    core = checked_core(sources["revision"], state)
    if lineage is None:
        job, payload = prepare(job_config(job_name, core), output)
    else:
        job, payload = research_recovery.reuse_input(job_config(job_name, core), output, worker_files(), lineage)
    workspace = modal.Workspace.from_context()
    inventory = preflight.inventory(modal, output, preflight.load_reconciliation(preflight.DEFAULT_RECONCILIATION))
    first = state["origin"]["baseline"]["summaries"][0]["cycle"] + "-01T00:00:00+00:00"
    storage = official_storage_quote()
    proof = cost_proof({key: str(value) for key, value in workspace.billing.rates().items()}, job, storage)
    snapshot = preflight.official_snapshot(workspace, first)
    names = {"app_name": "omni-" + attempt, "tags": {"omni-round": POLICY["round_id"], "omni-attempt": attempt}}
    purpose = json.dumps(
        {
            "job": "mature-" + job_name,
            "ownership": names,
            "sources_sha256": digest(sources),
            "controller_audit_sha256": sources["audit_sha256"],
            "input_sha256": hashlib.sha256(payload).hexdigest(),
            "job_config_sha256": job["config_sha256"],
            "cost_proof_sha256": digest(proof),
            **core,
            **(lineage or {}),
        },
        sort_keys=True,
    )
    following = reserve(state, snapshot, inventory, attempt, JOBS[job_name]["components"], sources["revision"], purpose)
    json_file(output / "remaining-protection.json", protect_remaining(following, job_name))
    for name, value in (("cost-proof", proof), ("billing-before", snapshot)):
        json_file(output / f"{name}.json", value)
    bound(output)
    transaction = store.write(following, blob)
    reserved = following["attempts"][-1]
    json_file(output / "reservation.json", {"transaction": transaction, "attempt": reserved})
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write("reserved=true\n")
    tag, body = release_identity(attempt, sources["revision"])
    receipt = {
        "schema": "omni-mature-controller-receipt-v2",
        "job": job_name,
        "source_revision": sources["revision"],
        "controller_audit_sha256": sources["audit_sha256"],
        "input_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "config_sha256": job["config_sha256"],
        "core_selection_sha256": core["core_selection_sha256"],
        "core_final_opening_sha256": core["core_final_opening_sha256"],
        "preparation_recovery": lineage,
        "success": False,
        "paid_call_started": False,
        "cleanup_checks": {"no_modal_executor_entered": True},
        "compute_stopped": True,
        "resources_verified": True,
        "release_tag": tag,
        "release_body": body,
        "assets": [],
        "attempt": attempt,
    }
    try:
        if find_draft(tag, body) is not None:
            raise ValueError("唯一研究草稿已存在，不重跑")
        json_file(output / "release-ownership.json", {"tag": tag, "body": body, "create_started": True})
        created = api(
            "releases",
            "POST",
            {
                "tag_name": tag,
                "target_commitish": sources["revision"],
                "name": "Unpublished limited mature-model research",
                "body": body,
                "draft": True,
                "prerelease": True,
            },
        )
        receipt["release_id"] = created["id"]
        receipt["assets"].append(upload_owned(tag, body, created["id"], output / "input.zip"))
        # executor entry後，沒有完整receipt時保留unknown，不能推測沒有付費。
        receipt.update(
            paid_call_started=True,
            compute_stopped=False,
            resources_verified=False,
            cleanup_checks={},
            paid_call_state="unknown_after_executor_entry",
        )
        runtime = synchronizer.create_blocking(execute)(
            names, output, [snapshot, inventory, proof, storage], job, payload
        )
        receipt.update(runtime)
        rows = execution.read_stopped_apps(receipt)
        json_file(output / "apps-after.json", rows)
        receipt = execution.reconcile_stop(receipt, rows, names)
        if (output / "result.zip").exists():
            receipt["result_verification"] = verify_results(output / "result.zip", job)
            receipt["assets"].append(upload_owned(tag, body, created["id"], output / "result.zip"))
        if not receipt.get("result_verification", {}).get("worker_success"):
            receipt["success"] = False
    except BaseException as error:
        receipt.update(success=False, error_type=type(error).__name__)
    finally:
        json_file(output / "research-receipt.json", receipt)
    bound(output)
    if not receipt["success"] or not receipt["resources_verified"]:
        raise SystemExit("成熟研究失敗或停止未核實；保留原額與證據，不自動重跑")


def finish_run(job, attempt, output):
    identity(job, attempt)
    sources = checked_sources()
    reserved = json.loads((output / "reservation.json").read_bytes())["attempt"]
    receipt = json.loads((output / "research-receipt.json").read_bytes())
    check_receipt_identity(job, attempt, sources, reserved, receipt)
    bounds = bound(output)
    record = api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}/artifacts?per_page=100")
    items = record["artifacts"]
    expected_name = "omni-" + attempt
    artifact_ok = (
        record["total_count"] == len(items) == 1
        and os.environ.get("OMNI_ARTIFACT_OUTCOME") == "success"
        and all(
            item["name"] == expected_name
            and item["size_in_bytes"] <= MAX_ARTIFACT_BYTES
            and 0
            < (
                datetime.fromisoformat(item["expires_at"].replace("Z", "+00:00"))
                - datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
            ).total_seconds()
            <= 7 * 86400 + 60
            for item in items
        )
    )
    owned = (
        api(f"releases/{receipt['release_id']}")
        if receipt.get("release_id")
        else find_draft(receipt["release_tag"], receipt["release_body"])
    )
    release_ok = (
        owned is not None
        and owned.get("draft") is True
        and owned.get("tag_name") == receipt["release_tag"]
        and owned.get("body") == receipt["release_body"]
        and owned.get("target_commitish") == sources["revision"]
        and {(asset["id"], asset["name"], asset["size"]) for asset in owned["assets"]}
        == {(asset["id"], asset["name"], asset["bytes"]) for asset in receipt["assets"]}
        and all(asset.get("roundtrip_verified") is True for asset in receipt["assets"])
        and {asset["name"] for asset in receipt["assets"]} <= {"input.zip", "result.zip"}
        and (
            receipt.get("success") is not True
            or (
                {asset["name"] for asset in receipt["assets"]} == {"input.zip", "result.zip"}
                and receipt.get("result_exported_before_stop") is True
                and receipt.get("research_final_opened") is True
            )
        )
    )
    receipt["cleanup_checks"].update(
        artifact_retention_and_size_bounded=artifact_ok,
        results_intentionally_retained_in_verified_owned_draft=release_ok,
    )
    receipt["resources_verified"] = receipt["resources_verified"] and artifact_ok and release_ok
    receipt["success"] = receipt["success"] and receipt["resources_verified"]
    receipt.update(
        artifact=bounds,
        artifact_api_records=[
            {key: item[key] for key in ("id", "name", "size_in_bytes", "created_at", "expires_at")} for item in items
        ],
    )
    store = GitHubLedger()
    state, blob = store.read()
    matching = [row for row in state["attempts"] if row["attempt_id"] == attempt]
    if len(matching) != 1 or reservation_identity(matching[0]) != reservation_identity(reserved):
        raise ValueError("成熟研究預約身分已變，不得重派")
    following = finish(state, attempt, receipt)
    transaction = store.write(following, blob)
    json_file(
        output / "finish-transaction.json",
        {
            "transaction": transaction,
            "attempt": next(row for row in following["attempts"] if row["attempt_id"] == attempt),
        },
    )


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=("run", "bound", "finish"))
    parser.add_argument("--job", required=True, choices=tuple(JOBS))
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "run":
        run(args.job, args.attempt_id, args.output)
    elif args.operation == "finish":
        finish_run(args.job, args.attempt_id, args.output)
    else:
        print(json.dumps(bound(args.output)))


if __name__ == "__main__":
    main()
