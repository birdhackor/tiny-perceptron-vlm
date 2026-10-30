"""只接續一次已證實在首寫前失敗的Kev準備；不放寬一般科學工作防重。"""

import hashlib
import io
import json
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from scripts.omni_revision.finance import digest
from scripts.omni_revision.formal_local import FROZEN, ROOT
from scripts.omni_revision.github_transport_probe import REPOSITORY, api, command
from scripts.omni_revision.staged_execution import MAX_INPUT_BYTES
from tiny_perceptron.omni_products.training_runtime import file_sha, json_file

PERMIT_NAME = "docs/course-revision-20261029-omni/infrastructure/mature-research-preparation-recovery-permit-v1.json"
ORIGINAL_NAME = "docs/course-revision-20261029-omni/infrastructure/mature-research-runtime-review-original-v1.json"
ORIGINAL_SHA = "d947cf06c95c28c60f34ab02f70a0bb4a3ec6f5d3348e6af2b08945e68082d40"
IDENTITY = {
    "job": "kev",
    "attempt": "mature-kev-41004743884-1",
    "source_revision": "ee799d24fd1ea26108af709ec41e49db63d58e31",
    "row_sha256": "5639aefbe9acf390f652fe9e5edfcb175d0360fdef404910e921a9ac93b20bbc",
    "receipt_sha256": "163c961242a6780e665bcb01f0b40a8e18129c78eb9a90b4d99a109783eaaaa0",
    "job_config_sha256": "73f75133744798cdf1bd9c870c3120c2e6f77647a2321adbc16f87de012f9328",
    "release_id": 428663042,
    "release_tag": "omni-mature-candidate-mature-kev-41004743884-1",
    "asset_id": 690960942,
    "asset_name": "input.zip",
    "asset_bytes": 434670,
    "asset_sha256": "8652c3843caeabf93dbb3ce99a24dd06632b27ab1d7db0c45da0004b83540823",
    "app_id": "ap-tb4m5Jl1b8Zwbi3wwVonu7",
    "sandbox_id": "sb-yxbPtKwCRI3u2pImYXT1OC",
}
LINEAGE_KEYS = (
    "recovery_of",
    "preparation_recovery_permit_sha256",
    "original_input_payload_sha256",
    "original_job_config_sha256",
    "original_row_sha256",
)


def require(value, message):
    if not value:
        raise ValueError("成熟研究準備修復拒絕：" + message)


def purpose(row):
    try:
        value = json.loads(row["purpose"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def lineage(permit_sha):
    return {
        "recovery_of": IDENTITY["attempt"],
        "preparation_recovery_permit_sha256": permit_sha,
        "original_input_payload_sha256": IDENTITY["asset_sha256"],
        "original_job_config_sha256": IDENTITY["job_config_sha256"],
        "original_row_sha256": IDENTITY["row_sha256"],
    }


def checked_permit():
    path, original_path = ROOT / PERMIT_NAME, ROOT / ORIGINAL_NAME
    require(path.is_file() and not path.is_symlink(), "缺本人獨立簽核的一次性permit")
    require(original_path.is_file() and not original_path.is_symlink(), "缺保留的真原判")
    require(file_sha(original_path) == ORIGINAL_SHA, "原判已改")
    permit = json.loads(path.read_bytes())
    original = json.loads(original_path.read_bytes())
    require(
        permit.get("schema") == "omni-mature-preparation-recovery-permit-v1"
        and permit.get("status") == "pass_for_exact_preparation_recovery"
        and permit.get("reviewer_task") == original.get("reviewer_task")
        and str(permit.get("reviewer_task", "")).startswith("/root/")
        and permit.get("identity") == IDENTITY
        and permit.get("original_runtime_review") == {"path": ORIGINAL_NAME, "sha256": ORIGINAL_SHA}
        and type(permit.get("maximum_new_attempts")) is int
        and permit.get("maximum_new_attempts") == 1
        and permit.get("new_bound_usd") == "4.30",
        "permit身分、獨立原判、次數或原上界不同",
    )
    semantics = permit.get("source_path_proof", {})
    require(
        semantics.get("sdk_version") == "1.6.0"
        and semantics.get("first_write_failed_before_any_fs_rpc") is True
        and semantics.get("sha_marker_not_sent_by_failed_path") is True
        and semantics.get("provider_historical_NN_telemetry_directly_observed") is False,
        "不能用沒有輸出代替確定的pre-marker來源證明",
    )
    require(isinstance(permit.get("created_at"), str), "缺permit真建立時間")
    created = datetime.fromisoformat(permit["created_at"])
    require(
        created.tzinfo is not None
        and created.utcoffset().total_seconds() == 0
        and datetime.fromisoformat(original["created_at"]) <= created <= datetime.now(UTC),
        "permit必須在真原判之後建立",
    )
    return permit, file_sha(path)


def eligibility(state, job):
    previous = [row for row in state["attempts"] if purpose(row).get("job") == "mature-" + job]
    if not previous:
        return None
    _, permit_sha = checked_permit()
    require(job == "kev" and len(previous) == 1, "一般防重保留，不允許第二次修復或其他工作例外")
    row = previous[0]
    require(
        row["attempt_id"] == IDENTITY["attempt"]
        and digest(row) == IDENTITY["row_sha256"]
        and digest(row["receipt"]) == IDENTITY["receipt_sha256"]
        and row["status"] == "failed-stopped"
        and row["revision"] == IDENTITY["source_revision"]
        and row["reserved_usd"] == "4.30",
        "只接受原failed-stopped紀錄與全額保留，不能改寫或退款",
    )
    receipt = row["receipt"]
    require(
        receipt.get("phase") == "uploading_fixed_input_payload"
        and receipt.get("error_type") == "TypeError"
        and not receipt.get("input_payload_uploaded")
        and not receipt.get("result_exported_before_stop")
        and receipt.get("app_id") == IDENTITY["app_id"]
        and receipt.get("sandbox_id") == IDENTITY["sandbox_id"]
        and receipt.get("observed_owned_sandbox_ids") == [IDENTITY["sandbox_id"]]
        and receipt.get("stopped_sandboxes") == {IDENTITY["sandbox_id"]: 3}
        and receipt.get("remaining_owned_sandboxes") == []
        and receipt.get("stop_errors") == {}
        and receipt.get("compute_stopped") is True
        and receipt.get("resources_verified") is True
        and bool(receipt.get("cleanup_checks"))
        and all(value is True for value in receipt["cleanup_checks"].values()),
        "已知資源、pre-marker路徑或完整停止無法核實",
    )
    require(
        not any(
            purpose(item).get("recovery_of") == IDENTITY["attempt"]
            or purpose(item).get("preparation_recovery_permit_sha256") == permit_sha
            for item in state["attempts"]
        ),
        "此permit已使用，失敗也不再重派",
    )
    release = api(f"releases/{IDENTITY['release_id']}")
    body = (
        "Owned unpublished limited mature-model research; "
        + IDENTITY["attempt"]
        + "; source "
        + IDENTITY["source_revision"]
    )
    require(
        release.get("id") == IDENTITY["release_id"]
        and release.get("draft") is True
        and release.get("tag_name") == IDENTITY["release_tag"]
        and release.get("body") == body
        and release.get("target_commitish") == IDENTITY["source_revision"]
        and len(release.get("assets", [])) == 1,
        "原owned輸入草稿、來源或資產集合已變",
    )
    asset = release["assets"][0]
    require(
        asset.get("id") == IDENTITY["asset_id"]
        and asset.get("name") == IDENTITY["asset_name"]
        and asset.get("size") == IDENTITY["asset_bytes"]
        and asset.get("digest") == "sha256:" + IDENTITY["asset_sha256"],
        "原輸入asset身分／大小／SHA不同",
    )
    return lineage(permit_sha)


def reuse_input(expected_job, output, worker_names, lineage):
    """逐bytes重用原owned封包；新來源只允許改準備層，不重打有mtime的ZIP。"""
    require(lineage.get("recovery_of") == IDENTITY["attempt"], "缺精確lineage")
    with tempfile.TemporaryDirectory(prefix="omni-kev-preparation-recovery-") as temporary:
        command(
            [
                "gh",
                "release",
                "download",
                IDENTITY["release_tag"],
                "--repo",
                REPOSITORY,
                "--pattern",
                IDENTITY["asset_name"],
                "--dir",
                temporary,
            ]
        )
        payload = (Path(temporary) / IDENTITY["asset_name"]).read_bytes()
    require(
        len(payload) == IDENTITY["asset_bytes"] <= MAX_INPUT_BYTES
        and hashlib.sha256(payload).hexdigest() == IDENTITY["asset_sha256"],
        "真取回的原ZIP不同",
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        require(
            len(entries) == len({row.filename for row in entries}) <= 512
            and sum(row.file_size for row in entries) <= MAX_INPUT_BYTES,
            "原ZIP展開超界或重複路徑",
        )
        for row in entries:
            name = PurePosixPath(row.filename)
            require(
                name.as_posix() == row.filename
                and not name.is_absolute()
                and ".." not in name.parts
                and "\\" not in row.filename
                and ":" not in row.filename
                and not stat.S_ISLNK(row.external_attr >> 16),
                "原ZIP不安全路徑",
            )
        raw_config = archive.read("job.json")
        require(hashlib.sha256(raw_config).hexdigest() == IDENTITY["job_config_sha256"], "原job bytes不同")
        job = json.loads(raw_config)
        require({k: v for k, v in job.items() if k != "input_files"} == expected_job, "科學條件或core前置已變")
        require({row.filename for row in entries} == set(job["input_files"]) | {"job.json"}, "原輸入檔集合不完整")
        for name, binding in job["input_files"].items():
            raw = archive.read(name)
            require(
                len(raw) == binding["bytes"] and hashlib.sha256(raw).hexdigest() == binding["sha256"], "原entry不同"
            )
        bundled_workers = {name for name in job["input_files"] if name.startswith(("scripts/", "tiny_perceptron/"))}
        require(bundled_workers == set(worker_names), "worker檔案集合已變")
        for name in worker_names:
            require(name in job["input_files"] and archive.read(name) == (ROOT / name).read_bytes(), "worker來源已改")
        for name in ("data-definition.json", "model-selection.json", "final-opening.json"):
            require(archive.read(name) == (FROZEN / name).read_bytes(), "資料／選模／開封已變")
    for name, raw in (("input.zip", payload), ("job.json", raw_config)):
        with (output / name).open("xb") as stream:
            stream.write(raw)
    json_file(
        output / "input-preparation.json",
        {
            "prepared_before_paid_create": True,
            "model_outputs_used": False,
            "reused_original_owned_input_asset": True,
            "input_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "input_payload_bytes": len(payload),
            **lineage,
        },
    )
    return {**job, "config_sha256": hashlib.sha256(raw_config).hexdigest()}, payload
