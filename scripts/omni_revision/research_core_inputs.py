"""成熟研究前先取回真核心成品，重用嚴格selection入口；只載入、不做NN預測。"""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import PurePosixPath

import torch

from scripts.omni_revision import core_final
from scripts.omni_revision.formal_local import FROZEN, ROOT
from scripts.omni_revision.github_transport_probe import REPOSITORY, api, command
from tiny_perceptron.omni_products.training_runtime import file_sha


def check_metadata(revision):
    filenames = (
        "data-definition.json",
        "metrics.json",
        "formal-recipes.json",
        "codec-selection.json",
        "model-selection.json",
        "final-opening.json",
    )
    for path in [*(FROZEN / name for name in filenames), core_final.AUDIT]:
        name = path.relative_to(ROOT).as_posix()
        if (
            path.is_symlink()
            or subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT) != path.read_bytes()
        ):
            raise ValueError("成熟研究核心前置須先逐bytes提交")
    selected = json.loads((FROZEN / "model-selection.json").read_bytes())
    required = {
        "schema",
        "created_at",
        "final_opened",
        "formal_recipe_sha256",
        "data_definition_sha256",
        "metrics_sha256",
        "implementation_sha256",
        "selected",
        "model_files",
        "all_formal_jobs",
        "temperature",
        "policy_feasible",
        "threshold",
        "selection_evidence",
        "sound_map",
        "codec_selection_sha256",
    }
    if (
        not required <= selected.keys()
        or selected["schema"] != "omni-core-model-selection-v1"
        or selected["final_opened"] is not False
    ):
        raise ValueError("缺完整核心selection，不能以摘要或空bindings開成熟研究")
    sources = core_final.checked_audit()
    audit = json.loads(core_final.AUDIT.read_bytes())
    if re.fullmatch(r"/root/[A-Za-z0-9_/.-]+", audit.get("reviewer_task", "")) is None:
        raise ValueError("核心final audit必須是真正非空獨立reviewer")
    opened = json.loads((FROZEN / "final-opening.json").read_bytes())
    fixed = {
        "schema": "omni-core-final-opening-v1",
        "model_selection_sha256": file_sha(FROZEN / "model-selection.json"),
        "data_definition_sha256": file_sha(FROZEN / "data-definition.json"),
        "metrics_sha256": file_sha(FROZEN / "metrics.json"),
        "evaluation_sources_sha256": sources,
        "independent_audit_sha256": file_sha(core_final.AUDIT),
        "final_opened": True,
    }
    if set(opened) != {*fixed, "created_at", "runtime"} or any(opened.get(k) != v for k, v in fixed.items()):
        raise ValueError("核心opening schema／資料／選擇／完整來源／獨立audit不符")
    runtime = opened["runtime"]
    if (
        set(runtime) != {"device", "torch", "python"}
        or runtime["device"] != "cpu"
        or runtime["torch"] != str(torch.__version__)
        or re.fullmatch(r"3\.13\.[0-9]+", runtime["python"]) is None
    ):
        raise ValueError("核心opening不是已固定的CPU/Torch/Python backend")
    when = datetime.fromisoformat(opened["created_at"])
    if when.tzinfo is None:
        raise ValueError("核心opening時間需有真實時區")
    return {
        "core_selection_sha256": fixed["model_selection_sha256"],
        "core_final_opening_sha256": file_sha(FROZEN / "final-opening.json"),
        "files": {name: file_sha(FROZEN / name) for name in filenames},
    }


def completed_core(state):
    rows = [
        row
        for row in state["attempts"]
        if row["purpose"].startswith("{") and json.loads(row["purpose"]).get("job") == "cpu-core-formal"
    ]
    if len(rows) != 1:
        raise ValueError("只有一個實際核心正式嘗試")
    row = rows[0]
    receipt = row.get("receipt", {})
    if (
        row["status"] != "completed"
        or receipt.get("success") is not True
        or receipt.get("all_training_processes_exited") is not True
        or receipt.get("no_modal_compute_created") is not True
    ):
        raise ValueError("核心工作沒有完整完成與保存")
    if json.loads(row["purpose"])["formal_recipe_sha256"] != file_sha(FROZEN / "formal-recipes.json"):
        raise ValueError("核心耐久帳與本次recipe不同")
    tag = "omni-core-candidate-" + row["attempt_id"]
    body = "Owned unpublished core weights and full optimizer/RNG; " + row["attempt_id"] + "; source " + row["revision"]
    owned = api(f"releases/{receipt['release_id']}")
    if (
        receipt.get("release_tag") != tag
        or owned.get("draft") is not True
        or owned.get("tag_name") != tag
        or owned.get("body") != body
        or owned.get("target_commitish") != row["revision"]
    ):
        raise ValueError("核心完整產物不是本輪精確owned draft")
    if {(x["id"], x["name"], x["size"]) for x in owned["assets"]} != {
        (x["id"], x["name"], x["bytes"]) for x in receipt["assets"]
    }:
        raise ValueError("核心provider資產與耐久帳不同")
    assets = [x for x in receipt["assets"] if x["name"] == "core-checkpoints.zip"]
    if (
        len(assets) != 1
        or assets[0].get("roundtrip_verified") is not True
        or not 0 < assets[0]["bytes"] <= 128 * 1024**2
        or re.fullmatch("[0-9a-f]{64}", assets[0]["sha256"]) is None
    ):
        raise ValueError("缺有界完整core ZIP及真roundtrip來源")
    return tag, assets[0]


def unpack_verified(path, destination):
    with zipfile.ZipFile(path) as archive:
        files = archive.infolist()
        names = {row.filename for row in files}
        if len(names) != len(files) or len(files) > 512 or sum(row.file_size for row in files) > 256 * 1024**2:
            raise ValueError("完整核心封包展開超界或重複路徑")
        for row in files:
            name = PurePosixPath(row.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in row.filename
                or (row.external_attr >> 16) & 0o170000 == 0o120000
                or (
                    row.filename != "package-file-manifest.json"
                    and name.parts[0] not in ("core", "frozen", "selection-evidence")
                )
            ):
                raise ValueError("完整核心封包含不允許路徑")
        manifest = json.loads(archive.read("package-file-manifest.json"))
        if set(manifest) != names - {"package-file-manifest.json"}:
            raise ValueError("完整核心逐檔manifest缺漏")
        for name, binding in manifest.items():
            raw = archive.read(name)
            if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
                raise ValueError("完整核心逐檔SHA／大小不符")
        archive.extractall(destination)


def checked_core(revision, state):
    identity = check_metadata(revision)
    tag, asset = completed_core(state)
    with tempfile.TemporaryDirectory(prefix="omni-mature-core-prerequisite-") as temporary:
        from pathlib import Path

        folder = Path(temporary)
        command(
            ["gh", "release", "download", tag, "--repo", REPOSITORY, "--pattern", asset["name"], "--dir", str(folder)]
        )
        package = folder / asset["name"]
        if package.stat().st_size != asset["bytes"] or file_sha(package) != asset["sha256"]:
            raise ValueError("實際取回完整核心成品與預定SHA不同")
        unpack_verified(package, folder / "verified")
        extracted = folder / "verified"
        for name in (
            "data-definition.json",
            "metrics.json",
            "formal-recipes.json",
            "codec-selection.json",
            "model-selection.json",
        ):
            if (extracted / "frozen" / name).read_bytes() != (FROZEN / name).read_bytes():
                raise ValueError("已提交核心manifest不是實際訓練包中的原選擇")
        evidence = ROOT / "outputs/omni-core-selection-v1"
        if evidence.exists():
            raise ValueError("研究checkout不覆寫既有core selection evidence")
        shutil.copytree(extracted / "selection-evidence", evidence)
        # 與真正開final共用完整10job/5bindings/T/tau/map/continuation/cal/policy檢查。
        # 此呼叫只load safetensors及讀原始receipts，不執行NN預測。
        core_final.checked_selection(extracted / "core", identity["core_selection_sha256"])
    return {
        **identity,
        "core_prerequisite_asset": {key: asset[key] for key in ("id", "name", "bytes", "sha256")},
        "complete_core_revalidated_without_predictions": True,
    }
