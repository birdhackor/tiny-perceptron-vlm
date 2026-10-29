"""唯讀查既有課程App依賴與已建Image，提前核對可沿用的執行環境。"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scripts.omni_revision.finance import POLICY

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = "docs/course-revision-20261029-omni/infrastructure/image-candidates-v1.json"
PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "accelerate",
    "safetensors",
    "huggingface-hub",
    "pillow",
    "qwen-omni-utils",
)


def candidates(path):
    document = json.loads(Path(path).read_bytes())
    if document.get("workspace") != POLICY["workspace"] or document.get("environment") != POLICY["environment"]:
        raise ValueError("Image來源清冊必須是既有birdhackor/main")
    values = document.get("apps")
    if not isinstance(values, list) or not 1 <= len(values) <= 24:
        raise ValueError("明示既有App清單需1至24筆")
    identifiers = [row["app_id"] for row in values]
    if len(set(identifiers)) != len(identifiers) or any(
        not re.fullmatch(r"ap-[A-Za-z0-9]{10,}", x) for x in identifiers
    ):
        raise ValueError("App ID重複或格式不符")
    if any(not row.get("primary_log_sha256") or not row.get("primary_job_url") for row in values):
        raise ValueError("每個既有App需有原始課程job來源")
    return document


async def inspect_existing(document, api, client):
    """只用唯讀RPC；缺失物件照實記錄，沒有建置或建立物件的fallback。"""
    results = []
    image_origins = {}
    for row in document["apps"]:
        app_id = row["app_id"]
        try:
            response = await asyncio.wait_for(client._stub.AppGetLayout(api.AppGetLayoutRequest(app_id=app_id)), 8)
            objects = sorted({item.object_id for item in response.app_layout.objects})
            image_ids = [identifier for identifier in objects if re.fullmatch(r"im-[A-Za-z0-9]{10,}", identifier)]
            for image_id in image_ids:
                image_origins.setdefault(image_id, []).append(app_id)
            results.append({"app_id": app_id, "status": "read", "object_count": len(objects), "image_ids": image_ids})
        except Exception as error:
            results.append(
                {
                    "app_id": app_id,
                    "status": "unavailable",
                    "error_type": type(error).__name__,
                    "grpc_status": getattr(getattr(error, "status", None), "name", None),
                }
            )
    registry = {"status": "not_queried_app_images_available"}
    named_origins = {}
    if not image_origins:
        try:
            response = await asyncio.wait_for(
                client._stub.ImageListTags(
                    api.ImageListTagsRequest(environment_name=POLICY["environment"], max_objects=24)
                ),
                8,
            )
            tags = []
            for item in response.items:
                if re.fullmatch(r"im-[A-Za-z0-9]{10,}", item.image_id):
                    image_origins.setdefault(item.image_id, [])
                    named_origins.setdefault(item.image_id, []).append(item.tag)
                    tags.append({"tag": item.tag, "image_id": item.image_id})
            registry = {"status": "read", "tags": tags, "more_pages_uninspected": bool(response.next_page_token)}
        except Exception as error:
            registry = {
                "status": "unavailable",
                "error_type": type(error).__name__,
                "grpc_status": getattr(getattr(error, "status", None), "name", None),
            }
    images = []
    for image_id in sorted(image_origins)[:24]:
        try:
            response = await asyncio.wait_for(client._stub.ImageFromId(api.ImageFromIdRequest(image_id=image_id)), 8)
            if response.image_id != image_id:
                raise ValueError("ImageFromId回傳不同ID")
            metadata = response.metadata
            images.append(
                {
                    "image_id": image_id,
                    "status": "existing_image_resolved",
                    "source_app_ids": image_origins[image_id],
                    "source_image_tags": named_origins.get(image_id, []),
                    "python_version_info": metadata.python_version_info,
                    "image_builder_version": metadata.image_builder_version,
                    "selected_python_packages": {
                        name: metadata.python_packages[name] for name in PACKAGES if name in metadata.python_packages
                    },
                }
            )
        except Exception as error:
            images.append(
                {
                    "image_id": image_id,
                    "status": "unavailable",
                    "error_type": type(error).__name__,
                    "grpc_status": getattr(getattr(error, "status", None), "name", None),
                }
            )
    return results, images, len(image_origins), registry


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidates", type=Path, default=ROOT / DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("MODAL_ENVIRONMENT", "main") != POLICY["environment"]:
        raise ValueError("必須使用既有main environment")
    document = candidates(args.candidates)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {}
    for path in [
        Path(__file__),
        args.candidates,
        ROOT / ".github/workflows/omni-revision.yml",
        ROOT / "scripts/omni_revision/finance.py",
    ]:
        name = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        if subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT) != raw:
            raise ValueError("唯讀清冊工具／來源必須與實際commit一致")
        sources[name] = hashlib.sha256(raw).hexdigest()
    import modal
    from google.protobuf.empty_pb2 import Empty
    from modal.client import _Client
    from modal_proto import api_pb2

    if modal.__version__ != "1.6.0":
        raise ValueError("唯讀RPC介面核對版本必須modal==1.6.0")

    async def run():
        client = await _Client.from_env()
        try:
            workspace = await asyncio.wait_for(client._stub.WorkspaceNameLookup(Empty()), 8)
            if workspace.username != POLICY["workspace"]:
                raise ValueError("認證workspace不是既有birdhackor")
            return await inspect_existing(document, api_pb2, client)
        finally:
            await client._close()

    apps, images, total, registry = asyncio.run(run())
    receipt = {
        "schema_version": 1,
        "kind": "actual_client_only_existing_image_inventory",
        "observed_at": datetime.now(UTC).isoformat(),
        "workspace": POLICY["workspace"],
        "environment": POLICY["environment"],
        "source_revision": revision,
        "modal_sdk_version": modal.__version__,
        "source_files": sources,
        "rpc_allowlist": ["WorkspaceNameLookup", "AppGetLayout", "ImageListTags", "ImageFromId"],
        "apps": apps,
        "images": images,
        "named_image_registry": registry,
        "image_dependencies_found": total,
        "image_dependency_limit": 24,
        "remote_compute_started": False,
        "image_created_or_built": False,
        "limitations": [
            "Image存在與packages metadata不是GPU可用、CUDA初始化、模型載入或訓練成功。",
            "不存在的舊App/Image照實記unavailable，沒有建置或執行fallback。",
            "後續任何付費邊界仍須新鮮官方billing/inventory、完整硬上界及durable CAS reserve。",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "image-inventory.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "apps_checked": len(apps),
                "existing_images_resolved": sum(x["status"] == "existing_image_resolved" for x in images),
                "remote_compute_started": False,
            }
        )
    )


if __name__ == "__main__":
    main()
