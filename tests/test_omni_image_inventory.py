import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from scripts.omni_revision import image_inventory as inventory


def test_direct_image_ids_are_bound_to_existing_course_log():
    document = inventory.candidates(inventory.ROOT / inventory.DEFAULT_CANDIDATES)
    assert len(document["images"]) == 4
    assert len({row["image_id"] for row in document["images"]}) == 4


@pytest.mark.parametrize("field,value", [("primary_source_sha256", "0" * 64), ("exact_source_excerpt", "wrong")])
def test_direct_image_source_tampering_rejected(tmp_path, field, value):
    document = json.loads((inventory.ROOT / inventory.DEFAULT_CANDIDATES).read_bytes())
    document["images"][0][field] = value
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Image"):
        inventory.candidates(path)


class ReadOnlyStub:
    def __init__(self, image_id, unavailable):
        self.image_id = image_id
        self.unavailable = unavailable
        self.calls = []

    async def AppGetLayout(self, request):
        self.calls.append(("AppGetLayout", request.app_id))
        return SimpleNamespace(app_layout=SimpleNamespace(objects=[SimpleNamespace(object_id=self.image_id)]))

    async def ImageFromId(self, request):
        self.calls.append(("ImageFromId", request.image_id))
        if self.unavailable:
            raise RuntimeError("old image unavailable")
        metadata = SimpleNamespace(
            python_version_info="3.11", image_builder_version="fixture", python_packages={"torch": "fixture"}
        )
        return SimpleNamespace(image_id=request.image_id, metadata=metadata)

    def __getattr__(self, name):
        self.calls.append((name, "forbidden"))
        raise AssertionError(f"unexpected RPC: {name}")


@pytest.mark.parametrize("unavailable", [False, True])
def test_direct_and_app_images_deduplicated_without_build_fallback(unavailable):
    document = inventory.candidates(inventory.ROOT / inventory.DEFAULT_CANDIDATES)
    document = copy.deepcopy(document)
    document["apps"] = document["apps"][:1]
    image_id = document["images"][0]["image_id"]
    stub = ReadOnlyStub(image_id, unavailable)
    api = SimpleNamespace(AppGetLayoutRequest=SimpleNamespace, ImageFromIdRequest=SimpleNamespace)
    _, images, count, registry = asyncio.run(inventory.inspect_existing(document, api, SimpleNamespace(_stub=stub)))
    assert count == 4
    assert len(images) == 4
    assert len([call for call in stub.calls if call[0] == "ImageFromId"]) == 4
    assert {call[0] for call in stub.calls} == {"AppGetLayout", "ImageFromId"}
    assert registry["status"] == "not_queried_explicit_or_app_images_available"
    assert {row["status"] for row in images} == ({"unavailable"} if unavailable else {"existing_image_resolved"})
