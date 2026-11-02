"""發布保護的離線負例；不把模擬 API 當成真實傳輸證據。"""

import hashlib
import json
import shutil

import pytest

from scripts import publish_c_v3_release as release


def test_corrupt_package_stops_before_any_network(tmp_path, monkeypatch):
    package = tmp_path / "model.zip"
    package.write_bytes(b"altered")
    monkeypatch.setattr(release, "gh", lambda *args: pytest.fail("不得呼叫網路"))
    with pytest.raises(ValueError, match="SHA-256"):
        release.verify_file(package)


def test_exact_file_identity_includes_size(tmp_path):
    package = tmp_path / "fixed.zip"
    package.write_bytes(b"fixed")
    digest = hashlib.sha256(b"fixed").hexdigest()
    release.verify_file(package, 5, digest)
    with pytest.raises(ValueError):
        release.verify_file(package, 6, digest)
    package.write_bytes(b"other")
    with pytest.raises(ValueError):
        release.verify_file(package, 5, digest)


def test_existing_asset_collision_and_public_retarget_are_rejected():
    revision = "a" * 40
    draft = {"id": release.RELEASE_ID, "tag_name": release.TAG, "draft": True, "assets": []}
    assert release.checked_release(draft, revision) is None
    asset = {"id": 12, "name": release.NAME, "size": release.SIZE}
    assert release.checked_release({**draft, "assets": [asset]}, revision) == asset
    for bad in (
        {**draft, "id": 123},
        {**draft, "tag_name": "another-model"},
        {**draft, "assets": [{**asset, "size": 1}]},
        {**draft, "assets": [asset, asset]},
        {**draft, "draft": False, "target_commitish": "b" * 40},
    ):
        with pytest.raises(ValueError):
            release.checked_release(bad, revision)


def test_local_process_cannot_publish(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(release, "gh", lambda *args: pytest.fail("不得呼叫網路"))
    with pytest.raises(ValueError, match="Actions runner"):
        release.publish()


def test_tag_checks_accept_only_known_missing_draft_or_matching_commit():
    revision = "a" * 40
    good = {"ref": f"refs/tags/{release.TAG}", "object": {"type": "commit", "sha": revision}}
    assert release.check_tag_response(404, {}, revision, True) is False
    assert release.check_tag_response(200, good, revision, True) is True
    assert release.check_tag_response(200, good, revision, False) is True
    for status, reference, draft in (
        (404, {}, False),
        (403, {}, True),
        (500, {}, True),
        (200, {**good, "object": {"type": "commit", "sha": "b" * 40}}, True),
        (200, {**good, "object": {"type": "tag", "sha": revision}}, True),
        (200, {}, True),
    ):
        with pytest.raises(ValueError):
            release.check_tag_response(status, reference, revision, draft)


def test_qualified_descriptor_cannot_drop_an_implementation_even_with_drift(tmp_path):
    source = release.ROOT / "docs/decision-product/v3/public-release.json"
    descriptor = json.loads(source.read_bytes())
    path = tmp_path / "public-release.json"
    path.write_bytes(source.read_bytes())
    assert len(descriptor["implementation"]) == 17
    for name in descriptor["implementation"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(release.ROOT / name, target)
    assert release.verify_implementation(tmp_path, path) == descriptor
    service = tmp_path / "tiny_perceptron/decision_product/service.py"
    service.write_bytes(service.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="實作"):
        release.verify_implementation(tmp_path, path)
    for mapping in (
        {},
        {name: digest for name, digest in descriptor["implementation"].items() if not name.endswith("/service.py")},
    ):
        path.write_text(json.dumps({**descriptor, "implementation": mapping}))
        with pytest.raises(ValueError, match="完整17檔"):
            release.verify_implementation(tmp_path, path)
