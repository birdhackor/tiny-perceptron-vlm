"""發布保護的離線負例；不把模擬 API 當成真實傳輸證據。"""

import hashlib

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
