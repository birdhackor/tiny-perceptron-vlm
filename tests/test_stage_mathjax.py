"""資產下載或快取受損時，網站不可發布不同版本或不完整的公式程式。"""

import hashlib
import io
import json

import pytest

from scripts.stage_mathjax import prepare_assets, stage_assets, validate_manifest


@pytest.fixture
def assets(tmp_path):
    names = ["LICENSE", "tex-chtml.js", "output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff"]
    files = []
    cache = tmp_path / "outputs/mathjax-cache/3.2.2"
    for name in names:
        raw = ("test-only:" + name).encode()
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files.append(
            {
                "path": name,
                "upstream_path": name if name == "LICENSE" else "es5/" + name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    value = {
        "schema_version": 1,
        "name": "MathJax",
        "version": "3.2.2",
        "repository": "mathjax/MathJax",
        "license": "Apache-2.0",
        "revision": "6" * 40,
        "files": files,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(value))
    return tmp_path, manifest, cache, value


def test_complete_verified_cache_builds_without_network(assets, monkeypatch):
    root, manifest, cache, value = assets

    def no_network(*args, **kwargs):
        raise AssertionError("完整快取不應連網")

    monkeypatch.setattr("scripts.stage_mathjax.urllib.request.urlopen", no_network)
    prepared = prepare_assets(root, manifest)
    destination = root / "site/javascripts/mathjax"
    stage_assets(destination, prepared)
    for item in value["files"]:
        assert (destination / item["path"]).read_bytes() == (cache / item["path"]).read_bytes()


def test_corrupt_existing_cache_is_rejected(assets):
    root, manifest, cache, _ = assets
    (cache / "tex-chtml.js").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="指紋不符"):
        prepare_assets(root, manifest)


@pytest.mark.parametrize("correct", [True, False])
def test_missing_asset_is_pinned_and_verified_before_cache_write(assets, monkeypatch, correct):
    root, manifest, cache, value = assets
    target = cache / "tex-chtml.js"
    original = target.read_bytes()
    target.unlink()
    urls = []

    def remote(url, timeout):
        urls.append(url)
        assert timeout == 30
        return io.BytesIO(original if correct else b"wrong upstream bytes")

    monkeypatch.setattr("scripts.stage_mathjax.urllib.request.urlopen", remote)
    if correct:
        prepare_assets(root, manifest)
        assert target.read_bytes() == original
    else:
        with pytest.raises(ValueError, match="指紋不符"):
            prepare_assets(root, manifest)
        assert not target.exists()
    assert urls == ["https://raw.githubusercontent.com/mathjax/MathJax/" + value["revision"] + "/es5/tex-chtml.js"]


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "a\\b", "."])
def test_manifest_cannot_write_outside_asset_tree(assets, path):
    _, _, _, value = assets
    value["files"][0]["path"] = path
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_duplicate_paths_cannot_hide_missing_assets(assets):
    _, _, _, value = assets
    value["files"][1] = dict(value["files"][0])
    with pytest.raises(ValueError):
        validate_manifest(value)


@pytest.mark.parametrize("link_kind", ["file", "parent", "destination"])
def test_stage_rejects_symlinks_without_changing_external_files(assets, link_kind):
    root, manifest, _, _ = assets
    prepared = prepare_assets(root, manifest)
    destination = root / "site/mathjax"
    outside = root / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"must remain unchanged")
    destination.parent.mkdir(parents=True)
    if link_kind == "destination":
        destination.symlink_to(outside, target_is_directory=True)
    else:
        destination.mkdir()
        if link_kind == "file":
            (destination / "LICENSE").symlink_to(sentinel)
        else:
            (destination / "output").symlink_to(outside, target_is_directory=True)
    before = {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="symlink"):
        stage_assets(destination, prepared)
    after = {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()}
    assert after == before


def test_stage_rechecks_source_after_prepare(assets):
    root, manifest, cache, _ = assets
    prepared = prepare_assets(root, manifest)
    (cache / "tex-chtml.js").write_bytes(b"changed after prepare")
    with pytest.raises(ValueError, match="指紋不符"):
        stage_assets(root / "site/mathjax", prepared)
