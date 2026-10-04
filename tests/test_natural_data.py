"""Bounded public transport and unsafe/corrupt archive fixtures, not model tests."""

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import fetch_natural_data as data


def make_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.mtime = 0
            if kind in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            else:
                member.linkname = "../../outside"
                archive.addfile(member)


class Response(io.BytesIO):
    def __init__(self, body, url, *, length=True):
        super().__init__(body)
        self.url = url
        self.headers = {"Content-Length": str(len(body))} if length else {}

    def geturl(self):
        return self.url


@pytest.fixture
def public_fixture(tmp_path):
    source = tmp_path / "public"
    source.mkdir()
    archives, files = [], []
    for archive_path, group in data.ARCHIVES.items():
        content = (group + " structural fixture, not actual training data\n").encode()
        name = group + "/example.txt"
        file = {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        path = source / Path(archive_path).name
        make_archive(path, [(name, content, tarfile.REGTYPE)])
        archives.append(
            {"path": archive_path, "bytes": path.stat().st_size, "sha256": data.sha256(path), "files": [file]}
        )
        files.append(file)
    manifest = {
        "schema_version": 1,
        "dataset_version": "test-fixture-only",
        "archives": archives,
        "files": files,
        "rows": [
            {"id": "image-train", "split": "train", "image": "vision/example.txt"},
            {"id": "ocr-validation", "split": "validation", "image": "ocr/example.txt"},
            {"id": "text-test", "split": "test"},
        ],
        "audio_rows": [{"id": "audio-train", "split": "train", "audio": "speech/example.txt"}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    calls = []

    def opener(request, *, timeout):
        calls.append((request.full_url, dict(request.header_items()), timeout))
        body = (
            path.read_bytes()
            if request.full_url.endswith("/docs/natural-assistant/manifest.json")
            else (source / Path(request.full_url).name).read_bytes()
        )
        return Response(body, request.full_url)

    return source, path, manifest, opener, calls


def run_fetch(fixture, output, *, opener=None):
    _, path, _, default_opener, _ = fixture
    return data.fetch_data(
        path, output, revision="a" * 40, manifest_sha256=data.sha256(path), opener=opener or default_opener
    )


def test_exact_pinned_anonymous_transport_and_verified_atomic_tree(public_fixture, tmp_path):
    _, _, manifest, _, calls = public_fixture
    output = tmp_path / "download"
    result = run_fetch(public_fixture, output)
    assert result["status"] == "downloaded_verified"
    assert result["files_verified"] == 3
    assert result["rows"] == {"train": 1, "validation": 1, "test": 1}
    assert result["audio_rows"] == {"train": 1, "validation": 0, "test": 0}
    assert len(calls) == 4
    assert all("/" + "a" * 40 + "/" in url and timeout == 30 for url, _, timeout in calls)
    assert all(not any(key.lower() == "authorization" for key in headers) for _, headers, _ in calls)
    assert all(record["authentication"] == "none" for record in result["transport"])
    assert data.verify_directory(manifest, output) == output
    assert not list(tmp_path.glob(".natural-data-*"))


def test_existing_exact_data_is_verified_offline_and_modified_data_stays_untouched(public_fixture, tmp_path):
    _, _, _, _, calls = public_fixture
    output = tmp_path / "download"
    run_fetch(public_fixture, output)
    calls.clear()
    assert run_fetch(public_fixture, output)["status"] == "existing_verified"
    assert calls == []
    changed = output / "vision/example.txt"
    changed.write_text("My edited data")
    with pytest.raises(ValueError, match="不會覆蓋"):
        run_fetch(public_fixture, output)
    assert changed.read_text() == "My edited data"
    assert calls == []


def test_extra_files_and_symlink_destination_are_preserved_and_refused(public_fixture, tmp_path):
    _, _, _, _, calls = public_fixture
    output = tmp_path / "download"
    run_fetch(public_fixture, output)
    calls.clear()
    local = output / "my-note.txt"
    local.write_text("Keep this")
    with pytest.raises(ValueError, match="多出"):
        run_fetch(public_fixture, output)
    assert local.read_text() == "Keep this"
    local.unlink()
    link = tmp_path / "link"
    try:
        link.symlink_to(output, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable on this machine")
    with pytest.raises(ValueError, match="連結"):
        run_fetch(public_fixture, link)
    assert calls == []


def test_local_manifest_pin_and_remote_manifest_pin_are_both_required(public_fixture, tmp_path):
    _, path, _, opener, calls = public_fixture
    with pytest.raises(ValueError, match="本機"):
        data.fetch_data(path, tmp_path / "first", revision="a" * 40, manifest_sha256="0" * 64, opener=opener)
    assert calls == []

    def different_remote(request, *, timeout):
        assert request.full_url.endswith("manifest.json")
        body = path.read_bytes()
        return Response(body[:-1] + b" ", request.full_url)

    with pytest.raises(ValueError, match="SHA-256"):
        run_fetch(public_fixture, tmp_path / "second", opener=different_remote)
    assert not (tmp_path / "second").exists()
    assert not list(tmp_path.glob(".natural-data-*"))


def test_mutable_commit_and_inconsistent_archive_allowlist_cannot_start_download(public_fixture, tmp_path):
    _, path, manifest, opener, calls = public_fixture
    with pytest.raises(ValueError, match="Git"):
        data.fetch_data(path, tmp_path / "first", revision="main", manifest_sha256=data.sha256(path), opener=opener)
    bad = copy.deepcopy(manifest)
    bad["files"][0] = {**bad["files"][0], "sha256": "0" * 64}
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="逐檔清單"):
        run_fetch(public_fixture, tmp_path / "second")
    assert calls == []


@pytest.mark.parametrize("change", ["size", "sha", "too-many-bytes", "interruption"])
def test_bounded_transport_failure_removes_all_partial_downloads(public_fixture, tmp_path, change):
    source, path, _, opener, calls = public_fixture
    if change == "sha":
        file = source / "natural-vision-v3.tar.gz"
        content = file.read_bytes()
        file.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    elif change == "size":
        file = source / "natural-vision-v3.tar.gz"
        file.write_bytes(file.read_bytes() + b"x")
    elif change == "too-many-bytes":
        original = opener

        def opener(request, *, timeout):
            if request.full_url.endswith("manifest.json"):
                return original(request, timeout=timeout)
            file = source / Path(request.full_url).name
            return Response(file.read_bytes() + b"x", request.full_url, length=False)
    else:
        original = opener

        def opener(request, *, timeout):
            if request.full_url.endswith("manifest.json"):
                return original(request, timeout=timeout)
            raise OSError("fixture network interruption")

    with pytest.raises((ValueError, OSError)):
        data.fetch_data(
            path, tmp_path / "download", revision="a" * 40, manifest_sha256=data.sha256(path), opener=opener
        )
    assert not (tmp_path / "download").exists()
    assert not list(tmp_path.glob(".natural-data-*"))


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE]
)
def test_links_devices_and_directories_cannot_be_extracted(public_fixture, tmp_path, kind):
    source, _, manifest, _, _ = public_fixture
    specification = copy.deepcopy(manifest["archives"][0])
    archive = source / Path(specification["path"]).name
    name = specification["files"][0]["path"]
    make_archive(archive, [(name, b"", kind)])
    specification.update(bytes=archive.stat().st_size, sha256=data.sha256(archive))
    with pytest.raises(ValueError, match="連結、裝置"):
        data.extract_checked(archive, specification, tmp_path / "output")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "name",
    ["../outside", "/tmp/outside", "vision/../../outside", "vision\\outside", "C:/outside", "vision//example.txt"],
)
def test_tar_paths_cannot_escape_or_alias_destination(public_fixture, tmp_path, name):
    source, _, manifest, _, _ = public_fixture
    specification = copy.deepcopy(manifest["archives"][0])
    archive = source / Path(specification["path"]).name
    make_archive(archive, [(name, b"x", tarfile.REGTYPE)])
    specification.update(bytes=archive.stat().st_size, sha256=data.sha256(archive))
    with pytest.raises(ValueError, match="路徑"):
        data.extract_checked(archive, specification, tmp_path / "output")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("change", ["content", "missing", "duplicate", "extra"])
def test_archive_hash_alone_is_insufficient_without_every_member_allowlist(public_fixture, tmp_path, change):
    source, _, manifest, _, _ = public_fixture
    specification = copy.deepcopy(manifest["archives"][0])
    archive = source / Path(specification["path"]).name
    name = specification["files"][0]["path"]
    original = b"vision structural fixture, not actual training data\n"
    members = [(name, original, tarfile.REGTYPE)]
    if change == "content":
        members = [(name, original[:-1] + b"x", tarfile.REGTYPE)]
    elif change == "missing":
        members = []
    elif change == "duplicate":
        members *= 2
    else:
        members.append(("vision/extra.txt", b"extra", tarfile.REGTYPE))
    make_archive(archive, members)
    specification.update(bytes=archive.stat().st_size, sha256=data.sha256(archive))
    with pytest.raises(ValueError):
        data.extract_checked(archive, specification, tmp_path / "output")


def test_frozen_real_manifest_splits_are_not_all_reported_as_training():
    manifest = data.load_manifest(data.MANIFEST, data.DEFAULT_MANIFEST_SHA256)
    result = data.summary(manifest, data.DEFAULT_REVISION, data.DEFAULT_MANIFEST_SHA256)
    assert result["rows"] == {"train": 272, "validation": 52, "test": 66}
    assert result["audio_rows"] == {"train": 24, "validation": 6, "test": 12}
    assert result["download_bytes"] == 70648031
    assert result["files"] == 345
    assert sum(result["rows"].values()) == 390


def test_manifest_byte_limits_are_checked_before_any_network(public_fixture, tmp_path, monkeypatch):
    _, path, _, opener, calls = public_fixture
    monkeypatch.setattr(data, "MAX_TOTAL_BYTES", 100)
    with pytest.raises(ValueError, match="512 MiB"):
        data.fetch_data(
            path, tmp_path / "download", revision="a" * 40, manifest_sha256=data.sha256(path), opener=opener
        )
    assert calls == []
