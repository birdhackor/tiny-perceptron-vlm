"""V4 fixed-byte downloader fixtures, without real HTTP or model inference."""

import copy
import gzip
import hashlib
import io
import itertools
import json
import tarfile
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from scripts import fetch_natural_data as data

REVISION = "b" * 40
V4_MANIFEST_PATH = "docs/natural-assistant/v4/manifest.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode()


def archive_bytes(members):
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, raw in sorted(members.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mtime, member.mode = len(raw), 0, 0o644
                archive.addfile(member, io.BytesIO(raw))
    return output.getvalue()


class Response(io.BytesIO):
    def __init__(self, raw, url, *, length=True):
        super().__init__(raw)
        self.url = url
        self.headers = {"Content-Length": str(len(raw))} if length else {}

    def geturl(self):
        return self.url


@pytest.fixture
def v4_public(tmp_path):
    members = {
        "vision": {"vision/photo.jpg": b"fixed v4 photo fixture", "vision/NOTICE.txt": b"DOCCI CC-BY-4.0 attribution"},
        "ocr": {"ocr/crop.png": b"fixed v4 crop fixture", "ocr/NOTICE.txt": b"NVIDIA and Commons attribution"},
        "voice": {
            "voice/asr.wav": b"fixed original ASR WAV fixture",
            "voice/question.wav": b"fixed original human-read question WAV fixture",
            "voice/NOTICE.txt": b"FLEURS CC-BY-4.0; AISHELL human read questions, not spontaneous conversation",
            "voice/AISHELL-Apache-2.0-LICENSE.txt": b"Apache License, Version 2.0 fixture",
        },
    }
    archives, files, responses = [], [], {}
    for archive_path, group in data.V4_ARCHIVES.items():
        body = archive_bytes(members[group])
        declared = [
            {"path": name, "bytes": len(raw), "sha256": digest(raw)} for name, raw in sorted(members[group].items())
        ]
        archives.append({"path": archive_path, "bytes": len(body), "sha256": digest(body), "files": declared})
        files.extend(declared)
        responses[Path(archive_path).name] = body
    manifest = {
        "schema_version": 1,
        "dataset_version": "natural-assistant-v4",
        "archives": archives,
        "files": files,
        "rows": [
            {"id": "scene", "split": "train", "image": "vision/photo.jpg"},
            {"id": "ocr", "split": "validation", "image": "ocr/crop.png"},
            {"id": "chat", "split": "test"},
        ],
        "audio_rows": [
            {"id": "fleurs-asr", "split": "test", "task": "speech_transcription", "audio": "voice/asr.wav"},
            {"id": "aishell-question", "split": "validation", "task": "speech_chat", "audio": "voice/question.wav"},
        ],
    }
    path = tmp_path / "v4-manifest.json"
    path.write_bytes(canonical(manifest))
    calls = []

    def opener(request, *, timeout):
        calls.append({"url": request.full_url, "headers": dict(request.header_items()), "timeout": timeout})
        parsed = urlsplit(request.full_url)
        if parsed.hostname == "raw.githubusercontent.com":
            assert parsed.path == f"/{data.REPOSITORY}/{REVISION}/{V4_MANIFEST_PATH}"
            raw = path.read_bytes()
        else:
            assert parsed.hostname == "media.githubusercontent.com"
            raw = responses[Path(parsed.path).name]
        return Response(raw, request.full_url)

    return {
        "path": path,
        "manifest": manifest,
        "responses": responses,
        "members": members,
        "opener": opener,
        "calls": calls,
    }


def fetch(fixture, destination, opener=None):
    return data.fetch_data(
        fixture["path"],
        destination,
        revision=REVISION,
        manifest_sha256=data.sha256(fixture["path"]),
        opener=opener or fixture["opener"],
    )


def test_v4_fixed_sha_anonymous_request_mock_extracts_exact_member_bytes(v4_public, tmp_path):
    destination = tmp_path / "download"
    result = fetch(v4_public, destination)
    assert result["status"] == "downloaded_verified"
    assert result["dataset_version"] == "natural-assistant-v4" and result["git_revision"] == REVISION
    assert result["manifest_sha256"] == data.sha256(v4_public["path"])
    assert result["files_verified"] == 8
    assert result["rows"] == {"train": 1, "validation": 1, "test": 1}
    assert result["audio_rows"] == {"train": 0, "validation": 1, "test": 1}
    assert [archive["id"] for archive in result["archives"]] == ["vision", "ocr", "voice"]
    assert len(v4_public["calls"]) == 4
    for call in v4_public["calls"]:
        parsed = urlsplit(call["url"])
        assert parsed.scheme == "https" and not parsed.query and not parsed.fragment
        assert f"/{REVISION}/" in parsed.path and call["timeout"] == 30
        assert not {key.lower() for key in call["headers"]} & {"authorization", "cookie", "proxy-authorization"}
        assert call["headers"]["Accept-encoding"] == "identity"
    assert result["remote_manifest"]["url"].endswith("/" + V4_MANIFEST_PATH)
    assert result["remote_manifest"]["authentication"] == "none"
    for record, expected in zip(result["transport"], v4_public["manifest"]["archives"], strict=True):
        assert record["authentication"] == "none"
        assert record["bytes"] == expected["bytes"] and record["sha256"] == expected["sha256"]
        assert record["url"].endswith("/" + expected["path"])
    for group in v4_public["members"].values():
        for name, raw in group.items():
            assert (destination / name).read_bytes() == raw
    assert data.verify_directory(v4_public["manifest"], destination) == destination
    assert not list(tmp_path.glob(".natural-data-*"))


def test_v4_existing_verified_tree_is_offline_and_preserves_changed_bytes(v4_public, tmp_path):
    destination = tmp_path / "download"
    fetch(v4_public, destination)
    v4_public["calls"].clear()
    assert fetch(v4_public, destination)["status"] == "existing_verified"
    assert v4_public["calls"] == []
    recording = destination / "voice/asr.wav"
    recording.write_bytes(b"user modified recording")
    with pytest.raises(ValueError):
        fetch(v4_public, destination)
    assert recording.read_bytes() == b"user modified recording" and v4_public["calls"] == []


@pytest.mark.parametrize(
    "v3_groups", [bits for bits in itertools.product((False, True), repeat=3) if len(set(bits)) == 2]
)
def test_all_v3_v4_mixed_archive_combinations_are_rejected_before_http(v4_public, tmp_path, v3_groups):
    manifest = copy.deepcopy(v4_public["manifest"])
    for index, use_v3 in enumerate(v3_groups):
        if use_v3:
            manifest["archives"][index]["path"] = list(data.ARCHIVES)[index]
            if index == 2:
                for declaration in manifest["archives"][index]["files"]:
                    declaration["path"] = declaration["path"].replace("voice/", "speech/", 1)
                for declaration in manifest["files"]:
                    if declaration["path"].startswith("voice/"):
                        declaration["path"] = declaration["path"].replace("voice/", "speech/", 1)
                for row in manifest["audio_rows"]:
                    row["audio"] = row["audio"].replace("voice/", "speech/", 1)
    v4_public["path"].write_bytes(canonical(manifest))
    with pytest.raises(ValueError, match="同版本"):
        fetch(v4_public, tmp_path / "download")
    assert v4_public["calls"] == [] and not (tmp_path / "download").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "unlisted"])
def test_v4_requires_exactly_the_three_declared_archive_paths(v4_public, tmp_path, change):
    manifest = copy.deepcopy(v4_public["manifest"])
    if change == "missing":
        manifest["archives"].pop()
    elif change == "duplicate":
        manifest["archives"][1] = manifest["archives"][0]
    else:
        manifest["archives"][0]["path"] = "assets/training/natural-vision-v5.tar.gz"
    v4_public["path"].write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        fetch(v4_public, tmp_path / "download")
    assert v4_public["calls"] == []


@pytest.mark.parametrize("change", ["remote-manifest-sha", "archive-sha", "too-many-bytes"])
def test_v4_changed_remote_bytes_are_rejected_with_no_installed_or_partial_tree(v4_public, tmp_path, change):
    if change == "archive-sha":
        name = "natural-voice-v4.tar.gz"
        body = v4_public["responses"][name]
        v4_public["responses"][name] = body[:-1] + bytes([body[-1] ^ 1])

    def opener(request, *, timeout):
        response = v4_public["opener"](request, timeout=timeout)
        body = response.read()
        if change == "remote-manifest-sha" and "raw.githubusercontent.com" in request.full_url:
            body = body[:-1] + b" "
        elif change == "too-many-bytes" and request.full_url.endswith("natural-voice-v4.tar.gz"):
            return Response(body + b"x", request.full_url, length=False)
        return Response(body, request.full_url)

    with pytest.raises(ValueError):
        fetch(v4_public, tmp_path / "download", opener=opener)
    assert not (tmp_path / "download").exists() and not list(tmp_path.glob(".natural-data-*"))


def test_v4_manifest_above_old_4mib_limit_is_bounded_and_sha_checked(v4_public, monkeypatch):
    manifest = copy.deepcopy(v4_public["manifest"])
    manifest["attribution_fixture"] = "x" * (4 * 1024 * 1024)
    v4_public["path"].write_bytes(canonical(manifest))
    assert 4 * 1024 * 1024 < v4_public["path"].stat().st_size < data.MAX_MANIFEST_BYTES
    assert (
        data.load_manifest(v4_public["path"], data.sha256(v4_public["path"]))["dataset_version"]
        == "natural-assistant-v4"
    )
    monkeypatch.setattr(data, "MAX_MANIFEST_BYTES", v4_public["path"].stat().st_size - 1)
    with pytest.raises(ValueError):
        data.load_manifest(v4_public["path"], data.sha256(v4_public["path"]))
    assert v4_public["calls"] == []
