"""Frozen v4 asset contracts; local byte fixtures only, without model inference."""

import copy
import hashlib
import importlib.util
import io
import json
import os
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

import PIL
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "docs/natural-assistant/v4/data"
DATA = ROOT / "outputs/natural-v4/data"
FLEURS_URL = (
    "https://huggingface.co/datasets/google/fleurs/resolve/"
    "d7c758a6dceecd54a98cac43404d3d576e721f07/data/cmn_hans_cn/audio/test.tar.gz"
)


def load_script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = load_script("build_natural_v4_assets")
ocr = load_script("prepare_natural_v4_ocr")
voice = load_script("replay_natural_v4_voice")


def fingerprint(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class Response(io.BytesIO):
    def __init__(self, raw, *, status=200, headers=None):
        super().__init__(raw)
        self.status = status
        self.headers = headers or {}


def wav_bytes():
    output = io.BytesIO()
    sf.write(output, [0.0, 0.125, -0.125] * 64, 16000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def tar_bytes(members):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, raw in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return output.getvalue()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    metadata = root / "docs/natural-assistant/v4/data"
    data = root / "outputs/natural-v4/data"
    monkeypatch.setattr(assets, "ROOT", root)
    monkeypatch.setattr(assets, "METADATA", metadata)
    raw = b"a fixed public photo fixture\n"
    photo = {
        "image": "vision/photo.jpg",
        "family": "photo-family",
        "split": "validation",
        "url": "https://storage.googleapis.com/docci/thumbnails/fixture.jpg?generation=1",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        **fingerprint(raw),
    }
    row = {
        "id": "photo-question",
        "family": photo["family"],
        "split": photo["split"],
        "task": "scene",
        "image": photo["image"],
        "user": "照片內容？",
        "answer": "測試照片",
        "history": [],
    }
    write_json(metadata / "vision-sources.json", {"rows": [photo]})
    write_json(metadata / "vision-activity-sources.json", {"rows": []})
    write_rows(metadata / "vision-labels/validation.jsonl", [row])
    write_rows(metadata / "chat-labels.jsonl", [])
    write_rows(metadata / "ocr-labels.jsonl", [])
    write_rows(metadata / "presence-root-labels.jsonl", [])
    write_json(metadata / "ocr-sources.json", {"sources": [], "artifacts": []})
    write_json(metadata / "chat-sources.json", {})
    for name in ("voice-sources.json", "voice-question-sources.json"):
        document = json.loads((METADATA / name).read_text())
        document["audio_rows"] = []
        write_json(metadata / name, document)
    license_source = ROOT / "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE"
    license_target = root / "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE"
    license_target.parent.mkdir(parents=True)
    license_target.write_bytes(license_source.read_bytes())
    target = data / photo["image"]
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    return {"root": root, "metadata": metadata, "data": data, "photo": photo, "row": row, "raw": raw}


def build_fixture(frozen):
    return assets.build(frozen["data"], frozen["root"] / "assets/training", frozen["root"] / "manifest.json")


def order_label(frozen):
    source = {
        "source_id": "heldout-photo",
        "dataset": "Wikimedia Commons individually licensed photographs",
        "family": frozen["photo"]["family"],
        "split": frozen["photo"]["split"],
        "image_relative": frozen["photo"]["image"],
        "image_bytes": frozen["photo"]["bytes"],
        "image_sha256": frozen["photo"]["sha256"],
    }
    artifact = {
        "path": source["image_relative"],
        "kind": "source_image",
        "source_id": source["source_id"],
        **fingerprint(frozen["raw"]),
    }
    write_json(frozen["metadata"] / "ocr-sources.json", {"sources": [source], "artifacts": [artifact]})
    return {
        "id": "heldout-photo-two-lines",
        "source_id": source["source_id"],
        "family": source["family"],
        "split": source["split"],
        "task": "ocr_transcription",
        "question": "請依先上後下讀出兩行。",
        "answer": "第一行\n第二行",
        "image_relative": source["image_relative"],
        "image_sha256": source["image_sha256"],
    }


def test_optional_ocr_order_source_can_be_absent_and_all_raw_annotation_files_are_declared(frozen):
    manifest = build_fixture(frozen)
    assert not (frozen["metadata"] / "ocr-order-labels.jsonl").exists()
    paths = {item["path"] for item in manifest["annotation_sources"]}
    expected = {str(path.relative_to(frozen["root"])) for path in frozen["metadata"].rglob("*.jsonl")}
    assert paths == expected
    assert not any(item["path"].endswith("ocr-order-labels.jsonl") for item in manifest["annotation_sources"])
    for item in manifest["annotation_sources"]:
        raw = (frozen["root"] / item["path"]).read_bytes()
        assert {key: item[key] for key in ("bytes", "sha256")} == fingerprint(raw)


def test_optional_ocr_order_source_preserves_multiline_reference_family_and_raw_sha(frozen):
    label = order_label(frozen)
    path = frozen["metadata"] / "ocr-order-labels.jsonl"
    write_rows(path, [label])
    manifest = build_fixture(frozen)
    row = next(row for row in manifest["rows"] if row["id"] == "ocr-v4:" + label["id"])
    assert row["task"] == row["references"]["kind"] == "ocr_order"
    assert row["answer"] == row["references"]["text"] == "第一行\n第二行"
    assert row["references"]["strip_whitespace"] is False
    assert (row["family"], row["split"]) == (label["family"], label["split"])
    declaration = next(
        item for item in manifest["annotation_sources"] if item["path"].endswith("ocr-order-labels.jsonl")
    )
    assert declaration == {"path": str(path.relative_to(frozen["root"])), **fingerprint(path.read_bytes())}


def test_raw_optional_annotation_layout_changes_sha_even_when_rows_are_identical(frozen):
    path = frozen["metadata"] / "ocr-order-labels.jsonl"
    write_rows(path, [order_label(frozen)])
    first = build_fixture(frozen)
    raw = path.read_bytes()
    path.write_bytes(raw + b"\n")  # The reader ignores blank lines; raw source provenance must still change.
    second = build_fixture(frozen)
    assert first["rows"] == second["rows"]
    assert first["annotation_sources"] != second["annotation_sources"]
    declaration = next(item for item in second["annotation_sources"] if item["path"].endswith("ocr-order-labels.jsonl"))
    assert {key: declaration[key] for key in ("bytes", "sha256")} == fingerprint(raw + b"\n")


def test_optional_ocr_order_source_cannot_duplicate_existing_annotation_ids(frozen):
    label = order_label(frozen)
    write_rows(frozen["metadata"] / "ocr-labels.jsonl", [label])
    write_rows(frozen["metadata"] / "ocr-order-labels.jsonl", [label])
    with pytest.raises(ValueError, match="Repeated"):
        assets.assemble()


def test_optional_ocr_order_source_cannot_change_its_frozen_photograph_split(frozen):
    label = dict(order_label(frozen), family="unrelated-new-family", split="test")
    write_rows(frozen["metadata"] / "ocr-order-labels.jsonl", [label])
    with pytest.raises(ValueError):
        build_fixture(frozen)


def test_actual_optional_ocr_order_rows_bind_frozen_sources_and_preserve_line_order():
    path = METADATA / "ocr-order-labels.jsonl"
    if not path.is_file():
        pytest.skip("optional reviewed OCR order rows are absent")
    original_bytes = path.read_bytes()
    labels = assets.json_lines(path)
    assert labels
    spec = json.loads((METADATA / "ocr-sources.json").read_text())
    sources = {source["source_id"]: source for source in spec["sources"]}
    artifacts = {artifact["path"]: artifact for artifact in spec["artifacts"]}
    assembled, _ = assets.assemble()
    assembled_ids = Counter(row["id"] for row in assembled)
    assembled = {row["id"]: row for row in assembled}
    for label in labels:
        source = sources[label["source_id"]]
        artifact = artifacts[label["image_relative"]]
        assert artifact["source_id"] == label["source_id"]
        assert (label["family"], label["split"]) == (source["family"], source["split"])
        assert label["image_sha256"] == artifact["sha256"]
        assert fingerprint((DATA / label["image_relative"]).read_bytes()) == {
            key: artifact[key] for key in ("bytes", "sha256")
        }
        assert "\n" in label["answer"]
        identifier = "ocr-v4:" + label["id"]
        assert assembled_ids[identifier] == 1
        row = assembled[identifier]
        assert row["task"] == row["references"]["kind"] == "ocr_order"
        assert row["references"]["text"] == row["answer"] == label["answer"]
        assert row["references"]["strip_whitespace"] is False
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "../escape",
        "vision/../escape",
        "/absolute",
        "vision\\photo.jpg",
        "vision//a",
        "./vision/a",
        "vision/a/",
        "vision/\x00.jpg",
    ],
)
def test_archive_paths_reject_noncanonical_or_escaping_names(name):
    with pytest.raises(ValueError):
        assets.relative(name)


@pytest.mark.parametrize("kind", ["file-symlink", "ancestor-symlink", "directory", "fifo"])
def test_package_requires_regular_files_inside_data_root(tmp_path, kind):
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "photo.jpg").write_bytes(b"outside data root")
    if kind == "file-symlink":
        (data / "photo.jpg").symlink_to(outside / "photo.jpg")
    elif kind == "ancestor-symlink":
        (data / "linked").symlink_to(outside, target_is_directory=True)
    elif kind == "directory":
        (data / "photo.jpg").mkdir()
    else:
        os.mkfifo(data / "photo.jpg")
    name = "linked/photo.jpg" if kind == "ancestor-symlink" else "photo.jpg"
    with pytest.raises(ValueError):
        assets.package_selected(data, {name}, tmp_path / "bundle.tar.gz")


def test_package_deterministic_gzip_tar_headers_members_and_sha(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    payloads = {"vision/z.jpg": b"z bytes", "ocr/a.png": b"a bytes"}
    for name, raw in payloads.items():
        path = data / name
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    members = assets.package_selected(data, set(payloads), first)
    os.utime(data / "vision/z.jpg", (123456, 123456))
    os.chmod(data / "ocr/a.png", 0o600)
    assert assets.package_selected(data, reversed(list(payloads)), second) == members
    assert first.read_bytes() == second.read_bytes()
    header = first.read_bytes()[:10]
    assert header[:3] == b"\x1f\x8b\x08"
    assert header[3] & 0x08 == 0  # gzip must not embed the destination filename.
    assert struct.unpack("<I", header[4:8])[0] == 0
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == sorted(payloads)
        for member in archive:
            assert member.isfile() and not member.issym() and not member.islnk()
            assert (member.mode, member.mtime, member.uid, member.gid) == (0o644, 0, 0, 0)
            assert member.uname == member.gname == ""
            assert archive.extractfile(member).read() == payloads[member.name]
    assert members == [{"path": name, **fingerprint(payloads[name])} for name in sorted(payloads)]


@pytest.mark.parametrize("raw", [b"wrong but same byte count", b"too short", b"fixed source bytes" * 10])
def test_download_rejects_changed_short_or_overlong_bytes(tmp_path, monkeypatch, raw):
    wanted = b"the exact frozen bytes!!"
    destination = tmp_path / "photo.jpg"
    destination.write_bytes(b"previous local bytes")
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: Response(raw))
    with pytest.raises(ValueError):
        assets.download_exact(
            "https://public.example/photo",
            destination,
            **{"expected_bytes": len(wanted), "expected_sha256": fingerprint(wanted)["sha256"]},
        )
    assert destination.read_bytes() == b"previous local bytes"
    assert not destination.with_suffix(".jpg.download").exists()


def test_download_reuses_only_exact_frozen_bytes_and_refuses_unbounded_requests(tmp_path, monkeypatch):
    raw = b"exact public bytes"
    destination = tmp_path / "image.jpg"
    destination.write_bytes(raw)
    monkeypatch.setattr(
        assets.urllib.request, "urlopen", lambda *a, **k: pytest.fail("valid cached bytes must not be downloaded")
    )
    assets.download_exact("https://public.example/photo", destination, len(raw), fingerprint(raw)["sha256"])
    for url, size in [
        ("http://public.example/photo", len(raw)),
        ("file:///tmp/photo", len(raw)),
        ("https://public.example/photo", 0),
        ("https://public.example/photo", assets.MAX_DOWNLOAD_BYTES + 1),
    ]:
        with pytest.raises(ValueError):
            assets.download_exact(url, destination, size, fingerprint(raw)["sha256"])


def test_exact_download_retries_temporary_errors_but_not_integrity_failures(tmp_path, monkeypatch):
    raw = b"an exact frozen photograph"
    calls, sleeps = [], []

    def open_response(request, **kwargs):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "temporary", {}, None)
        return Response(raw)

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    destination = tmp_path / "photo.jpg"
    assets.download_exact("https://public.example/photo", destination, len(raw), fingerprint(raw)["sha256"])
    assert len(calls) == 2 and len(sleeps) == 1
    destination.unlink()
    calls.clear()
    sleeps.clear()

    def changed(request, **kwargs):
        calls.append(request.full_url)
        return Response(b"x" * len(raw))

    monkeypatch.setattr(assets.urllib.request, "urlopen", changed)
    with pytest.raises(ValueError):
        assets.download_exact("https://public.example/photo", destination, len(raw), fingerprint(raw)["sha256"])
    assert len(calls) == 1 and sleeps == []


def test_exact_download_permanent_errors_and_exhausted_retries_are_bounded(tmp_path, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    for status in (404, 503):
        calls.clear()
        sleeps.clear()

        def failed(request, **kwargs):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, status, "failure", {}, None)

        monkeypatch.setattr(assets.urllib.request, "urlopen", failed)
        with pytest.raises((urllib.error.HTTPError, RuntimeError)):
            assets.download_exact(
                "https://public.example/photo", tmp_path / "photo.jpg", 1, fingerprint(b"x")["sha256"]
            )
        if status == 404:
            assert len(calls) == 1 and sleeps == []
        else:
            assert 2 <= len(calls) <= 4 and len(sleeps) == len(calls) - 1
        assert not (tmp_path / "photo.jpg.download").exists()


def test_build_rejects_changed_local_asset_instead_of_freezing_new_bytes(frozen):
    target = frozen["data"] / frozen["photo"]["image"]
    original = target.read_bytes()
    target.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError):
        build_fixture(frozen)


def test_family_questions_cannot_cross_splits(frozen):
    sibling = dict(frozen["row"], id="sibling-question", split="test", user="另一問題？")
    write_rows(frozen["metadata"] / "vision-labels/validation.jsonl", [frozen["row"], sibling])
    with pytest.raises(ValueError):
        assets.assemble()


@pytest.mark.parametrize("renamed", [False, True])
def test_same_source_cannot_cross_splits_by_relabelling_its_family(frozen, renamed):
    sibling = dict(frozen["row"], id="fake-independent-question", family="unrelated-family", split="test")
    if renamed:
        sibling["image"] = "vision/renamed.jpg"
        (frozen["data"] / sibling["image"]).write_bytes(frozen["raw"])
        source = dict(frozen["photo"], image=sibling["image"], family=sibling["family"], split=sibling["split"])
        write_json(frozen["metadata"] / "vision-activity-sources.json", {"rows": [source]})
    write_rows(frozen["metadata"] / "vision-labels/validation.jsonl", [frozen["row"], sibling])
    with pytest.raises(ValueError):
        build_fixture(frozen)


def test_crop_question_cannot_override_its_original_photo_family_or_split(frozen):
    source = {
        "source_id": "commons-fixture",
        "dataset": "Wikimedia Commons individually licensed photographs",
        "family": "photo-family",
        "split": "validation",
        "image_relative": frozen["photo"]["image"],
        "image_bytes": len(frozen["raw"]),
        "image_sha256": fingerprint(frozen["raw"])["sha256"],
    }
    raw = b"frozen crop bytes"
    name = "ocr/crops/fixture.png"
    target = frozen["data"] / name
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    artifact = {"path": name, "kind": "crop", "source_id": source["source_id"], **fingerprint(raw)}
    write_json(frozen["metadata"] / "ocr-sources.json", {"sources": [source], "artifacts": [artifact]})
    label = {
        "id": "crop-question",
        "source_id": source["source_id"],
        "family": "pretend-independent-crop",
        "split": "test",
        "task": "ocr_transcription",
        "question": "文字？",
        "answer": "測試",
        "image_relative": name,
        "image_sha256": artifact["sha256"],
    }
    write_rows(frozen["metadata"] / "ocr-labels.jsonl", [label])
    with pytest.raises(ValueError):
        build_fixture(frozen)


def test_build_refuses_an_asset_missing_from_frozen_source_declarations(frozen):
    name = "vision/unattributed.jpg"
    (frozen["data"] / name).write_bytes(b"undeclared bytes")
    write_rows(frozen["metadata"] / "vision-labels/validation.jsonl", [dict(frozen["row"], image=name)])
    with pytest.raises(ValueError):
        build_fixture(frozen)


def test_multiple_questions_do_not_duplicate_archived_source(frozen):
    sibling = dict(frozen["row"], id="photo-question-2", user="第二問題？", task="text_presence", answer="沒有")
    write_rows(frozen["metadata"] / "vision-labels/validation.jsonl", [frozen["row"], sibling])
    manifest = build_fixture(frozen)
    assert len(manifest["rows"]) == 2
    assert len({row["image"] for row in manifest["rows"]}) == 1
    photo_files = [row for row in manifest["files"] if row["path"] == frozen["photo"]["image"]]
    assert photo_files == [{"path": frozen["photo"]["image"], **fingerprint(frozen["raw"])}]
    assert not any("review-galler" in row["path"] for row in manifest["files"])


def test_build_is_byte_identical_and_records_every_archive_member(frozen):
    first = build_fixture(frozen)
    original = {row["path"]: (frozen["root"] / row["path"]).read_bytes() for row in first["archives"]}
    second = build_fixture(frozen)
    assert first == second
    for archive in first["archives"]:
        path = frozen["root"] / archive["path"]
        assert path.read_bytes() == original[archive["path"]]
        assert fingerprint(path.read_bytes()) == {key: archive[key] for key in ("bytes", "sha256")}
        with tarfile.open(path, "r:gz") as bundle:
            assert bundle.getnames() == [row["path"] for row in archive["files"]]
            for member, declared in zip(bundle, archive["files"], strict=True):
                assert fingerprint(bundle.extractfile(member).read()) == {
                    key: declared[key] for key in ("bytes", "sha256")
                }


def test_cli_verification_reads_the_frozen_manifest_before_overwriting_same_path(frozen, monkeypatch):
    expected = build_fixture(frozen)
    manifest_path = frozen["root"] / "manifest.json"
    original_bytes = manifest_path.read_bytes()
    changed = dict(expected, dataset_version="a changed snapshot must fail verification")

    def changed_build(data_root, archives, output):
        output.write_bytes(assets.canonical(changed))
        return changed

    monkeypatch.setattr(assets, "build", changed_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_natural_v4_assets.py",
            "--data-root",
            str(frozen["data"]),
            "--assets",
            str(frozen["root"] / "assets/training"),
            "--manifest",
            str(manifest_path),
            "--verify-manifest",
            str(manifest_path),
        ],
    )
    with pytest.raises(ValueError):
        assets.main()
    assert manifest_path.read_bytes() == original_bytes


def test_aishell_archive_includes_exact_apache_license_and_source_attribution(frozen):
    document = json.loads((METADATA / "voice-question-sources.json").read_text())
    raw = wav_bytes()
    row = dict(document["audio_rows"][0], **audio_row(raw))
    document["audio_rows"] = [row]
    write_json(frozen["metadata"] / "voice-question-sources.json", document)
    target = frozen["data"] / "voice" / row["audio"]
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    manifest = build_fixture(frozen)
    archive = next(item for item in manifest["archives"] if item["path"].endswith("natural-voice-v4.tar.gz"))
    license_bytes = (ROOT / "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE").read_bytes()
    with tarfile.open(frozen["root"] / archive["path"], "r:gz") as bundle:
        assert bundle.extractfile("voice/AISHELL-Apache-2.0-LICENSE.txt").read() == license_bytes
        notice = bundle.extractfile("voice/NOTICE.txt").read().decode("utf-8")
        assert "https://www.openslr.org/33" in notice and "human read" in notice.lower()
        attribution = json.loads(bundle.extractfile("voice/ATTRIBUTION.json").read())
        source = next(s for s in attribution["sources"] if s["path"].endswith("voice-question-sources.json"))
        assert source["metadata"]["official_corpus_license"] == "Apache-2.0"
        assert source["metadata"]["attribution"] == document["attribution"]


@pytest.mark.parametrize(
    "status,content_range,raw",
    [
        (200, "bytes 0-7/16", b"abcdefgh"),
        (206, "bytes 0-7/17", b"abcdefgh"),
        (206, "bytes 1-8/16", b"abcdefgh"),
        (206, "bytes 0-7/16", b"abc"),
        (206, "bytes 0-7/16", b"abcdefghi"),
    ],
)
def test_nvidia_ranges_reject_full_or_inexact_server_responses(monkeypatch, status, content_range, raw):
    monkeypatch.setattr(
        ocr.urllib.request,
        "urlopen",
        lambda *a, **k: Response(raw, status=status, headers={"Content-Range": content_range}),
    )
    reader = ocr.RangeFile("https://public.example/frozen.h5", size=16, limit=8)
    reader.block_size = 8
    with pytest.raises(ValueError):
        reader.read(1)


def test_nvidia_range_cache_and_total_transfer_limit(monkeypatch):
    requests = []

    def open_range(request, **kwargs):
        requests.append(request.get_header("Range"))
        return Response(b"abcdefgh", status=206, headers={"Content-Range": "bytes 0-7/16"})

    monkeypatch.setattr(ocr.urllib.request, "urlopen", open_range)
    reader = ocr.RangeFile("https://public.example/frozen.h5", size=16, limit=8)
    reader.block_size = 8
    assert reader.read(3) == b"abc"
    reader.seek(1)
    assert reader.read(7) == b"bcdefgh"
    assert requests == ["bytes=0-7"]
    assert reader.transferred == 8
    assert reader.receipts == [{"start": 0, "end": 7, **fingerprint(b"abcdefgh"), "attempts": 1}]
    with pytest.raises(ValueError):
        reader.read(1)
    with pytest.raises(ValueError):
        reader.seek(0)
        reader.read(9)


def test_nvidia_retries_the_failed_range_without_restarting_cached_bytes(monkeypatch):
    requests, sleeps = [], []

    def open_range(request, **kwargs):
        value = request.get_header("Range")
        requests.append(value)
        if requests == ["bytes=0-7"]:
            return Response(b"abcdefgh", status=206, headers={"Content-Range": "bytes 0-7/16"})
        if requests.count("bytes=8-15") == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "temporary", {}, None)
        return Response(b"ijklmnop", status=206, headers={"Content-Range": "bytes 8-15/16"})

    monkeypatch.setattr(ocr.urllib.request, "urlopen", open_range)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    reader = ocr.RangeFile("https://public.example/frozen.h5", size=16, limit=24)
    reader.block_size = 8
    assert reader.read(8) == b"abcdefgh"
    assert reader.read(8) == b"ijklmnop"
    assert requests == ["bytes=0-7", "bytes=8-15", "bytes=8-15"]
    assert len(sleeps) == 1 and reader.transferred == 16
    assert reader.reserved_transfer == 24  # Both requests and the failed attempt consume the transport budget.


def test_ocr_retries_only_temporary_http_failures(monkeypatch):
    raw = b"frozen public photograph"
    calls, sleeps = [], []

    def open_response(request, **kwargs):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "temporary", {}, None)
        return Response(raw)

    monkeypatch.setattr(ocr.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(ocr.time, "sleep", sleeps.append)
    assert ocr.download("https://public.example/photo", fingerprint(raw)["sha256"], len(raw)) == raw
    assert len(calls) == 2 and sleeps == [1]
    calls.clear()
    sleeps.clear()

    def missing(request, **kwargs):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "permanent", {}, None)

    monkeypatch.setattr(ocr.urllib.request, "urlopen", missing)
    with pytest.raises(RuntimeError):
        ocr.download("https://public.example/missing", fingerprint(raw)["sha256"], len(raw))
    assert len(calls) == 1 and sleeps == []


def test_ocr_existing_commons_crop_rebuild_matches_actual_frozen_bytes(tmp_path, monkeypatch):
    if not (METADATA / "ocr-sources.json").exists():
        pytest.skip("local reviewed OCR metadata not present")
    original = json.loads((METADATA / "ocr-sources.json").read_text())
    assert PIL.__version__ == original["encoding"]["pillow"]
    crop = next(a for a in original["artifacts"] if a["kind"] == "crop" and a["source_id"].startswith("commons-"))
    source = next(s for s in original["sources"] if s["source_id"] == crop["source_id"])
    selected = [a for a in original["artifacts"] if a["source_id"] == source["source_id"]]
    if not all((DATA / a["path"]).is_file() for a in selected):
        pytest.skip("actual frozen Commons source/crops are not in this checkout")
    spec = copy.deepcopy(original)
    spec["sources"], spec["artifacts"] = [source], selected
    raw = (DATA / source["image_relative"]).read_bytes()
    requests = []

    def open_photo(request, **kwargs):
        assert request.full_url == source["download_url"]
        requests.append(request.full_url)
        return Response(raw)

    monkeypatch.setattr(ocr.urllib.request, "urlopen", open_photo)
    sources = tmp_path / "sources.json"
    write_json(sources, spec)
    output = tmp_path / "rebuilt"
    receipt = ocr.rebuild(sources, output)
    assert requests == [source["download_url"]]
    assert receipt["verified_artifacts"] == len(selected)
    assert receipt["nvidia_range_http_bytes"] == 0
    for artifact in selected:
        assert (output / artifact["path"]).read_bytes() == (DATA / artifact["path"]).read_bytes()
        assert fingerprint((output / artifact["path"]).read_bytes()) == {
            key: artifact[key] for key in ("bytes", "sha256")
        }


def audio_row(raw):
    return {
        "id": "fixed-recording",
        "audio": "audio/test/fixed.wav",
        "source_member": "test/fixed.wav",
        "sample_rate": 16000,
        "channels": 1,
        "num_samples": 192,
        **fingerprint(raw),
    }


def test_voice_replay_writes_exact_original_wav_and_reuses_it(tmp_path, monkeypatch):
    raw = wav_bytes()
    row = audio_row(raw)
    url = FLEURS_URL
    requests = []

    def open_archive(request, **kwargs):
        requests.append(request.full_url)
        return Response(tar_bytes({"unused.wav": b"not selected", row["source_member"]: raw}))

    monkeypatch.setattr(voice.urllib.request, "urlopen", open_archive)
    first = voice.replay_archive((url, [row]), tmp_path)
    assert (tmp_path / row["audio"]).read_bytes() == raw
    assert first["recordings"] == 1 and first["reused"] == 0
    second = voice.replay_archive((url, [row]), tmp_path)
    assert len(requests) == 1 and second["reused"] == 1 and second["compressed_bytes_read"] == 0


@pytest.mark.parametrize("metadata_name", ["voice-sources.json", "voice-question-sources.json"])
def test_voice_replay_matches_an_actual_frozen_corpus_recording(tmp_path, monkeypatch, metadata_name):
    if not (METADATA / metadata_name).is_file():
        pytest.skip("reviewed v4 voice source metadata not present")
    document = json.loads((METADATA / metadata_name).read_text())
    row = next(row for row in document["audio_rows"] if row["split"] != "train")
    original = DATA / "voice" / row["audio"]
    if not original.is_file():
        pytest.skip("actual frozen source WAV not present in this checkout")
    url = (
        row["source_archive"]["url"]
        if "source_archive" in row
        else document["source_files"][row["source_split"]]["audio_archive"]["url"]
    )
    raw = original.read_bytes()
    archive = tar_bytes({row["source_member"]: raw})
    requests = []

    def open_archive(request, **kwargs):
        assert request.full_url == url
        requests.append(request.full_url)
        return Response(archive)

    monkeypatch.setattr(voice.urllib.request, "urlopen", open_archive)
    receipt = voice.replay_archive((url, [row]), tmp_path)
    assert requests == [url] and receipt["recordings"] == 1
    assert (tmp_path / row["audio"]).read_bytes() == raw
    assert fingerprint((tmp_path / row["audio"]).read_bytes()) == {key: row[key] for key in ("bytes", "sha256")}


def test_voice_replay_rejects_corrupt_sha_duplicate_member_and_wrong_wav_metadata(tmp_path, monkeypatch):
    raw = wav_bytes()
    row = audio_row(raw)
    url = FLEURS_URL
    monkeypatch.setattr(
        voice.urllib.request, "urlopen", lambda *a, **k: Response(tar_bytes({row["source_member"]: raw}))
    )
    with pytest.raises(ValueError):
        voice.replay_archive((url, [row, dict(row, id="duplicate")]), tmp_path)
    with pytest.raises(ValueError):
        voice.replay_archive((url, [dict(row, sha256="0" * 64)]), tmp_path)
    with pytest.raises(ValueError):
        voice.replay_archive((url, [dict(row, sample_rate=8000)]), tmp_path)
    assert not (tmp_path / row["audio"]).exists()


def test_voice_replay_retries_a_temporary_archive_error(tmp_path, monkeypatch):
    raw = wav_bytes()
    row = audio_row(raw)
    calls, sleeps = [], []

    def open_archive(request, **kwargs):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "temporary", {}, None)
        return Response(tar_bytes({row["source_member"]: raw}))

    monkeypatch.setattr(voice.urllib.request, "urlopen", open_archive)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    receipt = voice.replay_archive((FLEURS_URL, [row]), tmp_path)
    assert receipt["recordings"] == 1 and len(calls) == 2 and len(sleeps) == 1
    assert (tmp_path / row["audio"]).read_bytes() == raw


def test_voice_replay_permanent_failure_is_not_retried_and_retry_count_is_bounded(tmp_path, monkeypatch):
    row = audio_row(wav_bytes())
    calls, sleeps = [], []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    for status in (404, 503):
        calls.clear()
        sleeps.clear()

        def failed(request, **kwargs):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, status, "failure", {}, None)

        monkeypatch.setattr(voice.urllib.request, "urlopen", failed)
        with pytest.raises((urllib.error.HTTPError, RuntimeError)):
            voice.replay_archive((FLEURS_URL, [row]), tmp_path)
        if status == 404:
            assert len(calls) == 1 and sleeps == []
        else:
            assert 2 <= len(calls) <= 4 and len(sleeps) == len(calls) - 1


def test_voice_replay_does_not_overwrite_duplicate_destination_paths(tmp_path, monkeypatch):
    first = audio_row(wav_bytes())
    second = dict(first, id="other-id", source_member="test/other.wav")
    monkeypatch.setattr(
        voice.urllib.request, "urlopen", lambda *a, **k: pytest.fail("duplicate paths must be rejected before download")
    )
    with pytest.raises(ValueError):
        voice.replay_archive((FLEURS_URL, [first, second]), tmp_path)


def test_voice_reconstruct_refuses_an_unpinned_unofficial_source_before_download(frozen, monkeypatch):
    raw = wav_bytes()
    row = dict(audio_row(raw), split="test", source_split="test", family="fixed-family")
    document = {
        "revision": "a" * 40,
        "audio_rows": [row],
        "source_files": {"test": {"audio_archive": {"url": "https://public.example/unofficial.tar.gz"}}},
    }
    write_json(frozen["metadata"] / "voice-sources.json", document)
    monkeypatch.setattr(
        assets.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("unofficial corpus must be rejected before downloading"),
    )
    with pytest.raises(ValueError):
        assets.reconstruct_voices(frozen["data"])


@pytest.mark.parametrize("module", [voice, ocr])
def test_reconstruction_destinations_refuse_symlink_parent_escape(tmp_path, module):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        if module is voice:
            voice.target_path(output, {"audio": "linked/escape.wav"})
        else:
            ocr.safe_path(output, "linked/escape.png")


def test_voice_bounded_stream_stops_overlong_compressed_download():
    reader = assets.BoundedReader(io.BytesIO(b"12345"), maximum=4)
    assert reader.read(4) == b"1234"
    with pytest.raises(ValueError):
        reader.read(1)
    replay_reader = voice.BoundedReader(io.BytesIO(b"12345"), limit=4)
    assert replay_reader.read(4) == b"1234"
    with pytest.raises(ValueError):
        replay_reader.read(1)


def test_current_voice_selection_preserves_30_asr_only_and_8_human_read_questions():
    if not (METADATA / "voice-sources.json").exists():
        pytest.skip("reviewed v4 source metadata not present")
    rows, audio = assets.assemble()
    assert not any(row.get("audio") for row in rows)
    assert Counter(row["task"] for row in audio) == {"speech_transcription": 30, "speech_chat": 8}
    assert not any(row["split"] == "train" for row in audio)
    for row in audio:
        assert row["synthetic"] is False
        if row["task"] == "speech_chat":
            assert "human read" in row["references"]["scope"].lower()
            assert "not spontaneous" in row["references"]["scope"].lower()
        else:
            assert "not a human question" in row["recording_domain"].lower()


def test_current_ocr_families_keep_all_originals_crops_and_questions_together():
    if not (METADATA / "ocr-sources.json").exists():
        pytest.skip("reviewed v4 OCR source metadata not present")
    spec = json.loads((METADATA / "ocr-sources.json").read_text())
    sources = {row["source_id"]: row for row in spec["sources"]}
    artifacts = {row["path"]: row for row in spec["artifacts"]}
    families = defaultdict(set)
    identities = defaultdict(set)
    for row in assets.json_lines(METADATA / "ocr-labels.jsonl"):
        source = sources[row["source_id"]]
        artifact = artifacts[row["image_relative"]]
        assert artifact["source_id"] == source["source_id"]
        assert row["family"] == source["family"] and row["split"] == source["split"]
        assert row["image_sha256"] == artifact["sha256"]
        families[row["family"]].add(row["split"])
        identities[source["image_sha256"]].add(row["split"])
    assert all(len(splits) == 1 for splits in families.values())
    assert all(len(splits) == 1 for splits in identities.values())
    for source in sources.values():
        license_url = urllib.parse.urlsplit(source["license_url"])
        assert source["license"] and license_url.scheme in {"http", "https"}
        assert license_url.hostname == "creativecommons.org"
        if source["dataset"] == "nvidia/OCR-Synthetic-Multilingual-v1":
            assert "BY" in source["license"].upper()
        elif source["dataset"] == "Wikimedia Commons individually licensed photographs":
            assert source["source_page"].startswith("https://commons.wikimedia.org/")
            assert source["artist_html"] or source.get("attribution_html")
        else:
            assert source["dataset"] == "DOCCI"
            assert source["download_url"].startswith("https://storage.googleapis.com/docci/")
            assert source["artist_html"] == "Jason Baldridge and family"
