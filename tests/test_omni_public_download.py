"""公開新產物的archive完整性與不覆寫使用者檔案；離線人工bytes。"""

import hashlib
import json
import zipfile

import pytest

from scripts.omni_revision import public_download as loader


def package(tmp_path, extra=None, wrong_inner_hash=False):
    files = {
        "core/codec/training/weights.safetensors": b"explicit weight bytes fixture, never deserialized",
        "core/codec/training/training-result.json": b'{"explicit_fixture":true}',
        "core/codec/job-result.json": b'{"explicit_fixture":true}',
        "core/codec/training/latest-continuation.pt": b"explicit continuation bytes, never unpickled",
    }
    files.update(extra or {})
    manifest = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()}
    if wrong_inner_hash:
        manifest["core/codec/training/weights.safetensors"]["sha256"] = "0" * 64
    path = tmp_path / "codec.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
        archive.writestr("package-file-manifest.json", json.dumps(manifest))
    asset = {"job": "codec", "name": "codec.zip", "bytes": path.stat().st_size, "sha256": loader.file_sha(path)}
    return path, asset, files


def selected_files(files):
    return {
        name: {
            "sha256": hashlib.sha256(raw).hexdigest(),
            **({"bytes": len(raw)} if name.endswith(".safetensors") else {}),
        }
        for name, raw in files.items()
        if name.endswith(("weights.safetensors", "training-result.json", "job-result.json"))
    }


def test_inference_download_excludes_continuation_and_can_reuse_exact_cache(tmp_path):
    path, asset, files = package(tmp_path)
    target = tmp_path / "downloaded"
    receipt = loader.install(path, asset, target, expected_files=selected_files(files))
    assert (target / "core/codec/training/weights.safetensors").read_bytes() == files[
        "core/codec/training/weights.safetensors"
    ]
    assert not (target / "core/codec/training/latest-continuation.pt").exists()
    assert not any(name.endswith(".pt") for name in receipt)
    assert loader.install(path, asset, target, expected_files=selected_files(files)) == receipt


def test_explicit_continuation_option_copies_without_unpickling(tmp_path):
    path, asset, files = package(tmp_path)
    target = tmp_path / "full"
    loader.install(path, asset, target, full_state=True, expected_files=selected_files(files))
    assert (target / "core/codec/training/latest-continuation.pt").read_bytes() == files[
        "core/codec/training/latest-continuation.pt"
    ]


@pytest.mark.parametrize("failure", ["outer_sha", "inner_sha", "escape"])
def test_bad_or_outside_archive_never_installs_weights(tmp_path, failure):
    path, asset, files = package(
        tmp_path,
        {"core/codec/../../escape": b"outside"} if failure == "escape" else None,
        wrong_inner_hash=failure == "inner_sha",
    )
    if failure == "outer_sha":
        asset["sha256"] = "0" * 64
    target = tmp_path / "rejected"
    with pytest.raises(ValueError):
        loader.install(path, asset, target, expected_files=selected_files(files))
    assert not (target / "core/codec/training/weights.safetensors").exists()
    assert not (tmp_path / "escape").exists()


def test_existing_user_weights_are_preserved(tmp_path):
    path, asset, files = package(tmp_path)
    target = tmp_path / "old"
    weight = target / "core/codec/training/weights.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"user edited data")
    with pytest.raises(ValueError, match="不覆寫"):
        loader.install(path, asset, target, expected_files=selected_files(files))
    assert weight.read_bytes() == b"user edited data"


@pytest.mark.parametrize(
    "member",
    ["C:/escaped.txt", "C:escaped.txt", "name:stream", "CON", "NUL.txt", "../x", "x/./y", "x//y", "tail.", "folder\\x"],
)
def test_cross_platform_unsafe_members_rejected_before_install(tmp_path, member):
    assert not loader.portable_member("core/codec/" + member)
    path, asset, files = package(tmp_path, {"core/codec/" + member: b"outside or ambiguous"})
    target = tmp_path / "rejected"
    # Windows zipfile會把fixture的反斜線換成/；此時內層清單不符也必須拒絕。
    with pytest.raises(ValueError, match="路徑|manifest"):
        loader.install(path, asset, target, expected_files=selected_files(files))
    assert not target.exists()


def test_windows_case_collisions_rejected_even_on_posix(tmp_path):
    path, asset, files = package(tmp_path, {"core/codec/training/WEIGHTS.safetensors": b"different case bytes"})
    target = tmp_path / "rejected"
    with pytest.raises(ValueError, match="重複路徑"):
        loader.install(path, asset, target, expected_files=selected_files(files))
    assert not target.exists()


@pytest.mark.parametrize("filename", ["weights.safetensors", "training-result.json", "job-result.json"])
def test_self_consistent_archive_must_match_selected_artifacts(tmp_path, filename):
    path, asset, files = package(tmp_path)
    expected = selected_files(files)
    name = next(name for name in expected if name.endswith("/" + filename))
    expected[name] = {**expected[name], "sha256": "f" * 64}
    target = tmp_path / "wrong-selection"
    with pytest.raises(ValueError, match="已選定SHA"):
        loader.install(path, asset, target, expected_files=expected)
    assert not target.exists()


def test_late_inner_hash_failure_does_not_leave_earlier_weights(tmp_path):
    path, asset, files = package(tmp_path, {"core/codec/training/last.json": b"late bytes"})
    with zipfile.ZipFile(path) as archive:
        payload = {row.filename: archive.read(row.filename) for row in archive.infolist()}
    manifest = json.loads(payload["package-file-manifest.json"])
    manifest["core/codec/training/last.json"]["sha256"] = "0" * 64
    payload["package-file-manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in payload.items():
            archive.writestr(name, raw)
    asset.update(bytes=path.stat().st_size, sha256=loader.file_sha(path))
    target = tmp_path / "rejected"
    with pytest.raises(ValueError, match="逐檔SHA"):
        loader.install(path, asset, target, expected_files=selected_files(files))
    assert not target.exists()


def test_later_conflicting_user_file_prevents_all_new_installs(tmp_path):
    path, asset, files = package(tmp_path, {"core/codec/training/last.json": b"new version"})
    target = tmp_path / "existing"
    existing = target / "core/codec/training/last.json"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep user bytes")
    with pytest.raises(ValueError, match="不覆寫"):
        loader.install(path, asset, target, expected_files=selected_files(files))
    assert existing.read_bytes() == b"keep user bytes"
    assert not (existing.parent / "weights.safetensors").exists()


def dangling_link(link, outside):
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("此Windows環境未提供建立symlink權限")


def test_dangling_receipt_symlink_rejected_before_network(tmp_path, monkeypatch):
    target = tmp_path / "download"
    target.mkdir()
    outside = tmp_path / "outside.json"
    dangling_link(target / "download-receipt.json", outside)
    monkeypatch.setattr(loader, "descriptor", lambda _: {})
    with pytest.raises(ValueError, match="symlink"):
        loader.download(tmp_path / "unused.json", target)
    assert not outside.exists()


def test_dangling_wav_symlink_rejected(tmp_path):
    import torch

    from scripts.omni_revision.public_demo import save_wave

    outside = tmp_path / "outside.wav"
    link = tmp_path / "output.wav"
    dangling_link(link, outside)
    with pytest.raises(FileExistsError):
        save_wave(link, torch.zeros(80))
    assert not outside.exists()


@pytest.mark.parametrize(
    "filename", ["training/weights.safetensors", "training/training-result.json", "job-result.json"]
)
def test_installed_checkpoint_rejected_if_selected_weight_or_receipt_differs(tmp_path, filename):
    path, asset, files = package(tmp_path)
    root = tmp_path / "installed"
    loader.install(path, asset, root, expected_files=selected_files(files))
    selected = {
        "selected": {"codec": "codec"},
        "model_files": {
            "codec": {
                "job": "codec",
                "weights": selected_files(files)["core/codec/training/weights.safetensors"],
                "training_result_sha256": hashlib.sha256(files["core/codec/training/training-result.json"]).hexdigest(),
            }
        },
        "all_formal_jobs": {
            "codec": {"job_result_sha256": hashlib.sha256(files["core/codec/job-result.json"]).hexdigest()}
        },
    }
    loader.verify_installed(root, selected, "codec")
    (root / "core/codec" / filename).write_bytes(b"different but legitimate checkpoint")
    with pytest.raises(ValueError, match="已選定成品"):
        loader.verify_installed(root, selected, "codec")
