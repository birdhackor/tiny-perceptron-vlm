"""Optional legal proof backup: closed bytes, deterministic archive, no publication."""

import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("proof", ROOT / "scripts/build_natural_v4_research_proof.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "review.txt").write_bytes(b"original review observations\n")
    (root / "crop.jpg").write_bytes(b"legal licensed photograph byte fixture")
    return root


def test_archive_is_deterministic_closed_regular_and_unpadded(source, tmp_path):
    generated = {"PROOF/NOTICE.txt": b"source and author notice\n"}
    names = ["review.txt", "crop.jpg", "review.txt"]
    one, two = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    index = proof.package(source, names, one, generated)
    proof.package(source, names[::-1], two, generated)
    assert one.read_bytes() == two.read_bytes()
    assert one.read_bytes()[4:8] == b"\0" * 4
    assert not one.read_bytes()[3] & 8  # gzip embeds no host output filename
    assert len(index["files"]) == 2
    assert sum(row["bytes"] for row in index["files"]) == sum((source / n).stat().st_size for n in set(names))
    with tarfile.open(one) as archive:
        members = archive.getmembers()
        assert [m.name for m in members] == sorted({"review.txt", "crop.jpg", "PROOF/NOTICE.txt", "PROOF/INDEX.json"})
        assert all(m.isfile() and m.mtime == m.uid == m.gid == 0 and m.uname == m.gname == "" for m in members)
        assert json.loads(archive.extractfile("PROOF/INDEX.json").read()) == index
    proof.verify_archive(one, index, generated)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../review.txt", "./review.txt", "a\\b", "nul\0", "."])
def test_rejects_unsafe_source_and_generated_paths(source, tmp_path, name):
    with pytest.raises(ValueError):
        proof.package(source, [name], tmp_path / "bad.tar.gz", {})
    with pytest.raises(ValueError):
        proof.package(source, ["review.txt"], tmp_path / "bad.tar.gz", {name: b"bad"})
    assert not (tmp_path / "bad.tar.gz").exists()


def test_rejects_source_symlink_parent_and_non_regular_file(source, tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("This host provides no symlink API")
    try:
        (source / "link").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("This host does not permit symlink creation")
    (tmp_path / "outside.txt").write_bytes(b"outside")
    with pytest.raises(ValueError, match="regular files without symlinks"):
        proof.safe_file(source, "link/outside.txt")
    with pytest.raises(ValueError, match="regular files"):
        proof.safe_file(source, "link")
    with pytest.raises(ValueError, match="regular files"):
        proof.safe_file(source, "missing")


def test_source_change_aborts_without_partial_or_final_archive(source, tmp_path, monkeypatch):
    original = proof.file_digest

    def altered(path):
        result = original(path)
        path.write_bytes(b"changed after index creation")
        return result

    monkeypatch.setattr(proof, "file_digest", altered)
    target = tmp_path / "proof.tar.gz"
    with pytest.raises(ValueError, match="changed during"):
        proof.package(source, ["review.txt"], target, {})
    assert not target.exists()
    assert not target.with_suffix(".gz.partial").exists()


def test_immutable_output_and_reserved_namespace(source, tmp_path):
    target = tmp_path / "proof.tar.gz"
    proof.package(source, ["review.txt"], target, {})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        proof.package(source, ["review.txt"], target, {})
    assert target.read_bytes() == original
    with pytest.raises(ValueError, match="reserved"):
        proof.package(source, ["review.txt"], tmp_path / "two.tar.gz", {"PROOF/INDEX.json": b"fake index"})
    with pytest.raises(ValueError, match="overlap"):
        proof.package(source, ["PROOF/NOTICE.txt"], tmp_path / "two.tar.gz", {"PROOF/NOTICE.txt": b"fake"})


def test_total_raw_bytes_are_bounded(source, tmp_path, monkeypatch):
    monkeypatch.setattr(proof, "MAX_TOTAL_BYTES", 4)
    with pytest.raises(ValueError, match="512 MiB bound"):
        proof.package(source, ["review.txt"], tmp_path / "large.tar.gz", {})


@pytest.mark.parametrize("defect", ["changed_bytes", "missing", "extra", "duplicate", "symlink", "notice_change"])
def test_local_verifier_rejects_archive_tampering(source, tmp_path, defect):
    generated = {"PROOF/NOTICE.txt": b"source notice"}
    original = tmp_path / "original.tar.gz"
    index = proof.package(source, ["review.txt"], original, generated)
    with tarfile.open(original) as archive:
        records = [(m.name, archive.extractfile(m).read()) for m in archive]
    if defect == "missing":
        records = [item for item in records if item[0] != "review.txt"]
    if defect == "extra":
        records.append(("other.txt", b"undeclared"))
    if defect == "duplicate":
        records.append(records[-1])
    target = tmp_path / "altered.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for name, raw in records:
            if (defect == "changed_bytes" and name == "review.txt") or (
                defect == "notice_change" and name == "PROOF/NOTICE.txt"
            ):
                raw = bytes([raw[0] ^ 1]) + raw[1:]
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            if defect == "symlink" and name == "review.txt":
                info.type, info.linkname, info.size = tarfile.SYMTYPE, "outside", 0
                raw = b""
            archive.addfile(info, io.BytesIO(raw))
    with pytest.raises(ValueError):
        proof.verify_archive(target, index, generated)


@pytest.mark.parametrize("license", ["CC BY-SA 4.0", "CC BY-NC 4.0", "unknown", ""])
def test_rejects_unselected_sa_nc_or_unknown_source_licenses(license):
    with pytest.raises(ValueError):
        proof.license_key(license)


def test_selection_excludes_candidate_cards_html_and_old_ocr(tmp_path):
    root = tmp_path
    required = [
        "LICENSE",
        ".github/workflows/natural-data-v4.yml",
        "docs/natural-assistant/v4/manifest.json",
        *(
            f"scripts/{name}.py"
            for name in (
                "build_natural_v4_research_proof",
                "build_natural_v4_assets",
                "fetch_natural_data",
                "prepare_natural_v4_ocr",
                "replay_natural_v4_voice",
            )
        ),
        *(f"docs/natural-assistant/v4/data/{name}" for name in proof.SOURCE_FILES),
        *(
            f"tests/{name}.py"
            for name in (
                "test_natural_v4_assets",
                "test_natural_v4_data_download",
                "test_natural_v4_data_workflow",
                "test_natural_v4_research_proof",
            )
        ),
    ]
    keep = f"{proof.EVIDENCE}/vision/sources/docci-descriptions.jsonl"
    reject = [
        f"{proof.EVIDENCE}/ocr/original-research/preview-SA.jpg",
        f"{proof.EVIDENCE}/vision/sources/dataset-README.html",
        f"{proof.EVIDENCE}/chat-speech/TaiwanChat-NC-card.md",
        f"{proof.EVIDENCE}/experiment/qwen3-vl-readme.md",
    ]
    for name in [*required, keep, *reject]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    selected = proof.selected_files(root)
    assert keep in selected
    assert not set(reject) & set(selected)
    assert selected == sorted(set(selected))


def test_existing_receipt_refuses_work_before_build(tmp_path, monkeypatch):
    import sys

    receipts = tmp_path / "receipts"
    receipts.mkdir()
    target = tmp_path / "must-not-build.tar.gz"
    monkeypatch.setattr(sys, "argv", ["proof", "--archive", str(target), "--receipt-dir", str(receipts)])
    with pytest.raises(FileExistsError, match="receipts are immutable"):
        proof.main()
    assert not target.exists()
