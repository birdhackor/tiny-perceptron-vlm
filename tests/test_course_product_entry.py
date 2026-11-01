"""固定公開成品的下載邊界與主線入口，不使用網路或真實權重。"""

import json
import zipfile

import pytest

from scripts.course_product import RELEASE, ROOT, dispatch_request, install_archive, sha256, verify_bundle


@pytest.fixture
def archive(tmp_path):
    payload = {
        "bundle.json": b'{"status":"qualified"}',
        "policy.json": b"{}",
        "tokenizer.json": b"{}",
        "weights.safetensors": b"test fixture, not neural weights",
    }
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w") as target:
        for name, data in payload.items():
            target.writestr("tiny-perceptron-c-v3/bundle/" + name, data)
        target.writestr("../untrusted.py", b"must not be extracted or executed")
    descriptor = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()),
        "bundle": {name: sha256(data) for name, data in payload.items()},
    }
    return path, descriptor


def test_install_only_pinned_inference_payload_and_allow_identical_reuse(archive, tmp_path):
    path, descriptor = archive
    output = tmp_path / "installed"
    install_archive(path, output, descriptor)
    install_archive(path, output, descriptor)
    assert {p.name for p in output.iterdir()} == set(descriptor["bundle"])
    assert not (tmp_path / "untrusted.py").exists()
    verify_bundle(output, descriptor)


def test_corrupt_archive_cannot_create_a_bundle(archive, tmp_path):
    path, descriptor = archive
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        install_archive(path, tmp_path / "installed", descriptor)
    assert not (tmp_path / "installed").exists()


def test_valid_zip_with_wrong_inner_payload_is_rejected(archive, tmp_path):
    path, descriptor = archive
    descriptor["bundle"]["policy.json"] = "0" * 64
    with pytest.raises(ValueError, match="policy.json"):
        install_archive(path, tmp_path / "installed", descriptor)
    assert not (tmp_path / "installed").exists()


def test_existing_different_version_is_preserved(archive, tmp_path):
    path, descriptor = archive
    output = tmp_path / "installed"
    output.mkdir()
    (output / "weights.safetensors").write_bytes(b"a different experiment")
    with pytest.raises(ValueError):
        install_archive(path, output, descriptor)
    assert (output / "weights.safetensors").read_bytes() == b"a different experiment"


def test_qualified_source_identity_is_the_one_shipped_in_the_course():
    descriptor = json.loads(RELEASE.read_text(encoding="utf-8"))
    for relative, digest in descriptor["implementation"].items():
        assert sha256((ROOT / relative).read_bytes()) == digest
    summary = ROOT / "docs/decision-product/v3/qualification-summary.json"
    assert sha256(summary.read_bytes()) == descriptor["qualification_summary_sha256"]


@pytest.mark.parametrize("incoming", [{"text": "add three numbers"}, [], None])
def test_dispatch_cannot_claim_new_c_supports_legacy_text_auto(incoming):
    with pytest.raises(ValueError, match="devices"):
        dispatch_request(None, incoming)
