"""Downloaded student weights must match the public release, not a moving branch."""

import hashlib

import pytest

from scripts.fetch_course_models import fetch_model


def test_download_pins_public_revision_and_verifies_bytes(tmp_path, monkeypatch):
    source = tmp_path / "cached.pt"
    source.write_bytes(b"reviewed weights")
    calls = []

    def download(repo, filename, **options):
        calls.append((repo, filename, options))
        return str(source)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    revision = "a" * 40
    spec = {
        "id": "sft",
        "repo": "example/public",
        "revision": revision,
        "files": [
            {
                "path": "course/sft/model.pt",
                "output": "model.pt",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": source.stat().st_size,
            }
        ],
    }
    folder = fetch_model(spec, tmp_path / "downloads")
    assert (folder / "model.pt").read_bytes() == source.read_bytes()
    assert calls == [("example/public", "course/sft/model.pt", {"revision": revision, "token": False})]
    source.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="does not match"):
        fetch_model(spec, tmp_path / "downloads")
    spec["revision"] = "main"
    with pytest.raises(ValueError, match="pinned HF commit"):
        fetch_model(spec, tmp_path / "downloads")


@pytest.mark.parametrize("bad_path", ["../escape.pt", "/absolute.pt", "..\\escape.pt"])
def test_manifest_cannot_write_outside_selected_model(tmp_path, bad_path):
    spec = {"id": bad_path, "revision": "a" * 40, "files": []}
    with pytest.raises(ValueError, match="Invalid model file path"):
        fetch_model(spec, tmp_path)
