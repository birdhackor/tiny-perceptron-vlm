"""離線核對資料包的雜湊、檔案邊界與不覆寫契約。"""

import hashlib
import io
import tarfile

import pytest

from tiny_perceptron.assets import sha256, unpack_asset


def make_asset(tmp_path, name="samples/train.jsonl"):
    archive = tmp_path / "sample.tar.gz"
    payloads = {name: b'{"text":"one"}\n', "samples/LICENSE": b"original terms\n"}
    with tarfile.open(archive, "w:gz") as package:
        for path, content in payloads.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            package.addfile(member, io.BytesIO(content))
    asset = {
        "id": "sample",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "files": [
            {"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for path, content in payloads.items()
        ],
    }
    return archive, asset


def test_unpack_idempotent_and_preserves_edits(tmp_path):
    archive, asset = make_asset(tmp_path)
    output = tmp_path / "data"
    assert unpack_asset(archive, asset, output)["files_written"] == 2
    assert unpack_asset(archive, asset, output)["files_written"] == 0
    path = output / "samples/train.jsonl"
    path.write_text("reader edits")
    with pytest.raises(ValueError, match="既有資料不同"):
        unpack_asset(archive, asset, output)
    assert path.read_text() == "reader edits"


def test_archive_hash_verified_before_writing(tmp_path):
    archive, asset = make_asset(tmp_path)
    content = archive.read_bytes()
    archive.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    output = tmp_path / "data"
    with pytest.raises(ValueError, match="SHA-256"):
        unpack_asset(archive, asset, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute/escape"])
def test_archive_cannot_escape_destination(tmp_path, name):
    archive, asset = make_asset(tmp_path, name)
    with pytest.raises(ValueError, match="路徑"):
        unpack_asset(archive, asset, tmp_path / "data")
    assert not (tmp_path / "escape").exists()
