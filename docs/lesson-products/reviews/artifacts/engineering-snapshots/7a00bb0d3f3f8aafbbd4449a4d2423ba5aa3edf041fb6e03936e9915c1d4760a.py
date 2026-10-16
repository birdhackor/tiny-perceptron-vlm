"""只測虛構發布包及拒絕邊界；不連網、不reserve、不啟動Modal。"""

import hashlib
import io
import tarfile

import pytest

from scripts.lesson_products.publication import unpack_verified


def archive(tmp_path, entries):
    path = tmp_path / "transport.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                stream.addfile(member)
            else:
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_transport_bytes_are_checked_before_extract_and_scope_is_bounded(tmp_path):
    path, digest = archive(tmp_path, [("b/model.pt", b"fictional, not weights", "file")])
    with pytest.raises(ValueError, match="SHA"):
        unpack_verified(path, "0" * 64, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    unpack_verified(path, digest, tmp_path / "good")
    assert (tmp_path / "good/b/model.pt").read_bytes() == b"fictional, not weights"
    with pytest.raises(ValueError, match="全新"):
        unpack_verified(path, digest, tmp_path / "good")


@pytest.mark.parametrize(
    "entries",
    [
        [("../outside", b"bad", "file")],
        [("/outside", b"bad", "file")],
        [("b/alias", b"", "symlink")],
        [("x", b"first", "file"), ("x", b"second", "file")],
    ],
)
def test_transport_rejects_outside_paths_links_and_duplicates(tmp_path, entries):
    path, digest = archive(tmp_path, entries)
    with pytest.raises(ValueError, match="相對路徑"):
        unpack_verified(path, digest, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()
