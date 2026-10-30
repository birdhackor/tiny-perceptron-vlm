"""外部Kev只下載到隔離工作素材；普通Git保存版本、授權與逐檔SHA。"""

import hashlib
import urllib.request
from pathlib import Path

REVISION = "2074cc44256200bb37540103fa4572901faa10f3"
FILES = {
    "kev/__init__.py": (0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "kev/checkpoint.py": (23232, "02062399977ee0da316dd52e03a79fa5e10134aa49c0b5ed801598c52042dc32"),
    "kev/model.py": (35153, "2ff3b2e67c048e3bc5a0d1a32aafe50804ed315a5564fce06bb4b95697c5271c"),
    "kev/shared_prefix.py": (9671, "bc6dedbad734cca227f3ec6746223b3b03363b0c045aa9f2493d886b1828e71e"),
    "LICENSE": (11343, "6b08bb37982c233aa12bcbdf19106da12f3f4fcf800773ea42e28ebeddd34fb8"),
}


def source_url(name):
    if name not in FILES:
        raise ValueError("不在事前列明的Kev來源範圍")
    return f"https://raw.githubusercontent.com/jaredpalmer/kev/{REVISION}/{name}"


def verify(root):
    root = Path(root)
    for name, (size, expected) in FILES.items():
        path = root / name
        if (
            path.is_symlink()
            or path.stat().st_size != size
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("外部Kev程式與固定版本不一致")
    return {
        name: {"source_url": source_url(name), "bytes": size, "sha256": digest}
        for name, (size, digest) in FILES.items()
    }


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for name, (size, expected) in FILES.items():
        url = source_url(name)
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.url != url:
                raise ValueError("外部固定來源不接受轉址")
            raw = response.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("外部來源下載SHA／大小不符")
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return {"revision": REVISION, "license": "Apache-2.0", "files": verify(root), "ordinary_git_fulltext": False}
