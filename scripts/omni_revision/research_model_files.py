"""完整固定公開模型檔案；模型載入只讀local snapshot，避免附屬檔落回main。"""

import fnmatch
import hashlib
from pathlib import Path, PurePosixPath

MODELS = {
    "Qwen/Qwen2.5-Omni-3B": ("f75b40e3da2003cdd6e1829b1f420ca70797c34e", 20 * 1024**3),
    "openai/whisper-small": ("973afd24965f72e36ca33b3055d56a652f456b4d", 1024**3),
}
PATTERNS = ("*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja", "*.model", "*.tiktoken")


def file_digests(path):
    size = path.stat().st_size
    sha256, git_sha1 = hashlib.sha256(), hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha256.update(block)
            git_sha1.update(block)
    return sha256.hexdigest(), git_sha1.hexdigest()


def frozen_snapshot(model, revision):
    from huggingface_hub import HfApi, snapshot_download

    if model not in MODELS or revision != MODELS[model][0]:
        raise ValueError("成熟模型只下載明示固定的完整版本")
    info = HfApi(token=False).model_info(model, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("公開模型metadata並非要求的commit")
    files = [row for row in info.siblings if any(fnmatch.fnmatchcase(row.rfilename, pattern) for pattern in PATTERNS)]
    names = {row.rfilename for row in files}
    if (
        len(files) != len(names)
        or not 0 < len(files) <= 128
        or any(type(row.size) is not int or row.size < 0 for row in files)
        or sum(row.size for row in files) > MODELS[model][1]
    ):
        raise ValueError("公開模型檔案數／下載bytes超出固定上界")
    if (
        "config.json" not in names
        or not any(name.endswith(".safetensors") for name in names)
        or (model.startswith("Qwen/") and "spk_dict.pt" not in names)
    ):
        raise ValueError("完整固定模型缺config／神經權重／speaker")
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("公開模型metadata含非法檔案路徑")
    root = Path(
        snapshot_download(repo_id=model, revision=revision, token=False, allow_patterns=sorted(names), max_workers=4)
    )
    if root.name != revision or not root.is_dir():
        raise ValueError("下載沒有落在精確commit snapshot")
    observed = {}
    for row in files:
        path = root / row.rfilename
        if path.stat().st_size != row.size:
            raise ValueError("固定模型檔案大小與metadata不同")
        digest, git_blob = file_digests(path)
        expected = getattr(row.lfs, "sha256", None) if row.lfs is not None else None
        if (expected is not None and digest != expected) or (expected is None and git_blob != row.blob_id):
            raise ValueError("固定模型檔案的LFS SHA256／Git blob身分不同")
        observed[row.rfilename] = {
            "bytes": row.size,
            "sha256": digest,
            "source_blob_id": row.blob_id,
            "source_lfs_sha256": expected,
        }
    return root, {
        "schema": "omni-fixed-public-model-input-files-v1",
        "model": model,
        "revision": revision,
        "local_files_only_for_neural_load": True,
        "implicit_tokens_permitted": False,
        "files": observed,
        "ordinary_git_weight_bytes": 0,
    }


def component_parameter_dtypes(model):
    """只記參數的實際dtype／numel，不以一個model.dtype代表所有零件或所有算子。"""
    result = {}
    for name in ("thinker", "talker", "token2wav"):
        counts = {}
        for parameter in getattr(model, name).parameters():
            key = str(parameter.dtype)
            counts[key] = counts.get(key, 0) + parameter.numel()
        result[name] = counts
    return result
