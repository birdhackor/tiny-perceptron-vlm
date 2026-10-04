"""Fetch, verify and open only the reviewed, immutable public practical release.

The committed manifest is the trust anchor. There is deliberately no fallback
to a mutable HF branch, a private training checkpoint, or an untrained adapter.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "docs/natural-assistant/public-release.json"
REQUIREMENTS = ROOT / "requirements-natural.txt"
MAX_RELEASE_BYTES = 128 * 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024
REQUIRED_FILES = {"adapter_model.safetensors", "adapter_config.json", "README.md", "release-provenance.json"}
ALLOWED_FILES = REQUIRED_FILES | {"runtime-versions.json", "LICENSE.md", "THIRD_PARTY_NOTICES.md"}
CODE_FILES = {"tiny_perceptron/natural_assistant.py", "tiny_perceptron/natural_ui.py"}
DEPENDENCIES = {
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "accelerate",
    "numpy",
    "pillow",
    "soundfile",
    "huggingface-hub",
    "safetensors",
    "scipy",
}
LORA_TARGETS = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
LORA_WEIGHT = re.compile(
    r"base_model\.model\..*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Release file paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != value
        or any(part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in path.parts)
    ):
        raise ValueError("Release file paths cannot escape their directory")
    return path


def exact_digest(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(rf"[a-f0-9]{{{length}}}", value):
        raise ValueError(f"{label} needs an exact lowercase {length}-hex pin")


def model_pin(value, label):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("repo"), str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value["repo"])
    ):
        raise ValueError(f"{label} needs owner/repository")
    exact_digest(value.get("revision"), 40, label)


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Public natural release needs schema_version=1")
    if manifest.get("reviewed") is not True or manifest.get("anonymous_download_verified") is not True:
        raise ValueError("The release must be reviewed and its anonymous downloads verified")
    if not isinstance(manifest.get("release_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}", manifest["release_id"]
    ):
        raise ValueError("Release id must be a single bounded directory name")
    model_pin(manifest, "Public release")
    for key in ("base_model", "asr_model"):
        model_pin(manifest.get(key), key)
    exact_digest(manifest.get("git_revision"), 40, "Source Git revision")
    for key in ("manifest_sha256", "approval_sha256", "requirements_sha256"):
        exact_digest(manifest.get(key), 64, key)
    prefix = safe_relative(manifest.get("prefix"))
    if prefix.parts != ("natural-v3", manifest["release_id"]):
        raise ValueError("Public prefix must identify exactly this natural-v3 release")
    versions = manifest.get("dependency_versions")
    if (
        not isinstance(versions, dict)
        or set(versions) != DEPENDENCIES
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[a-z0-9.-]+)?", value)
            for value in versions.values()
        )
    ):
        raise ValueError("Release must pin every practical dependency version")
    runtime = manifest.get("runtime", {})
    if (
        not isinstance(runtime, dict)
        or runtime.get("python") != "3.12"
        or set(runtime)
        != {
            "python",
            "min_pixels",
            "max_pixels",
            "max_tokens",
            "max_new_tokens",
            "seed",
        }
    ):
        raise ValueError("Release needs the reviewed Python 3.12 and inference limits")
    for key in ("min_pixels", "max_pixels", "max_tokens", "max_new_tokens"):
        if type(runtime[key]) is not int or runtime[key] <= 0:
            raise ValueError("Inference limits must be positive integers")
    if (
        runtime["min_pixels"] > runtime["max_pixels"]
        or runtime["max_pixels"] > 16_000_000
        or runtime["max_tokens"] > 32768
        or runtime["max_new_tokens"] >= runtime["max_tokens"]
        or type(runtime["seed"]) is not int
    ):
        raise ValueError("Inference limits are inconsistent or outside the student route")
    if type(manifest.get("adapter_parameters")) is not int or not 0 < manifest["adapter_parameters"] < 50_000_000:
        raise ValueError("Release must state a bounded adapter parameter count")
    code = manifest.get("code_files")
    if not isinstance(code, dict) or set(code) != CODE_FILES:
        raise ValueError("Release must bind the tested chat core and interface bytes")
    for path, digest in code.items():
        safe_relative(path)
        exact_digest(digest, 64, "Student source file")
    files = manifest.get("files")
    if not isinstance(files, list) or not 4 <= len(files) <= len(ALLOWED_FILES):
        raise ValueError("Release needs a bounded inference-only file allowlist")
    remote, local, total = set(), set(), 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each public file needs reviewed metadata")
        path, output = safe_relative(item.get("path")), safe_relative(item.get("output"))
        if path.parts[: len(prefix.parts)] != prefix.parts or len(path.parts) <= len(prefix.parts):
            raise ValueError("Public file is outside the approved release prefix")
        if output.as_posix() not in ALLOWED_FILES or path.name != output.name:
            raise ValueError("Only explicitly named inference adapter, config, card and version files may be fetched")
        if path.as_posix() in remote or output.as_posix() in local:
            raise ValueError("Release paths must be unique")
        remote.add(path.as_posix())
        local.add(output.as_posix())
        if type(item.get("bytes")) is not int or not 0 < item["bytes"] <= MAX_RELEASE_BYTES:
            raise ValueError("Each public file needs a bounded positive byte length")
        total += item["bytes"]
        exact_digest(item.get("sha256"), 64, "Public file SHA-256")
        if (
            item.get("redistribution_approved") is not True
            or not isinstance(item.get("license"), str)
            or not item["license"].strip()
        ):
            raise ValueError("Every public file needs its reviewed redistribution license")
    if not REQUIRED_FILES <= local or total > MAX_RELEASE_BYTES:
        raise ValueError("Release is incomplete or too large for the inference-only student route")
    return manifest


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON files cannot contain duplicate keys")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def verify_adapter(directory, manifest):
    """Check inert safetensors layout and LoRA-only keys before PEFT loads it."""
    config = read_json(directory / "adapter_config.json")
    if not isinstance(config, dict):
        raise ValueError("Adapter config must be a JSON object")
    rank = config.get("r")
    if (
        config.get("peft_type") != "LORA"
        or config.get("task_type") != "CAUSAL_LM"
        or config.get("inference_mode") is not True
        or config.get("base_model_name_or_path") != manifest["base_model"]["repo"]
        or config.get("revision") not in (None, manifest["base_model"]["revision"])
        or type(rank) is not int
        or not 0 < rank <= 256
        or config.get("target_modules") != LORA_TARGETS
        or config.get("modules_to_save") is not None
        or config.get("auto_mapping") is not None
        or config.get("bias", "none") != "none"
        or config.get("use_dora", False)
        or config.get("rank_pattern", {})
    ):
        raise ValueError("Adapter config is not the reviewed language-attention LoRA architecture")
    file = directory / "adapter_model.safetensors"
    with file.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("Adapter safetensors header is missing")
        length = struct.unpack("<Q", raw_length)[0]
        if not 0 < length <= MAX_HEADER_BYTES or length + 8 >= file.stat().st_size:
            raise ValueError("Adapter safetensors header has invalid bounds")
        header = json.loads(stream.read(length), object_pairs_hook=unique_object)
    if not isinstance(header, dict) or header.get("__metadata__", {}) not in ({}, {"format": "pt"}):
        raise ValueError("Adapter cannot carry unreviewed tensor metadata")
    shapes, intervals, parameters = {}, [], 0
    payload_bytes = file.stat().st_size - 8 - length
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if (
            not LORA_WEIGHT.fullmatch(name)
            or not isinstance(tensor, dict)
            or set(tensor) != {"dtype", "shape", "data_offsets"}
        ):
            raise ValueError("Public weights must contain only the declared language q/v LoRA tensors")
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if (
            dtype not in {"F32", "F16", "BF16"}
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(size) is not int or not 0 < size <= 65536 for size in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or not 0 <= offsets[0] < offsets[1] <= payload_bytes
            or offsets[1] - offsets[0] != shape[0] * shape[1] * (4 if dtype == "F32" else 2)
        ):
            raise ValueError("LoRA tensor shape, dtype or byte bounds are invalid")
        shapes[name] = shape
        intervals.append(offsets)
        parameters += shape[0] * shape[1]
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("LoRA tensor payload has overlapping or undeclared bytes")
        cursor = end
    if not shapes or cursor != payload_bytes or parameters != manifest["adapter_parameters"]:
        raise ValueError("Adapter tensor count or payload differs from the reviewed release")
    for name, shape in shapes.items():
        is_a = ".lora_A." in name
        partner = name.replace(".lora_A.", ".lora_B.") if is_a else name.replace(".lora_B.", ".lora_A.")
        if partner not in shapes or shape[0 if is_a else 1] != rank:
            raise ValueError("Every LoRA projection requires a compatible A/B rank pair")
    provenance = read_json(directory / "release-provenance.json")
    if not isinstance(provenance, dict) or set(provenance) != {
        "approval_sha256",
        "git_revision",
        "manifest_sha256",
        "source",
        "base_model",
        "asr_model",
    }:
        raise ValueError("Public provenance must contain only the declared source/model pins")
    source = provenance["source"]
    model_pin(source, "Reviewed adapter source")
    if set(source) != {"repo", "revision", "prefix"} or safe_relative(source.get("prefix")).parts[0] != "natural-v3":
        raise ValueError("Public provenance cannot include private training state")
    for key in ("approval_sha256", "git_revision", "manifest_sha256", "base_model", "asr_model"):
        if provenance.get(key) != manifest[key]:
            raise ValueError("Public source/model provenance differs from the committed student manifest")
    if not (directory / "README.md").read_text(encoding="utf-8").strip():
        raise ValueError("Public model card cannot be empty")
    versions_file = directory / "runtime-versions.json"
    if versions_file.exists():
        versions = read_json(versions_file)
        if (
            not isinstance(versions, dict)
            or set(versions) != {"python", "dependencies"}
            or versions.get("python") != "3.12"
            or versions.get("dependencies") != manifest["dependency_versions"]
        ):
            raise ValueError("Public dependency versions differ from the reviewed student manifest")


def verify_release(manifest, directory, *, receipt=True):
    validate_manifest(manifest)
    directory = Path(directory).absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Student release must be a real existing directory")
    expected = {item["output"] for item in manifest["files"]}
    if receipt:
        expected.add("verified-release.json")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Student release cannot contain symbolic links")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != expected:
        raise ValueError("Student folder must contain exactly the approved files and download receipt")
    for item in manifest["files"]:
        file = directory / item["output"]
        if file.stat().st_size != item["bytes"] or sha256(file) != item["sha256"]:
            raise ValueError(f"Public file does not match its exact SHA/size: {item['output']}")
    verify_adapter(directory, manifest)
    if receipt and read_json(directory / "verified-release.json") != manifest:
        raise ValueError("Local download receipt differs from the committed public manifest")
    return directory


def fetch_release(manifest, output, *, downloader=None):
    validate_manifest(manifest)
    target = Path(output).absolute()
    if target.exists() or target.is_symlink():
        raise ValueError("Download folder already exists; verify it or choose another --output")
    if any(parent.is_symlink() for parent in target.parents):
        raise ValueError("Download destination cannot pass through a symbolic link")
    target.parent.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    with tempfile.TemporaryDirectory(prefix=".natural-release-", dir=target.parent) as temporary:
        stage = Path(temporary)
        for item in manifest["files"]:
            cached = Path(downloader(manifest["repo"], item["path"], revision=manifest["revision"], token=False))
            if not cached.is_file() or cached.stat().st_size != item["bytes"] or sha256(cached) != item["sha256"]:
                raise ValueError(f"Anonymous download differs from the reviewed SHA/size: {item['path']}")
            shutil.copyfile(cached, stage / item["output"])
        verify_release(manifest, stage, receipt=False)
        (stage / "verified-release.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(target)
    return verify_release(manifest, target)


def check_runtime(manifest):
    """Catch the common mistake of opening this model in the tiny CPU venv."""
    if sys.version_info[:2] != (3, 12):
        raise ValueError("此成品請使用獨立的 Python 3.12 環境；原本教材的 .venv 繼續保留。")
    if sha256(REQUIREMENTS) != manifest["requirements_sha256"]:
        raise ValueError("requirements-natural.txt 與這個公開成品的版本不同，請更新專案。")
    requirements = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            match = re.fullmatch(r"([A-Za-z0-9_-]+)==([^\s]+)", line)
            if not match:
                raise ValueError("Student requirements must use only explicit package==version pins")
            requirements[match[1].lower()] = match[2]
    if requirements != manifest["dependency_versions"]:
        raise ValueError("公開成品的套件版本與安裝清單不一致。")
    for package, expected in requirements.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"缺少 {package}，請在 .venv-natural 安裝 requirements-natural.txt。") from error
        if installed.split("+", 1)[0] != expected:
            raise ValueError(f"{package} 需要 {expected}，目前是 {installed}；請重新安裝獨立環境的套件。")
    for relative, expected in manifest["code_files"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError("專案程式與這個成品驗證時的版本不同，請依公開版本更新專案。")


def student_options(manifest, directory, *, device="cuda", dtype=None, cache_dir=None, local_files_only=False):
    validate_manifest(manifest)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Student route supports the documented cpu or cuda devices")
    dtype = dtype or ("float32" if device == "cpu" else "bfloat16")
    if dtype not in {"float32", "float16", "bfloat16"} or device == "cpu" and dtype != "float32":
        raise ValueError("CPU 路線使用 float32；CUDA 可使用 bfloat16 或 float16。")
    runtime = manifest["runtime"]
    return SimpleNamespace(
        model=manifest["base_model"]["repo"],
        model_revision=manifest["base_model"]["revision"],
        asr_model=manifest["asr_model"]["repo"],
        asr_revision=manifest["asr_model"]["revision"],
        adapter=Path(directory),
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        min_pixels=runtime["min_pixels"],
        max_pixels=runtime["max_pixels"],
        max_tokens=runtime["max_tokens"],
        max_new_tokens=runtime["max_new_tokens"],
        seed=runtime["seed"],
        output=ROOT / "outputs/natural-student",
        data_root=ROOT / "outputs/natural-student/ui-data",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/natural-assistant/release")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify", action="store_true", help="Verify an existing student folder without downloading")
    modes.add_argument(
        "--serve", action="store_true", help="Verify an existing folder and open the pinned local chat core"
    )
    modes.add_argument("--list", action="store_true", help="Show the actual published pins; do not download")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error("此完整版尚未發布可驗證的公開權重；不會改抓 main 或私人訓練檔。")
    try:
        manifest = validate_manifest(read_json(args.manifest))
        if args.list:
            print(
                json.dumps(
                    {key: manifest[key] for key in ("release_id", "repo", "revision", "base_model", "asr_model")},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if args.verify or args.serve:
            directory = verify_release(manifest, args.output)
        else:
            directory = fetch_release(manifest, args.output)
        if args.serve:
            check_runtime(manifest)
            options = student_options(
                manifest,
                directory,
                device=args.device,
                dtype=args.dtype,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            )
            import torch

            from tiny_perceptron.natural_ui import serve

            if args.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("找不到可用的 NVIDIA GPU；請確認 CUDA 安裝，或改用 --device cpu。")
            if args.device == "cuda" and options.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("這張 GPU 不支援 bfloat16，請加上 --dtype float16。")
            if not 0 <= args.port <= 65535:
                raise ValueError("port 必須是 0 到 65535 的整數。")
            serve(options, port=args.port)
        else:
            print(str(directory))
    except (KeyError, ValueError, OSError, UnicodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
