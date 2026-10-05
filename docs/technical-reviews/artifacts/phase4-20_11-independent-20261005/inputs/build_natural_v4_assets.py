"""Build immutable, attributed v4 LFS bundles from reviewed, fixed source selections.

Reconstruction retrieves the same bytes and crop specifications. It never chooses
new questions, changes labels, trains a model, or consumes a private credential.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import io
import json
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "docs/natural-assistant/v4/data"
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
MAX_AUDIO_ARCHIVE_READ = 64 * 1024 * 1024
VOICE_PINS = {
    "voice-sources.json": "https://huggingface.co/datasets/google/fleurs/resolve/d7c758a6dceecd54a98cac43404d3d576e721f07/",
    "voice-question-sources.json": "https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/bbe295d530192a4cd41644b711c9aecd087df653/",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(name):
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or "\x00" in name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or path.as_posix() != name
    ):
        raise ValueError("Dataset files must have canonical relative POSIX paths")
    return path


def safe_destination(data_root, name):
    root = Path(data_root).absolute()
    target = root / relative(name)
    for candidate in (root, *target.parents, target):
        if candidate.is_symlink():
            raise ValueError("Dataset paths must not contain symlinks")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset destination escapes its data root")
    return target


def regular_file(data_root, name):
    path = safe_destination(data_root, name)
    if not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Snapshots require existing regular files inside their data root")
    return path


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def download_exact(url, path, expected_bytes, expected_sha256):
    if not url.startswith("https://") or not 0 < expected_bytes <= MAX_DOWNLOAD_BYTES:
        raise ValueError("Public downloads must be bounded HTTPS files")
    path = Path(path)
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        raise ValueError("Download destination must not contain symlinks")
    if path.is_file() and path.stat().st_size == expected_bytes and sha256(path) == expected_sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-perceptron-vlm-fixed-v4/1"})
    try:
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                    count = 0
                    while block := response.read(min(1024 * 1024, expected_bytes + 1 - count)):
                        count += len(block)
                        if count > expected_bytes:
                            raise ValueError("Source download exceeds its frozen byte count")
                        stream.write(block)
                if temporary.stat().st_size != expected_bytes or sha256(temporary) != expected_sha256:
                    raise ValueError("Source download differs from its frozen bytes/SHA-256")
                temporary.replace(path)
                return
            except urllib.error.HTTPError as error:
                if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(2**attempt)
    finally:
        temporary.unlink(missing_ok=True)


class BoundedReader:
    def __init__(self, stream, maximum=MAX_AUDIO_ARCHIVE_READ):
        self.stream, self.maximum, self.bytes_read = stream, maximum, 0

    def read(self, size=-1):
        if size < 0:
            size = self.maximum - self.bytes_read + 1
        raw = self.stream.read(min(size, self.maximum - self.bytes_read + 1))
        self.bytes_read += len(raw)
        if self.bytes_read > self.maximum:
            raise ValueError("Fixed audio selection exceeds the bounded source stream")
        return raw


def reconstruct_voices(data_root):
    groups = {}
    destinations = set()
    for name, prefix in VOICE_PINS.items():
        document = json_file(METADATA / name)
        for row in document["audio_rows"]:
            if row["split"] == "train":
                continue
            url = (
                row["source_archive"]["url"]
                if "source_archive" in row
                else document["source_files"][row["source_split"]]["audio_archive"]["url"]
            )
            if not url.startswith(prefix) or any(part in {".", ".."} for part in url[len(prefix) :].split("/")):
                raise ValueError("Voice source must be a fixed revision of its official corpus")
            destination = safe_destination(data_root, "voice/" + str(relative(row["audio"])))
            if destination in destinations:
                raise ValueError("Duplicate frozen recording destination")
            destinations.add(destination)
            groups.setdefault(url, []).append(row)
    helper = Path(__file__).with_name("replay_natural_v4_voice.py")
    spec = importlib.util.spec_from_file_location("fixed_v4_voice_replay", helper)
    voice = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(voice)
    for item in groups.items():
        voice.replay_archive(item, Path(data_root) / "voice")


def reconstruct(data_root):
    pictures = []
    for name in ("vision-sources.json", "vision-activity-sources.json", "vision-activity-heldout-sources.json"):
        document = METADATA / name
        if document.is_file():
            pictures.extend(json_file(document)["rows"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                download_exact, row["url"], safe_destination(data_root, row["image"]), row["bytes"], row["sha256"]
            )
            for row in pictures
        ]
        for future in futures:
            future.result()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_natural_v4_ocr.py"),
            "--sources",
            str(METADATA / "ocr-sources.json"),
            "--output",
            str(data_root),
        ],
        check=True,
    )
    reconstruct_voices(data_root)


def assemble():
    rows = []
    for path in sorted((METADATA / "vision-labels").glob("*.jsonl")):
        rows.extend(json_lines(path))
    rows.extend(json_lines(METADATA / "chat-labels.jsonl"))
    used_presence = set()
    ocr_labels = json_lines(METADATA / "ocr-labels.jsonl")
    extra_order = METADATA / "ocr-order-labels.jsonl"
    if extra_order.is_file():
        ocr_labels.extend(json_lines(extra_order))
    for original in ocr_labels:
        row = dict(original)
        row["id"] = "ocr-v4:" + original["id"]
        row["image"] = str(relative(original["image_relative"]))
        row["user"] = original["question"]
        row.setdefault("history", [])
        if original["task"] == "chinese_presence":
            row["task"] = "text_presence"
            row["references"] = {"kind": "exact", "accepted": [original["answer"]]}
            used_presence.add((row["image"], row["user"]))
        elif original["task"] == "ocr_transcription":
            ordered = "\n" in original["answer"]
            row["task"] = "ocr_order" if ordered else "ocr"
            row["references"] = {
                "kind": "ocr_order" if ordered else "ocr",
                "text": original["answer"],
                "strip_whitespace": not ordered,
                "policy": "NFKC; preserve case, punctuation and original Chinese script. Single-line tasks ignore Unicode whitespace; explicit multi-line tasks preserve internal spaces/newlines and reading order.",
            }
        else:
            raise ValueError("Unknown curated OCR task")
        rows.append(row)
    for row in json_lines(METADATA / "presence-root-labels.jsonl"):
        key = (row["image"], row["user"])
        if key not in used_presence:
            rows.append(row)
            used_presence.add(key)
    audio_rows = []
    for name, task in (("voice-sources.json", "speech_transcription"), ("voice-question-sources.json", "speech_chat")):
        for original in json_file(METADATA / name)["audio_rows"]:
            if original["split"] == "train":
                continue
            row = dict(original, task=task, audio="voice/" + str(relative(original["audio"])))
            row["original_source_task"] = original["task"]
            if task == "speech_chat":
                row["references"] = {
                    "kind": "manual",
                    "rubric": original["semantic_rubric"],
                    "scope": "Human read question, paired reference-text / actual-ASR responses; not spontaneous conversation accuracy.",
                }
            audio_rows.append(row)
    families, identifiers = {}, set()
    for row in rows + audio_rows:
        if row["id"] in identifiers:
            raise ValueError("Repeated dataset row ID")
        identifiers.add(row["id"])
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split")
        if row["family"] in families and families[row["family"]] != row["split"]:
            raise ValueError("Related dataset family crosses splits")
        families[row["family"]] = row["split"]
        for key in ("image", "audio"):
            if row.get(key):
                relative(row[key])
    return rows, audio_rows


def validate_assets(data_root, rows, audio_rows, sources):
    """Bind labels to their original source and check bytes before any packaging."""
    declarations, source_splits = {}, {}

    def identity(digest, split):
        previous = source_splits.setdefault(digest, split)
        if previous != split:
            raise ValueError("Identical source bytes cross dataset splits")

    def declare(name, size, digest, family, split):
        name = str(relative(name))
        entry = {"bytes": size, "sha256": digest, "family": family, "split": split}
        if name in declarations and declarations[name] != entry:
            raise ValueError("A physical asset has conflicting frozen source declarations")
        declarations[name] = entry
        identity(digest, split)

    documents = {Path(source["path"]).name: source["metadata"] for source in sources}
    for name in ("vision-sources.json", "vision-activity-sources.json", "vision-activity-heldout-sources.json"):
        for source in documents.get(name, {}).get("rows", []):
            declare(source["image"], source["bytes"], source["sha256"], source["family"], source["split"])
    ocr = documents.get("ocr-sources.json", {})
    originals = {source["source_id"]: source for source in ocr.get("sources", [])}
    if len(originals) != len(ocr.get("sources", [])):
        raise ValueError("Duplicate OCR original source declaration")
    for source in originals.values():
        identity(source["image_sha256"], source["split"])
    for artifact in ocr.get("artifacts", []):
        if artifact["source_id"] not in originals:
            raise ValueError("Crop has no declared original source")
        original = originals[artifact["source_id"]]
        declare(artifact["path"], artifact["bytes"], artifact["sha256"], original["family"], original["split"])
    for name in VOICE_PINS:
        for source in documents.get(name, {}).get("audio_rows", []):
            if source["split"] != "train":
                declare(
                    "voice/" + source["audio"], source["bytes"], source["sha256"], source["family"], source["split"]
                )
    selected = set()
    for row in rows + audio_rows:
        for key in ("image", "audio"):
            if not row.get(key):
                continue
            name = row[key]
            if name not in declarations:
                raise ValueError("Dataset asset is absent from frozen source declarations")
            expected = declarations[name]
            if (row["family"], row["split"]) != (expected["family"], expected["split"]):
                raise ValueError("Question changes its physical source family or split")
            if row.get("image_sha256") and row["image_sha256"] != expected["sha256"]:
                raise ValueError("Question image SHA differs from frozen source")
            selected.add(name)
    for name in sorted(selected):
        path = regular_file(data_root, name)
        expected = declarations[name]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Local asset differs from its frozen source bytes/SHA-256")
    return selected


def write_notices(data_root, sources):
    documents = {Path(source["path"]).name: source["metadata"] for source in sources}
    notices = {
        "vision": (
            "DOCCI photographs and source descriptions\n"
            "Attribution: Google LLC; photographs by Jason Baldridge and family.\n"
            "License: Creative Commons Attribution 4.0 International.\n"
            "https://creativecommons.org/licenses/by/4.0/\n"
            "Source: https://huggingface.co/datasets/google/docci\n"
            "Changes: fixed photograph subset; AI-authored Traditional-Chinese questions and answers.\n"
            "Individual pinned URLs, copyright information, source text and review provenance: ATTRIBUTION.json.\n"
        ),
        "ocr": (
            "NVIDIA OCR-Synthetic-Multilingual-v1 and individually licensed Wikimedia Commons photographs\n"
            "NVIDIA Corporation: Creative Commons Attribution 4.0 International.\n"
            "https://creativecommons.org/licenses/by/4.0/\n"
            "Commons photos: each author, license, source page and modification are listed in ATTRIBUTION.json.\n"
            "This subset uses only CC BY 3.0/4.0, CC0 or public-domain photo sources.\n"
            "https://creativecommons.org/licenses/by/3.0/\n"
            "https://creativecommons.org/publicdomain/zero/1.0/\n"
            "Changes: image crops and AI-authored visible-region Chinese transcriptions.\n"
            "DOCCI negatives retain the photograph attribution and CC BY 4.0 license described in ATTRIBUTION.json.\n"
        ),
        "voice": (
            "FLEURS original Mandarin human recordings and transcripts\n"
            "Attribution: Google FLEURS contributors.\n"
            "License: Creative Commons Attribution 4.0 International.\n"
            "https://creativecommons.org/licenses/by/4.0/\n"
            "AISHELL-1 original recordings and transcripts: Beijing Shell Shell Technology Co., Ltd.\n"
            "License: Apache License 2.0; see AISHELL-Apache-2.0-LICENSE.txt.\n"
            "Official license source: https://www.openslr.org/33/\n"
            "Changes: fixed source subsets; original WAV bytes and transcripts preserved.\n"
            "These are human read recordings, not spontaneous conversations.\n"
            "Exact contributor attribution, source URLs, pins and per-file hashes: ATTRIBUTION.json.\n"
        ),
    }
    files = {}
    for kind, notice in notices.items():
        name = f"{kind}/NOTICE.txt"
        target = safe_destination(data_root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(notice, encoding="utf-8", newline="\n")
        files[kind] = {name}
    if documents.get("voice-question-sources.json", {}).get("audio_rows"):
        license_path = ROOT / "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE"
        name = "voice/AISHELL-Apache-2.0-LICENSE.txt"
        safe_destination(data_root, name).write_bytes(license_path.read_bytes())
        files["voice"].add(name)
    return files


def package_selected(data_root, names, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with destination.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(names):
                path = regular_file(data_root, name)
                content = path.read_bytes()
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(content), 0o644, 0
                archive.addfile(member, io.BytesIO(content))
                files.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return files


def build(data_root, assets, output):
    rows, audio_rows = assemble()
    source_names = (
        "vision-sources.json",
        "vision-activity-sources.json",
        "vision-activity-heldout-sources.json",
        "ocr-sources.json",
        "chat-sources.json",
        "voice-sources.json",
        "voice-question-sources.json",
    )
    sources = [
        {
            "path": str((METADATA / name).relative_to(ROOT)),
            "sha256": sha256(METADATA / name),
            "metadata": json_file(METADATA / name),
        }
        for name in source_names
        if (METADATA / name).is_file()
    ]
    attribution = {
        "dataset_version": "natural-assistant-v4",
        "sources": sources,
        "label_notice": "Traditional-Chinese photo QA, Commons region transcriptions and course chat adaptations are AI-authored; source language labels and actual inspection provenance are preserved. No claim of independent human Chinese gold.",
        "changes": "Fixed source subsets, explicitly attributed Chinese adaptations and image region crops. No audio model is trained in this branch.",
    }
    archives, files = [], []
    names = validate_assets(data_root, rows, audio_rows, sources)
    notices = write_notices(data_root, sources)
    for kind in ("vision", "ocr", "voice"):
        chosen = {name for name in names if PurePosixPath(name).parts[0] == kind}
        extra = f"{kind}/ATTRIBUTION.json"
        target = safe_destination(data_root, extra)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical(attribution))
        chosen.add(extra)
        chosen.update(notices[kind])
        archive = assets / f"natural-{kind}-v4.tar.gz"
        members = package_selected(data_root, chosen, archive)
        files.extend(members)
        archives.append(
            {
                "path": f"assets/training/{archive.name}",
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
                "files": members,
            }
        )
    if sum(item["bytes"] for item in archives) > 512 * 1024 * 1024:
        raise ValueError("Combined compressed snapshots exceed the remote preparation limit")
    manifest = {
        "schema_version": 1,
        "dataset_version": "natural-assistant-v4",
        "sources": sources,
        "annotation_sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(METADATA.rglob("*.jsonl"))
        ],
        "archives": archives,
        "files": files,
        "rows": rows,
        "audio_rows": audio_rows,
        "counts": {
            split: dict(Counter(row["task"] for row in rows + audio_rows if row["split"] == split))
            for split in ("train", "validation", "test")
        },
        "split_policy": "DOCCI similarity clusters, Commons physical source/photograph families, OpenAssistant message trees and audio sentence/speaker families are grouped before their derived questions. Source SHA checks prevent byte duplicate leakage. Unknown foundation pretraining overlap is not ruled out.",
        "scope": "Bounded teaching sample. Only a fresh language q/v LoRA is trained. FLEURS evaluates transcription; AISHELL human read questions evaluate paired typed-reference and actual-ASR chat. Preserve independent denominators and do not call read-question performance unrestricted voice-chat accuracy.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "outputs/natural-v4/data")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets/training")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/natural-assistant/v4/manifest.json")
    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args()
    expected = json_file(args.verify_manifest) if args.verify_manifest else None
    if args.reconstruct:
        reconstruct(args.data_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v4-manifest-", dir=args.manifest.parent) as temporary:
        candidate = Path(temporary) / "manifest.json"
        manifest = build(args.data_root, args.assets, candidate)
        if expected is not None and manifest != expected:
            raise ValueError("Reconstructed manifest differs from the exact committed snapshot")
        candidate.replace(args.manifest)
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "archive_bytes": sum(item["bytes"] for item in manifest["archives"]),
                "manifest_sha256": sha256(args.manifest),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
