"""Make an optional external proof snapshot; never upload or modify Git state."""

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = "docs/natural-assistant/evidence/v4-research"
ALLOWED_DIRECTORIES = (
    "asset-helper-review",
    "asr-cpu-validation",
    "chat-gold-review",
    "ci-portability-review",
    "data-helper-review",
    "experiment",
    "full-token-audit",
    "ocr-gold-review",
    "ocr-order-author-review",
    "ocr-order-peer-review",
    "source-status-review",
    "vision",
    "vision-activity-peer-review",
    "vision-gold-review",
    "vision-labels-b",
    "vision-labels-root",
    "workflow-helper-review",
)
CHAT_FILES = (
    "AISHELL1-license-source-receipt.json",
    "AISHELL1-transcript-first-1MiB.raw.txt",
    "AISHELL1-transcript-prefix-inspection.json",
    "chat-curation-initial-summary.json",
    "chat-data-integrity-check.json",
    "freeze-chat-ledger-source.py.txt",
    "fleurs-train-small-actual-inspection.json",
    "oasst2-candidate-filter-receipt.json",
    "oasst2-candidate-relaxed-filter-receipt.json",
    "oasst2-data-download-receipt.json",
    "oasst2-selected-original-messages.jsonl",
)
SOURCE_FILES = (
    "vision-sources.json",
    "vision-activity-sources.json",
    "vision-activity-heldout-sources.json",
    "ocr-sources.json",
    "chat-sources.json",
    "voice-sources.json",
    "voice-question-sources.json",
)
VISION_SOURCE_FILES = {
    "docci-descriptions.jsonl",
    "docci-clusters.jsonl",
    "docci-image-archive-object-metadata.json",
    "docci-descriptions-object-metadata.json",
    "source-fetch-receipt.json",
    "secondary-fetch-receipt.json",
    "tertiary-fetch-receipt.json",
    "coco-terms-fetch-receipt.json",
}
EXCLUDED_EXPERIMENT_CARDS = {"qwen3-vl-finetune-readme.md", "qwen3-vl-readme.md", "whisper-turbo.README.md"}
MAX_TOTAL_BYTES = 512 * 1024 * 1024


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(raw)
    return value.hexdigest()


def safe_file(root, name):
    canonical_member(name)
    target = root / name
    if any(p.is_symlink() for p in (target, *target.parents)) or not target.is_file():
        raise ValueError("Proof sources must be regular files without symlinks")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Proof source escapes its repository")
    return target


def canonical_member(name):
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or "\x00" in name
        or path.as_posix() != name
    ):
        raise ValueError("Proof members need canonical relative POSIX paths")
    return name


def selected_files(root):
    names = set()
    base = root / EVIDENCE
    for directory in ALLOWED_DIRECTORIES:
        for path in (base / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                if base / "vision/sources" in path.parents:
                    if path.parent != base / "vision/sources" or path.name not in VISION_SOURCE_FILES:
                        continue
                if directory == "experiment" and path.name in EXCLUDED_EXPERIMENT_CARDS:
                    continue
                names.add(path.relative_to(root).as_posix())
    for path in (base / "ocr/nvidia").glob("*.jpg"):
        names.add(path.relative_to(root).as_posix())
    for name in (
        "before-review-local-rebuild-receipt.json",
        "fresh-rebuild-receipt.json",
        "nvidia-multiline-overlap-audit.json",
        "nvidia-single-overlap-audit.json",
        "rebuild-range-protection-checks.json",
    ):
        if (base / "ocr" / name).is_file():
            names.add(f"{EVIDENCE}/ocr/{name}")
    for name in CHAT_FILES:
        if (base / "chat-speech" / name).is_file():
            names.add(f"{EVIDENCE}/chat-speech/{name}")
    for path in (base / "chat-speech/fleurs-metadata").glob("*.tsv"):
        names.add(path.relative_to(root).as_posix())
    metadata = root / "docs/natural-assistant/v4/data"
    names.update(f"docs/natural-assistant/v4/data/{name}" for name in SOURCE_FILES)
    names.update(path.relative_to(root).as_posix() for path in metadata.rglob("*.jsonl"))
    names.update(
        (
            "LICENSE",
            "scripts/build_natural_v4_research_proof.py",
            "scripts/build_natural_v4_assets.py",
            "scripts/fetch_natural_data.py",
            "scripts/prepare_natural_v4_ocr.py",
            "scripts/replay_natural_v4_voice.py",
            ".github/workflows/natural-data-v4.yml",
            "docs/natural-assistant/v4/manifest.json",
            "tests/test_natural_v4_assets.py",
            "tests/test_natural_v4_data_download.py",
            "tests/test_natural_v4_data_workflow.py",
            "tests/test_natural_v4_research_proof.py",
        )
    )
    for name in names:
        if "original-research" in name or "ocr-samples" in name:
            raise ValueError("Historical unselected OCR research is never bundled")
        safe_file(root, name)
    return sorted(names)


def license_key(value):
    normalized = re.sub(r"[\s-]", "", value).upper()
    if normalized not in {"CCBY4.0", "CCBY3.0", "CC0", "PUBLICDOMAIN", "APACHE2.0", "MIT"}:
        raise ValueError(f"Unsupported proof source license: {value}")
    return normalized


def attribution(root):
    metadata = root / "docs/natural-assistant/v4/data"
    documents = {name: json.loads((metadata / name).read_bytes()) for name in SOURCE_FILES}
    originals = {}

    def add_original(sha, record, metadata_name):
        entry = originals.setdefault(sha, record)
        reference = {"metadata": metadata_name, "dataset": record["dataset"], "source_id": record["source_id"]}
        entry.setdefault("references", []).append(reference)

    for name in SOURCE_FILES[:3]:
        for row in documents[name]["rows"]:
            license_key(row["license"])
            add_original(
                row["sha256"],
                {
                    "dataset": "google/docci",
                    "source_id": row["example_id"],
                    "sha256": row["sha256"],
                    "artist": row["image_creator"],
                    "licensor": row["licensor"],
                    "license": row["license"],
                    "license_url": row["license_url"],
                    "source_url": row["url"],
                },
                name,
            )
    for source in documents["ocr-sources.json"]["sources"]:
        license_key(source["license"])
        add_original(
            source["image_sha256"],
            {
                "dataset": source["dataset"],
                "source_id": source["source_id"],
                "sha256": source["image_sha256"],
                "artist": source.get("artist", source.get("artist_html", "")),
                "credit": source.get("credit_html", ""),
                "attribution": source.get("attribution_html", ""),
                "license": source["license"],
                "license_url": source["license_url"],
                "source_url": source.get("source_page", source.get("source_url", source["download_url"])),
            },
            "ocr-sources.json",
        )
    corpora = []
    for name, fields in (
        ("chat-sources.json", documents["chat-sources.json"]["upstream"]),
        ("voice-sources.json", documents["voice-sources.json"]),
        ("voice-question-sources.json", documents["voice-question-sources.json"]),
    ):
        license = fields.get("license", fields.get("official_corpus_license"))
        license_key(license)
        corpora.append(
            {
                "metadata": "docs/natural-assistant/v4/data/" + name,
                "license": license,
                "attribution": fields["attribution"],
                "revision": fields["revision"],
            }
        )
    return {
        "photographs_and_pages": sorted(originals.values(), key=lambda item: (item["dataset"], item["source_id"])),
        "text_and_speech_corpora": corpora,
        "modifications": "Review montages, source image enlargements/crops, AI-authored Chinese annotations, and original audit observations; source bytes and historical reports preserved.",
        "proof_code_and_reports": "Course-authored code/reports retain the repository MIT license; source text and photographs retain their individual licenses.",
        "third_party_software": {
            "project": "Hugging Face Transformers 4.57.6",
            "license": "Apache 2.0",
            "copyright": "Individual original copyright and license headers are preserved in each source file.",
            "source_url": "https://github.com/huggingface/transformers/tree/v4.57.6",
        },
    }


def package(root, names, archive, generated):
    for name in generated:
        canonical_member(name)
        if not name.startswith("PROOF/") or name == "PROOF/INDEX.json":
            raise ValueError("Generated proof metadata must use its reserved PROOF namespace")
    if set(names) & (set(generated) | {"PROOF/INDEX.json"}):
        raise ValueError("Raw and generated proof members overlap")
    files = [
        {"archive_member": name, "bytes": safe_file(root, name).stat().st_size, "sha256": file_digest(root / name)}
        for name in sorted(set(names))
    ]
    if sum(item["bytes"] for item in files) >= MAX_TOTAL_BYTES:
        raise ValueError("Optional research proof exceeds its 512 MiB bound")
    index = {
        "schema_version": 1,
        "scope": "Optional external legal v4 research-proof snapshot; not training data or an upload receipt",
        "raw_file_access": "Raw paths below are archive members; they require external archive download, SHA verification and extraction, and are not promised in a normal Git checkout.",
        "files": files,
        "exclusions": [
            "All original-research and historical ocr-samples",
            "Unselected chat/speech cards and copyrighted full texts without confirmed redistribution",
            "NC/SA candidate previews",
            "Model weights and GPU artifacts",
        ],
    }
    generated = {**generated, "PROOF/INDEX.json": canonical(index)}
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".partial")
    if temporary.exists() or archive.exists():
        raise FileExistsError("Proof snapshots are immutable; select a new output path")
    try:
        with temporary.open("xb") as output, gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as bundle:
                expected = {item["archive_member"]: item for item in files}
                for name in sorted([*expected, *generated]):
                    raw = generated[name] if name in generated else safe_file(root, name).read_bytes()
                    if name in expected and (
                        len(raw) != expected[name]["bytes"] or digest(raw) != expected[name]["sha256"]
                    ):
                        raise ValueError("Proof source changed during snapshot creation")
                    member = tarfile.TarInfo(name)
                    member.size, member.mode, member.mtime = len(raw), 0o644, 0
                    bundle.addfile(member, io.BytesIO(raw))
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return index


def verify_archive(archive, index, generated):
    expected = {item["archive_member"]: item for item in index["files"]}
    generated = {**generated, "PROOF/INDEX.json": canonical(index)}
    expected.update({name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in generated.items()})
    seen = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            canonical_member(member.name)
            item = expected.get(member.name)
            if not member.isfile() or member.name in seen or item is None or member.size != item["bytes"]:
                raise ValueError("Proof archive members differ from the closed index")
            with bundle.extractfile(member) as stream:
                raw = stream.read(item["bytes"] + 1)
            if len(raw) != item["bytes"] or digest(raw) != item["sha256"]:
                raise ValueError("Proof member differs from its closed SHA index")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Proof archive is missing indexed members")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "outputs/natural-v4/research-proof-v1/natural-v4-research-proof-v1.tar.gz",
    )
    parser.add_argument("--receipt-dir", type=Path, default=ROOT / "docs/natural-assistant/v4/research-proof-v1")
    args = parser.parse_args()
    if args.receipt_dir.exists():
        raise FileExistsError("Proof receipts are immutable; select a new receipt directory")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    credits = attribution(ROOT)
    notice = (
        b"Optional v4 research-proof backup; original source and historical review bytes.\n"
        b"Raw files require extracting the external archive and are not present in an ordinary Git checkout.\n"
        b"DOCCI: Jason Baldridge and family / Google LLC; NVIDIA OCR: NVIDIA Corporation; CC BY 4.0.\n"
        b"Commons: individual artists/credits/source pages and CC BY 3.0/4.0 or CC0 licenses in ATTRIBUTION.json.\n"
        b"FLEURS: Alexis Conneau et al.; CC BY 4.0. OpenAssistant: contributors / LAION; Apache 2.0.\n"
        b"AISHELL: Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng / Beijing Shell Shell Technology; Apache 2.0.\n"
        b"https://creativecommons.org/licenses/by/4.0/\nhttps://creativecommons.org/licenses/by/3.0/\n"
        b"https://creativecommons.org/publicdomain/zero/1.0/\nhttps://www.openslr.org/33/\n"
        b"Changes: source subsets, review montages, enlargements/crops and AI-authored annotations.\n"
        b"Course-authored proof code/reports: MIT. Individual underlying source rights remain as attributed.\n"
        b"Transformers 4.57.6 software source: Apache 2.0; original copyright/license headers retained.\n"
        b"No draft release or external upload is performed by this builder.\n"
    )
    apache = ROOT / "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE"
    generated = {
        "PROOF/NOTICE.txt": notice,
        "PROOF/ATTRIBUTION.json": canonical(credits),
        "PROOF/LICENSE-MIT.txt": (ROOT / "LICENSE").read_bytes(),
        "PROOF/LICENSE-APACHE-2.0.txt": apache.read_bytes(),
    }
    index = package(ROOT, selected_files(ROOT), args.archive, generated)
    verify_archive(args.archive, index, generated)
    args.receipt_dir.mkdir(parents=True, exist_ok=False)
    (args.receipt_dir / "proof-index.json").write_bytes(canonical(index))
    (args.receipt_dir / "NOTICE.txt").write_bytes(notice)
    (args.receipt_dir / "ATTRIBUTION.json").write_bytes(canonical(credits))
    receipt = {
        "schema_version": 1,
        "snapshot": "natural-v4-research-proof-v1",
        "code_revision_observed": revision,
        "archive_name": args.archive.name,
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": file_digest(args.archive),
        "index_sha256": digest(canonical(index)),
        "raw_files": len(index["files"]),
        "raw_bytes": sum(item["bytes"] for item in index["files"]),
        "all_raw_member_bytes_sha256_verified_locally": True,
        "ordinary_checkout_has_raw_evidence": False,
        "draft_release_created": False,
        "external_archive_uploaded": False,
        "external_archive_url": None,
        "ci_readable_artifacts": [
            "receipt.json",
            "proof-index.json",
            "NOTICE.txt",
            "ATTRIBUTION.json",
            "SHA256SUMS",
            "release-notes.md",
        ],
    }
    (args.receipt_dir / "receipt.json").write_bytes(canonical(receipt))
    (args.receipt_dir / "SHA256SUMS").write_text(
        f"{receipt['archive_sha256']}  {args.archive.name}\n{receipt['index_sha256']}  proof-index.json\n"
    )
    (args.receipt_dir / "release-notes.md").write_text(
        f"Optional immutable v1 legal research-proof snapshot ({len(index['files'])} raw files).\n\n"
        "This release has not been created or uploaded by the builder. The coordinator can attach the archive and the small receipt/index/notices to a GitHub draft release targeting an existing reviewed commit.\n\n"
        "Raw evidence requires archive extraction after SHA verification. Ordinary checkout and CI consume the receipt/index; reader or technical-review registry records must link these small artifacts or an eventual published archive URL, rather than assume raw evidence paths exist in checkout.\n\n"
        f"Archive SHA-256: `{receipt['archive_sha256']}`\n\n"
        "Excluded: original historical OCR research/samples, unselected SA/NC candidate previews/cards, copyrighted full texts without confirmed redistribution, model weights. Training/scoring/future reader reviews require later separate immutable snapshots.\n"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
