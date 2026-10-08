"""Fetch a small, reproducible set of human Mandarin recordings from FLEURS.

The original, pinned Hugging Face revision retains streaming tar archives.  The
current Parquet conversion stores an entire split's audio in one row group,
which is unnecessarily expensive for this teaching sample.  We keep the first
distinct sentence IDs encountered in each original archive, with no selection
based on an ASR model's outputs.  No text is generated from the audio here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
REVISION = "d7c758a6dceecd54a98cac43404d3d576e721f07"
DATASET = "google/fleurs"
CONFIG = "cmn_hans_cn"
VERSION = "fleurs-mandarin-human-small-v1"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
CARD_SHA256 = "878990a8b3832046622035b1b2e4f1effa138176876372fd5a2365d0a469f14c"
TSV_SHA256 = {
    "train": "89c4a48ffaf2811bf64a4c38a65ccf436e4e46feaf1ba762ba137fc324532200",
    "dev": "6b4efd804b543048feb278db06f3b58b5ea171cdd4ba072e328ad630ca25384b",
    "test": "5734461648f816181d7dab5fc79204b18c4b9bc2cd5138225b25c72d18385d21",
}
ARCHIVE_SHA256 = {
    "train": "812346f79314a46347e1ed481c4948626fa63a6220e511907d1ec5860855df68",
    "dev": "3bc33212d5974eef7feb04bc4792458d6cd7e14ff10a1a24772f3c45ea87a822",
    "test": "09d19ad18f5d7e91076880807e866cd16abd924c7052b55f71cdae91714fc166",
}


class CountedReader:
    """Count the bytes actually requested from a partial archive stream."""

    def __init__(self, response):
        self.response = response
        self.bytes_read = 0

    def read(self, size=-1):
        value = self.response.read(size)
        self.bytes_read += len(value)
        return value


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-perceptron-vlm-public-data/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def metadata(split: str, directory: Path) -> tuple[dict, dict]:
    url = f"{BASE}/data/{CONFIG}/{split}.tsv"
    raw = download(url)
    actual = sha256(raw)
    if actual != TSV_SHA256[split]:
        raise ValueError(f"Pinned {split} transcription checksum mismatch: {actual}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{split}.tsv").write_bytes(raw)
    rows = {}
    for line in raw.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"Unexpected transcription row with {len(fields)} fields")
        sentence, filename, original, normalized, _, frames, gender = fields
        if filename in rows:
            raise ValueError(f"Repeated filename: {filename}")
        rows[filename] = {
            "sentence_id": int(sentence),
            "filename": filename,
            "raw_transcription": original,
            "normalized_transcription": normalized,
            "expected_num_samples": int(frames),
            "gender": gender.lower(),
        }
    return rows, {"url": url, "sha256": actual, "bytes": len(raw)}


def recordings(
    source_split: str,
    split: str,
    count: int,
    annotations: dict,
    output: Path,
) -> tuple[list[dict], dict]:
    url = f"{BASE}/data/{CONFIG}/audio/{source_split}.tar.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-perceptron-vlm-public-data/1"})
    selected = []
    sentence_ids = set()
    scanned_files = 0
    directory = output / "audio" / split
    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=90) as response:
        stream = CountedReader(response)
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                filename = Path(member.name).name
                if filename not in annotations:
                    continue
                scanned_files += 1
                note = annotations[filename]
                sentence_id = note["sentence_id"]
                if sentence_id in sentence_ids:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Unable to read {member.name}")
                raw = extracted.read()
                sound = sf.info(io.BytesIO(raw))
                if sound.samplerate != 16000 or sound.channels != 1:
                    raise ValueError(f"Unexpected audio format in {filename}: {sound}")
                if sound.frames != note["expected_num_samples"]:
                    raise ValueError(f"Transcript sample count does not match {filename}")
                relative = Path("audio") / split / filename
                (output / relative).write_bytes(raw)
                row_id = f"fleurs-cmn-{source_split}-{Path(filename).stem}"
                selected.append(
                    {
                        "id": row_id,
                        "split": split,
                        "task": "speech_chat",
                        "audio": relative.as_posix(),
                        "user": note["raw_transcription"],
                        "answer": None,
                        "raw_transcription": note["raw_transcription"],
                        "normalized_transcription": note["normalized_transcription"],
                        "sample_rate": sound.samplerate,
                        "num_samples": sound.frames,
                        "duration_seconds": sound.duration,
                        "speaker": None,
                        "family": f"fleurs-cmn-sentence-{sentence_id}",
                        "gender": note["gender"],
                        "sha256": sha256(raw),
                        "bytes": len(raw),
                        "source": "fleurs-mandarin",
                        "source_split": source_split,
                        "source_member": member.name,
                        "synthetic": False,
                    }
                )
                sentence_ids.add(sentence_id)
                if len(selected) == count:
                    break
        compressed_bytes_read = stream.bytes_read
    if len(selected) != count:
        raise ValueError(f"Requested {count} {split} clips, found only {len(selected)}")
    return selected, {
        "url": url,
        "sha256_from_huggingface_lfs_metadata": ARCHIVE_SHA256[source_split],
        "full_archive_downloaded_and_verified": False,
        "compressed_bytes_read": compressed_bytes_read,
        "metadata_matched_files_scanned": scanned_files,
    }


def prepare(output: Path, counts: dict[str, int]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    official = output / "official-sources"
    official.mkdir(parents=True, exist_ok=True)
    card_url = f"{BASE}/README.md"
    card = download(card_url)
    if sha256(card) != CARD_SHA256:
        raise ValueError("Pinned FLEURS dataset-card checksum mismatch")
    (official / "FLEURS_DATASET_CARD.md").write_bytes(card)
    (output / "LICENSE.txt").write_text(
        "FLEURS Mandarin human-recording subset\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
        "Official license URL: https://creativecommons.org/licenses/by/4.0/\n"
        "Attribution: Alexis Conneau et al., FLEURS: Few-shot Learning Evaluation "
        "of Universal Representations of Speech (2022).\n"
        "Paper: https://arxiv.org/abs/2205.12446\n"
        f"Dataset: https://huggingface.co/datasets/{DATASET}\n"
        f"Source revision: {REVISION}\n"
        "Modification notice: selection of a small subset; original WAV bytes "
        "and original/normalized reference transcriptions are retained.\n"
        "This notice refers to the official license; it is not a replacement "
        "for its legal terms. The official dataset card is included under "
        "official-sources/FLEURS_DATASET_CARD.md.\n",
        encoding="utf-8",
    )
    all_metadata = {}
    source_files = {}
    for split in counts:
        rows, source = metadata(split, official)
        all_metadata[split] = rows
        source_files[split] = {"transcripts": source}
    sentence_sets = {split: {row["sentence_id"] for row in rows.values()} for split, rows in all_metadata.items()}
    pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for left, right in pairs:
        overlap = sentence_sets[left] & sentence_sets[right]
        if overlap:
            raise ValueError(f"Source sentence IDs overlap in {left}/{right}: {overlap}")
    audio_rows = []
    for source_split, count in counts.items():
        split = "validation" if source_split == "dev" else source_split
        rows, source = recordings(source_split, split, count, all_metadata[source_split], output)
        audio_rows.extend(rows)
        source_files[source_split]["audio_archive"] = source
    manifest = {
        "schema_version": 1,
        "dataset_version": VERSION,
        "sources": [
            {
                "id": "fleurs-mandarin",
                "dataset": DATASET,
                "config": CONFIG,
                "revision": REVISION,
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "card_url": f"https://huggingface.co/datasets/{DATASET}/blob/{REVISION}/README.md",
                "paper_url": "https://arxiv.org/abs/2205.12446",
                "attribution": "Alexis Conneau et al., FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech (2022).",
                "language": "Mandarin Chinese",
                "script": "simplified Chinese; original transcript preserved",
                "recording_type": "human read speech, not a conversational dialogue corpus",
                "speaker_ids_available": False,
                "speaker_split_claim": "The official card states that training speakers differ from dev/test speakers. It does not establish dev/test speaker disjointness.",
                "card_file": "official-sources/FLEURS_DATASET_CARD.md",
                "card_sha256": CARD_SHA256,
                "license_file": "LICENSE.txt",
                "files": source_files,
            }
        ],
        "selection": {
            "policy": "first distinct sentence IDs encountered in each pinned source archive",
            "counts": counts,
            "based_on_model_outputs": False,
            "source_sentence_ids_disjoint": True,
            "sentence_id_counts_in_source": {split: len(values) for split, values in sentence_sets.items()},
        },
        "rows": [],
        "audio_rows": audio_rows,
        "notes": [
            "The user field is the reference transcript for ASR evaluation, and the matching typed-input baseline. Never give it to the model on the audio-input inference route.",
            "The raw orthography and the source-normalized transcription are both retained. Define the evaluation normalization before comparing ASR outputs.",
            "FLEURS has no assistant response labels. answer=null is deliberate: this preparation does not invent a correct chat reply or label ASR transcripts as assistant responses.",
            "Partial tar streaming does not verify the full archive checksum. Each downloaded WAV is hashed, and metadata is checked against a pinned SHA-256.",
            "The small subset is a teaching smoke test. It cannot establish robustness to all Mandarin speakers, Taiwanese accents, spontaneous speech, noise, or multi-speaker overlap.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/natural/speech")
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--validation", type=int, default=6)
    parser.add_argument("--test", type=int, default=12)
    args = parser.parse_args()
    counts = {"train": args.train, "dev": args.validation, "test": args.test}
    if any(value <= 0 for value in counts.values()):
        parser.error("Each split must request at least one recording")
    manifest = prepare(args.output.resolve(), counts)
    summary = {
        "manifest": str(args.output.resolve() / "manifest.json"),
        "recordings": len(manifest["audio_rows"]),
        "duration_seconds": round(sum(row["duration_seconds"] for row in manifest["audio_rows"]), 3),
        "audio_bytes": sum(row["bytes"] for row in manifest["audio_rows"]),
        "compressed_bytes_read": sum(
            entry["audio_archive"]["compressed_bytes_read"] for entry in manifest["sources"][0]["files"].values()
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
