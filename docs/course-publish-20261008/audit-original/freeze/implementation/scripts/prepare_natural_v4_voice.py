"""Select fresh, source-disjoint FLEURS recordings before any ASR inference."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import prepare_natural_speech as previous  # noqa: E402


def main():
    output = ROOT / "outputs/natural-v4/data/voice"
    metadata_root = ROOT / "docs/natural-assistant/evidence/v4-research/chat-speech/fleurs-metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    old_bundle = ROOT / "assets/training/natural-speech-v3.tar.gz"
    with tarfile.open(old_bundle, "r:gz") as archive:
        old = json.load(archive.extractfile("speech/manifest.json"))
    excluded = {row["family"] for row in old["audio_rows"]}
    excluded_recordings = {row["id"] for row in old["audio_rows"]}
    rows, sources = [], {}
    counts = {"train": 36, "dev": 12, "test": 18}
    used_families = set()
    for source_split, count in counts.items():
        annotation, receipt = previous.metadata(source_split, metadata_root)
        split = "validation" if source_split == "dev" else source_split
        url = f"{previous.BASE}/data/{previous.CONFIG}/audio/{source_split}.tar.gz"
        chosen = []
        request = urllib.request.Request(url, headers={"User-Agent": "tiny-perceptron-vlm-fresh-voice/1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            stream = previous.CountedReader(response)
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    filename = Path(member.name).name
                    if not member.isfile() or filename not in annotation:
                        continue
                    note = annotation[filename]
                    family = f"fleurs-cmn-sentence-{note['sentence_id']}"
                    if family in excluded or family in used_families:
                        continue
                    duration = note["expected_num_samples"] / 16000
                    if not 2 <= duration <= 25:
                        continue
                    raw = archive.extractfile(member).read()
                    info = sf.info(io.BytesIO(raw))
                    if (info.samplerate, info.channels, info.frames) != (16000, 1, note["expected_num_samples"]):
                        raise ValueError(f"Source sound metadata mismatch: {member.name}")
                    relative = Path("audio") / split / filename
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    row_id = f"fleurs-cmn-v4-{source_split}-{Path(filename).stem}"
                    chosen.append(
                        {
                            "id": row_id,
                            "split": split,
                            "task": "speech_chat",
                            "family": family,
                            "audio": str(relative),
                            "user": note["raw_transcription"],
                            "answer": None,
                            "raw_transcription": note["raw_transcription"],
                            "normalized_transcription_from_source": note["normalized_transcription"],
                            "source": "fleurs-mandarin-v4",
                            "source_split": source_split,
                            "source_member": member.name,
                            "source_sentence_id": note["sentence_id"],
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "bytes": len(raw),
                            "sample_rate": info.samplerate,
                            "channels": info.channels,
                            "num_samples": info.frames,
                            "duration_seconds": info.duration,
                            "speaker": None,
                            "gender": note["gender"],
                            "synthetic": False,
                            "recording_domain": "human read speech; not a human question to an assistant",
                            "inspection": "SoundFile metadata and official transcript, not listening or ASR inference",
                        }
                    )
                    used_families.add(family)
                    if len(chosen) == count:
                        break
            compressed_bytes_read = stream.bytes_read
        if len(chosen) != count:
            raise ValueError(f"Not enough fresh {source_split} recordings")
        rows.extend(chosen)
        sources[source_split] = {
            "transcripts": receipt,
            "audio_archive": {
                "url": url,
                "sha256_from_huggingface_lfs_metadata": previous.ARCHIVE_SHA256[source_split],
                "full_archive_downloaded_and_verified": False,
                "compressed_bytes_read": compressed_bytes_read,
                "selection": "First eligible distinct sentence in original tar order; no model output used",
            },
        }
    if excluded & used_families:
        raise ValueError("New source sentences overlap the old v3 subset")
    result = {
        "schema_version": 1,
        "dataset_version": "fleurs-mandarin-fresh-v4",
        "dataset": previous.DATASET,
        "config": previous.CONFIG,
        "revision": previous.REVISION,
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Alexis Conneau et al., FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech (2022)",
        "source_files": sources,
        "modification_notice": "Selected fresh original WAV bytes and original reference transcripts; no generated response label or transcript correction",
        "audio_root": str(output.relative_to(ROOT)),
        "audio_rows": rows,
        "counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
        "total_seconds": sum(row["duration_seconds"] for row in rows),
        "audio_bytes": sum(row["bytes"] for row in rows),
        "excluded_v3_sentence_families": sorted(excluded),
        "excluded_v3_recordings": sorted(excluded_recordings),
        "sentence_family_disjointness_checked": True,
        "speaker_disjointness": "Official card states train speakers differ from dev/test; individual IDs unavailable in this selected source format",
        "metrics_policy_frozen_before_asr": {
            "raw_CER": "Character Levenshtein distance on verbatim official raw transcript; punctuation retained",
            "normalized_CER": "NFKC then remove all Unicode whitespace; retain punctuation and original Simplified/Traditional characters",
            "optional_script_equivalence": "Separate named diagnostic only; do not replace the raw prediction or ground truth",
            "LM_response_quality": "Separate from CER. Any authored restatement instruction is text/system context, not something the human said in the WAV",
            "split_usage": "train audio/transcripts for training-only preparation; validation for choosing small vs turbo; test opened for final inference after ASR/model selection freeze",
        },
        "notes": [
            "These source utterances are read statements, not unrestricted voice-chat questions.",
            "Answer remains null: a transcript is not a reference assistant answer.",
            "No source audio was listened to and no ASR model was run during this preparation.",
            "2..25 second duration filter is predeclared for the existing <=30 second input route; no quality selection using model outputs.",
        ],
    }
    destination = ROOT / "docs/natural-assistant/v4/data/voice-sources.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "LICENSE.txt").write_text(
        f"FLEURS human Mandarin fresh v4 subset\nLicense: Creative Commons Attribution 4.0 International.\n{result['license_url']}\n"
        f"Attribution: {result['attribution']}\nDataset: https://huggingface.co/datasets/google/fleurs\nRevision: {previous.REVISION}\n"
        f"Modification: {result['modification_notice']}\n"
    )
    print(json.dumps({key: result[key] for key in ("counts", "total_seconds", "audio_bytes")}, indent=2))


if __name__ == "__main__":
    main()
