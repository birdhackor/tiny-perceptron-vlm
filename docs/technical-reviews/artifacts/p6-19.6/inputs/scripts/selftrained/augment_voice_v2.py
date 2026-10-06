#!/usr/bin/env python3
"""Create deterministic train-only transforms of the original human voice data.

Original train bytes remain an exact prefix. Heldout data/assets are preserved.
The transforms are derived examples of the same 62 recordings, not new humans,
speakers, independent utterances, or manually verified transcriptions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import shutil
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261006
GENERATOR = "voice-train-augmentation-v2-fixedpoint-1"
REVISION = "40ce77cb32a384e4d50a568e1ec39ac804019d33"
ALLOWED_OTHER_APPEND_FILES = {"text-tools-train.jsonl", "ocr-train.jsonl"}
VARIANTS = (
    ("speed090", 9, 10, None, 0, 0),
    ("speed110", 11, 10, None, 0, 0),
    ("speed095", 19, 20, None, 0, 0),
    ("speed105", 21, 20, None, 0, 0),
    ("noise30_shift", 1, 1, 30, 100, 200),
    ("noise25_shift", 1, 1, 25, 250, 50),
    ("speed090_noise28_shift", 9, 10, 28, 150, 100),
    ("speed110_noise28_shift", 11, 10, 28, 80, 180),
)
# Fixed, prequantized Hann-windowed sinc FIR kernels (sum 65536). Speed-up
# filtering reduces aliasing. Integer kernels avoid platform-specific DSP bytes.
FIR = {
    (21, 20): (
        0,
        0,
        -30,
        130,
        -306,
        508,
        -626,
        507,
        0,
        -987,
        2446,
        -4243,
        6132,
        -7802,
        8952,
        56174,
        8952,
        -7802,
        6132,
        -4243,
        2446,
        -987,
        0,
        507,
        -626,
        508,
        -306,
        130,
        -30,
        0,
        0,
    ),
    (11, 10): (
        0,
        -16,
        63,
        -90,
        0,
        282,
        -728,
        1156,
        -1244,
        641,
        882,
        -3289,
        6225,
        -9077,
        11155,
        53616,
        11155,
        -9077,
        6225,
        -3289,
        882,
        641,
        -1244,
        1156,
        -728,
        282,
        0,
        -90,
        63,
        -16,
        0,
    ),
}
# Approximate 10**(-snr_db/20), quantized to Q20 once, not computed at runtime.
NOISE_GAIN_Q20 = {25: 58966, 28: 41746, 30: 33159}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def rounded_divide(values, divisor):
    """Integer round-to-nearest, ties towards positive infinity."""
    return (values + divisor // 2) // divisor


def resample_speed(samples: np.ndarray, numerator: int, denominator: int) -> np.ndarray:
    samples = samples.astype(np.int64)
    if numerator == denominator:
        return samples.copy()
    if numerator > denominator:
        kernel = np.asarray(FIR[(numerator, denominator)], dtype=np.int64)
        padded = np.pad(samples, (len(kernel) // 2, len(kernel) // 2), mode="edge")
        samples = rounded_divide(np.convolve(padded, kernel, mode="valid"), 65536)
    # Endpoint-inclusive interpolation spans the whole recording, with no crop.
    count = ((len(samples) - 1) * denominator + numerator - 1) // numerator + 1
    positions = np.arange(count, dtype=np.int64) * numerator
    indexes = np.minimum(positions // denominator, len(samples) - 1)
    fractions = positions % denominator
    right = np.minimum(indexes + 1, len(samples) - 1)
    return rounded_divide(samples[indexes] * (denominator - fractions) + samples[right] * fractions, denominator)


def pcm_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    memory = io.BytesIO()
    with wave.open(memory, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.astype("<i2").tobytes())
    return memory.getvalue()


def transform(samples, sample_rate, parent_sha, variant, seed):
    name, numerator, denominator, snr, leading_ms, trailing_ms = variant
    signal = resample_speed(samples, numerator, denominator)
    leading = sample_rate * leading_ms // 1000
    trailing = sample_rate * trailing_ms // 1000
    signal = np.pad(signal, (leading, trailing))
    transformed = signal.copy()
    noise_seed = int(digest(f"{GENERATOR}|{seed}|{parent_sha}|{name}".encode())[:8], 16)
    signal_energy = int(np.dot(signal, signal))
    noise_energy = 0
    if snr is not None:
        # Stationary synthetic background noise only, not generated human speech.
        rng = np.random.Generator(np.random.PCG64(noise_seed))
        noise = rng.integers(-32768, 32768, len(signal), dtype=np.int64)
        noise -= rounded_divide(int(noise.sum()), len(noise))
        noise = rounded_divide(np.convolve(np.pad(noise, (1, 1), mode="edge"), [1, 2, 1], mode="valid"), 4)
        raw_energy = int(np.dot(noise, noise))
        if not signal_energy or not raw_energy:
            raise ValueError("Cannot augment zero-energy waveform")
        ratio_q20 = math.isqrt((signal_energy << 40) // raw_energy)
        gain_q20 = rounded_divide(ratio_q20 * NOISE_GAIN_Q20[snr], 1 << 20)
        noise = rounded_divide(noise * gain_q20, 1 << 20)
        noise_energy = int(np.dot(noise, noise))
        transformed += noise
    peak = int(np.abs(transformed).max())
    attenuation_numerator, attenuation_denominator = (32700, peak) if peak > 32700 else (1, 1)
    if peak > 32700:
        transformed = rounded_divide(transformed * 32700, peak)
    if np.abs(transformed).max() > 32767:
        raise ValueError("Unexpected PCM clipping")
    metadata = {
        "variant": name,
        "speed_numerator": numerator,
        "speed_denominator": denominator,
        "interpolation": "endpoint-inclusive fixed-point linear; fixed 31-tap anti-alias FIR for speed-up",
        "leading_padding_samples": leading,
        "trailing_padding_samples": trailing,
        "noise": "deterministic PCG64 discrete-uniform noise, fixed [1,2,1]/4 low-pass" if snr is not None else None,
        "noise_seed": noise_seed if snr is not None else None,
        "target_noise_snr_db": snr,
        "pre_attenuation_signal_energy": signal_energy,
        "pre_attenuation_noise_energy": noise_energy,
        "peak_attenuation_numerator": attenuation_numerator,
        "peak_attenuation_denominator": attenuation_denominator,
        "sample_rate": sample_rate,
        "channels": 1,
        "encoding": "PCM_16 WAV",
        "samples": len(transformed),
        "speech_cropped": False,
    }
    return transformed.astype(np.int16), metadata


def preserve_original_files(source: Path, destination: Path):
    """Copy only missing files; preserve precisely permitted reviewed appenders."""
    entries = []
    for original in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = original.relative_to(source)
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        if relative.as_posix() == "voice-train.jsonl":
            if not target.read_bytes().startswith(original.read_bytes()):
                raise ValueError("Existing voice train does not preserve original bytes as prefix")
        elif relative.as_posix() in ALLOWED_OTHER_APPEND_FILES:
            if not target.read_bytes().startswith(original.read_bytes()):
                raise ValueError(f"Reviewed train append must preserve original prefix: {relative}")
        elif file_digest(original) != file_digest(target):
            raise ValueError(f"Existing original/heldout file changed: {relative}")
        entries.append({"file": relative.as_posix(), "sha256": file_digest(original), "bytes": original.stat().st_size})
    return entries


def frozen_parent_files(source: Path, destination: Path) -> dict[str, str]:
    """Bind distributed provenance to the frozen package, never local caches."""
    frozen = json.loads((ROOT / "docs/selftrained/manifest.json").read_text(encoding="utf-8"))
    recipe = json.loads((ROOT / "docs/selftrained/package-recipe.json").read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in frozen["records"] + frozen["assets"]}
    packaged = {entry["path"]: entry for entry in recipe["source_files"]}
    if set(expected) != set(packaged):
        raise ValueError("Frozen parent manifest and package recipe list different files")
    identities = {}
    for relative, entry in sorted(expected.items()):
        path = source / relative
        # The four frozen package notices live outside producers' data roots.
        # Resolve only those missing package files via the frozen V1 recipe.
        if not path.exists():
            path = ROOT / packaged[relative]["source"]
        if (
            entry["sha256"] != packaged[relative]["sha256"]
            or entry["bytes"] != packaged[relative]["bytes"]
            or file_digest(path) != entry["sha256"]
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError(f"Source file differs from frozen parent package: {relative}")
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if relative in ALLOWED_OTHER_APPEND_FILES | {"voice-train.jsonl"}:
            if not target.read_bytes().startswith(path.read_bytes()):
                raise ValueError(f"Existing train does not preserve frozen parent prefix: {relative}")
        elif file_digest(target) != entry["sha256"]:
            raise ValueError(f"Existing immutable packaged file changed: {relative}")
        identities[relative] = entry["sha256"]
    return identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=ROOT / "outputs/selftrained/data")
    parser.add_argument("--output-data", type=Path, default=ROOT / "outputs/selftrained-v2/data")
    parser.add_argument("--variants-per-wave", type=int, choices=(4, 6, 8), default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    source, destination = args.source_data.resolve(), args.output_data.resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError("Output must be separate from immutable source data")
    original_files = preserve_original_files(source, destination)
    stable_parent_sha = frozen_parent_files(source, destination)
    original_train = (source / "voice-train.jsonl").read_bytes()
    if not original_train.endswith(b"\n"):
        raise ValueError("Source JSONL must end in newline to preserve its exact bytes as prefix")
    records = [json.loads(line) for line in original_train.decode("utf-8").splitlines() if line.strip()]
    by_audio = defaultdict(list)
    for record in records:
        if record["split"] != "train" or not record["task"].startswith("voice"):
            raise ValueError("Only original training voice rows may be augmented")
        by_audio[record["audio"]].append(record)
    variants = VARIANTS[: args.variants_per_wave]
    added, assets = [], []
    for relative, parents in sorted(by_audio.items()):
        parent_path = (source / relative).resolve()
        if not parent_path.is_relative_to(source):
            raise ValueError("Audio must be inside source asset root")
        if len({r["group_id"] for r in parents}) != 1 or len({r["supervision"]["intent_id"] for r in parents}) != 1:
            raise ValueError("Inconsistent source group or source intent label")
        samples, sample_rate = sf.read(parent_path, dtype="int16")
        if samples.ndim != 1 or sample_rate != 8000:
            raise ValueError("Expected native mono 8kHz source waveform")
        parent_sha = file_digest(parent_path)
        for variant in variants:
            transformed, metadata = transform(samples, sample_rate, parent_sha, variant, args.seed)
            data = pcm_wav(transformed, sample_rate)
            checksum = digest(data)
            asset = f"audio/voice-augmentation-v2/{checksum}.wav"
            target = destination / asset
            if target.exists() and file_digest(target) != checksum:
                raise ValueError("Existing derived asset checksum conflict")
            atomic_write(target, data)
            asset_metadata = {
                "audio": asset,
                "sha256": checksum,
                "bytes": len(data),
                "parent_audio": relative,
                "parent_sha256": parent_sha,
                "parent_pcm_int16_sha256": digest(samples.astype("<i2").tobytes()),
                "group_id": parents[0]["group_id"],
                "original_human_recordings": 1,
                "new_independent_human_recordings": 0,
                "transformation": metadata,
            }
            assets.append(asset_metadata)
            for parent in parents:
                record = copy.deepcopy(parent)
                record["id"] = parent["id"] + "-aug-v2-" + variant[0]
                record["audio"] = asset
                record["augmentation"] = {
                    "generator": GENERATOR,
                    "variant": variant[0],
                    "parent_id": parent["id"],
                    "parent_audio": relative,
                    "parent_audio_sha256": parent_sha,
                    "new_independent_human_recording": False,
                }
                added.append(record)
    suffix = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in added).encode("utf-8")
    output_train = original_train + suffix
    output_path = destination / "voice-train.jsonl"
    existing = output_path.read_bytes()
    manifest_dir = destination.parent / "manifests"
    manifest_path = manifest_dir / "voice-augmentation-v2.json"
    if existing not in (original_train, output_train):
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if old.get("generator") != GENERATOR or old.get("output_voice_train_sha256") != digest(existing):
            raise ValueError("Refusing to overwrite unknown appended voice training rows")
    atomic_write(output_path, output_train)
    # Recheck source tree and all immutable assets/heldout bytes after writing.
    if original_files != preserve_original_files(source, destination):
        raise ValueError("Source files changed while generating augmentation")
    if not output_path.read_bytes().startswith(original_train):
        raise ValueError("Original voice training bytes are not an exact prefix")
    all_records = records + added
    if len({r["id"] for r in all_records}) != len(all_records):
        raise ValueError("Duplicate augmented row IDs")
    for record in added:
        parent = next(
            r
            for r in by_audio[record["augmentation"]["parent_audio"]]
            if r["id"] == record["augmentation"]["parent_id"]
        )
        if any(record[key] != parent[key] for key in ("group_id", "split", "task", "messages", "supervision")):
            raise ValueError("Augmentation altered history, label, task or split")
        if record.get("modality_message_index") != parent.get("modality_message_index") or record.get(
            "evaluation"
        ) != parent.get("evaluation"):
            raise ValueError("Augmentation altered public modality/evaluation structure")
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR,
        "seed": args.seed,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "source": {
            "dataset": "PolyAI/minds14",
            "config": "zh-CN",
            "revision": REVISION,
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "author": "PolyAI; Gerz et al., Multilingual and Cross-Lingual Intent Detection from Spoken Data (2021)",
            "official_card": f"https://huggingface.co/datasets/PolyAI/minds14/blob/{REVISION}/README.md",
        },
        "counts": {
            "original_training_rows": len(records),
            "original_human_training_recordings": len(by_audio),
            "original_training_source_groups": len({r["group_id"] for r in records}),
            "transforms_per_parent": len(variants),
            "derived_audio_files": len(assets),
            "added_training_rows": len(added),
            "total_training_rows": len(all_records),
            "new_independent_human_recordings": 0,
            "new_speakers_claimed": 0,
            "task_rows": dict(Counter(r["task"] for r in all_records)),
            "intent_rows": dict(Counter(r["supervision"]["intent"] for r in all_records)),
        },
        "source_voice_train_sha256": digest(original_train),
        "output_voice_train_sha256": digest(output_train),
        "parent_manifest": {
            "path": "docs/selftrained/manifest.json",
            "sha256": file_digest(ROOT / "docs/selftrained/manifest.json"),
        },
        "parent_package_recipe": {
            "path": "docs/selftrained/package-recipe.json",
            "sha256": file_digest(ROOT / "docs/selftrained/package-recipe.json"),
        },
        "parent_packaged_files_count": len(stable_parent_sha),
        "parent_file_sha256": stable_parent_sha,
        "parent_packaged_files_sha256": digest(
            json.dumps(stable_parent_sha, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ),
        "preservation": {
            "voice_train_original_bytes_exact_prefix": True,
            "original_files_and_heldout_media_verified": True,
            "entire_source_tree_checked_at_runtime": True,
            "distributed_parent_identity": "frozen V1 packaged files; producer caches/raw/API metadata excluded",
            "allowed_other_reviewed_prefix_append_files": sorted(ALLOWED_OTHER_APPEND_FILES),
            "heldout_jsonl": {
                p.name: file_digest(destination / p.name)
                for p in sorted(source.glob("voice-*.jsonl"))
                if p.name in {"voice-validation.jsonl", "voice-test.jsonl"}
            },
        },
        "transformations": [
            dict(
                zip(
                    (
                        "variant",
                        "speed_numerator",
                        "speed_denominator",
                        "target_noise_snr_db",
                        "leading_ms",
                        "trailing_ms",
                    ),
                    v,
                    strict=True,
                )
            )
            for v in variants
        ],
        "fir_coefficients_q16": {f"{n}/{d}": list(k) for (n, d), k in FIR.items()},
        "assets": assets,
        "limits": [
            "Same original utterance, source group and intent for every derived example",
            "No new humans/speakers or speaker-disjoint claim",
            "No crop, ASR, transcript prompt, relabel, pretrained weights or test generation",
            "Original U-Law bytes retained; derivatives decoded, transformed and encoded as deterministic PCM16",
            "Stationary noise is synthetic background augmentation, not synthetic human speech",
            "Pure gain augmentation omitted because per-record log-mel standardization largely cancels it",
            "No claim of increased independent source counts or measured validation improvement",
        ],
    }
    atomic_write(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    notice = (
        "# V2 training-only MInDS-14 audio modifications\n\n"
        "Source attribution: PolyAI; Gerz et al., Multilingual and Cross-Lingual Intent Detection from Spoken Data (2021). "
        f"https://huggingface.co/datasets/PolyAI/minds14 revision `{REVISION}`. "
        "CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . Original card and notices remain in the original data/manifests.\n\n"
        "Changes: training audio only, fixed-point speed resampling, gentle deterministic synthetic background noise, "
        "leading/trailing padding without cutting speech, optional peak attenuation, PCM16 encoding and appended training rows. "
        "Original human recordings, source labels, group IDs, response/history templates and evaluation structure remain the same. "
        "No new independent utterances or speakers are claimed; no endorsement by the source authors is implied. "
        "The project-authored teaching replies retain the project's MIT license.\n\n"
        "Validation/test JSONL and original media bytes remain unchanged. Other reviewed text/OCR training appenders "
        "may add rows while retaining their original exact byte prefix. No human listening was performed.\n"
    )
    atomic_write(manifest_dir / "voice-augmentation-v2-NOTICE.md", notice.encode())
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "voice_train_sha256": digest(output_train),
                "manifest_sha256": file_digest(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
