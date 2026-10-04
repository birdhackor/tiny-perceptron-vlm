"""Fetch exactly the frozen v4 WAVs, checking each file against its declared SHA."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import io
import json
import re
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FLEURS_REVISION = "d7c758a6dceecd54a98cac43404d3d576e721f07"
AISHELL_REVISION = "bbe295d530192a4cd41644b711c9aecd087df653"
FLEURS_BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}"
AISHELL_BASE = f"https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/{AISHELL_REVISION}"
SOURCE_LIMITS = {
    **{f"{FLEURS_BASE}/data/cmn_hans_cn/audio/{split}.tar.gz": 32 * 1024 * 1024 for split in ("train", "dev", "test")},
    **{
        f"{AISHELL_BASE}/data_aishell/wav/{speaker}.tar.gz": 48 * 1024 * 1024
        for speaker in ("S0005", "S0007", "S0023", "S0019", "S0031", "S0039", "S0042", "S0032")
    },
}
MAX_WAV_BYTES = 4 * 1024 * 1024
MAX_ATTEMPTS = 3


def validate_source_url(url):
    """Only the exact official corpus archives and revisions used by v4 are allowed."""
    if not isinstance(url, str) or url not in SOURCE_LIMITS:
        raise ValueError("Recording source must be an exact fixed-revision official v4 archive URL")
    return SOURCE_LIMITS[url]


def canonical_relative(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Recording paths must be nonempty canonical relative POSIX paths")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
        raise ValueError("Recording paths must be canonical relative POSIX paths")
    return path


def validate_rows(rows):
    members, destinations, ids = set(), set(), set()
    for row in rows:
        member = str(canonical_relative(row["source_member"]))
        destination = str(canonical_relative(row["audio"]))
        if member in members or destination in destinations or row["id"] in ids:
            raise ValueError("Duplicate frozen archive member, destination path or recording ID")
        members.add(member)
        destinations.add(destination)
        ids.add(row["id"])
        if type(row["bytes"]) is not int or not 0 < row["bytes"] <= MAX_WAV_BYTES:
            raise ValueError("Frozen WAV byte count must be positive and bounded")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Frozen WAV needs a lowercase SHA-256 digest")


def validate_voice_document(name, metadata):
    """Validate a frozen document before it can supply any download URL."""
    configurations = {
        "voice-sources.json": ("google/fleurs", FLEURS_REVISION),
        "voice-question-sources.json": ("AISHELL/AISHELL-1", AISHELL_REVISION),
    }
    if name not in configurations:
        raise ValueError("Unknown v4 voice metadata document")
    dataset, revision = configurations[name]
    if metadata.get("revision") != revision or metadata.get("dataset", dataset) != dataset:
        raise ValueError("Voice metadata differs from the fixed official corpus/revision")
    validate_rows(metadata["audio_rows"])
    for row in metadata["audio_rows"]:
        if name == "voice-sources.json":
            split = row["source_split"]
            url = metadata["source_files"][split]["audio_archive"]["url"]
            if split not in {"train", "dev", "test"} or url != f"{FLEURS_BASE}/data/cmn_hans_cn/audio/{split}.tar.gz":
                raise ValueError("FLEURS source archive does not match its fixed split")
        else:
            url = row["source_archive"]["url"]
            if not url.startswith(AISHELL_BASE + "/"):
                raise ValueError("AISHELL source differs from its fixed corpus")
        validate_source_url(url)


class BoundedReader:
    def __init__(self, response, limit):
        self.response, self.limit, self.bytes_read = response, limit, 0

    def read(self, count=-1):
        if count == 0:
            return b""
        remaining = self.limit - self.bytes_read
        if remaining <= 0:
            raise ValueError("Frozen WAV replay exceeded the bounded compressed read")
        requested = min(count, remaining) if count >= 0 else remaining
        try:
            value = self.response.read(requested)
        except http.client.IncompleteRead as error:
            self.bytes_read += len(error.partial)
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # Unknown partial reads conservatively consume the whole requested allowance.
            self.bytes_read += requested
            raise
        self.bytes_read += len(value)
        if len(value) > requested:
            raise ValueError("Source response exceeded its bounded read request")
        return value


def check_audio(raw, row):
    if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(f"WAV bytes differ from frozen recording {row['id']}")
    info = sf.info(io.BytesIO(raw))
    if (info.samplerate, info.channels, info.frames) != (row["sample_rate"], row["channels"], row["num_samples"]):
        raise ValueError(f"WAV metadata differs from frozen recording {row['id']}")


def target_path(output, row):
    relative = canonical_relative(row["audio"])
    root = Path(output).absolute()
    target = root.joinpath(*relative.parts)
    current = root
    if current.is_symlink():
        raise ValueError("Output root cannot be a symbolic link")
    if current.exists() and not current.is_dir():
        raise ValueError("Output root must be a directory")
    for index, component in enumerate(relative.parts):
        current /= component
        if current.is_symlink():
            raise ValueError("Audio destination contains a symbolic link")
        if index < len(relative.parts) - 1 and current.exists() and not current.is_dir():
            raise ValueError("Audio destination parent must be a directory")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Audio destination resolves outside its output root")
    if target.exists() and not target.is_file():
        raise ValueError("Audio destination must be a regular file")
    return target


def write_payload(output, name, raw):
    """Atomically install bytes after checking the destination and its ancestors."""
    target = target_path(output, {"audio": name})
    target.parent.mkdir(parents=True, exist_ok=True)
    target = target_path(output, {"audio": name})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".voice-replay-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        target_path(output, {"audio": name})
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replay_archive(item, output):
    url, rows = item
    limit = validate_source_url(url)
    validate_rows(rows)
    expected = {row["source_member"]: row for row in rows}
    remaining = {}
    reused = 0
    for member, row in expected.items():
        target = target_path(output, row)
        if target.exists():
            check_audio(target.read_bytes(), row)
            reused += 1
        else:
            remaining[member] = row
    if not remaining:
        return {"url": url, "recordings": len(rows), "reused": reused, "compressed_bytes_read": 0}
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-vlm-frozen-voice-replay/1"})
    downloaded = 0
    for attempt in range(MAX_ATTEMPTS):
        stream = None
        try:
            if downloaded >= limit:
                raise ValueError("Retries exhausted the total bounded compressed download")
            with urllib.request.urlopen(request, timeout=90) as response:
                stream = BoundedReader(response, limit - downloaded)
                with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                    seen = set()
                    for member in archive:
                        name = str(canonical_relative(member.name))
                        if name in seen:
                            raise ValueError("Duplicate tar member encountered in the read archive prefix")
                        seen.add(name)
                        if member.issym() or member.islnk():
                            raise ValueError("Symbolic/hard links are not allowed in the read archive prefix")
                        if name not in remaining:
                            continue
                        row = remaining[name]
                        if not member.isfile() or member.size != row["bytes"]:
                            raise ValueError("Expected frozen WAV must be a regular file of its exact byte size")
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ValueError("Expected frozen WAV was not a regular archive file")
                        raw = extracted.read(row["bytes"] + 1)
                        check_audio(raw, row)
                        write_payload(output, row["audio"], raw)
                        remaining.pop(name)
                        if not remaining:
                            break
                if remaining:
                    raise ValueError(f"Frozen members absent from pinned archive: {sorted(remaining)}")
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead) as error:
            if not remaining:
                break
            if isinstance(error, urllib.error.HTTPError) and not (error.code in {408, 429} or 500 <= error.code < 600):
                raise
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            time.sleep(2**attempt)
        finally:
            if stream is not None:
                downloaded += stream.bytes_read
    return {
        "url": url,
        "recordings": len(rows),
        "reused": reused,
        "compressed_bytes_read": downloaded,
        "attempts": attempt + 1,
        "compressed_bytes_limit_including_retries": limit,
        "archive_integrity_scope": "Unique canonical members in the read prefix only; unread archive tail not inspected",
    }


def write_notices(source_metadata, output):
    """Emit reproducible UTF-8 license/notice bytes inside the voice asset root."""
    fleurs = source_metadata["voice-sources.json"]
    aishell = source_metadata["voice-question-sources.json"]
    for name, metadata in source_metadata.items():
        validate_voice_document(name, metadata)
    texts = {
        "LICENSE.txt": (
            "FLEURS human Mandarin fresh v4 subset\n"
            "License: Creative Commons Attribution 4.0 International.\n"
            f"{fleurs['license_url']}\nAttribution: {fleurs['attribution']}\n"
            "Dataset: https://huggingface.co/datasets/google/fleurs\n"
            f"Revision: {fleurs['revision']}\nModification: {fleurs['modification_notice']}\n"
        ),
        "AISHELL-NOTICE.txt": (
            "AISHELL-1 original human recordings and transcripts\n"
            f"Attribution: {aishell['attribution']}\n"
            "Official dataset license: Apache License 2.0, https://www.openslr.org/33\n"
            "Official HF dataset: https://huggingface.co/datasets/AISHELL/AISHELL-1\n"
            f"Revision: {aishell['revision']}\nModification: {aishell['modification_notice']}\n"
            "These are human read questions, not spontaneous conversations.\n"
        ),
    }
    files = {name: text.encode("utf-8") for name, text in texts.items()}
    apache = ROOT / "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE"
    files["AISHELL-Apache-2.0-LICENSE.txt"] = apache.read_bytes()
    for name, raw in files.items():
        write_payload(output, name, raw)
    return [
        {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in sorted(files.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "docs/natural-assistant/v4/data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/natural-v4/data/voice")
    parser.add_argument(
        "--include-train", action="store_true", help="Optional unused FLEURS train audio; excluded by default"
    )
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    if not 1 <= options.workers <= 8:
        raise ValueError("Voice replay worker count must be between 1 and 8")
    grouped = defaultdict(list)
    all_rows = []
    source_metadata = {}
    for name in ("voice-sources.json", "voice-question-sources.json"):
        metadata = json.loads((options.metadata_dir / name).read_text())
        validate_voice_document(name, metadata)
        source_metadata[name] = metadata
        for row in metadata["audio_rows"]:
            if row["split"] not in {"validation", "test"} and not options.include_train:
                continue
            if name == "voice-sources.json":
                url = metadata["source_files"][row["source_split"]]["audio_archive"]["url"]
            else:
                url = row["source_archive"]["url"]
            validate_source_url(url)
            grouped[url].append(row)
            all_rows.append(row)
    if len({r["id"] for r in all_rows}) != len(all_rows) or len({r["audio"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate frozen recording ID or destination across voice documents")
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as executor:
        results = list(executor.map(lambda item: replay_archive(item, options.output), grouped.items()))
    receipt = {
        "recordings": len(all_rows),
        "bytes": sum(r["bytes"] for r in all_rows),
        "selection": "Exact frozen member IDs only; no new question/transcript selection or model inference",
        "all_recording_sha256_verified": True,
        "original_wav_bytes_preserved": True,
        "full_archive_hash_verified": False,
        "train_audio_included": options.include_train,
        "source_archives": results,
    }
    options.output.mkdir(parents=True, exist_ok=True)
    receipt["license_files"] = write_notices(source_metadata, options.output)
    write_payload(
        options.output,
        "replay-receipt.json",
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "source_archives"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
