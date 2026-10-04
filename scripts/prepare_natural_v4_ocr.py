#!/usr/bin/env python3
"""Rebuild the frozen, reviewed OCR image selection; never select or relabel.

Public HTTP downloads only. No credentials are read. NVIDIA's 10 GB HDF5
shard is accessed using exact, bounded HTTP ranges (128 MiB maximum).
Every original photo/page and generated PNG must match its frozen SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import PIL
from PIL import Image


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(data: bytes, expected_sha: str, expected_bytes: int, what: str) -> None:
    if len(data) != expected_bytes or digest(data) != expected_sha:
        raise ValueError(f"Frozen source changed or corrupt bytes: {what}")


def safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes output directory: {relative}")
    return candidate


def download(url: str, expected_sha: str, expected_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-perceptron-vlm-frozen-ocr-rebuild/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200:
                    raise ValueError(f"Unexpected HTTP {response.status}: {url}")
                data = response.read(expected_bytes + 1)
            check(data, expected_sha, expected_bytes, url)
            return data
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(f"Public image download failed: {url}: HTTP {error.code}") from error
            time.sleep(2**attempt)
        except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError, ConnectionError) as error:
            if attempt == 3:
                raise RuntimeError(f"Public image download failed: {url}: {error}") from error
            time.sleep(2**attempt)
    raise AssertionError("Unreachable")


class RangeFile(io.RawIOBase):
    """Seekable HDF5 reader that refuses a server's full-file response."""

    def __init__(self, url: str, size: int, limit: int):
        self.url, self.size, self.limit = url, size, limit
        self.position = 0
        self.transferred = 0
        self.reserved_transfer = 0
        self.block_size = 256 * 1024
        self.cache: dict[int, bytes] = {}
        self.receipts: list[dict] = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, amount=-1):
        if amount < 0:
            amount = self.size - self.position
        if amount > self.limit:
            raise ValueError("Unbounded HDF5 read refused")
        stop_position = min(self.position + amount, self.size)
        chunks = []
        while self.position < stop_position:
            number = self.position // self.block_size
            start = number * self.block_size
            end = min(start + self.block_size, self.size) - 1
            if number not in self.cache:
                size = end - start + 1
                request = urllib.request.Request(
                    self.url + f"?download=true&range_probe={start}-{end}",
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "tiny-perceptron-vlm-frozen-ocr-rebuild/1.0",
                    },
                )
                # Charge every attempt conservatively, including failed reads,
                # so retrying a partial response cannot escape the byte limit.
                # Successfully cached blocks survive subsequent retries.
                for attempt in range(4):
                    if self.reserved_transfer + size > self.limit:
                        raise ValueError("NVIDIA HTTP range transfer exceeds frozen limit")
                    self.reserved_transfer += size
                    try:
                        with urllib.request.urlopen(request, timeout=45) as response:
                            if (
                                response.status != 206
                                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                            ):
                                raise ValueError("Server did not honor exact bounded HTTP range")
                            data = response.read(size + 1)
                        if len(data) < size:
                            raise ConnectionError("Truncated HTTP range response")
                        if len(data) > size:
                            raise ValueError("HTTP range length mismatch")
                        break
                    except urllib.error.HTTPError as error:
                        if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                            raise RuntimeError(
                                f"Pinned NVIDIA range failed: {start}-{end}: HTTP {error.code}"
                            ) from error
                        time.sleep(2**attempt)
                    except (
                        urllib.error.URLError,
                        http.client.RemoteDisconnected,
                        TimeoutError,
                        ConnectionError,
                    ) as error:
                        if attempt == 3:
                            raise RuntimeError(f"Pinned NVIDIA range failed: {start}-{end}: {error}") from error
                        time.sleep(2**attempt)
                self.transferred += size
                self.cache[number] = data
                self.receipts.append(
                    {"start": start, "end": end, "bytes": size, "sha256": digest(data), "attempts": attempt + 1}
                )
            take = min(stop_position - self.position, end - self.position + 1)
            offset = self.position - start
            chunks.append(self.cache[number][offset : offset + take])
            self.position += take
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def rebuild(sources_file: Path, output: Path, local_sources: Path | None = None) -> dict:
    raw_spec = sources_file.read_bytes()
    spec = json.loads(raw_spec)
    if PIL.__version__ != spec["encoding"]["pillow"]:
        raise ValueError(f"Pillow must be {spec['encoding']['pillow']}; found {PIL.__version__}")
    artifacts = spec["artifacts"]
    paths = [a["path"] for a in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate artifact paths in frozen source manifest")
    sources = {s["source_id"]: s for s in spec["sources"]}
    if len(sources) != len(spec["sources"]):
        raise ValueError("Duplicate source IDs")
    needed = {a["source_id"] for a in artifacts}
    output.mkdir(parents=True, exist_ok=True)
    range_reader = None
    original_count = 0
    crop_count = 0
    http_bytes = 0
    with tempfile.TemporaryDirectory(prefix="frozen-ocr-pages-") as cache:
        cached: dict[str, Path] = {}
        # An offline rebuild still reads original photographs/pages, never
        # previously generated crops, and verifies every frozen source SHA.
        # This is useful after public HTTP acquisition has already succeeded.
        if local_sources is not None:
            for sid in needed:
                source = sources[sid]
                original = safe_path(local_sources, source["image"])
                data = original.read_bytes()
                check(data, source["image_sha256"], source["image_bytes"], str(original))
                page = Path(cache) / (sid.replace(":", "_") + ".jpg")
                page.write_bytes(data)
                cached[sid] = page
        nvidia_sources = [
            s
            for s in spec["sources"]
            if s["dataset"] == "nvidia/OCR-Synthetic-Multilingual-v1" and s["source_id"] in needed
        ]
        # Fetch smaller independent photos first. A transient Commons error then
        # fails before repeating the larger bounded NVIDIA acquisition.
        photo_ids = sorted(needed - cached.keys() - {s["source_id"] for s in nvidia_sources})

        def fetch_photo(sid):
            source = sources[sid]
            data = download(source["download_url"], source["image_sha256"], source["image_bytes"])
            page = Path(cache) / (sid.replace(":", "_") + ".jpg")
            page.write_bytes(data)
            return sid, page, len(data)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for sid, page, size in pool.map(fetch_photo, photo_ids):
                cached[sid] = page
                http_bytes += size
        print(f"Verified {len(photo_ids)} public source photographs", flush=True)
        if nvidia_sources and local_sources is None:
            import h5py
            import numpy

            if h5py.__version__ != spec["encoding"]["h5py"] or numpy.__version__ != spec["encoding"]["numpy"]:
                raise ValueError("h5py/numpy versions differ from the frozen acquisition environment")
            upstream = spec["nvidia"]
            range_reader = RangeFile(upstream["url"], upstream["size"], upstream["max_http_bytes"])
            with h5py.File(range_reader, "r") as shard:
                for source in sorted(nvidia_sources, key=lambda s: s["index"]):
                    index = source["index"]
                    data = shard["images"][index].tobytes()
                    check(data, source["image_sha256"], source["image_bytes"], source["source_id"])
                    sample_id = shard["sample_ids"][index]
                    if isinstance(sample_id, bytes):
                        sample_id = sample_id.decode("utf-8")
                    if sample_id != source["sample_id"]:
                        raise ValueError("Pinned NVIDIA sample ID mismatch")
                    page = Path(cache) / (source["source_id"] + ".jpg")
                    page.write_bytes(data)
                    cached[source["source_id"]] = page
                    if len(cached) % 40 == 0:
                        print(
                            f"Verified {len(cached) - len(photo_ids)}/{len(nvidia_sources)} NVIDIA pages; {range_reader.transferred} range bytes",
                            flush=True,
                        )
            http_bytes += range_reader.transferred
        for artifact in artifacts:
            source = sources[artifact["source_id"]]
            page = cached[artifact["source_id"]]
            if artifact["kind"] == "source_image":
                data = page.read_bytes()
                original_count += 1
            elif artifact["kind"] == "crop":
                with Image.open(page) as original:
                    image = original.convert("RGB")
                    if list(image.size) != source.get("dimensions", source.get("size")):
                        raise ValueError("Frozen image dimensions changed")
                    box = artifact["crop_xyxy"]
                    if not (
                        len(box) == 4 and 0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height
                    ):
                        raise ValueError("Frozen crop outside source image")
                    buffer = io.BytesIO()
                    image.crop(box).save(buffer, format="PNG", **spec["encoding"]["png_save_kwargs"])
                    data = buffer.getvalue()
                crop_count += 1
            else:
                raise ValueError(f"Unknown artifact kind: {artifact['kind']}")
            check(data, artifact["sha256"], artifact["bytes"], artifact["path"])
            target = safe_path(output, artifact["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    receipt = {
        "sources_manifest_sha256": digest(raw_spec),
        "artifact_count": len(artifacts),
        "input_original_source_count": len(needed),
        "nvidia_referenced_source_pages": len(nvidia_sources),
        "source_image_count": original_count,
        "crop_count": crop_count,
        "verified_artifacts": len(artifacts),
        "http_bytes": http_bytes,
        "nvidia_range_http_bytes": range_reader.transferred if range_reader else 0,
        "nvidia_range_conservative_attempt_bytes": range_reader.reserved_transfer if range_reader else 0,
        "nvidia_ranges": range_reader.receipts if range_reader else [],
        "pillow": PIL.__version__,
        "selection_or_label_changes": False,
        "source_method": "verified original local photographs/pages"
        if local_sources
        else "public HTTP URLs and pinned bounded HDF5 ranges",
    }
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Data root; artifacts preserve ocr/ and shared vision/ paths"
    )
    parser.add_argument("--receipt", type=Path, help="Optional proof JSON outside image archives")
    parser.add_argument(
        "--local-sources",
        type=Path,
        help="Optional repository root holding previously downloaded original photographs/pages; never uses existing crop PNGs",
    )
    args = parser.parse_args()
    receipt = rebuild(args.sources, args.output, args.local_sources)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "nvidia_ranges"}, indent=2))


if __name__ == "__main__":
    main()
