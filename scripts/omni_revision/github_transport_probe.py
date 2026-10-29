"""在 Actions 驗證完整小型 checkpoint 的 Release 傳輸，不啟動 Modal。

只用標準庫建立兩參數線性模型的一次 SGD 更新、momentum 和 RNG 狀態。
這是傳輸／同環境接續運算 fixture，不是 A/C 的訓練或能力證據。
"""

import argparse
import hashlib
import io
import json
import os
import random
import struct
import subprocess
import tarfile
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY = "birdhackor/tiny-perceptron-vlm"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


def update(weights, optimizer, rng):
    """同一運算順序下可精確重放的微型 SGD＋momentum。"""
    examples = [(1.0, 0.5 + rng.gauss(0, 0.01)), (2.0, 1.0 + rng.gauss(0, 0.01))]
    errors = [weights[0] * x + weights[1] - y for x, y in examples]
    gradients = [sum(error * x for error, (x, _) in zip(errors, examples)), sum(errors)]
    for i, gradient in enumerate(gradients):
        optimizer["velocity"][i] = optimizer["momentum"] * optimizer["velocity"][i] + gradient
        weights[i] -= optimizer["learning_rate"] * optimizer["velocity"][i]


def make_fixture(path):
    rng = random.Random(113)
    weights = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)]
    optimizer = {"learning_rate": 0.01, "momentum": 0.9, "velocity": [0.0, 0.0]}
    update(weights, optimizer, rng)
    files = {
        "weights.bin": struct.pack("<2d", *weights),
        "training-state.json": canonical({"optimizer": optimizer, "rng": rng.getstate(), "next_step": 1}),
        "config.json": canonical({"model": "y=w*x+b", "dtype": "little-endian-float64", "parameters": 2}),
    }
    files["manifest.json"] = canonical({name: checksum(raw) for name, raw in files.items()})
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(raw), 0
            archive.addfile(info, io.BytesIO(raw))
    if path.stat().st_size > 32768:
        raise ValueError("Transport fixture exceeds its 32KiB bound")
    update(weights, optimizer, rng)
    return struct.pack("<2d", *weights)


def verify_fixture(path, expected):
    with tarfile.open(path, "r:gz") as archive:
        names = {item.name for item in archive.getmembers()}
        if names != {"weights.bin", "training-state.json", "config.json", "manifest.json"}:
            raise ValueError("Checkpoint fixture contains unexpected files")
        files = {name: archive.extractfile(name).read() for name in names}
    manifest = json.loads(files["manifest.json"])
    if manifest != {name: checksum(raw) for name, raw in files.items() if name != "manifest.json"}:
        raise ValueError("Inner checkpoint digests differ")
    weights = list(struct.unpack("<2d", files["weights.bin"]))
    training = json.loads(files["training-state.json"])
    if training["next_step"] != 1:
        raise ValueError("Continuation lost its next-step identity")
    rng = random.Random()
    rng.setstate(tuples(training["rng"]))
    update(weights, training["optimizer"], rng)
    if struct.pack("<2d", *weights) != expected:
        raise ValueError("Optimizer/RNG continuation differs from uninterrupted execution")


def command(arguments, payload=None, allow_missing=False):
    result = subprocess.run(arguments, input=payload, capture_output=True, timeout=90)
    if result.returncode:
        if allow_missing and b"HTTP 404" in result.stderr:
            return None
        # Redirect URLs and credential diagnostics must never enter a public log.
        raise RuntimeError("GitHub transport command rejected; inspect the operation stage")
    return result.stdout


def api(endpoint, method="GET", payload=None, allow_missing=False):
    arguments = ["gh", "api", f"repos/{REPOSITORY}/{endpoint}", "--method", method]
    if payload is not None:
        arguments += ["--input", "-"]
    raw = command(arguments, canonical(payload) if payload is not None else None, allow_missing)
    return json.loads(raw) if raw else None


def find_draft(tag, body):
    """REST tag lookup omits drafts; authenticated release listing includes them.

    A bounded scan can find our exact nonce. Failure to find it cannot prove that
    a create request with an unknown outcome never reached the server.
    """
    for page in range(1, 6):
        releases = api(f"releases?per_page=100&page={page}")
        for release in releases:
            if release.get("tag_name") == tag:
                if release.get("draft") is not True or release.get("body") != body:
                    raise ValueError("Release nonce exists but ownership differs")
                return release
        if len(releases) < 100:
            break
    return None


def probe(output):
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise ValueError("Use this probe only in the authorized repository's Actions runner")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    tag = "omni-transport-probe-" + uuid.uuid4().hex
    body = "Unpublished checkpoint permission fixture; owned nonce " + tag
    release_id = None
    create_started = False
    result = {
        "schema": "omni-actions-checkpoint-transport-v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "source_revision": revision,
        "repository": REPOSITORY,
        "tag": tag,
        "modal_compute_started": False,
        "stage": "create_fixture",
        "scope": "Authenticated draft asset transport and exact CPU fixture continuation; not product training, public anonymous download or GPU resume verification.",
        "success": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="omni-transport-") as temporary:
            directory = Path(temporary)
            package = directory / "checkpoint-fixture.tar.gz"
            expected = make_fixture(package)
            result.update(archive_bytes=package.stat().st_size, archive_sha256=checksum(package.read_bytes()))
            result["stage"] = "create_draft"
            create_started = True
            release = api(
                "releases",
                "POST",
                {
                    "tag_name": tag,
                    "target_commitish": revision,
                    "name": "Temporary checkpoint transport probe",
                    "body": body,
                    "draft": True,
                    "prerelease": True,
                },
            )
            release_id = release["id"]
            if release["draft"] is not True or release["tag_name"] != tag or release["body"] != body:
                raise ValueError("Draft identity is unexpected")
            result.update(release_id=release_id, stage="upload_asset")
            command(["gh", "release", "upload", tag, str(package), "--repo", REPOSITORY])
            release = api(f"releases/{release_id}")
            if len(release["assets"]) != 1 or release["assets"][0]["name"] != package.name:
                raise ValueError("Draft asset identity is unexpected")
            result.update(asset_id=release["assets"][0]["id"], stage="download_asset")
            destination = directory / "download"
            destination.mkdir()
            command(
                [
                    "gh",
                    "release",
                    "download",
                    tag,
                    "--repo",
                    REPOSITORY,
                    "--pattern",
                    package.name,
                    "--dir",
                    str(destination),
                ]
            )
            received = destination / package.name
            if checksum(received.read_bytes()) != result["archive_sha256"]:
                raise ValueError("Release transport changed archive bytes")
            verify_fixture(received, expected)
            result.update(success=True, authenticated_roundtrip=True, optimizer_rng_exact_continuation=True)
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        result["cleanup_completed"] = False
        try:
            owned = api(f"releases/{release_id}", allow_missing=True) if release_id else find_draft(tag, body)
            if owned is None and (release_id is not None or not create_started):
                result["cleanup_completed"] = True
            elif owned is None:
                result["creation_outcome_unknown"] = True
            elif owned["draft"] is True and owned["tag_name"] == tag and owned["body"] == body:
                api(f"releases/{owned['id']}", "DELETE")
                result["cleanup_completed"] = api(f"releases/{owned['id']}", allow_missing=True) is None
        except Exception as error:
            result["cleanup_error_type"] = type(error).__name__
        (output / "github-transport.json").write_bytes(canonical(result) + b"\n")
        print(json.dumps({key: result.get(key) for key in ("stage", "success", "cleanup_completed", "archive_bytes")}))
    if not result["success"] or not result["cleanup_completed"]:
        raise RuntimeError("Checkpoint transport preflight is incomplete; inspect the safe receipt")
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    probe(args.output)


if __name__ == "__main__":
    main()
