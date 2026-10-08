"""Read four anchored safety files and run a CPU audit without Modal compute.

The authenticated Volume API streams existing files only. Private checkpoints,
training state and source records stay in a temporary runner directory and are
deleted before artifact upload. This entrypoint cannot select arbitrary files.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {
    "run_id": "gha-37046840520-1",
    "revision": "a864a60bbf72583afc9bbaf45e052bd4fe076c62",
    "hf_revision": "f535d05a0a16cb68573c502dc32d2e2f55f9b085",
    "volume_prefix": "course-v1/safety",
    "files": {
        "pku-pilot.pt": (488461, "8f5e7857c1d5422bbf3adcef95a9ecba8f598b8a33ad2e0d931e4f08a8155c3b"),
        "pku-excerpts/train.jsonl": (25823, "42d11e4a1a03c4e546ab5bc4b86c8a7f11d2104c1f23d37ce94e6dd4e43dd21a"),
        "pku-excerpts/validation.jsonl": (3267, "b540a63a6b9cd83edd40fd914b89c8a502e19bea895327921c918b61607a8424"),
        "pku-excerpts/test.jsonl": (3283, "6109322ca582bbbf14ee5a083213313927cca88baac5b63d2d6561e550da4499"),
    },
}
PROBE = ROOT / "docs/technical-reviews/artifacts/integration-9.1-private-probe.py.txt"


def read_bounded(volume, relative, maximum):
    data = bytearray()
    try:
        for chunk in volume.read_file(f"{SOURCE['volume_prefix']}/{relative}"):
            if not isinstance(chunk, bytes) or len(data) + len(chunk) > maximum:
                raise ValueError("Existing audit file exceeds its anchored size")
            data.extend(chunk)
    except Exception:
        # SDK diagnostics can contain signed URLs. Do not log them.
        raise RuntimeError("Cannot read the anchored existing safety audit file") from None
    return bytes(data)


def main():
    if not PROBE.is_file():
        raise RuntimeError("The independently written CPU audit is missing")
    volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
    raw = read_bounded(volume, "result.json", 2 * 1024 * 1024)
    result = json.loads(raw)
    if result.get("revision") != SOURCE["revision"] or result.get("modal", {}).get("run_id") != SOURCE["run_id"]:
        raise RuntimeError("Existing Volume result differs from the anchored formal run")
    manifest = {row["path"]: row for row in result["artifacts"]}
    verified = []
    with tempfile.TemporaryDirectory(prefix="private-safety-audit-") as temporary:
        directory = Path(temporary)
        for relative, (size, digest) in SOURCE["files"].items():
            if manifest[relative]["bytes"] != size or manifest[relative]["sha256"] != digest:
                raise RuntimeError("Formal result manifest differs from the anchored file")
            data = read_bounded(volume, relative, size)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("Existing formal audit file size or SHA-256 differs")
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            verified.append({"path": relative, "bytes": size, "sha256": digest})
        process = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(PROBE),
                "--input",
                str(directory),
                "--output",
                str(directory / "proof.json"),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            timeout=180,
        )
        if process.returncode:
            raise RuntimeError("Independent CPU audit failed; private diagnostics are not published")
        proof = json.loads((directory / "proof.json").read_text())
        # The reviewed probe emits aggregate measurements only. Never upload
        # temporary model/data files or captured stdout/stderr.
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "scope": "Read-only authenticated Modal Volume streaming of fixed formal files; CPU audit on the GitHub runner. No Modal function/container, GPU, retraining, HF upload or student release.",
        "source": {key: value for key, value in SOURCE.items() if key != "files"},
        "verified_files": verified,
        "probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "proof": proof,
    }
    output = ROOT / "outputs/modal-course/pku-existing-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "existing_files_verified": len(verified), "gpu_used": False}))


if __name__ == "__main__":
    main()
