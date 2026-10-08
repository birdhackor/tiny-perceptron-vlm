"""Download only a selected course model and verify its pinned public HF files."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/course-experiments/public-models.json"


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in value:
        raise ValueError(f"Invalid model file path: {value!r}")
    return path


def fetch_model(spec, output):
    from huggingface_hub import hf_hub_download

    revision = spec["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Published model revision must be a pinned HF commit")
    folder = Path(output) / safe_relative(spec["id"])
    folder.mkdir(parents=True, exist_ok=True)
    for file in spec["files"]:
        remote = safe_relative(file["path"])
        destination = folder / safe_relative(file["output"])
        cached = Path(hf_hub_download(spec["repo"], remote.as_posix(), revision=revision, token=False))
        digest = hashlib.sha256(cached.read_bytes()).hexdigest()
        if digest != file["sha256"] or cached.stat().st_size != file["bytes"]:
            raise ValueError(f"Model file does not match the published manifest: {remote}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
        print(json.dumps({"path": str(destination), "sha256": digest}, ensure_ascii=False))
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", help="Experiment ID shown by --list; no implicit download of all models")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/course")
    args = parser.parse_args()
    if not MANIFEST.is_file():
        parser.error("The public model manifest has not been published yet")
    models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    if args.list or not args.model:
        for spec in models:
            print(json.dumps({key: spec[key] for key in ("id", "repo", "revision")}, ensure_ascii=False))
        return
    selected = [spec for spec in models if spec["id"] == args.model]
    if len(selected) != 1:
        parser.error("Unknown model; use --list to see the available experiments")
    fetch_model(selected[0], args.output)


if __name__ == "__main__":
    main()
