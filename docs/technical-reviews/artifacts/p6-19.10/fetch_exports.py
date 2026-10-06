"""Read only the immutable release recipe, download and hash final inference files."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
RECIPE = REPO / "docs/selftrained/v2-jobs/release-final-four-exports.json"
release = json.loads(RECIPE.read_bytes())["release"]
PUBLIC_REVISION = "979cdfacc588ad0536f1c64fff96f264571cf054"  # current chapter 19.11 immutable HF pin


def fetch_export(export):
    name, revision = export["name"], PUBLIC_REVISION
    rows = []
    for entry in export["files"]:
        url = (f"https://huggingface.co/{release['repo_id']}/resolve/{revision}/"
               f"{release['prefix']}/{name}/{entry['path']}")
        local = Path("/tmp/p6-19.10-exports") / name / entry["path"]
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists() or hashlib.sha256(local.read_bytes()).hexdigest() != entry["sha256"]:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            local.write_bytes(response.content)
        data = local.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        assert sha == entry["sha256"]
        assert len(data) == entry["bytes"]
        # Persist full metadata and exact raw header; model weights remain reproducibly pinned.
        durable = ROOT / "public" / name / entry["path"]
        durable.parent.mkdir(parents=True, exist_ok=True)
        if entry["path"] == "model.safetensors":
            header_length = int.from_bytes(data[:8], "little")
            durable = durable.with_name("model.safetensors.header.json")
            durable.write_bytes(data[8:8 + header_length])
        else:
            durable.write_bytes(data)
        rows.append({"name": name, "revision": revision, "training_git_revision": export["revision"], "url": url,
                     "path": entry["path"], "bytes": len(data), "sha256": sha,
                     "snapshot_path": str(durable.relative_to(REPO)),
                     "snapshot_sha256": hashlib.sha256(durable.read_bytes()).hexdigest()})
    return rows


if __name__ == "__main__":
    final = [e for e in release["exports"] if e["name"] in ("moe-joint", "dense-joint")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch_export, final))
    out = {"recipe_path": str(RECIPE.relative_to(REPO)),
           "recipe_sha256": hashlib.sha256(RECIPE.read_bytes()).hexdigest(), "exports": results}
    (ROOT / "public-fetch.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
