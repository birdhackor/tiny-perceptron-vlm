"""Read-only provider acquisition. No asset payload or model inference is requested."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("public-location-v1.json")
CACHE = Path("/workspace/work/reference/omni-P5-model-guide")
CACHE.mkdir(parents=True, exist_ok=True)
REPO = "birdhackor/tiny-perceptron-vlm"
TAGS = ["omni-core-candidate-cpu-core-40988311100-1",
        "omni-final-candidate-cpu-final-41004229262-1",
        "omni-mature-candidate-mature-kev-41011677439-1"]

def acquire(tag):
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                "User-Agent": "independent-model-guide-technical-review"})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read()
        code = response.status
    cache = CACHE / (tag + ".json")
    cache.write_bytes(raw)
    value = json.loads(raw)
    return {"url": url, "read_at": datetime.now(UTC).isoformat(), "http_status": code,
            "full_response_bytes": len(raw), "full_response_sha256": hashlib.sha256(raw).hexdigest(),
            "outside_git_full_response": str(cache),
            "release": {key: value[key] for key in ("id", "tag_name", "target_commitish", "draft", "prerelease", "published_at", "html_url")},
            "assets": [{key: row[key] for key in ("id", "name", "size", "digest", "browser_download_url")}
                       for row in value["assets"]]}

with ThreadPoolExecutor(max_workers=3) as pool:
    releases = list(pool.map(acquire, TAGS))
descriptor = json.loads((ROOT / "docs/omni-products/v1/public-release.json").read_bytes())
assets = {row["name"]: row for row in releases[0]["assets"]}
checks = []
for kind, expected in descriptor["components"].items():
    actual = assets[expected["name"]]
    matches = (actual["id"] == expected["asset_id"] and actual["size"] == expected["bytes"]
               and actual["digest"] == "sha256:" + expected["sha256"])
    assert matches
    checks.append({"component": kind, "asset_id_bytes_sha_match": matches,
                   "browser_download_url": actual["browser_download_url"]})
assert all(not row["release"]["draft"] for row in releases)
result = {"created_at": datetime.now(UTC).isoformat(),
          "reviewer_task": "/root/omni_p5_technical_model_guide", "anonymous_requests": True,
          "authorization_header_sent": False, "model_asset_payloads_downloaded": 0,
          "releases": releases, "manifest_checks": checks,
          "scope": "Provider's original public metadata and asset identity; no new payload roundtrip or inference."}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"metadata_releases": len(releases), "matching_components": len(checks),
                  "model_payload_downloads": 0, "output": str(OUT.relative_to(ROOT))}))
