import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
DATA = ROOT / "docs/natural-assistant/v4"
REV = "3a282925bb46bb68c87fed4157ae4df5f6de30ee"
manifest = json.loads((DATA / "manifest.json").read_bytes())
result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
          "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; no model loaded"},
          "accessed_on": datetime.now(UTC).date().isoformat(), "requests": []}


def get(url, maximum):
    cmd = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "55", "--max-filesize", str(maximum), url]
    proc = subprocess.run(cmd, capture_output=True)
    record = {"url": url, "command": cmd, "exit_code": proc.returncode, "stderr": proc.stderr.decode(errors="replace"),
              "bytes": len(proc.stdout), "sha256": hashlib.sha256(proc.stdout).hexdigest()}
    result["requests"].append(record)
    return proc.stdout if proc.returncode == 0 else None, record


url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{REV}/docs/natural-assistant/v4/manifest.json"
raw, record = get(url, 16 * 1024 * 1024)
record["matches_local_manifest"] = raw == (DATA / "manifest.json").read_bytes() if raw is not None else False
print(json.dumps(record, ensure_ascii=False), flush=True)

for archive in manifest["archives"]:
    url = f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{REV}/{archive['path']}"
    raw, record = get(url, archive["bytes"] + 1)
    if raw is not None:
        expected = {f["path"]: f for f in archive["files"]}
        record["archive_sha_matches"] = record["sha256"] == archive["sha256"]
        record["archive_size_matches"] = len(raw) == archive["bytes"]
        actual = {}
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    raise ValueError("壓縮包內有非一般檔案")
                body = tar.extractfile(member).read()
                actual[member.name] = {"path": member.name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        record["members"] = len(actual)
        record["members_match_manifest"] = actual == expected
    print(json.dumps(record, ensure_ascii=False), flush=True)
    raw = None

raw, record = get("https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines", 16 * 1024 * 1024)
if raw is not None:
    captions = {r["example_id"]: r["description"] for line in raw.splitlines() if (r := json.loads(line))}
    checked = [r for r in manifest["rows"] if r["task"] == "scene"]
    record["checked_rows"] = len(checked)
    record["mismatched_original_captions"] = [r["id"] for r in checked if r["references"]["official_description"] != captions.get(r["source"]["original_id"])]
    record["original_version"] = record["sha256"]
print(json.dumps(record, ensure_ascii=False), flush=True)
raw = None

ocr = json.loads((DATA / "data/ocr-sources.json").read_bytes())
used = {a["source_id"] for a in ocr["artifacts"]}
commons = [s for s in ocr["sources"] if s["source_id"] in used and s["source_id"].startswith("commons-")]
for start in range(0, len(commons), 40):
    group = commons[start:start + 40]
    url = "https://commons.wikimedia.org/w/api.php?" + urlencode({"action": "query", "format": "json", "pageids": "|".join(str(s["source_page_id"]) for s in group),
            "prop": "imageinfo", "iiprop": "extmetadata", "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist"})
    raw, record = get(url, 2 * 1024 * 1024)
    if raw is not None:
        try:
            pages = json.loads(raw)["query"]["pages"]
            record["licenses"] = [{"source_id": s["source_id"], "declared": s["license"],
                "official": pages[str(s["source_page_id"])]["imageinfo"][0]["extmetadata"]["LicenseShortName"]["value"]} for s in group]
        except (KeyError, json.JSONDecodeError) as error:
            record["parse_error"] = str(error)
    print(json.dumps(record, ensure_ascii=False), flush=True)

for name, version in (("Pillow", "12.3.0"), ("h5py", "3.16.0"), ("numpy", "2.5.3"), ("soundfile", "0.13.1")):
    raw, record = get(f"https://pypi.org/pypi/{name}/{version}/json", 2 * 1024 * 1024)
    if raw is not None:
        d = json.loads(raw)
        record["release"] = {"name": d["info"]["name"], "version": d["info"]["version"], "requires_python": d["info"]["requires_python"],
                             "files": len(d["urls"]), "cp313_wheels": sum("cp313" in f["filename"] for f in d["urls"])}
    print(json.dumps(record, ensure_ascii=False), flush=True)

output = ART / "remote-audit.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
