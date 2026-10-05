"""Verify original immutable inputs; do not open model weight files."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parents[1]
j = json.loads((ART / "inputs/docs/course-experiments/results/tokenizer.json").read_text())
digest = lambda raw: hashlib.sha256(raw).hexdigest()
report = {"revision": j["revision"], "result_sha256": digest((ART / "inputs/docs/course-experiments/results/tokenizer.json").read_bytes()), "historical_runtime": {"torch": j["torch_version"], "python": j["python_version"], "device": j["device"], "tokenizers": "not recorded in original result; current reproduction uses 0.23.2"}, "code": [], "artifacts": [], "assets": []}
for name, expected in j["code_sha256"].items():
    raw = subprocess.check_output(["git", "show", f"{j['revision']}:{name}"], cwd=ROOT)
    observed = digest(raw)
    report["code"].append({"path": name, "expected": expected, "observed": observed, "match": expected == observed})
    assert observed == expected, name
for item in j["artifacts"]:
    if item["path"].endswith(".pt"):
        report["artifacts"].append({**item, "inspection": "metadata only; model file intentionally not opened or copied"})
        continue
    p = ART / "inputs/original-tokenizer" / item["path"]
    if not p.is_file():
        report["artifacts"].append({**item, "inspection": "current-run.json absent from local candidate directory; irrelevant to tokenizer literal roundtrip"})
        continue
    raw = p.read_bytes()
    entry = {**item, "original_candidate": "outputs/text-behavior-interface-check/tokenizer/" + item["path"], "permanent_snapshot": p.relative_to(ROOT).as_posix(), "observed_sha256": digest(raw), "observed_bytes": len(raw), "match": digest(raw) == item["sha256"] and len(raw) == item["bytes"]}
    assert entry["match"], item["path"]
    if p.suffix == ".jsonl":
        rows = [json.loads(s) for s in raw.decode().splitlines() if s.strip()]
        entry["records"] = len(rows)
        entry["raw_utf8_bytes"] = sum(len(r["text"].encode()) for r in rows)
        assert len(rows) == j["results"]["data"][p.stem]["records"]
    report["artifacts"].append(entry)
for asset in j["assets"]:
    p = ROOT / asset["archive"]
    raw = p.read_bytes()
    entry = {"id": asset["id"], "archive": asset["archive"], "expected_sha256": asset["archive_sha256"], "observed_sha256": digest(raw), "expected_bytes": asset["archive_bytes"], "observed_bytes": len(raw), "match": digest(raw) == asset["archive_sha256"] and len(raw) == asset["archive_bytes"], "training_records": asset["training_records"], "source_metadata": asset["source_metadata"], "members": []}
    assert entry["match"], asset["id"]
    with tarfile.open(p) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for item in asset["files"]:
            matching = [m for name, m in members.items() if name == item["path"] or name.endswith("/" + item["path"])]
            assert len(matching) == 1, item["path"]
            content = archive.extractfile(matching[0]).read()
            match = digest(content) == item["sha256"] and len(content) == item["bytes"]
            assert match, item["path"]
            entry["members"].append({"path": item["path"], "sha256": digest(content), "bytes": len(content), "match": match})
    report["assets"].append(entry)
report["executed_code_sha256"] = digest(Path(__file__).read_bytes())
(ART / "execution/input-provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"historical_revision": j["revision"], "historical_code_hashes_checked": len(report["code"]), "original_tokenizer_and_data_files_checked": sum(x.get("match") is True for x in report["artifacts"]), "asset_archives_checked": len(report["assets"]), "asset_members_checked": sum(len(x["members"]) for x in report["assets"]), "model_weights_opened": 0, "result": "all assertions passed"}, ensure_ascii=False, indent=2))
