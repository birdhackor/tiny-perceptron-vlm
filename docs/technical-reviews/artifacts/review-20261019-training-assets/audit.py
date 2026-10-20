"""核對固定資料、資料入口與重建格式；只保存小型查核收據。"""

import contextlib
import gzip
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import wave
from collections import Counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    return {"command": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def fetch(url):
    argv = ["curl", "--location", "--fail", "--silent", "--show-error", "--max-time", "90", url]
    result = subprocess.run(argv, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    return result.stdout, {"command": argv, "bytes": len(result.stdout), "sha256": digest(result.stdout)}


def entries(raw, declared):
    contents = {}
    metadata = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as package:
        for member in package:
            if member.isfile():
                contents[member.name] = package.extractfile(member).read()
                metadata.append({"path": member.name, "bytes": member.size, "sha256": digest(contents[member.name]),
                                 "mtime": member.mtime, "uid": member.uid, "gid": member.gid, "mode": member.mode})
    observed = {row["path"]: (row["bytes"], row["sha256"]) for row in metadata}
    expected = {row["path"]: (row["bytes"], row["sha256"]) for row in declared}
    return contents, {"file_count": len(metadata), "unpacked_bytes": sum(row["bytes"] for row in metadata),
                      "all_file_hashes_match": observed == expected, "all_mtime_zero": all(r["mtime"] == 0 for r in metadata),
                      "all_modes_0644": all(r["mode"] == 0o644 for r in metadata)}


def rebuild(contents):
    result = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=result, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as package:
            for name, content in sorted(contents.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(content), 0o644, 0
                package.addfile(member, io.BytesIO(content))
    return result.getvalue()


def rows(raw):
    return [json.loads(line) for line in raw.decode().splitlines() if line]


def main():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    import pyarrow.parquet as pq
    import torch
    from tiny_perceptron.assets import unpack_asset
    from scripts.course_experiments.common import split_records
    from scripts.course_experiments.text import _deduplicate_text
    from scripts.course_experiments.modalities import _resample_8_to_16
    import numpy as np
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-training-assets/audit.py",
        "environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "platform": platform.platform(),
                        "git_head": head},
        "freeze_source_sha256": digest((ROOT / "docs/course-review-20261019/freeze-01/sources/training-assets.md").read_bytes()),
        "current_source_sha256": digest((ROOT / "assets/training/README.md").read_bytes()),
        "fetch_list": run([sys.executable, "scripts/fetch_training_assets.py", "--list"]),
        "git_lfs_version": run(["git", "lfs", "version"]),
        "git_attributes": run(["git", "check-attr", "filter", "diff", "merge", "--", "assets/training/tinystories-v1.tar.gz"]),
        "ignored_data": run(["git", "check-ignore", "data/training/example.jsonl"]),
        "packages": [],
    }
    manifest = json.loads((ROOT / "assets/training/manifest.json").read_bytes())
    result["pilot_totals"] = {"packages": len(manifest["assets"]),
                              "training_records": sum(a["training_records"] for a in manifest["assets"]),
                              "archive_bytes": sum(a["archive_bytes"] for a in manifest["assets"])}
    result["pilot_totals"]["MiB"] = result["pilot_totals"]["archive_bytes"] / 2**20
    pilot_contents = {}
    for asset in manifest["assets"]:
        url = f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{head}/{asset['archive']}"
        raw, receipt = fetch(url)
        content, summary = entries(raw, asset["files"])
        summary.update({"id": asset["id"], "download": receipt, "archive_bytes_match": len(raw) == asset["archive_bytes"],
                        "archive_hash_match": digest(raw) == asset["archive_sha256"],
                        "in_memory_rebuild_sha256": digest(rebuild(content)),
                        "in_memory_rebuild_matches": digest(rebuild(content)) == digest(raw),
                        "jsonl": []})
        for name, data in content.items():
            if name.endswith(".jsonl") and not name.endswith("rows.manifest.jsonl"):
                records = rows(data)
                summary["jsonl"].append({"path": name, "count": len(records), "keys": sorted(records[0]) if records else []})
        if asset["id"] == "tinystories":
            name = "text-initial/tinystories-train-prefix-complete.txt"
            summary["complete_story_markers"] = content[name].count(b"<|endoftext|>")
        if asset["id"] in ("tinystories", "chinese-poetry"):
            name = "text-initial/" + ("tinystories-train-512.jsonl" if asset["id"] == "tinystories" else "chinese-classical-train-365.jsonl")
            parts = split_records(_deduplicate_text(rows(content[name])), seed=42)
            families = {side: {r["family"] for r in values} for side, values in parts.items()}
            summary["executed_text_split"] = {"counts": {k: len(v) for k,v in parts.items()},
                                              "no_exact_text_family_overlap": not any(families[a] & families[b] for a,b in [("train","test"),("train","validation"),("validation","test")]),
                                              "near_duplicate_clustering": "_deduplicate_text only normalizes whitespace and removes exact text matches"}
        if asset["id"] == "fashion-mnist":
            name = next(name for name in content if name.endswith(".parquet"))
            summary["source_parquet_rows"] = pq.ParquetFile(io.BytesIO(content[name])).metadata.num_rows
            name = next(name for name in content if name.endswith("/train.jsonl"))
            summary["pilot_label_counts"] = dict(Counter(str(r["label"]) for r in rows(content[name])))
        if asset["id"] == "fsdd":
            rates = Counter()
            for name,data in content.items():
                if name.endswith(".wav"):
                    with wave.open(io.BytesIO(data)) as source:
                        rates[source.getframerate()] += 1
            summary["wav_sample_rates"] = dict(rates)
        if asset["id"] == "pku-safe-rlhf":
            name = next(name for name in content if name.endswith("train-first-100.jsonl"))
            pairs = rows(content[name])
            summary["both_responses_unsafe"] = sum(not r["is_response_0_safe"] and not r["is_response_1_safe"] for r in pairs)
        pilot_contents[asset["id"]] = content
        result["packages"].append(summary)
    # 以小型自製資料測試不覆寫與雙重雜湊契約，暫存資料限制在指定目錄。
    with tempfile.TemporaryDirectory(dir=ARTIFACT, prefix="synthetic-") as work:
        work = Path(work)
        payload = {"sample/train.jsonl": b'{"text":"one"}\n', "sample/LICENSE": b"self-authored example\n"}
        archive = work / "sample.tar.gz"
        raw = rebuild(payload)
        archive.write_bytes(raw)
        asset = {"id":"sample", "archive_bytes":len(raw), "archive_sha256":digest(raw),
                 "files":[{"path":n,"bytes":len(b),"sha256":digest(b)} for n,b in payload.items()]}
        destination = work / "data"
        first = unpack_asset(archive,asset,destination)
        second = unpack_asset(archive,asset,destination)
        (destination / "sample/train.jsonl").write_text("自訂內容")
        try:
            unpack_asset(archive,asset,destination)
        except ValueError as exc:
            conflict = str(exc)
        modified_preserved = (destination / "sample/train.jsonl").read_text() == "自訂內容"
        broken = {**asset, "files": [{**r, "sha256":"0"*64} for r in asset["files"]]}
        try:
            unpack_asset(archive,broken,work / "bad-files")
        except ValueError as exc:
            inner_hash_error = str(exc)
        archive.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        try:
            unpack_asset(archive,asset,work / "bad-archive")
        except ValueError as exc:
            archive_hash_error = str(exc)
        result["unpacker_behavior"] = {"first_written":first["files_written"],"second_written":second["files_written"],
                                        "conflict_error":conflict,"modified_content_preserved":modified_preserved,
                                        "inner_hash_error":inner_hash_error,"archive_hash_error":archive_hash_error,
                                        "failed_destinations_absent":not (work/"bad-files").exists() and not (work/"bad-archive").exists()}
    # 對下載入口只攔截 unpack，實際執行 argparse 與 all 的選取分支。
    cli = importlib.import_module("scripts.fetch_training_assets")
    selected = []
    def fake_unpack(archive,asset,destination):
        selected.append(asset["id"])
        return {"asset":asset["id"],"destination":str(destination)}
    fake_calls = []
    with patch.object(sys,"argv",["fetch_training_assets.py","--asset","all","--output",str(ARTIFACT / "unused")]), patch.object(cli,"unpack_asset",fake_unpack), patch.object(cli.subprocess,"run",lambda *a,**kw: fake_calls.append(a[0])):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            cli.main()
    result["all_selection"] = {"ids":selected,"lfs_commands":fake_calls,"mode":"真實 argparse/選取；LFS 子行程及解包以攔截函式隔離，不宣稱真實 git lfs pull 已執行"}
    source = np.linspace(-0.5,0.5,80,dtype="float32")
    resampled = _resample_8_to_16(source)
    result["resample"] = {"input_samples":len(source),"output_samples":len(resampled),"dtype":str(resampled.dtype),
                          "finite":bool(np.isfinite(resampled).all()),"duration_before_seconds":len(source)/8000,"duration_after_seconds":len(resampled)/16000}
    plan = json.loads((ROOT / "docs/course-experiments/plan.json").read_bytes())
    result["experiments"] = {"count":len(plan["sequence"]),"entries":[{k:e[k] for k in ("id","module","function","dependencies","assets")} for e in plan["sequence"]],
                             "functions_present": all(f"def {e['function']}(" in (ROOT / f"scripts/course_experiments/{e['module']}.py").read_text() for e in plan["sequence"])}
    manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_bytes())
    pkg = manifest["package"]
    raw, receipt = fetch(f"https://media.githubusercontent.com/media/{pkg['repo_id']}/{pkg['revision']}/{pkg['path']}")
    content, summary = entries(raw,manifest["records"]+manifest["assets"])
    summary.update({"download":receipt,"archive_bytes_match":len(raw)==pkg["bytes"],"archive_hash_match":digest(raw)==pkg["sha256"],
                    "jsonl_count":len(manifest["records"]),"other_file_count":len(manifest["assets"]),"record_counts":{},"unique_audio":{},"ocr":{}})
    for item in manifest["records"]:
        records = rows(content[item["path"]])
        summary["record_counts"][item["path"]] = len(records)
        if "voice" in item["path"] or "audio" in item["path"]:
            summary["unique_audio"][item["path"]] = {"keys":sorted(records[0]),"audio_field_values":{k:len({str(r.get(k)) for r in records}) for k in records[0] if any(x in k for x in ["audio","speaker","family","intent"])},"first_record":records[0]}
        if item["path"].startswith("ocr-"):
            summary["ocr"][item["path"]] = {"keys":sorted(records[0]),"first_record":records[0],"answer_characters":sorted({c for r in records for c in str(r.get("answer",r.get("text","")))})}
    summary["split_totals"] = {side:sum(count for name,count in summary["record_counts"].items() if name.endswith(f"-{side}.jsonl")) for side in ("train","validation","test")}
    summary["source_notices"] = {name:data.decode() for name,data in content.items() if name in ["DATA-NOTICE.md","licenses/minds14/PROJECT-NOTICE.md","provenance/v2/voice-augmentation-v2-NOTICE.md"]}
    result["selftrained_v2"] = summary
    result["source_hashes"] = {}
    paths = ["assets/training/manifest.json",".gitattributes",".gitignore","scripts/fetch_training_assets.py","scripts/build_training_assets.py","tiny_perceptron/assets.py","scripts/course_experiments/plan.py","docs/course-experiments/plan.json","scripts/course_experiments/text.py","scripts/course_experiments/common.py","scripts/course_experiments/modalities.py","scripts/course_experiments/alignment.py","scripts/course_experiments/compression.py","scripts/course_experiments/applications.py","docs/selftrained/v2-manifest.json","docs/selftrained/TRAINING.md","docs/selftrained/v2-public-cpu-commands.md","docs/natural-assistant/v4/DATA.md","docs/natural-assistant/v4/TRAINING.md","docs/natural-assistant/v4/STUDENT.md","docs/natural-assistant/manifest.json"]
    for name in paths:
        p=ROOT/name
        if p.is_file():
            result["source_hashes"][name]=digest(p.read_bytes())
    target = ARTIFACT / "audit-output.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(target.relative_to(ROOT)),"sha256":digest(target.read_bytes()),"pilot_totals":result["pilot_totals"],"selftrained_v2_totals":summary["split_totals"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
