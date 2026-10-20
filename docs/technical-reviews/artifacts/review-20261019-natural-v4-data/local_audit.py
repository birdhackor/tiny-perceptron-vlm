import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
DATA = ROOT / "docs/natural-assistant/v4"
REV = "3a282925bb46bb68c87fed4157ae4df5f6de30ee"
STUDENT = "ee17b8871cfe2ba9807b796dd11f339b5dae3b5c"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def splits(rows, key=None):
    return {s: len({r[key] for r in rows if r["split"] == s}) if key else sum(r["split"] == s for r in rows)
            for s in ("train", "validation", "test")}


def git_file(ref, path):
    return subprocess.check_output(["git", "show", ref + ":" + path], cwd=ROOT)


def pinned_module(ref, path):
    raw = git_file(ref, path)
    spec = importlib.util.spec_from_loader("reviewed_" + Path(path).stem, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(ROOT / path)
    exec(compile(raw, ref + ":" + path, "exec"), mod.__dict__)
    return mod


m = load(DATA / "manifest.json")
result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
          "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; no model loaded"}}
paths = ["docs/course-review-20261019/freeze-03/sources/natural-v4-data.md",
         "docs/course-review-20261019/freeze-04.json", "docs/natural-assistant/v4/DATA.md",
         "docs/natural-assistant/v4/manifest.json", "docs/natural-assistant/v4/selection.json",
         "scripts/fetch_natural_data.py", "scripts/build_natural_v4_assets.py",
         "scripts/prepare_natural_v4_ocr.py", "scripts/replay_natural_v4_voice.py",
         "docs/selftrained/v2-manifest.json"]
paths += [str(p.relative_to(ROOT)) for p in (DATA / "data").rglob("*.json*")]
paths += ["tiny_perceptron/natural_assistant.py", "tiny_perceptron/natural_concepts.py", "scripts/selftrained/prepare_vision_ocr.py"]
result["inputs"] = {p: {"sha256": sha((ROOT / p).read_bytes()), "bytes": (ROOT / p).stat().st_size} for p in sorted(paths)}
result["pins"] = []
for ref, path in [(STUDENT, "scripts/fetch_natural_data.py"), (STUDENT, "docs/natural-assistant/v4/manifest.json"),
                  (REV, "docs/natural-assistant/v4/manifest.json"), (REV, "scripts/build_natural_v4_assets.py"),
                  (REV, "scripts/prepare_natural_v4_ocr.py"), (REV, "scripts/replay_natural_v4_voice.py")]:
    raw = git_file(ref, path)
    result["pins"].append({"command": f"git show {ref}:{path}", "sha256": sha(raw), "bytes": len(raw),
                           "same_as_working_tree": raw == (ROOT / path).read_bytes()})

table = defaultdict(Counter)
for row in m["rows"]:
    if row["task"] == "scene":
        category = "docci_" + row["references"]["qa_type"]
    elif row["task"] in ("ocr", "ocr_order"):
        category = ("nvidia_" if row["synthetic"] else "commons_") + row["task"]
    elif row["task"] == "chat":
        category = "oasst2" if row["source"] == "oasst2-curated-zh-v4" else "project_chat"
    else:
        category = row["task"]
    table[category][row["split"]] += 1
for row in m["audio_rows"]:
    table[row["task"]][row["split"]] += 1
result["question_table"] = {k: {s: v[s] for s in ("train", "validation", "test")} for k, v in table.items()}
result["row_totals"] = splits(m["rows"])
result["audio_totals"] = splits(m["audio_rows"])
result["family_cross_split"] = []
for key in ("family", "image", "audio"):
    groups = defaultdict(set)
    for row in m["rows"] + m["audio_rows"]:
        if row.get(key):
            groups[row[key]].add(row["split"])
    result["family_cross_split"].extend({"key": key, "value": k, "splits": sorted(v)} for k, v in groups.items() if len(v) > 1)

vision = []
for name in ("vision-sources.json", "vision-activity-sources.json", "vision-activity-heldout-sources.json"):
    vision += load(DATA / "data" / name)["rows"]
retained = {r["image"] for r in m["rows"] if r.get("image", "").startswith("vision/")}
unused = sorted(r["image"] for r in vision if r["image"] not in retained)
vision = [r for r in vision if r["image"] in retained]
result["docci"] = {"unused_acquired_images": unused, "unique_images": len({r["image"] for r in vision}), "images": splits(vision, "image"),
                   "families": splits(vision, "family"),
                   "publisher_thumbnails": all("storage.googleapis.com/docci/thumbnails/" in r["url"] and "generation=" in r["url"] for r in vision)}
vrows = [r for r in m["rows"] if r["task"] == "scene"]
result["vision_annotation_records"] = {"rows": len(vrows),
    "missing_original_support_or_review": [r["id"] for r in vrows if not (r["references"].get("official_description") and r["references"].get("support_sentences") and r["source"].get("image_sha256") and r["source"].get("image_review"))],
    "recorded_methods": sorted({r["source"]["image_review"]["method"] for r in vrows}),
    "cat_records": [{"id": r["id"], "answer": r["answer"], "user": r["user"], "qa_type": r["references"]["qa_type"],
                     "original_caption_sha256": sha(r["references"]["official_description"].encode()),
                     "source_image_sha256": r["source"]["image_sha256"]} for r in vrows if "train_01827" in r["id"]]}
ocr = load(DATA / "data/ocr-sources.json")
by_id = {s["source_id"]: s for s in ocr["sources"]}
used = {a["source_id"] for a in ocr["artifacts"]}
nv = [s for s in ocr["sources"] if s["source_id"] in used and s["dataset"] == "nvidia/OCR-Synthetic-Multilingual-v1"]
co = [s for s in ocr["sources"] if s["source_id"] in used and s["source_id"].startswith("commons-")]
co_crops = [a for a in ocr["artifacts"] if a["kind"] == "crop" and a["source_id"].startswith("commons-")]
result["ocr_materials"] = {"nvidia_retained_pages": len(nv), "nvidia_families": splits(nv, "family"),
    "nvidia_crops": sum(a["kind"] == "crop" and a["source_id"] in {s["source_id"] for s in nv} for a in ocr["artifacts"]),
    "commons_originals": splits(co, "source_id"), "commons_families": splits(co, "family"),
    "commons_crops": Counter(by_id[a["source_id"]]["split"] for a in co_crops),
    "commons_license_types": dict(Counter(s["license"] for s in co)),
    "merged_commons_families": {f: [s["title"] for s in co if s["family"] == f] for f, n in Counter(s["family"] for s in co).items() if n > 1},
    "encoding": ocr["encoding"], "limits": ocr["limits"]}
ol = lines(DATA / "data/ocr-labels.jsonl") + lines(DATA / "data/ocr-order-labels.jsonl")
transcription = [r for r in ol if r["source_id"].startswith("commons-") and r["task"] == "ocr_transcription"]
result["commons_transcription_language"] = {"counts": dict(Counter(r["language"] for r in transcription)),
    "non_zh_hant": [{"id": r["id"], "split": r["split"], "language": r["language"], "answer": r["answer"]} for r in transcription if r["language"] != "zh_hant"],
    "extra_order_splits": splits(lines(DATA / "data/ocr-order-labels.jsonl"))}
chat = lines(DATA / "data/chat-labels.jsonl")
project = [r for r in chat if r["source"] == "project-authored-traditional-chinese-v4"]
source = [r for r in chat if r["source"] == "oasst2-curated-zh-v4"]
ledger = lines(DATA / "data/chat-curation-ledger.jsonl")
changes = [c for r in ledger for c in r["changes"]]
result["chat"] = {"source_families": splits(source, "family"), "project_families": splits(project, "family"),
                  "project_license_synthetic": dict(Counter((str(r["license"]) + "/" + str(r["synthetic"])) for r in project)),
                  "ledger_trees": len(ledger), "changes": len(changes),
                  "malformed_changes": [c for c in changes if not all(k in c for k in ("kind", "before", "after"))],
                  "source_ids_present": all(r.get("source_message_ids") for r in source),
                  "original_snapshot_exists": (ROOT / load(DATA / "data/chat-sources.json")["source_message_snapshot"]).is_file()}
result["voices"] = {"tasks": dict(Counter(r["task"] for r in m["audio_rows"])),
    "training": sum(r["split"] == "train" for r in m["audio_rows"]),
    "non_null_answers": [r["id"] for r in m["audio_rows"] if r["answer"] is not None],
    "synthetic": sum(r["synthetic"] for r in m["audio_rows"]),
    "aishell_spaces_removed": all(r["user"] == r["original_spaced_source_transcription"].replace(" ", "") for r in m["audio_rows"] if r["task"] == "speech_chat"),
    "aishell_speakers": {s: sorted({r["source_speaker_id"] for r in m["audio_rows"] if r["split"] == s and r["task"] == "speech_chat"}) for s in ("validation", "test")},
    "fleurs_speaker_information": load(DATA / "data/voice-sources.json")["speaker_disjointness"]}
result["bundles"] = [{"path": a["path"], "compressed_bytes": a["bytes"], "uncompressed_bytes": sum(f["bytes"] for f in a["files"]),
    "files": len(a["files"]), "text_notices": [f["path"] for f in a["files"] if f["path"].endswith((".txt", ".json"))]} for a in m["archives"]]
result["bundle_total"] = {"compressed_bytes": sum(a["bytes"] for a in m["archives"]), "uncompressed_bytes": sum(f["bytes"] for f in m["files"])}
result["lfs_pointers"] = {a["path"]: (ROOT / a["path"]).read_text() for a in m["archives"]}
result["selection"] = load(DATA / "selection.json")
result["selftrained_scope"] = load(ROOT / "docs/selftrained/v2-manifest.json")["scope"]
build = pinned_module(REV, "scripts/build_natural_v4_assets.py")
rows, audio = build.assemble()
result["pinned_assembly"] = {"rows_equal": rows == m["rows"], "audio_equal": audio == m["audio_rows"]}
fetch = pinned_module(STUDENT, "scripts/fetch_natural_data.py")
summary = fetch.summary(fetch.load_manifest(DATA / "manifest.json", sha((DATA / "manifest.json").read_bytes())), REV, sha((DATA / "manifest.json").read_bytes()))
result["pinned_list_result"] = summary
sys.path.insert(0, str(ROOT))
from tiny_perceptron import natural_assistant as runtime
from tiny_perceptron.natural_concepts import text_error_report
sample = project[-1]
result["runtime_behaviors"] = {"raw_cer": text_error_report("今天去臺北", "今天去台北"),
    "empty_reference": text_error_report("", "字"),
    "history_preserved": runtime.messages_for(sample, ART)[1:-1] == [
        {"role": t["role"], "content": [{"type": "text", "text": t["content"]}]} for t in sample["history"]],
    "same_sample_on_repeat": runtime.training_row_at([sample], 0, 1)["id"] == runtime.training_row_at([sample], 9, 1)["id"]}

class Response(io.BytesIO):
    def __init__(self, body, url):
        super().__init__(body)
        self.url = url
        self.headers = {"Content-Length": str(len(body))}

    def geturl(self):
        return self.url


with tempfile.TemporaryDirectory(prefix="fixture-", dir=ART) as tmp:
    base = Path(tmp)
    archives, files, responses = [], [], {}
    for archive, group in fetch.V4_ARCHIVES.items():
        body = (group + " exact bytes").encode()
        name = group + "/example.bin"
        record = {"path": name, "bytes": len(body), "sha256": sha(body)}
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            member = tarfile.TarInfo(name)
            member.size = len(body)
            tar.addfile(member, io.BytesIO(body))
        raw = buf.getvalue()
        files.append(record)
        archives.append({"path": archive, "bytes": len(raw), "sha256": sha(raw), "files": [record]})
        responses[archive] = raw
    fm = {"schema_version": 1, "dataset_version": "natural-assistant-v4-fixture", "archives": archives, "files": files, "rows": [], "audio_rows": []}
    fp = base / "manifest.json"
    fp.write_text(json.dumps(fm))
    calls = []

    def opener(request, timeout):
        calls.append({"url": request.full_url, "headers": dict(request.header_items())})
        raw = fp.read_bytes() if "raw.githubusercontent.com" in request.full_url else next(v for k, v in responses.items() if request.full_url.endswith(k))
        return Response(raw, request.full_url)

    def run_fetch():
        return fetch.fetch_data(fp, base / "download", revision=REV, manifest_sha256=sha(fp.read_bytes()), opener=opener)

    first = run_fetch()
    before = len(calls)
    second = run_fetch()
    changed = base / "download/voice/example.bin"
    changed.write_bytes(b"user-modified")
    try:
        run_fetch()
        rejection = "unexpected acceptance"
    except ValueError as error:
        rejection = str(error)
    result["pinned_download_fixture"] = {"initial": first["status"], "files_verified": first["files_verified"], "calls": calls,
        "reuse": second["status"], "reuse_and_mismatch_network_calls": len(calls) - before,
        "mismatch_rejection": rejection, "changed_bytes_preserved": changed.read_bytes() == b"user-modified"}

spec = importlib.util.spec_from_file_location("checked_ocr", ROOT / "scripts/prepare_natural_v4_ocr.py")
ocrmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocrmod)
result["ocr_pinned_ast_equal"] = __import__("ast").dump(__import__("ast").parse(git_file(REV, "scripts/prepare_natural_v4_ocr.py"))) == __import__("ast").dump(__import__("ast").parse((ROOT / "scripts/prepare_natural_v4_ocr.py").read_bytes()))
reader = ocrmod.RangeFile("https://example.invalid/frozen.h5", 1024, 10)
try:
    reader.read(11)
except ValueError as error:
    result["range_limit_rejection"] = str(error)
reader = ocrmod.RangeFile("https://example.invalid/frozen.h5", 1024, 1024)


class FullResponse(io.BytesIO):
    status = 200
    headers = {}


previous = ocrmod.urllib.request.urlopen
ocrmod.urllib.request.urlopen = lambda *a, **k: FullResponse(b"not-a-range")
try:
    reader.read(1)
except ValueError as error:
    result["full_file_response_rejection"] = str(error)
finally:
    ocrmod.urllib.request.urlopen = previous
from importlib import metadata
result["dependency_versions"] = {}
for name in ("pillow", "h5py", "numpy", "soundfile"):
    try:
        result["dependency_versions"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        result["dependency_versions"][name] = "not installed in review interpreter"
out = ART / "local-audit.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(out.relative_to(ROOT)), "sha256": sha(out.read_bytes()), "question_table": result["question_table"],
                  "docci": result["docci"], "ocr_materials": result["ocr_materials"], "bundle_total": result["bundle_total"],
                  "pinned_assembly": result["pinned_assembly"], "fixture": result["pinned_download_fixture"]}, ensure_ascii=False, indent=2))
