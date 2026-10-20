"""獨立重算指紋、測試表格與語音編輯距離，保存來源版本。"""

import hashlib
import copy
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153"
BLIND = EVIDENCE / "blind-review"
INPUTS = {}


def read(path):
    raw = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def normalize(text, remove_space=False):
    result = unicodedata.normalize("NFKC", text).strip()
    return "".join(result.split()) if remove_space else result


def distance(left, right):
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        table[i][0] = i
    for j in range(len(right) + 1):
        table[0][j] = j
    for i, x in enumerate(left, 1):
        for j, y in enumerate(right, 1):
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + (x != y))
    return table[-1][-1]


def complete(record):
    tokens = record["generated_token_ids"]
    return bool(tokens) and tokens[-1] in record["eos_token_ids"]


manifest = read(ROOT / "docs/natural-assistant/v4/manifest.json")
selection = read(ROOT / "docs/natural-assistant/v4/selection.json")
asr_selection = read(ROOT / "docs/natural-assistant/v4/asr-selection.json")
release = read(ROOT / "docs/natural-assistant/v4/public-release.json")
audit = read(EVIDENCE / "selected-final-test-audit-summary.json")
result = read(BLIND / "result.json")
records = read(BLIND / "generations-base.json")
transcripts = read(BLIND / "transcripts.json")
scores = read(BLIND / "scored/scores.json")
subsets = read(BLIND / "descriptive-subsets.json")
grades = read(BLIND / "combined/grades.json")
packet = read(BLIND / "blind-packet.json")
protocol = read(ROOT / "docs/natural-assistant/v4/validation-protocol-lower-lr.json")

rows = {row["id"]: row for row in manifest["rows"] if row["split"] == "test"}
audio = {row["id"]: row for row in manifest["audio_rows"] if row["split"] == "test"}
transcript_map = {row["id"]: row for row in transcripts}
manual = {row["case_id"]: row for row in grades["grades"]}
record_map = {row["task"] + ":" + row["id"]: row for row in records}
case_decisions = {row["case_id"]: row for row in scores["case_decisions"]}
computed = []
sources_by_group = defaultdict(list)
for record in records:
    key = record["task"] + ":" + record["id"]
    row = rows.get(record["id"], audio.get(record["id"]))
    assert row is not None
    if record["task"] == "scene":
        group = "photo_summary" if row["references"]["qa_type"] == "scene" else "photo_fact"
    else:
        group = {"chat": "text_chat", "text_presence": "text_presence", "ocr": "single_ocr",
                 "ocr_order": "ordered_ocr", "typed_chat": "voice_typed_reference_chat",
                 "speech_chat": "voice_actual_asr_chat"}[record["task"]]
    expected_user = transcript_map[row["id"]]["transcript"] if record["task"] == "speech_chat" else row["user"]
    assert record["user"] == expected_user
    assert record["reference_answer"] == row["answer"]
    assert record.get("image") == row.get("image")
    assert len(record["generated_token_ids"]) == record["generated_tokens"]
    ended = complete(record)
    assert ended == record["ended_with_eos"]
    semantic = manual[key]["passed"] if key in manual else normalize(record["prediction"], group == "single_ocr") == normalize(row["answer"], group == "single_ocr")
    passed = ended and semantic
    assert passed == case_decisions[key]["passed"]
    computed.append({"case_id": key, "group": group, "passed": passed, "complete": ended})
    sources_by_group[group].append(row)

assert len(record_map) == len(records) == len(case_decisions) == 178
group_counts = Counter(row["group"] for row in computed)
pass_counts = Counter(row["group"] for row in computed if row["passed"])
assert dict(group_counts) == {k: v for k, v in scores["denominators"].items() if k in group_counts}
assert dict(pass_counts) == scores["correct_counts"]
truncated = [{"case_id": row["task"] + ":" + row["id"], "task": row["task"],
              "tokens": len(row["generated_token_ids"]), "stop_reason": row["stop_reason"]}
             for row in records if not complete(row)]
assert len(truncated) == 7 and all(row["task"] == "chat" and row["tokens"] == 384 for row in truncated)

asr_rows = []
for record in transcripts:
    row = audio[record["id"]]
    assert record["reference_transcript"] == row["user"]
    assert record["audio_sha256"] == row["sha256"]
    ref, hyp = row["user"], record["transcript"]
    nr, nh = normalize(ref, True), normalize(hyp, True)
    metrics = {"id": row["id"], "source": row["source"], "task": row["task"],
               "synthetic": row["synthetic"], "recording_domain": row.get("recording_domain"),
               "raw_errors": distance(ref, hyp), "raw_characters": len(ref),
               "normalized_errors": distance(nr, nh), "normalized_characters": len(nr)}
    assert metrics["raw_errors"] == record["raw_errors"]
    assert metrics["normalized_errors"] == record["errors"]
    asr_rows.append(metrics)
asr_totals = {key: sum(row[key] for row in asr_rows) for key in ["raw_errors", "raw_characters", "normalized_errors", "normalized_characters"]}
asr_totals["raw_percent"] = f'{100 * asr_totals["raw_errors"] / asr_totals["raw_characters"]:.2f}%'
asr_totals["normalized_percent"] = f'{100 * asr_totals["normalized_errors"] / asr_totals["normalized_characters"]:.2f}%'

subset_results = {}
for tag in ["action", "relation"]:
    tagged = [row for row in sources_by_group["photo_fact"] if row["references"]["qa_type"] == tag]
    decisions = [case_decisions["scene:" + row["id"]] for row in tagged]
    saved = subsets["photo_fact_subsets"][tag]
    assert {row["case_id"] for row in decisions} == {row["case_id"] for row in saved["cases"]}
    assert len(decisions) == saved["denominator"]
    assert sum(row["passed"] for row in decisions) == saved["correct"]
    subset_results[tag] = {"correct": sum(row["passed"] for row in decisions), "count": len(decisions),
                           "questions": [row["user"] for row in tagged]}

context = [row for row in sources_by_group["text_chat"] if row.get("context_dependent")]
presence = Counter(row["answer"] for row in sources_by_group["text_presence"])
photos = {row["image"] for row in sources_by_group["photo_summary"]}
fact_photos = Counter(row["image"] for row in sources_by_group["photo_fact"])
assert photos == set(fact_photos) and all(v == 2 for v in fact_photos.values())
ocr_images = {row["image"] for key in ["text_presence", "single_ocr", "ordered_ocr"] for row in sources_by_group[key]}
assert len(ocr_images) < sum(group_counts[key] for key in ["text_presence", "single_ocr", "ordered_ocr"])
assert set(row["id"] for row in sources_by_group["voice_typed_reference_chat"]) == set(row["id"] for row in sources_by_group["voice_actual_asr_chat"])

binding = audit["artifact_binding"]
assert binding == scores["artifact_binding"]
for name, expected in [("generations-base.json", binding["generation_file"]["sha256"]),
                       ("transcripts.json", binding["transcripts_sha256"]),
                       ("result.json", binding["test_result_sha256"]),
                       ("blind-packet.json", scores["blind_packet_sha256"]),
                       ("combined/grades.json", scores["manual_grades_sha256"])]:
    assert hashlib.sha256((BLIND / name).read_bytes()).hexdigest() == expected
assert selection["selected_variant"] == release["selected_variant"] == binding["selected_variant"] == "base"
assert not binding["adapters"] and release["adapter_parameters"] == result["trainable_parameters"] == 0
assert asr_selection["selected_variant"] == "turbo"
assert (result["model"], result["model_revision"]) == (release["base_model"]["repo"], release["base_model"]["revision"])
assert (result["asr_model"], result["asr_revision"]) == (release["asr_model"]["repo"], release["asr_model"]["revision"])

fingerprints = []
for value in [b"base A, adapter B", b"base A, adapter C", b"base a, adapter B"]:
    obj = hashlib.sha256(value)
    fingerprints.append({"input": value.decode("ascii"), "hex": obj.hexdigest(), "prefix": obj.hexdigest()[:12],
                         "input_type": type(value).__name__, "digest_type": type(obj.digest()).__name__,
                         "digest_bytes": len(obj.digest()), "hexdigest_chars": len(obj.hexdigest()),
                         "digest_hex_matches_hexdigest": obj.digest().hex() == obj.hexdigest()})
accepted = []
for value in [b"abc", bytearray(b"abc"), memoryview(b"abc"), "abc"]:
    try:
        accepted.append({"type": type(value).__name__, "accepted": True, "sha256": hashlib.sha256(value).hexdigest()})
    except TypeError as error:
        accepted.append({"type": type(value).__name__, "accepted": False, "error": str(error)})

spec = importlib.util.spec_from_file_location("release_verifier", ROOT / "scripts/fetch_natural_release.py")
release_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_module)
fixture_manifest = copy.deepcopy(release)
provenance = {key: release[key] for key in ["approval_sha256", "git_revision", "manifest_sha256", "base_model", "asr_model", "selected_variant"]}
provenance["source"] = None
payloads = {"README.md": b"Fixture model card\n", "release-provenance.json": json.dumps(provenance).encode()}
for item in fixture_manifest["files"]:
    payload = payloads[item["output"]]
    item["bytes"] = len(payload)
    item["sha256"] = hashlib.sha256(payload).hexdigest()
release_checks = []
with tempfile.TemporaryDirectory(dir=ARTIFACT, prefix="download-fixture-") as temporary:
    directory = Path(temporary)
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    release_module.verify_release(fixture_manifest, directory, receipt=False)
    release_checks.append({"input": "內容與大小符合完整清單", "accepted": True})
    card = directory / "README.md"
    for name, payload in [("等長內容改動", b"X" + payloads["README.md"][1:]),
                          ("檔案大小改動", payloads["README.md"] + b"X")]:
        card.write_bytes(payload)
        try:
            release_module.verify_release(fixture_manifest, directory, receipt=False)
        except ValueError as error:
            release_checks.append({"input": name, "accepted": False, "error": str(error)})
        else:
            raise AssertionError("不相符的內容不應通過核對")

for relative in ["scripts/score_natural_v4_test.py", "scripts/score_natural_v4_validation.py",
                 "scripts/fetch_natural_release.py", "tiny_perceptron/natural_assistant.py",
                 "docs/natural-assistant/v4/STUDENT.md", "docs/natural-assistant/v4/DATA.md",
                 "docs/natural-assistant/v4/TRAINING.md", "docs/technical-review-guide.md",
                 "docs/course-review-20261019/freeze-01/sources/20.13.md",
                 "docs/course-review-20261019/freeze-04.json"]:
    INPUTS[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-13/verify.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu; standard library only", "unicode": unicodedata.unidata_version},
    "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "input_sha256": INPUTS,
    "fingerprints": fingerprints, "hashlib_input_types": accepted,
    "release_verification_fixture": release_checks,
    "identical_input_full_digest_equal": hashlib.sha256(b"base A, adapter B").digest() == hashlib.sha256(b"base A, adapter B").digest(),
    "group_counts": dict(group_counts), "correct_counts": dict(pass_counts), "all_case_decisions_match_saved_scores": True,
    "complete": sum(row["complete"] for row in computed), "truncated": truncated,
    "subsets": subset_results,
    "photo_scope": {"unique_summary_images": len(photos), "facts_per_image": sorted(set(fact_photos.values())),
                    "qa_types": dict(Counter(row["references"]["qa_type"] for row in sources_by_group["photo_fact"])),
                    "unique_ocr_images_across_tasks": len(ocr_images), "presence_labels": dict(presence)},
    "context_dependent_chat": [{"id": row["id"], "passed": case_decisions["chat:" + row["id"]]["passed"], "history_messages": len(row["history"])} for row in context],
    "chat_history_lengths": dict(Counter(len(row.get("history", [])) for row in sources_by_group["text_chat"])),
    "ocr_order_prompts": [row["user"] for row in sources_by_group["ordered_ocr"]],
    "photo_summary_prompts": sorted({row["user"] for row in sources_by_group["photo_summary"]}),
    "photo_summary_prompt_counts": dict(Counter(row["user"] for row in sources_by_group["photo_summary"])),
    "photo_summary_without_sentence_limit": [{"id": row["id"], "user": row["user"], "system": row.get("system"),
                                               "rubric": row["references"]["rubric"]}
                                              for row in sources_by_group["photo_summary"]
                                              if not any(term in row["user"] for term in ["一至兩句", "一兩句"])],
    "manual_grade_groups": sorted({case["group"] for case in packet["cases"]}),
    "asr_audio_seconds_range": [min(row["duration_seconds"] for row in audio.values()), max(row["duration_seconds"] for row in audio.values())],
    "asr_totals": asr_totals, "asr_records": asr_rows,
    "model_binding": {"core": release["base_model"], "asr": release["asr_model"], "core_parameters": result["total_parameters"],
                      "asr_parameters": result["asr"]["parameters"], "adapters": binding["adapters"],
                      "generation": protocol["generation"], "test_versions": result["versions"]},
    "limitations": "重算使用原始生成、參考答案與既有語意標註；沒有重新訓練、推論、聽錄音或重新替照片評分。人工語意標註的正確性不是本次重算能自動證明的事。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
