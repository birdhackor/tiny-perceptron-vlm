"""Independent bounded CPU audit: metadata, raw records, frozen score arithmetic.

No model loading, training, inference, or dataset downloads.
"""
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import fetch_natural_release as release
from scripts import score_natural_v4_test as scoring


def read(path):
    return json.loads((ROOT / path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(text, whitespace=False):
    text = unicodedata.normalize("NFKC", text).strip()
    return "".join(text.split()) if whitespace else text


def distance(a, b):
    # Independently written standard dynamic-programming edit distance.
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


env = {"python": sys.version, "platform": platform.platform(), "device": "cpu"}
for name in ["torch", "transformers", "peft", "numpy", "pillow", "huggingface-hub"]:
    try:
        env[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        env[name] = "not installed (no model imports)"
(HERE / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
print("ENV", json.dumps(env))

manifest = read("docs/natural-assistant/v4/manifest.json")
public = read("docs/natural-assistant/v4/public-release.json")
release.validate_manifest(public)
assert public["selected_variant"] == "base" and public["adapter_parameters"] == 0
options = release.student_options(public, HERE, device="cpu", local_files_only=True)
assert options.adapter is None and options.dtype == "float32"
assert options.max_new_tokens == 384 and options.max_tokens == 2048
assert options.model_revision == public["base_model"]["revision"]
assert options.asr_revision == public["asr_model"]["revision"]
assert public["requirements_sha256"] == sha(ROOT / "requirements-natural.txt")
for file, expected in public["code_files"].items():
    assert sha(ROOT / file) == expected
bad = dict(public, adapter_parameters=1)
try:
    release.validate_manifest(bad)
except ValueError:
    pass
else:
    raise AssertionError("Base manifest incorrectly accepted an adapter")
for dtype in ["float16", "bfloat16"]:
    try:
        release.student_options(public, HERE, device="cpu", dtype=dtype)
    except ValueError:
        pass
    else:
        raise AssertionError("CPU wrong dtype accepted")
print("DELIVERY", json.dumps({"selected": public["selected_variant"], "adapter": options.adapter,
    "models": [public["base_model"], public["asr_model"]], "runtime": public["runtime"],
    "public_files": [{k:f[k] for k in ["output", "bytes", "sha256"]} for f in public["files"]]}))

for kind, count in [("qwen", 2127532032), ("whisper", 808878080)]:
    meta = json.loads((HERE / "official" / (kind + "-model-api.json")).read_bytes())
    pin = public["base_model" if kind == "qwen" else "asr_model"]
    assert (meta["id"], meta["sha"]) == (pin["repo"], pin["revision"])
    assert meta["safetensors"]["total"] == count
    assert sum(meta["safetensors"]["parameters"].values()) == count
    print("OFFICIAL_PARAMETER_METADATA", kind, meta["sha"], count)

expected, audio = scoring.expected_cases(manifest)
denoms = Counter(c["group"] for c in expected.values())
base = "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/"
generations = read(base + "generations-base.json")
transcripts = read(base + "transcripts.json")
grades = read(base + "combined/grades.json")
saved = read(base + "scored/scores.json")
raw_report = read(base + "result.json")
audit = read("docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/selected-final-test-audit-summary.json")
binding = audit["artifact_binding"]
for file, pointer in [("generations-base.json", "combined_generations_sha256"), ("transcripts.json", "transcripts_sha256"), ("result.json", "test_result_sha256")]:
    assert sha(ROOT / (base + file)) == binding[pointer]
for file, expected_sha in [("docs/natural-assistant/v4/manifest.json", binding["manifest_sha256"]),
                          ("docs/natural-assistant/v4/selection.json", binding["selection_sha256"]),
                          ("docs/natural-assistant/v4/asr-selection.json", binding["asr_selection_sha256"])]:
    assert sha(ROOT / file) == expected_sha
assert raw_report["total_parameters"] == 2127532032 and raw_report["trainable_parameters"] == 0
assert raw_report["asr"]["parameters"] == 808878080
assert set(raw_report["variants"]) == {"base"}
assert raw_report["execution"]["adapters"] == []
assert raw_report["execution"]["selection_sha256"] == binding["selection_sha256"]
assert raw_report["max_tokens"] == 2048 and raw_report["max_pixels"] == 524288
assert raw_report["model_revision"] == public["base_model"]["revision"]
assert raw_report["asr_revision"] == public["asr_model"]["revision"]
assert saved["manual_grades_sha256"] == sha(ROOT / (base + "combined/grades.json"))
assert grades["blind_packet_sha256"] == sha(ROOT / (base + "blind-packet.json"))
assert {g["case_id"] for g in grades["grades"]} == {k for k,c in expected.items() if c["group"] in scoring.MANUAL}
manual = {g["case_id"]: g["passed"] for g in grades["grades"]}
records = {scoring.case_id(r["id"], r["task"]): r for r in generations}
assert len(generations) == len(records) == len(expected) == 178 and records.keys() == expected.keys()
complete, truncated, counts, decisions = 0, [], Counter(), []
for key, c in expected.items():
    r, group = records[key], c["group"]
    ids = r["generated_token_ids"]
    assert len(ids) == r["generated_tokens"] <= 384
    assert all(type(x) is int and x >= 0 for x in ids)
    eos = bool(ids and ids[-1] in r["eos_token_ids"])
    assert eos == scoring.eos_complete(r, 384)
    assert eos == r["ended_with_eos"]
    if eos:
        complete += 1
    else:
        assert len(ids) == 384 and r["truncated"] and r["task"] == "chat"
        truncated.append(key)
    row = c["row"]
    if group in scoring.MANUAL:
        semantic = manual[key]
    else:
        semantic = norm(row["answer"], group == "single_ocr") == norm(r["prediction"], group == "single_ocr")
    passed = bool(eos and semantic)
    counts[group] += passed
    decisions.append({"case_id": key, "group": group, "decoder_complete": eos,
                      "rubric_passed": semantic, "passed": passed})
    if not group.startswith("voice_"):
        assert r["user"] == row["user"] and r["reference_answer"] == row["answer"]
        assert r["image"] == row.get("image") and r["family"] == row["family"]
assert complete == 171 and len(truncated) == 7
assert counts == saved["correct_counts"]
assert decisions == saved["case_decisions"]
assert set(truncated) == set(saved["incomplete_case_ids"])
assert all(saved["denominators"][k] == v for k,v in denoms.items())
print("SCORE_REAGGREGATION", json.dumps({"denominators": denoms, "correct": counts,
      "complete": complete, "truncated": len(truncated), "manual_grade_rows": len(manual)}))

photo_rows = [c["row"] for c in expected.values() if c["group"] in {"photo_summary", "photo_fact"}]
assert len({r["image"] for r in photo_rows}) == 42
assert len(photo_rows) == 126
presence = [c["row"] for c in expected.values() if c["group"] == "text_presence"]
assert Counter(r["answer"] for r in presence) == {"有": 10, "沒有": 8}
assert len({c["row"]["image"] for c in expected.values() if c["group"] == "single_ocr"}) == 10
assert len({c["row"]["image"] for c in expected.values() if c["group"] == "ordered_ocr"}) == 3
subset = read(base + "descriptive-subsets.json")
for qa, numerator, denominator in [("action", 3, 11), ("relation", 21, 29)]:
    keys = [key for key,c in expected.items() if c["group"] == "photo_fact" and c["row"]["references"]["qa_type"] == qa]
    assert len(keys) == denominator
    observed = sum(next(d["passed"] for d in decisions if d["case_id"] == key) for key in keys)
    assert observed == numerator
    assert {x["case_id"] for x in subset["photo_fact_subsets"][qa]["cases"]} == set(keys)
    print("SUBSET", qa, observed, denominator)
context_keys = [key for key,c in expected.items() if c["group"] == "text_chat" and c["row"].get("history")]
assert len(context_keys) == 1 and records[context_keys[0]]["truncated"]
print("NONINDEPENDENCE", "42 images shared by 126 photo questions; 4 recordings shared by both 4-question voice paths")
print("CHAT_CONTEXT", context_keys, "failed; seven capped cases remain among thirteen")

audio_by_id = {r["id"]: r for r in audio}
assert len(transcripts) == len(audio_by_id) == 22
raw_error = raw_chars = normalized_error = normalized_chars = 0
voice_chars = voice_error = 0
for r in transcripts:
    source = audio_by_id[r["id"]]
    assert r["reference_transcript"] == source["user"]
    assert r["audio_sha256"] == source["sha256"]
    assert r["raw_token_ids"] == r["expected_decoder_prompt_ids"] + r["generated_token_ids"]
    assert scoring.asr_eos_complete(r)
    a,b = r["reference_transcript"],r["transcript"]
    re,ne = distance(a,b),distance(norm(a,True),norm(b,True))
    assert re == r["raw_errors"] and ne == r["errors"]
    raw_error += re; raw_chars += len(a)
    normalized_error += ne; normalized_chars += len(norm(a,True))
    if r["task"] == "speech_chat":
        voice_error += re; voice_chars += len(a)
        for task,user in [("typed_chat", a), ("speech_chat", b)]:
            gen = records[scoring.case_id(r["id"],task)]
            assert gen["user"] == user
assert (raw_error,raw_chars,normalized_error,normalized_chars) == (112,674,96,661)
assert (voice_error,voice_chars) == (0,42)
assert Counter(r["task"] for r in transcripts) == {"speech_transcription":18,"speech_chat":4}
print("ASR_RECOMPUTED", raw_error, raw_chars, f"{100*raw_error/raw_chars:.2f}%",
      normalized_error, normalized_chars, f"{100*normalized_error/normalized_chars:.2f}%", "AISHELL",voice_error,voice_chars)

selection = read("docs/natural-assistant/v4/asr-selection.json")
asr_dir = "docs/natural-assistant/evidence/v4-research/asr-cpu-validation/"
small,turbo = [read(asr_dir+v+"-records.json") for v in ["small","turbo"]]
assert len(small) == len(turbo) == 16
assert [r["id"] for r in small] == [r["id"] for r in turbo]
for a,b in zip(small,turbo):
    assert (a["audio_sha256"],a["reference_transcript"],a["split"]) == (b["audio_sha256"],b["reference_transcript"],"validation")
    assert a["id"] not in audio_by_id
for variant,rows in [("small",small),("turbo",turbo)]:
    re = sum(distance(r["reference_transcript"],r["transcript"]) for r in rows)
    rc = sum(len(r["reference_transcript"]) for r in rows)
    ne = sum(distance(norm(r["reference_transcript"],True),norm(r["transcript"],True)) for r in rows)
    nc = sum(len(norm(r["reference_transcript"],True)) for r in rows)
    assert abs(re/rc - selection["raw_CER"][variant]) < 1e-15
    assert abs(ne/nc - selection["normalized_CER"][variant]) < 1e-15
    print("ASR_SELECTION_RECOMPUTED",variant,"validation",len(rows),"raw",re,rc,"normalized",ne,nc)
assert selection["raw_CER"]["turbo"] < selection["raw_CER"]["small"]
assert selection["normalized_CER"]["turbo"] < selection["normalized_CER"]["small"]

assert norm("Ａ 你\n好！", True) == "A你好!"
assert norm("繁體",True) != norm("繁体",True)
assert norm("上\n下") != norm("上下")
assert norm("上 下",True) == norm("上下",True)
print("NORMALIZATION_VARIANTS", "NFKC punctuation changes disclosed; whitespace ignored only on single-line/CER; script and explicit order preserved")

fence = (HERE / "original-fence/fence-1.py").read_bytes()
exec(compile(fence,"original-20.13-fence","exec"),{})
for a,b,equal in [(b"base A, adapter B",b"base A, adapter B",True),
                  (b"base A, adapter B",b"base A, adapter C",False),
                  (b"base A, adapter B",b"base A, adapter B\n",False)]:
    assert (hashlib.sha256(a).digest() == hashlib.sha256(b).digest()) == equal
print("HASH_VARIANTS", "same bytes=True; different adapter=False; extra newline=False; digest length=32 bytes")
print("ALL_BOUNDED_ASSERTIONS_PASSED")
