"""Write this reviewer's new report; never read the previous report."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PREFIX = HERE.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_19_3"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def artifact(identifier, name, kind, description, **extra):
    return {"id": identifier, "path": PREFIX + "/" + name, "sha256": sha(HERE / name),
            "kind": kind, "description": description, **extra}

records = json.loads((HERE / "execution-record.json").read_text())
env = {"python": "3.13.5", "torch": "2.14.1+cpu", "pytest": "9.1.1", "device": "cpu",
       "cuda_build": "None", "threads": "1", "network_for_execution": "offline"}
artifacts = [
    artifact("section", "section-original.md", "source_snapshot", "19.3 raw UTF-8 bytes originally read; no newline normalization"),
    artifact("read-scope", "read-scope.json", "source_snapshot", "Actual section range and raw-byte fingerprint; no whole-chapter version claimed"),
    artifact("inspection", "inspection.json", "source_snapshot", "Independent AST ranges, original-source locators, raw JSON pointers, scope and honest tool-failure records"),
    artifact("acquisition", "source-acquisition.json", "source_snapshot", "Fetched original HTTPS URLs, exact versions, byte counts, access date and hashes"),
    artifact("fence", "original-fence.py", "code", "Unmodified original Python fence executed by this reviewer"),
    artifact("bounded-code", "verify_19_3.py", "code", "Bounded independent data reconstruction, fingerprints, original fence, baselines, seed variants and negative controls"),
    artifact("report-writer", "write_report.py", "code", "This reviewer's complete new report writer; does not read the old report"),
    artifact("commands", "execution-record.json", "execution", "Actual subprocess commands, CPU/offline environment, timeout, timing and exit codes",
             command="; ".join(" ".join(row["argv"]) for row in records),
             result="Independent verification exit 0; selected original tests exit 0", environment=env),
    artifact("cpu-result", "verification-stdout.txt", "execution", "Actual original-fence stdout and independently computed data-integrity results",
             command=" ".join(records[0]["argv"]), result="ALL_ASSERTIONS_PASSED; no model scores", environment=env),
    artifact("tests-result", "existing-tests-stdout.txt", "execution", "Three selected original test functions including five audio seed cases",
             command=" ".join(records[1]["argv"]), result="7 passed in 1.81s", environment=env),
    artifact("cpu-stderr", "verification-stderr.txt", "source_snapshot", "Empty stderr of final successful verification"),
    artifact("tests-stderr", "existing-tests-stderr.txt", "source_snapshot", "Empty stderr of successful original tests"),
    artifact("derivation", "counterexample.md", "derivation", "Exact arithmetic baseline and score-change counterexample, independently derived"),
]
source_files = {
    "sklearn-cv": ("sklearn-cross-validation-1.7.2.rst", "official_docs", "scikit-learn grouped evaluation and train/validation/test rules", "1.7.2"),
    "sklearn-leakage": ("sklearn-common-pitfalls-1.7.2.rst", "official_docs", "scikit-learn data leakage rules", "1.7.2"),
    "minds-schema": ("minds14-readme-40ce77c.md", "official_docs", "PolyAI MInDS-14 published dataset schema", "40ce77cb32a384e4d50a568e1ec39ac804019d33"),
    "python-counter": ("cpython-collections-3.13.5.rst", "official_docs", "CPython Counter API", "v3.13.5"),
    "python-set": ("cpython-stdtypes-3.13.5.rst", "official_docs", "CPython set-comprehension and intersection API", "v3.13.5"),
    "frozen-data": ("data-1df3353.json", "repository_code", "Frozen capstone original rows and generation manifest", "1df335318bda03fd771807f66976953231d5a00b; capstone-small-world-v2, seed 42"),
    "dataset-code": ("current-capstone.py", "repository_code", "Original current capstone dataset and actual input construction", "Current raw SHA-256 d02d1cd86cb4f6bbfbc1718c140107a5d6d3a128a9e20e183d4a9cd38e45edbe"),
    "input-tests": ("capstone-test-1df3353.py", "repository_code", "Original actual-input/family integrity tests", "1df335318bda03fd771807f66976953231d5a00b; current file byte-identical"),
    "modality-code": ("current-multimodal.py", "repository_code", "Original synthetic scene, waveform and acoustic feature construction", "Current original bytes captured 2026-10-05; full SHA-256 identifies version"),
    "tokenizer-code": ("current-data.py", "repository_code", "Original UTF-8 byte tokenizer contract", "Current original bytes captured 2026-10-05; full SHA-256 identifies version"),
}
acquisitions = {row["name"]: row for row in json.loads((HERE / "source-acquisition.json").read_text())}
inspection = json.loads((HERE / "inspection.json").read_text())
original_notes = {
    "sklearn-cv": "Read raw 1.7.2 RST lines 10-21, 60-82 and 627-654: validation for choices, final test held out, dependent groups absent from paired training folds.",
    "sklearn-leakage": "Read raw 1.7.2 RST lines 77-118: information unavailable at prediction, no fit or model choices using test, split before preprocessing.",
    "minds-schema": "Read fixed-commit original YAML schema lines 112-154, dataset description 915-938 and Data Instances/Data Fields 970-1002; independently enumerate all 15 config feature blocks; six fields, no speaker/session field; example path contains ADDRESS.",
    "python-counter": "Read official v3.13.5 Counter original RST lines 243-272: counts of iterable items, dictionary values, missing-key zero. Matches installed Python 3.13.5.",
    "python-set": "Read official v3.13.5 stdtypes original RST lines 4438-4455 and 4497-4508: comprehension makes a set; & returns common members. Matches installed Python 3.13.5.",
    "frozen-data": "First inspected top-level keys/types, then /manifest/version, /seed, /split_unit, /sha256, /counts, /families, /task_majority_baselines, /task_label_counts, /split_policy, /label_definition, /majority_baseline_scope, /limitations and /splits/{train,validation,test}; these are raw rows and method/baseline contracts. Fetched commit file hash equals original local file hash. No old review or author result-correction fields read.",
    "dataset-code": "AST-selected and read DATA_VERSION, digest, _row, build_dataset, modality_tensors, prompt_ids, encode_record, prepare_batch; original lines 1-35 and 124-338. Full untouched file preserved, unnecessary model/training/inference methods not read.",
    "input-tests": "AST-selected actual_input_sha and three selected data-integrity tests; read lines 1-98. Fetched original commit file and current file SHA both 8446cf66b1a82de26696cd1d9ab2bf8dcf86b1a3a11979809a4bc41bc5f9b918. Executed only three selected functions, not full-model tests.",
    "modality-code": "AST-selected tone 44-46, mel_filter_bank 49-59, log_mel 62-69 and scene 177-190; read 1-75, 177-190. Inspected 3x16x16 image generation and 16-band frame-mean acoustic summary construction.",
    "tokenizer-code": "AST-selected ByteTokenizer and encode; read lines 1-30. UTF-8 bytes map to IDs by adding eight specials; actual prompt_ids include system/user and modality markers.",
}
sources = []
for identifier, (name, kind, title, version) in source_files.items():
    artid = "snapshot-" + identifier
    artifacts.append(artifact(artid, "sources/" + name, "source_snapshot", title + " — byte-exact original snapshot"))
    row = {"id": identifier, "kind": kind, "title": title, "version": version,
           "verified": True, "inspection_note": original_notes[identifier], "artifact_id": artid}
    if kind == "official_docs":
        row.update(url=acquisitions[name]["url"], accessed_on="2026-10-05", checked_original=True,
                   authority_reason=("Producer PolyAI publishes this fixed-version dataset card and schema in its own dataset repository." if identifier == "minds-schema"
                                     else "Version-tagged raw documentation in the project's official upstream source repository."))
    else:
        row.update(path=PREFIX + "/sources/" + name, sha256=sha(HERE / "sources" / name))
    sources.append(row)
for stage in ["pretrain", "sft", "joint", "dpo"]:
    artifacts.append(artifact("manifest-" + stage, "sources/" + stage + "-data-manifest.json", "source_snapshot",
                              "Original " + stage + " /version, /seed, /counts and /sha256 provenance; model scores not inspected"))
artifacts.append(artifact("student-provenance", "sources/student-report-original.json", "source_snapshot",
                          "Full untouched original retained; only /data_manifest/version and /data_manifest/sha256 inspected"))
sources += [
    {"id": "cpu", "kind": "execution", "title": "Independent bounded CPU data verification", "verified": True, "artifact_id": "cpu-result"},
    {"id": "selected-tests", "kind": "execution", "title": "Selected original integrity tests on CPU", "verified": True, "artifact_id": "tests-result"},
    {"id": "baseline-logic", "kind": "derivation", "title": "Constant payloads, denominator and current-truth counterexample", "verified": True,
     "details": "See counterexample.md: high correct on 3/6 labels; green/square each 9/9; unchanged constant high output scores 0/1 on low then 1/1 on high without examining audio; exact integer arithmetic."},
]

def ev(source_id, locator, supports):
    return {"source_id": source_id, "locator": locator, "supports": supports}

def verified(identifier, kind, statement, location, scope, evidence, artifact_ids, expected=None, observed=None, details=None, denominators=None):
    claim = {"id": identifier, "kind": kind, "statement": statement, "location": location, "scope": scope,
             "status": "verified", "evidence": evidence, "artifact_ids": artifact_ids}
    if kind in {"numeric", "software", "empirical"}:
        claim["verification"] = {"method": "executed", "expected": expected, "observed": observed, "details": details}
        if kind == "numeric":
            claim["verification"]["tolerance"] = "Exact integer/string equality; no floating tolerance needed. Baseline fractions 3/6=0.5 and 9/9=1 exactly."
        if denominators:
            claim["verification"]["denominators"] = denominators
    return claim

claims = [
    verified("family-separation", "concept",
             "Group original material and its dependent descendants before train/validation/final-test splitting; original/crop, paraphrases and paired swapped scenes must stay in the same family.",
             "19.3 paragraphs 1-2 and natural-product-data paragraph after the Python example",
             "A methodological plan for natural text/images/OCR/recordings and an implemented synthetic family example. Grouping blocks this defined dependence; it does not guarantee every semantic or template similarity is absent, nor accept an unbuilt product.",
             [ev("sklearn-cv", "RST 627-654, grouped data and GroupKFold", "Dependent samples must be grouped and held-out groups absent from training; domain-specific IDs define the group."),
              ev("sklearn-leakage", "RST 104-118", "Split before preprocessing rather than mixing derived inputs across roles.")], ["inspection"]),
    verified("split-roles", "concept",
             "Training updates parameters, validation selects settings, and final test is reserved for final evaluation; reused final questions cannot be described as new material.",
             "19.3 paragraph 2",
             "Evaluation protocol, not proof that future product engineering or an unseen-phrasing test has been completed.",
             [ev("sklearn-cv", "RST 60-82", "Distinguishes training, hyperparameter validation and held-out final testing, including test-set selection leakage."),
              ev("sklearn-leakage", "RST 80-89", "Test data must not be used to make model choices.")], ["inspection"]),
    verified("python-fence", "software",
             "The unmodified Python example builds seed-42 data, uses set intersections for family overlap, and Counter/manifest payload counts for task, pitch and constant-answer baselines.",
             "19.3 original Python fence (source lines 117-137)",
             "Dataset construction and counting only. No parameter update, model inference or scored model prediction is executed; ordinary Python APIs are grouped in this claim.",
             [ev("python-counter", "Counter original RST 243-272", "Iterable category strings are counted as dictionary keys and count values."),
              ev("python-set", "stdtypes original RST 4438-4455, 4500-4505", "Set comprehension deduplicates family strings and & returns common members."),
              ev("dataset-code", "build_dataset 139-282; original fence snapshot", "Defines the seed, split family strings and constant payload manifest accessed by the example."),
              ev("cpu", "stdout ORIGINAL_FENCE_BEGIN..END", "Actual original example executes without modification and prints the independently checked values.")],
             ["fence", "bounded-code", "cpu-result"], "Original fence executes on CPU with three zero family intersections and documented counters.",
             "Original fence completed; outputs 0/0/0 intersections, 552/84/90 task totals, pitch high3/low3, high3/6, green9/9, square9/9.",
             "exec(compile(original bytes)) without bootstrap edits; negative controls detect copied family/input; no model object loaded."),
    verified("dataset-counts", "numeric",
             "capstone-small-world-v2 seed42 contains 552 training, 84 validation and 90 final-test records, totaling 726; task descendants share numeric, modality, audio and context-family split membership.",
             "19.3 details paragraph 1 and original-fence task counts",
             "Specified synthetic dataset version only; these are records/questions, not samples of natural-product ability or model performance.",
             [ev("frozen-data", "/manifest/version, /seed, /counts, /families; /splits/{train,validation,test}", "Exact original generated rows and provenance."),
              ev("dataset-code", "build_dataset 139-247", "Six rows per numeric family, held modality pairs, three pure-tone variants per audio family, four context descendants, deterministic split then deduplication."),
              ev("cpu", "stdout counts, train_task_counts, validation_task_counts, test_task_counts", "Reconstruction equals every frozen row and manifest; row IDs are recomputed.")],
             ["snapshot-frozen-data", "cpu-result"], "552+84+90=726; generated and frozen rows/manifest equal.",
             "Exact counts train552/validation84/test90 and total726; full row and manifest equality passed.",
             "Sum record lengths, count task categories, recompute every record ID and assert all generated rows match the pinned JSON.", {"train_records": 552, "validation_records": 84, "test_records": 90, "total_records": 726}),
    verified("audio-baselines", "numeric",
             "Pure-tone low/high each have 8/1/1 families across train/validation/test, three frequency variants per family; final audio has low3/high3 and a constant-high baseline 3/6. Joint final data has 18 same-question rows, low9/high9.",
             "19.3 paragraph after fence and details paragraphs 3-4",
             "Synthetic tone labels and arithmetic payload baselines only; no speech recognition or trained-model score. Family counts and question counts are different denominators.",
             [ev("dataset-code", "build_dataset 159-194, 216-247; modality_tensors 285-296", "Audio family generation and deterministic balance swaps; 440/880 Hz bases with family offsets and three frequency variants; joint questions identical."),
              ev("frozen-data", "/splits/test filtered task audio/joint; /manifest/families/*/audio; /manifest/task_label_counts", "Raw pitch generation truth and per-task labels."),
              ev("cpu", "stdout audio_summary, joint_test, small_seed_variants", "Counts recomputed for original seed and bounded alternative seeds, including exact per-family three-row checks.")],
             ["cpu-result", "tests-result", "derivation"], "Audio low/high families8/1/1 with three descendants, test3/6 fixed high, joint low9/high9 of18.",
             "Train audio48 (24 each), validation6/test6 (3 each); per-class families8/1/1; test fixed high3/6; joint18 with9 each and one question.",
             "Counter over actual row pitch labels, distinct family names and joint question strings; manifest majority baseline is generated truth, not model prediction.",
             {"pure_audio_train": 48, "pure_audio_validation": 6, "pure_audio_test": 6, "joint_test": 18}),
    verified("image-baselines", "numeric",
             "Blue-circle is reserved for validation and green-square for final test; training still contains every basic color/shape. All final images are green squares, so constant green and square payloads each score 9/9 on their respective tasks.",
             "19.3 fence discussion, details paragraphs 1 and 5",
             "These are held-out color/shape combinations within a synthetic generator. They are neither new colors/shapes nor evidence that a trained model looked at images.",
             [ev("dataset-code", "build_dataset 159-182, 237-242; modality_tensors 285-290", "Visual families group color/shape and all positions/brightness/joint descendants, then deduplicate repeated records."),
              ev("modality-code", "scene 177-190", "Actual RGB generator uses the recorded color/shape truth and spatial offset."),
              ev("frozen-data", "/splits/test image_color/image_shape; /manifest/task_majority_baselines/test", "Original input specs and exact task baselines."),
              ev("cpu", "stdout image_family_summary, image_color_fixed_baseline, image_shape_fixed_baseline", "Recomputed all class/family coverage and each constant answer's numerator/denominator.")],
             ["cpu-result", "derivation"], "Train all3 colors and both2 shapes; validation blue-circle/test green-square; constants9/9 each.",
             "Train colors blue/green/red and circle/square; reserved pair membership exact; constant green9/9 and square9/9.",
             "Inspect every image spec and compare its expected answer payload, preserving task-specific denominators9 rather than pooling image/joint rows.",
             {"test_image_color": 9, "test_image_shape": 9}),
    verified("reserved-numbers-and-templates", "numeric",
             "The numbers1:2 family has six descendants only in final test, including both ordered calculator questions, unavailable variants, concept and tool-return; calculator questions use the same two templates found in training.",
             "19.3 details paragraphs 2 and 6",
             "This is a held-out operand family and genuine reserved example, with previously seen templates. No unfamiliar-phrasing evaluation or general Chinese comprehension is established.",
             [ev("dataset-code", "build_dataset 146-158 and 210-217", "Canonical a<=b numeric family, both orientations and deterministic pre-score reservation of1:2."),
              ev("frozen-data", "/splits/test filtered family numbers:1:2; /splits/* filtered calculator task", "The six raw questions/answers and actual templates in each split."),
              ev("cpu", "stdout held_1_2, calculator_templates; selected original held-out-family test", "Exact six records and normalized two question templates independently verified.")],
             ["cpu-result", "tests-result"], "Six1:2 rows in test and none in train/validation; calculator template sets equal.",
             "Six test rows (calculator2/unavailable2/concept1/tool_return1); two identical templates across all three splits.",
             "Filter actual rows by canonical family; replace only numeric substrings to compare calculator phrasing without changing words.", {"reserved_family_records": 6}),
    verified("frozen-fingerprints", "software",
             "The pinned data.json stores all726 rows and generation specs; recomputed train/validation/test SHA prefixes are e0a419727952/afedd4dc84ce/aef4b7a876ff, preserved per split across stages and recorded student comparisons, while the three distinct roles have different fingerprints.",
             "19.3 details paragraph 7",
             "Raw data version and provenance consistency only; does not establish trained-model performance or validate any future pipeline. SHA is over JSON rows via the original ensure_ascii=False, sort_keys=True digest contract.",
             [ev("frozen-data", "pinned commit file full SHA053f4e...; /manifest/sha256; /splits/*; /modality_generation", "Downloaded designated original is byte-identical to local version and contains rows plus specs."),
              ev("dataset-code", "digest30-31; manifest270-277", "Defines serialized row fingerprint calculation."),
              ev("cpu", "stdout computed_split_sha256, frozen_stage_provenance", "Independently recomputed full hashes and checked four stage manifest /sha256 pointers plus student /data_manifest/sha256.")],
             ["cpu-result", "snapshot-frozen-data", "manifest-pretrain", "manifest-sft", "manifest-joint", "manifest-dpo", "student-provenance"],
             "Full hashes match pinned manifest and stated prefixes; each stage/comparison preserves the corresponding split hash; all three role hashes differ.",
             "train e0a419727952eceb700e5bc8f47bbc7d7c961ba4e14acca3f68b73c360e20cf9; validation afedd4dc84ce22a8867f7ebc5cc7804c7403d01d1cf29a68cf04985c2ff208a9; test aef4b7a876ff8966c0a7413d342017fd69038855ac8f55c28b7c66edb8808ecd; all provenance comparisons pass.",
             "Rehash raw ordered lists and verify exact byte equivalence of pinned/local JSON; read only named raw provenance pointers, not model-result summaries."),
    verified("actual-input-nonoverlap", "software",
             "The original CPU integrity test fingerprints actual prompt IDs and generated image/acoustic feature bytes; all three cross-split family and exact-input intersections are zero, without establishing semantic/template novelty or model ability.",
             "19.3 details final paragraph",
             "Equality of the chosen input representation under the specified current generator. Exact duplicate exclusion is narrower than independence, unseen semantics, unseen wording or audio/vision reliance.",
             [ev("input-tests", "actual_input_sha45-50; test_family_split_and_actual_prompt_modal_content_do_not_leak53-62", "Hashes actual prompt IDs and feature numpy bytes, then checks all three split pairs."),
              ev("dataset-code", "modality_tensors285-296; prompt_ids299-305; prepare_batch321-338", "Actual model-input representation comes from the stored specs and system/user/modality IDs, excluding answer targets."),
              ev("tokenizer-code", "ByteTokenizer.encode20-21", "Prompt UTF-8 byte IDs used by the test."),
              ev("cpu", "stdout cross_split_overlaps and negative_controls", "Independent reimplementation yields zero originals and one overlap after inserting a copied train row into validation sets."),
              ev("selected-tests", "tests-result stdout", "Selected original test functions execute:7 passed.")],
             ["cpu-result", "tests-result", "bounded-code"], "All pairs show family0/input0; copied-row negative control shows1/1; original tests pass.",
             "train/validation0/0, train/test0/0, validation/test0/0; copied train family/input each1;7 selected test cases passed.",
             "Hash numerical prompt list plus RGB tensor bytes and16-band log-mel frame-mean bytes; independently recompute all726 rows; no model loading/inference."),
    verified("minds-and-sidechannel-scope", "concept",
             "MInDS-14's published schema has no speaker ID, so unique source rows do not establish a new-speaker evaluation; speaker/session grouping requires those identifiers. Transcripts, intent labels and label-bearing filenames must be excluded from the task input to avoid answer side channels.",
             "19.3 natural-product-data paragraph after fence",
             "Verified against the fixed PolyAI dataset-card version only. This does not prove absence of speakers in original recordings, prohibit audio models from predicting intent, or claim future product input filtering is implemented.",
             [ev("minds-schema", "fixed-commit YAML features112-154 and Data Fields990-1002; example path976", "Only path/audio/transcription/english_transcription/intent_class/lang_id; no speaker/session. Example ADDRESS path itself encodes intent."),
              ev("sklearn-cv", "RST627-654", "Unseen-subject evaluation needs group IDs and all rows of the subject held together."),
              ev("sklearn-leakage", "RST80-89", "Information unavailable in legitimate prediction can lead to optimistic test estimates.")],
             ["snapshot-minds-schema", "cpu-result", "inspection"]),
    {"id": "score-change-counterexample", "kind": "numeric", "status": "verified",
     "statement": "A score change alone cannot prove correct audio use; paired checks must hold image/question fixed, change audio, update the pitch truth and check each answer against it.",
     "location": "19.3 details paragraph4 and image-pair check recommendation paragraph5",
     "scope": "Necessary evaluation logic and a future acceptance recommendation; no paired-model test is claimed performed by this review or by this section's simple counting fence.",
     "evidence": [ev("baseline-logic", "counterexample.md first paragraph", "Constant high output goes from0/1 on low to1/1 on high while never inspecting the waveform; a score change therefore does not imply audio recognition."),
                  ev("frozen-data", "/splits/test filtered joint task", "There are18 joint rows with one unchanged question, supporting the stated sample scope; raw specs define changed pitch truth.")],
     "artifact_ids": ["derivation", "cpu-result"],
     "verification": {"method": "hand_calculation", "expected": "Unchanged constant high output can change correctness after the truth changes from low to high.",
                      "observed": "0/1 before and1/1 after, while the output function is independent of audio; synthetic constants also explain3/6 and9/9 baselines.",
                      "details": "Compare the constant payload high to current per-question low/high truth; numerator changes, denominator remains one question. This is a counterexample, not measured model inference.",
                      "tolerance": "Exact Boolean/integer arithmetic."}},
]

body = (HERE / "section-original.md").read_bytes()
raw = (ROOT / "course/chapters/19.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
index = next(i for i, header in enumerate(headers) if header[0].startswith(b"## 19.3 "))
current = raw[headers[index].start():headers[index + 1].start() if index + 1 < len(headers) else len(raw)]
assert current == body, "Section changed; this reviewer must independently inspect a new version before pass."
assert TASK == "/root/phase4_factual_coordinator/factual_19_3"
report = {"schema_version": 1, "review_stage": "technical", "lesson_id": "19.3",
          "source": "course/chapters/19.md#19.3", "source_sha256": hashlib.sha256(body).hexdigest(),
          "reviewer_task": TASK, "reviewer_context": "fresh", "reviewed_on": "2026-10-05",
          "verdict": "pass", "figure_sha256": {}, "sources": sources, "artifacts": artifacts,
          "claims": claims, "issues": [],
          "checks": {
              "factual_accuracy": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "Independently checked grouping/holdout concepts against official exact-version docs, schema against PolyAI original, and every material raw-data/CPU claim with separate scope. No prior reports or author result-review summaries used."},
              "numeric_verification": {"status": "pass", "claim_ids": [c["id"] for c in claims if c["kind"] == "numeric"], "details": "Exact record/family/task counts, baseline numerators and question denominators independently recomputed; six reserved-number rows and same two templates checked; all numerical values are data facts/arithmetic, not trained-model scores."},
              "figure_consistency": {"status": "not_applicable", "claim_ids": [], "details": "No image/SVG reference in19.3 raw bytes. This section uses sets, integer counts and named material families without requiring a spatial diagram; no render/view was claimed."},
              "source_verification": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "Own fetched original scikit-learn1.7.2 and CPython3.13.5 raw docs, fixed PolyAI commit card, pinned1df3353 raw data/test files; inspected exact code methods and named JSON pointers; permanent byte-exact snapshots and hashes recorded."},
              "limitations": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "No GPU/training/model downloads/model score re-evaluation. Zero overlap means exact representation equality exclusion, with shared templates and image constant-answer shortcuts still present. New natural-data family policies and paired tests remain product plans; unseen-speaker/phrasing or general Chinese capability is not accepted."},
          },
          "read_scope": {"section_lines": [108, 161], "section_snapshot_artifact_id": "section", "inspection_artifact_id": "inspection", "chapter_introduction_read": False,
                         "whole_chapter_read_or_hash_claimed": False, "frozen_input_scope": "section raw UTF-8 bytes only"}}
(ROOT / "docs/technical-reviews/19.3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"reviewer_task": TASK, "source_sha256": report["source_sha256"], "verdict": report["verdict"],
                  "claims": len(claims), "report_sha256": sha(ROOT / "docs/technical-reviews/19.3.json")}, ensure_ascii=False))
