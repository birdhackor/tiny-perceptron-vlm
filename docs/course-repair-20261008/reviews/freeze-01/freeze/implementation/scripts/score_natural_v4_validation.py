"""Export blind semantic review inputs, then select from exact validation counts.

Reads existing artifacts only. Never generates answers or changes frozen gold.
Give graders only blind-packet.json and grades-template.json, not private-map.json.
"""

import argparse
import hashlib
import json
import re
import secrets
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_perceptron import natural_assistant as core  # noqa: E402

WEIGHTS = {
    "photo_summary": 270,
    "photo_fact": 135,
    "text_presence": 840,
    "single_ocr": 1512,
    "ordered_ocr": 5040,
    "text_chat": 560,
    "voice_typed_reference_chat": 1260,
    "voice_actual_asr_chat": 1260,
}
DENOMINATOR = 75600
MANUAL = {"photo_summary", "photo_fact", "text_chat", "voice_typed_reference_chat", "voice_actual_asr_chat"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Repeated JSON object key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)


def case_id(identifier, task):
    return task + ":" + identifier


def mapping_commitment(aliases, nonce):
    return hashlib.sha256(
        json.dumps({"aliases": aliases, "nonce": nonce}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def expected_cases(manifest, protocol):
    cases = {}
    visual = [row for row in manifest["rows"] if row["split"] == "validation"]
    audio = [row for row in manifest["audio_rows"] if row["split"] == "validation"]
    for row in visual:
        task = row["task"]
        if task == "scene":
            summary = row["references"].get("qa_type") == "scene"
            require(
                row["id"].endswith("/scene") if summary else row["id"].endswith(("/fact1", "/fact2")),
                "Unknown photo QA category",
            )
            group = "photo_summary" if summary else "photo_fact"
        else:
            require(task in {"chat", "text_presence", "ocr", "ocr_order"}, "Unknown validation task")
            group = {
                "chat": "text_chat",
                "ocr": "single_ocr",
                "ocr_order": "ordered_ocr",
                "text_presence": "text_presence",
            }[task]
        cases[case_id(row["id"], task)] = {"row": row, "task": task, "group": group}
    for row in audio:
        require(row["task"] in {"speech_chat", "speech_transcription"}, "Unexpected v4 audio task")
        if row["task"] == "speech_chat":
            for task, group in (("typed_chat", "voice_typed_reference_chat"), ("speech_chat", "voice_actual_asr_chat")):
                cases[case_id(row["id"], task)] = {"row": row, "task": task, "group": group}
    counts = Counter(case["group"] for case in cases.values())
    declared = protocol["validation_denominators"]
    require(
        all(counts[group] == declared[group] > 0 for group in WEIGHTS), "Frozen validation group denominators changed"
    )
    require(
        len(cases) == declared["lm_generations_per_candidate"] and len(audio) == declared["asr_recordings"],
        "Frozen LM/ASR denominators changed",
    )
    require(
        sum(counts[group] * weight for group, weight in WEIGHTS.items()) == DENOMINATOR,
        "Integer macro weights do not normalize to one",
    )
    return cases, audio


def eos_complete(record, cap):
    tokens, eos = record.get("generated_token_ids"), record.get("eos_token_ids")
    return bool(
        isinstance(tokens, list)
        and tokens
        and all(type(token) is int for token in tokens)
        and len(tokens) <= cap
        and isinstance(eos, list)
        and eos
        and all(type(token) is int for token in eos)
        and tokens[-1] in eos
        and record.get("generated_tokens") == len(tokens)
        and record.get("ended_with_eos") is True
        and record.get("stop_reason") == "eos"
        and record.get("truncated") is False
        and record.get("completion_unknown") is False
    )


def load_artifacts(manifest_path, protocol_path, validation_dir, data_root):
    manifest, root = core.load_manifest(manifest_path, data_root)
    protocol = read_json(protocol_path)
    cases, audio = expected_cases(manifest, protocol)
    candidates = protocol["candidate_order"]
    require(
        candidates == ["base", "adapter-step-001039", "adapter-step-002077"], "Unexpected predeclared v4 candidates"
    )
    directory = Path(validation_dir)
    report_path = directory / "result.json"
    report = read_json(report_path)
    execution = report.get("execution", {})
    manifest_sha = core.sha256(manifest_path)
    require(
        report.get("status") == "completed"
        and report.get("split") == "validation"
        and execution.get("stage") == "validation",
        "Needs the actual completed validation stage",
    )
    require(
        report.get("manifest_sha256") == manifest_sha == execution.get("manifest_sha256"),
        "Validation belongs to a different frozen manifest",
    )
    require(
        report.get("model") == core.MODEL_ID and report.get("model_revision") == core.MODEL_REVISION,
        "Unexpected foundation model pin",
    )
    for field in ("min_pixels", "max_pixels"):
        require(report.get(field) == protocol["generation"][field], "Validation pixel settings differ from protocol")
    require(
        report.get("max_tokens") == 2048 and report.get("seed") == protocol["training"]["seed"],
        "Validation token/seed settings differ from protocol",
    )
    selection_path = Path(protocol["asr"]["selection_file"])
    if not selection_path.is_absolute():
        selection_path = ROOT / selection_path
    require(core.sha256(selection_path) == protocol["asr"]["selection_sha256"], "ASR selection bytes changed")
    asr = read_json(selection_path)
    require(
        (report.get("asr_model"), report.get("asr_revision"))
        == core.ASR_VARIANTS[protocol["asr"]["variant"]]
        == (asr["model"], asr["revision"]),
        "ASR pin differs from frozen choice",
    )
    require(
        execution.get("runtime_options", {}).get("asr_variant") == protocol["asr"]["variant"],
        "Actual ASR runtime choice differs",
    )
    require(set(report.get("variants", {})) == set(candidates), "Validation must contain all predeclared candidates")
    require(
        report.get("requested_visual_text_rows")
        == len([row for row in manifest["rows"] if row["split"] == "validation"])
        and report.get("requested_audio_rows") == len(audio)
        and report.get("requested_audio_chat_rows") == len([row for row in audio if row["task"] == "speech_chat"]),
        "Actual validation request counts differ from frozen data",
    )
    transcripts_path = directory / "transcripts.json"
    transcripts = read_json(transcripts_path)
    transcript_index = {}
    source_audio = {row["id"]: row for row in audio}
    for record in transcripts:
        identifier = record.get("id")
        require(
            identifier in source_audio and identifier not in transcript_index, "Unknown/repeated ASR validation row"
        )
        row = source_audio[identifier]
        require(
            record.get("reference_transcript") == row["user"] and record.get("task") == row["task"],
            "ASR source transcript/task changed",
        )
        require(record.get("audio_sha256") == manifest["asset_sha256"][row["audio"]], "ASR used a different recording")
        require(isinstance(record.get("transcript"), str), "Missing actual ASR hypothesis")
        transcript_index[identifier] = record
    require(
        set(transcript_index) == set(source_audio) and report.get("asr", {}).get("completed") is True,
        "ASR validation is incomplete",
    )
    adapters = execution.get("adapters", [])
    adapter_index = {item["variant"]: item for item in adapters}
    require(
        len(adapter_index) == len(adapters) == 2 and set(adapter_index) == set(candidates[1:]),
        "Missing/repeated archived adapter descriptors",
    )
    for variant in candidates[1:]:
        item = adapter_index[variant]
        require(item.get("checkpoint") == variant.removeprefix("adapter-"), "Adapter checkpoint/variant differs")
        require(
            re.fullmatch(r"[a-f0-9]{64}", str(item.get("adapter_sha256", ""))) is not None
            and item.get("adapter_run_id") == execution.get("adapter_run_id")
            and bool(item.get("adapter_run_id")),
            "Invalid adapter train-run/weight binding",
        )
    records, record_files = {}, []
    for variant in candidates:
        summary = report["variants"][variant]
        require(
            summary.get("completed") is True and summary.get("generation_count") == len(cases),
            "Candidate generation count is incomplete",
        )
        path = directory / ("generations-" + variant + ".json")
        index = {}
        for record in read_json(path):
            require(
                isinstance(record.get("id"), str) and isinstance(record.get("task"), str),
                "Missing generation row identity",
            )
            key = case_id(record["id"], record["task"])
            require(key in cases and key not in index, "Unknown/repeated generation row")
            case = cases[key]
            row = case["row"]
            require(
                record.get("variant") == variant
                and record.get("split") == "validation"
                and isinstance(record.get("prediction"), str),
                "Generation variant/split/raw answer changed",
            )
            user = (
                transcript_index[row["id"]]["transcript"] if case["group"] == "voice_actual_asr_chat" else row["user"]
            )
            require(record.get("user") == user, "Generation user differs from frozen prompt/shared actual ASR")
            if case["group"] == "voice_actual_asr_chat":
                require(
                    record.get("reference_user") == row["user"] and record.get("transcript") == user,
                    "Voice comparison lost original reference/shared ASR",
                )
            require(record.get("image") == row.get("image"), "Generation source image differs")
            index[key] = record
        require(set(index) == set(cases), "Missing candidate generation rows")
        records[variant] = index
        record_files.append({"variant": variant, "name": path.name, "sha256": core.sha256(path)})
    binding = {
        "manifest_sha256": manifest_sha,
        "protocol_sha256": core.sha256(protocol_path),
        "validation_result_sha256": core.sha256(report_path),
        "generation_files": record_files,
        "transcripts_sha256": core.sha256(transcripts_path),
        "asr_selection_sha256": core.sha256(selection_path),
        "validation_run_id": execution.get("run_id"),
        "batch_id": execution.get("batch_id"),
        "adapters": adapters,
        "model": report["model"],
        "model_revision": report["model_revision"],
    }
    require(
        bool(binding["validation_run_id"]) and bool(binding["batch_id"]), "Missing actual validation execution identity"
    )
    return manifest, protocol, cases, records, binding, root


def export_blind(artifacts, output):
    _, protocol, cases, records, binding, root = artifacts
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "Refuse to replace an existing blind-review packet")
    output.mkdir(parents=True, exist_ok=True)
    variants = list(protocol["candidate_order"])
    secrets.SystemRandom().shuffle(variants)
    aliases = dict(zip(("A", "B", "C"), variants, strict=True))
    nonce = secrets.token_hex(32)
    packet = {
        "schema_version": 1,
        "scope": "Validation-only blind semantic grading. Grade original intent/rubric; inspect actual full source images. Do not consult private-map.json or runtime default scores.",
        "cases": [],
        "candidate_mapping_commitment": mapping_commitment(aliases, nonce),
    }
    for key, case in cases.items():
        if case["group"] not in MANUAL:
            continue
        row = case["row"]
        rubric = (
            row.get("predeclared_semantic_rubric")
            if case["group"] == "text_chat"
            else row.get("references", {}).get("rubric")
        )
        require(isinstance(rubric, str) and bool(rubric.strip()), "Missing frozen manual semantic rubric")
        for alias in sorted(aliases):
            record = records[aliases[alias]][key]
            complete = eos_complete(record, protocol["generation"]["max_new_tokens"])
            packet["cases"].append(
                {
                    "case_id": key,
                    "candidate": alias,
                    "group": case["group"],
                    "user": row["user"],
                    "model_user": record["user"],
                    "system": row.get("system"),
                    "history": row.get("history", []),
                    "source_image": str(core.asset_path(row["image"], root)) if row.get("image") else None,
                    "reference_example": row.get("answer"),
                    "rubric": rubric,
                    "prediction": record["prediction"],
                    "decoder_complete": complete,
                    "source_image_inspection_required": bool(row.get("image")) and complete,
                }
            )
    packet_path = output / "blind-packet.json"
    core.write_json(packet_path, packet)
    packet_sha = core.sha256(packet_path)
    core.write_json(
        output / "private-map.json",
        {
            "schema_version": 1,
            "scope": "Do not provide to graders; preserve before grading, then commit with completed grades for audit",
            "aliases": aliases,
            "mapping_nonce": nonce,
            "artifact_binding": binding,
            "blind_packet_sha256": packet_sha,
        },
    )
    core.write_json(
        output / "grades-template.json",
        {
            "schema_version": 1,
            "blind_packet_sha256": packet_sha,
            "grader": "",
            "grades": [
                {
                    "case_id": case["case_id"],
                    "candidate": case["candidate"],
                    "passed": None,
                    "reason": "",
                    "source_image_inspected": None,
                }
                for case in packet["cases"]
            ],
        },
    )
    return {"manual_grade_count": len(packet["cases"]), "blind_packet_sha256": packet_sha}


def score_blind(artifacts, blind_dir, grades_path):
    _, protocol, cases, records, binding, _ = artifacts
    directory = Path(blind_dir)
    private = read_json(directory / "private-map.json")
    packet_path = directory / "blind-packet.json"
    packet = read_json(packet_path)
    require(
        private["artifact_binding"] == binding and private["blind_packet_sha256"] == core.sha256(packet_path),
        "Validation artifacts or blind packet changed after export",
    )
    aliases = private["aliases"]
    require(
        set(aliases) == {"A", "B", "C"} and set(aliases.values()) == set(protocol["candidate_order"]),
        "Invalid blinded candidate mapping",
    )
    require(
        packet.get("candidate_mapping_commitment") == mapping_commitment(aliases, private.get("mapping_nonce")),
        "Blinded candidate mapping changed after export",
    )
    grades = read_json(grades_path)
    require(
        grades.get("schema_version") == 1
        and grades.get("blind_packet_sha256") == private["blind_packet_sha256"]
        and isinstance(grades.get("grader"), str)
        and grades["grader"].strip(),
        "Manual grades must bind the original blind packet and name grader",
    )
    expected = {(key, alias) for key, case in cases.items() if case["group"] in MANUAL for alias in aliases}
    manual = {}
    for grade in grades.get("grades", []):
        key = (grade.get("case_id"), grade.get("candidate"))
        require(key in expected and key not in manual, "Unknown/repeated manual grade")
        require(
            type(grade.get("passed")) is bool and isinstance(grade.get("reason"), str) and grade["reason"].strip(),
            "Every manual grade needs a Boolean decision and reason",
        )
        case = cases[key[0]]
        record = records[aliases[key[1]]][key[0]]
        if case["row"].get("image") and eos_complete(record, protocol["generation"]["max_new_tokens"]):
            require(
                grade.get("source_image_inspected") is True, "Photo semantic grades require actual source inspection"
            )
        manual[key] = grade
    require(set(manual) == expected, "Missing manual grades; never reduce the frozen denominator")
    reverse = {variant: alias for alias, variant in aliases.items()}
    variants = {}
    for variant in protocol["candidate_order"]:
        counts = {group: 0 for group in WEIGHTS}
        incomplete = []
        for key, case in cases.items():
            record = records[variant][key]
            complete = eos_complete(record, protocol["generation"]["max_new_tokens"])
            if not complete:
                incomplete.append(key)
            passed = (
                manual[(key, reverse[variant])]["passed"]
                if case["group"] in MANUAL
                else core.score_output(case["row"], record["prediction"])["passed"]
            )
            counts[case["group"]] += int(complete and passed)
        variants[variant] = {
            "correct_counts": counts,
            "primary_numerator": sum(counts[group] * weight for group, weight in WEIGHTS.items()),
            "primary_denominator": DENOMINATOR,
            "incomplete_case_ids": incomplete,
            "all_generations_complete": not incomplete,
        }
    base = variants["base"]
    for variant in protocol["candidate_order"][1:]:
        candidate = variants[variant]
        gates = {}
        for group in WEIGHTS:
            rule = protocol["adapter_nonregression_gates_vs_base"][group + "_correct_count_minimum"]
            require(rule in {"base", "base minus 1"}, "Unknown predeclared nonregression rule")
            minimum = max(0, base["correct_counts"][group] - int(rule == "base minus 1"))
            gates[group] = {
                "minimum": minimum,
                "actual": candidate["correct_counts"][group],
                "passed": candidate["correct_counts"][group] >= minimum,
            }
        candidate["gates"] = gates
        candidate["eligible"] = candidate["all_generations_complete"] and all(gate["passed"] for gate in gates.values())
    chosen = "base"
    for variant in protocol["candidate_order"][1:]:
        if (
            variants[variant]["eligible"]
            and variants[variant]["primary_numerator"] > variants[chosen]["primary_numerator"]
        ):
            chosen = variant
    scores = {
        "schema_version": 1,
        "scope": "Validation-only exact macro and predeclared gates; decoder EOS does not prove semantic correctness",
        "artifact_binding": binding,
        "blind_packet_sha256": private["blind_packet_sha256"],
        "manual_grades_sha256": core.sha256(grades_path),
        "grader": grades["grader"],
        "denominators": protocol["validation_denominators"],
        "integer_weights": WEIGHTS,
        "variants": variants,
        "selected_variant": chosen,
        "base_fallback_scope": "Base fallback may contain incorrect/incomplete responses; no population or universal-capability claim",
    }
    selection = {
        "schema_version": 1,
        "dataset_manifest_sha256": binding["manifest_sha256"],
        "validation_run_id": binding["validation_run_id"],
        "validation_result_sha256": binding["validation_result_sha256"],
        "selected_variant": chosen,
        "criterion": f"Frozen protocol SHA {binding['protocol_sha256']}; exact integer primary /75600, all-EOS and nonregression gates, earlier archive on ties",
        "decision": "Choose eligible strict-improvement adapter"
        if chosen != "base"
        else "Select base; neither archive meets the strict-improvement standard",
        "validation_protocol_sha256": binding["protocol_sha256"],
        "manual_grades_sha256": scores["manual_grades_sha256"],
        "blind_packet_sha256": private["blind_packet_sha256"],
        "base_model": {"model": binding["model"], "revision": binding["model_revision"]},
    }
    if chosen != "base":
        adapter = next(item for item in binding["adapters"] if item["variant"] == chosen)
        selection.update(
            adapter_run_id=adapter["adapter_run_id"],
            adapter_sha256=adapter["adapter_sha256"],
            adapter_checkpoint=adapter["checkpoint"],
        )
    return scores, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "score"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/natural-assistant/v4/manifest.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/natural-assistant/v4/validation-protocol.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "outputs/natural-v4/data")
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--grades", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifacts = load_artifacts(args.manifest, args.protocol, args.validation_dir, args.data_root)
    if args.mode == "export":
        result = export_blind(artifacts, args.blind_dir)
    else:
        require(args.grades is not None and args.output is not None, "score needs --grades and --output")
        scores, selection = score_blind(artifacts, args.blind_dir, args.grades)
        require(
            not args.output.exists() or not any(args.output.iterdir()), "Refuse to overwrite a prior scored decision"
        )
        args.output.mkdir(parents=True, exist_ok=True)
        core.write_json(args.output / "scores.json", scores)
        selection["validation_scoring_sha256"] = core.sha256(args.output / "scores.json")
        core.write_json(args.output / "selection.json", selection)
        result = {
            "selected_variant": selection["selected_variant"],
            "selection_sha256": core.sha256(args.output / "selection.json"),
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
