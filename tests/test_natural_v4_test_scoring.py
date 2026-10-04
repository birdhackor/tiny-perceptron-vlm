"""Adversarial final-test artifact checks with constructed, committed fixtures.

No model loading, training, real inference, or real semantic grades occur here.
"""

import copy
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import score_natural_v4_test as scoring

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    scoring.core.write_json(path, value)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture(params=["base", "adapter-step-001039"])
def artifacts(tmp_path, request):
    root = tmp_path / "repo"
    root.mkdir()
    for name in (*scoring.SOURCE_FILES, "scripts/score_natural_v4_validation.py", "scripts/score_natural_v4_test.py"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    data = root / "data"
    data.mkdir()
    Image.new("RGB", (4, 4)).save(data / "picture.png")
    (data / "audio.wav").write_bytes(b"Synthetic integrity fixture: not decoded as audio")
    asr_path = root / "asr.json"
    repo, pin = scoring.core.ASR_VARIANTS["turbo"]
    write(asr_path, {"model": repo, "revision": pin})
    protocol = scoring.read_json(ROOT / "docs/natural-assistant/v4/validation-protocol.json")
    protocol["asr"].update(selection_file="asr.json", selection_sha256=scoring.core.sha256(asr_path))
    protocol_path = root / "protocol.json"
    write(protocol_path, protocol)
    rows = []
    for group, task in (
        ("photo_summary", "scene"),
        ("photo_fact", "scene"),
        ("text_presence", "text_presence"),
        ("single_ocr", "ocr"),
        ("ordered_ocr", "ocr_order"),
        ("text_chat", "chat"),
    ):
        for index in range(scoring.COUNTS[group]):
            suffix = "/scene" if group == "photo_summary" else f"/fact{index % 2 + 1}" if group == "photo_fact" else ""
            row = {
                "id": f"fixture-{group}-{index}{suffix}",
                "task": task,
                "split": "test",
                "family": f"fixture:{group}:{index}",
                "user": "這是測試問題。",
                "answer": "有",
                "history": [],
                "system": "遵循測試上下文。",
            }
            if task == "scene":
                row.update(
                    image="picture.png",
                    references={
                        "kind": "manual",
                        "qa_type": "scene" if group == "photo_summary" else "object",
                        "rubric": "Fixture-only rubric: identify the book.",
                    },
                )
            elif task == "chat":
                row.update(
                    answer="原始示例，不要求逐字匹配",
                    predeclared_semantic_rubric="Fixture-only rubric: equivalent wording accepted.",
                )
            elif task in {"ocr", "ocr_order"}:
                answer = "ＡＢ 文字" if task == "ocr" else "甲乙\n丙 丁"
                row.update(
                    image="picture.png",
                    answer=answer,
                    references={"kind": task, "text": answer, "strip_whitespace": task == "ocr"},
                )
            else:
                row.update(image="picture.png", references={"kind": "exact", "accepted": ["有"]})
            rows.append(row)
    audio = [
        {
            "id": f"fixture-voice-{index}",
            "task": "speech_chat" if index < 4 else "speech_transcription",
            "family": f"voice:{index}",
            "split": "test",
            "user": "Ａ B。",
            "audio": "audio.wav",
            "answer": None,
            "references": {"kind": "manual", "rubric": "Fixture-only voice semantic rubric."},
        }
        for index in range(22)
    ]
    manifest = {
        "schema_version": 1,
        "dataset_version": "natural-assistant-v4",
        "rows": rows,
        "audio_rows": audio,
        "files": {
            name: {"sha256": scoring.core.sha256(data / name), "bytes": (data / name).stat().st_size}
            for name in ("picture.png", "audio.wav")
        },
    }
    manifest_path = root / "manifest.json"
    write(manifest_path, manifest)
    variant = request.param
    selection = {
        "schema_version": 1,
        "dataset_manifest_sha256": scoring.core.sha256(manifest_path),
        "validation_protocol_sha256": scoring.core.sha256(protocol_path),
        "validation_run_id": "validation-fixture",
        "validation_result_sha256": "d" * 64,
        "validation_scoring_sha256": "c" * 64,
        "manual_grades_sha256": "e" * 64,
        "blind_packet_sha256": "f" * 64,
        "selected_variant": variant,
        "criterion": "Frozen fixture validation criterion",
        "decision": "Fixture pre-test decision",
        "base_model": {"model": scoring.core.MODEL_ID, "revision": scoring.core.MODEL_REVISION},
    }
    adapters = []
    if variant != "base":
        selection.update(adapter_run_id="train-fixture", adapter_sha256="a" * 64, adapter_checkpoint="step-001039")
        adapters.append(
            {
                "variant": variant,
                "checkpoint": "step-001039",
                "adapter_run_id": "train-fixture",
                "adapter_sha256": "a" * 64,
            }
        )
    selection_path = root / "selection.json"
    write(selection_path, selection)
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "Synthetic frozen pre-test inputs",
    )
    revision = git(root, "rev-parse", "HEAD")
    run = tmp_path / "review"
    run.mkdir()
    transcripts = [
        {
            "id": row["id"],
            "task": row["task"],
            "reference_transcript": row["user"],
            "transcript": "AB。",
            "audio_sha256": scoring.core.sha256(data / "audio.wav"),
            "raw_token_ids": [3, 4, 1, 2],
            "raw_token_count": 4,
            "expected_decoder_prompt_ids": [3, 4],
            "decoder_prompt_matches": True,
            "generated_token_ids": [1, 2],
            "generated_token_count": 2,
            "max_new_tokens": 128,
            "eos_token_ids": [2],
            "ended_with_eos": True,
            "stop_reason": "eos",
            "truncated": False,
            "completion_unknown": False,
            "raw_errors": 999999,
            "errors": 999999,
        }
        for row in audio
    ]
    write(run / "transcripts.json", transcripts)
    runtime = {
        "steps": 2077,
        "seed": 42,
        "max_seconds": 3300,
        "max_pixels": 524288,
        "learning_rate": 0.0001,
        "asr_variant": "turbo",
        "asr_model": repo,
        "asr_revision": pin,
        "checkpoint_steps": [],
        "adapter_checkpoint": selection.get("adapter_checkpoint", ""),
        "adapter_checkpoints": [],
    }
    proof = {
        key: selection.get(key)
        for key in (
            "validation_run_id",
            "validation_result_sha256",
            "selected_variant",
            "adapter_run_id",
            "adapter_sha256",
            "criterion",
            "decision",
        )
    }
    proof["adapter_checkpoint"] = selection.get("adapter_checkpoint", "")
    report = {
        "schema_version": 1,
        "status": "completed",
        "split": "test",
        "selected_only": True,
        "manifest_sha256": scoring.core.sha256(manifest_path),
        "model": scoring.core.MODEL_ID,
        "model_revision": scoring.core.MODEL_REVISION,
        "asr_model": repo,
        "asr_revision": pin,
        "asset_sha256": {name: scoring.core.sha256(data / name) for name in ("picture.png", "audio.wav")},
        "min_pixels": 65536,
        "max_pixels": 524288,
        "max_tokens": 2048,
        "seed": 42,
        "requested_visual_text_rows": 170,
        "requested_audio_rows": 22,
        "requested_audio_chat_rows": 4,
        "asr": {"completed": True, "raw_micro_cer": 999999},
        "variants": {variant: {"completed": True, "generation_count": 178}},
        "execution": {
            "stage": "evaluate",
            "manifest_sha256": scoring.core.sha256(manifest_path),
            "revision": revision,
            "run_id": "test-fixture",
            "batch_id": "fixture-v4",
            "selection_sha256": scoring.core.sha256(selection_path),
            "pretest_selection": proof,
            "runtime_options": runtime,
            "runtime_options_sha256": scoring.digest(
                json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
            ),
            "adapters": adapters,
            "adapter_run_id": selection.get("adapter_run_id"),
            "adapter_sha256": selection.get("adapter_sha256"),
            "adapter_checkpoint": selection.get("adapter_checkpoint", ""),
        },
    }
    write(run / "result.json", report)
    generations = []
    for row in rows + [dict(row, task=task) for row in audio[:4] for task in ("typed_chat", "speech_chat")]:
        actual = row["task"] == "speech_chat"
        record = {
            "id": row["id"],
            "split": "test",
            "task": row["task"],
            "variant": variant,
            "family": row["family"],
            "reference_answer": row.get("answer"),
            "user": "AB。" if actual else row["user"],
            "image": row.get("image"),
            "prediction": row.get("answer") or "Synthetic response",
            "generated_token_ids": [1, 2],
            "generated_tokens": 2,
            "eos_token_ids": [2],
            "ended_with_eos": True,
            "stop_reason": "eos",
            "truncated": False,
            "completion_unknown": False,
            "reached_max_new_tokens": False,
        }
        if actual:
            record.update(reference_user=row["user"], transcript=record["user"])
        generations.append(record)
    write(run / f"generations-{variant}.json", generations)
    write(run / "generations.json", generations)
    return SimpleNamespace(
        root=root,
        data=data,
        manifest=manifest_path,
        protocol=protocol_path,
        selection=selection_path,
        run=run,
        tmp=tmp_path,
        variant=variant,
    )


def load(fixture):
    return scoring.load_artifacts(
        fixture.manifest, fixture.protocol, fixture.run, fixture.data, fixture.selection, repo_root=fixture.root
    )


def mutate(fixture, filename, change):
    path = fixture.run / filename
    value = scoring.read_json(path)
    change(value)
    write(path, value)
    if filename.startswith("generations-"):
        write(fixture.run / "generations.json", value)


def grade(fixture, passed=True):
    artifacts = load(fixture)
    blind = fixture.tmp / "blind"
    exported = scoring.export_blind(artifacts, blind)
    assert exported["manual_grade_count"] == 147
    grades = scoring.read_json(blind / "grades-template.json")
    grades.update(
        grader="Synthetic unit-test grader; no actual semantic review",
        independent_of_training_and_selection=True,
        private_mapping_consulted=False,
    )
    for record in grades["grades"]:
        record.update(passed=passed, reason="Synthetic test decision", source_image_inspected=True)
    path = fixture.tmp / "grades.json"
    write(path, grades)
    return artifacts, blind, path


def test_full_counts_rational_scores_and_independent_asr_cer(artifacts):
    loaded, blind, path = grade(artifacts)
    selection_before = artifacts.selection.read_bytes()
    result = scoring.score_blind(loaded, blind, path)
    assert result["correct_counts"] == scoring.COUNTS
    assert result["descriptive_capability_macro"] == {"numerator": 1, "denominator": 1, "decimal": 1.0}
    assert result["descriptive_response_micro"]["numerator"] == 1
    assert result["asr"]["count"] == 22
    assert result["asr"]["raw_errors"] == 44 and result["asr"]["raw_reference_characters"] == 88
    assert result["asr"]["raw_micro_cer"]["decimal"] == 0.5
    assert result["asr"]["normalized_errors"] == 0
    assert result["asr"]["normalized_micro_cer"]["decimal"] == 0
    assert result["asr"]["incomplete_case_ids"] == []
    assert artifacts.selection.read_bytes() == selection_before
    assert "primary_denominator" not in result and "integer_weights" not in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "time_limit_partial"),
        ("split", "validation"),
        ("selected_only", False),
        ("requested_visual_text_rows", 124),
        ("requested_audio_rows", 16),
        ("requested_audio_chat_rows", 8),
        ("model_revision", "b" * 40),
        ("asset_sha256", {}),
        ("seed", True),
    ],
)
def test_report_validation_and_count_risks(artifacts, field, value):
    mutate(artifacts, "result.json", lambda report: report.update({field: value}))
    with pytest.raises(ValueError):
        load(artifacts)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_variant",
        "stage",
        "selection_sha",
        "proof",
        "checkpoint",
        "runtime_hash",
        "comparison",
        "missing_revision",
    ],
)
def test_selection_execution_drift_rejected(artifacts, mutation):
    def change(report):
        execution = report["execution"]
        if mutation == "extra_variant":
            report["variants"]["adapter-step-002077"] = {"completed": True, "generation_count": 178}
        elif mutation == "stage":
            execution["stage"] = "validation"
        elif mutation == "selection_sha":
            execution["selection_sha256"] = "b" * 64
        elif mutation == "proof":
            execution["pretest_selection"]["decision"] = "Pick based on test outputs"
        elif mutation == "checkpoint":
            execution["adapter_checkpoint"] = "step-002077"
        elif mutation == "runtime_hash":
            execution["runtime_options_sha256"] = "b" * 64
        elif mutation == "comparison":
            execution["runtime_options"]["adapter_checkpoints"] = ["step-002077"]
            execution["runtime_options_sha256"] = scoring.digest(
                json.dumps(execution["runtime_options"], sort_keys=True, separators=(",", ":")).encode()
            )
        else:
            execution.pop("revision")

    mutate(artifacts, "result.json", change)
    with pytest.raises(ValueError):
        load(artifacts)


def test_selected_weight_drift_and_base_adapter_contamination(artifacts):
    def change(report):
        execution = report["execution"]
        execution["adapter_sha256"] = "b" * 64
        if execution["adapters"]:
            execution["adapters"][0]["adapter_sha256"] = "b" * 64

    mutate(artifacts, "result.json", change)
    with pytest.raises(ValueError, match="weights|drift"):
        load(artifacts)


def test_uncommitted_selection_and_gold_are_rejected_even_with_matching_report(artifacts):
    selection = scoring.read_json(artifacts.selection)
    selection["decision"] = "Changed after test"
    write(artifacts.selection, selection)
    mutate(
        artifacts,
        "result.json",
        lambda report: report["execution"].update(selection_sha256=scoring.core.sha256(artifacts.selection)),
    )
    with pytest.raises(ValueError, match="differs from execution Git commit"):
        load(artifacts)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "unknown", "prompt", "image", "voice_transcript", "reference", "fractional_count"],
)
def test_raw_generation_rows_and_shared_asr(artifacts, mutation):
    def change(rows):
        if mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows.append(copy.deepcopy(rows[0]))
        elif mutation == "unknown":
            rows[0]["id"] = "unknown-case"
        elif mutation == "prompt":
            rows[0]["user"] = "Changed request"
        elif mutation == "image":
            rows[0]["image"] = "other.png"
        elif mutation == "reference":
            rows[0]["reference_answer"] = "Changed gold"
        elif mutation == "fractional_count":
            rows[0]["generated_tokens"] = 2.0
        else:
            next(row for row in rows if row["task"] == "speech_chat")["user"] = (
                "Reference transcript substituted for actual ASR"
            )

    mutate(artifacts, f"generations-{artifacts.variant}.json", change)
    if mutation == "fractional_count":
        loaded, blind, path = grade(artifacts)
        assert len(scoring.score_blind(loaded, blind, path)["incomplete_case_ids"]) == 1
    else:
        with pytest.raises(ValueError):
            load(artifacts)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "hypothesis", "recording", "reference", "extra_file", "combined_extra"]
)
def test_asr_and_extra_candidate_artifacts(artifacts, mutation):
    if mutation == "extra_file":
        write(artifacts.run / "generations-extra.json", [])
    elif mutation == "combined_extra":
        mutate(artifacts, "generations.json", lambda rows: rows.append(copy.deepcopy(rows[0])))
    else:

        def change(rows):
            if mutation == "missing":
                rows.pop()
            elif mutation == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            elif mutation == "hypothesis":
                rows[0]["transcript"] = None
            elif mutation == "recording":
                rows[0]["audio_sha256"] = "b" * 64
            else:
                rows[0]["reference_transcript"] = "Changed source text"

        mutate(artifacts, "transcripts.json", change)
    with pytest.raises(ValueError):
        load(artifacts)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "nonboolean", "unknown", "photo_uninspected", "dependent", "mapping_consulted"]
)
def test_manual_grade_denominator_and_independence(artifacts, mutation):
    loaded, blind, path = grade(artifacts)
    grades = scoring.read_json(path)
    if mutation == "missing":
        grades["grades"].pop()
    elif mutation == "duplicate":
        grades["grades"].append(copy.deepcopy(grades["grades"][0]))
    elif mutation == "nonboolean":
        grades["grades"][0]["passed"] = 1
    elif mutation == "unknown":
        grades["grades"][0]["case_id"] = "unknown-case"
    elif mutation == "photo_uninspected":
        grades["grades"][0]["source_image_inspected"] = False
    elif mutation == "dependent":
        grades["independent_of_training_and_selection"] = False
    else:
        grades["private_mapping_consulted"] = True
    write(path, grades)
    with pytest.raises(ValueError):
        scoring.score_blind(loaded, blind, path)


@pytest.mark.parametrize("mutation", ["no_eos_ids", "truncated", "unknown", "wrong_final_token", "over_cap"])
def test_eos_masks_manual_pass_and_keeps_full_denominators(artifacts, mutation):
    def change(rows):
        record = rows[0]
        if mutation == "no_eos_ids":
            record.pop("eos_token_ids")
        elif mutation == "truncated":
            record["truncated"] = True
        elif mutation == "unknown":
            record["completion_unknown"] = True
        elif mutation == "wrong_final_token":
            record["generated_token_ids"][-1] = 1
        else:
            record.update(generated_token_ids=[1] * 384 + [2], generated_tokens=385)

    mutate(artifacts, f"generations-{artifacts.variant}.json", change)
    loaded, blind, path = grade(artifacts)
    result = scoring.score_blind(loaded, blind, path)
    assert result["correct_counts"]["photo_summary"] == 41
    assert result["denominators"]["photo_summary"] == 42
    assert result["group_accuracy"]["photo_summary"]["denominator"] == 42
    assert result["descriptive_response_micro"]["numerator"] == 177
    macro = Fraction(1) - Fraction(1, 42 * 2 * 5)
    assert result["descriptive_capability_macro"]["numerator"] == macro.numerator
    assert result["descriptive_capability_macro"]["denominator"] == macro.denominator


def test_exact_cap_eos_complete_and_chat_semantic_override(artifacts):
    def change(rows):
        row = next(row for row in rows if row["task"] == "chat")
        row.update(
            prediction="Accepted semantic paraphrase",
            generated_token_ids=[1] * 383 + [2],
            generated_tokens=384,
            reached_max_new_tokens=True,
        )

    mutate(artifacts, f"generations-{artifacts.variant}.json", change)
    loaded, blind, path = grade(artifacts)
    result = scoring.score_blind(loaded, blind, path)
    assert result["all_generations_complete"] and result["correct_counts"]["text_chat"] == 13


def test_declared_exact_normalization_preserves_order_script_case_and_punctuation(artifacts):
    def change(rows):
        presence = next(row for row in rows if row["task"] == "text_presence")
        presence["prediction"] = "有的，圖片有中文字"
        ocr = [row for row in rows if row["task"] == "ocr"]
        ocr[0]["prediction"] = "AB\n\t文字"
        ocr[1]["prediction"] = "ab文字"
        ocr[2]["prediction"] = "AB文字。"
        ocr[3]["prediction"] = "AB文字體"
        ordered = [row for row in rows if row["task"] == "ocr_order"]
        ordered[0]["prediction"] = "甲乙 丙 丁"
        ordered[1]["prediction"] = "甲乙\n丙丁"
        ordered[2]["prediction"] = " 甲乙\n丙 丁 "

    mutate(artifacts, f"generations-{artifacts.variant}.json", change)
    loaded, blind, path = grade(artifacts)
    result = scoring.score_blind(loaded, blind, path)
    assert result["correct_counts"]["text_presence"] == 17
    assert result["correct_counts"]["single_ocr"] == 7
    assert result["correct_counts"]["ordered_ocr"] == 1
    assert scoring.normalized("鸭") != scoring.normalized("鴨")


@pytest.mark.parametrize("mutation", ["packet", "mapping", "result", "grades_hash"])
def test_bound_review_tampering_rejected(artifacts, mutation):
    loaded, blind, grades_path = grade(artifacts)
    if mutation == "packet":
        packet = scoring.read_json(blind / "blind-packet.json")
        packet["cases"][0]["prediction"] = "Changed answer"
        write(blind / "blind-packet.json", packet)
        private = scoring.read_json(blind / "private-map.json")
        private["blind_packet_sha256"] = scoring.core.sha256(blind / "blind-packet.json")
        write(blind / "private-map.json", private)
    elif mutation == "mapping":
        private = scoring.read_json(blind / "private-map.json")
        private["mapping_nonce"] = "Changed nonce"
        write(blind / "private-map.json", private)
    elif mutation == "result":
        mutate(artifacts, "result.json", lambda report: report.update(elapsed_seconds=123))
        loaded = load(artifacts)
    else:
        grades = scoring.read_json(grades_path)
        grades["blind_packet_sha256"] = "b" * 64
        write(grades_path, grades)
    with pytest.raises(ValueError):
        scoring.score_blind(loaded, blind, grades_path)


def test_cli_writes_only_scores_and_refuses_replacing_review(artifacts):
    _, blind, grades = grade(artifacts)
    output = artifacts.tmp / "scored"
    command = [
        sys.executable,
        str(artifacts.root / "scripts/score_natural_v4_test.py"),
        "score",
        "--manifest",
        str(artifacts.manifest),
        "--protocol",
        str(artifacts.protocol),
        "--data-root",
        str(artifacts.data),
        "--test-dir",
        str(artifacts.run),
        "--selection",
        str(artifacts.selection),
        "--blind-dir",
        str(blind),
        "--grades",
        str(grades),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=artifacts.root, check=True)
    assert json.loads(result.stdout)["selected_variant"] == artifacts.variant
    assert {path.name for path in output.iterdir()} == {"scores.json"}
    failed = subprocess.run(command, capture_output=True, text=True, cwd=artifacts.root, check=False)
    assert failed.returncode != 0 and "Refuse to overwrite" in failed.stderr
    with pytest.raises(ValueError, match="Refuse to replace"):
        scoring.export_blind(load(artifacts), blind)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicated.json"
    path.write_text('{"passed":false,"passed":true}')
    with pytest.raises(ValueError, match="Repeated JSON object key"):
        scoring.read_json(path)


def test_partial_manifest_never_shrinks_frozen_test_denominator(artifacts):
    manifest = scoring.read_json(artifacts.manifest)
    manifest["rows"].pop()
    with pytest.raises(ValueError, match="denominators changed"):
        scoring.expected_cases(manifest)


def test_group_tradeoffs_use_test_fractions_not_validation_weights(artifacts):
    loaded, blind, path = grade(artifacts, passed=False)
    grades = scoring.read_json(path)
    summary = next(record for record in grades["grades"] if record["case_id"].startswith("scene:fixture-photo_summary"))
    summary["passed"] = True
    write(path, grades)
    result = scoring.score_blind(loaded, blind, path)
    # 31 exact cases pass; one summary is 1/(42*2) of photo capability,
    # not the validation summary weight 270/75600 = 1/280.
    expected = (Fraction(1, 84) + 3) / 5
    assert result["descriptive_capability_macro"]["numerator"] == expected.numerator
    assert result["descriptive_capability_macro"]["denominator"] == expected.denominator
    assert result["descriptive_response_micro"]["numerator"] == 16
    assert result["descriptive_response_micro"]["denominator"] == 89


def test_asr_unknown_completion_is_disclosed_and_cer_can_exceed_one(artifacts):
    def change(rows):
        rows[-1]["transcript"] = "Long synthetic hypothesis that greatly exceeds the reference"
        rows[-1].pop("eos_token_ids")

    mutate(artifacts, "transcripts.json", change)
    loaded, blind, path = grade(artifacts)
    result = scoring.score_blind(loaded, blind, path)
    assert result["asr"]["incomplete_case_ids"] == ["fixture-voice-21"]
    assert result["asr"]["records"][-1]["raw_errors"] > 4
    assert result["asr"]["normalized_micro_cer"]["numerator"] > 0
    assert result["correct_counts"] == scoring.COUNTS


@pytest.mark.parametrize("module", ["self", "validation", "core"])
@pytest.mark.parametrize("stage", ["export", "score"])
def test_actual_grading_source_drift_rejected_with_stale_loaded_artifacts(artifacts, monkeypatch, module, stage):
    source_module = {"self": scoring, "validation": scoring.validation, "core": scoring.core}[module]
    source = artifacts.tmp / f"copied-{module}.py"
    source.write_bytes(Path(source_module.__file__).read_bytes())
    monkeypatch.setattr(source_module, "__file__", str(source))
    if stage == "score":
        loaded, blind, grades = grade(artifacts)
    else:
        loaded = load(artifacts)
    source.write_bytes(source.read_bytes() + b"\n# Synthetic scoring-source drift after load\n")
    with pytest.raises(ValueError, match="Scoring source bytes changed"):
        if stage == "score":
            scoring.score_blind(loaded, blind, grades)
        else:
            scoring.export_blind(loaded, artifacts.tmp / "blind")


def test_copied_cli_casefold_change_after_blind_export_cannot_change_scores(artifacts):
    mutate(
        artifacts,
        f"generations-{artifacts.variant}.json",
        lambda rows: next(row for row in rows if row["task"] == "ocr").update(prediction="ab文字"),
    )
    loaded, blind, grades = grade(artifacts)
    original_scores = scoring.score_blind(loaded, blind, grades)
    assert original_scores["correct_counts"]["single_ocr"] == 9
    copied_scorer = artifacts.root / "scripts/score_natural_v4_test.py"
    original = copied_scorer.read_text()
    needle = 'return "".join(text.split()) if strip_whitespace else text'
    assert original.count(needle) == 1
    copied_scorer.write_text(
        original.replace(needle, 'return ("".join(text.split()) if strip_whitespace else text).casefold()')
    )
    output = artifacts.tmp / "changed-scored"
    command = [
        sys.executable,
        str(copied_scorer),
        "score",
        "--manifest",
        str(artifacts.manifest),
        "--protocol",
        str(artifacts.protocol),
        "--data-root",
        str(artifacts.data),
        "--test-dir",
        str(artifacts.run),
        "--selection",
        str(artifacts.selection),
        "--blind-dir",
        str(blind),
        "--grades",
        str(grades),
        "--output",
        str(output),
    ]
    run = subprocess.run(command, capture_output=True, text=True, cwd=artifacts.root, check=False)
    assert run.returncode != 0
    assert "Test artifacts or blind packet changed after export" in run.stderr
    assert not output.exists()
    expected_sources = scoring.scoring_sources()
    assert original_scores["artifact_binding"]["scoring_sources"] == expected_sources
    private = scoring.read_json(blind / "private-map.json")
    assert private["artifact_binding"]["scoring_sources"] == expected_sources


def test_fixture_preserves_crlf_bytes_under_inherited_autocrlf_true(tmp_path, monkeypatch):
    inherited_config = tmp_path / "inherited-gitconfig"
    inherited_config.write_bytes(b"[core]\n\tautocrlf = true\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(inherited_config))

    # Reproduce the failure on constructed bytes before the fixture-local fix.
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    json_path = baseline / "selection.json"
    json_path.write_bytes(b'{"synthetic": true}\r\n')
    git(baseline, "init", "-q")
    assert git(baseline, "config", "--get", "core.autocrlf") == "true"
    git(baseline, "add", ".")
    git(
        baseline,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "Synthetic CRLF baseline",
    )
    revision = git(baseline, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="differs from execution Git commit"):
        scoring.committed_bytes(json_path, revision, baseline)

    # Construct CRLF JSON independently of the current platform's newline mode.
    original_write = write

    def write_crlf(path, value):
        original_write(path, value)
        raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
        Path(path).write_bytes(raw.replace(b"\n", b"\r\n"))

    monkeypatch.setitem(globals(), "write", write_crlf)
    fixture = artifacts.__wrapped__(tmp_path, SimpleNamespace(param="base"))
    assert git(fixture.root, "config", "--local", "--get", "core.autocrlf") == "false"
    assert b"\r\n" in fixture.selection.read_bytes()
    loaded = load(fixture)
    assert loaded["binding"]["selected_variant"] == "base"
    assert len(loaded["cases"]) == 178
