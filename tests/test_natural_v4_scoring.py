"""Adversarial artifact/selection checks with explicit synthetic runtime fixtures.

No weight downloads, real predictions or semantic grading are performed here.
"""

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v4_scoring", ROOT / "scripts/score_natural_v4_validation.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


@pytest.fixture
def artifacts(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (4, 4)).save(data / "picture.png")
    (data / "audio.wav").write_bytes(b"Synthetic integrity fixture; never decoded as audio")
    protocol = scoring.read_json(ROOT / "docs/natural-assistant/v4/validation-protocol.json")
    asr_path = tmp_path / "asr.json"
    repo, pin = scoring.core.ASR_VARIANTS["turbo"]
    scoring.core.write_json(asr_path, {"model": repo, "revision": pin})
    protocol["asr"].update(selection_file=str(asr_path), selection_sha256=scoring.core.sha256(asr_path))
    protocol_path = tmp_path / "protocol.json"
    scoring.core.write_json(protocol_path, protocol)
    rows = []
    for group, task in (
        ("photo_summary", "scene"),
        ("photo_fact", "scene"),
        ("text_presence", "text_presence"),
        ("single_ocr", "ocr"),
        ("ordered_ocr", "ocr_order"),
        ("text_chat", "chat"),
    ):
        for index in range(protocol["validation_denominators"][group]):
            suffix = (
                "/scene" if group == "photo_summary" else (f"/fact{index % 2 + 1}" if group == "photo_fact" else "")
            )
            row = {
                "id": f"fixture-{group}-{index}" + suffix,
                "task": task,
                "split": "validation",
                "family": f"fixture:{group}:{index}",
                "user": "這是一個測試問題。",
                "answer": "有",
                "history": [],
            }
            if task == "scene":
                row.update(
                    image="picture.png",
                    references={
                        "kind": "manual",
                        "qa_type": "scene" if group == "photo_summary" else "object",
                        "rubric": "Test-only rubric: identify the book.",
                    },
                )
            elif task == "chat":
                row.update(
                    answer="原始示例，不要求逐字匹配",
                    predeclared_semantic_rubric="Test-only semantic rubric; paraphrases accepted.",
                )
            elif task == "ocr_order":
                row.update(
                    image="picture.png",
                    answer="甲乙\n丙丁",
                    references={"kind": "ocr_order", "text": "甲乙\n丙丁", "strip_whitespace": False},
                )
            elif task == "ocr":
                row.update(
                    image="picture.png",
                    answer="文字 AB",
                    references={"kind": "ocr", "text": "文字 AB", "strip_whitespace": True},
                )
            else:
                row.update(image="picture.png", references={"kind": "exact", "accepted": ["有"]})
            rows.append(row)
    audio = []
    for index in range(16):
        task = "speech_chat" if index < 4 else "speech_transcription"
        audio.append(
            {
                "id": f"fixture-voice-{index}",
                "task": task,
                "family": f"voice:{index}",
                "split": "validation",
                "user": f"原始問題{index}",
                "audio": "audio.wav",
                "references": {"kind": "manual", "rubric": "Test-only voice question rubric."},
            }
        )
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
    manifest_path = tmp_path / "manifest.json"
    scoring.core.write_json(manifest_path, manifest)
    run = tmp_path / "validation"
    run.mkdir()
    transcripts = [
        {
            "id": row["id"],
            "task": row["task"],
            "reference_transcript": row["user"],
            "transcript": "實際 ASR 測試文字",
            "audio_sha256": scoring.core.sha256(data / "audio.wav"),
        }
        for row in audio
    ]
    scoring.core.write_json(run / "transcripts.json", transcripts)
    report = {
        "status": "completed",
        "split": "validation",
        "manifest_sha256": scoring.core.sha256(manifest_path),
        "model": scoring.core.MODEL_ID,
        "model_revision": scoring.core.MODEL_REVISION,
        "asr_model": repo,
        "asr_revision": pin,
        "min_pixels": 65536,
        "max_pixels": 524288,
        "max_tokens": 2048,
        "seed": 42,
        "requested_visual_text_rows": 124,
        "requested_audio_rows": 16,
        "requested_audio_chat_rows": 4,
        "asr": {"completed": True},
        "execution": {
            "stage": "validation",
            "manifest_sha256": scoring.core.sha256(manifest_path),
            "run_id": "validation-fixture",
            "batch_id": "fixture-v4",
            "adapter_run_id": "train-fixture",
            "runtime_options": {"asr_variant": "turbo"},
            "adapters": [
                {
                    "variant": variant,
                    "checkpoint": variant.removeprefix("adapter-"),
                    "adapter_run_id": "train-fixture",
                    "adapter_sha256": digit * 64,
                }
                for variant, digit in zip(protocol["candidate_order"][1:], ("a", "b"), strict=True)
            ],
        },
        "variants": {variant: {"completed": True, "generation_count": 132} for variant in protocol["candidate_order"]},
    }
    scoring.core.write_json(run / "result.json", report)
    for variant in protocol["candidate_order"]:
        generated = []
        for row in rows + [dict(row, task=task) for row in audio[:4] for task in ("typed_chat", "speech_chat")]:
            actual = row["task"] == "speech_chat"
            record = {
                "id": row["id"],
                "split": "validation",
                "task": row["task"],
                "variant": variant,
                "user": "實際 ASR 測試文字" if actual else row["user"],
                "image": row.get("image"),
                "prediction": row.get("answer", "語意測試回答"),
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
            generated.append(record)
        scoring.core.write_json(run / f"generations-{variant}.json", generated)
    return SimpleNamespace(data=data, manifest=manifest_path, protocol=protocol_path, run=run, tmp=tmp_path)


def load(fixture):
    return scoring.load_artifacts(fixture.manifest, fixture.protocol, fixture.run, fixture.data)


def blind_and_grade(fixture, predicate=lambda variant, case: False):
    loaded = load(fixture)
    blind = fixture.tmp / "blind"
    result = scoring.export_blind(loaded, blind)
    assert result["manual_grade_count"] == 303
    private = scoring.read_json(blind / "private-map.json")
    grades = scoring.read_json(blind / "grades-template.json")
    grades["grader"] = "Synthetic artifact-test grader; not actual semantic review"
    for grade in grades["grades"]:
        grade.update(
            passed=predicate(private["aliases"][grade["candidate"]], loaded[2][grade["case_id"]]),
            reason="Synthetic unit-test decision",
            source_image_inspected=True,
        )
    path = fixture.tmp / "grades.json"
    scoring.core.write_json(path, grades)
    return loaded, blind, path


def test_equal_rational_scores_keep_earlier_archive_and_clamp_negative_gates(artifacts):
    facts = set()

    def grade(variant, case):
        if variant == "adapter-step-001039":
            return case["group"] == "photo_summary" and case["row"]["id"].startswith("fixture-photo_summary-0/")
        if variant == "adapter-step-002077" and case["group"] == "photo_fact":
            facts.add(case["row"]["id"])
            return len(facts) <= 2
        return False

    loaded, blind, path = blind_and_grade(artifacts, grade)
    scores, selection = scoring.score_blind(loaded, blind, path)
    one, two = (scores["variants"][v] for v in ("adapter-step-001039", "adapter-step-002077"))
    assert one["primary_numerator"] == two["primary_numerator"]
    assert selection["selected_variant"] == "adapter-step-001039"
    assert one["gates"]["photo_fact"]["minimum"] == 0
    assert one["gates"]["text_chat"]["minimum"] == 0
    assert selection["adapter_checkpoint"] == "step-001039"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonboolean", "unknown_case", "photo_uninspected"])
def test_manual_artifact_rejections(artifacts, mutation):
    loaded, blind, path = blind_and_grade(artifacts)
    grades = scoring.read_json(path)
    if mutation == "missing":
        grades["grades"].pop()
    elif mutation == "duplicate":
        grades["grades"].append(grades["grades"][0])
    elif mutation == "nonboolean":
        grades["grades"][0]["passed"] = 1
    elif mutation == "unknown_case":
        grades["grades"][0]["case_id"] = "not-validation"
    else:
        grades["grades"][0]["source_image_inspected"] = False
    scoring.core.write_json(path, grades)
    with pytest.raises(ValueError):
        scoring.score_blind(loaded, blind, path)


def test_duplicate_raw_row_and_changed_asr_chain_are_rejected(artifacts):
    path = artifacts.run / "generations-base.json"
    original = scoring.read_json(path)
    scoring.core.write_json(path, original + [original[0]])
    with pytest.raises(ValueError, match="repeated generation"):
        load(artifacts)
    original[-1]["user"] = "Changed ASR only for one candidate"
    scoring.core.write_json(path, original)
    with pytest.raises(ValueError, match="shared actual ASR"):
        load(artifacts)


def test_unknown_eos_cannot_be_hidden_by_complete_report_or_manual_pass(artifacts):
    path = artifacts.run / "generations-adapter-step-001039.json"
    records = scoring.read_json(path)
    records[0].pop("eos_token_ids")
    scoring.core.write_json(path, records)
    loaded, blind, grades = blind_and_grade(artifacts, lambda variant, case: variant != "base")
    scores, selection = scoring.score_blind(loaded, blind, grades)
    assert not scores["variants"]["adapter-step-001039"]["eligible"]
    assert len(scores["variants"]["adapter-step-001039"]["incomplete_case_ids"]) == 1
    assert selection["selected_variant"] == "adapter-step-002077"


def test_exact_cap_eos_is_complete_and_chat_uses_semantic_override(artifacts):
    path = artifacts.run / "generations-adapter-step-001039.json"
    records = scoring.read_json(path)
    chat = next(record for record in records if record["task"] == "chat")
    chat.update(
        prediction="A semantically accepted test paraphrase",
        generated_token_ids=[1] * 383 + [2],
        generated_tokens=384,
        reached_max_new_tokens=True,
    )
    scoring.core.write_json(path, records)
    loaded, blind, grades = blind_and_grade(
        artifacts, lambda variant, case: variant == "adapter-step-001039" and case["group"] == "text_chat"
    )
    scores, selection = scoring.score_blind(loaded, blind, grades)
    assert scores["variants"]["adapter-step-001039"]["correct_counts"]["text_chat"] == 9
    assert scores["variants"]["adapter-step-001039"]["all_generations_complete"]
    assert selection["selected_variant"] == "adapter-step-001039"


def test_changed_blind_mapping_and_grade_packet_hash_are_rejected(artifacts):
    loaded, blind, grades = blind_and_grade(artifacts)
    private = scoring.read_json(blind / "private-map.json")
    private["aliases"]["A"], private["aliases"]["B"] = private["aliases"]["B"], private["aliases"]["A"]
    scoring.core.write_json(blind / "private-map.json", private)
    with pytest.raises(ValueError, match="mapping changed"):
        scoring.score_blind(loaded, blind, grades)


def test_artifact_binding_blocks_new_gold_and_result_changes(artifacts):
    loaded, blind, grades = blind_and_grade(artifacts)
    report = scoring.read_json(artifacts.run / "result.json")
    report["elapsed_seconds"] = 123
    scoring.core.write_json(artifacts.run / "result.json", report)
    with pytest.raises(ValueError, match="artifacts.*changed"):
        scoring.score_blind(load(artifacts), blind, grades)
    manifest = scoring.read_json(artifacts.manifest)
    manifest["rows"][0]["answer"] = "Changed frozen gold"
    scoring.core.write_json(artifacts.manifest, manifest)
    with pytest.raises(ValueError, match="different frozen manifest"):
        load(artifacts)


def test_exact_score_ties_with_base_keep_base(artifacts):
    loaded, blind, path = blind_and_grade(artifacts, lambda variant, case: True)
    scores, selection = scoring.score_blind(loaded, blind, path)
    assert {entry["primary_numerator"] for entry in scores["variants"].values()} == {75600}
    assert selection["selected_variant"] == "base" and "adapter_sha256" not in selection


def test_duplicate_json_object_key_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"passed":false,"passed":true}')
    with pytest.raises(ValueError, match="Repeated JSON object key"):
        scoring.read_json(path)


def test_cli_decision_matches_actual_modal_pretest_selection_contract(artifacts):
    _, blind, grades = blind_and_grade(artifacts, lambda variant, case: variant != "base")
    output = artifacts.tmp / "scored"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_natural_v4_validation.py"),
            "score",
            "--manifest",
            str(artifacts.manifest),
            "--protocol",
            str(artifacts.protocol),
            "--data-root",
            str(artifacts.data),
            "--validation-dir",
            str(artifacts.run),
            "--blind-dir",
            str(blind),
            "--grades",
            str(grades),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "adapter-step-001039" in result.stdout
    selection = scoring.read_json(output / "selection.json")
    assert selection["validation_scoring_sha256"] == scoring.core.sha256(output / "scores.json")
    # Use the real current gate functions, without importing Modal or invoking it.
    tree = ast.parse((ROOT / "scripts/modal_natural.py").read_text())
    names = {"safe_name", "checkpoint_names", "validate_selection"}
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[]
    )
    namespace = {"re": re}
    exec(compile(module, "actual_modal_selection_functions", "exec"), namespace)
    assert (
        namespace["validate_selection"](
            selection, scoring.core.sha256(artifacts.manifest), "train-fixture", "step-001039"
        )
        == selection
    )
