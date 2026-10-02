"""教師實際停止、風格真值與完整題目，是蒸餾比較不可省掉的條件。"""

import json
from types import SimpleNamespace

import pytest
import torch

from scripts.course_experiments import compression
from scripts.course_experiments.behavior import _conversation, _style_metrics, _style_record
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM


@pytest.mark.parametrize("suffix", [[5, 128], [128, 2], [2], []])
def test_hard_targets_preserve_actual_control_ids_and_termination(tmp_path, monkeypatch, suffix):
    teacher = TinyLM(ModelConfig(width=8, max_length=32))
    row = _conversation("1+1=?", "2", "arithmetic-probe")

    def controlled_generation(model, ids, **kwargs):
        return torch.cat((ids, ids.new_tensor([suffix])), dim=1)

    monkeypatch.setattr(compression, "generate", controlled_generation)
    ctx = SimpleNamespace(output=tmp_path, device="cpu")
    targets, audit = compression._hard_targets(ctx, teacher, [row], "probe", 4)
    assert targets[0]["_hard_ids"] == suffix
    assert targets[0]["_hard_eos"] == (2 in suffix)
    assert audit["audit"][0]["teacher_ids"] == suffix
    assert audit["audit"][0]["eos"] == (2 in suffix)
    if suffix:
        _, labels = compression._example(targets[0], 32)
        assert labels[labels != IGNORE].tolist() == suffix
        assert audit["usable_target_records"] == 1 and audit["zero_target_records"] == 0
        if 5 in suffix:
            assert "<image>" in audit["audit"][0]["teacher_answer"]
            assert audit["invalid_control_records"] == audit["invalid_control_tokens"] == 1
            assert audit["audit"][0]["invalid_special_tokens"][0]["id"] == 5
    else:
        assert audit["zero_target_records"] == 1 and audit["zero_target_families"] == [row["family"]]
        assert audit["usable_target_records"] == 0
        with pytest.raises(ValueError, match="不可捏造 EOS"):
            compression._example(targets[0], 32)


def style_records():
    arithmetic = split_records(arithmetic_records(), seed=42)["test"]
    rows = [_style_record(row, style, True) for row in arithmetic for style in ("concise", "vivid", "json")]
    for day in (3, 8, 17):
        date = f"2026-10-{day:02d}"
        rows += [
            _conversation(f"task=date;date={date};confirm", f"已確認{date}。", date, style="clarification"),
            _conversation(f"task=date;id={day};date=?;confirm", "請提供日期。", date, style="clarification"),
        ]
    return rows


def evaluated_answers(rows, answers=None):
    tok = ByteTokenizer()
    answers = answers if answers is not None else [row["messages"][-1]["content"] for row in rows]
    samples = []
    for row, answer in zip(rows, answers, strict=True):
        gold = row["messages"][-1]["content"]
        samples.append(
            {
                "family": row["family"],
                "question": row["messages"][-2]["content"],
                "expected": gold,
                "generated": answer,
                "generated_ids": tok.encode(answer) + [tok.eos_id],
                "exact": answer == gold,
            }
        )
    return {"examples": len(rows), "generated_samples": samples}


def test_all_27_gold_style_answers_include_clarification_in_the_denominator():
    rows = style_records()
    assert len(rows) == 27
    score = compression._style_scores(evaluated_answers(rows), rows)
    assert score["examples"] == score["content_correct"] == 27
    assert score["content_accuracy"] == 1
    assert score["rubric"]["clarification"]["content_correct"] == 6
    assert score["json_valid"] == score["json_examples"] == 7


def test_date_year_only_and_json_boolean_are_not_correct_style_answers():
    rows = style_records()
    date = next(row for row in rows if row["messages"][-1]["content"].startswith("已確認"))
    score = compression._style_scores(evaluated_answers([date], ["2026"]), [date])
    assert score["content_correct"] == 0
    json_row = next(row for row in rows if row["style"] == "json")
    score = compression._style_scores(evaluated_answers([json_row], ['{"answer": true}']), [json_row])
    assert score["content_correct"] == score["json_valid"] == 0


@pytest.mark.parametrize("style", ["concise", "vivid", "json", "clarification"])
def test_hidden_trailing_control_token_invalidates_every_style_rubric(style):
    row = next(row for row in style_records() if row["style"] == style)
    evaluation = evaluated_answers([row])
    sample = evaluation["generated_samples"][0]
    sample["generated_ids"].insert(-1, 5)
    sample["exact"] = False
    # 原 decoder 把 <image> 丟掉，仍會留下完整正確模板。
    assert sample["generated"] == row["messages"][-1]["content"]
    # shared rubric 也必須以 raw IDs 阻擋，不能只信可見答案或既有 exact 旗標。
    shared = _style_metrics({"samples": [dict(sample, exact=True)]}, [row])["rubric"][style]
    assert shared["content_correct"] == shared["style_correct"] == shared["json_valid"] == 0
    score = compression._style_scores(evaluation, [row])
    assert "<image>" in score["samples"][0]["generated"]
    assert score["samples"][0]["invalid_special_tokens"][0]["id"] == 5
    assert score["content_correct"] == score["json_valid"] == 0
    assert score["rubric"][style]["style_correct"] == 0


def test_style_scores_require_samples_to_match_the_original_records():
    rows = style_records()
    evaluation = evaluated_answers(rows)
    evaluation["generated_samples"] = evaluation["generated_samples"][::-1]
    with pytest.raises(ValueError, match="題目順序"):
        compression._style_scores(evaluation, rows)
    evaluation = evaluated_answers(rows)
    evaluation["generated_samples"].pop()
    with pytest.raises(ValueError, match="zip"):
        compression._style_scores(evaluation, rows)
    with pytest.raises(ValueError, match="非空"):
        compression._style_scores({"examples": 0, "generated_samples": []}, [])


def test_gsm8k_challenge_keeps_only_complete_prompts_with_generation_room(tmp_path, monkeypatch):
    original = [
        {"question": "Too long. " * 30, "answer": "worked answer\n#### 7"},
        {"question": "There are 2 apples and 3 pears. How many fruits?", "answer": "2+3=5.\n#### 5"},
    ]
    path = tmp_path / "gsm8k-train-first200.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in original))
    monkeypatch.setattr(compression, "extract_asset", lambda ctx, identifier: tmp_path)
    rows, selection = compression._challenge(SimpleNamespace(), maximum=128, reserved_tokens=24)
    assert len(rows) == 1 and rows[0]["question"] == original[1]["question"] and rows[0]["answer"] == "5"
    assert len(compression._prompt(rows[0])) + 24 <= 128
    assert selection["source_rows"] == 2 and selection["eligible_rows"] == selection["selected_rows"] == 1
    assert selection["skipped_rows"] == 1 and selection["selected_source_rows"] == [1]
    assert selection["skipped"][0]["source_row"] == 0
    rows, selection = compression._challenge(SimpleNamespace(), maximum=32, reserved_tokens=24)
    assert rows == [] and selection["status"] == "not_run" and selection["reason"]
    assert "exact_match" not in selection and "correct" not in selection
