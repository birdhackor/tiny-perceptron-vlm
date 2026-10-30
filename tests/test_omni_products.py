"""新產品的機制／資料隔離契約；不以隨機模型成績替代 sealed final。"""

from dataclasses import replace

import pytest
import torch

from tiny_perceptron.omni_products.av import AVConfig, AVModel, collate_av, local_audio
from tiny_perceptron.omni_products.choice import Candidate, ChoiceModel, collate_questions
from tiny_perceptron.omni_products.data import (
    Event,
    av_pair,
    av_waveform_swap_control,
    c_splits,
    fingerprint,
    manifest,
    split_a,
    visible_c_facts,
)
from tiny_perceptron.omni_products.questions import panel_c, training_questions
from tiny_perceptron.omni_products.reliability import accepted_rows, select_policy, select_temperature
from tiny_perceptron.omni_products.signal_metrics import analyzer_controls
from tiny_perceptron.omni_products.sound import ProgramTalker, SoundCodec


def test_split_counts_and_c_visible_erasure_collisions_are_excluded():
    frozen = manifest()
    assert frozen["counts"]["A"] == {"train": 38, "dev": 16, "final": 200}
    assert frozen["counts"]["sound"]["final"] == 200
    assert frozen["counts"]["C"]["final"] == 60
    observed = {}
    for split, families in c_splits().items():
        for index, family in enumerate(families):
            if split == "train":
                panel = ("complete", "removed", "missing")[index % 3]
                indices = [index + variant for variant in range(3)]
            else:
                panel = (
                    "missing"
                    if split in {"policy", "final"} and index >= 54
                    else "removed"
                    if split in {"policy", "final"} and index >= 48
                    else "complete"
                )
                indices = [index]
            for actual in indices:
                key = fingerprint(visible_c_facts(family, actual, panel))
                assert key not in observed or observed[key] == (split, family)
                observed[key] = split, family


def test_counterfactual_pairs_keep_observations_and_only_move_time():
    for panel in range(4):
        for family in split_a()["dev"]:
            left, right = av_pair(family, panel=panel)
            assert (left[1], right[1]) == (0, 1)
            a, b = left[0], right[0]
            assert torch.equal(a.frames, b.frames) and torch.equal(a.waveform, b.waveform)
            assert torch.equal(a.frame_ms, b.frame_ms) and a.duration_ms == b.duration_ms
            assert a.audio_start_ms != b.audio_start_ms
            a.validate()
            b.validate()


def test_same_metadata_different_waveforms_can_change_the_answer():
    left, right = av_waveform_swap_control(split_a()["dev"][0])
    a, b = left[0], right[0]
    assert a.audio_start_ms == b.audio_start_ms and a.waveform.shape == b.waveform.shape
    assert torch.equal(a.frames, b.frames) and not torch.equal(a.waveform, b.waveform)
    assert left[1] != right[1]


def test_local_features_ignore_other_blocks_and_tail_padding_keeps_valid_anchor():
    waveform = torch.randn(7000) * 0.1
    baseline, starts, ends, anchors = local_audio(waveform)
    changed = waveform.clone()
    changed[3200:] *= -3
    after, *_ = local_audio(changed)
    assert torch.equal(baseline[0], after[0])
    assert (ends - starts).max().item() <= 200
    assert anchors[-1].item() == (400 + 437.5) / 2


def test_time_representations_have_equal_trainable_capacity_and_missing_inputs_finite():
    additive = AVModel(AVConfig(representation="additive"))
    rope = AVModel(AVConfig(representation="time-rope"))
    assert sum(x.numel() for x in additive.parameters()) == sum(x.numel() for x in rope.parameters())
    item, _ = av_pair(split_a()["dev"][0])[0]
    batch = collate_av([(replace(item, image_present=False, audio_present=False), 2)])
    assert torch.isfinite(rope(batch)).all()


@pytest.mark.parametrize("change", ["duplicate", "unicode", "oversize", "empty"])
def test_choice_rejects_ambiguous_or_invalid_input(change):
    question = panel_c("dev")[0]["questions"]["choice"]
    if change == "duplicate":
        question = replace(question, candidates=(Candidate("x", "same"), Candidate("y", "same")))
    elif change == "unicode":
        question = replace(question, state="\ud800")
    elif change == "oversize":
        question = replace(question, text="字" * 43)
    else:
        question = replace(question, state=" ")
    with pytest.raises(ValueError):
        question.validate()


def test_choice_reordering_recompute_and_q1_q2_q1_cache_contract():
    torch.manual_seed(10)
    model = ChoiceModel().eval()
    questions = panel_c("dev")[0]["questions"]
    q1, q2 = questions["choice"], questions["rubric"]
    normal = model.predict(q1)
    model.predict(q2)
    repeated = model.predict(q1)
    reordered = model.predict(replace(q1, candidates=tuple(reversed(q1.candidates))))
    assert torch.allclose(torch.tensor(normal["logits"]), torch.tensor(repeated["logits"]), atol=1e-5, rtol=0)
    by_id = dict(zip(reordered["ids"], reordered["logits"]))
    assert max(abs(score - by_id[stable_id]) for stable_id, score in zip(normal["ids"], normal["logits"])) <= 1e-5
    batch = collate_questions([q1])
    with torch.no_grad():
        shared, recomputed = model(batch), model(batch, recompute=True)
    assert torch.allclose(shared["logits"], recomputed["logits"], atol=1e-5, rtol=0)
    assert torch.allclose(shared["candidate_hidden"], recomputed["candidate_hidden"], atol=1e-5, rtol=0)
    memory = model.state_memory(q1.state)
    with torch.no_grad():
        model.state_encoder.norm.bias.add_(0.01)
    with pytest.raises(ValueError, match="cache"):
        model.predict(q1, memory=memory)


def test_info_and_rubric_have_only_their_fixed_labels():
    model = ChoiceModel().eval()
    questions = panel_c("dev")[1]["questions"]
    assert set(model.predict(questions["info"])["ids"]) == {"yes", "no"}
    assert set(model.predict(questions["rubric"])["ids"]) == {"0", "1", "2"}
    assert max(len(question.state.encode()) for question in training_questions()) <= 512


def test_codec_gradients_update_all_three_modules_and_eval_never_resets():
    torch.manual_seed(4)
    model = SoundCodec()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    blocks = torch.randn(16, 80).clamp(-1, 1)
    optimizer.zero_grad()
    model(blocks)["loss"].backward()
    optimizer.step()
    for prefix in ("encoder", "codebook", "decoder"):
        assert any(
            not torch.equal(before[name], value)
            for name, value in model.state_dict().items()
            if name.startswith(prefix)
        )
    model.eval()
    with pytest.raises(ValueError, match="eval"):
        model.reset_unused_training_codes(torch.zeros(128), torch.randn(16, 32))


def test_codec_chunks_equal_offline_and_future_codes_cannot_change_prefix():
    model = SoundCodec().eval()
    codes = torch.tensor([1, 3, 5, 7])
    offline = model.decode_codes(codes)
    chunked = torch.cat(list(model.chunks(codes)))
    changed = model.decode_codes(torch.tensor([1, 3, 17, 19]))
    assert torch.equal(offline, chunked) and torch.equal(offline[:160], changed[:160])


def test_talker_does_not_force_reference_length_or_eos():
    model = ProgramTalker().eval()
    program = (Event(220, 100),)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(-100)
        model.output.bias[model.eos] = 100
    assert len(model.generate(program)["codes"]) == 0
    with torch.no_grad():
        model.output.bias.fill_(-100)
        model.output.bias[0] = 100
    generated = model.generate(program, max_blocks=17)
    assert len(generated["codes"]) == 17 and generated["capped"] and not generated["eos"]


def test_signal_analyzer_positive_and_failure_controls():
    result = analyzer_controls()
    assert result["all_passed"] and len(result["checks"]) == 9


def reliability_fixture(count):
    families = []
    for i in range(count):
        record = {"ids": ["action", "none"], "logits": [100, 0], "gold": "action"}
        families.append(
            {
                "family": str(i),
                "tasks": {kind: dict(record) for kind in ("choice", "info", "rubric")},
                "choice_info": {"ids": ["yes", "no"], "logits": [100, 0]},
                "world_supported": True,
                "choice_information_sufficient": True,
            }
        )
    return families


def test_temperature_ties_choose_one_and_policy_ties_choose_higher_tau():
    calibration = reliability_fixture(80)
    for family in calibration:
        for record in family["tasks"].values():
            record["logits"] = [0, 0]
    assert select_temperature(calibration)["temperature"] == 1
    assert select_policy(reliability_fixture(60), 1)["threshold"] == 0.99


def test_accepted_error_includes_unknown_information_even_if_choice_happens_correct():
    rows = reliability_fixture(2)
    rows[0]["choice_information_sufficient"] = False
    rows[1]["world_supported"] = False
    result = accepted_rows(rows, 1, 0.9)
    assert result["accepted"] == 2 and result["errors"] == 2
