"""獨立原判六個反例的實際修正；指標門檻沒有因模型失敗而放寬。"""

import pytest
import torch

from tiny_perceptron.omni_products.choice import ChoiceModel
from tiny_perceptron.omni_products.data import Event, av_pair, sound_splits, sound_wave, split_a
from tiny_perceptron.omni_products.questions import panel_c
from tiny_perceptron.omni_products.reliability import select_temperature
from tiny_perceptron.omni_products.signal_metrics import program_success
from tiny_perceptron.omni_products.sound import CodecConfig, ProgramTalker, SoundCodec, program_schedule


def test_fixed_length_av_does_not_reveal_event_position_to_previous_oracle():
    correct, total, modes = 0, 0, set()
    for panel in range(4):
        for family in split_a()["dev"]:
            for item, gold in av_pair(family, panel=panel):
                assert len(item.waveform) == 11200
                center = item.audio_start_ms + len(item.waveform) / 16 - 60
                index = int(torch.searchsorted(item.frame_ms, torch.tensor(center), right=True)) - 1
                predicted = int(item.frames[index].sum().item() == 225)
                correct += predicted == gold
                total += 1
                modes.add(int((item.waveform.abs() > 0.01).nonzero()[0]) // 3200)
    assert modes == {0, 2} and correct / total < 0.85


def test_arbitrary_partial_padding_cannot_change_valid_output_or_vq_losses():
    codec = SoundCodec()
    blocks = torch.randn(3, 80) * 0.1
    valid = torch.ones_like(blocks, dtype=torch.bool)
    valid[0, 40:] = False
    valid[2] = False
    changed = blocks.clone()
    changed[~valid] = 99
    before, after = codec(blocks, valid), codec(changed, valid)
    assert torch.equal(before["waveform"][valid], after["waveform"][valid])
    for field in ("codes", "reconstruction_loss", "codebook_loss", "commitment_loss"):
        assert torch.equal(before[field], after[field])
    assert before["codes"][2] == -1 and len(before["encoded"]) == 2


def test_hop40_and80_have_matching_targets_conditioning_and_generation_caps():
    for hop in (40, 80):
        codec = SoundCodec(CodecConfig(hop=hop)).eval()
        program = (Event(220, 100),)
        codes, _ = codec.encode_wave(sound_wave(program))
        schedule = program_schedule(program, hop)
        talker = ProgramTalker(hop=hop)
        assert len(schedule) == len(codes) + 1
        assert talker.config()["hop"] == hop and talker.max_blocks * hop == 16000
        previous = torch.full((1, len(schedule)), talker.bos, dtype=torch.long)
        assert talker(previous, schedule.unsqueeze(0)).shape[1] == len(codes) + 1


def test_analyzer_accepts_all_development_programs_at_both_original_phases():
    for program in sound_splits()["dev"]:
        for phase in (0, 0.8):
            assert program_success(sound_wave(program, phase), program)["success"]


def test_duration_rest_and_total_length_are_independently_checked():
    target = (Event(440, 200, 100), Event(880, 200))

    def wave(first_ms, rest_ms, second_ms, prefix_ms=0):
        tone1 = 0.6 * torch.sin(2 * torch.pi * 440 * torch.arange(first_ms * 8) / 8000)
        tone2 = 0.6 * torch.sin(2 * torch.pi * 880 * torch.arange(second_ms * 8) / 8000)
        return torch.cat((torch.zeros(prefix_ms * 8), tone1, torch.zeros(rest_ms * 8), tone2))

    assert not program_success(wave(140, 130, 200, 30), target)["success"]
    assert not program_success(wave(220, 50, 230), target)["success"]
    assert not program_success(torch.cat((sound_wave(target), torch.zeros(800))), target)["success"]


def test_extreme_logits_nll_uses_logsumexp_without_underflow_tie():
    families = [
        {
            "family": str(i),
            "tasks": {
                kind: {"ids": ["a", "b"], "gold": "b", "logits": [10000, 0]} for kind in ("choice", "info", "rubric")
            },
        }
        for i in range(80)
    ]
    result = select_temperature(families)
    assert result["temperature"] == 4
    assert min(row["family_equal_type_equal_nll"] for row in result["grid"]) == 2500


def test_public_memory_tensor_cannot_poison_internal_cache():
    model = ChoiceModel().eval()
    question = panel_c("dev")[0]["questions"]["choice"]
    baseline = model.predict(question)
    memory = model.state_memory(question.state)
    memory.values.add_(100)
    after = model.predict(question)
    assert baseline["logits"] == after["logits"]


@pytest.mark.parametrize("extra_ms", [5, 10, 15])
def test_short_unknown_frequency_event_in_required_rest_is_not_erased(extra_ms):
    target = (Event(440, 200, 100), Event(880, 200))
    waveform = sound_wave(target)
    start = 220 * 8
    waveform[start : start + extra_ms * 8] = 0.6 * torch.sin(2 * torch.pi * 660 * torch.arange(extra_ms * 8) / 8000)
    assert not program_success(waveform, target)["success"]
