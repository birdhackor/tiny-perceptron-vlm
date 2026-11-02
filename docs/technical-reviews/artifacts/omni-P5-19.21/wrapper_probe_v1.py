"""Exercise the real wrapper using artificial inputs and fixed fake outputs, never trained/final AV inference."""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.omni_products.av import AVConfig
from tiny_perceptron.omni_products.data import AVInput, split_a
from tiny_perceptron.omni_products.inference import SOUND_MAP, OmniAssistant


class FakeAV(nn.Module):
    def __init__(self, label):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.config = AVConfig()
        self.label = label

    def forward(self, batch):
        logits = torch.full((len(batch["frames"]), 3), -1.0)
        logits[:, self.label] = 2
        return logits


class FakeTalker:
    def __init__(self):
        self.programs = []

    def generate(self, program):
        self.programs.append(program)
        return {"codes": torch.tensor([0])}


class FakeCodec:
    def decode_codes(self, codes):
        return torch.zeros(8)


class FakeOldA:
    def __init__(self):
        self.calls = []

    def reply(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "stub-old-A"


def main():
    torch.set_num_threads(1)
    item = AVInput(torch.zeros(2, 1, 32, 32), torch.tensor([0.0, 1000.0]), torch.zeros(11200), 100, 2000)
    results = []
    for label, name in enumerate(("circle", "square", "unknown")):
        wrapper = OmniAssistant.__new__(OmniAssistant)
        wrapper.av, wrapper.talker, wrapper.codec = FakeAV(label), FakeTalker(), FakeCodec()
        record = wrapper.video(item, with_sound=True)
        assert record["answer"] == name and record["logits"] == [-1.0 if i != label else 2.0 for i in range(3)]
        assert ("sound" in record) == (name != "unknown")
        if name != "unknown":
            assert wrapper.talker.programs == [SOUND_MAP[name]]
        else:
            assert not wrapper.talker.programs
        results.append({"answer": name, "sound_generated": "sound" in record,
                        "same_answer_selects_sound_program": name == "unknown" or wrapper.talker.programs == [SOUND_MAP[name]]})
    wrapper.old_a = FakeOldA()
    old_input = {"messages": [{"role": "user", "content": "fixture"}], "task": "text", "tools": False}
    assert wrapper.old_reply(old_input) == "stub-old-A"
    assert wrapper.old_a.calls[0][0] is old_input["messages"]
    assert wrapper.old_a.calls[0][1] == {"task": "text", "max_new_tokens": 128, "tools": False}
    split = split_a()
    assert [len(split[k]) for k in ("train", "dev", "final")] == [38, 16, 200]
    assert len(set(sum(split.values(), []))) == 254
    invalid = []
    for name, kwargs in (
        ("17 frames", {"frames": torch.zeros(17, 1, 32, 32), "frame_ms": torch.arange(17) * 100}),
        ("wrong sample rate", {"sample_rate": 8000}),
        ("audio out of clip", {"audio_start_ms": 1900}),
        ("duplicate frame timestamps", {"frame_ms": torch.tensor([0.0, 0.0])}),
        ("nonfinite waveform", {"waveform": torch.tensor([float("nan")])}),
    ):
        fields = dict(item.__dict__)
        fields.update(kwargs)
        try:
            AVInput(**fields).validate()
        except ValueError as error:
            invalid.append({"case": name, "rejected": True, "message": str(error)})
        else:
            raise AssertionError(name)
    result = {"created_at": datetime.now(UTC).isoformat(),
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
              "video_records": results, "old_reply_delegation": True,
              "family_split_sizes": {key: len(value) for key, value in split.items()},
              "family_split_disjoint": True, "invalid_inputs": invalid,
              "scope": "Real OmniAssistant.video/old_reply and AVInput validation with fake AV logits, fake Talker/codec/old-A, synthetic zero inputs. No learned/final model loaded or measured."}
    path = Path(__file__).with_name("wrapper-probe-result-v1.json")
    with path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
