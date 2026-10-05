"""Execute original loaders using tiny recording doubles; no pretrained weights."""
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import fetch_natural_release as release
from tiny_perceptron import natural_assistant as core

calls = []


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class CoreFactory:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        calls.append({"api": "Qwen3VLForConditionalGeneration.from_pretrained", "name": name,
                      "kwargs": {k: str(v) for k,v in kwargs.items()}})
        return Model()


class ASRFactory:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        calls.append({"api": "WhisperForConditionalGeneration.from_pretrained", "name": name,
                      "kwargs": {k: str(v) for k,v in kwargs.items()}})
        return Model()


class ProcessorFactory:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        calls.append({"api": "Processor.from_pretrained", "name": name,
                      "kwargs": {k: str(v) for k,v in kwargs.items()}})
        return object()


class AdapterFactory:
    @classmethod
    def from_pretrained(cls, model, adapter, *, is_trainable):
        calls.append({"api": "PeftModel.from_pretrained", "adapter": str(adapter),
                      "is_trainable": is_trainable})
        return model


public = json.loads((ROOT / "docs/natural-assistant/v4/public-release.json").read_bytes())
options = release.student_options(public, Path(__file__).parent, device="cpu", local_files_only=True)
transformers = ModuleType("transformers")
for name, value in {"AutoProcessor": ProcessorFactory, "Qwen3VLForConditionalGeneration": CoreFactory,
                    "WhisperProcessor": ProcessorFactory, "WhisperForConditionalGeneration": ASRFactory}.items():
    setattr(transformers,name,value)
peft = ModuleType("peft")
peft.PeftModel = AdapterFactory
with patch.dict(sys.modules, {"transformers": transformers, "peft": peft}):
    model, processor = core.load_core(options, adapter=options.adapter)
    assert not model.training and not model.weight.requires_grad
    assert [x["api"] for x in calls] == ["Processor.from_pretrained", "Qwen3VLForConditionalGeneration.from_pretrained"]
    assert all(x["kwargs"]["revision"] == public["base_model"]["revision"] for x in calls)
    assert calls[0]["kwargs"]["min_pixels"] == "65536" and calls[0]["kwargs"]["max_pixels"] == "524288"
    assert calls[1]["kwargs"]["attn_implementation"] == "sdpa" and calls[1]["kwargs"]["dtype"] == "torch.float32"
    asr, _ = core.load_asr(options)
    assert not asr.training
    assert all(x["kwargs"]["revision"] == public["asr_model"]["revision"] for x in calls[-2:])
    core.load_core(options, adapter="fictional-adapter-for-contract-check")
    assert calls[-1] == {"api":"PeftModel.from_pretrained", "adapter":"fictional-adapter-for-contract-check", "is_trainable":False}
assert len(calls) == 7
try:
    release.check_runtime(public)
except ValueError as exc:
    assert "3.12" in str(exc)
    print("EXPECTED_CPU_REVIEW_ENV_REJECTION", str(exc))
else:
    raise AssertionError("Review .venv must not be misrepresented as the Python3.12 natural-model runtime")
print(json.dumps({"scope":"Original loaders executed with recording doubles, not real transformers/peft or pretrained inference",
                  "calls":calls},ensure_ascii=False,indent=2))
print("LOADER_CONTRACT_ASSERTIONS_PASSED")
