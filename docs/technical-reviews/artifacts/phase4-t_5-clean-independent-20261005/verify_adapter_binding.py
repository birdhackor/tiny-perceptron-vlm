"""Bounded loader contract check with untrained tiny model and fabricated adapter; no capability evaluation."""
import copy
import hashlib
import json
import platform
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.adapters import base_state_sha256, load_lora_adapter
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=16))
state = copy.deepcopy(model.state_dict())
payload = {"format_version": "lora-v1", "config": asdict(model.config), "base_sha256": base_state_sha256(state), "scaling": "alpha/rank", "adapter": {"output": {"a": torch.full((2, 8), 0.1), "b": torch.full((264, 2), 0.2), "rank": 2, "alpha": 4}}}
result = {"purpose": __doc__, "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}, "cases": {}}
with tempfile.TemporaryDirectory(prefix="t5-loader-") as directory:
    path = Path(directory) / "fabricated-adapter.pt"
    torch.save(payload, path)
    information = load_lora_adapter(model, path)
    result["cases"]["matching_native_base"] = {"base_sha256_verified": information["base_sha256_verified"], "scaling": information["scaling"], "modules": information["modules"]}
    try:
        load_lora_adapter(model, path)
        raise AssertionError("duplicate adapter must fail")
    except ValueError as error:
        result["cases"]["duplicate_adapter"] = str(error)
    other = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=16))
    other.load_state_dict(state)
    with torch.no_grad():
        other.output.weight[0, 0] += 0.1
    try:
        load_lora_adapter(other, path)
        raise AssertionError("wrong base must fail")
    except ValueError as error:
        assert "base_sha256" in str(error)
        result["cases"]["modified_base"] = str(error)
result["temporary_weights_removed"] = not path.exists()
assert result["temporary_weights_removed"]
result["loader_source_sha256"] = hashlib.sha256((ROOT / "tiny_perceptron/adapters.py").read_bytes()).hexdigest()
(HERE / "adapter-binding.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
