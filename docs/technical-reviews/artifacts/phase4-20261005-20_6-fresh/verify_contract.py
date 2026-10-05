"""Bounded argument/freeze-contract probe; loaders are recording CPU doubles."""
import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import torch
from torch import nn

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron import natural_assistant as implementation

spec = importlib.util.spec_from_file_location("review_cli", ROOT / "scripts/natural_assistant.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
options = cli.parser().parse_args(["train", "--output", "/tmp/unused-20_6-contract", "--device", "cpu", "--dtype", "float32", "--local-files-only"])
assert options.learning_rate == 3e-5 and options.lora_rank == 8
calls = []

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Linear(2, 2)
        self.model.language_model = nn.Linear(2, 2)
        self.config = types.SimpleNamespace(use_cache=True)

    def gradient_checkpointing_enable(self, **kwargs):
        calls.append({"gradient_checkpointing_enable": kwargs})

class ModelLoader:
    @staticmethod
    def from_pretrained(name, **kwargs):
        calls.append({"model_loader": name, "kwargs": {k: str(v) for k, v in kwargs.items()}})
        return Core()

class ProcessorLoader:
    @staticmethod
    def from_pretrained(name, **kwargs):
        calls.append({"processor_loader": name, "kwargs": {k: str(v) for k, v in kwargs.items()}})
        return object()

class LoraConfig:
    def __init__(self, **kwargs):
        self.values = kwargs

def get_peft_model(model, config):
    assert all(not p.requires_grad for p in model.parameters())
    calls.append({"lora_config_arguments": config.values})
    model.lora_A = nn.Parameter(torch.zeros(1))
    model.lora_B = nn.Parameter(torch.zeros(1))
    return model

transformers_double = types.ModuleType("transformers")
transformers_double.AutoProcessor = ProcessorLoader
transformers_double.Qwen3VLForConditionalGeneration = ModelLoader
peft_double = types.ModuleType("peft")
peft_double.LoraConfig = LoraConfig
peft_double.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
peft_double.get_peft_model = get_peft_model
sys.modules["transformers"] = transformers_double
sys.modules["peft"] = peft_double
model, _ = implementation.load_core(options, train=True)
assert model.training and model.config.use_cache is False
assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
assert all(p.requires_grad for n, p in model.named_parameters() if "lora_" in n)
config = next(c["lora_config_arguments"] for c in calls if "lora_config_arguments" in c)
assert config["r"] == 8 and config["lora_alpha"] == 16 and config["lora_dropout"] == 0.0
assert config["target_modules"] == implementation.LORA_TARGETS

# Execute the official regex helper's unchanged AST without PEFT dependencies.
official = ART / "sources/peft-v0.18.1-utils-other.py"
node = next(n for n in ast.parse(official.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "match_target_against_key")
namespace = {"re": re}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(official), "exec"), namespace)
match = namespace["match_target_against_key"]
checks = {
    "model.language_model.layers.0.self_attn.q_proj": True,
    "model.language_model.layers.27.self_attn.v_proj": True,
    "model.language_model.layers.0.self_attn.k_proj": False,
    "model.language_model.layers.0.mlp.q_proj": False,
    "model.visual.blocks.0.attn.q_proj": False,
    "whisper.model.decoder.layers.0.self_attn.q_proj": False,
}
for key, expected in checks.items():
    assert bool(match(config["target_modules"], key)) == expected
results = {
    "parser_train_defaults": {"learning_rate": options.learning_rate, "lora_rank": options.lora_rank},
    "load_core_recorded_arguments": calls,
    "base_frozen_before_adapter_creation": True,
    "base_and_visual_frozen_after_creation": True,
    "only_added_lora_parameters_trainable_in_double": True,
    "official_regex_helper_checks": checks,
    "scope": "Executed original load_core argument/freeze orchestration with recording CPU loader doubles, not actual Transformers/PEFT model loading. Actual selected tensor shapes/names are checked separately from original training measurements. No training loop was invoked.",
}
(ART / "contract-results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
