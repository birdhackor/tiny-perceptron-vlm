"""核對共享區塊的參數、呼叫次數與保留的中間 tensor；不量測訓練品質。"""

import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from tiny_perceptron.model import Block, ModelConfig, TinyLM

torch.manual_seed(0)
torch.set_num_threads(1)
torch.set_default_device("cpu")
config = ModelConfig(width=8)
records = []
for mode, rounds in [("stacked", 3), ("shared", 1), ("shared", 3), ("shared", 5)]:
    shared = Block(config)
    blocks = nn.ModuleList([Block(config) for _ in range(rounds)] if mode == "stacked" else [shared] * rounds)
    calls = []
    saved = []
    states = []

    def track(module, args, output):
        calls.append(id(module))

    def pack(tensor):
        saved.append({"shape": list(tensor.shape), "numel": tensor.numel()})
        return tensor

    handles = [block.register_forward_hook(track) for block in dict.fromkeys(blocks)]
    state = torch.randn(1, 3, 8, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        for block in blocks:
            before = state
            state, cache, auxiliary = block(state)
            states.append({"changed": bool(not torch.equal(before, state)), "shape": list(state.shape)})
        state.square().mean().backward()
    optimizer = torch.optim.Adam(blocks.parameters())
    optimizer.step()
    first = next(iter(blocks))
    records.append({
        "mode": mode,
        "rounds": rounds,
        "parameter_numel": sum(p.numel() for p in blocks.parameters()),
        "parameter_tensors": len(list(blocks.parameters())),
        "named_parameters_without_dedup": len(list(blocks.named_parameters(remove_duplicate=False))),
        "unique_block_objects": len(set(map(id, blocks))),
        "block_calls": len(calls),
        "output_shape": list(state.shape),
        "state_updates": states,
        "single_block_parameter_breakdown": {name: p.numel() for name, p in first.named_parameters()},
        "adam_exp_avg_numel": sum(s["exp_avg"].numel() for s in optimizer.state.values()),
        "adam_exp_avg_sq_numel": sum(s["exp_avg_sq"].numel() for s in optimizer.state.values()),
        "saved_tensor_events": len(saved),
        "saved_tensor_numel_with_repeated_references": sum(s["numel"] for s in saved),
        "saved_tensor_event_shapes": saved,
    })
    for handle in handles:
        handle.remove()

moe = Block(ModelConfig(width=8, experts=2, top_k=1))
moe_state = torch.randn(1, 3, 8)
for _ in range(3):
    moe_state, _, _ = moe(moe_state)
tied = TinyLM(ModelConfig(width=8, tied=True))
local_source = {}
for label, obj in [("Module.parameters", nn.Module.parameters), ("Module.named_parameters", nn.Module.named_parameters), ("Module._named_members", nn.Module._named_members), ("Module.named_modules", nn.Module.named_modules), ("Adam._init_group", torch.optim.Adam._init_group)]:
    text = inspect.getsource(obj)
    local_source[label] = {"file": inspect.getfile(obj), "source": text, "source_sha256": hashlib.sha256(text.encode()).hexdigest()}
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-11/supplemental-check.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(torch.get_default_dtype()), "seed": "0", "threads": "1"},
    "records": records,
    "moe_shared_three_round_output_shape": list(moe_state.shape),
    "tied_embedding_output_same_parameter": tied.embedding.weight is tied.output.weight,
    "local_pytorch_source": local_source,
    "limits": "saved_tensors_hooks counts save events, including references to parameters and repeated tensors; these are not allocated bytes or peak memory. No quality or speed benchmark was run.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
