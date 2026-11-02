"""chapter-04 導讀的短 CPU 診斷；不代表其他編號節的完整審閱。"""

import hashlib
import inspect
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.attention import manual_attention
from tiny_perceptron.model import Block, ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("diagnostic-output-v1.json")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
torch.set_num_threads(1)
torch.manual_seed(42)
freeze = json.loads(FREEZE.read_bytes())
pages = {p["page_id"]: p for p in freeze["pages"]}


def section(page_id):
    p = pages[page_id]
    raw = (ROOT / p["snapshot"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == p["source_sha256"]
    current = (ROOT / p["source"]).read_text()
    if p.get("selector") == "before-first-section":
        body = current.split("\n## ", 1)[0] + "\n"
    else:
        start = re.search(r"^## " + re.escape(page_id) + r" .+$", current, re.M).start()
        rest = current[start:]
        nxt = re.search(r"\n## ", rest)
        body = rest[: nxt.start() + 1] if nxt else rest
    assert body.encode() == raw, page_id
    return body


intro = section("chapter-04")
assert not re.findall(r"!\[[^\]]*\]\([^)]+\)", intro)
assert pages["chapter-04"]["figures_sha256"] == {}

# Content changes the attention read weights; the mask forbids future positions.
keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
q1, q2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
w1, w2 = (q1 @ keys.T).softmax(-1), (q2 @ keys.T).softmax(-1)
assert w1.argmax().item() == 0 and w2.argmax().item() == 1
q = torch.zeros(1, 1, 3, 2)
v = torch.tensor([[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]])
allowed = torch.ones(3, 3, dtype=torch.bool).tril()[None, None]
attended, weights = manual_attention(q, q, v, allowed)
changed_v = v.clone()
changed_v[:, :, 2] = 1000.0
changed_attended, _ = manual_attention(q, q, changed_v, allowed)
assert torch.equal(attended[:, :, :2], changed_attended[:, :, :2])
assert torch.equal(weights[~allowed], torch.zeros_like(weights[~allowed]))

# Every position is processed by the same complete DenseFFN.
dense = DenseFFN(4)
x = torch.ones(1, 2, 4)
y = dense(x)
assert torch.equal(y[:, 0], y[:, 1])
x[0, 1, 0] = 2
changed_y = dense(x)
assert torch.equal(y[:, 0], changed_y[:, 0])
assert not torch.equal(y[:, 1], changed_y[:, 1])

# Observe the full causal language-model route, with no optimizer update.
config = ModelConfig(vocab_size=20, width=8, layers=2)
model = TinyLM(config)
events = []
hooks = []
for name in ["embedding", "position", "blocks.0.attention", "blocks.0.ffn", "blocks.1.attention", "blocks.1.ffn", "final_norm", "output"]:
    module = model.get_submodule(name)

    def save_event(module, inputs, output, name=name):
        tensor = output[0] if isinstance(output, tuple) else output
        events.append({"module": name, "output_shape": list(tensor.shape)})

    hooks.append(module.register_forward_hook(save_event))
ids = torch.tensor([[1, 2, 3]])
result = model(ids)
for hook in hooks:
    hook.remove()
assert result["logits"].shape == (1, 3, 20)
assert events[0]["module"] == "embedding" and events[1]["module"] == "position"
assert events[-1]["module"] == "output"
first = {p.data_ptr() for p in model.blocks[0].parameters()}
second = {p.data_ptr() for p in model.blocks[1].parameters()}
assert not first & second
changed_ids = ids.clone()
changed_ids[0, 2] = 4
assert torch.equal(result["logits"][:, :2], model(changed_ids)["logits"][:, :2])

# The linked future topic uses one shared block repeatedly on changing states.
stacked = nn.ModuleList([Block(ModelConfig(width=8)) for _ in range(3)])
shared = Block(ModelConfig(width=8))
looped = nn.ModuleList([shared, shared, shared])
initial_state = torch.randn(1, 3, 8)
arrangements = []
for name, blocks in [("stacked", stacked), ("looped", looped)]:
    state = initial_state.clone()
    states_differ = []
    for block in blocks:
        previous = state
        state, _, _ = block(state)
        states_differ.append(not torch.equal(previous, state))
    arrangements.append({"design": name, "unique_parameter_elements": sum(p.numel() for p in blocks.parameters()), "block_calls": len(blocks), "state_changes_each_call": states_differ, "output_shape": list(state.shape)})
assert arrangements[0]["unique_parameter_elements"] == 3 * arrangements[1]["unique_parameter_elements"]
assert all(r["block_calls"] == 3 for r in arrangements)
assert len({id(block) for block in looped}) == 1

# Check only the introductory roadmap, not every claim in these destination pages.
roadmap = []
for page_id, needle in [("4.4", "class DenseFFN"), ("4.5", "class Block"), ("4.6", "class TinyLM")]:
    body = section(page_id)
    assert needle in body
    roadmap.append({"page_id": page_id, "heading": body.splitlines()[0], "declares": needle, "source_sha256": pages[page_id]["source_sha256"]})
assert "梯度" in section("4.7") and "層" in section("4.8").splitlines()[0]
assert "同一組區塊參數" in section("14.11")
later_uses = []
for page_id in ["5.1", "5.7", "5.9", "5.13", "5.15", "7.5", "7.7", "7.11"]:
    body = section(page_id)
    assert "TinyLM(" in body
    later_uses.append({"page_id": page_id, "source_sha256": pages[page_id]["source_sha256"], "usage": next(line for line in body.splitlines() if "TinyLM(" in line)})

output = {
    "reviewer_task": "/root/omni_technical_coordinator/technical_chapter_04",
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_chapter_04-v1/diagnose-chapter-04-v1.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "threads": str(torch.get_num_threads()), "seed": "42", "dtype": "torch.float32"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    "intro_sha256": hashlib.sha256(intro.encode()).hexdigest(),
    "attention": {"argmax_key_by_query": [w1.argmax().item(), w2.argmax().item()], "future_weights_are_zero": True, "future_value_change_keeps_earlier_outputs": True},
    "dense": {"same_input_same_output_across_positions": True, "change_position_1_keeps_position_0": True, "change_position_1_changes_position_1": True, "output_shape": list(y.shape)},
    "model": {"ids": ids.tolist(), "route": events, "logits_shape": list(result["logits"].shape), "layer_parameter_storage_disjoint": True, "future_id_change_keeps_earlier_logits": True},
    "stacked_and_shared": arrangements,
    "roadmap": roadmap,
    "alignment_then_depth_roadmap_present": True,
    "14.11_shared_parameter_destination_present": True,
    "later_TinyLM_uses": later_uses,
    "pytorch_original_source_inspected": [{"api": obj.__qualname__, "installed_file": inspect.getfile(obj), "start_line": inspect.getsourcelines(obj)[1], "file_sha256": hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()} for obj in [nn.Module.parameters, nn.Module.named_parameters, nn.Linear.forward]],
    "result": "All assertions passed. No training or optimizer update was run.",
    "scope": "Only chapter-04 introductory mechanism and roadmap; destination-page snippets are context, not separate lesson sign-offs. No quality, speed, GPU, or generalization conclusion."
}
with OUT.open("x", encoding="utf-8") as handle:
    json.dump(output, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(output, ensure_ascii=False))
