"""Artificial CPU cache fixture, using the pinned Kev method excerpt and HF cache API.

No tokenizer, weights, neural model forward, training, service, or final-family
inference is run. The fake backbone deliberately mutates a replica to expose aliasing.
"""

import copy
import functools
import hashlib
import inspect
import json
import platform
import textwrap
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

OUT = Path(__file__).resolve().parent
receipt = json.loads((OUT / "primary-source-receipts-v1.json").read_text())
source = receipt["excerpts"]["kev_rows_hidden"]["text"]
assert hashlib.sha256(source.encode()).hexdigest() == receipt["excerpts"]["kev_rows_hidden"]["sha256"]
layer_types = receipt["excerpts"]["qwen_layer_types"]["layer_types"]
assert len(layer_types) == 32 and layer_types.count("linear_attention") == 24 and layer_types.count("full_attention") == 8


def row_budget(rows, prefix_len=0, budget=16384):
    return max(1, budget // (prefix_len + max(len(r) for r in rows)))


def make_cache():
    # The config controls cache classes only; no model is instantiated.
    cfg = Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"])
    cache = DynamicCache(config=cfg)
    assert isinstance(cache.layers[0], LinearAttentionCacheLayerMixin)
    cache.update_conv_state(torch.tensor([[[1.0, 2.0]]]), 0, conv_kernel_size=2)
    cache.update_recurrent_state(torch.tensor([[[[2.6]]]]), 0)
    cache.update(torch.tensor([[[[1.0], [2.0]]]]), torch.tensor([[[[1.0], [2.0]]]]), 1)
    return cache


def snapshot(cache):
    linear, full = cache.layers
    return {"conv": linear.conv_states[0].tolist(), "r": linear.recurrent_states[0].tolist(), "keys": full.keys.tolist(), "values": full.values.tolist(), "flags": [linear.is_conv_states_initialized[0], linear.is_recurrent_states_initialized[0], linear.has_previous_state[0]]}


class FakeBackbone:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, position_ids, attention_mask, past_key_values, use_cache):
        assert use_cache and torch.equal(position_ids, torch.full_like(input_ids, 2))
        batch, length = input_ids.shape
        assert torch.equal(attention_mask, torch.ones((batch, length + 2), dtype=torch.long))
        linear, full = past_key_values.layers
        self.calls.append({"batch_size": batch, "prefix_before_mutation": snapshot(past_key_values)})
        step = input_ids.float().sum(-1)
        linear.conv_states[0].add_(step[:, None, None])
        linear.recurrent_states[0].add_(step[:, None, None, None])
        new_kv = input_ids.float()[:, None, :, None]
        full.update(new_kv, new_kv)
        hidden = torch.stack([linear.recurrent_states[0].flatten(1).sum(1), linear.conv_states[0].flatten(1).sum(1)], -1)
        return SimpleNamespace(last_hidden_state=hidden[:, None, :].expand(batch, length, 2))


class Fixture:
    training = False
    device = "cpu"

    def __init__(self):
        self.lm = FakeBackbone()

    def _pad_rows(self, rows):
        ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
        pos = torch.tensor([r[1] for r in rows], dtype=torch.long)
        return ids, pos, torch.ones_like(ids)


def fixture_run(method_source, budget):
    namespace = {"copy": copy, "torch": torch, "LinearAttentionCacheLayerMixin": LinearAttentionCacheLayerMixin, "rows_per_pass": functools.partial(row_budget, budget=budget)}
    exec(compile(textwrap.dedent(method_source), "<pinned-kev-rows-hidden-excerpt>", "exec"), namespace)
    model = Fixture()
    method = namespace["_rows_hidden"].__get__(model)
    cache = make_cache()
    before = snapshot(cache)
    results = method([([3], [2]), ([4], [2]), ([3], [2])], cache=cache, prefix_len=2)
    return {"before": before, "after": snapshot(cache), "results": [r.tolist() for r in results], "calls": model.lm.calls}


batch = fixture_run(source, 16384)
chunks = fixture_run(source, 3)
for run in [batch, chunks]:
    assert run["before"] == run["after"]
    assert run["results"][0] == run["results"][2]
assert batch["results"] == chunks["results"]
assert [c["batch_size"] for c in batch["calls"]] == [3]
assert [c["batch_size"] for c in chunks["calls"]] == [1, 1, 1]

# Negative control removes only the two dictionary copies before reorder_cache.
dirty = source.replace("target.conv_states = source.conv_states.copy()", "pass  # deliberately omit conv dictionary isolation")
dirty = dirty.replace("target.recurrent_states = source.recurrent_states.copy()", "pass  # deliberately omit recurrent dictionary isolation")
assert dirty != source
polluted = fixture_run(dirty, 3)
assert polluted["before"] != polluted["after"]
assert polluted["results"][0] != polluted["results"][2]

cache_file = Path(inspect.getfile(LinearAttentionCacheLayerMixin))
original_cache_digest = next(s["sha256"] for s in receipt["sources"] if s["source_id"] == "hf_cache")
cache_digest = hashlib.sha256(cache_file.read_bytes()).hexdigest()
assert cache_digest == original_cache_digest
result = {"reviewer_task": "/root/omni_p5_technical_16_15", "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "device": "CPU"}, "scope": "Exact pinned Kev _rows_hidden method excerpt and actual Transformers cache classes, driven by an artificial mutating fake backbone; no model weights or neural forward", "kev_method_source_sha256": hashlib.sha256(source.encode()).hexdigest(), "installed_cache_source_matches_official_v5_17_0": cache_digest, "fixed_base_layer_counts": {"total": len(layer_types), "linear_attention": layer_types.count("linear_attention"), "full_attention": layer_types.count("full_attention")}, "one_batch": batch, "one_row_chunks": chunks, "negative_control_without_dict_copies": polluted, "assertions_passed": True}
target = OUT / "prefix-api-execution-v1.json"
with target.open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"result": "pass", "artifact": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "clean": chunks["results"], "dirty": polluted["results"]}))
