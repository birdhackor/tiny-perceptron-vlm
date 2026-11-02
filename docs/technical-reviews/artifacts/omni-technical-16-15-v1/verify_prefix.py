"""直接執行Kev所釘_rows_hidden與官方cache reorder方法；LM僅作狀態寫入探針。"""

import ast
import copy
import hashlib
import json
import platform
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

BASE = Path('/workspace/work/reference/omni-technical-16-15-v1')
OUT = Path(__file__).resolve().parent
COMMAND = 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-16-15-v1/verify_prefix.py'


def method(text, class_name, method_name):
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    code = textwrap.dedent(ast.get_source_segment(text, node))
    namespace = {'torch': torch, 'copy': copy, 'rows_per_pass': rows_per_pass,
                 'LinearAttentionCacheLayerMixin': globals().get('LinearAttentionCacheLayerMixin')}
    exec(compile('from __future__ import annotations\n' + code, 'pinned-' + method_name, 'exec'), namespace)
    return namespace[method_name]


# 此全域函式也取自Kev原文；預設值取原始ROW_PASS_TOKENS常數。
kev = (BASE / 'kev-model-2074cc4.py').read_text()
tree = ast.parse(kev)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rows_per_pass')
exec(compile(ast.get_source_segment(kev, fn), 'pinned-rows_per_pass', 'exec'), globals() | {'ROW_PASS_TOKENS': 16384}, local := {})
rows_per_pass = local['rows_per_pass']

cache_text = (BASE / 'transformers-cache-v5.17.0.py').read_text()
# 接合的容器只提供所取方法需要的欄位；不宣稱載入完整Transformers。
LinearAttentionCacheLayerMixin = type('LinearAttentionCacheLayerMixin', (), {
    '__init__': method(cache_text, 'LinearAttentionCacheLayerMixin', '__init__'),
    'reorder_cache': method(cache_text, 'LinearAttentionCacheLayerMixin', 'reorder_cache'),
})
FullLayer = type('FullLayer', (), {
    'reorder_cache': method(cache_text, 'CacheLayerMixin', 'reorder_cache'),
    'get_seq_length': lambda self: self.keys.shape[-2],
})
Cache = type('Cache', (), {'reorder_cache': method(cache_text, 'Cache', 'reorder_cache')})
lin = LinearAttentionCacheLayerMixin()
lin.device = torch.device('cpu')
lin.conv_states[0] = torch.tensor([[[2.]]])
lin.recurrent_states[0] = torch.tensor([[[[2.6]]]])
lin.is_conv_states_initialized[0] = True
lin.is_recurrent_states_initialized[0] = True
lin.has_previous_state[0] = True
lin.conv_kernel_size[0] = 1
full = FullLayer()
full.keys = torch.tensor([[[[1.], [2.]]]])
full.values = full.keys.clone()
prefix = Cache()
prefix.layers = [full, lin]

initial = copy.deepcopy(prefix)
observed_starts = []


def lm_probe(input_ids, position_ids, attention_mask, past_key_values, use_cache):
    a, b = past_key_values.layers
    observed_starts.append({'batch': input_ids.shape[0], 'K': a.keys.tolist(),
                            'conv': b.conv_states[0].tolist(), 'r': b.recurrent_states[0].tolist(),
                            'attention_mask': attention_mask.tolist()})
    assert a.keys.data_ptr() != prefix.layers[0].keys.data_ptr()
    assert b.conv_states[0].data_ptr() != prefix.layers[1].conv_states[0].data_ptr()
    assert b.recurrent_states[0].data_ptr() != prefix.layers[1].recurrent_states[0].data_ptr()
    # 模擬依分支輸入更新cache；檢查這些in-place寫入不能回到caller的prefix。
    delta = input_ids[:, 0].float()
    a.keys.add_(delta[:, None, None, None])
    a.values.add_(delta[:, None, None, None])
    b.conv_states[0].add_(delta[:, None, None])
    b.recurrent_states[0].add_(delta[:, None, None, None])
    return SimpleNamespace(last_hidden_state=input_ids[..., None].float())


cls = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
           and any(isinstance(m, ast.FunctionDef) and m.name == '_rows_hidden' for m in n.body))
Holder = type('Holder', (), {'_pad_rows': method(kev, cls, '_pad_rows'),
                            '_rows_hidden': method(kev, cls, '_rows_hidden')})
holder = Holder()
holder.device, holder.pad_id, holder.training = torch.device('cpu'), 0, False
holder.lm = lm_probe
rows = [([3], [2]), ([4], [2]), ([3], [2])]
outputs = holder._rows_hidden(rows, cache=prefix, prefix_len=2)
for row in rows:
    holder._rows_hidden([row], cache=prefix, prefix_len=2)

assert torch.equal(prefix.layers[0].keys, initial.layers[0].keys)
assert torch.equal(prefix.layers[0].values, initial.layers[0].values)
assert torch.equal(prefix.layers[1].conv_states[0], initial.layers[1].conv_states[0])
assert torch.equal(prefix.layers[1].recurrent_states[0], initial.layers[1].recurrent_states[0])
assert len(observed_starts) == 4
result = {
    'created_at': datetime.now(UTC).isoformat(), 'command': COMMAND,
    'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                    'device': 'cpu', 'platform': platform.platform()},
    'source_sha256': {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
                      for name in ['kev-model-2074cc4.py', 'transformers-cache-v5.17.0.py']},
    'methods': ['Kev._rows_hidden', 'Kev._pad_rows', 'Kev.rows_per_pass',
                'CacheLayerMixin.reorder_cache', 'LinearAttentionCacheLayerMixin.__init__',
                'LinearAttentionCacheLayerMixin.reorder_cache', 'Cache.reorder_cache'],
    'observed_starts': observed_starts, 'probe_outputs': [x.tolist() for x in outputs],
    'result': 'batched and sequential branches received copied initial KV/conv/recurrent buffers; caller prefix unchanged after deliberate in-place probe writes',
    'scope': 'exact pinned method bodies with minimal cache containers and an instrumented LM; no pretrained Qwen/Kev inference, serving endpoint, GPU or numerical parity claim',
}
with (OUT / 'prefix-execution.json').open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
