"""以人工 backbone 回傳值檢查 Kev 固定原碼的呼叫路徑；不載入模型。"""

import ast
import hashlib
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

root = Path(__file__).resolve().parents[4]
source = Path('/workspace/work/reference/omni-technical-20-14-v1/kev-model.py').read_bytes()
tree = ast.parse(source.decode())
selected = {'rows_of'}
method_names = {'forward_batch', 'forward_rows_batch', 'probs', 'probs_and_prefix', '_branch_rows_from_prefix'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DecisionModel')
nodes += [n for n in model.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
scope = {'torch': torch, 'F': F, 'nn': nn, 'math': math, 're': re}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'kev-original-control-flow', 'exec'), scope)

# 人工編碼已符合原 rows_of 契約：兩個 state token，接兩個獨立 question branch。
enc = {'ids': [10, 11, 20, 21, 22, 30, 31, 32], 'pos': [0, 1, 2, 3, 4, 2, 3, 4],
       'seg': [0, 0, 1, 1, 1, 2, 2, 2], 'decide_idx': [4, 7], 'opt_idx': [[3], [6]]}
calls = []
prefix_marker = object()
fake = SimpleNamespace(device='cpu')
fake.rows_form = lambda encs: True
fake.head = lambda decide, options: torch.zeros(len(options))


def hidden(rows, cache=None, prefix_len=0):
    calls.append({'rows': [ids for ids, _ in rows], 'prefix_supplied': cache is prefix_marker,
                  'prefix_len': prefix_len})
    return [torch.zeros(len(ids), 2) for ids, _ in rows]


def prefix(record):
    calls.append({'prefix_called': True, 'state_tokens': record['seg'].count(0)})
    return 2, prefix_marker, None


fake._rows_hidden = hidden
fake.forward_rows_batch = lambda encs, shared_prefix=False: scope['forward_rows_batch'](fake, encs, shared_prefix)
fake._branch_rows_from_prefix = lambda record, cache: scope['_branch_rows_from_prefix'](fake, record, cache)
fake.prefix = prefix
fake.probs_and_prefix = lambda record: scope['probs_and_prefix'](fake, record)
scope['forward_batch'](fake, [enc])
forward_calls = calls[:]
assert forward_calls == [{'rows': [[10, 11, 20, 21, 22], [10, 11, 30, 31, 32]],
                          'prefix_supplied': False, 'prefix_len': 0}]
calls.clear()
scope['probs'](fake, enc)
assert calls == [{'prefix_called': True, 'state_tokens': 2},
                 {'rows': [[20, 21, 22], [30, 31, 32]], 'prefix_supplied': True, 'prefix_len': 2}]
result = {'created_at': datetime.now(UTC).isoformat(),
          'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-20-14-v1/kev-control-flow.py',
          'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'device': 'cpu'},
          'source_sha256': hashlib.sha256(source).hexdigest(),
          'original_methods_executed': sorted(method_names), 'forward_calls': forward_calls, 'probs_calls': calls,
          'all_assertions_passed': True,
          'scope': '只驗原碼控制流與行內容。backbone 隱向量與 head 輸出均人工；未驗 Kev 實際張量快取、權重或兩路機率等價。'}
out = Path(__file__).with_name('kev-control-flow-result.json')
with out.open('x') as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'all_assertions_passed': True, 'output': str(out.relative_to(root)),
                  'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}, ensure_ascii=False))
