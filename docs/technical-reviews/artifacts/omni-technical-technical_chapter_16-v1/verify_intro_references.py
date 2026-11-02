"""只核第16章導讀直接依賴的切片／程式指向，不代替16.x逐節審查。"""
import ast
import hashlib
import inspect
import json
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json'
pages = {p['page_id']: p for p in json.loads(FREEZE.read_text())['pages']}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def current_slice(page):
    raw = (ROOT / page['source']).read_bytes()
    if page['selector'] == 'before-first-section':
        return raw[:re.search(rb'^## ', raw, re.M).start()]
    lid = page['selector'].removeprefix('section:')
    start = re.search(rb'^## ' + re.escape(lid.encode()) + rb'(?=\s)', raw, re.M)
    assert start, lid
    end = re.search(rb'^## ', raw[start.end():], re.M)
    return raw[start.start(): start.end() + end.start()] if end else raw[start.start():]

def body(lid):
    return (ROOT / pages[lid]['snapshot']).read_text()

def blocks(lid):
    return re.findall(r'```python\n(.*?)```', body(lid).split('<details>')[0], re.S)

print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__,
                  'torch_git': torch.version.git_version, 'platform': platform.platform(),
                  'device': 'cpu', 'threads': torch.get_num_threads()}, ensure_ascii=False))
ids = ['chapter-16', '3.7', '7.23', '14.11', '19.17', '16.1', '16.3', '16.4',
       '16.5', '16.6', '16.7', '16.8', '16.9', '16.10', '16.11', '16.12', '16.13', '16.14']
for lid in ids:
    p = pages[lid]
    frozen = (ROOT / p['snapshot']).read_bytes()
    current = current_slice(p)
    assert frozen == current and sha(frozen) == p['source_sha256'], lid
    print(json.dumps({'check': 'freeze_current_exact_slice', 'id': lid,
                      'snapshot': p['snapshot'], 'sha256': sha(frozen)}, ensure_ascii=False))

for lid in ['3.7', '16.3', '16.4', '16.5', '16.8', '14.11']:
    ns = {'__name__': '__review__'}
    for index, code in enumerate(blocks(lid), 1):
        print(f'EXECUTE frozen lesson={lid} block={index} sha256={sha(code.encode())}')
        exec(compile(code, f'{pages[lid]["snapshot"]}:python-block-{index}', 'exec'), ns)
    if lid == '16.5':
        from tiny_perceptron.attention import attention_mask as library_mask
        parsed = [ast.parse(blocks(lid)[1]), ast.parse(inspect.getsource(library_mask))]
        assert ast.dump(parsed[0], include_attributes=False) == ast.dump(parsed[1], include_attributes=False)
        position = torch.arange(4)
        segment = torch.tensor([[0, 0, 1, 1]])
        for valid in (None, torch.tensor([[True, True, True, False]])):
            assert torch.equal(ns['attention_mask'](position, position, valid, segment),
                               library_mask(position, position, valid, segment))
        print('PASS embedded attention_mask AST and outputs match current library')

# 檢查函式庫KV順序與兩backend；不作速度或品質量測。
from tiny_perceptron.attention import CausalAttention
x = torch.randn(1, 4, 16)
manual = CausalAttention(16, heads=4, kv_heads=1, backend='manual').eval()
sdpa = CausalAttention(16, heads=4, kv_heads=1, backend='sdpa').eval()
sdpa.load_state_dict(manual.state_dict())
with torch.no_grad():
    out_m, cache_m = manual(x)
    out_s, cache_s = sdpa(x)
    assert cache_m[0].shape == (1, 1, 4, 4)
    assert cache_s[0].shape == (1, 1, 4, 4)
    assert torch.allclose(out_m, out_s, atol=1e-6)
    _, previous = manual(x[:, :3])
    _, appended = manual(x[:, 3:], cache=previous)
    assert appended[0].shape == (1, 1, 4, 4)
    assert torch.equal(appended[0][:, :, :3], previous[0])
print('PASS CausalAttention stores unexpanded KV, appends previous KV, and both backends agree for this CPU input')

# 用人工時鐘驗計時邊界，單位為sentinel成本，並非實際秒數或訓練模型結果。
import tiny_perceptron.lesson_products.inference as inference
class Clock:
    value = 0.0
    def tick(self, amount):
        self.value += amount
    def now(self):
        return self.value
clock = Clock()
class B:
    auto_policy = {'version': 1, 'low_max_events': 0, 'medium_max_events': 4, 'sum_depth': 1}
    def synchronize(self):
        pass
    def solve(self, state, loops):
        clock.tick(7 * loops)
        return {'loops': loops, 'mock': True}
class C:
    def synchronize(self):
        pass
    def judge(self, state, kind):
        cost = 3 if kind == 'choice' else 2
        clock.tick(cost)
        return {'selected': 'boxes' if kind == 'choice' else 'yes', 'seconds': float(cost)}
original_perf_counter = inference.time.perf_counter
try:
    inference.time.perf_counter = clock.now
    state = '兩盒追蹤;A=2;B=6;事件=交換;查詢=B'
    results = {effort: inference.auto_solve(B(), C(), state, effort=effort) for effort in ('auto', 'low', 'high')}
finally:
    inference.time.perf_counter = original_perf_counter
assert results['auto']['b_loops'] == 3 and results['auto']['seconds'] == 26
assert results['low']['seconds'] == 12 and results['high']['seconds'] == 40
assert results['low']['seconds'] + results['high']['seconds'] == 52
print(json.dumps({'check': 'auto_solve_mock_clock_cost_scope', 'units': 'artificial sentinels, not measured seconds',
                  'auto_total': results['auto']['seconds'], 'auto_router': results['auto']['c_seconds'],
                  'low_total': results['low']['seconds'], 'high_total': results['high']['seconds'],
                  'low_then_high_total': results['low']['seconds'] + results['high']['seconds']}, ensure_ascii=False))
print('PASS all bounded checks; no GPU, training, benchmark, or model-quality evaluation')
