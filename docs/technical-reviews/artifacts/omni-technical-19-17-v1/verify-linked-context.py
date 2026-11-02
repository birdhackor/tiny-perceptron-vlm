"""核對 19.17 明確連結的前置與新版輸入設計，不執行新版訓練。"""

import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

from tiny_perceptron.lesson_products.data import CANDIDATES, c_rows, family, split_of
from tiny_perceptron.lesson_products.inference import Product, parse_state


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
freeze = json.loads((ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json').read_bytes())
pins = []
for ident in ('7.23', '19.14', '1.15', '19.24'):
    p = next(p for p in freeze['pages'] if p['page_id'] == ident)
    raw = (ROOT / p['snapshot']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == p['source_sha256']
    current = (ROOT / p['source']).read_text()
    section = re.search(r'^## ' + re.escape(ident) + r' .*?(?=^## |\Z)', current, re.M | re.S).group()
    assert raw.decode() == section
    pins.append({'page_id': ident, 'snapshot': p['snapshot'], 'sha256': p['source_sha256'], 'current_exact_slice_match': True})

p24 = next(p for p in freeze['pages'] if p['page_id'] == '19.24')
blocks24 = re.findall(r'```python\n(.*?)```', (ROOT / p24['snapshot']).read_text(), re.S)
scope = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(blocks24[0], p24['snapshot'] + ':block1', 'exec'), scope)
select_route = scope['select_route']
request = {'text': '計算三個數', 'numbers': [1, 2, 3]}
judgment = {'route': 'b_sum', 'p_max': 0.96, 'info_yes': True}
accepted = select_route(request, judgment, tau=0.9)
assert accepted == {'route': 'b_sum', 'state': '三數加法;x=1;y=2;z=3', 'loops': 3}
assert parse_state(accepted['state']) == {'task': 'sum'}
rejected = select_route({**request, 'numbers': [1, 2, 7]}, judgment, tau=0.9, manual='b_sum')
assert rejected['route'] == 'handoff'
av = select_route({'text': '影片問題', 'has_video': True, 'has_audio': True}, {'route': 'a_av', 'p_max': 0.96, 'info_yes': True}, tau=0.9)
assert av == {'route': 'a_av', 'output': 'canonical semantic result'}

salt = 'finite-worlds-v1-coverage-1'
w = {'task': 'boxes', 'a': 2, 'b': 6, 'events': [('swap', 0)], 'query': 'B'}
w['family_id'] = family('boxes', (w['a'], w['b'], tuple(w['events'])))
split = split_of(w['family_id'], salt)
derived = list(c_rows(w, split))
assert all(r['split'] == split and r['family_id'] == w['family_id'] for r in derived)
assert set(CANDIDATES['choice']) == {'card', 'address', 'app', 'boxes', 'sum', 'none'}

contract_path = ROOT / 'docs/course-revision-20261029-omni/CONTRACTS.md'
contract = contract_path.read_text()
for text in ('尚無新版能力結果', 'UTF-8 bytes使用完整0–255', '原B權重、config、tokenizer與depth policy固定', 'C路由只讀文字及附件存在資訊'):
    assert text in contract
b = Product('/workspace/work/reference/omni-technical-19-17-v1/b', 'b')
rejected_new_unicode = []
try:
    b.tokenizer.encode('🦉')
    raise AssertionError('finite B tokenizer accepted unseen emoji')
except ValueError as error:
    rejected_new_unicode.append(str(error))
byte_probe = '新說法🦉'.encode('utf-8')
assert byte_probe.decode('utf-8') == '新說法🦉' and all(0 <= n <= 255 for n in byte_probe)
result = {
    'created_at': datetime.now(UTC).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-17-v1/verify-linked-context.py',
    'environment': {'python': platform.python_version(), 'device': 'cpu'},
    'linked_slice_pins': pins,
    'source_quote_scope': '19.17 L60 is a forward reference to the new C interface design, not a claim that new C training or integrated final evaluation has completed. CONTRACTS.md L3 explicitly says new capability results do not exist yet; 19.24 final paragraph says formal integration remains to be completed.',
    'contract': {'path': str(contract_path.relative_to(ROOT)), 'sha256': hashlib.sha256(contract_path.read_bytes()).hexdigest(), 'inspected_lines': '3;35–49'},
    'linked_19_24_actual_embedded_block1_stdout': stdout.getvalue(),
    'adapter_accepted': accepted,
    'adapter_rejected_even_manual': rejected,
    'av_route_placeholder': av,
    'av_limit': 'This function sees presence flags and returns a placeholder, with no video contents or recognition. No new C/A model inference performed.',
    'c_shared_family_example': {'family_id': w['family_id'], 'split': split, 'descendant_rows': len(derived), 'all_same_split': True},
    'old_b_finite_tokenizer_rejection': rejected_new_unicode,
    'byte_encoding_probe': {'text': '新說法🦉', 'utf8_bytes': list(byte_probe), 'roundtrip_equal': True, 'scope': 'Python encoding fact only; not a new C model test or evidence of semantic understanding.'},
    'result': 'All exact current-slice pins, family, old-B and linked adapter checks passed; future interface scope explicitly retained.',
}
target = OUT / 'linked-context-output.json'
with target.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'path': str(target.relative_to(ROOT)), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'result': result['result']}, ensure_ascii=False))
