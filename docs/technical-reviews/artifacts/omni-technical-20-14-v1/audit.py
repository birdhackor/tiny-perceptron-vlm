"""20.14 審閱：執行凍結課文，並核對原始碼／設定中的有限契約。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-20-14-v1')
OUT = Path(__file__).with_name('audit-result.json')
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json'
pages = {p['page_id']: p for p in json.loads(FREEZE.read_text())['pages']}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen(page_id):
    page = pages[page_id]
    raw = (ROOT / page['snapshot']).read_bytes()
    assert digest(raw) == page['source_sha256']
    return raw.decode()


def run_code(code, label):
    captured, scope = io.StringIO(), {}
    with contextlib.redirect_stdout(captured):
        exec(compile(code, label, 'exec'), scope)
    return captured.getvalue().splitlines(), scope


def first_code(body):
    return re.findall(r'^```python\n(.*?)^```', body, re.M | re.S)[0]


body = frozen('20.14')
chapter = (ROOT / pages['20.14']['source']).read_text()
current = re.search(r'^## 20\.14 .*?(?=^## |\Z)', chapter, re.M | re.S).group()
assert current == body
code = first_code(body)
default_lines, default_scope = run_code(code, '20.14-frozen-code')
expected_default = [
    '獨立分支 [(3, 12), (3, 15)]',
    '原背景 2 10',
    '共用可變清單 [(3, 12), (4, 17)]',
    '被改過的背景 4 17',
]
assert default_lines == expected_default
assert default_scope['shared']['kv'] is default_scope['prefix']['kv']
assert default_scope['shared']['recurrent'] is default_scope['prefix']['recurrent']
assert code.count('("資訊足夠", 5)') == 1
exercise_code = code.replace('("資訊足夠", 5)', '("資訊足夠", 1)')
exercise_lines, exercise_scope = run_code(exercise_code, '20.14-exercise-code')
assert exercise_scope['separate'] == [(3, 12), (3, 11)]
assert exercise_scope['contaminated'] == [(3, 12), (4, 13)]

# 只執行直接前置的隨機初值計算例；不執行其校準／訓練安排。
prerequisite_code = first_code(frozen('19.23'))
prerequisite_lines, prerequisite_scope = run_code(prerequisite_code, '19.23-frozen-first-code')
assert '重算／共享最大差 0.0 重排最大差 0.0' in prerequisite_lines
assert prerequisite_scope['reorder_error'] <= 1e-5
assert [len(x['ids']) for x in prerequisite_scope['shared'][:3]] == [3, 2, 3]

# 以 AST 取原始碼中的小型純函式與 PointerHead；不匯入或載入 Kev 底座。
kev_raw = (CACHE / 'kev-model.py').read_bytes()
kev_tree = ast.parse(kev_raw.decode())
names = {'ContextOverflow', 'user_tokens', 'encode', 'rows_of', 'PointerHead'}
constant_names = {'SPECIAL', 'MAX_STATE', 'MAX_BRANCH', 'MAX_PACKED', 'OPT_NONE', 'OPT_DECIDE', '_SPECIAL_RE'}
nodes = []
for node in kev_tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
        nodes.append(node)
    if isinstance(node, ast.Assign):
        assigned = {x.id for t in node.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
        if assigned & constant_names:
            nodes.append(node)
scope = {'math': math, 're': re, 'torch': torch, 'nn': nn}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'kev-selected-original-ast', 'exec'), scope)


class LiteralTokenizer:
    """每個字元一個人工 ID，僅檢查原程式的位置／邊界，不模擬 Qwen tokenizer。"""

    def convert_tokens_to_ids(self, token):
        return 1000 + scope['SPECIAL'].index(token)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])


record = {'state': 'S', 'questions': [{'instr': 'Q', 'options': ['A', 'B'], 'label': 0}]}
enc = scope['encode'](LiteralTokenizer(), record)
state_ids, positions, rows = scope['rows_of'](enc)
assert len(rows) == 1 and rows[0]['ids'][-1] == 1004
assert enc['ids'][enc['decide_idx'][0]] == 1004
assert all(enc['ids'][x] == 1003 for x in enc['opt_idx'][0])
assert enc['opt_idx'][0][-1] < enc['decide_idx'][0]
head = scope['PointerHead'](2, dp=2).float().eval()
with torch.no_grad():
    for layer in [head.q, head.k]:
        layer.weight.copy_(torch.eye(2))
        layer.bias.zero_()
    q, opts = torch.tensor([1., 0.]), torch.eye(2)
    logits = head(q, opts)
    expected_logits = torch.tensor([1 / math.sqrt(2), 0.])
    assert torch.allclose(logits, expected_logits, atol=1e-7, rtol=0)
    head.temperature = 2.41
    calibrated = head(q, opts)
    assert torch.allclose(calibrated, expected_logits / 2.41, atol=1e-7, rtol=0)

qwen = json.loads((CACHE / 'qwen35-config.json').read_text())['text_config']
counts = Counter(qwen['layer_types'])
assert qwen['num_hidden_layers'] == len(qwen['layer_types']) == 32
assert counts == {'linear_attention': 24, 'full_attention': 8}
adapter = json.loads((CACHE / 'kev4b-adapter.json').read_text())
training = json.loads((CACHE / 'kev4b-training.json').read_text())
base_revision = '1001bb4d826a52d1f399e183466143f4da7b741b'
assert adapter['revision'] is None
assert training['base_revision'] == training['args']['base_revision'] == base_revision
assert adapter['base_model_name_or_path'] == training['args']['base'] == 'Qwen/Qwen3.5-4B-Base'
card = (CACHE / 'kev4b-card.md').read_text()
assert '| Languages | English |' in card
assert '| Validated context length | 8,192 tokens (see Long documents) |' in card
assert 'T = 2.41' in card and 'States of up to 65,536 tokens are served' in card
serve_limit = next(ast.literal_eval(n.value) for n in kev_tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'SERVE_MAX_STATE' for t in n.targets))
assert serve_limit == 65536

result = {
    'created_at': datetime.now(UTC).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-20-14-v1/audit.py',
    'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                    'device': 'cpu', 'platform': platform.platform(), 'threads': str(torch.get_num_threads())},
    'lesson': {'snapshot': pages['20.14']['snapshot'], 'source_sha256': digest(body.encode()),
               'current_slice_matches_snapshot': True, 'code_sha256': digest(code.encode()),
               'default_stdout': default_lines, 'exercise_stdout': exercise_lines,
               'shallow_nested_lists_alias_prefix': True},
    'prerequisite_19_23': {'snapshot': pages['19.23']['snapshot'],
                           'source_sha256': pages['19.23']['source_sha256'],
                           'executed_first_code_sha256': digest(prerequisite_code.encode()),
                           'stdout': prerequisite_lines, 'scope': '隨機初值 CPU 參考計算；無訓練與語義品質結論'},
    'kev_selected_original_ast': {'source_sha256': digest(kev_raw),
                                'selected_definitions': sorted(names), 'synthetic_tokenizer': True,
                                'decide_after_all_option_closings': True,
                                'raw_logits': logits.tolist(), 'temperature_2_41_logits': calibrated.tolist(),
                                'scope': '只執行固定原碼的邊界編碼與人工權重 PointerHead，未載入 Kev/Qwen 權重或神經快取'},
    'official_config_and_card': {'num_hidden_layers': 32, 'layer_type_counts': dict(counts),
                                 'serve_max_state': serve_limit, 'card_validated_context_length': 8192,
                                 'card_temperature': 2.41, 'card_language': 'English',
                                 'adapter_revision': adapter['revision'], 'training_base_revision': base_revision,
                                 'scope': '已讀官方原碼／設定／卡片中的明示值；沒有重跑卡片的品質評測或檢查 head.pt'},
    'all_assertions_passed': True,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'all_assertions_passed': True, 'output': str(OUT.relative_to(ROOT)),
                  'sha256': digest(OUT.read_bytes()), 'environment': result['environment']}, ensure_ascii=False))
