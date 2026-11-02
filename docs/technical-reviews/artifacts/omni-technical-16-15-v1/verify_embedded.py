"""16.15：從指定凍結稿取出原 Python，直接執行並保存小份核對結果。"""

import contextlib
import copy
import hashlib
import inspect
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/reference/omni-technical-16-15-v1')
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix10.json'
COMMAND = 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-16-15-v1/verify_embedded.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def execute(code):
    namespace = {'__name__': '__lesson_audit__'}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, 'frozen-16.15-embedded.py', 'exec'), namespace)
    return namespace, output.getvalue()


freeze = json.loads(FREEZE.read_bytes())
page = next(x for x in freeze['pages'] if x['page_id'] == '16.15')
raw = (ROOT / page['snapshot']).read_bytes()
assert sha(raw) == page['source_sha256']
for path, digest in page['figures_sha256'].items():
    assert sha((ROOT / path).read_bytes()) == digest
code = re.findall(r'```python\n(.*?)```', raw.decode(), re.S)
assert len(code) == 1
code = code[0]
ns, stdout = execute(code)
expected_stdout = re.search(r'```text\n(.*?)```', raw.decode(), re.S)[1]
assert stdout == expected_stdout, (stdout, expected_stdout)
assert torch.equal(ns['memory'], ns['saved'])
assert torch.equal(ns['shared'][0][-1], ns['shared'][2][-1])
assert ns['shared'][1][-1][-1].item() > 0

# 各步皆重新執行原程式；僅改教材指定的一行。
exercise1 = code.replace('    m[0].add_(q)  # 錯誤的寫回；正確answer沒有這一行\n', '')
assert exercise1 != code
ex1, ex1_stdout = execute(exercise1)
ex1_delta = (ex1['first'][-1] - ex1['last'][-1]).abs().max().item()
assert ex1_delta == 0
exercise2 = code.replace('clean = [advance(copy.deepcopy(prefix), [x])', 'clean = [advance(prefix, [x])')
assert exercise2 != code
ex2, ex2_stdout = execute(exercise2)
assert ex2['clean'] == ns['polluted']
assert ex2['prefix']['r'] == ex2['clean'][-1]

# 刪掉KV紀錄不會回復已累積的r／conv。
cropped = copy.deepcopy(ns['dirty_prefix'])
before = {'r': cropped['r'], 'conv': list(cropped['conv'])}
del cropped['kv'][2:]
assert {'r': cropped['r'], 'conv': cropped['conv']} == before

# 官方所釘配置的層型計數，不執行模型權重推論。
config = json.loads((CACHE / 'qwen35-config-1001bb4.json').read_bytes())
types = config['text_config']['layer_types']
counts = {x: types.count(x) for x in sorted(set(types))}
assert counts == {'full_attention': 8, 'linear_attention': 24}

# 舊版Block原文直接執行；只替換同列後方的問題位置，觀察前方材料特徵。
old = (CACHE / 'old-c-model-2b3332c.py').read_text()
block = old[old.index('class Block('):old.index('\n\nclass LoopedSolver(')]
old_ns = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional}
exec(compile(block, 'pinned-old-C-Block.py', 'exec'), old_ns)
from types import SimpleNamespace

torch.manual_seed(0)
old_block = old_ns['Block'](SimpleNamespace(heads=1, width=2, ff=4)).eval()
x1 = torch.tensor([[[1., 0.], [0., 2.], [2., 0.]]])
x2 = x1.clone()
x2[:, 2] = torch.tensor([0., 2.])
with torch.no_grad():
    old_block_delta = (old_block(x1, torch.ones(1, 3, dtype=torch.bool))[:, :2]
                       - old_block(x2, torch.ones(1, 3, dtype=torch.bool))[:, :2]).abs().max().item()
assert old_block_delta > 0

# 實裝官方Module/Dropout的eval行為；不把eval當作所有平台決定性保證。
dropout = torch.nn.Dropout(0.5).eval()
drop_input = torch.ones(4, requires_grad=True)
drop_output = dropout(drop_input)
assert torch.equal(drop_input, drop_output) and drop_output.requires_grad
installed_hashes = {}
for cls, cached in [(torch.nn.Module, 'torch-module-5c4886.py'),
                    (torch.nn.Dropout, 'torch-dropout-5c4886.py')]:
    path = Path(inspect.getsourcefile(cls))
    actual_sha = sha(path.read_bytes())
    assert actual_sha == sha((CACHE / cached).read_bytes())
    installed_hashes[str(path)] = actual_sha

result = {
    'created_at': datetime.now(UTC).isoformat(),
    'command': COMMAND,
    'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                    'torch_git': torch.version.git_version, 'platform': platform.platform(),
                    'device': 'cpu', 'threads': str(torch.get_num_threads())},
    'source_snapshot': page['snapshot'], 'source_sha256': sha(raw),
    'freeze': str(FREEZE.relative_to(ROOT)), 'freeze_sha256': sha(FREEZE.read_bytes()),
    'embedded_code_sha256': sha(code.encode()),
    'embedded_stdout': stdout, 'matches_published_stdout_exactly': True,
    'inputs': {'state': ns['state'].tolist(), 'candidates': ns['candidates'].tolist(),
               'questions': [x.tolist() for x in ns['questions']], 'dtype': str(ns['state'].dtype)},
    'intermediates': {'M': ns['memory'].tolist(),
                      'Q1_weights': ns['shared'][0][1].tolist(),
                      'Q1_hidden': ns['shared'][0][2].tolist(),
                      'Q1_scores': ns['shared'][0][3].tolist(),
                      'Q2_none_unrounded': ns['shared'][1][-1][-1].item()},
    'readonly_check': {'unchanged_M': True, 'Q1_repeat_identical': True, 'max_difference': ns['error']},
    'exercise1': {'only_change': 'remove m[0].add_(q)', 'Q1_probability_difference': ex1_delta},
    'exercise2': {'only_change': 'clean line: copy.deepcopy(prefix) -> prefix',
                  'clean': ex2['clean'], 'prefix_after_clean': ex2['prefix'],
                  'later_polluted': ex2['polluted']},
    'kv_crop_check': {'kept_kv_positions': len(cropped['kv']), 'remaining_r': cropped['r'],
                       'remaining_conv': cropped['conv'], 'r_and_conv_unchanged': True},
    'pinned_base_layer_counts': counts,
    'pinned_old_Block': {'seed': 0, 'width': 2, 'heads': 1, 'ff': 4,
                         'changed_only_last_position': True, 'earlier_material_feature_max_difference': old_block_delta,
                         'scope': 'directly executed pinned Block with synthetic features; no trained-C or quality claim'},
    'dropout_eval': {'identity': True, 'requires_grad': drop_output.requires_grad},
    'installed_official_source_sha256': installed_hashes,
    'result': 'all assertions passed; one embedded block and two single-line exercise variants executed on CPU; no model weights loaded',
}
target = OUT / 'execution.json'
with target.open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
