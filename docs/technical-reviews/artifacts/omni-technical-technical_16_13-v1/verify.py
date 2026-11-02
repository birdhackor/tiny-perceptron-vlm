"""16.13 的教材原碼、練習與圖格核驗；不使用 GPU 或訓練。"""
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import pathlib
import platform
import re
import sys
import xml.etree.ElementTree as ET

import torch
from tiny_perceptron.attention import attention_mask

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).with_name('execution.json')
assert not OUT.exists(), '不可覆寫既有執行紀錄'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def section(path, sec):
    text = (ROOT / path).read_text()
    return re.search(r'^## ' + re.escape(sec) + r' .*?(?=^## |\Z)', text, re.M | re.S).group()

def run(code):
    ns = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, '<教材原碼或明示練習變體>', 'exec'), ns)
    return ns, out.getvalue()

freeze_path = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(p for p in freeze['pages'] if p['page_id'] == '16.13')
body = (ROOT / page['snapshot']).read_text()
assert digest(body.encode()) == page['source_sha256']
assert section('course/chapters/16.md', '16.13') == body
code = re.findall(r'```python\n(.*?)```', body, re.S)[0]
ns, stdout = run(code)
assert stdout == ('完整因果 允許格數 21 位置 3 可讀 [0, 1, 2, 3]\n'
                  '滑動視窗 允許格數 15 位置 3 可讀 [1, 2, 3]\n'
                  '區塊局部 允許格數 12 位置 3 可讀 [3]\n')
assert tuple(attention_mask(ns['position'], ns['position']).shape) == (1, 1, 6, 6)
expected = {
    'causal': [[k <= q for k in range(6)] for q in range(6)],
    'sliding': [[k <= q and q-k < 3 for k in range(6)] for q in range(6)],
    'block_local': [[k <= q and k//3 == q//3 for k in range(6)] for q in range(6)],
}
for name, matrix in expected.items():
    assert ns[name].tolist() == matrix
    assert ns[name].dtype == torch.bool and ns[name].shape == (6, 6)
assert ns['position'].dtype == torch.int64

variants = {}
for label, modified in [
    ('remove_causal_intersection', code.replace('sliding = causal & (', 'sliding = (')),
    ('exercise_query_4', code.replace('query = 3', 'query = 4')),
    ('exercise_group_size_2', code.replace('group = position // 3', 'group = position // 2')),
]:
    state, output = run(modified)
    variants[label] = {'modified_code_sha256': digest(modified.encode()), 'stdout': output}
    if label == 'remove_causal_intersection':
        assert state['sliding'].sum().item() == 30
        assert state['position'][state['sliding'][3]].tolist() == [1,2,3,4,5]
    elif label == 'exercise_query_4':
        assert [state['position'][state[n][4]].tolist() for n in expected] == [list(range(5)), [2,3,4], [3,4]]
    else:
        assert state['group'].tolist() == [0,0,1,1,2,2]
        assert state['block_local'].sum().item() == 9
        assert state['position'][state['block_local'][3]].tolist() == [2,3]
        assert torch.equal(ns['causal'], state['causal'])
        assert torch.equal(ns['sliding'], state['sliding'])

# 實際讀取目前必要前提，不沿用舊快照中的函式。
prerequisites = {}
for path, sec in [('course/chapters/16.md','16.5'), ('course/chapters/16.md','16.9'),
                  ('course/chapters/16.md','16.12'), ('course/first-steps.md','W.2')]:
    current = section(path, sec)
    prerequisites[sec] = {'source': path+'#'+sec, 'section_sha256': digest(current.encode())}
code_16_5 = re.findall(r'```python\n(.*?)```', section('course/chapters/16.md','16.5'), re.S)
function_code = next(c for c in code_16_5 if 'def attention_mask(' in c)
from_text = {}; exec(function_code, from_text)
assert torch.equal(from_text['attention_mask'](ns['position'],ns['position'],segments=ns['group'][None]),
                   attention_mask(ns['position'],ns['position'],segments=ns['group'][None]))
assert inspect.getsource(attention_mask).strip() in function_code
prerequisites['16.5']['attention_mask_embedded_matches_current_implementation'] = True

code_16_9 = re.findall(r'```python\n(.*?)```', section('course/chapters/16.md','16.9'), re.S)[0]
state_2, output_2 = run(code_16_9)
code_16_9_one = code_16_9.replace('range(0, 3, 2)', 'range(0, 3, 1)').replace('start + 2', 'start + 1')
state_1, output_1 = run(code_16_9_one)
value_2 = (state_2['numerator']/state_2['denominator']).item()
value_1 = (state_1['numerator']/state_1['denominator']).item()
assert round(value_1,4) == round(value_2,4) == 24.2857
assert output_2.count('分母/分子') == 2 and output_1.count('分母/分子') == 3

# 依 SVG 座標逐格重建三張表，與獨立列舉規則及教材結果比對。
figure_path = 'course/figures/rewrite-16-13-visible-pairs.svg'
figure_raw = (ROOT / figure_path).read_bytes()
assert digest(figure_raw) == page['figures_sha256'][figure_path]
svg = ET.fromstring(figure_raw)
rects = svg.findall('{http://www.w3.org/2000/svg}rect')
figure_matrices = {}
for name, base_y in [('causal',148),('sliding',508),('block_local',868)]:
    matrix = []
    for q in range(6):
        row = []
        for k in range(6):
            matches = [r for r in rects if r.get('x') == str(69+38*k) and r.get('y') == str(base_y+38*q)]
            assert len(matches) == 1
            assert matches[0].get('width') == matches[0].get('height') == '34'
            fill = matches[0].get('fill'); assert fill in ('#107b6e','#e2e8f0')
            row.append(fill == '#107b6e')
        matrix.append(row)
    assert matrix == expected[name]
    figure_matrices[name] = [''.join('1' if b else '0' for b in row) for row in matrix]

nbraw = (ROOT/page['notebook']).read_bytes()
nb = json.loads(nbraw)
nb_code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
assert any(code.strip() == c.strip() for c in nb_code)
assert digest(nbraw) == page['notebook_sha256']
result = {
    'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
    'reviewer_task': '/root/omni_technical_coordinator/technical_16_13',
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_16_13-v1/verify.py',
    'environment': {'python':sys.version.split()[0], 'torch':torch.__version__, 'torch_git':torch.version.git_version,
                    'device':'cpu', 'platform':platform.platform()},
    'freeze': {'path':freeze_path, 'sha256':digest((ROOT/freeze_path).read_bytes())},
    'source_sha256': page['source_sha256'],
    'embedded_code_sha256': digest(code.encode()),
    'main_stdout':stdout,
    'tensor_contract': {'position_dtype':str(ns['position'].dtype), 'mask_dtype':str(ns['causal'].dtype),
                        'helper_shape':[1,1,6,6], 'sliced_shape':[6,6], 'dense_elements_per_mask':36,
                        'reduction_dtype':str(ns['causal'].sum().dtype), 'groups':ns['group'].tolist()},
    'independent_cell_checks': {'checked_cells':108, 'matched':True, 'rows':figure_matrices},
    'variants':variants,
    'current_prerequisites':prerequisites,
    '16.9_current_embedded_tiling': {'block_2_stdout':output_2, 'block_1_stdout':output_1,
                                   'block_2_result':value_2, 'block_1_result':value_1,
                                   'tolerance': 'rounded to 4 decimals; no bitwise equality claim'},
    'notebook': {'path':page['notebook'], 'sha256':digest(nbraw), 'embedded_block_matches':True},
    'result':'PASS: exact embedded block, all three exercise variants, current 16.5 helper, 16.9 block sizes 2/1 and 108 SVG cells match.',
    'limits':'Deterministic CPU mask/count and scalar demonstrations only; no GPU kernel, timing, memory benchmark or model-quality claim.'
}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
