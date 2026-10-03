"""1.12 的有界 CPU 原例、兩練習、失敗反例與固定報告核對；不執行訓練。"""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import torch
from tiny_perceptron.data import split_documents, toy_documents
from scripts.train_simple import examples

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/technical-reviews/artifacts/1.12-execution.json'

def section(path, lesson):
    raw = (ROOT / path).read_bytes()
    heads = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
    selected = next(i for i,h in enumerate(heads) if h[0].startswith(('## '+lesson+' ').encode()))
    end = heads[selected+1].start() if selected+1 < len(heads) else len(raw)
    return raw[heads[selected].start():end]

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def first_code(raw):
    return re.search(rb'```python\n(.*?)```', raw, flags=re.S)[1].decode()

body = section('course/chapters/01.md', '1.12')
original = first_code(body)
results = []
for name, code, expected_error in (
    ('原例：η=0.1', original, None),
    ('練習：η=0.6', original.replace('0.1 * w.grad', '0.6 * w.grad'), None),
    ('練習：η=2（預期 AssertionError）', original.replace('0.1 * w.grad', '2 * w.grad'), 'AssertionError'),
    ('移除 no_grad（預期 RuntimeError）', original.replace('with torch.no_grad():\n    w -= 0.1 * w.grad', 'w -= 0.1 * w.grad'), 'RuntimeError'),
):
    ns = {}
    stdout = io.StringIO()
    error = None
    with redirect_stdout(stdout):
        try:
            exec(compile(code, name, 'exec'), ns)
        except (AssertionError, RuntimeError) as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
    actual_error = None if error is None else error['type']
    assert actual_error == expected_error, (name, error)
    w = ns['w']
    row = {'case': name, 'code_sha256': digest(code.encode()), 'stdout': stdout.getvalue(),
           'expected_exception': expected_error, 'observed_exception': error,
           'w': w.item(), 'w_grad': w.grad.item(), 'w_requires_grad': w.requires_grad,
           'w_is_leaf': w.is_leaf, 'w_dtype': str(w.dtype), 'before': ns['before'].item(),
           'grad_mode_after': torch.is_grad_enabled(),
           'parameter_and_gradient_storage_are_distinct': w.data_ptr() != w.grad.data_ptr()}
    if 'after' in ns:
        row.update(after=ns['after'].item(), after_requires_grad=ns['after'].requires_grad,
                   after_grad_fn=type(ns['after'].grad_fn).__name__)
    results.append(row)

# 有界 API 觀察：求導、取出純 Python 數值、原地更新各自只做一次。
w = torch.tensor(1.0, requires_grad=True)
loss = (w - 3).square()
parameter_id, parameter_storage, value_before = id(w), w.data_ptr(), w.item()
loss.backward()
item = w.item()
api = {'before_backward': value_before, 'after_backward': w.item(), 'gradient': w.grad.item(),
       'item_python_type': type(item).__name__, 'after_item': w.item()}
with torch.no_grad():
    w -= 0.1 * w.grad
api.update(same_python_object_after_update=id(w) == parameter_id,
           same_storage_after_update=w.data_ptr() == parameter_storage,
           old_loss_after_parameter_update=loss.item(),
           new_loss_after_parameter_update=(w - 3).square().item(),
           gradient_after_update=w.grad.item())
assert api['before_backward'] == api['after_backward'] == api['after_item'] == 1.0
assert api['same_python_object_after_update'] and api['same_storage_after_update']
assert api['old_loss_after_parameter_update'] == 4.0

# 只執行直接前置 1.11 原例，確認 backward 後參數及梯度分開。
prereq = section('course/chapters/01.md', '1.11')
ns = {}
stdout = io.StringIO()
with redirect_stdout(stdout):
    exec(compile(first_code(prereq), '1.11 原例', 'exec'), ns)
pre = {'stdout': stdout.getvalue(), 'w': ns['w'].tolist(), 'gradient': ns['w'].grad.tolist(),
       'shape': list(ns['w'].shape), 'gradient_shape': list(ns['w'].grad.shape),
       'distinct_storage': ns['w'].data_ptr() != ns['w'].grad.data_ptr()}
assert pre['w'] == [1.0, 2.0] and pre['gradient'] == [-4.0, 2.0]

# T.3 的內建資料及平均分母；只建短文、窗口，不更新模型或下載資料。
parts = split_documents(toy_documents(), seed=42)
vocabulary = {c:i+2 for i,c in enumerate(sorted(set(''.join(parts['train']))))}
data = {}
for name, docs in parts.items():
    x, y = examples(docs, vocabulary, 1)
    data[name] = {'documents': docs, 'document_count': len(docs), 'next_character_targets': int(y.numel()),
                  'input_shape': list(x.shape), 'target_shape': list(y.shape)}
assert [data[k]['document_count'] for k in ('train','validation','test')] == [9,1,2]
assert [data[k]['next_character_targets'] for k in ('train','validation','test')] == [103,11,22]
fixed_path = ROOT / 'docs/course-experiments/results/simple_models.json'
fixed_raw = fixed_path.read_bytes()
fixed = json.loads(fixed_raw)
bigram = fixed['results']['runs']['bigram']
fixed_check = {'path': str(fixed_path.relative_to(ROOT)), 'sha256': digest(fixed_raw),
               'revision': fixed['revision'], 'device': fixed['device'], 'seed': fixed['seed'],
               'torch_version': fixed['torch_version'], 'python_version': fixed['python_version'],
               'steps': bigram['steps'], 'train_document_count': fixed['results']['data']['train']['records'],
               'before_train_nll': bigram['before_nll']['train'],
               'after_train_nll': bigram['after_nll_same_post_update_time']['train'],
               'rounded_4dp': [f"{bigram['before_nll']['train']:.4f}", f"{bigram['after_nll_same_post_update_time']['train']:.4f}"],
               'generation_samples': bigram['samples'],
               'interpretation': '只讀固定正式報告，沒有重跑正式實驗或訓練；數值是逐下一字目標平均 NLL。'}
assert fixed_check['rounded_4dp'] == ['3.7733','1.0568']
assert fixed_check['steps'] == 200 and fixed_check['train_document_count'] == 9

# 實際閱讀的本節及必要前置均無 SVG 引用，故沒有待 render 的圖。
svg_scan = {}
for path, lesson in [('course/chapters/01.md','1.12'),('course/chapters/01.md','1.11'),('course/training.md','T.3')]:
    raw = section(path,lesson)
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+\.svg)\)|(?:src|href)=["\']([^"\']+\.svg)["\']',raw.decode())
    svg_scan[path+'#'+lesson] = {'section_sha256':digest(raw),'embedded_svg_references':refs}
    assert not refs

env = {'python':sys.version, 'python_executable':sys.executable, 'torch':torch.__version__,
       'torch_git_version':torch.version.git_version, 'device':'cpu', 'cuda_build':str(torch.version.cuda),
       'cuda_available':str(torch.cuda.is_available()), 'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS',''),
       'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS',''), 'torch_threads':str(torch.get_num_threads())}
record = {'reviewed_on':'2026-10-03','command':'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/1.12-verify.py',
          'environment':env,'original_section_sha256':digest(body),'cases':results,'api_checks':api,
          'prerequisite_1_11':pre,'toy_document_split_and_denominators':data,'fixed_formal_report_check':fixed_check,
          'svg_scan':svg_scan,'result':'全部有界檢查通過；η=2 的 AssertionError 和移除 no_grad 的 RuntimeError 均按預期出現。'}
OUT.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'result':record['result'],'environment':env,'cases':results,'fixed_formal_report_check':fixed_check},ensure_ascii=False,indent=2))
