"""Bounded, source-first first-steps review; no installation or training."""
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path('/tmp/tpb').resolve()
OUT = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-first-steps'
TASK = '/root/c_v3_technical_coord/canonical_first_steps'
COMMAND = ('UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb '
           'UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python '
           'docs/technical-reviews/artifacts/c-v3-v21-canonical-first-steps/probe-v1.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(source, label, namespace):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(source, label, 'exec'), namespace)
    return stream.getvalue()


os.chdir(ROOT)
torch.set_num_threads(1)
torch.manual_seed(42)
started = datetime.now(UTC).isoformat()
body = (ROOT / 'course/first-steps.md').read_text()
namespace = {'__name__': '__main__'}
blocks = []
for number, match in enumerate(re.finditer(r'^```python\n(.*?)^```', body, re.M | re.S), 1):
    code = match.group(1)
    line = body[:match.start()].count('\n') + 1
    blocks.append({'block': number, 'line': line, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
                   'stdout': execute(code, f'course/first-steps.md:{line}', namespace)})
assert len(blocks) == 8, len(blocks)

scores = torch.tensor([[60.,70.,80.],[80.,90.,100.]])
changed = scores.clone()
changed[0,2] = 110
assert scores.mean(1).tolist() == [70.,90.]
assert scores.mean(0).tolist() == [70.,80.,90.]
assert changed.mean(1).tolist() == [80.,90.]
assert changed.mean(0).tolist() == [70.,80.,105.]
tensor_results = {'shape': list(scores.shape), 'first_row': scores[0].tolist(),
                  'item': scores[1,2].item(), 'mean_dim1': scores.mean(1).tolist(),
                  'mean_dim0': scores.mean(0).tolist(), 'mean_negative_dim': scores.mean(-1).tolist(),
                  'changed_mean1': changed.mean(1).tolist(), 'changed_mean0': changed.mean(0).tolist(),
                  'reshape': scores.reshape(6).tolist(),
                  'recovered': scores.reshape(6).reshape(2,3).tolist(),
                  'btd_index_shape': list(torch.arange(24).reshape(2,4,3)[0].shape)}
failures = {}
for label, operation in (
    ('item_many', lambda: scores.item()),
    ('reshape_eight', lambda: scores.reshape(8)),
    ('matmul_inner_mismatch', lambda: torch.zeros(2,3) @ torch.zeros(2,2)),
    ('list_index', lambda: ['貓','狗'][2]),
    ('dictionary_key', lambda: {'貓':0}['狗']),
    ('missing_file', lambda: (OUT/'file-that-does-not-exist').read_text()),
    ('log_zero', lambda: math.log(0)),
    ('syntax', lambda: compile('for x in [1]\n    print(x)', '<bad-syntax>', 'exec')),
):
    try:
        operation()
    except Exception as error:
        failures[label] = {'class': type(error).__name__, 'message': str(error)}
    else:
        raise AssertionError(f'expected failure: {label}')

animals = ['貓','狗']
record = {'animal':'貓','caption':'貓在睡覺'}
record['caption'] = '新的說明'
scores_list = [5,9,7]
syntax_examples = {
    'slice_exclude_last':[5,2,7][:-1], 'slice_tail':[5,2,7][1:], 'range':list(range(3)),
    'set_count':len(set('貓狗貓')), 'sorted':sorted([3,1,2]),
    'reverse_sorted':sorted([3,1,2],reverse=True), 'unicode':[ord('狗'),ord('貓')],
    'unicode_sorted':sorted(set('貓狗貓')), 'key_len':max(['貓','小花貓'],key=len),
    'lambda':max(range(3),key=lambda i:scores_list[i]), 'zip':list(zip(animals,[3,1])),
    'dict_zip':dict(zip(animals,[0,1])), 'enumerate':list(enumerate(animals)),
    'reversed':list(reversed(['貓','狗','鳥'])), 'join':'' .join(['貓','看','狗']),
    'join_separator':'、'.join(animals), 'list_comprehension':[x*2 for x in [1,2,3]],
    'generator_sum':sum(x*2 for x in [1,2,3]), 'next':next(x for x in [3,8,12] if x>5),
    'next_default':next((x for x in [3] if x>5),None), 'f_string':f'看到{animals[0]}',
    'floor_positive':7//2, 'floor_negative':-7//2, 'dict_change_keeps_list':animals,
    'duplicate_key_replaces':{'cat':1,'cat':2},
}
exercise_animals = ['鳥','貓','狗']
assert len(exercise_animals)==3
syntax_examples['exercise_stdout'] = execute(
    'print(animals[0],len(animals))\nfor animal in animals:\n    print("看到",animal)\nassert len(animals)==3\n',
    '<W.2 exercise>', {'animals':exercise_animals})
syntax_examples['first_function'] = execute('def first(items):\n    return items[0]\nprint(first(["貓","狗"]))\n', '<function>', {})

recipes = torch.tensor([[2.,1.,0.],[1.,2.,1.]])
weights = torch.tensor([[10.,0.],[20.,1.],[30.,4.]])
layer = torch.nn.Linear(3,2)
with torch.no_grad():
    layer.weight.copy_(weights.T)
    layer.bias.copy_(torch.tensor([5.,0.]))
linear = {'matmul':(recipes@weights).tolist(), 'shape':list((recipes@weights).shape),
          'weight_shape':list(layer.weight.shape), 'bias5':layer(recipes).tolist()}
with torch.no_grad():
    layer.bias.copy_(torch.tensor([8.,0.]))
linear['bias8'] = layer(recipes).tolist()
assert linear['bias8'] == [[48.,1.],[88.,6.]]

gradients=[]
for rate in (0.1,0.6,1.1):
    w=torch.tensor(1.,requires_grad=True)
    loss=(w-3).square()
    loss.backward()
    before={'w':w.item(),'loss':loss.item(),'gradient':w.grad.item()}
    with torch.no_grad():
        w-=rate*w.grad
    after={'w':w.item(),'loss':(w-3).square().item()}
    assert abs(after['w']-(1+4*rate)) < 1e-6
    assert abs(after['loss']-(1+4*rate-3)**2) < 2e-6
    gradients.append({'rate':rate,'before':before,'after':after})
w=torch.tensor(1.,requires_grad=True)
(w-3).square().backward()
first_grad=w.grad.item()
(w-3).square().backward()
accumulated=w.grad.item()
w.grad=None
(w-3).square().backward()
assert (first_grad,accumulated,w.grad.item())==(-4.,-8.,-4.)
log_costs=[{'p':p,'cost':-math.log(p),'rounded':round(-math.log(p),3)} for p in (0.9,0.5,0.1,0.25)]
assert [x['rounded'] for x in log_costs] == [0.105,0.693,2.303,1.386]

notebook=ROOT/'notebooks/01/1.1.ipynb'
nb=json.loads(notebook.read_text())
nb_runs=[]
for edited in (False,True):
    ns={'__name__':'__main__'}
    outputs=[]
    for index,cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code' or cell.get('metadata',{}).get('course_figure'):
            continue
        code=''.join(cell['source'])
        if edited:
            code=code.replace('text = "貓看狗，狗看貓。"','text = "鳥看狗，狗看鳥。"')
        outputs.append({'cell':index,'stdout':execute(code,f'1.1.ipynb:cell{index}',ns)})
    nb_runs.append({'edited':edited,'outputs':outputs})
assert nb_runs[1]['outputs'][-1]['stdout'].strip().endswith('鳥看狗，狗看鳥。')

root_paths={'cwd':str(Path.cwd()),'pyproject':Path('pyproject.toml').exists(),'package':Path('tiny_perceptron').exists()}
os.chdir(ROOT/'notebooks')
notebook_paths={'cwd':str(Path.cwd()),'pyproject':Path('pyproject.toml').exists(),'package':Path('tiny_perceptron').exists()}
os.chdir(ROOT)
assert root_paths['pyproject'] and root_paths['package']
assert not notebook_paths['pyproject'] and not notebook_paths['package']
env_check=subprocess.run([sys.executable,'scripts/check_env.py'],cwd=ROOT,capture_output=True,text=True,check=True)

render=[]
for stem in ('rewrite-W-3-axes','rewrite-W-6-gradient-step'):
    source=ROOT/f'course/figures/{stem}.svg'
    target=OUT/f'{stem}-render-v1.png'
    args=['inkscape',str(source),'--export-type=png',f'--export-filename={target}','--export-width=640']
    run=subprocess.run(args,capture_output=True,text=True,check=True)
    render.append({'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),
                   'output':str(target.relative_to(ROOT)),'sha256':sha(target),
                   'command':args,'stdout':run.stdout,'stderr':run.stderr})

inputs=['course/first-steps.md','pyproject.toml','scripts/check_env.py','scripts/build_course.py',
        'notebooks/01/1.1.ipynb','.gitignore','course/lesson-index.json',
        'docs/course-revision-20261101-c-v3/reader-phase-proof-v21-v2.json',
        'docs/course-revision-20261101-c-v3/technical-stage-open-v21-v1.json',
        'docs/course-revision-20261101-c-v3/source-freeze-v21.json',
        'docs/course-revision-20261101-c-v3/numbered-source-freeze-v21.json']
result={'schema_version':1,'reviewer_task':TASK,'started_at':started,'completed_at':datetime.now(UTC).isoformat(),
        'command':COMMAND,'environment':{'python':platform.python_version(),'torch':str(torch.__version__),
        'device':'cpu','platform':platform.platform(),'inkscape':subprocess.check_output(['inkscape','--version'],text=True).strip()},
        'inputs_sha256':{x:sha(ROOT/x) for x in inputs},'python_blocks':blocks,'syntax_examples':syntax_examples,
        'tensor':tensor_results,'expected_errors':failures,'log_costs':log_costs,'linear':linear,
        'gradient_steps':gradients,'gradient_accumulation':[first_grad,accumulated,w.grad.item()],
        'notebook_1_1':{'sha256':sha(notebook),'method':'fresh Python namespace per version; local bootstrap executed; figure display cells skipped; not Jupyter UI or Colab execution','runs':nb_runs},
        'paths':{'root':root_paths,'notebooks':notebook_paths},
        'check_env':{'command':[sys.executable,'scripts/check_env.py'],'returncode':env_check.returncode,'stdout':env_check.stdout,'stderr':env_check.stderr},
        'figure_renders':render,
        'scope':'Only first-steps snippets and stated local CPU examples. No install, clone, GUI/kernel registration, Windows, macOS, Colab, all-book output audit, model training or product validation.'}
with (OUT/'execution-v1.json').open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(json.dumps({'artifact':str((OUT/'execution-v1.json').relative_to(ROOT)), 'sha256':sha(OUT/'execution-v1.json'),
                  'python_blocks':len(blocks),'result':'all bounded assertions passed','environment':result['environment']},ensure_ascii=False))
