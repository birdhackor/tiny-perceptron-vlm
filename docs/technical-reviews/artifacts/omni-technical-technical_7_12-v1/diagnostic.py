"""7.12: execute the frozen snippet and audit saved data/recipe metadata; no training."""
import ast
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.prepare_data import generate_records
from scripts.course_experiments import common, text, run

CACHE = Path('/workspace/work/reference/omni-technical-technical_7_12-v1')
OUTPUT = Path(__file__).with_name('diagnostic-result.json')
COMMAND = 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_7_12-v1/diagnostic.py'
def digest(data):
    return hashlib.sha256(data).hexdigest()

def section(path, identifier):
    raw = (ROOT / path).read_text(encoding='utf-8')
    found = re.search(r'^## ' + re.escape(identifier) + r'\s.*?(?=^## |\Z)', raw, re.M | re.S)
    assert found, (path, identifier)
    return found.group()

snapshot = ROOT / 'docs/course-review-20261019/freeze-01/sources/7.12.md'
body = snapshot.read_text(encoding='utf-8')
expected_sha = 'f8f786938e665040e676e2a0c9f8b283a1526fbe79b50fe3a09a5d934be54f7a'
assert digest(snapshot.read_bytes()) == expected_sha
assert section('course/chapters/07.md', '7.12') == body
snippets = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(snippets) == 1
snippet = snippets[0]
base = {}
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(compile(snippet, str(snapshot) + ':17-22', 'exec'), base)
assert base['all_pairs'] == [('紅','圓'),('紅','方'),('藍','圓'),('藍','方')]
assert base['train'] == [('紅','圓'),('藍','圓'),('藍','方')]
assert base['held_out'] == {('紅','方')}
assert set(base['train']) | base['held_out'] == set(base['all_pairs'])
assert {c for c, _ in base['train']} == {'紅','藍'}
assert {s for _, s in base['train']} == {'圓','方'}
call_names = sorted({node.func.id for node in ast.walk(ast.parse(snippet))
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)})
assert call_names == ['print', 'set']
exercise = {}
exercise_stream = io.StringIO()
exercise_code = snippet.replace('held_out = {("紅", "方")}', 'held_out = {("藍", "圓"), ("藍", "方")}')
assert exercise_code != snippet
with contextlib.redirect_stdout(exercise_stream):
    exec(compile(exercise_code, '<7.12 exercise>', 'exec'), exercise)
assert exercise['train'] == [('紅','圓'),('紅','方')]
assert not any(c == '藍' for c, _ in exercise['train'])
rows = [line.split('|')[1:-1] for line in body.splitlines() if line.startswith('|')][2:]
assert len(rows) == 4
for row in rows:
    _, prompt, answer, _ = [cell.strip() for cell in row]
    assert answer in prompt
    assert answer == ('圓形' if '圓形' in prompt else '方塊')

record_path = ROOT / 'docs/course-experiments/results/sft.json'
record = json.loads(record_path.read_bytes())
records = generate_records('attributes-sft')
parts = common.split_records(records, seed=record['seed'])
assert len(records) == 60
assert len({row['family'] for row in records}) == 12
split_stats = {}
for name, selected in parts.items():
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected).encode()
    stats = {'records':len(selected), 'families':len({r['family'] for r in selected}), 'sha256':digest(encoded)}
    assert all(stats[key] == record['results']['data'][name][key] for key in stats)
    split_stats[name] = stats
for a, b in [('train','validation'),('train','test'),('validation','test')]:
    assert not ({r['family'] for r in parts[a]} & {r['family'] for r in parts[b]})
recipe_calls = []
checkpoint_calls = []
def fake_fit(model, selected, ctx, **kwargs):
    recipe_calls.append(dict(kwargs, records=len(selected), records_sha256=common.records_sha256(selected)))
    return {'steps':kwargs['steps'], 'checkpoint':kwargs['name']+'.pt'}
probe_output = CACHE / 'recipe-boundary-probe'
probe_output.mkdir(exist_ok=False)
ctx = SimpleNamespace(device='cpu', output=probe_output, seed=42, step_scale=1.0)
with patch.object(text,'new_lm',return_value=object()), patch.object(text,'fit_lm',side_effect=fake_fit), patch.object(text,'_evaluations',return_value={}), patch.object(text,'save_checkpoint',side_effect=lambda path,*args,**kw:checkpoint_calls.append(str(path))), patch.object(text,'_natural_chat_pilot',return_value={'scope':'mocked: no optimization, evaluation, download or GPU execution'}):
    recipe = text.run_sft(ctx)
assert [(x['mode'],x['steps'],x['name'],x['records']) for x in recipe_calls] == [('sft',900,'model',45),('text',250,'pretrain',45),('sft',900,'pretrain-sft',45)]
assert recipe_calls[0]['records_sha256'] == record['results']['training']['records_sha256']
assert recipe_calls[1]['records_sha256'] == record['results']['pretrain_then_sft']['pretraining']['records_sha256']
assert recipe_calls[2]['records_sha256'] == record['results']['pretrain_then_sft']['sft']['records_sha256']
assert record['results']['training']['steps'] == 900
assert record['results']['pretrain_then_sft']['pretraining']['steps'] == 250
assert record['results']['pretrain_then_sft']['sft']['steps'] == 900
assert record['seed'] == 42
assert record['gpu'] == 'NVIDIA L4' and record['device'] == 'cuda'
assert f"{record['elapsed_seconds']:.1f}" == '31.7'
assert record['status'] == 'completed' and record['evidence_status'] == 'complete_run'

cli_calls = []
def fake_execute(*args, **kwargs):
    cli_calls.append({'experiment_id':args[0], 'device':args[1], 'output':str(args[2]), 'dependencies':str(args[3]), 'assets':str(args[4]), **kwargs})
    return {'experiment_id':'sft','elapsed_seconds':0.0,'evidence_status':'review_cli_boundary_only','artifacts':[]}
with patch.object(run,'execute',side_effect=fake_execute), patch.object(sys,'argv',['run','--experiment','sft','--device','cuda']), contextlib.redirect_stdout(io.StringIO()):
    run.main()
assert cli_calls[0]['output'] == str(ROOT / 'outputs/course-experiments/course-v1/sft')
assert cli_calls[0]['device'] == 'cuda' and cli_calls[0]['step_scale'] == 1.0
list_assets = subprocess.run([sys.executable,'-m','scripts.course_experiments.run','--list-assets','sft'],cwd=ROOT,text=True,capture_output=True,check=True)
assert list_assets.stdout.strip() == 'assets/training/ultrachat-sft-v1.tar.gz'
fetch_spec = importlib.util.spec_from_file_location('review_fetch', ROOT/'scripts/fetch_training_assets.py')
fetch = importlib.util.module_from_spec(fetch_spec)
fetch_spec.loader.exec_module(fetch)
fetch_calls = []
with patch.object(sys,'argv',['fetch','--asset','ultrachat-sft']), patch.object(fetch.subprocess,'run',side_effect=lambda *args,**kwargs:fetch_calls.append({'kind':'git_lfs_boundary_only','argv':args[0]})), patch.object(fetch,'unpack_asset',side_effect=lambda archive,asset,destination:fetch_calls.append({'kind':'unpack_boundary_only','archive':str(archive),'asset':asset['id'],'destination':str(destination)}) or {'scope':'review boundary probe; no delivery verification'}), contextlib.redirect_stdout(io.StringIO()):
    fetch.main()
assert fetch_calls[-1]['asset'] == 'ultrachat-sft'
assert fetch_calls[-1]['destination'] == str(ROOT / 'data/training')

historical_run = subprocess.check_output(['git','show',record['revision']+':scripts/course_experiments/run.py'],cwd=ROOT)
assert digest(historical_run) == record['code_sha256']['scripts/course_experiments/run.py']
file_paths = ['scripts/prepare_data.py','scripts/course_experiments/common.py','scripts/course_experiments/text.py','scripts/course_experiments/run.py','scripts/fetch_training_assets.py','tiny_perceptron/assets.py','tiny_perceptron/data.py','assets/training/manifest.json','docs/course-experiments/results/sft.json']
prereqs = []
for path, ident, quote in [('course/chapters/07.md','7.11','for a in range(4)'),('course/chapters/01.md','1.2','兩個集合的 `&` 是交集'),('course/chapters/05.md','5.12','切分時整個家族放在同一邊'),('course/training.md','T.4','fixed-sft-recipe')]:
    selected = section(path,ident)
    assert quote in selected
    prereqs.append({'source':path+'#'+ident, 'section_sha256':digest(selected.encode()), 'checked_quote':quote,'svg_references':re.findall(r'!\[[^\]]*\]\(([^)]+\.svg)\)',selected)})
result = {
 'created_at':datetime.datetime.now(datetime.UTC).isoformat(),
 'reviewer_task':'/root/omni_technical_coordinator/technical_7_12',
 'command':COMMAND,
 'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'platform':platform.platform(),'device':'cpu','training':'none'},
 'source_sha256':expected_sha,
 'inline_code_sha256':digest(snippet.encode()),
 'inline_stdout':stream.getvalue(),
 'all_pairs':base['all_pairs'],'train':base['train'],'held_out':sorted(base['held_out']),
 'base_counts':{'all':4,'train':3,'held_out':1},
 'table_answers_are_present_in_prompts':True,
 'inline_call_names':call_names,
 'exercise_stdout':exercise_stream.getvalue(),
 'exercise_train':exercise['train'],'exercise_held_out':sorted(exercise['held_out']),
 'base_attribute_coverage':{'colors':sorted({c for c,_ in base['train']}),'shapes':sorted({s for _,s in base['train']})},
 'historical_record_sha256':digest(record_path.read_bytes()),
 'recreated_splits_match_recorded_jsonl_sha256':split_stats,
 'families_total':12,'records_total':60,
 'recipe_boundary_probe':{'scope':'actual run_sft with model creation, training, evaluation, checkpoint writes and UltraChat pilot mocked; no training result','calls':recipe_calls,'checkpoint_paths':checkpoint_calls},
 'historic_run':{k:record[k] for k in ['revision','seed','device','gpu','python_version','torch_version','elapsed_seconds','timing_scope','evidence_status','status']},
 'rounded_elapsed_seconds':f"{record['elapsed_seconds']:.1f}",
 'historical_run_py_sha256':digest(historical_run),
 'historical_run_py_cache_path':str(CACHE/'historical-course-experiments-run.py'),
 'cli_boundary_only':cli_calls,'list_assets_stdout':list_assets.stdout,
 'fetch_boundary_only':fetch_calls,
 'current_prerequisite_sections':prereqs,
 'current_file_sha256':{path:digest((ROOT/path).read_bytes()) for path in file_paths},
 'verification_scope':'Exact frozen inline Python and exercise executed; pure data generation and CLI/recipe boundaries audited. Historical runtime metadata read and arithmetically checked, not reproduced. No model training, no GPU and no paid work.'
}
with OUTPUT.open('x',encoding='utf-8') as handle:
    json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
print(json.dumps({'output':str(OUTPUT.relative_to(ROOT)),'sha256':digest(OUTPUT.read_bytes()),'inline_stdout':result['inline_stdout'],'base_counts':result['base_counts'],'exercise_train':result['exercise_train'],'historic_splits':split_stats,'recipe_steps':[x['steps'] for x in recipe_calls],'rounded_elapsed_seconds':result['rounded_elapsed_seconds'],'status':'all assertions passed; no training'},ensure_ascii=False,indent=2))
