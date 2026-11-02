"""Independent T.5 audit: bounded CPU only, no complete recipe, GPU, download or publication."""
from collections import Counter
from datetime import datetime, UTC
from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = Path('/workspace/work/omni-technical-t-5-v1')
WORK.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
sys.path.insert(0, str(ROOT))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def execute(args, timeout=50):
    command = ['uv', 'run', '--extra', 'cpu', 'python', *map(str, args)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}, timeout=timeout)
    item = {'command': shlex.join(command), 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}
    commands.append(item)
    if result.returncode:
        raise RuntimeError(item)
    return result.stdout

commands = []
report = {'created_at': datetime.now(UTC).isoformat(),
          'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                          'device': 'cpu', 'threads': '2', 'platform': platform.platform()},
          'boundary': 'Only one update per generated model, plus dry-run, inference and source/CLI audits. '
                      'No 500-step training, fixed recipe execution, GPU, Modal, HF access or publication.'}

freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json'
page = next(x for x in json.loads(freeze_path.read_text())['pages'] if x['page_id'] == 'T.5')
snapshot = ROOT / page['snapshot']
report['freeze'] = {'path': str(freeze_path.relative_to(ROOT)), 'sha256': sha(freeze_path),
                    'snapshot': page['snapshot'], 'snapshot_sha256': sha(snapshot),
                    'matches': sha(snapshot) == page['source_sha256'], 'figures': page['figures_sha256']}

for script in ['scripts/prepare_data.py', 'scripts/train.py', 'scripts/evaluate.py', 'scripts/infer.py',
               'scripts/fetch_training_assets.py']:
    execute([script, '--help'])
execute(['-m', 'scripts.course_experiments.run', '--help'])
execute(['-m', 'scripts.course_experiments.run', '--list-assets', 'style'])
execute(['-m', 'scripts.course_experiments.run', '--list-assets', 'safety'])
execute(['-m', 'scripts.course_experiments.run', '--list-assets', 'lora'])

from scripts.prepare_data import generate_records
from scripts.train import parser, prepare_examples
from scripts.course_experiments.behavior import _safety_records, _style_record
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import ByteTokenizer, render_chat

report['generated'] = {}
for kind in ['style', 'safety']:
    execute(['scripts/prepare_data.py', '--kind', kind, '--output', WORK / 'generated'])
    directory = WORK / 'generated' / kind
    manifest = json.loads((directory / 'manifest.json').read_text())
    parts = {split: [json.loads(line) for line in (directory / f'{split}.jsonl').read_text().splitlines()]
             for split in ['train', 'validation', 'test']}
    family_sets = {s: set(r['family'] for r in rows) for s, rows in parts.items()}
    examples = []
    if kind == 'style':
        examples = [r for rows in parts.values() for r in rows if r['family'] == '2+3']
        classes = Counter(r['style'] for rows in parts.values() for r in rows)
    else:
        examples = [r for rows in parts.values() for r in rows if r['family'] == '0']
        classes = Counter(str(r['permission']) for rows in parts.values() for r in rows)
    settings = ['--task', 'sft', '--data', str(directory / 'train.jsonl'), '--train', '--steps', '500',
                '--max-length', '256', '--output', str(WORK / f'{kind}.pt')]
    args = parser().parse_args(settings)
    parsed = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    rendered = prepare_examples(args, ByteTokenizer())
    report['generated'][kind] = {'manifest': manifest, 'classes': dict(classes), 'examples': examples,
        'family_disjoint': all(not family_sets[a] & family_sets[b] for a,b in
                               [('train','validation'), ('train','test'), ('validation','test')]),
        'parsed_textbook_training': parsed, 'rendered_train_examples': len(rendered),
        'max_rendered_train_length': max(len(x) for x,y in rendered),
        'effective_train_answer_tokens': sum(int((y != -100).sum()) for x,y in rendered)}
    # Preserve the 500-step schedule and all model defaults; one actual update only.
    execute(['scripts/train.py', *settings, '--device', 'cpu', '--stop-after', '1'])
    execute(['scripts/evaluate.py', WORK / f'{kind}.pt', '--data', directory / 'validation.jsonl',
             '--mode', 'sft', '--tokens', '128', '--limit', 'all', '--device', 'cpu',
             '--output', WORK / f'{kind}-validation.json'])
    evaluation = json.loads((WORK / f'{kind}-validation.json').read_text())
    report['generated'][kind]['short_evaluation'] = {k:v for k,v in evaluation.items() if k != 'samples'}
    report['generated'][kind]['short_evaluation']['first_sample'] = evaluation['samples'][0]
    report['generated'][kind]['short_checkpoint_sha256'] = sha(WORK / f'{kind}.pt')
    report['generated'][kind]['short_evaluation_sha256'] = sha(WORK / f'{kind}-validation.json')

execute(['scripts/train.py', '--task', 'sft', '--data', WORK / 'generated/style/train.jsonl',
         '--steps', '500', '--max-length', '256', '--device', 'cpu'])
execute(['scripts/infer.py', WORK / 'style.pt', '--chat', '--prompt', 'style=vivid; 2+3=?',
         '--tokens', '128', '--device', 'cpu', '--json'])

fixed_arithmetic = split_records(arithmetic_records(), seed=42)
fixed_safety = split_records(_safety_records(), seed=42)
report['fixed_data_source_audit'] = {
    'arithmetic': {s: {'records':len(v), 'families': len(set(r['family'] for r in v))} for s,v in fixed_arithmetic.items()},
    'safety': {s: {'records':len(v), 'families':len(set(r['family'] for r in v)),
                  'kinds': dict(Counter(r['kind'] for r in v))} for s,v in fixed_safety.items()},
    'conditional_example': [_style_record({'a':2,'b':3,'family':'2+3'}, style, True)
                            for style in ['concise','vivid','json']],
    'scope': 'Data generation only; did not invoke run_style, run_safety or run_lora.'}

report['historical'] = {}
for name in ['style', 'safety', 'lora']:
    path = ROOT / f'docs/course-experiments/results/{name}.json'
    r = json.loads(path.read_text())
    selected = {k:r.get(k) for k in ['experiment_id','revision','device','seed','torch_version','python_version',
                 'gpu','elapsed_seconds','timing_scope','step_scale','evidence_status','assets','public_exports','hf']}
    selected['path'] = str(path.relative_to(ROOT)); selected['sha256'] = sha(path)
    selected['rounded_seconds'] = round(r['elapsed_seconds'],1)
    selected['run_code'] = []
    for code_path in ['scripts/course_experiments/behavior.py','scripts/course_experiments/common.py',
                      'scripts/course_experiments/run.py','tiny_perceptron/model.py']:
        got = subprocess.run(['git','show',f'{r["revision"]}:{code_path}'],cwd=ROOT,capture_output=True)
        entry = {'path':code_path,'git_show_returncode':got.returncode,
                 'recorded_sha256':r.get('code_sha256',{}).get(code_path)}
        if got.returncode == 0:
            history_path=WORK / f'history-{name}-{code_path.replace("/","-")}'
            history_path.write_bytes(got.stdout)
            digest=hashlib.sha256(got.stdout).hexdigest()
            entry.update(actual_sha256=digest, matches_recorded=digest == entry['recorded_sha256'],
                         local_path=str(history_path))
        else:
            entry['error']=got.stderr.decode(errors='replace')
        selected['run_code'].append(entry)
    results = r['results']
    if name == 'style':
        selected['configuration_and_denominators']={k:results[k] for k in ['arithmetic_data','content_training','data','training']}
        selected['configuration_and_denominators']['default_style_training']={s:v['training'] for s,v in results['default_style_runs'].items()}
        selected['evaluations']={s:{k:v for k,v in e.items() if k not in ['samples']} for s,e in results['after'].items()}
    elif name == 'safety':
        selected['configuration_and_denominators']={'data':results['data'],
            'runs': {s:{'training':v['training'], 'safety':{sp:{k:x for k,x in e.items() if k!='samples'} for sp,e in v['safety'].items()},
                       'arithmetic':{sp:{k:x for k,x in e.items() if k!='samples'} for sp,e in v['arithmetic'].items()}}
                     for s,v in results['runs'].items()},
            'pku_pilot':results['pku_pilot'], 'private_only_artifacts':results['private_only_artifacts'],
            'private_only_data':results['private_only_data']}
        selected['pku_text_fields_in_public_report'] = str(any('samples' in e for e in results['pku_pilot']['evaluation'].values()))
    else:
        selected['configuration_and_denominators']={k:results[k] for k in
           ['base_sha256','base_parameters','rank','alpha','scaling','a_b_a_max_difference','merge_max_difference']}
        selected['configuration_and_denominators']['runs']={s:{'training':v['training'],'data':v['data'],
            'layers':v['layers'],'base_parameters_unchanged':v['base_parameters_unchanged']} for s,v in results['runs'].items()}
        selected['configuration_and_denominators']['full_sft_training']=results['full_sft']['training']
    report['historical'][name]=selected

# Adapter loading and merging on an untrained miniature, deterministic weights; no quality claim.
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.adapters import base_state_sha256, load_lora_adapter
from tiny_perceptron.model import ModelConfig, TinyLM
torch.manual_seed(42)
base=TinyLM(ModelConfig(width=8,layers=1,heads=1,max_length=32)).eval()
original=copy.deepcopy(base.state_dict())
layer=LoRALinear(copy.deepcopy(base.output),rank=2,alpha=2)
with torch.no_grad():
    layer.a.fill_(0.02); layer.b.fill_(0.03)
payload={'format_version':'lora-v1','config':asdict(base.config),'base_sha256':base_state_sha256(original),
         'scaling':'alpha/rank','adapter':{'output':{'a':layer.a.detach(),'b':layer.b.detach(),'rank':2,'alpha':2}}}
adapter=WORK/'mini-adapter.pt';torch.save(payload,adapter)
loaded=copy.deepcopy(base)
info=load_lora_adapter(loaded,adapter,base_state=original)
probe=torch.tensor([[1,3,57,2,4]])
merged=copy.deepcopy(base)
with torch.no_grad(): merged.output.weight.copy_(layer.merged_weight())
diff=float((loaded(probe)['logits']-merged(probe)['logits']).abs().max())
errors={}
for label, candidate in [('wrong_base',copy.deepcopy(base)),('merged_base',merged)]:
    if label == 'wrong_base':
        with torch.no_grad(): candidate.output.weight.add_(0.001)
    try:
        load_lora_adapter(candidate,adapter)
        errors[label]='UNEXPECTED SUCCESS'
    except ValueError as error:
        errors[label]=str(error)
report['mini_adapter']={'config':asdict(base.config),'original_base_sha256':base_state_sha256(original),
    'loader_result':info,'merge_max_logit_difference':diff,'reject_results':errors,
    'scope':'Synthetic adapter A=0.02, B=0.03, rank=alpha=2; no training; only verifies loader/merge mechanics.'}

release=json.loads((ROOT/'docs/course-experiments/releases/safety.json').read_text())
public=json.loads((ROOT/'docs/course-experiments/public-models.json').read_text())
report['public_release']={'safety_release_files':[f['path'] for f in release['files']],
    'pku_named_release_files':[f['path'] for f in release['files'] if 'pku' in f['path'].lower()],
    'registry_top_level_type':type(public).__name__,
    'boundary':'Read local release/registry records only; did not contact a model host or download weights.'}
paths=['scripts/prepare_data.py','scripts/train.py','scripts/evaluate.py','scripts/infer.py',
       'scripts/fetch_training_assets.py','scripts/course_experiments/run.py','scripts/course_experiments/behavior.py',
       'scripts/course_experiments/common.py','scripts/course_experiments/text.py','tiny_perceptron/adapters.py',
       'tiny_perceptron/alignment.py','tiny_perceptron/model.py','tiny_perceptron/data.py','tiny_perceptron/training.py',
       'assets/training/manifest.json','assets/training/sources/pku-safe-rlhf.json',
       'docs/course-experiments/releases/safety.json','docs/course-experiments/public-models.json',
       'docs/course-experiments/README.md','docs/course-experiments/release-licenses.md','assets/training/README.md']
report['source_sha256']={p:sha(ROOT/p) for p in paths}
report['commands']=commands
report['completed_at']=datetime.now(UTC).isoformat()
with (HERE/'execution.json').open('x',encoding='utf-8') as stream:
    json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps({'artifact':str(HERE/'execution.json'),'sha256':sha(HERE/'execution.json'),
                  'commands':len(commands),'environment':report['environment'],
                  'generated':{k:v['manifest']['splits'] for k,v in report['generated'].items()},
                  'mini_adapter':report['mini_adapter']},ensure_ascii=False,indent=2))
