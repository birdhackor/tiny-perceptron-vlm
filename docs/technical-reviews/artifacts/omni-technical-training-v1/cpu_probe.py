"""Bounded, independent review of the frozen training page. No formal training."""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SCRATCH = Path('/workspace/work/omni-technical-training-v1')
SNAPSHOT = REPO / 'docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04/sources/training.md'
SCRATCH.mkdir(exist_ok=True)
for folder in ('outputs', 'checkpoints'):
    (SCRATCH / folder).mkdir(exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
rows = []


def run(label, args, allowed=(0,), timeout=40):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=SCRATCH, env=env,
                            text=True, capture_output=True, timeout=timeout)
    row = {'label': label, 'command': [sys.executable, *map(str, args)],
           'cwd': str(SCRATCH), 'returncode': result.returncode,
           'stdout': result.stdout, 'stderr': result.stderr}
    rows.append(row)
    if result.returncode not in allowed:
        raise RuntimeError(f'{label}: {result.returncode}: {result.stderr}')
    print(label, result.returncode, flush=True)
    return result


def script(name, *args):
    return [REPO / 'scripts' / name, *args]


body = SNAPSHOT.read_text()
python_blocks = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(python_blocks) == 2
for name, content in zip(('inline_save_start.py', 'inline_size_check.py'), python_blocks, strict=True):
    target = HERE / name
    if target.exists():
        assert target.read_text() == content
    else:
        target.write_text(content)

try:
    run('assets-list', script('fetch_training_assets.py', '--list'))
    run('models-list', script('fetch_course_models.py', '--list'))
    for task in ('text', 'sft', 'vision'):
        run(f'dry-{task}', script('train.py', '--task', task, '--device', 'cpu'))
    for model in ('bigram', 'mlp'):
        extra = [] if model == 'bigram' else ['--context', '3', '--width', '16']
        run(f'{model}-before', script('train_simple.py', '--model', model, *extra, '--seed', '42', '--device', 'cpu'))
        run(f'{model}-after', script('train_simple.py', '--model', model, *extra, '--seed', '42', '--device', 'cpu',
                                    '--train', '--steps', '200', '--output', f'checkpoints/{model}.pt'))
    for kind in ('toy-text', 'attributes-sft', 'style', 'safety', 'preference'):
        run(f'prepare-{kind}', script('prepare_data.py', '--kind', kind, '--seed', '42'))
    run('inline-save-start', [HERE / 'inline_save_start.py'])
    for mode, kind, tokens in [('text', 'toy-text', '32'), ('sft', 'attributes-sft', '24')]:
        run(f'evaluate-{kind}-before', script('evaluate.py', 'checkpoints/start.pt', '--data',
            f'data/generated/{kind}/validation.jsonl', '--mode', mode, '--tokens', tokens,
            '--device', 'cpu', '--limit', 'all', '--output', f'outputs/{kind}-before.json'))
        run(f'bounded-train-{kind}', script('train.py', '--task', mode, '--data',
            f'data/generated/{kind}/train.jsonl', '--checkpoint', 'checkpoints/start.pt',
            '--seed', '42', '--device', 'cpu', '--train', '--steps', '200' if mode == 'text' else '500',
            '--stop-after', '2', '--output', 'checkpoints/text.pt' if mode == 'text' else 'checkpoints/attributes.pt'))
    run('bounded-new-sft-stage', script('train.py', '--task', 'sft', '--data',
        'data/generated/attributes-sft/train.jsonl', '--checkpoint', 'checkpoints/text.pt', '--seed', '42',
        '--device', 'cpu', '--train', '--steps', '500', '--stop-after', '2', '--output', 'checkpoints/text-then-sft.pt'))
    for kind in ('style', 'safety'):
        run(f'bounded-train-{kind}', script('train.py', '--task', 'sft', '--data',
            f'data/generated/{kind}/train.jsonl', '--device', 'cpu', '--train', '--steps', '500',
            '--stop-after', '2', '--max-length', '256', '--output', f'checkpoints/{kind}.pt'))
        run(f'evaluate-{kind}', script('evaluate.py', f'checkpoints/{kind}.pt', '--data',
            f'data/generated/{kind}/validation.jsonl', '--mode', 'sft', '--tokens', '128', '--device', 'cpu',
            '--limit', 'all', '--output', f'outputs/{kind}-validation.json'))
    for modality in ('vision', 'audio'):
        run(f'encoder-{modality}', script('pretrain_encoders.py', '--modality', modality,
            '--device', 'cpu', '--train', '--steps', '300', '--output', f'checkpoints/{modality}-encoder.pt'))
    for task in ('vision', 'audio', 'joint'):
        extra = []
        if task != 'audio':
            extra += ['--vision-encoder', 'checkpoints/vision-encoder.pt']
        if task != 'vision':
            extra += ['--audio-encoder', 'checkpoints/audio-encoder.pt']
        run(f'bounded-modal-{task}', script('train.py', '--task', task, '--checkpoint', 'checkpoints/attributes.pt',
            *extra, '--freeze', 'partial' if task == 'joint' else 'projector', '--device', 'cpu',
            '--train', '--steps', '500', '--stop-after', '2', '--output', f'checkpoints/{task}.pt'))
        choices = {'vision':['--color','blue','--shape','circle','--prompt','shape?'],
                   'audio':['--frequency','880','--prompt','pitch?'],
                   'joint':['--color','red','--shape','square','--frequency','220','--prompt','joint?']}[task]
        run(f'infer-modal-{task}', script('infer_modal.py', f'checkpoints/{task}.pt', *choices, '--tokens','16','--device','cpu'))
    run('reject-modal-in-text-evaluator', script('evaluate.py','checkpoints/joint.pt','--device','cpu'), allowed=(1,))
    run('bounded-dpo', script('train.py','--task','dpo','--checkpoint','checkpoints/style.pt','--data',
        'data/generated/preference/train.jsonl','--device','cpu','--train','--steps','200','--stop-after','2',
        '--output','checkpoints/preferred.pt'))
    for ckpt in ('style', 'preferred'):
        run(f'infer-{ckpt}',script('infer.py',f'checkpoints/{ckpt}.pt','--chat','--prompt','0+5=?',
            '--tokens','32','--temperature','0','--device','cpu','--json'))
    for version, extra in [('baseline',[]),('rms',['--norm','rms']),('moe',['--experts','4','--top-k','2'])]:
        run(f'bounded-architecture-{version}',script('train.py','--task','text','--data',
            'data/generated/toy-text/train.jsonl','--device','cpu','--train','--steps','200','--stop-after','2',
            '--seed','42',*extra,'--output',f'checkpoints/{version}.pt'))
        run(f'evaluate-architecture-{version}',script('evaluate.py',f'checkpoints/{version}.pt','--data',
            'data/generated/toy-text/validation.jsonl','--mode','text','--tokens','32','--limit','all','--device','cpu',
            '--output',f'outputs/{version}-validation.json'))
    for bits in ('4','8'):
        run(f'quantize-{bits}',script('quantize.py','checkpoints/attributes.pt','--bits',bits,
            '--output',f'checkpoints/attributes-int{bits}.pt'))
    run('inline-size-check',[HERE/'inline_size_check.py'])
    for version in ('attributes','attributes-int4','attributes-int8'):
        run(f'infer-{version}',script('infer.py',f'checkpoints/{version}.pt','--chat','--prompt',
            'color=blue;shape=circle;pitch=low;color?','--tokens','24','--device','cpu','--json'))
        for split in ('validation','test'):
            run(f'eval-{version}-{split}',script('evaluate.py',f'checkpoints/{version}.pt','--data',
                f'data/generated/attributes-sft/{split}.jsonl','--split-label',split,'--mode','sft','--tokens','24',
                '--limit','all','--device','cpu','--output',f'outputs/{version}-{split}.json'))
    run('fixed-run-help',['-m','scripts.course_experiments.run','--help'])
    run('fixed-run-cuda-no-fallback',['-m','scripts.course_experiments.run','--experiment','simple_models',
        '--device','cuda','--output',str(SCRATCH/'fixed')],allowed=(1,))
    run('fixed-run-reject-mps',['-m','scripts.course_experiments.run','--experiment','simple_models',
        '--device','mps'],allowed=(2,))
finally:
    result = {'created_at':datetime.now(timezone.utc).isoformat(),
              'environment':{'python':platform.python_version(),'torch':str(torch.__version__),
                'torch_git_version':torch.version.git_version,'device':'cpu','platform':platform.platform()},
              'scope':'Exact T.2 dry-runs, T.3 200-step tiny models, both 300-step encoders, literal inline Python; other training is explicitly bounded to 2 updates. No quality or GPU claims.',
              'snapshot_sha256':hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),'runs':rows}
    (HERE/'cpu-probe-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
