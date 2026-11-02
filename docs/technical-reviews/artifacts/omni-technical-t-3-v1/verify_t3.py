"""T.3 independent review: exact lesson arguments, isolated output paths, short CPU only."""

import hashlib
import inspect
import json
import platform
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = Path('/workspace/work/omni-technical-t-3-v1')
WORK.mkdir(exist_ok=False)
(WORK / '.venv').symlink_to(ROOT / '.venv', target_is_directory=True)
(WORK / 'scripts').symlink_to(ROOT / 'scripts', target_is_directory=True)
sys.path.insert(0, str(ROOT))
from scripts.train_simple import examples
from tiny_perceptron.data import ByteTokenizer, split_documents, toy_documents
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.tokenization import generation_report
from tiny_perceptron.training import load_checkpoint, seed_everything

torch.set_num_threads(2)
receipt = {
    'created_at': datetime.now(UTC).isoformat(),
    'invocation': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-t-3-v1/verify_t3.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torch_git_version': torch.version.git_version, 'device': 'cpu',
                    'platform': platform.platform()},
    'work_directory': str(WORK),
    'commands': [],
    'limits': 'Only 200-step toy character models and 32-token tiny-model CPU inference. No GPU, Modal, paid tasks, publication, or full-scale training.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(lesson_command, argv=None, stdout_path=None, timeout=60):
    argv = shlex.split(lesson_command.split(' > ')[0]) if argv is None else argv
    result = subprocess.run(argv, cwd=WORK, capture_output=True, text=True, timeout=timeout)
    if stdout_path:
        (WORK / stdout_path).write_text(result.stdout)
    entry = {'lesson_command': lesson_command, 'executed_argv': argv, 'cwd': str(WORK),
             'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    receipt['commands'].append(entry)
    if result.returncode:
        raise RuntimeError(json.dumps(entry, ensure_ascii=False))
    return result.stdout


frozen = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.3.md'
body = frozen.read_text()
receipt['source_sha256'] = digest(frozen)
blocks = re.findall(r'```bash\n(.*?)\n```', body, re.S)
receipt['lesson_bash_blocks'] = blocks
run('mkdir -p outputs checkpoints')
for block in blocks[:3]:
    for command in block.splitlines():
        if command.startswith('.venv/bin/python scripts/train_simple.py'):
            run(command, stdout_path=command.split(' > ')[1])

parts = split_documents(toy_documents(), 42)
vocabulary = {char: index + 2 for index, char in enumerate(sorted(set(''.join(parts['train']))))}
receipt['data'] = {'documents': toy_documents(), 'splits': parts, 'vocabulary': vocabulary,
                   'vocab_size_including_boundary_and_unk': len(vocabulary)+2,
                   'deduplication_check': split_documents(toy_documents()+[toy_documents()[0]],42)==parts,
                   'example_tensors': {}}
for context in [1,3,5]:
    receipt['data']['example_tensors'][str(context)] = {}
    for name, documents in parts.items():
        x,y = examples(documents,vocabulary,context)
        receipt['data']['example_tensors'][str(context)][name] = {
            'x_shape': list(x.shape), 'y_shape': list(y.shape), 'x_dtype': str(x.dtype),
            'y_dtype': str(y.dtype), 'target_count': y.numel(),
            'first_rows': x[:5].tolist(), 'first_targets': y[:5].tolist(),
            'boundary_targets': int((y==0).sum()), 'unknown_targets': int((y==1).sum())}

receipt['simple_reports'] = {}
for name in ['bigram', 'mlp']:
    before = json.loads((WORK/f'outputs/{name}-before.json').read_text())
    after = json.loads((WORK/f'outputs/{name}-after.json').read_text())
    payload=torch.load(WORK/f'checkpoints/{name}.pt',map_location='cpu',weights_only=True)
    model=BigramLM(len(vocabulary)+2) if name=='bigram' else ContextMLP(len(vocabulary)+2,3,16)
    model.load_state_dict(payload['model'])
    post={}
    for part in ['train','validation']:
        x,y=examples(parts[part],vocabulary,payload['context'])
        with torch.no_grad():
            logits=model(x)
            mean=F.cross_entropy(logits,y).item()
            sum_count=(F.cross_entropy(logits,y,reduction='sum')/y.numel()).item()
        post[part]={'loss':mean,'sum_divided_by_targets':sum_count,'targets':y.numel(),
                    'shape':list(logits.shape),'dtype':str(logits.dtype)}
        assert abs(mean-after[f'{part if part=="train" else "validation"}_loss']) < 1e-6
    try:
        load_checkpoint(WORK/f'checkpoints/{name}.pt','cpu')
    except ValueError as error:
        rejected=str(error)
    else:
        raise AssertionError('simple checkpoint unexpectedly accepted by Transformer loader')
    receipt['simple_reports'][name]={'before':before,'after':after,
        'loss_deltas':{key:after[key]-before[key] for key in ['train_loss','validation_loss']},
        'checkpoint_keys':sorted(payload),'checkpoint_sha256':digest(WORK/f'checkpoints/{name}.pt'),
        'post_update_recalculation':post,'transformer_loader_rejection':rejected}

# Independently instrument the unmodified lesson entrypoint: backward without step leaves weights unchanged.
import runpy
original_argv=sys.argv
original_step=torch.optim.AdamW.step
step_calls=[]
def capture_step(self,*args,**kwargs):
    before={id(p):p.detach().clone() for group in self.param_groups for p in group['params']}
    result=original_step(self,*args,**kwargs)
    step_calls.append(max(float((p-before[id(p)]).abs().max().detach()) for group in self.param_groups for p in group['params']))
    return result
torch.optim.AdamW.step=capture_step
import contextlib,io
captured=io.StringIO()
sys.argv=['scripts/train_simple.py','--model','bigram','--seed','42','--device','cpu']
with contextlib.redirect_stdout(captured):
    runpy.run_path(str(ROOT/'scripts/train_simple.py'),run_name='__main__')
sys.argv=original_argv
torch.optim.AdamW.step=original_step
receipt['dry_run_instrumentation']={'optimizer_step_calls':len(step_calls),'stdout':captured.getvalue()}
assert not step_calls

receipt['mlp_parameter_shapes']={str(context): {'shapes':{n:list(p.shape) for n,p in ContextMLP(17,context,16).named_parameters()},
    'parameters':sum(p.numel() for p in ContextMLP(17,context,16).parameters())} for context in [1,3,5]}
tok=ByteTokenizer()
receipt['handwritten_generation_example']={'encoded_A':tok.encode('A'),'eos_id':tok.eos_id,
    'report':generation_report(tok,[73,2]),'without_eos':generation_report(tok,[73])}
assert generation_report(tok,[73,2])['answer']=='A'

# Public download is 1.6 MB and inference is tiny; redirect only the output destination.
run('.venv/bin/python scripts/fetch_course_models.py --list')
run('.venv/bin/python scripts/fetch_course_models.py --model text_foundation',
    ['.venv/bin/python','scripts/fetch_course_models.py','--model','text_foundation','--output',str(WORK/'checkpoints/course')], timeout=120)
run('.venv/bin/python scripts/infer.py checkpoints/course/text_foundation/model.pt --prompt "color=blue;shape=circle;" --tokens 32 --device cpu --json')
payload=torch.load(WORK/'checkpoints/course/text_foundation/model.pt',map_location='cpu',weights_only=True)
receipt['public_model']={'keys':sorted(payload),'config':payload['config'],'metadata':payload.get('metadata'),
    'optimizer':payload.get('optimizer'),'training_state':payload.get('training_state'),
    'model_sha256':digest(WORK/'checkpoints/course/text_foundation/model.pt'),
    'card_sha256':digest(WORK/'checkpoints/course/text_foundation/README.md')}

# This exact fixed experiment is also a tiny CPU teaching demonstration, not a quality reproduction.
run('.venv/bin/python -m scripts.course_experiments.run --experiment simple_models --device cpu',
    ['.venv/bin/python','-m','scripts.course_experiments.run','--experiment','simple_models','--device','cpu',
     '--output',str(WORK/'outputs/course-experiments/course-v1')])
raw=json.loads((WORK/'outputs/course-experiments/course-v1/simple_models/result.json').read_text())
receipt['fixed_experiment_current_cpu']={k:raw[k] for k in ['device','seed','torch_version','python_version',
    'elapsed_seconds','timing_scope','step_scale','evidence_status','results','artifacts']}
receipt['completed_at']=datetime.now(UTC).isoformat()
(HERE/'execution.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'completed','receipt':str(HERE/'execution.json'),
    'sha256':digest(HERE/'execution.json'),'work_directory':str(WORK)},ensure_ascii=False))
