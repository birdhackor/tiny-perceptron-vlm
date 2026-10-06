"""Bounded R.4 review: read specified pointers; no training or held-out generation."""
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path('/workspace/selftrained-v2')
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig, PaddingSafeMoE
from tiny_perceptron.modern import DenseFFN

OUT = ROOT / 'docs/technical-reviews/artifacts/p6-R.4/checks'
catalog = []

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(rel, pointers):
    p = ROOT / rel
    v = json.loads(p.read_bytes())
    catalog.append({'path': rel, 'sha256_rawbytes': sha(p),
                    'topkeys': {k: type(x).__name__ for k, x in v.items()},
                    'inspected_pointers': pointers})
    return v

summary = {'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                           'device': 'cpu'}, 'training': [], 'final': [], 'cpu': [], 'structure': []}
for p in sorted((ROOT / 'docs/selftrained/results/training-raw').glob('*/raw/train-receipt.json')):
    rel = p.relative_to(ROOT).as_posix()
    r = read(rel, ['/architecture', '/stage', '/steps', '/tokens', '/target_tokens',
        '/train_records', '/validation_records', '/test_used_for_selection', '/seed',
        '/origin/kind', '/origin/seed', '/stage_history/*/stage', '/stage_history/*/checkpoint_sha256',
        '/stage_history/*/selection', '/interrupted', '/completed_requested_steps',
        '/selected_checkpoint_available', '/inference_exported', '/config'])
    ex_rel = (p.parent / 'execution.json').relative_to(ROOT).as_posix()
    e = read(ex_rel, ['/status', '/returncode', '/revision', '/command'])
    assert r['origin']['kind'] == 'all-neural-weights-random'
    assert r['origin']['seed'] == r['seed']
    assert r['completed_requested_steps'] and not r['interrupted']
    assert r['selected_checkpoint_available'] and r['inference_exported']
    assert not r['test_used_for_selection']
    assert r['steps'] == int(e['command'][e['command'].index('--steps') + 1])
    assert r['tokens'] > 0 and r['train_records'] > 0 and r['validation_records'] > 0
    assert e['returncode'] == 0 and e['status'] == 'completed'
    summary['training'].append({'stage_archive': p.parents[1].name, 'steps': r['steps'],
        'tokens': r['tokens'], 'target_tokens': r['target_tokens'], 'architecture': r['architecture'],
        'origin': r['origin']['kind'], 'seed': r['seed'], 'train_records': r['train_records'],
        'validation_records': r['validation_records'], 'completed': True,
        'history': [{'stage': h['stage'], 'selected_sha': h['checkpoint_sha256'],
                     'selection': h['selection']} for h in r['stage_history']]})
assert len(summary['training']) == 15
probes = json.loads((OUT.parent / 'originals/anonymous-public-probe.json').read_bytes())
for a in ['moe', 'dense']:
    base = f'docs/selftrained/results/public-raw/{a}'
    f = read(base + '/freeze/frozen.json', ['/created_at', '/architecture', '/checkpoint_sha256',
        '/safe_weights_sha256', '/inference_manifest_sha256', '/selected_checkpoint_sha256',
        '/thresholds', '/selection', '/test_once'])
    m = read(base + '/test/metrics.json', ['/split', '/count', '/expected_count', '/evaluation_complete',
        '/limited_smoke', '/interrupted', '/checkpoint_sha256', '/safe_weights_sha256',
        '/inference_manifest_sha256', '/selected_checkpoint_sha256', '/thresholds',
        '/per_task_final_reply', '/teacher_forcing_used_for_generation'])
    j = read(base + '/test/evaluation-receipt.json', ['/split', '/expected_count', '/completed_count',
        '/status', '/checkpoint_sha256', '/protocol_sha256', '/resumed_from'])
    r = read(base + '/test/receipt.json', ['/status', '/returncode', '/revision', '/public_export'])
    assert m['count'] == m['expected_count'] == j['completed_count'] == j['expected_count'] == 3734
    assert m['split'] == j['split'] == 'test' and j['status'] == 'complete'
    assert m['evaluation_complete'] and not m['limited_smoke'] and not m['interrupted']
    assert not m['teacher_forcing_used_for_generation']
    assert f['test_once'] and f['selection'] == 'validation only'
    assert f['thresholds'] == m['thresholds']
    assert j['protocol_sha256'] == sha(ROOT / (base + '/freeze/frozen.json'))
    for k in ['checkpoint_sha256','safe_weights_sha256','inference_manifest_sha256','selected_checkpoint_sha256']:
        assert f[k] == m[k] == r['public_export'][k]
    assert j['checkpoint_sha256'] == m['checkpoint_sha256']
    assert r['returncode'] == 0 and r['status'] == 'completed'
    assert r['public_export']['public_export']['revision'] == '979cdfacc588ad0536f1c64fff96f264571cf054'
    probe = next(x for x in probes if x['architecture'] == a)
    assert probe['model.safetensors']['headers']['x-linked-etag'].strip('"') == m['safe_weights_sha256']
    assert probe['model-config.json']['status'] == 200
    tasks = m['per_task_final_reply']
    assert sum(t['count'] for t in tasks.values()) == 3734
    for task, t in tasks.items():
        for metric, z in t.items():
            if isinstance(z, dict) and {'numerator','denominator','rate'} <= z.keys():
                assert abs(z['numerator'] / z['denominator'] - z['rate']) < 1e-14
    rt = tasks['tool_call']['tool_roundtrip']
    assert rt['rate'] < f['thresholds']['tool_roundtrip'] == .95
    summary['final'].append({'architecture': a, 'count': m['count'], 'frozen_weights_sha': m['safe_weights_sha256'],
        'completed': True, 'public_revision': probe['revision'],
        'tool_roundtrip': rt, 'original_tool_threshold': .95,
        'passes_all_original_capability_thresholds': False,
        'tasks': {task: {'count': t['count'], 'semantic': t['semantic']} for task,t in tasks.items()}})
for p in sorted((ROOT / 'docs/selftrained/results/public-cpu-raw').glob('*result.json')):
    r = read(p.relative_to(ROOT).as_posix(), ['/task','/invocation_number','/argv','/returncode',
        '/stdout_sha256','/stderr_sha256','/actual_answer','/model/manifest_sha256',
        '/model/files/model.safetensors','/model/origin/kind','/public_source', '/generation_count'])
    assert r['returncode'] == 0
    argv = r['argv']; assert argv[argv.index('--device') + 1] == 'cpu'
    stdout = p.with_name(p.name.replace('-result.json','-stdout.txt'))
    stderr = p.with_name(p.name.replace('-result.json','-stderr.txt'))
    assert sha(stdout) == r['stdout_sha256'] and sha(stderr) == r['stderr_sha256']
    # Raw stdout is bound by hash, but no saved GPU comparisons or controller review is read.
    row = {'task': r['task'], 'invocation': r['invocation_number'], 'returncode': r['returncode'],
           'stdout_sha256': sha(stdout), 'stderr_sha256': sha(stderr)}
    if 'public_source' in r:
        assert r['public_source']['revision'] == '979cdfacc588ad0536f1c64fff96f264571cf054'
        assert r['public_source']['authentication'] == 'disabled'
        assert r['model']['origin']['kind'] == 'all-neural-weights-random'
        assert r['model']['files']['model.safetensors'] == summary['final'][0]['frozen_weights_sha']
        row.update(answer=r['actual_answer'], public_source=r['public_source'],
                   weights_sha=r['model']['files']['model.safetensors'],
                   generation_count=r['generation_count'])
    summary['cpu'].append(row)
torch.set_num_threads(2)
torch.manual_seed(271828)
for a in ['moe','dense']:
    config = SelftrainedConfig(**json.loads((OUT.parent / f'originals/{a}-model-config.json').read_bytes()))
    model = LimitedAssistant(config).eval()
    ids = torch.tensor([[11, *([7]*2), *([8]*32), *([9]*16), 12]], dtype=torch.long)
    payloads = [[{'kind':'image','values':torch.zeros(2,1,32,32),
        'coordinates':torch.tensor([[.25,.5],[.75,.5]])},
        {'kind':'ocr','values':torch.zeros(1,32,128)},
        {'kind':'audio','values':torch.zeros(24,40)}]]
    with torch.no_grad():
        out = model(ids, modalities=payloads)
    assert out['logits'].shape == (1,52,550) and torch.isfinite(out['logits']).all()
    assert [x['kind'] for x in out['perception']] == ['image','ocr','audio']
    assert all(isinstance(b.ffn, PaddingSafeMoE if a=='moe' else DenseFFN) for b in model.lm.blocks)
    assert model.lm.embedding.weight is model.lm.output.weight
    summary['structure'].append({'architecture':a,'parameters':sum(p.numel() for p in model.parameters()),
        'all_modalities_share_one_lm':True,'logits_shape':list(out['logits'].shape),
        'encoder_token_shapes': {'image':[2,256],'ocr':[32,256],'audio':[16,256]},
        'ffn_class':type(model.lm.blocks[0].ffn).__name__, 'no_generation_or_update':True})
# Autograd check does not update parameters and does not train any model.
w = torch.nn.Parameter(torch.tensor([2.0,3.0])); loss = (w*w).sum(); loss.backward()
assert torch.equal(w.grad, torch.tensor([4.0,6.0])) and torch.equal(w.detach(), torch.tensor([2.,3.]))
summary['autograd'] = {'input_weights':[2.,3.], 'loss':float(loss.detach()),
                       'gradient':w.grad.tolist(),'parameter_updated':False}
natural = ast.parse((ROOT/'tiny_perceptron/natural_assistant.py').read_text())
summary['mature_route'] = {}
for name in ['load_core','load_asr']:
    f = next(x for x in natural.body if isinstance(x,ast.FunctionDef) and x.name == name)
    calls = [ast.unparse(x.func) for x in ast.walk(f) if isinstance(x,ast.Call)]
    pretrained_calls = [x for x in calls if x.endswith('.from_pretrained')]
    assert len(pretrained_calls) >= 2
    summary['mature_route'][name] = {'line_range':[f.lineno,f.end_lineno], 'calls':pretrained_calls}
(OUT/'raw-catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
(OUT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'training_stages':len(summary['training']),
    'final_counts':[x['count'] for x in summary['final']],
    'tool_roundtrip_numerators':[x['tool_roundtrip']['numerator'] for x in summary['final']],
    'cpu_original_invocations':len(summary['cpu']), 'structure':summary['structure'],
    'autograd':summary['autograd'], 'environment':summary['environment']},ensure_ascii=False,indent=2))
