"""Bounded, independently authored T.4 CPU probes; never full-course training."""
import copy
import hashlib
import inspect
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch
from scripts.evaluate import answer_sample, evaluate
from tiny_perceptron.data import ByteTokenizer, render_chat, shifted
from tiny_perceptron.model import ModelConfig, TinyLM, generate, loss_sum, masked_loss
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything
from tiny_perceptron.tokenization import load_tokenizer

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
WORK = ROOT / 'outputs/natural-v4/factual-research/T.4/cpu'
WORK.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()), 'threads': str(torch.get_num_threads())}
commands = []

def run(label, args, expected=0, cwd=ROOT):
    command = [sys.executable, *map(str, args)]
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    commands.append({'label': label, 'command': command, 'cwd': str(cwd.relative_to(ROOT)), 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'expected_exit': expected})
    (OUT / 'cpu-commands.json').write_text(json.dumps({'environment': environment, 'commands': commands}, ensure_ascii=False, indent=2) + '\n')
    assert (result.returncode == 0) if expected == 0 else (result.returncode != 0), (label, result.stderr)
    return result

tok = ByteTokenizer()
for kind in ['toy-text', 'attributes-sft']:
    run('prepare-' + kind, ['scripts/prepare_data.py', '--kind', kind, '--seed', '42', '--output', WORK / 'generated'])
splits = {}
for kind, expected in [('toy-text', [9,1,2]), ('attributes-sft', [45,5,10])]:
    manifest = json.loads((WORK / 'generated' / kind / 'manifest.json').read_text())
    families = []
    splits[kind] = {}
    for split, count in zip(['train','validation','test'], expected, strict=True):
        path = WORK / 'generated' / kind / (split + '.jsonl')
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        fam = {r['family'] for r in rows}
        assert len(rows) == count and hashlib.sha256(path.read_bytes()).hexdigest() == manifest['splits'][split]['sha256']
        assert all(r['split'] == split and r['source'] == 'course-generated' and r['license'] == 'MIT' for r in rows)
        families.append(fam)
        effective = sum(len(r['text'].encode()) + 1 for r in rows) if kind == 'toy-text' else sum(len(r['messages'][-1]['content'].encode()) + 1 for r in rows)
        splits[kind][split] = {'records':len(rows), 'families':len(fam), 'effective_targets':effective, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not any(families[i] & families[j] for i in range(3) for j in range(i))
assert splits['toy-text']['validation']['effective_targets'] == 35
assert splits['attributes-sft']['validation']['effective_targets'] == 36

messages = [{'role':'user', 'content':'color=red;shape=circle;pitch=low;shape?'}, {'role':'assistant','content':'circle'}]
x,y = render_chat(messages)
assert y[y != -100].tolist() == tok.encode('circle') + [tok.eos_id]
assert x[(y != -100).nonzero()[0].item()].item() == tok.assistant_id
torch.manual_seed(42)
model = TinyLM(ModelConfig(width=32))
minimal_messages = [{'role':'user','content':'shape?'}, {'role':'assistant','content':'circle'}]
minimal = evaluate(model, [{'messages':minimal_messages}], mode='sft', max_new_tokens=2)
assert minimal['effective_tokens'] == 7 and minimal['samples'][0]['generated_ids'] == [143,30]
assert abs(minimal['mean_token_nll'] - 5.97386714390346) < 1e-6
assert minimal['exact_match'] == minimal['completed_exact_match'] == minimal['eos_rate'] == 0
(OUT / 'minimal-evaluation.json').write_text(json.dumps(minimal, ensure_ascii=False, indent=2) + '\n')
before = {n:p.detach().clone() for n,p in model.named_parameters()}
logits = model(x[None])['logits']; logits.retain_grad()
total, count = loss_sum(logits,y[None]); loss = masked_loss(logits,y[None]); loss.backward()
ignored_logits_grad = float(logits.grad[0,y == -100].abs().max())
context_embedding_grad = float(model.embedding.weight.grad[tok.encode('=')[0]].abs().sum())
unchanged = max(float((before[n]-p.detach()).abs().max()) for n,p in model.named_parameters())
assert ignored_logits_grad == 0 and context_embedding_grad > 0 and unchanged == 0
assert abs(float(loss.detach()) - float(total.detach()) / int(count)) < 1e-6
prefix = torch.tensor([[1,10,11,12,13]])
changed = prefix.clone(); changed[0,-1] = 14
with torch.no_grad():
    causal_difference = float((model(prefix)['logits'][:,:-1]-model(changed)['logits'][:,:-1]).abs().max())
plain = generate(model,prefix,4,use_cache=False)
cached = generate(model,prefix,4,use_cache=True)
assert causal_difference == 0 and torch.equal(plain,cached)
answers = {}
for label,ids in [('content_only',tok.encode('circle')),('completed',tok.encode('circle')+[2]),('space',tok.encode('circle ')+[2]),('illegal_role',tok.encode('circle')+[4,2])]:
    answers[label] = answer_sample(tok,ids,'circle')
assert answers['content_only']['exact_match'] and not answers['content_only']['completed_exact_match']
assert answers['completed']['completed_exact_match']
assert not answers['space']['exact_match'] and not answers['illegal_role']['exact_match']
for label,rows,mode in [('no_records',[],'sft'),('no_supervision',[{'messages':[{'role':'user','content':'?'}]}],'sft'),('too_long_sft',[{'messages':[{'role':'user','content':'?'*200},{'role':'assistant','content':'circle'}]}],'sft')]:
    try: evaluate(model,rows,mode=mode,max_new_tokens=2)
    except ValueError as e: answers[label] = {'expected_error':str(e)}
    else: raise AssertionError(label)
short = TinyLM(ModelConfig(width=8,max_length=4))
long_text = evaluate(short,[{'text':'abcdefghi'}],mode='text',max_new_tokens=2)
assert long_text['effective_tokens'] == 10 and long_text['loss_evaluated_records'] == 1 and long_text['generation_evaluated_records'] == 0
answers['text_generation_only_skip'] = {k:v for k,v in long_text.items() if k != 'samples'}

seed_everything(42)
start = TinyLM(ModelConfig(width=32,layers=1,heads=1,max_length=128))
save_checkpoint(WORK/'start.pt', start, step=0, metadata={'seed':42,'note':'untrained baseline'})
loaded, initial_payload = load_checkpoint(WORK/'start.pt')
assert initial_payload['step'] == 0 and initial_payload['optimizer'] is None
assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in start.state_dict().items())
run('reject-steps-zero',['scripts/train.py','--steps','0','--device','cpu','--output',WORK/'zero.pt'],expected=1)
assert not (WORK/'zero.pt').exists()
run('dry-run-no-checkpoint',['scripts/train.py','--task','sft','--data',WORK/'generated/attributes-sft/train.jsonl','--checkpoint',WORK/'start.pt','--device','cpu','--steps','8','--output',WORK/'dry.pt'])
assert not (WORK/'dry.pt').exists()
resume = {}
(WORK/'outputs').mkdir(exist_ok=True)
for mode,kind in [('text','toy-text'),('sft','attributes-sft')]:
    train_args = ['scripts/train.py','--task',mode,'--data',WORK/'generated'/kind/'train.jsonl','--device','cpu','--train','--steps','8','--batch-size','2','--seed','42','--lr','0.001']
    run(mode+'-full-8',[*train_args,'--checkpoint',WORK/'start.pt','--output',WORK/(mode+'-full.pt')])
    run(mode+'-stop-4',[*train_args,'--checkpoint',WORK/'start.pt','--stop-after','4','--output',WORK/(mode+'-middle.pt')])
    run(mode+'-resume-8',[*train_args,'--checkpoint',WORK/(mode+'-middle.pt'),'--resume','--output',WORK/(mode+'-resumed.pt')])
    full, fp = load_checkpoint(WORK/(mode+'-full.pt'))
    resumed, rp = load_checkpoint(WORK/(mode+'-resumed.pt'))
    difference = max(float((v-resumed.state_dict()[n]).abs().max()) for n,v in full.state_dict().items())
    assert difference == 0 and fp['step'] == rp['step'] == 8 and rp['optimizer']['state']
    resume[mode] = {'full_steps':8,'saved_step':4,'resumed_step':rp['step'],'max_weight_difference':difference,'schedule_steps':rp['metadata']['schedule_steps'],'peak_lr':rp['metadata']['peak_lr'],'optimizer_entries':len(rp['optimizer']['state'])}
    run(mode+'-reject-completed-resume',[*train_args,'--checkpoint',WORK/(mode+'-full.pt'),'--resume','--output',WORK/(mode+'-bad.pt')],expected=1)
    changed_schedule = [str(v) for v in train_args]; changed_schedule[changed_schedule.index('--steps')+1] = '9'
    run(mode+'-reject-changed-schedule',[*changed_schedule,'--checkpoint',WORK/(mode+'-middle.pt'),'--resume','--output',WORK/(mode+'-bad-plan.pt')],expected=1)
    run(mode+'-fresh-stage',[*train_args,'--checkpoint',WORK/(mode+'-full.pt'),'--stop-after','1','--output',WORK/(mode+'-fresh.pt')])
    _, fresh_payload = load_checkpoint(WORK/(mode+'-fresh.pt')); assert fresh_payload['step'] == 1
    stem = 'text' if mode == 'text' else 'attributes'
    for stage, checkpoint in [('before',WORK/'start.pt'),('after',WORK/(mode+'-full.pt'))]:
        eval_path = WORK/'outputs'/(stem+'-'+stage+'.json')
        run(stem+'-'+stage+'-evaluate',['scripts/evaluate.py',checkpoint,'--data',WORK/'generated'/kind/'validation.jsonl','--mode',mode,'--tokens','24','--device','cpu','--output',eval_path])
        infer_args = ['scripts/infer.py',checkpoint,'--prompt','color=' if mode == 'text' else 'color=red;shape=square;pitch=low;shape?','--tokens','24','--temperature','0','--cache','--device','cpu'] + (['--chat'] if mode == 'sft' else [])
        result = run(stem+'-'+stage+'-infer',infer_args)
        (WORK/'outputs'/(stem+'-prompt-'+stage+'.txt')).write_text(result.stdout)

source = (OUT/'section-read-original.md').read_text()
comparison_code = re.findall(r'```python\n(.*?)\n```',source,re.S)[-1]
comparison_path = WORK/'compare_validation.py'
comparison_path.write_text(comparison_code+'\n')
run('literal-comparison-code',[comparison_path],cwd=WORK)
comparison = json.loads((WORK/'outputs/attributes-comparison.json').read_text())
assert comparison['before']['effective_tokens'] == comparison['after']['effective_tokens'] == 36
before_path = WORK/'outputs/attributes-before.json'; original_before = before_path.read_bytes()
bad = json.loads(original_before); bad['effective_tokens'] += 1; before_path.write_text(json.dumps(bad))
(WORK/'outputs/attributes-comparison.json').unlink()
run('comparison-rejects-mismatched-denominator',[comparison_path],expected=1,cwd=WORK)
assert not (WORK/'outputs/attributes-comparison.json').exists()
before_path.write_bytes(original_before)

summary = {'environment':environment,'splits':splits,'minimal':{k:v for k,v in minimal.items() if k in ['mean_token_nll','effective_tokens','exact_match','completed_exact_match','eos_rate','samples']},'mask_and_update':{'assistant_target_ids':y[y != -100].tolist(),'ignored_logits_gradient_max':ignored_logits_grad,'user_context_embedding_gradient_sum':context_embedding_grad,'backward_parameter_max_change':unchanged,'causal_past_logits_max_difference':causal_difference,'cache_generation_ids_equal':torch.equal(plain,cached)},'matching_and_skips':answers,'resume':resume,'parameter_count':{'width64_layers2':sum(p.numel() for p in TinyLM(ModelConfig(width=64,layers=2)).parameters()),'derivation':'2*264*64 + 128*64 + 2*(4*64^2 + 8*64^2 + 5*64 + 4*64) + 2*64 = 141568; untied embeddings/output, position table, attention, dense FFN plus biases, block/final LayerNorm'},'scope':'32-step total local CPU training across two 8-step full and resumed pairs plus two one-step fresh stages; no GPU/full-course training or benchmark replication.'}
assert summary['parameter_count']['width64_layers2'] == 141568
(OUT/'cpu-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
