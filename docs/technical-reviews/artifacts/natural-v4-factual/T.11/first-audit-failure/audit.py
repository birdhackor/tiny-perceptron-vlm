"""Independent bounded T.11 record audit and CPU software probes; no GPU rerun."""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import torch

from scripts.course_experiments import applications as app
from scripts.course_experiments.common import split_records, text_examples, records_sha256
from scripts.prepare_data import generate_records
from scripts.evaluate import evaluate, answer_sample
from tiny_perceptron.data import ByteTokenizer, render_chat, IGNORE
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.training import save_checkpoint, load_checkpoint, seed_everything

torch.set_num_threads(2)
BASE = Path(__file__).parent
WORK = Path('outputs/natural-v4/factual-research/T.11')
WORK.mkdir(parents=True, exist_ok=True)
tok = ByteTokenizer()
out = {'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'cpu',
                       'cuda_available': torch.cuda.is_available(), 'threads': str(torch.get_num_threads())}}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def original(name):
    path = Path('docs/course-experiments/results') / (name + '.json')
    data = json.loads(path.read_text())
    out.setdefault('original_records', {})[name] = {'path': str(path), 'sha256': sha(path.read_bytes()),
        'device': data['device'], 'gpu': data['gpu'], 'torch': data['torch_version'], 'seed': data['seed'],
        'revision': data['revision']}
    return data['results']

def decoded(sample):
    ids = sample['generated_ids']
    content = ids[:-1] if ids and ids[-1] == 2 else ids
    clean = not any(x < 8 for x in content)
    text = bytes(x-8 for x in content if x >= 8).decode('utf-8', errors='replace')
    assert text == sample['generated']
    assert len(ids) == sample['generated_tokens']
    assert sample['eos'] == bool(ids and ids[-1] == 2)
    assert clean == (not sample['invalid_special_tokens'])
    return text, clean

def prompt_ids(messages):
    roles = {'system': 7, 'user': 3, 'assistant': 4}
    ids = [1]
    for msg in messages:
        ids += [roles[msg['role']]] + [b+8 for b in msg['content'].encode()] + [2]
    return ids + [4]

def splits_check(splits, summary):
    fams = [set(r['family'] for r in rows) for rows in splits.values()]
    assert all(not (a & b) for i, a in enumerate(fams) for b in fams[i+1:])
    for side, rows in splits.items():
        assert len(rows) == summary[side]['records']
        assert len({r['family'] for r in rows}) == summary[side]['families']
        assert records_sha256(rows) == summary[side]['sha256']
    return {s: {'records': len(rows), 'families': len({r['family'] for r in rows}),
                'sha256': records_sha256(rows)} for s, rows in splits.items()}

def sampled_targets(rows, steps, batch):
    # Reproduce only the deterministic question draws and count supervised targets; no model updates.
    lengths = [len(r['messages'][-1]['content'].encode()) + 1 for r in rows]
    rng = random.Random(42)
    return sum(sum(rng.choices(lengths, k=batch)) for _ in range(steps))

rag = original('rag')
rs = app._rag_splits(42)
rag_summary = {'split': splits_check(rs, rag['split'])}
for side, rows in rs.items():
    keys = {r['key'] for r in rows}
    assert all(d['text'].split()[0] in keys for r in rows for d in r['documents'])
assert len(rag['samples']) == 84
by_mode = {}
for row in rag['samples']:
    text, clean = decoded(row['generation']['samples'][0])
    assert row['generation']['input_ids'] == prompt_ids(row['generation']['messages'])
    fact = row['context_fact']
    correct = clean and text == f"{fact['address']}[{fact['source']}]"
    assert correct == row['fact_answer_correct']
    assert (clean and text == row['expected']) == row['exact_match']
    assert row['fact']['family'] == row['family']
    by_mode.setdefault(row['mode'], {})[row['family']] = correct
assert len(by_mode) == 7 and all(len(v) == 12 for v in by_mode.values())
paired = {m: sum(v and by_mode['correct_context'][fam] for fam, v in by_mode[m].items())
          for m in ('changed_address_context', 'changed_source_context')}
assert paired == {'changed_address_context': 5, 'changed_source_context': 2}
for row in rag['samples']:
    if row['mode'].startswith('changed_'):
        old, new = row['fact'], row['context_fact']
        assert old['key'] == new['key']
        field = 'address' if row['mode'] == 'changed_address_context' else 'source'
        assert old[field] != new[field]
        other = 'source' if field == 'address' else 'address'
        assert old[other] == new[other]
        assert row['documents'] == [{'id': new['source'], 'text': f"{new['key']} address={new['address']}"}]
assert len(rag['icl_samples_same_model_weights']) == 4
for row in rag['icl_samples_same_model_weights']:
    decoded(row['generation']['samples'][0])
rag_summary.update(generations=84, families=12, modes=7, icl_generations=4,
                   per_mode_correct={k: sum(v.values()) for k, v in by_mode.items()}, paired_correct=paired)
out['rag'] = rag_summary

tools = original('tools')
ts = split_records(app._tool_records(), 42)
tools_summary = {'split': splits_check(ts, tools['split'])}
counts = Counter()
for row in tools['samples'] + tools['one_step_cap_samples']:
    match = re.fullmatch(r'CALC:(add|multiply)\(([0-9]+),([0-9]+)\)', row['question'])
    truth = int(row['question'][5:]) if match is None else (int(match[2])+int(match[3]) if match[1]=='add' else int(match[2])*int(match[3]))
    assert truth == row['expected']
    context = [{'role':'system','content':app._TOOLS_SYSTEM},{'role':'user','content':row['question']}]
    executions = []
    for event in row['trace']:
        g = event['generation']; text, clean = decoded(g['samples'][0])
        assert g['input_ids'] == prompt_ids(context)
        try:
            action = json.loads(text)
            parsed = clean and isinstance(action,dict)
        except json.JSONDecodeError:
            parsed = False; action = None
        assert parsed == ('parsed' in event)
        if 'parsed' in event: assert action == event['parsed']
        if event['executed']:
            assert match and action == {'name':match[1], 'arguments':{'a':int(match[2]),'b':int(match[3])}}
            calculated = int(match[2])+int(match[3]) if match[1]=='add' else int(match[2])*int(match[3])
            assert calculated == event['tool_result'] and event['finite_result']
            executions.append(calculated)
            context += [{'role':'assistant','content':text},{'role':'user','content':f'TOOL_RESULT:{calculated}'}]
    correct = row['status']=='done' and type(row['answer']) in (int,float) and row['answer']==truth
    correct = correct and (not executions if match is None else bool(executions and row['answer']==executions[-1]))
    assert correct == row['correct']
    if row in tools['samples']:
        counts['tasks'] += 1; counts['correct'] += correct
        counts['first_action_parsed'] += 'parsed' in row['trace'][0]
        counts['executed_calls'] += len(executions)
    else:
        assert row['status']=='step_limit' and len(executions)==1 and row['max_steps']==1
failure = next(r for r in tools['samples'] if not r['correct'])
assert failure['question']=='CALC:multiply(9,9)'
assert failure['trace'][0]['tool_result']==81
assert failure['trace'][1]['generation']['samples'][0]['generated']=='{"done":true,"answer":8}}'
assert dict(counts)=={'tasks':20,'correct':19,'first_action_parsed':20,'executed_calls':18}
assert len(tools['one_step_cap_samples'])==18
tools_summary.update(dict(counts), capped_tasks=18, capped_executions=18,
    failure={'question':failure['question'],'tool_result':81,'final_json':'{"done":true,"answer":8}}'},
    injected_checks=len(tools['injected_protocol_checks_not_model_scores']))
out['tools'] = tools_summary

reason = original('reasoning')
cs = split_records(app._reasoning_records(), 42)
reason_summary = {'split':splits_check(cs,reason['split']), 'budgets':{}, 'training_targets':{}}
assert reason['comparison']['direct']['base_state_sha256']==reason['comparison']['steps']['base_state_sha256']
candidate_total=0
for mode, branch in reason['comparison'].items():
    target_rows=app._reasoning_sft(cs['train'],mode)
    counted=sampled_targets(target_rows,900,24)
    assert counted==branch['training']['effective_tokens']
    assert branch['training']['steps']==900
    reason_summary['training_targets'][mode]=counted
    budgets=[]
    for budget in branch['budgets']:
        k=budget['candidate_count'];covered=majority_correct=token_count=0;seconds=0
        assert len(budget['samples'])==24
        for row in budget['samples']:
            truth=row['a']+row['b']+row['c'];assert truth==row['truth']
            values=[];flags=[]
            assert len(row['candidates'])==k
            assert row['generation']['candidate_count']==k
            assert row['generation']['temperature']==0.7
            assert row['generation']['max_new_tokens']==(8 if mode=='direct' else 48)
            for candidate in row['candidates']:
                text,clean=decoded(candidate);text=text.strip();candidate_total+=1
                m=re.fullmatch(r'-?[0-9]+',text) if mode=='direct' else re.search(r';answer=(-?[0-9]+)$',text)
                value=(int(m[0]) if mode=='direct' else int(m[1])) if clean and m else None
                flags.append(value==truth);assert (value==truth)==candidate['final_correct']
                if value is not None:values.append(value)
                if mode=='steps':
                    m=re.fullmatch(r'(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)',text)
                    valid=False
                    if m and clean:
                        a,b,s,p,c,total,final=map(int,m.groups())
                        valid=a+b==s and p+c==total and s==p and total==final and (a,b,c)==(row['a'],row['b'],row['c']) and final==truth
                    assert valid==candidate['fully_verified']
                token_count+=len(candidate['generated_ids'])
            major=Counter(values).most_common(1)[0][0] if values else None
            covered+=any(flags);majority_correct+=major==truth;seconds+=row['generation']['seconds']
            assert any(flags)==row['oracle_coverage'] and major==row['majority_answer']
        assert covered==budget['oracle_coverage']['numerator']
        assert majority_correct==budget['majority_accuracy']['numerator']
        assert token_count==budget['generated_tokens'] and math.isclose(seconds,budget['generation_seconds'])
        budgets.append({'k':k,'oracle':covered,'majority':majority_correct,'questions':24,'candidates':24*k,
                        'tokens_including_eos':token_count,'generation_seconds_record':seconds})
    reason_summary['budgets'][mode]=budgets
assert candidate_total==720
assert reason_summary['budgets']['steps'][0]['oracle']==11
assert reason_summary['budgets']['steps'][3]['oracle']==14
assert reason_summary['budgets']['steps'][3]['majority']==12
reason_summary['total_candidates']=candidate_total
out['reasoning']=reason_summary

policy_summary={}
for label in ('strict','weak_proxy'):
    branch=reason['reinforce'][label];tr=branch['training']
    assert tr['steps']==1200 and tr['sampled_actions']==1200*64 and tr['effective_tokens']==0
    rows=branch['after']['samples'];greedy=sample_correct=proxy=enumerations=actions=0
    assert len(rows)==24
    for row in rows:
        truth=sum(row[k] for k in ('a','b','c'));assert truth==row['truth']
        greedy+=bool(re.fullmatch(r'[0-9]+',row['greedy_action']) and int(row['greedy_action'])==truth)
        assert len(row['samples'])==16
        for sample in row['samples']:
            action=sample['action_id'];text=str(action) if action<16 else ' '.join(map(str,range(16)))
            assert text==sample['generated'];strict=action<16 and action==truth;weak=str(truth) in text.split()
            assert strict==bool(sample['strict_reward']) and weak==bool(sample['proxy_reward'])
            sample_correct+=strict;proxy+=weak;enumerations+=action==16;actions+=1
    assert greedy==branch['after']['greedy_accuracy']['numerator']
    assert sample_correct==branch['after']['sample_accuracy']['numerator']
    policy_summary[label]={'updates':1200,'training_actions':76800,'autoregressive_tokens':0,
        'greedy_correct':greedy,'questions':24,'sample_correct':sample_correct,'sample_actions':actions,
        'proxy_reward':proxy,'enumeration_actions':enumerations}
assert policy_summary['strict']['greedy_correct']==policy_summary['strict']['sample_correct']==0
assert policy_summary['weak_proxy']['proxy_reward']==policy_summary['weak_proxy']['enumeration_actions']==384
assert policy_summary['weak_proxy']['sample_correct']==0
out['finite_policy']=policy_summary

pilot=reason['gsm8k_training_pilot']
raw=Path('data/training/reasoning-initial/gsm8k-train-first200.jsonl').read_bytes()
assert raw== (WORK/'gsm8k-original-first200.jsonl').read_bytes()
assert sha(raw)==pilot['source_sha256']
records=[json.loads(line) for line in raw.splitlines()]
assert len(records)==200
rows=[{'family':sha(row['question'].encode()),'source_index':i,
       'provenance':'GSM8K original human-authored training answer; not teacher generation',
       'messages':app._conversation(row['question'],row['answer'])} for i,row in enumerate(records)]
gs=split_records(rows,42);gsm_summary={'split':splits_check(gs,pilot['split']),'original_rows':200,'original_sha256':sha(raw)}
packing={};train_lengths=[]
for side,side_rows in gs.items():
    segments=[];skipped=0
    for row in side_rows:
        x,y=render_chat(row['messages']);counted=0
        for start in range(0,len(x),128):
            count=int((y[start:start+128]!=IGNORE).sum())
            if not count:skipped+=1;continue
            segments.append({'source_index':row['source_index'],'start':start,'length':len(x[start:start+128]),'effective_tokens':count})
            counted+=count
            if side=='train':train_lengths.append(count)
        assert counted==len(row['messages'][-1]['content'].encode())+1
    packing[side]={'segments':len(segments),'effective_tokens':sum(x['effective_tokens'] for x in segments),'question_only_segments_skipped':skipped}
    assert packing[side]==pilot['packing'][side]
sampler=random.Random(42);used=sum(sum(sampler.choices(train_lengths,k=16)) for _ in range(150))
assert used==pilot['training']['effective_tokens'] and pilot['training']['steps']==150
for row,continuation in zip(gs['test'],pilot['continuations'],strict=True):
    prefix=row['messages'][1]['content'].encode()[:40].decode('utf-8',errors='ignore')
    assert continuation['generation']['messages']==[{'role':'user','content':'Continue human answer: '+prefix}]
    decoded(continuation['generation']['samples'][0])
gsm_summary.update(packing=packing, training_updates=150, training_targets=used,
                   continuation_prompts=len(pilot['continuations']), input_mode='human-answer prefix; no solve accuracy')
out['gsm8k']=gsm_summary

# Actual small CPU quantization and evaluation software checks, separate from imaginary table.
data_root=WORK/'generated'
prepared=subprocess.run([sys.executable,'scripts/prepare_data.py','--kind','attributes-sft','--seed','42','--output',str(data_root)],capture_output=True,text=True)
assert prepared.returncode==0,prepared.stderr
manifest=json.loads((data_root/'attributes-sft/manifest.json').read_text())
validation=[json.loads(line) for line in (data_root/'attributes-sft/validation.jsonl').read_text().splitlines()]
assert manifest['seed']==42 and manifest['splits']['validation']['records']==5
assert manifest['splits']['validation']['sha256']==sha((data_root/'attributes-sft/validation.jsonl').read_bytes())
seed_everything(42);model=TinyLM(ModelConfig(width=8,layers=1,heads=1,max_length=128))
save_checkpoint(WORK/'probe-fp.pt',model,step=0,metadata={'seed':42,'scope':'untrained CPU software probe'})
qreports={}
for bits in (8,4):
    run=subprocess.run([sys.executable,'scripts/quantize.py',str(WORK/'probe-fp.pt'),'--bits',str(bits),'--output',str(WORK/f'probe-int{bits}.pt')],capture_output=True,text=True)
    assert run.returncode==0,run.stderr;qreports[str(bits)]=json.loads(run.stdout)
    qm,payload=load_checkpoint(WORK/f'probe-int{bits}.pt')
    assert torch.equal(qm.embedding.weight,model.embedding.weight)
    assert qm.embedding.weight.dtype==torch.float32 and payload['bits']==bits
    assert payload['optimizer'] is None
rejected=subprocess.run([sys.executable,'scripts/quantize.py',str(WORK/'probe-int8.pt'),'--bits','4','--output',str(WORK/'rejected.pt')],capture_output=True,text=True)
assert rejected.returncode!=0 and '避免反覆量化' in rejected.stderr
eval_report=evaluate(model,validation,mode='sft',max_new_tokens=2)
assert eval_report['effective_tokens']==36 and eval_report['metric_denominators']['exact_match']==5
assert eval_report['skipped']==[]
cases={}
for name,ids in [('normal',tok.encode('circle')+[2]),('no_eos',tok.encode('circle')),('space',tok.encode(' circle')+[2]),('illegal',[4]+tok.encode('circle')+[2])]:
    sample=answer_sample(tok,ids,'circle');cases[name]={k:sample[k] for k in ('exact_match','completed_exact_match','eos','invalid_special_tokens')}
assert cases['normal']['completed_exact_match'] and cases['no_eos']['exact_match'] and not cases['no_eos']['completed_exact_match']
assert not cases['space']['exact_match'] and not cases['illegal']['exact_match']
out['cpu_quantization_evaluation']={'scope':'untrained width8 one-layer CPU software only', 'quantize_cli':qreports,
    'requantization_exit':rejected.returncode,'requantization_rejection':rejected.stderr.splitlines()[-1],
    'validation_manifest':manifest['splits']['validation'],'effective_tokens':36,'generation_rows':5,'answer_edge_cases':cases}

# Actual two-update finite-policy run tests closure state omission and real nonzero updates.
ctx=SimpleNamespace(device='cpu',output=WORK/'policy-two-updates',seed=42,step_scale=1/600)
probe=app._reinforce(cs,ctx)
keys={}
for label in ('strict','weak_proxy'):
    payload=torch.load(ctx.output/f'policy-{label}.pt',weights_only=True)
    assert payload['step']==2 and 'baseline' not in payload and 'sampler_rng' not in payload
    assert 'torch_rng' in payload and 'optimizer' in payload
    keys[label]=sorted(payload)
out['cpu_saved_state']={'updates_each':2,'sampled_actions_each':128,'saved_keys':keys,
    'omitted':'moving baseline and private random.Random sampler state are closure values not in payload'}
logits=torch.tensor([0.,0.],requires_grad=True);loss=-(1.-.5)*logits.log_softmax(0)[1];loss.backward()
after=(logits-.1*logits.grad).softmax(0).detach().tolist()
assert logits.grad.tolist()==[.25,-.25]
out['policy_gradient_example']={'gradient':logits.grad.tolist(),'after_probabilities':after}

# Own one public-model CPU operation, never added to held-out score.
tool_model,_=load_checkpoint('checkpoints/course/tools/model.pt','cpu')
record=next(r for r in app._tool_records() if r['operation']=='add' and r['a']==2 and r['b']==3)
operation=app._tool_episode(tool_model,record,SimpleNamespace(device='cpu'))
assert operation['actual_tool_calls']==1 and operation['trace'][0]['tool_result']==5 and operation['answer']==5
assert operation['trace'][1]['generation']['input_ids']==prompt_ids([
    {'role':'system','content':app._TOOLS_SYSTEM},{'role':'user','content':'CALC:add(2,3)'},
    {'role':'assistant','content':operation['trace'][0]['generation']['samples'][0]['generated']},
    {'role':'user','content':'TOOL_RESULT:5'}])
out['own_cpu_tool_operation']={'question':'CALC:add(2,3)','calls':1,'result':5,'final_answer':operation['answer'],
    'raw_request':operation['trace'][0]['generation']['samples'][0]['generated'],
    'raw_final':operation['trace'][1]['generation']['samples'][0]['generated'],
    'scope':'single public-model operation; excluded from formal held-out scores'}

workflow=json.loads(Path('docs/course-experiments/student-checks/application-workflows.json').read_text())
public=workflow['operations']['tools'];posted=workflow['tools_posted_command']['output']
assert public['trace'][0]['executed'] and public['trace'][0]['tool_result']==5 and public['answer']==5
assert posted['trace'][0]['executed'] and posted['trace'][0]['tool_result']==5 and posted['answer']==5
assert workflow['tools_posted_command']['returncode']==0
out['public_workflow_record']={'path':'docs/course-experiments/student-checks/application-workflows.json',
    'sha256':sha(Path('docs/course-experiments/student-checks/application-workflows.json').read_bytes()),
    'recorded_request':public['trace'][0]['generation']['samples'][0]['generated'],
    'recorded_result':5,'recorded_final':public['trace'][1]['generation']['samples'][0]['generated'],
    'posted_command_returncode':0,'note':'operational inputs/outputs inspected; status metadata not used as authority'}

out['hypothetical_table']={'baseline_bytes':80000,'changed_bytes':50000,'saved_bytes':80000-50000,
    'quality_delta':4/5-4/5,'exercise_saved_bytes':80000-60000,'exercise_quality_delta':3/5-4/5,
    'conclusion':'table meets both hypothetical criteria; exercise only meets smaller-file criterion; no measured speed or memory'}
out['pass_at_k_n_equals_k']={str(k):[1-math.comb(k-c,k)/math.comb(k,k) for c in range(k+1)] for k in (1,2,4,8)}
assert all(v==[0.]+[1.]*int(k) for k,v in out['pass_at_k_n_equals_k'].items())
(BASE/'audit-result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
