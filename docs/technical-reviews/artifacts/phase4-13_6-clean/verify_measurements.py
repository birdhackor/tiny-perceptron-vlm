import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.data import ByteTokenizer
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.behavior import _preference_parts, _pair_examples

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
d = json.loads((BASE/'inputs/raw-dpo.json').read_text())
s = json.loads((BASE/'inputs/raw-style.json').read_text())
pointers = []
def pick(data, pointer, label='dpo'):
    value = data
    for key in pointer.strip('/').split('/'):
        value = value[key]
    pointers.append(label + ':' + pointer)
    return value

provenance = {k: pick(d, '/' + k) for k in ['experiment_id', 'revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu', 'step_scale']}
assert provenance['seed'] == 42 and provenance['step_scale'] == 1.0
raw_parts = _preference_parts(split_records(arithmetic_records(), seed=42))
splits = {}
for split, rows in raw_parts.items():
    expected = pick(d, '/results/data/' + split)
    raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    digest = hashlib.sha256(raw).hexdigest()
    families = {r['family'] for r in rows}
    assert digest == expected['sha256']
    assert len(rows) == expected['records'] and len(families) == expected['families']
    splits[split] = {'records': len(rows), 'families': len(families), 'raw_jsonl_sha256': digest}
assert [splits[k]['records'] for k in ('train','validation','test')] == [49,8,7]
assert splits['test']['families'] == 4
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not ({r['family'] for r in raw_parts[a]} & {r['family'] for r in raw_parts[b]})

def preference(pointer):
    samples = pick(d, pointer + '/samples')
    derived = {'records': len(samples), 'chosen_higher_absolute_probability': 0, 'relative_preference_improved': 0, 'mean_relative_margin': 0.0}
    for row in samples:
        absolute = row['policy_chosen_logp'] - row['policy_rejected_logp']
        relative = absolute - row['reference_margin']
        assert abs(absolute-row['policy_margin']) < 1e-9
        assert abs(relative-row['relative_margin']) < 1e-9
        derived['chosen_higher_absolute_probability'] += int(absolute > 0)
        derived['relative_preference_improved'] += int(relative > 0)
        derived['mean_relative_margin'] += relative / len(samples)
        assert row['chosen_answer_tokens'] == len(row['chosen'].encode()) + 1
        assert row['rejected_answer_tokens'] == len(row['rejected'].encode()) + 1
    for key,value in derived.items():
        observed = pick(d, pointer + '/' + key)
        assert abs(observed-value) < 1e-9
    return derived

def generation(data, pointer, label):
    samples = pick(data, pointer+'/samples', label)
    tok = ByteTokenizer()
    matches = 0
    ended = 0
    for row in samples:
        ids = row['generated_ids']
        eos = tok.eos_id in ids
        raw = ids[:ids.index(tok.eos_id)] if eos else ids
        exact = raw == tok.encode(row['expected'])
        assert exact == row['exact'] and eos == row['eos']
        assert tok.decode(raw) == row['generated']
        matches += exact
        ended += eos
    assert matches == pick(data, pointer+'/matches', label)
    assert len(samples) == pick(data, pointer+'/records', label)
    assert ended/len(samples) == pick(data, pointer+'/eos_rate', label)
    return {'records':len(samples),'matches':matches,'eos':ended,
            'example_2_plus_2':[r for r in samples if r['messages'][0]['content']=='2+2=?']}

table = {}
for split in ['validation','test']:
    table['before/'+split] = preference('/results/before/'+split)
    table['before/generated/'+split] = generation(s,'/results/content_evaluation/'+split,'style')
for name in ['model','beta1']:
    beta = pick(d,'/results/runs/'+name+'/beta')
    steps = pick(d,'/results/runs/'+name+'/training/steps')
    history = pick(d,'/results/runs/'+name+'/training/history')
    effective = pick(d,'/results/runs/'+name+'/training/effective_answer_tokens_both_sides')
    assert steps == 250 and history[-1]['step']==250
    pairs = _pair_examples(raw_parts['train'],128)
    sampler=random.Random(42)
    replay = sum(int((y != -100).sum()) for _ in range(250) for pair in sampler.choices(pairs,k=8) for x,y in pair)
    assert replay == effective == 9448
    table[name+'/training']={'beta':beta,'steps':steps,'batch_pairs':8,'sampled_pair_exposures':2000,'target_token_exposures_both_sides':replay}
    for split in ['validation','test']:
        table[name+'/'+split] = preference('/results/runs/'+name+'/preference/'+split)
        table[name+'/generated/'+split] = generation(d,'/results/runs/'+name+'/arithmetic/'+split,'dpo')

assert [(table[k+'/validation']['chosen_higher_absolute_probability'], table[k+'/test']['chosen_higher_absolute_probability'],table[k+'/test']['relative_preference_improved']) for k in ['before','model','beta1']] == [(0,1,0),(3,1,1),(0,0,0)]
assert [table[k+'/validation']['relative_preference_improved'] for k in ['model','beta1']] == [5,5]
assert abs(table['model/test']['mean_relative_margin'] - (-16.27347)) < 5e-6
assert abs(table['beta1/test']['mean_relative_margin'] - (-5.03428)) < 5e-6
for key,value in table.items():
    if '/generated/' in key:
        assert value['matches']==0 and value['eos']==value['records']
for name in ['model','beta1']:
    example=table[name+'/generated/test']['example_2_plus_2']
    assert len(example)==1 and example[0]['generated']=='8' and example[0]['eos']
model=TinyLM(ModelConfig(width=64,layers=2))
parameters=sum(p.numel() for p in model.parameters())
assert parameters == pick(s,'/results/content_training/parameters','style') == 141568
print(json.dumps({'environment': {'python':sys.version,'torch':torch.__version__,'device':'cpu'},
 'provenance':provenance,'raw_pointers_read':pointers,'splits':splits,'parameter_count':parameters,'table_recomputed':table,
 'scope':'Recomputed raw saved measurements, target-token sampling and architecture counts; no saved model inference, checkpoint loading, training or GPU.'},ensure_ascii=False,indent=2))
