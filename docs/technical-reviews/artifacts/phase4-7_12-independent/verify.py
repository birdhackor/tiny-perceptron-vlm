"""Bounded offline verification; never calls model construction, fit or generation."""
import ast
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OUT = BASE / 'verification'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
assert torch.version.cuda is None
ns = {'json': json, 'hashlib': hashlib, 'random': random, 'torch': torch}
exec(compile((BASE/'historical-code/tiny_perceptron/data.py').read_bytes(), 'historical-data.py', 'exec'), ns)

def load_functions(relative, names):
    tree = ast.parse((BASE/'historical-code'/relative).read_bytes())
    chosen = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in chosen} == set(names)
    exec(compile(ast.Module(body=chosen, type_ignores=[]), relative, 'exec'), ns)

load_functions('scripts/prepare_data.py', ['conversation', 'generate_records'])
load_functions('scripts/course_experiments/common.py', ['split_records','records_sha256','text_examples'])
raw_section = (BASE/'inputs/section.md').read_bytes()
fence = (BASE/'inputs/fence-1.py').read_bytes()
original = {}
exec(compile(fence, 'original-fence-1.py', 'exec'), original)
assert original['all_pairs'] == [('紅','圓'),('紅','方'),('藍','圓'),('藍','方')]
assert original['train'] == [('紅','圓'),('藍','圓'),('藍','方')]
assert original['held_out'] == {('紅','方')}

def check_split(held_out):
    pairs = original['all_pairs']
    train = [pair for pair in pairs if pair not in held_out]
    return {'held_out':sorted(held_out), 'train':train,
            'intersection':sorted(set(train)&held_out),
            'train_colors':sorted({c for c,s in train}),
            'train_shapes':sorted({s for c,s in train})}

variants = {'original':check_split({('紅','方')}),
            'blue_unseen':check_split({('藍','圓'),('藍','方')}),
            'restored':check_split({('紅','方')}),
            'none_held':check_split(set()),
            'all_held':check_split(set(original['all_pairs'])),
            'out_of_domain_held':check_split({('綠','圓')})}
assert variants['blue_unseen']['train'] == [('紅','圓'),('紅','方')]
assert variants['blue_unseen']['train_colors'] == ['紅']
assert set(variants['restored']['train_colors']) == {'紅','藍'}
assert set(variants['restored']['train_shapes']) == {'圓','方'}
assert all(not item['intersection'] for item in variants.values())
# Two wordings of the same held-out pair stay together when keyed by attributes.
paraphrases = [{'color':'紅','shape':'方','text':'紅色方塊是什麼形狀？','answer':'方塊'},
              {'color':'紅','shape':'方','text':'一個紅方形，請說出形狀','answer':'方塊'}]
assert all((row['color'],row['shape']) in original['held_out'] for row in paraphrases)
(OUT/'four-grid.json').write_text(json.dumps({'variants':variants,'paraphrase_rows':paraphrases,
 'assertion_scope':'Disjointness only: empty training and out-of-domain holdout also satisfy the original assertion.'},ensure_ascii=False,indent=2)+'\n')

record = json.loads((BASE/'inputs/docs/course-experiments/results/sft.json').read_text())
r = record['results']
rows = ns['generate_records']('attributes-sft')
parts = ns['split_records'](rows,seed=record['seed'])
assert len(rows) == 60 and len({row['family'] for row in rows}) == 12
assert record['seed'] == 42
families = {k:{row['family'] for row in v} for k,v in parts.items()}
assert not (families['train']&families['validation'] or families['train']&families['test'] or families['validation']&families['test'])
rebuilt_summary = {}
for split, selected in parts.items():
    content = ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected).encode()
    (OUT/(split+'.jsonl')).write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    assert sha == r['data'][split]['sha256']
    assert len(selected) == r['data'][split]['records']
    assert len(families[split]) == r['data'][split]['families']
    examples = ns['text_examples'](selected,mode='sft',max_length=128)
    token_count = sum(int((y != -100).sum()) for x,y in examples)
    rebuilt_summary[split]={'records':len(selected),'families':sorted(families[split]),'sha256':sha,
      'effective_answer_tokens':token_count,'attributes':{
          key:sorted({row['family'].split(':')[i] for row in selected})
          for i,key in enumerate(['color','shape','pitch'])}}
    assert all(len([row for row in selected if row['family']==family]) == 5 for family in families[split])
    if split != 'train':
        assert token_count == r['after'][split]['effective_tokens']

assert rebuilt_summary['validation']['effective_answer_tokens']==36
assert rebuilt_summary['test']['effective_answer_tokens']==69
assert ns['records_sha256'](parts['train'])==r['training']['records_sha256']
# Replay sampler choices and label lengths only; no parameters/forward/backward/update.
sft_lengths = [int((y!=-100).sum()) for x,y in ns['text_examples'](parts['train'],'sft',128)]
text_records = [{'text':row['messages'][0]['content']+row['messages'][1]['content'],'family':row['family']} for row in parts['train']]
assert ns['records_sha256'](text_records) == r['pretrain_then_sft']['pretraining']['records_sha256']
text_lengths = [int((y!=-100).sum()) for x,y in ns['text_examples'](text_records,'text',128)]

def sampled_tokens(lengths, steps):
    sampler=random.Random(record['seed'])
    return sum(sum(sampler.choices(lengths,k=16)) for _ in range(steps))

budgets={'direct':r['training']['steps'],
         'pretraining':r['pretrain_then_sft']['pretraining']['steps'],
         'continuation':r['pretrain_then_sft']['sft']['steps']}
assert budgets == {'direct':900,'pretraining':250,'continuation':900}
assert sampled_tokens(sft_lengths,900) == r['training']['effective_tokens'] == 101091
assert sampled_tokens(text_lengths,250) == r['pretrain_then_sft']['pretraining']['effective_tokens'] == 191742
assert r['pretrain_then_sft']['sft']['effective_tokens']==101091
assert budgets['direct']!=budgets['pretraining']+budgets['continuation']

evaluations = []
for path, stage in [('before',r['before']),('after',r['after']),
                    ('pretrain_then_sft.before_sft',r['pretrain_then_sft']['before_sft']),
                    ('pretrain_then_sft.after_sft',r['pretrain_then_sft']['after_sft'])]:
    for split,m in stage.items():
        samples=m['samples']; assert len(samples)==len(parts[split])==m['records']==m['examples']
        matches=ended=0
        for row,sample in zip(parts[split],samples,strict=True):
            assert sample['messages']==row['messages'][:-1]
            assert sample['expected']==row['messages'][-1]['content']
            ids=sample['generated_ids']; tok=ns['ByteTokenizer']()
            assert len(ids)<=32
            raw=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            exact=raw==tok.encode(sample['expected']); eos=tok.eos_id in ids
            assert exact==sample['exact'] and eos==sample['eos']
            assert tok.decode(raw)==sample['generated']
            matches+=int(exact); ended+=int(eos)
        assert matches==m['matches'] and matches/len(samples)==m['exact_match']
        assert ended/len(samples)==m['eos_rate']
        assert abs(m['nll_sum']/m['effective_tokens']-m['nll'])<=1e-12
        evaluations.append({'path':path,'split':split,'samples':len(samples),'matches':matches,
          'exact_match':matches/len(samples),'eos_count':ended,'effective_tokens':m['effective_tokens'],
          'scope':'Recomputed saved IDs/booleans/denominators only; NLL numerator and model behavior were not rerun.'})
(OUT/'historical-reconstruction.json').write_text(json.dumps({'original_revision':record['revision'],
  'seed':record['seed'],'rows':len(rows),'combinations':12,'dimensions':[3,2,2],
  'rebuilt_splits':rebuilt_summary,'update_budgets':budgets,
  'sampled_tokens':{'direct':101091,'pretraining':191742,'continuation':101091},
  'evaluation_rechecks':evaluations,
  'task_contract':{'describe':'shape','shape?':'shape','color?':'color','pitch?':'pitch','joint?':'shape,pitch'},
  'scope':'Historical records checked and deterministic data/sampler reconstruction only; no training, inference or weights.'},ensure_ascii=False,indent=2)+'\n')
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,
 'torch':torch.__version__,'torch_git_version':torch.version.git_version,'cuda_build':str(torch.version.cuda),
 'device':'cpu','threads':str(torch.get_num_threads()),'section_sha256':hashlib.sha256(raw_section).hexdigest()},indent=2)+'\n')
print(json.dumps({'checks':'passed','four_grid_train':original['train'],'blue_holdout_train':variants['blue_unseen']['train'],
 'historical_combinations':12,'split_records':[len(parts[k]) for k in ['train','validation','test']],
 'split_families':[len(families[k]) for k in ['train','validation','test']],
 'budgets':budgets,'verified_saved_evaluations':len(evaluations)},ensure_ascii=False,indent=2))
