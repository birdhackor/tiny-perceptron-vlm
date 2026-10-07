import json,sys,pathlib,torch,random,hashlib
from scripts.prepare_data import generate_records
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records,text_examples,records_sha256
from tiny_perceptron.data import IGNORE,pad_batch
b=pathlib.Path('docs/technical-reviews/artifacts/p7_technical_a');r=json.load(open('docs/course-experiments/results/sft_ablation.json'));data={};parts={}
for name,rows in [('attributes',generate_records('attributes-sft')),('arithmetic',arithmetic_records())]:
 p=split_records(rows,seed=r['seed']);parts[name]=p;data[name]={}
 for s,rs in p.items():
  sha=hashlib.sha256(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs).encode()).hexdigest();assert sha==r['results']['data'][name][s]['sha256']
  examples=text_examples(rs,mode='sft');data[name][s]={'records':len(rs),'sha256':sha,'effective_targets':sum(int((y!=IGNORE).sum()) for x,y in examples)}
counters={}
for name,rows in [('b-only',parts['arithmetic']['train']),('replay',parts['attributes']['train']+parts['arithmetic']['train'])]:
 q=r['results']['runs'][name]['training'];assert records_sha256(rows)==q['records_sha256'];examples=text_examples(rows,mode='sft');rng=random.Random(r['seed']);count=sum(int((y!=IGNORE).sum()) for step in range(q['steps']) for x,y in rng.choices(examples,k=16));assert count==q['effective_tokens']
 # Check actual padded label count differs from grid count and equals individual count.
 x,y,valid=pad_batch(examples[:16]);assert int((y!=IGNORE).sum())==sum(int((z!=IGNORE).sum()) for a,z in examples[:16])
 counters[name]={'updates':q['steps'],'batch':16,'records':len(rows),'exposures_rebuilt':count,'batch_total_cells':y.numel(),'batch_effective_labels':int((y!=IGNORE).sum()),'train_unique_targets':sum(int((z!=IGNORE).sum()) for a,z in examples)}
variants={}
for longlength,longcount in [(20,10),(1,10),(20,5)]:
 short=90;long=longlength*longcount;variants[f'length{longlength}-count{longcount}']={'tokens':[short,long],'ratios':[short/(short+long),long/(short+long)]}
assert abs(180/(180+20)-.9)<1e-12
out={'command':'PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/check-5-16.py','python':sys.version,'torch':torch.__version__,'device':'CPU data/label/sampler reconstruction, no training','variants':variants,'records_ratio_for_token90_10':180,'source_splits':data,'historical_counters':counters,'ratio':counters['replay']['exposures_rebuilt']/counters['b-only']['exposures_rebuilt'],'original_revision':r['revision'],'raw_sha256':hashlib.sha256(pathlib.Path('docs/course-experiments/results/sft_ablation.json').read_bytes()).hexdigest(),'scope':'No reading7.16 curriculum, no quality/forgetting measurements used, no original base checkpoint replay or500-step training'}
p=b/'5-16-original-checks.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
