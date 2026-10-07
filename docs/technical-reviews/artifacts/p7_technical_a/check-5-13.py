import json,random,sys,subprocess,ast,hashlib
from pathlib import Path
from tiny_perceptron.model import TinyLM,ModelConfig
from scripts.course_experiments.common import split_records,text_examples,records_sha256
b=Path('docs/technical-reviews/artifacts/p7_technical_a');r=json.load(open('docs/course-experiments/results/text_foundation.json'));raw=r['results']
pool=[{'text':f'object={i};color={color};shape={shape}.','family':str(i)} for i in range(80) for color,shape in [('red' if i%2 else 'blue','circle' if i%3 else 'square')]]
parts=split_records(pool,seed=r['seed']);data={}
for s,rows in parts.items():
 blob=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows).encode();sha=hashlib.sha256(blob).hexdigest();assert sha==raw['scaling_data'][s]['sha256']
 data[s]={'records':len(rows),'sha256':sha,'unique_targets':sum(len(y) for x,y in text_examples(rows)),'families':sorted(x['family'] for x in rows)}
checks={}
for w in [16,32]:
 count=sum(x.numel() for x in TinyLM(ModelConfig(width=w)).parameters())
 for n in [16,64]:
  q=raw['scaling'][f'scaling-w{w}-n{n}'];examples=text_examples(parts['train'][:n]);sampler=random.Random(r['seed']);exposure=sum(len(y) for step in range(150) for x,y in sampler.choices(examples,k=4))
  assert exposure==q['training']['effective_tokens'];assert count==q['parameters'];assert records_sha256(parts['train'][:n])==q['training']['records_sha256']
  evaluation={}
  for s in ['validation','test']:
   v=q['evaluation'][s];assert v['effective_tokens']==data[s]['unique_targets'];avg=v['nll_sum']/v['effective_tokens'];assert abs(avg-v['nll'])<1e-12
   evaluation[s]={'sum':v['nll_sum'],'targets':v['effective_tokens'],'mean':avg,'rounded5':f'{avg:.5f}'}
  checks[f'w{w}-n{n}']={'parameters':count,'training_records':n,'unique_training_targets':sum(len(y) for x,y in examples),'updates':150,'batch':4,'exposures_rebuilt':exposure,'train_nll_recorded_not_recalculated':q['training']['final_loss'],'train_nll_rounded5':f"{q['training']['final_loss']:.5f}",'evaluation':evaluation}
old=subprocess.check_output(['git','show',r['revision']+':scripts/course_experiments/text.py'],text=True);cur=Path('scripts/course_experiments/text.py').read_text()
def function(t,name):return next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name==name)
a=function(old,'run_text_foundation');z=function(cur,'run_text_foundation');assert ast.dump(a,include_attributes=False)==ast.dump(z,include_attributes=False)
first=data['validation']['families'];second=data['test']['families'];assert not(set(first)&set(second))
plan={'params8':TinyLM(ModelConfig(width=8)).description()['parameters'],'params16':TinyLM(ModelConfig(width=16)).description()['parameters'],'original_budget':4*8*100,'half_length_100':4*4*100,'half_length_steps':3200/(4*4),'flops8':2*8*8,'flops16':2*16*16}
out={'command':'PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/check-5-13.py','python':sys.version,'device':'CPU new parameter counts/data reconstruction only, no training','plan':plan,'splits':data,'checks':checks,'historical_run_function_AST_matches':True,'original_function_sha256':hashlib.sha256(ast.get_source_segment(old,a).encode()).hexdigest(),'scope':'Saved original GPU loss values, validation/test sums/counts recomputed; train mean remains reported original final evaluation, no saved weight replay or long training.'}
p=b/'5-13-original-checks.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
