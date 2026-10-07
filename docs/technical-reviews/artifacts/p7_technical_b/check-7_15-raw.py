import hashlib,json,platform,torch
from pathlib import Path
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records,records_sha256
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer
B=Path('docs/technical-reviews/artifacts/p7_technical_b');j=json.loads((B/'sources/sft-ablation-result.json').read_text());r=j['results'];a=split_records(generate_records('attributes-sft'),seed=42);b=split_records(arithmetic_records(),seed=42);out={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','B_split_counts':{},'measurements':{},'steps':r['runs']['b-only']['training']['steps'],'train_records':r['runs']['b-only']['training']['records']};tok=ByteTokenizer()
for split,rows in b.items():
 h=hashlib.sha256(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows).encode()).hexdigest();assert h==r['data']['arithmetic'][split]['sha256'];out['B_split_counts'][split]={'records':len(rows),'families':len({z['family'] for z in rows}),'sha256':h}
assert records_sha256(b['train'])==r['runs']['b-only']['training']['records_sha256']
for stage,values in [('before',r['before']),('after',r['runs']['b-only'])]:
 out['measurements'][stage]={}
 for task,parts in [('A_attributes',a),('B_arithmetic',b)]:
  z=values[task]['test'];exact=eos=0
  for row,s in zip(parts['test'],z['samples'],strict=True):
   assert row['messages'][:-1]==s['messages'] and row['messages'][-1]['content']==s['expected'];ids=s['generated_ids'];raw=ids[:ids.index(2)] if 2 in ids else ids;ok=raw==tok.encode(s['expected']);assert ok==s['exact'] and tok.decode(raw)==s['generated'];exact+=ok;eos+=2 in ids
  assert exact==z['matches'] and eos/len(parts['test'])==z['eos_rate'] and abs(z['nll_sum']/z['effective_tokens']-z['nll'])<1e-12
  out['measurements'][stage][task]={'matches':exact,'records':len(parts['test']),'EOS':eos,'meanNLL':z['nll'],'NLLsum':z['nll_sum'],'targets':z['effective_tokens']}
assert out['measurements']['before']['A_attributes']['matches']==5 and out['measurements']['after']['A_attributes']['matches']==0
assert out['measurements']['before']['B_arithmetic']['matches']==out['measurements']['after']['B_arithmetic']['matches']==0
assert out['steps']==500 and out['train_records']==49
out['first_A_after']=r['runs']['b-only']['A_attributes']['test']['samples'][0]
before=json.loads((B/'check-7_15-output.json').read_text());before['fixed_A'].append('紅色物體是圓。請回答形狀。');assert all(v is None for row in before['measurements'].values() for v in row.values());out['add_question_only']={'A_count':len(before['fixed_A']),'all_measurements_still_None':True}
print(json.dumps(out,ensure_ascii=False,indent=2))
