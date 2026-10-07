import hashlib,json,platform,random,torch
from pathlib import Path
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records,records_sha256,text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer
B=Path('docs/technical-reviews/artifacts/p7_technical_b/');j=json.loads((B/'sources/sft-ablation-result.json').read_text());r=j['results'];a=split_records(generate_records('attributes-sft'),seed=42);b=split_records(arithmetic_records(),seed=42);tok=ByteTokenizer()
out={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','measurements':{},'budget':{},'recipe':{}}
for stage,v in [('before',r['before']),('b-only',r['runs']['b-only']),('replay',r['runs']['replay'])]:
 out['measurements'][stage]={}
 for task,parts in [('A_attributes',a),('B_arithmetic',b)]:
  out['measurements'][stage][task]={}
  for split in ['validation','test']:
   z=v[task][split];exact=eos=0
   for row,s in zip(parts[split],z['samples'],strict=True):
    assert row['messages'][:-1]==s['messages'] and row['messages'][-1]['content']==s['expected']
    ids=s['generated_ids'];raw=ids[:ids.index(2)] if 2 in ids else ids;ok=raw==tok.encode(s['expected'])
    assert ok==s['exact'] and tok.decode(raw)==s['generated'];exact+=ok;eos+=2 in ids
   assert exact==z['matches'] and eos/len(parts[split])==z['eos_rate'] and abs(z['nll_sum']/z['effective_tokens']-z['nll'])<1e-12
   out['measurements'][stage][task][split]={'exact':exact,'records':len(parts[split]),'EOS':eos,'NLLsum':z['nll_sum'],'targets':z['effective_tokens'],'meanNLL':z['nll']}
for name,rows in [('b-only',b['train']),('replay',a['train']+b['train'])]:
 v=r['runs'][name]['training'];assert records_sha256(rows)==v['records_sha256'];examples=text_examples(rows,'sft',128);sampler=random.Random(42)
 tokens=sum(sum(int((y!=-100).sum()) for x,y in sampler.choices(examples,k=16)) for _ in range(500))
 assert tokens==v['effective_tokens'];assert v['steps']==500
 out['budget'][name]={'records':len(rows),'steps':500,'batch_size':16,'sampled_targets':tokens,'records_sha256':records_sha256(rows)}
assert out['budget']['b-only']['sampled_targets']==18453 and out['budget']['replay']['sampled_targets']==36384
out['budget']['target_ratio']=36384/18453
aa=[{'source':'A','text':'舊題一','answer_tokens':2},{'source':'A','text':'舊題二','answer_tokens':2}]
bb=[{'source':'B','text':'新題一','answer_tokens':2},{'source':'B','text':'新題二','answer_tokens':2}]
for name,recipe in [('base',aa+bb),('variation',[aa[0],bb[0],bb[1],bb[0]])]:
 out['recipe'][name]={'sources':[z['source'] for z in recipe],'targets':sum(z['answer_tokens'] for z in recipe)}
for row in bb:row['answer_tokens']=4
v=[aa[0],bb[0],bb[1],bb[0]];out['recipe']['longB']={'A_targets':2,'B_targets':12,'total':sum(z['answer_tokens'] for z in v)}
pdf=Path('outputs/phase7-source-cache/p7_technical_b/replay-1811.11682.pdf')
txt=pdf.with_suffix('.txt').read_text();excerpt='\n'.join(txt.splitlines()[124:136]+txt.splitlines()[157:160])
(B/'sources/replay-excerpt.json').write_text(json.dumps({'url':'https://arxiv.org/abs/1811.11682','version':'v2 2019-11-26','locator':'Section3 opening and losses new/replay paragraph','original_pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'reviewer':'/root/p7_technical_b','scope':'Current network training on mixture of new/replayed data, no claims about supervised LM guaranteed outcome','excerpt':excerpt},ensure_ascii=False,indent=2)+'\n')
out['replay_paper_pdf_sha256']=hashlib.sha256(pdf.read_bytes()).hexdigest()
print(json.dumps(out,ensure_ascii=False,indent=2))
