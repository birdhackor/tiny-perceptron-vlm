import copy,hashlib,json,platform,random,torch
from pathlib import Path
from scripts.prepare_data import generate_records
from scripts.course_experiments.common import split_records,records_sha256,text_examples
from tiny_perceptron.data import ByteTokenizer
B=Path('docs/technical-reviews/artifacts/p7_technical_b');src=Path('docs/course-experiments/results/sft_ablation.json');j=json.loads(src.read_text());r=j['results'];dest=B/'sources/sft-ablation-result.json';assert not dest.exists();dest.write_bytes(src.read_bytes())
parts=split_records(generate_records('attributes-sft'),seed=42);clean=parts['train'];noisy=copy.deepcopy(clean);corrupt=[]
for i,row in enumerate(noisy):
 ans=row['messages'][-1]['content']
 if ans in ('circle','square') and len(corrupt)<max(1,len(noisy)//10):
  row['messages'][-1]['content']='square' if ans=='circle' else 'circle';corrupt.append({'row':i,'family':row['family'],'correct':ans,'wrong':row['messages'][-1]['content']})
assert corrupt==r['corruptions'] and len(corrupt)==4 and len({a['family'] for a in corrupt})==2
out={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','raw_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'historical_revision':j['revision'],'corrupted_count':len(corrupt),'train_records':len(clean),'corrupt_percent':100*len(corrupt)/len(clean),'corrupted_families':sorted({a['family'] for a in corrupt}),'runs':{},'code_hashes':{}}
tok=ByteTokenizer()
for name,rows in [('clean',clean),('noisy',noisy)]:
 v=r['runs'][name];assert records_sha256(rows)==v['training']['records_sha256'];examples=text_examples(rows,'sft',128);sampler=random.Random(42);tokens=sum(sum(int((y!=-100).sum()) for x,y in sampler.choices(examples,k=16)) for step in range(300));assert tokens==v['training']['effective_tokens']==33733
 test=v['attributes']['test'];assert test['records']==len(test['samples'])==10;exact=0
 for record,s in zip(parts['test'],test['samples'],strict=True):
  assert record['messages'][:-1]==s['messages'] and record['messages'][-1]['content']==s['expected'];ids=s['generated_ids'];raw=ids[:ids.index(2)] if 2 in ids else ids;ok=raw==tok.encode(s['expected']);assert ok==s['exact'] and tok.decode(raw)==s['generated'];exact+=ok
 assert exact==test['matches'] and abs(test['nll_sum']/test['effective_tokens']-test['nll'])<1e-12
 out['runs'][name]={'steps':v['training']['steps'],'sampled_target_tokens':tokens,'test_matches':exact,'test_records':10,'test_nll_sum':test['nll_sum'],'test_target_tokens':test['effective_tokens'],'test_mean_nll':test['nll'],'train_records_sha256':records_sha256(rows)}
assert [out['runs'][n]['test_matches'] for n in ['clean','noisy']]==[7,6]
for p in ['scripts/course_experiments/text.py','scripts/course_experiments/common.py','tiny_perceptron/data.py','tiny_perceptron/model.py']:
 h=hashlib.sha256(Path(p).read_bytes()).hexdigest();assert h==j['code_sha256'][p];out['code_hashes'][p]=h
print(json.dumps(out,ensure_ascii=False,indent=2))
