import hashlib,inspect,json,platform
from pathlib import Path
from scripts.prepare_data import generate_records
from scripts.course_experiments.common import split_records,records_sha256,fit_lm
B=Path('docs/technical-reviews/artifacts/p7_technical_b');j=json.loads((B/'sources/sft-result.json').read_text());r=j['results'];parts=split_records(generate_records('attributes-sft'),seed=42)
for k,rows in parts.items():
 h=hashlib.sha256(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows).encode()).hexdigest();assert h==r['data'][k]['sha256']
train=parts['train'];text=[{'text':v['messages'][0]['content']+v['messages'][1]['content'],'family':v['family']} for v in train];st=r['pretrain_then_sft'];assert len(train)==45 and records_sha256(text)==st['pretraining']['records_sha256'];assert records_sha256(train)==r['training']['records_sha256']==st['sft']['records_sha256']
sig=inspect.signature(fit_lm);assert sig.parameters['batch_size'].default==16 and sig.parameters['lr'].default==.003
assert [r['training']['steps'],st['pretraining']['steps'],st['sft']['steps']]==[900,250,900]
print(json.dumps({'python':platform.python_version(),'device':'cpu','historical_seed':j['seed'],'counts':{k:len(v) for k,v in parts.items()},'random_sft_steps':r['training']['steps'],'text_then_sft_steps':[st['pretraining']['steps'],st['sft']['steps']],'SFT_records_sha256':records_sha256(train),'text_records_sha256':records_sha256(text),'batch_size_default':sig.parameters['batch_size'].default,'lr_default':sig.parameters['lr'].default,'same_materials_hash_and_splits':True},indent=2))
