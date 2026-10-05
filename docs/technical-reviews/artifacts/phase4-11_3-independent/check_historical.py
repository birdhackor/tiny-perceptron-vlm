"""Read only explicit measurement/config/provenance/sample pointers; no models."""
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def sha(b):
    return hashlib.sha256(b).hexdigest()
def ptr(obj,pointer):
    x=obj
    for key in pointer.strip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~')
        x=x[int(key)] if isinstance(x,list) else x[key]
    return x
receipts=[]
def read(name,pointers):
    src=ROOT/'docs/course-experiments/results'/name
    raw=src.read_bytes()
    obj=json.loads(raw)
    shutil.copyfile(src,OUT/('raw-'+name))
    selected={p:ptr(obj,p) for p in pointers}
    receipts.append({'source':str(src.relative_to(ROOT)),'full_sha256':sha(raw),'copied_original':str((OUT/('raw-'+name)).relative_to(ROOT)),'inspected_pointers':pointers,'excluded':'notes, review, scope, *_scope_correction, evidence_status, status and author summaries not read'})
    return selected

provenance=['/revision','/seed','/device','/torch_version','/python_version','/step_scale']
train_fields=['config','modal_config','initial_loss','final_loss','history','steps','effective_tokens','effective_targets','weights_changed','nonzero_gradient_seen','trainable_parameters']
eval_fields=['examples','correct','exact_match','mean_token_nll','effective_tokens','eos_rate','generation_errors','invalid_special_tokens','ablation','skipped','samples']
code_files=['tiny_perceptron/model.py','tiny_perceptron/data.py','tiny_perceptron/multimodal.py','scripts/course_experiments/modalities.py','scripts/course_experiments/common.py']
paths=provenance+['/results/training/'+f for f in train_fields]+['/results/'+split+'/'+f for split in ['before','validation','test'] for f in eval_fields]+['/results/language_weights_unchanged','/results/data/seed','/results/data/split_policy']+['/results/data/splits/'+split+'/'+f for split in ['train','validation','test'] for f in ['count','sha256','records']]+['/code_sha256/'+f.replace('~','~0').replace('/','~1') for f in code_files]
# JSON pointer helper uses raw path components; code hashes are explicitly selected
# from the registered map below, without recursive traversal.
p=read('projector.json',paths)
code_checks={}
for filename in code_files:
    original=p['/code_sha256/'+filename.replace('/','~1')]
    current=sha((ROOT/filename).read_bytes())
    assert original==current
    code_checks[filename]={'original':original,'inspected_current':current,'equal':True}
splits={s:p['/results/data/splits/'+s+'/records'] for s in ['train','validation','test']}
for s,rows in splits.items():
    assert len(rows)==p['/results/data/splits/'+s+'/count']
    assert sha(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode())==p['/results/data/splits/'+s+'/sha256']
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not ({r['family'] for r in splits[a]} & {r['family'] for r in splits[b]})
def encode(t):return [b+8 for b in t.encode('utf-8')]
evals={}
for split in ['before','validation','test']:
    samples=p['/results/'+split+'/samples']
    for x in samples:
        ids=x['generated_ids'];raw=ids[:ids.index(2)] if 2 in ids else ids
        assert x['exact_match']==(raw==encode(x['target']))
        assert x['eos']==(2 in ids)
        assert x['generated']==bytes(t-8 for t in raw if t>=8).decode('utf-8',errors='replace')
    correct=sum(x['exact_match'] for x in samples)
    effective=sum(len(x['target'].encode())+1 for x in samples)
    eos=sum(x['eos'] for x in samples)
    assert len(samples)==p['/results/'+split+'/examples']
    assert correct==p['/results/'+split+'/correct']
    assert effective==p['/results/'+split+'/effective_tokens']
    assert correct/len(samples)==p['/results/'+split+'/exact_match']
    assert eos/len(samples)==p['/results/'+split+'/eos_rate']
    assert p['/results/'+split+'/skipped']==[]
    evals[split]={'examples':len(samples),'correct':correct,'effective_answer_plus_EOS_tokens':effective,'eos_count':eos,'mean_token_nll_nats':p['/results/'+split+'/mean_token_nll'],'samples':samples}
history=p['/results/training/history']
assert len(history)==p['/results/training/steps']
effective=sum(x['effective_targets'] for x in history)
assert effective==p['/results/training/effective_targets']==p['/results/training/effective_tokens']
assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in history)
assert p['/results/training/weights_changed'] and p['/results/training/nonzero_gradient_seen']
assert p['/results/language_weights_unchanged']
assert p['/results/training/final_loss']<p['/results/training/initial_loss']
assert evals['test']['mean_token_nll_nats']<evals['before']['mean_token_nll_nats']

sft_fields=['matches','records','examples','exact_match','effective_tokens','eos_rate','samples']
s=read('sft.json',provenance+['/results/checkpoint']+['/results/after/test/'+f for f in sft_fields])
samples=s['/results/after/test/samples']
for sample in samples:
    generated_ids=sample['generated_ids']
    raw=generated_ids[:generated_ids.index(2)] if 2 in generated_ids else generated_ids
    assert sample['exact']==(raw==encode(sample['expected']))
    assert sample['eos']==(2 in generated_ids)
assert sum(x['exact'] for x in samples)==s['/results/after/test/matches']
assert len(samples)==s['/results/after/test/examples']==s['/results/after/test/records']
assert s['/results/after/test/matches']/len(samples)==s['/results/after/test/exact_match']
assert sum(len(x['expected'].encode())+1 for x in samples)==s['/results/after/test/effective_tokens']
assert sum(x['eos'] for x in samples)/len(samples)==s['/results/after/test/eos_rate']
print('SFT sample schema',[(k,type(v).__name__) for k,v in samples[0].items()])
report={'projector_provenance':{f:p['/'+f] for f in ['revision','seed','device','torch_version','python_version','step_scale']},'training':{f:p['/results/training/'+f] for f in train_fields if f!='history'},'history_recomputed_steps':len(history),'history_recomputed_effective_targets':effective,'split_counts':{s:len(rows) for s,rows in splits.items()},'split_offsets':{s:sorted({row['offset'] for row in rows}) for s,rows in splits.items()},'split_family_disjoint':True,'language_weights_unchanged_recorded':p['/results/language_weights_unchanged'],'evaluation':evals,'sft_text_base_selected_checkpoint':s['/results/checkpoint'],'sft_text_base_provenance':{f:s['/'+f] for f in ['revision','seed','device','torch_version']},'sft_text_test':{f:s['/results/after/test/'+f] for f in sft_fields},'scope':'Arithmetic and scoring reconstruction from historical original JSON; no rerun of old models or training; historical update flag is not an independent checkpoint comparison.'}
report['selected_original_code_sha_checks']=code_checks
(OUT/'historical-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(OUT/'raw-pointer-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
