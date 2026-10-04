from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[5]
DEST=Path(__file__).parent
names=['simple_models','text_foundation','real_text','tokenizer','sft','sft_ablation','style','safety','lora','encoders','projector','vqa','real_modal','ocr','dpo','posttraining','capstone_student','modern','moe','efficiency','flash_probe','precision','quantization','distillation','multimodal_distillation','joint','rag','tools','reasoning']
out={'scope':'Independent CPU recomputation of existing fixed experiment records; not reviewer GPU replication. Original files remain in existing repository paths.', 'records':{}, 'arithmetic_checks':[]}
def walk(x,path,record):
    if isinstance(x,dict):
        for nk in ['nll','mean_token_nll']:
            if nk in x and 'nll_sum' in x and x.get('effective_tokens',0)>0:
                observed=x['nll_sum']/x['effective_tokens'];expected=x[nk]
                out['arithmetic_checks'].append({'record':record,'locator':path+'/'+nk,'expected':expected,'observed':observed,'tolerance':1e-9,'passed':abs(expected-observed)<=1e-9,'denominator':x['effective_tokens']})
        if 'numerator' in x and 'denominator' in x and 'rate' in x and isinstance(x['denominator'],(int,float)) and x['denominator']>0:
            a=x['numerator']/x['denominator'];b=x['rate']
            out['arithmetic_checks'].append({'record':record,'locator':path+'/rate','expected':b,'observed':a,'tolerance':1e-12,'passed':b is not None and abs(a-b)<=1e-12,'denominator':x['denominator']})
        samples=x.get('samples',x.get('generated_samples'))
        if isinstance(samples,list) and samples and all(isinstance(s,dict) for s in samples):
            key=next((k for k in ['exact_match','exact','correct','strict_correct','match'] if all(k in s and isinstance(s[k],bool) for s in samples)),None)
            target=next((k for k in ['matches','correct'] if k in x and isinstance(x[k],int)),None)
            if key and target:
                actual=sum(s[key] for s in samples)
                out['arithmetic_checks'].append({'record':record,'locator':path+'/'+target,'expected':x[target],'observed':actual,'tolerance':0,'passed':actual==x[target],'denominator':len(samples),'details':'Count independently recomputed from all preserved sample flags; check original raw IDs separately for content evidence.'})
        for k,v in x.items():walk(v,path+'/'+str(k),record)
    elif isinstance(x,list):
        for i,v in enumerate(x):walk(v,path+'/'+str(i),record)
def digest(x):
    if isinstance(x,dict):
        d={}
        for k,v in x.items():
            if k in ['history','loss_trace','scaler_history','samples','generated_samples','code_sha256','artifacts','public_exports','build_configuration','profiler_events','kernel_events','raw_logits','logits','records','examples'] and isinstance(v,list):
                d[k]={'count':len(v),'first':v[0] if k in ['samples','generated_samples'] and v else None}
            elif k in ['train','validation','test'] and isinstance(v,list):d[k]={'count':len(v),'first':v[0] if v else None}
            elif k in ['code_sha256','artifacts','public_exports','runtime','config','metadata','classes','provenance']:
                d[k]=digest(v)
            else:d[k]=digest(v)
        return d
    if isinstance(x,list):
        if len(x)>16:return {'count':len(x),'first':digest(x[0]),'last':digest(x[-1])}
        return [digest(v) for v in x]
    return x
for name in names:
    f=ROOT/'docs/course-experiments/results'/f'{name}.json';r=json.loads(f.read_text());walk(r['results'],'/results',name)
    meta={k:r.get(k) for k in ['device','seed','torch_version','python_version','gpu','revision','elapsed_seconds','evidence_status','step_scale','timing_scope']}
    out['records'][name]={'path':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'metadata':meta,'digest':digest(r['results'])}
out['total_checks']=len(out['arithmetic_checks']);out['failed_checks']=[x for x in out['arithmetic_checks'] if not x['passed']]
(DEST/'record-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for n,r in out['records'].items():
    (DEST/('record-digest-'+n+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checked_records':len(names),'total_arithmetic_checks':out['total_checks'],'failures':out['failed_checks']},indent=2))
