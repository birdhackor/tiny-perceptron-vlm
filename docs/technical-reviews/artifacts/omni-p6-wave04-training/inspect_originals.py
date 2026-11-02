"""Read original result JSON and hash existing files; do not execute any trained model."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-p6-wave04-training-originals')
CORE = Path('/workspace/work/omni-core-completed-40988311100/verified')
FINAL = Path('/workspace/work/omni-final-completed-41004229262/verified')
out = {'created_at':datetime.now(timezone.utc).isoformat(), 'reviewer_task':'/root/omni_p6_technical_training', 'environment':{'python':platform.python_version(),'device':'cpu','operation':'JSON recomputation and byte hashing only; no neural forward or training'},'inputs':{},'historical_results':{},'scope':'Historical JSON is inspected, not reproduced. All original external fulltexts remain outside Git.'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):
    p=Path(p)
    if not p.is_absolute():p=ROOT/p
    out['inputs'][str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
    return json.loads(p.read_text())

expected={'simple_models':4.3,'text_foundation':16.0,'real_text':23.9,'tokenizer':16.2,'sft':31.7,'sft_ablation':23.5,'style':29.5,'safety':22.2,'lora':24.5,'encoders':10.2,'projector':15.9,'vqa':83.8,'dpo':18.7,'posttraining':2.1,'modern':44.6,'moe':65.7,'efficiency':32.3,'precision':15.0,'flash_probe':4.5,'quantization':7.8,'distillation':90.1,'joint':33.4,'multimodal_distillation':53.0}
for name,wanted in expected.items():
    d=read('docs/course-experiments/results/'+name+'.json')
    assert round(d['elapsed_seconds'],1)==wanted,(name,d['elapsed_seconds'])
    row={k:d.get(k) for k in ['device','gpu','seed','revision','elapsed_seconds','timing_scope','evidence_status','step_scale']}
    row['quoted_seconds']=wanted
    r=d['results']
    if name in ['simple_models','text_foundation','sft']:
        row['data_counts']={k:v['records'] for k,v in r['data'].items() if isinstance(v,dict) and 'records' in v}
        if 'training' in r:row['steps']=r['training']['steps'];row['effective_tokens']=r['training'].get('effective_tokens')
    if name=='real_text':row['steps']={k:v['training']['steps'] for k,v in r['runs'].items()}
    if name=='tokenizer':row['runs']={k:{'keys':list(v),'steps':v.get('training',{}).get('steps')} for k,v in r['runs'].items()}
    if name=='vqa':
        p=r['variants']['partial']
        row['data_counts']={k:v['count'] for k,v in r['data']['splits'].items()}
        row['family_overlap']= {a+'/'+b:sorted(set(v['family'] for v in r['data']['splits'][a]['records']) & set(v['family'] for v in r['data']['splits'][b]['records'])) for a,b in [('train','validation'),('train','test'),('validation','test')]}
        row['partial']={}
        for key in ['test','text_before','text_after']:
            samples=p[key]['samples']
            correct=sum(s['generated_ids']==[b+8 for b in s.get('target',s.get('expected')).encode()]+[2] for s in samples)
            reported=p[key].get('correct',p[key].get('matches'))
            assert correct==reported
            row['partial'][key]={'recomputed_completed_exact':correct,'examples':len(samples),'reported_correct_or_matches':reported,'effective_tokens':p[key]['effective_tokens']}
        row['freeze_scope']=p['freeze_scope']
    if name in ['modern','moe']:
        row['data_counts']={k:v['records'] for k,v in r['dataset'].items()}
        row['variants']={k:{'config':v['model']['config'],'steps':v['training']['steps'],'effective_tokens':v['training']['effective_tokens'],'batch_size':v['training']['batch_size']} for k,v in r['variants'].items()}
    if name=='quantization':row['steps']=r['training']['steps'];row['same_fp32_source']=r['same_fp32_source']
    if name=='distillation':
        row['matched_students']=r['matched_students'];row['tasks']={}
        for task,v in r['tasks'].items():
            pairs={}
            for key,a in v['runs'].items():
                if key.endswith('_ce'):
                    b=v['runs'][key+'_kl'];ta,tb=a['training'],b['training']
                    pair={k:ta[k] for k in ['initialization_sha256','batch_plan_sha256','steps','batch_size','effective_supervised_tokens']}
                    for k in pair:assert ta[k]==tb[k],(task,key,k)
                    pairs[key]=pair
            data={k:v['data'][k] for k in ['sha256','counts','family_intersections','student_training_records','student_training_selection']}
            challenge=v.get('out_of_domain_gsm8k',{}).get('selection') if v.get('out_of_domain_gsm8k') else None
            challenge={k:challenge[k] for k in ['source_rows','eligible_rows','selected_rows','skipped_rows','selected_source_rows','scope']} if challenge else None
            row['tasks'][task]={'data':data,'pairs':pairs,'teacher_frozen_and_unchanged':v['teacher_frozen_and_unchanged'],'out_of_domain_selection':challenge}
    if name=='flash_probe':row['verification']={k:r[k] for k in ['status','schedule_completed','verification_passed','supported_routes','configuration']};row['route_status']={k:v['status'] for k,v in r['routes'].items()}
    if name=='posttraining':row['scope']=r['scope'];row['label_source']=r['label_source'];row['contexts']={k:v['contexts'] for k,v in r['splits'].items()}
    out['historical_results'][name]=row

selection=read('docs/omni-products/v1/model-selection.json')
opening=read(FINAL/'frozen/final-opening.json')
assert opening['model_selection_sha256']==sha(ROOT/'docs/omni-products/v1/model-selection.json')
assert sha(CORE/'frozen/model-selection.json')==sha(FINAL/'frozen/model-selection.json')==opening['model_selection_sha256']
out['formal_jobs']={}
for job in sorted((CORE/'core').iterdir()):
    r=read(job/'training/training-result.json');j=read(job/'job-result.json')
    assert r['complete'] and j['success'] and r['optimizer_updates_observed']
    files={}
    for name,record in r['files'].items():
        p=job/'training'/name
        assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],p
        files[name]=record
    out['formal_jobs'][job.name]={'complete':r['complete'],'updates':r['updates'],'runtime':r['runtime'],'final_opened':r['final_opened'],'finished_at':j['finished_at'],'best_update':r['best']['update'],'files_verified_by_bytes_and_sha256':files,'source_unchanged_during_training':r['source_unchanged_during_training']}
assert len(out['formal_jobs'])==10
assert all(j['finished_at']<opening['created_at'] for j in out['formal_jobs'].values())
out['selection']={k:selection[k] for k in ['created_at','final_opened','selected','temperature','threshold','policy_feasible']}
out['selection']['final_opening_created_at']=opening['created_at']
out['selection']['same_selection_in_core_and_final']=True
temperature=read(CORE/'selection-evidence/temperature.json')
assert min(temperature['grid'],key=lambda x:x['family_equal_type_equal_nll'])['temperature']==selection['temperature']
policy=read(CORE/'selection-evidence/policy.json')
assert policy['threshold']==selection['threshold']
out['selection']['policy_threshold_from_preserved_grid']=policy['threshold']
out['selected_weight_bytes']={k:(CORE/'core'/selection['selected'][k]/'training/weights.safetensors').stat().st_size for k in ['av','choice','codec','talker']}
out['selected_weight_total_bytes']=sum(out['selected_weight_bytes'].values())
out['selected_weight_total_mib']=out['selected_weight_total_bytes']/1048576
assert out['selected_weight_total_bytes']==2192940
choice=read(FINAL/'measurements/choice/results.json')['primary']
p=choice['policy'];accepted=sum(x['accepted'] for x in p['rows']);errors=sum(x['accepted'] and x['error'] for x in p['rows'])
assert accepted==p['accepted'] and errors==p['errors']
out['final_choice']={'type_results':{k:v['primary'] for k,v in choice['types'].items()},'policy':{'families':len(p['rows']),'accepted':accepted,'errors':errors,'risk_recomputed':errors/accepted,'coverage_recomputed':accepted/len(p['rows'])}}
out['final_gates']=read(FINAL/'final-report.json')
out['frozen_C_metrics']=read(FINAL/'frozen/metrics.json')['C']
provider=read(CACHE/'core-jobs.json')
out['provider_core_jobs']=[{k:j[k] for k in ['name','conclusion','labels','started_at','completed_at']} for j in provider['jobs']]
release=read(CACHE/'release-metadata.json');descriptor=read('docs/omni-products/v1/public-release.json')
assets={r['name']:r for r in release['assets']}
out['public_release']={'tag':release['tag_name'],'published_at':release['published_at'],'components':{}}
for kind,row in descriptor['components'].items():
    a=assets[row['name']]
    assert a['id']==row['asset_id'] and a['size']==row['bytes'] and a['digest']=='sha256:'+row['sha256']
    out['public_release']['components'][kind]={'name':a['name'],'bytes':a['size'],'digest':a['digest'],'metadata_matches_descriptor':True}
tools=read('docs/course-experiments/results/tools.json')['results']
discrepancies=[]
for i,sample in enumerate(tools['samples']):
    for t,trace in enumerate(sample['trace']):
        g=trace['generation'];ids=[1];roles={'system':7,'user':3,'assistant':4}
        for message in g['messages']:ids += [roles[message['role']]]+[b+8 for b in message['content'].encode()]+[2]
        ids += [4]
        if ids!=g['input_ids']:
            discrepancies.append({'sample':i,'step':t,'logged_message_count':len(g['messages']),'actual_input_id_count':len(g['input_ids']),'reconstructed_message_id_count':len(ids),'actual_decoded_body':bytes(x-8 for x in g['input_ids'] if x>=8).decode('utf-8',errors='replace')})
out['tool_trace_message_discrepancies']={'count':len(discrepancies),'first':discrepancies[:2]}
out['v2_historical_result']={k:v for k,v in read('docs/selftrained/results/v2-final-public-results.json').items() if k!='architectures'}
out['legacy_product_training']={}
for kind in ['b','c']:
    receipt=read(CACHE/f'legacy-{kind}-training-receipt.json')
    out['legacy_product_training'][kind]={k:receipt[k] for k in ['product','mode','random_initialization','steps_this_attempt','last_step','selected_step','stop_reason','initial_weights_sha256','selected_weights_sha256','weights_changed','environment']}
out['primary_source_receipts']=read(CACHE/'receipts.json')
destination=Path(__file__).with_name('original-data-inspection.json')
destination.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(destination.relative_to(ROOT)),'bytes':destination.stat().st_size,'sha256':sha(destination),'history_reports':len(out['historical_results']),'formal_jobs':len(out['formal_jobs']),'selected_weight_bytes':out['selected_weight_total_bytes'],'tool_discrepancies':len(discrepancies)},ensure_ascii=False))
