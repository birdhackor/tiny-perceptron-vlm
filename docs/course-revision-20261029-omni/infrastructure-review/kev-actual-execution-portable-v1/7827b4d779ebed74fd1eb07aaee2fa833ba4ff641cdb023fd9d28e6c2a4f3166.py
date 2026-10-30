"""Independent actual Kev provenance, retention and raw-row arithmetic; no NN or dispatch."""
from collections import Counter
from copy import deepcopy
from datetime import UTC,datetime
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import socket
import subprocess
import sys
from unittest.mock import patch
import zipfile

import torch

REPO=Path('/workspace/work/omni-models-v1')
BASE=Path('/workspace/work/omni-kev-actual-independent-verification-v1')
REV='010a639f6db4889ecbac5801dcab7a7a4409e8f0'
ORIGINAL_REV='ee799d24fd1ea26108af709ec41e49db63d58e31'
ATTEMPT='mature-kev-41011677439-1'
OLD_ATTEMPT='mature-kev-41004743884-1'
AUDIT_NAME='docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v3.json'
AUDIT_SHA='53a5f325e6b4800f0f12be56df31551423a9bbbbf6a070753f4e62f1d5559a61'
PERMIT='929b40a464f2cedea3021e33db3fcdabc6570552aa361dfc969ce17fd0be2fd8'
INPUT_SHA='8652c3843caeabf93dbb3ce99a24dd06632b27ab1d7db0c45da0004b83540823'
RESULT_SHA='1261b695cd6f1e1b429880cb0554fe3ff7f6c8231d3e70e6429bc8505a7bb566'
ARTIFACT_SHA='9c77780f6c4d3a086ec3469cc9fb0cc288c5baf3e9b4fec2076d95eab990a70c'
CORE_SELECTION='935594413eda401f4fddc81b57f0245cd6a28219dda19e4e27b9d0f751385e20'
CORE_OPEN='ce962df3c95b0967d8397c6929433c9117d6fe60cede80266fce333964306843'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
when=lambda t:datetime.fromisoformat(t.replace('Z','+00:00'))

def require(condition,message):
    if not condition:raise AssertionError(message)
def close(a,b,message,tolerance=1e-9):
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)==set(b),message+' keys')
        for k in a:close(a[k],b[k],message+'/'+k,tolerance)
    elif isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),message+' count')
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,message+'/'+str(i),tolerance)
    elif type(a) in (int,float) and type(b) in (int,float):require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=tolerance,abs_tol=1e-10),message)
    else:require(a==b,message)

def git_bytes(name,revision=REV):return subprocess.check_output(['git','show',revision+':'+name],cwd=REPO)

# Import only the actually executed immutable project source, not future worktree edits.
audit_raw=git_bytes(AUDIT_NAME);require(hashlib.sha256(audit_raw).hexdigest()==AUDIT_SHA,'actual v3 audit bytes')
audit=json.loads(audit_raw);require(audit['status']=='pass_for_reviewed_scope' and len(audit['files'])==96,'independent exact96 source audit')
SOURCE=BASE/'source-010a639f'
names=set(audit['files'])|{AUDIT_NAME}
names.update('docs/omni-products/v1/'+n for n in ['data-definition.json','metrics.json','formal-recipes.json','codec-selection.json','model-selection.json','final-opening.json'])
for name in sorted(names):
    raw=git_bytes(name);target=SOURCE/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():require(target.read_bytes()==raw,'preserved immutable reviewer projection')
    else:target.write_bytes(raw)
    if name in audit['files']:require(hashlib.sha256(raw).hexdigest()==audit['files'][name],'committed audited96 source '+name)
sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import research_execution as execution,research_recovery as recovery,research_measurements as measurements,finance,core_final
require(Path(execution.__file__).is_relative_to(SOURCE),'actual revision import isolation')

# Every ZIP is bounded, path-checked and read in full before any local extraction.
def zip_payload(path,expected_count,expected_sha=None):
    if expected_sha:require(sha(path)==expected_sha,'outer ZIP SHA '+path.name)
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[x.filename for x in infos]
        require(len(infos)==expected_count and len(names)==len(set(names)) and len(names)<=1024 and sum(x.file_size for x in infos)<=256*1024**2,'bounded unique ZIP '+path.name)
        for x in infos:
            p=PurePosixPath(x.filename)
            require(not x.is_dir() and not p.is_absolute() and p.as_posix()==x.filename and '..' not in p.parts and '\\' not in x.filename and ':' not in x.filename and ((x.external_attr>>16)&0o170000)!=0o120000,'safe regular POSIX ZIP path')
        return {name:z.read(name) for name in names}

def unpack(raw,directory):
    directory.mkdir(exist_ok=True)
    for name,value in raw.items():
        p=directory/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():require(p.read_bytes()==value,'do not overwrite differing review evidence')
        else:p.write_bytes(value)

outer=zip_payload(BASE/'provider-artifact.zip',21,ARTIFACT_SHA);unpack(outer,BASE/'artifact')
require((BASE/'provider-artifact.zip').stat().st_size==780592,'artifact780592 bytes')
require(sha(BASE/'download/input.zip')==sha(BASE/'original-input/input.zip')==INPUT_SHA and (BASE/'download/input.zip').stat().st_size==434670,'true original identical input ZIP recovery')
require(outer['input.zip']==(BASE/'download/input.zip').read_bytes() and outer['result.zip']==(BASE/'download/result.zip').read_bytes(),'artifact and owned Release original payloads identical')
input_raw=zip_payload(BASE/'download/input.zip',71,INPUT_SHA);result_raw=zip_payload(BASE/'download/result.zip',16,RESULT_SHA)
unpack(result_raw,BASE/'verified')
job=json.loads(input_raw['job.json']);require(input_raw['job.json']==outer['job.json'] and hashlib.sha256(input_raw['job.json']).hexdigest()=='73f75133744798cdf1bd9c870c3120c2e6f77647a2321adbc16f87de012f9328','original exact job config bytes')
job={**job,'config_sha256':hashlib.sha256(input_raw['job.json']).hexdigest()}
require(len(job['input_files'])==70 and set(input_raw)==set(job['input_files'])|{'job.json'},'all70 bound input files/71 ZIP entries')
for name,binding in job['input_files'].items():require(len(input_raw[name])==binding['bytes'] and hashlib.sha256(input_raw[name]).hexdigest()==binding['sha256'],'all original input entry hashes')
workers={n for n in job['input_files'] if n.startswith(('scripts/','tiny_perceptron/'))}
require(len(workers)==62 and workers==set(execution.worker_files()),'exact62 unchanged worker file set')
for name in workers:require(input_raw[name]==git_bytes(name)==git_bytes(name,ORIGINAL_REV),'worker unchanged from original to recovery')
for name in ['data-definition.json','model-selection.json','final-opening.json']:
    require(input_raw[name]==git_bytes('docs/omni-products/v1/'+name)==git_bytes('docs/omni-products/v1/'+name,ORIGINAL_REV),'fixed scientific/core metadata')
require(job['core_selection_sha256']==CORE_SELECTION and job['core_final_opening_sha256']==CORE_OPEN and hashlib.sha256(input_raw['model-selection.json']).hexdigest()==CORE_SELECTION and hashlib.sha256(input_raw['final-opening.json']).hexdigest()==CORE_OPEN,'immutable own model/temperature/threshold opening binding')
require(job['gpu']=='L40S' and job['cpu']==4 and job['memory_mib']==49152 and job['worker_seconds']==1800 and job['sandbox_seconds']==2100 and job['own_training_permitted'] is False,'unchanged approved resources and no own training')
manifest=json.loads(result_raw['file-manifest.json']);require(set(manifest)==set(result_raw)-{'file-manifest.json'},'all15 result entries bound by manifest')
for name,binding in manifest.items():require(len(result_raw[name])==binding['bytes'] and hashlib.sha256(result_raw[name]).hexdigest()==binding['sha256'],'result per-file SHA/size '+name)
require(all(PurePosixPath(n).suffix in {'.json','.jsonl','.log'} for n in result_raw),'only own result/log files; no model weights or third-party full text exported')

run=load(BASE/'provider-run.json');jobs=load(BASE/'provider-jobs.json');release=load(BASE/'provider-release.json');original_release=load(BASE/'provider-original-release.json');artifact_meta=load(BASE/'provider-artifacts.json');state_envelope=load(BASE/'connector-ledger.json');state=json.loads(state_envelope['content'])
require(run['id']==41011677439 and run['run_attempt']==1 and run['event']=='workflow_dispatch' and run['head_branch']=='main' and run['head_sha']==REV and run['status']=='completed' and run['conclusion']=='success','native run exact revision/manual attempt')
require(jobs['total_count']==len(jobs['jobs'])==1,'one native job');native_job=jobs['jobs'][0]
require(native_job['id']==123918730063 and native_job['head_sha']==REV and native_job['conclusion']=='success' and all(x['status']=='completed' and x['conclusion']=='success' for x in native_job['steps']),'all native Kev steps success')
require(sha(BASE/'provider-native.log')=='5d9025d1412a3d64ab52605c8ff7d5cb601bf7e09297ae7867fad75e60614e45' and (BASE/'provider-native.log').stat().st_size==43357,'fresh native log exact bytes')
require(artifact_meta['total_count']==len(artifact_meta['artifacts'])==1,'single exact provider artifact');artifact=artifact_meta['artifacts'][0]
require(artifact['id']==12958341052 and artifact['size_in_bytes']==780592 and artifact['digest']=='sha256:'+ARTIFACT_SHA and not artifact['expired'] and artifact['workflow_run']['head_sha']==REV and artifact['name']=='omni-'+ATTEMPT,'actual provider artifact')
require(0<(when(artifact['expires_at'])-when(artifact['created_at'])).total_seconds()<=7*86400+60,'actual seven day artifact retention')
row=next(x for x in state['attempts'] if x['attempt_id']==ATTEMPT);old=next(x for x in state['attempts'] if x['attempt_id']==OLD_ATTEMPT);purpose=json.loads(row['purpose']);receipt=row['receipt']
require(state['reserved_total_usd']=='10.90' and row['reserved_usd']==old['reserved_usd']=='4.30' and old['status']=='failed-stopped' and row['status']=='completed','full original and recovery bounds retained, no refund')
require(row['revision']==REV and row['reservation_sha256']==finance.digest(finance.reservation_identity(row)),'durable exact reserved identity')
require(finance.digest(old)=='5639aefbe9acf390f652fe9e5edfcb175d0360fdef404910e921a9ac93b20bbc' and finance.digest(old['receipt'])=='163c961242a6780e665bcb01f0b40a8e18129c78eb9a90b4d99a109783eaaaa0','original failed row/receipt have not been rewritten')
require(not any(x['attempt_id']=='mature-kev-41009648132-1' for x in state['attempts']),'intermediate test failure has no ledger reservation')
permit,permit_sha=recovery.checked_permit();require(permit_sha==PERMIT and permit['maximum_new_attempts']==1 and permit['new_bound_usd']=='4.30','original signed one-use permit')
consumed=[x for x in state['attempts'] if json.loads(x['purpose']).get('preparation_recovery_permit_sha256')==PERMIT]
require(len(consumed)==1 and consumed[0]['attempt_id']==ATTEMPT and purpose['recovery_of']==OLD_ATTEMPT and purpose['original_input_payload_sha256']==INPUT_SHA and purpose['original_job_config_sha256']==job['config_sha256'],'exact permit consumed once')
require(purpose['input_sha256']==INPUT_SHA and purpose['job_config_sha256']==job['config_sha256'],'durable purpose binds original input and config')
sources=json.loads(outer['sources.json']);require(sources['revision']==REV and len(sources['files'])==96 and sources['files']==audit['files'] and sources['audit_sha256']==AUDIT_SHA and finance.digest(sources)==purpose['sources_sha256'],'exact96 controller source provenance')
execution.check_receipt_identity('kev',ATTEMPT,sources,json.loads(outer['reservation.json'])['attempt'],receipt)
require(receipt['success'] is True and receipt['compute_stopped'] is True and receipt['resources_verified'] is True and all(v is True for v in receipt['cleanup_checks'].values()),'successful completed resource receipt')
require(receipt['result_exported_before_stop'] is True and receipt['input_payload_uploaded'] is True and receipt['result_sha256']==RESULT_SHA and receipt['research_final_opened'] is True,'exact result exported before stop')
require(release['id']==receipt['release_id']==428701352 and release['draft'] and release['target_commitish']==REV and release['tag_name']=='omni-mature-candidate-'+ATTEMPT and release['body']=='Owned unpublished limited mature-model research; '+ATTEMPT+'; source '+REV,'exact owned recovery draft')
require({(a['id'],a['name'],a['size']) for a in release['assets']}=={(a['id'],a['name'],a['bytes']) for a in receipt['assets']} and {a['name'] for a in release['assets']}=={'input.zip','result.zip'},'exact two assets no third-party weights')
for a in release['assets']:require(a['digest']=='sha256:'+sha(BASE/'download'/a['name']),'both provider payload digests and fresh readback')
require(original_release['id']==428663042 and original_release['draft'] and original_release['target_commitish']==ORIGINAL_REV and original_release['tag_name']=='omni-mature-candidate-'+OLD_ATTEMPT and len(original_release['assets'])==1 and original_release['assets'][0]['id']==690960942 and original_release['assets'][0]['digest']=='sha256:'+INPUT_SHA,'original failed owned input draft preserved')
core=next(x for x in state['attempts'] if x['attempt_id']=='cpu-core-40988311100-1');core_release=load(BASE/'provider-core-release.json')
require(core_release['id']==428573191 and core_release['draft'] and len(core_release['assets'])==11 and core_release['target_commitish']==core['revision'] and core_release['body']=='Owned unpublished core weights and full optimizer/RNG; '+core['attempt_id']+'; source '+core['revision'],'original core draft body/target untouched')
require({(a['id'],a['name'],a['size']) for a in core_release['assets']}=={(a['id'],a['name'],a['bytes']) for a in core['receipt']['assets']},'all11 original core asset identities unchanged')
original_native=load(BASE/'provider-original-jobs.json')['jobs'][0];intermediate=load(BASE/'provider-intermediate-jobs.json')['jobs'][0]
require(original_native['conclusion']=='failure' and intermediate['conclusion']=='failure' and next(s for s in intermediate['steps'] if s['number']==7)['conclusion']=='skipped','native prior failure and pre-CAS intermediate skip')

apps=json.loads(outer['apps-after.json']);runtime=json.loads(outer['runtime-receipt.json']);owned=json.loads(outer['owned-resource.json']);app_by_id={x['app_id']:x for x in apps}
terminal=[]
for label,x in [('original',old),('recovery',row)]:
    r=x['receipt'];a=app_by_id[r['app_id']];ids=r['observed_owned_sandbox_ids']
    require(ids==[r['sandbox_id']] and r['stopped_sandboxes']=={r['sandbox_id']:3} and r['remaining_owned_sandboxes']==[] and r['stop_errors']=={} and a['state']=='stopped' and str(a['tasks'])=='0','all known historical IDs terminal')
    require(when(a['stopped_at'])<=when(receipt['finished_at']),'both apps stopped by verified completion time')
    terminal.append({'role':label,'app_id':r['app_id'],'sandbox_id':r['sandbox_id'],'SandboxWait_terminal_status':3,'app_state':'stopped','tasks':0,'provider_app_stopped_at':a['stopped_at']})
require(owned['app_id']==runtime['app_id']==receipt['app_id'] and owned['sandbox_id']==runtime['sandbox_id']==receipt['sandbox_id'],'exact same owned runtime resource IDs')
require(runtime['result_exported_before_stop'] and runtime['result_sha256']==RESULT_SHA and outer['stdout.txt'].decode().strip()=='OMNI_JOB_READY' and when(runtime['cleanup_started_at'])<when(runtime['finished_at'])<=when(runtime['cleanup_deadline_at']),'ready/export/source verified stop ordering and cleanup deadline')
modal_live=load(BASE/'provider-modal-terminal-readback.json');require(modal_live['error_type']=='AuthError' and modal_live['records']==[],'live Modal probe limitation recorded honestly')

NN_CALLS=[];NETWORK_CALLS=[]
def no_nn(*a,**kw):NN_CALLS.append(True);raise AssertionError('NN replay forbidden')
def no_network(*a,**kw):NETWORK_CALLS.append(True);raise AssertionError('network forbidden during pure metric replay')
def prob(logits,T):
    xs=[v/T for v in logits];m=max(xs);w=[math.exp(v-m) for v in xs];return [v/sum(w) for v in w]
def pred(x):return min(range(len(x['ids'])),key=lambda i:(-x['logits'][i],x['ids'][i]))
def cls(rows,T):
    n=len(rows);correct=0;losses=[];brier=[];bins={}
    for x in rows:
        p=prob(x['logits'],T);gold=x['ids'].index(x['gold']);hit=x['answer']==x['gold'];correct+=hit;zs=[v/T for v in x['logits']];m=max(zs)
        losses.append(m+math.log(sum(math.exp(z-m) for z in zs))-zs[gold]);brier.append(sum((v-int(i==gold))**2 for i,v in enumerate(p)));bins.setdefault(min(9,int(max(p)*10)),[]).append((max(p),hit))
    return {'observations':n,'accuracy':correct/n,'NLL':sum(losses)/n,'Brier':sum(brier)/n,'ECE10':sum(len(b)/n*abs(sum(x for x,_ in b)/len(b)-sum(y for _,y in b)/len(b)) for b in bins.values())}
def acceptance(rows,T,tau):
    output=[]
    for x in rows:
        c=x['tasks']['choice'];p=prob(c['logits'],T);answer=c['ids'][pred(c)];info=x['choice_info'];ok=tau is not None and max(p)>=tau and answer!='none' and info['ids'][pred(info)]=='yes';error=ok and (answer!=c['gold'] or not x['world_supported'] or not x['choice_information_sufficient'])
        output.append({'family':x['family'],'accepted':ok,'error':bool(error),'answer':answer,'p_max':max(p)})
    n=sum(x['accepted'] for x in output);e=sum(x['error'] for x in output)
    return {'rows':output,'families':len(rows),'accepted':n,'errors':e,'coverage':n/len(rows),'empirical_risk':e/n if n else None}

with patch.object(torch.nn.Module,'_call_impl',no_nn),patch.object(torch.Tensor,'backward',no_nn),patch.object(socket.socket,'connect',no_network),patch.object(socket,'getaddrinfo',no_network),patch.object(finance.GitHubLedger,'write',side_effect=AssertionError('ledger writes forbidden')):
    finance.validate_state(state,datetime.now(UTC))
    strict=execution.verify_results(BASE/'download/result.zip',job)
    data=json.loads(input_raw['data-definition.json']);metrics=load(SOURCE/'docs/omni-products/v1/metrics.json');selection=json.loads(result_raw['kev-reliability-selection.json']);research=json.loads(result_raw['research-result.json'])
    calibration=json.loads(result_raw['kev-calibration.json']);policy=json.loads(result_raw['kev-policy-predictions.json'])
    require(len(calibration)==80 and len(policy)==60,'fixed calibration80/policy60')
    temperature_grid=[]
    for T in metrics['C']['temperature_grid']:
        loss=sum(sum(cls([x['tasks'][k]],T)['NLL'] for k in ('choice','info','rubric'))/3 for x in calibration)/80
        temperature_grid.append({'temperature':T,'family_equal_type_equal_nll':loss})
    close(temperature_grid,selection['temperature']['grid'],'independent33-point equal-family/equal-type calibration NLL')
    best=min(temperature_grid,key=lambda x:(x['family_equal_type_equal_nll'],abs(x['temperature']-1),x['temperature']))['temperature'];require(best==selection['temperature']['temperature']==1.8340080864093424,'original Kev T reproduced without new model outputs')
    policy_grid=[{'threshold':t,**acceptance(policy,best,t)} for t in metrics['C']['threshold_grid']]
    feasible=[x for x in policy_grid if x['accepted']>0 and x['empirical_risk']<=0.05];chosen=max(feasible,key=lambda x:(x['coverage'],x['threshold']))
    close(policy_grid,selection['policy']['grid'],'original policy grid');require(chosen['threshold']==selection['policy']['threshold']==0.8 and selection['policy']['feasible'],'original maximum-coverage feasible Kev tau')
    require(selection['head_temperature_during_raw_logits']==1.0 and selection['upstream_temperature']==2.406050072164233 and selection['final_opened'] is False,'raw/upstream/new T distinctions')
    raw=[json.loads(line) for line in result_raw['kev-raw-predictions.jsonl'].decode().splitlines()]
    require(len(raw)==260 and Counter((x['split'],x['language']) for x in raw)==Counter({('calibration','english'):80,('policy','english'):60,('final','english'):60,('final','chinese'):60}),'all260 raw family-view records')
    views={language:json.loads(result_raw['kev-final-'+language+'.json']) for language in ('english','chinese')}
    final_summaries={};all_expected=[('calibration','english',x) for x in calibration]+[('policy','english',x) for x in policy]
    for language,v in views.items():
        rows=v['raw']['rows'];all_expected.extend(('final',language,x) for x in rows)
        require({x['family'] for x in rows}=={x['family'] for x in data['C']['panels']['final']} and len(rows)==60,'same frozen60 final families per language')
        for label,T,tau in [('raw',1.,None),('upstream_temperature',selection['upstream_temperature'],None),('new_temperature_policy',best,chosen['threshold'])]:
            block=v[label];require(block['rows']==rows,'all three final views reuse same actual predictions')
            for kind in ('choice','info','rubric'):
                values=[x['tasks'][kind] for x in rows];close(cls(values,1.),block['types'][kind]['raw'],'independent raw metrics');close(cls(values,T),block['types'][kind]['calibrated'],'independent calibrated metrics')
                require(block['types'][kind]['primary']['correct']==sum(x['answer']==x['gold'] for x in values) and block['types'][kind]['primary']['families']==60,'type fixed60 counts')
            expected_policy=acceptance(rows,T,tau);close(expected_policy,{k:v for k,v in block['policy'].items() if k!='error_wilson95_among_accepts'},'raw policy decision/error replay')
        final_summaries[language]={'families':60,'counts':{k:v['raw']['types'][k]['primary']['correct'] for k in ('choice','info','rubric')},'new_temperature_policy':{k:x for k,x in v['new_temperature_policy']['policy'].items() if k!='rows'},'metrics':{label:{k:x['calibrated'] for k,x in block['types'].items()} for label,block in v.items()}}
    require([(x['split'],x['language'],x['prediction']) for x in raw]==all_expected,'raw JSONL exact stage order and same reported predictions')
    require({x['family'] for x in views['english']['raw']['rows']}=={x['family'] for x in views['chinese']['raw']['rows']},'two state language views do not increase independent n')
    cache=research['cache'];require(len(cache['rows'])==8 and all(re.fullmatch('[0-9a-f]{64}',r['prefix_before']) and r['prefix_before']==r['prefix_after'] for r in cache['rows']),'all8 recorded prefix fingerprints stable')
    require(all(0<=r['uncached_cached_max_abs']<=1e-5 and r['Q1_Q2_Q1_probability_max_abs']==0 for r in cache['rows']) and cache['all_observed_within_tolerance'],'all recorded cache deltas within fixed tolerance')
    # Strong semantic negatives: rebuild the inner manifest so corruption is not merely a hash error.
    negative=[]
    for mode in ('missing_calibration','missing_final_chinese','missing_raw_record','cache_summary_lie'):
        changed=dict(result_raw)
        if mode=='missing_calibration':changed['kev-calibration.json']=json.dumps(calibration[:-1]).encode()
        elif mode=='missing_final_chinese':
            x=deepcopy(views['chinese']);x['raw']['rows'].pop();changed['kev-final-chinese.json']=json.dumps(x).encode()
        elif mode=='missing_raw_record':changed['kev-raw-predictions.jsonl']=('\n'.join(json.dumps(x) for x in raw[:-1])+'\n').encode()
        else:
            x=deepcopy(research);x['cache']['rows'][0]['prefix_after']='0'*64;changed['research-result.json']=json.dumps(x).encode();s=json.loads(changed['worker-summary.json']);s['research_result_sha256']=hashlib.sha256(changed['research-result.json']).hexdigest();changed['worker-summary.json']=json.dumps(s).encode()
        changed['file-manifest.json']=json.dumps({n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in changed.items() if n!='file-manifest.json'}).encode()
        p=BASE/('negative-'+mode+'.zip')
        with zipfile.ZipFile(p,'w') as z:
            for n,v in changed.items():z.writestr(n,v)
        try:execution.verify_results(p,job)
        except ValueError:negative.append({'case':mode,'rejected':True,'manifest_valid':True})
        else:raise AssertionError('semantic incomplete output accepted')
    try:recovery.eligibility(state,'kev')
    except ValueError as e:require('第二次修復' in str(e) or '已使用' in str(e),'one-time recovery consumed state')
    else:raise AssertionError('used permit eligible for another attempt')
require(not NN_CALLS and not NETWORK_CALLS,'no NN and no metric-replay network')
for name,binding in audit['files'].items():require(sha(SOURCE/name)==binding,'immutable executed-source snapshot unchanged')
result={'schema':'omni-kev-actual-independent-verification-evidence-v1','reviewer_task':'/root/omni_cpu_final_execution_review','status':'pass_for_actual_execution_and_preserved_research_result_integrity','actual_commit':REV,'source_count':96,'source_audit_sha256':AUDIT_SHA,'worker_files':62,'input_bound_files':70,'input_zip_entries':71,'input_sha256':INPUT_SHA,'result_entries':16,'result_manifest_entries':15,'result_sha256':RESULT_SHA,'artifact_entries':21,'artifact_sha256':ARTIFACT_SHA,'permit_sha256':PERMIT,'permit_consumptions':1,'original_failure_bound_retained_usd':'4.30','recovery_bound_retained_usd':'4.30','total_bounds_at_snapshot_usd':'10.90','strict_frozen_result_verifier':strict,'raw_family_view_records':260,'raw_prediction_vectors':1040,'calibration_families':80,'policy_families':60,'final_same_families_per_language':60,'selection':{'temperature':best,'threshold':chosen['threshold'],'upstream_temperature':selection['upstream_temperature'],'raw_head_temperature':1.0,'policy_calibration_accepted':chosen['accepted'],'policy_calibration_errors':chosen['errors']},'final':final_summaries,'cache':{'dev_families':8,'all_fingerprints_equal':True,'atol':1e-5,'max_uncached_cached_abs':max(x['uncached_cached_max_abs'] for x in cache['rows']),'max_Q1_Q2_Q1_probability_abs':0.0,'observation_scope':'Only recorded fingerprints and numeric controls rechecked; underlying prefix tensors were not exported or replayed.'},'historical_terminal_resources':terminal,'live_Modal_probe':{'status':'AuthError','observed_at':modal_live['observed_at'],'new_RPC_success_claimed':False,'no_user_token_requested':True},'semantic_negative_controls':negative,'source_worker_core_science_modified':False,'NN_forward_calls':0,'paid_create_calls':0,'dispatch_calls':0,'remote_writes':0,'limitations':['No fresh live Modal terminal RPC succeeded: the intentionally unbound local SDK profile returned AuthError. Terminal assessment uses the independently downloaded provider artifact runtime-receipt/apps-after at the actual recorded stop timestamps and exact source-verified lifecycle.','No model NN replay or weight loading was performed. Base/adapter revisions and head/config SHA enforcement are established by exact executed worker source and retained result/runtime identities; multi-GB model weight files were not re-downloaded or independently re-hashed.','Raw prefix/cache tensors are absent; stable recorded fingerprints and all eight recorded deltas were checked, not re-computed by a model.','English/Chinese differ only in visible-state rendering while questions stay English. Both views share the same60 held-out project-authored families; this is not120 independent cases, independent corpus, or proof of pretraining-unseen generalization.','This is not a fair own-C architecture comparison, broad capability proof, promotion gate, or any curriculum review round.']}
p=BASE/'verification-results-v1.json';require(not p.exists(),'new immutable verification result');p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'source_count':96,'worker_files':62,'raw_records':260,'final_english_counts':final_summaries['english']['counts'],'final_chinese_counts':final_summaries['chinese']['counts'],'historical_known_sandboxes_terminal':2,'live_Modal_probe':'AuthError recorded','NN_forward_calls':0},ensure_ascii=False))
