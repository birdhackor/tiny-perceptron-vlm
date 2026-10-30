"""Independent provider/archive/raw-result verification. Never runs any NN forward or training."""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import UTC,datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import socket
import subprocess
from unittest.mock import patch
import wave
import zipfile

import numpy as np
import torch
import sys

ROOT=Path('/workspace/work/omni-models-v1')
BASE=Path('/workspace/work/omni-final-actual-independent-verification-v1')
REV='ee799d24fd1ea26108af709ec41e49db63d58e31'
ATTEMPT='cpu-final-41004229262-1'
# Import the actual executed Git objects, independent of later research worktree edits.
SOURCE=BASE/'source-ee799d'
source_audit_path='docs/course-revision-20261029-omni/infrastructure/cpu-final-execution-audit-v2.json'
source_audit_raw=subprocess.check_output(['git','show',REV+':'+source_audit_path],cwd=ROOT)
source_audit=json.loads(source_audit_raw)
project_names=set(source_audit['files']) | {source_audit_path,'docs/course-revision-20261029-omni/infrastructure/final-evaluation-audit-v2.json'}
project_names.update('docs/omni-products/v1/'+name for name in ('data-definition.json','metrics.json','formal-recipes.json','codec-selection.json','model-selection.json','final-opening.json','public-release.json','mature-research-protocol.md'))
for name in sorted(project_names):
    raw_source=subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT)
    target=SOURCE/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=raw_source:raise AssertionError('immutable review Git snapshot differs')
    else:target.write_bytes(raw_source)
sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import cpu_final_execution as controller
from scripts.omni_revision import core_final,core_report,finance,core_evaluation
from tiny_perceptron.omni_products.data import Event
from tiny_perceptron.omni_products.inference import SOUND_MAP
from tiny_perceptron.omni_products.signal_metrics import program_success
from tiny_perceptron.selftrained.tools import parse_tool_call,ToolCallError
OPEN='ce962df3c95b0967d8397c6929433c9117d6fe60cede80266fce333964306843'
ZIP_SHA='50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
when=lambda value:datetime.fromisoformat(value.replace('Z','+00:00'))
RESULT={};NN_CALLS=[];NETWORK_CALLS=[]

def require(value,scope):
    if not value:raise AssertionError(scope)
def close(a,b,scope,tol=1e-11):
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)==set(b),scope+' keys')
        for k in a:close(a[k],b[k],scope+'/'+k,tol)
    elif isinstance(a,(list,tuple)):
        require(len(a)==len(b),scope+' len')
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,scope+'/'+str(i),tol)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):require(math.isclose(a,b,rel_tol=tol,abs_tol=tol),scope)
    else:require(a==b,scope)
def wilson(k,n):
    if n==0:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;s=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-s),min(1,c+s)]
def proportion(k,n):return {'correct':k,'families':n,'accuracy':k/n,'wilson95_family':wilson(k,n)}
def probabilities(logits,T):
    values=[x/T for x in logits];m=max(values);weights=[math.exp(x-m) for x in values];den=sum(weights)
    return [x/den for x in weights]
def answer(row):return min(range(len(row['ids'])),key=lambda i:(-row['logits'][i],row['ids'][i]))
def checked_prediction(row,T):
    p=probabilities(row['logits'],T);require(row['answer']==row['ids'][answer(row)],'stable raw-logit argmax')
    if 'probabilities' in row:close(p,row['probabilities'],'stored probabilities')
    if 'p_max' in row:close(max(p),row['p_max'],'stored maximum probability')
    return p

def archive_verify(path,destination):
    require(path.stat().st_size==1827422 and sha(path)==ZIP_SHA,'outer actual Release ZIP')
    with zipfile.ZipFile(path) as z:
        entries=z.infolist();names=[x.filename for x in entries]
        require(len(entries)==624 and len(names)==len(set(names)) and len(names)<=1024,'624 unique final entries')
        require(sum(x.file_size for x in entries)==13038648 and sum(x.file_size for x in entries)<=256*1024**2,'expanded bounds')
        for item in entries:
            p=PurePosixPath(item.filename)
            require(not item.is_dir() and not p.is_absolute() and p.as_posix()==item.filename and '..' not in p.parts and '\\' not in item.filename and ':' not in item.filename and ((item.external_attr>>16)&0o170000)!=0o120000,'safe final POSIX regular path')
            allowed=item.filename in {'package-file-manifest.json','core-inputs.json','execution-sources.json','final-report.json','frozen/mature-research-protocol.md'}
            allowed|=bool(re.fullmatch(r'frozen/(?:data-definition|metrics|formal-recipes|codec-selection|model-selection|final-opening|public-release)\.json',item.filename))
            allowed|=bool(re.fullmatch(r'measurements/(?:av|sound|choice|integration)/(?:part-origin|part-completion|results)\.json',item.filename))
            allowed|=bool(re.fullmatch(r'measurements/sound/(?:sound/sound|chain/chain|chain/gold-semantic)-[0-9]{3}\.wav',item.filename))
            require(allowed,'final archive allowlist')
        manifest=json.loads(z.read('package-file-manifest.json'))
        require(set(manifest)==set(names)-{'package-file-manifest.json'},'complete final inner manifest')
        all_bytes={name:z.read(name) for name in names}
        for name,binding in manifest.items():require(len(all_bytes[name])==binding['bytes'] and hashlib.sha256(all_bytes[name]).hexdigest()==binding['sha256'],'inner bytes/SHA '+name)
        destination.mkdir(exist_ok=True)
        for name,raw in all_bytes.items():
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():require(target.read_bytes()==raw,'do not overwrite different review unpack')
            else:target.write_bytes(raw)
    return manifest,all_bytes

# Provider snapshots were freshly fetched by this reviewer via connected GitHub and native read-only gh API.
run=load(BASE/'provider-run.json');jobs=load(BASE/'provider-jobs.json');release=load(BASE/'provider-release.json');art=load(BASE/'provider-artifacts-native.json')
require(run['id']==41004229262 and run['head_sha']==REV and run['head_branch']=='main' and run['event']=='workflow_dispatch' and run['run_attempt']==1 and run['status']=='completed' and run['conclusion']=='success','native run identity')
require(jobs['total_count']==len(jobs['jobs'])==1,'one native final job');job=jobs['jobs'][0]
require(job['id']==123895352619 and job['head_sha']==REV and job['status']=='completed' and job['conclusion']=='success','native final job')
steps={row['number']:row for row in job['steps']}
require(all(steps[i]['status']=='completed' and steps[i]['conclusion']=='success' for i in range(1,11)),'every native preparation/measurement/report/export/bound/finish step')
require(sha(BASE/'provider-native.log')=='e1881982490c952e18cafc7b68b13a9783e85cef9f51d478a41a2fab5f3b4de3' and (BASE/'provider-native.log').stat().st_size==47549,'independently fetched native log exact bytes')
prior=load(BASE/'provider-prior-failed-jobs.json')['jobs'][0]
require(prior['conclusion']=='failure' and next(x for x in prior['steps'] if x['number']==6)['conclusion']=='failure' and next(x for x in prior['steps'] if x['number']==7)['conclusion']=='skipped','earlier setup failed before NN step')
require(sha(ROOT/'outputs/omni-final-first-preparation-failure-41001706982.log')=='0971bee3b192ef67cf131df4c6c1852866b69be292e1029310b2e67fd4b1170d','prior failure log preserved')
pr=load(BASE/'provider-pr10.json');checks=load(BASE/'provider-pr10-checks.json')
require(pr['merged'] and pr['merge_commit_sha']==REV and pr['head']['sha']=='13455d8341ab95055a6d5668a4d70b0b12df9774','merged exact PR10')
require(checks['total_count']==len(checks['check_runs'])==12 and all(x['head_sha']==pr['head']['sha'] and x['conclusion']=='success' and x['status']=='completed' and when(x['completed_at'])<=when(pr['merged_at']) for x in checks['check_runs']),'all12 checks before merge')
state_envelope=load(BASE/'connector-actual_ledger.json');state=json.loads(state_envelope['content'])
rows=[x for x in state['attempts'] if json.loads(x['purpose']).get('job')=='cpu-core-final-v1']
require(len(rows)==1,'exactly one durable final attempt');row=rows[0];receipt=row['receipt'];purpose=json.loads(row['purpose'])
require(row['attempt_id']==ATTEMPT and row['revision']==REV and row['status']=='completed' and row['reserved_usd']=='0.10' and receipt['attempt']==ATTEMPT and receipt['source_revision']==REV and receipt['success'] is True,'exact completed final identity')
require(row['reservation_sha256']==finance.digest(finance.reservation_identity(row)),'unchanged reservation identity')
require(row['components_usd']==controller.transport.COMPONENTS and state['reserved_total_usd']=='2.30','no refund and bounded storage components')
require(receipt['no_training_performed'] and receipt['no_modal_compute_created'] and receipt['all_training_processes_exited'] and receipt['resources_verified'] and receipt['compute_stopped'] and all(receipt['cleanup_checks'].values()),'true terminal cleanup receipt')
require(receipt['completed_parts']==['av','sound','choice','integration'],'all four fixed parts once')
require(release['id']==receipt['release_id']==428660172 and release['draft'] and release['tag_name']=='omni-final-candidate-'+ATTEMPT and release['target_commitish']==REV and release['body']=='Owned unpublished fixed core final measurements; '+ATTEMPT+'; source '+REV,'exact owned final draft')
require(len(release['assets'])==len(receipt['assets'])==1,'one exact final asset');asset=release['assets'][0]
require(asset['id']==690952957 and asset['name']=='final-results.zip' and asset['size']==1827422 and asset['digest']=='sha256:'+ZIP_SHA and receipt['assets'][0]['sha256']==ZIP_SHA and receipt['assets'][0]['roundtrip_verified'],'provider asset digest/ledger actual roundtrip')
require(art['total_count']==len(art['artifacts'])==1,'single actual provider artifact');artifact=art['artifacts'][0]
require(artifact['id']==12954354171 and artifact['name']=='omni-final-41004229262-1' and not artifact['expired'] and 0<artifact['size_in_bytes']<=256*1024**2 and artifact['workflow_run']['id']==run['id'] and artifact['workflow_run']['head_sha']==REV,'actual artifact identity and size')
require(0<(when(artifact['expires_at'])-when(artifact['created_at'])).total_seconds()<=7*86400+60,'actual artifact7day retention')
require(sha(BASE/'provider-artifact.zip')==artifact['digest'].removeprefix('sha256:') and (BASE/'provider-artifact.zip').stat().st_size==artifact['size_in_bytes'],'artifact download exact digest')
core_row=next(x for x in state['attempts'] if x['attempt_id']=='cpu-core-40988311100-1');core_release=load(BASE/'provider-core-release.json')
require(core_release['id']==core_row['receipt']['release_id'] and core_release['draft'] and core_release['target_commitish']==core_row['revision'] and core_release['tag_name']==core_row['receipt']['release_tag'] and len(core_release['assets'])==11,'core draft unchanged and11 assets retained')
require({(x['id'],x['name'],x['size']) for x in core_release['assets']}=={(x['id'],x['name'],x['bytes']) for x in core_row['receipt']['assets']},'all11 core provider assets remain exact')
RESULT['provider']={'run_id':run['id'],'job_id':job['id'],'source_revision':REV,'status':'completed','conclusion':'success','run_attempt':1,'native_steps':[{k:x[k] for k in ('number','name','status','conclusion','started_at','completed_at')} for x in job['steps'] if x['number']<=10],'ledger_blob':state_envelope['sha'],'ledger_total_at_reviewer_fetch':'2.30','final_bound':'0.10','reserved_at':row['reserved_at'],'release_id':release['id'],'asset_id':asset['id'],'artifact_id':artifact['id'],'artifact_size':artifact['size_in_bytes'],'artifact_retention_seconds':(when(artifact['expires_at'])-when(artifact['created_at'])).total_seconds(),'core_assets_retained':11,'prior_setup_NN_step':'skipped','PR10_successful_checks':12}

manifest,raw=archive_verify(BASE/'download/final-results.zip',BASE/'verified')
verified=BASE/'verified';data=load(verified/'frozen/data-definition.json');metrics=load(verified/'frozen/metrics.json');selection=load(verified/'frozen/model-selection.json');opening=load(verified/'frozen/final-opening.json');sources=load(verified/'execution-sources.json')
require(sha(verified/'frozen/final-opening.json')==OPEN and opening['runtime']=={'device':'cpu','torch':'2.14.1+cpu','python':'3.13.5'},'fixed original opening/runtime')
for path in (verified/'frozen').iterdir():
    require(path.read_bytes()==(controller.FROZEN/path.name).read_bytes()==subprocess.check_output(['git','show',REV+':docs/omni-products/v1/'+path.name],cwd=ROOT),'unchanged committed frozen '+path.name)
source_before={name:sha(SOURCE/name) for name in controller.source_paths()};require(len(source_before)==99 and sources['revision']==REV and sources['files']==source_before,'actual99 source closure')
audit=load(controller.AUDIT);require(audit['status']=='pass_for_reviewed_scope' and audit['files']==source_before and sha(controller.AUDIT)==sources['audit_sha256'],'exact reviewed99 audit')
for name,binding in source_before.items():require(hashlib.sha256(subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT)).hexdigest()==binding,'actual committed99 bytes')
require(opening['evaluation_sources_sha256']==core_final.evaluation_sources() and len(opening['evaluation_sources_sha256'])==57,'unchanged57 evaluation sources')
eval_audit_path=SOURCE/'docs/course-revision-20261029-omni/infrastructure/final-evaluation-audit-v2.json';eval_audit=load(eval_audit_path)
require(sha(eval_audit_path)==opening['independent_audit_sha256'] and eval_audit['status']=='pass_for_reviewed_scope' and all(eval_audit['reviewed_files_sha256'][k]==v for k,v in opening['evaluation_sources_sha256'].items()),'original independent57 audit binding')
require(purpose['sources_sha256']==finance.digest(sources) and purpose['model_selection_sha256']==sha(verified/'frozen/model-selection.json') and purpose['final_opening_sha256']==OPEN,'durable purpose binds sources/selection/opening')
inputs=load(verified/'core-inputs.json')
require(inputs['core_final_opening_sha256']==OPEN and inputs['core_selection_sha256']==purpose['model_selection_sha256'] and all(v==sha(verified/'frozen'/k) for k,v in inputs['files'].items()),'all original six metadata bindings')
require(inputs['core_asset']['sha256']=='107c418e69f917294c7e2fd1a17b820285e8712d5ea4e1f1d62332285de17aee','original trained core ZIP')
with zipfile.ZipFile(BASE/'provider-artifact.zip') as z:
    entries=z.infolist();require(len(entries)==637 and sum(x.file_size for x in entries)==receipt['artifact']['total_bytes']<=256*1024**2,'actual artifact entire byte bound')
    require(all(not x.is_dir() and PurePosixPath(x.filename).as_posix()==x.filename and not PurePosixPath(x.filename).is_absolute() and '..' not in PurePosixPath(x.filename).parts and '\\' not in x.filename for x in entries),'artifact safe paths')
    for name in manifest:
        if name.startswith('measurements/') or name in ('final-report.json','core-inputs.json'):require(z.read(name)==raw[name],'artifact and Release same original results')
    require(z.read('sources.json')==raw['execution-sources.json'] and hashlib.sha256(z.read('final-results.zip')).hexdigest()==ZIP_SHA,'same source record and release ZIP inside artifact')
    require(json.loads(z.read('reservation.json'))['attempt']['reservation_sha256']==row['reservation_sha256'],'artifact holds exact original reservation')
    require(json.loads(z.read('cpu-final-receipt.json'))['package']==receipt['package'],'artifact run receipt matches durable finish package')
    snap=json.loads(z.read('billing-before.json'));inv=json.loads(z.read('inventory.json'))
    require(0<=(when(row['reserved_at'])-when(snap['observed_at'])).total_seconds()<=120 and 0<=(when(row['reserved_at'])-when(inv['observed_at'])).total_seconds()<=120,'real fresh billing/inventory before CAS')
    require(finance.digest(json.loads(z.read('cost-proof.json')))==purpose['cost_proof_sha256'],'real cost quote binding')
    require(json.loads(z.read('legacy-preflight.json'))['data_files']==8962 and json.loads(z.read('legacy-preflight.json'))['data_bytes']==127161811,'real pre-CAS legacy loader receipt')
    require(json.loads(z.read('legacy-download.json'))['anonymous_A_download']['authentication']=='disabled','real anonymous A receipt')
    starts=json.loads(z.read('final-run-started.json'));require(starts['attempt']==ATTEMPT and starts['source_revision']==REV and when(starts['started_at'])>=when(row['reserved_at']),'actual CAS before any final start')
    control_bytes=sum(x.file_size for x in entries if not x.filename.endswith('.zip'));require(control_bytes==receipt['artifact']['control_bytes']<=16*1024**2,'real16MiB control bound')
values={part:load(verified/'measurements'/part/'results.json') for part in ('av','sound','choice','integration')}
parts=[];previous=when(receipt['started_at'])
for part in values:
    directory=verified/'measurements'/part;origin=load(directory/'part-origin.json');end=load(directory/'part-completion.json')
    core_report.validate_part(part,directory,opening,OPEN)
    require(previous<=when(origin['created_at'])<=when(end['created_at'])<=when(receipt['finished_at']),'four part order and receipt chronology')
    previous=when(end['created_at']);parts.append({'part':part,'started_at':origin['created_at'],'completed_at':end['created_at'],'results_sha256':end['results_sha256'],'origin_sha256':end['part_origin_sha256']})
RESULT['archive']={'zip_bytes':1827422,'zip_sha256':ZIP_SHA,'entries':624,'expanded_bytes':13038648,'inner_manifest_entries':len(manifest),'inner_manifest_sha256':sha(verified/'package-file-manifest.json'),'frozen_files_compared_to_committed_bytes':8,'bound_core_metadata_files':6,'execution_sources':99,'evaluation_sources':57,'opening_sha256':OPEN,'parts':parts}

# Explicitly forbid NN, remote sockets and ledger writes during pure result arithmetic.
def no_nn(*args,**kwargs):NN_CALLS.append(True);raise AssertionError('NN forward forbidden')
def no_net(*args,**kwargs):NETWORK_CALLS.append(True);raise AssertionError('network forbidden during raw validation')
with patch.object(torch.nn.Module,'_call_impl',no_nn),patch.object(torch.Tensor,'backward',no_nn),patch.object(socket.socket,'connect',no_net),patch.object(socket,'getaddrinfo',no_net),patch.object(finance.GitHubLedger,'write',side_effect=AssertionError('ledger mutation forbidden')),patch.object(core_final,'evaluate',side_effect=AssertionError('NN final forbidden')):
    torch.set_num_threads(2)
    finance.validate_state(state,datetime.now(UTC))
    models=values['av']['models'];expected={v['family']:v for v in data['AV']['final']};av_summary={}
    for name,value in models.items():
        rows=value['records'];require(len(rows)==len({x['family'] for x in rows})==200 and {x['family'] for x in rows}==set(expected),'six same200 families')
        for x in rows:
            fixed=expected[x['family']];require(x['gold']==fixed['labels'] and x['panel']==fixed['panel'] and x['paired_correct']==(x['answers']==x['gold']) and x['waveform_swap_gold']==[0,1],'AV raw labels/panel/pair/swap')
        close(value['primary'],proportion(sum(x['answers']==x['gold'] for x in rows),200),'AV primary')
        for panel in range(4):
            subset=[x for x in rows if x['panel']==panel];require(len(subset)==50,'AV50 perpanel');close(value['panels'][str(panel)],proportion(sum(x['paired_correct'] for x in subset),50),'AV panel')
        for kind in ('image','audio'):close(value['remove_'+kind],proportion(sum(x['remove_'+kind+'_answer']==2 for x in rows),200),'missing cue full200')
        close(value['same_metadata_waveform_swap']['paired'],proportion(sum(x['waveform_swap_answers']==x['waveform_swap_gold'] for x in rows),200),'swap full200')
        av_summary[name]={'correct':value['primary']['correct'],'families':200,'panels_correct':{p:v['correct'] for p,v in value['panels'].items()},'remove_image_correct':value['remove_image']['correct'],'remove_audio_correct':value['remove_audio']['correct']}
    compare=values['av']['paired_family_comparison'];indices=torch.randint(200,(1000,200),generator=torch.Generator().manual_seed(20261029));vectors=[]
    def interval(vector):return {'mean_time_rope_minus_additive':float(vector.mean()),'paired_family_bootstrap95':torch.quantile(vector[indices].mean(1),torch.tensor([.025,.975],dtype=torch.float64)).tolist()}
    for seed in (11,29,47):
        left={x['family']:x['paired_correct'] for x in models[f'av-additive-s{seed}']['records']};right={x['family']:x['paired_correct'] for x in models[f'av-time-rope-s{seed}']['records']}
        vector=torch.tensor([int(right[f])-int(left[f]) for f in sorted(expected)],dtype=torch.float64);vectors.append(vector);close(interval(vector),compare['per_training_seed'][str(seed)],'paired common-family bootstrap per seed')
    close(interval(torch.stack(vectors).mean(0)),compare['three_seed_mean'],'paired bootstrap seed mean');require(compare['families']==200 and compare['replicates']==1000 and compare['resampling_seed']==20261029,'frozen bootstrap specification')
    RESULT['AV']={'models':av_summary,'selected':selection['selected']['av'],'bootstrap':compare,'independent_family_unit':200}

    T=selection['temperature'];tau=selection['threshold'];c_summary={}
    def classification(rows,temperature):
        losses=[];briers=[];hits=[];bins=defaultdict(list)
        for x in rows:
            p=probabilities(x['logits'],temperature);gold=x['ids'].index(x['gold']);hit=x['answer']==x['gold'];scaled=[v/temperature for v in x['logits']];m=max(scaled)
            losses.append(m+math.log(sum(math.exp(v-m) for v in scaled))-scaled[gold]);briers.append(sum((v-int(i==gold))**2 for i,v in enumerate(p)));hits.append(hit);bins[min(9,int(max(p)*10))].append((max(p),hit))
        n=len(rows);return {'observations':n,'accuracy':sum(hits)/n,'NLL':sum(losses)/n,'Brier':sum(briers)/n,'ECE10':sum(len(v)/n*abs(sum(x for x,_ in v)/len(v)-sum(y for _,y in v)/len(v)) for v in bins.values())}
    for label,temp,threshold in [('primary',T,tau),('data_control_raw_only',1.,None)]:
        block=values['choice'][label];rows=block['rows'];require(len(rows)==len({x['family'] for x in rows})==60,'C complete60 family rows')
        for x in rows:
            for task in x['tasks'].values():checked_prediction(task,temp)
            checked_prediction(x['choice_info'],temp)
        for kind in ('choice','info','rubric'):
            selected=[x['tasks'][kind] for x in rows];close(block['types'][kind]['primary'],proportion(sum(x['answer']==x['gold'] for x in selected),60),'C fixed raw60 score')
            close(classification(selected,1),block['types'][kind]['raw'],'C independent raw metrics');close(classification(selected,temp),block['types'][kind]['calibrated'],'C independent calibrated metrics')
        policy_rows=[]
        for x in rows:
            c=x['tasks']['choice'];p=probabilities(c['logits'],temp);accepted=threshold is not None and max(p)>=threshold and c['answer']!='none' and x['choice_info']['answer']=='yes';error=accepted and (c['answer']!=c['gold'] or not x['world_supported'] or not x['choice_information_sufficient'])
            policy_rows.append({'family':x['family'],'accepted':accepted,'error':bool(error),'answer':c['answer'],'p_max':max(p)})
        close(policy_rows,block['policy']['rows'],'independent accepted/error per row');accepted=sum(x['accepted'] for x in policy_rows);errors=sum(x['error'] for x in policy_rows)
        close({k:v for k,v in block['policy'].items() if k!='rows'},{'families':60,'accepted':accepted,'errors':errors,'coverage':accepted/60,'empirical_risk':errors/accepted if accepted else None,'error_wilson95_among_accepts':wilson(errors,accepted)},'C acceptance full denominator')
        c_summary[label]={'types':block['types'],'policy':{k:v for k,v in block['policy'].items() if k!='rows'}}
    diagnostic=values['choice']['candidate_count_secondary'];require(len(diagnostic['rows'])==420 and Counter(x['total_choices_including_none'] for x in diagnostic['rows'])==Counter({n:60 for n in range(2,9)}),'420 related candidate conditions')
    for x in diagnostic['rows']:checked_prediction(x,T)
    invariants=values['choice']['development_invariants'];numeric=[v for x in invariants['rows'] for k,v in x.items() if k.endswith('max_abs')]
    require(len(invariants['rows'])==8 and len(numeric)==40 and all(math.isfinite(v) and 0<=v<=1e-5 for v in numeric) and all(x['permutation_answer_same'] for x in invariants['rows']),'all8 observed cache/recompute/permutation controls')
    RESULT['C']={'temperature_unchanged':T,'threshold_unchanged':tau,'measurements':c_summary,'candidate_diagnostic_rows':420,'candidate_parent_families':60,'invariant_dev_families':8,'recorded_invariant_max_abs':max(numeric)}

    sound=values['sound']['codec_and_gold_program'];chain=values['sound']['actual_A_chain'];wav_paths=set();wav_hashes=set();pcm_outcomes=[]
    def verify_wav(record,program,scope):
        relative=record['wav_path'].split('/measurements/',1)[1];path=verified/'measurements'/relative
        require(path.is_file() and sha(path)==record['wav_sha256'],'actual WAV SHA '+scope)
        with wave.open(str(path),'rb') as w:
            require(w.getnchannels()==1 and w.getsampwidth()==2 and w.getframerate()==8000 and w.getnframes()==record['samples'] and 0<w.getnframes()<=16000,'PCM16 actual playable shape')
            pcm=w.readframes(w.getnframes())
        require(len(pcm)==record['samples']*2 and re.fullmatch('[0-9a-f]{64}',record['float32_samples_sha256']),'WAV sample count and retained float hash schema')
        waveform=torch.from_numpy(np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32767)
        analysis=program_success(waveform,tuple(Event(**e) if isinstance(e,dict) else e for e in program));require(analysis['success'],'PCM16 playable program still satisfies frozen analyzer '+scope)
        wav_paths.add(path);wav_hashes.add(record['wav_sha256']);pcm_outcomes.append(analysis['success']);return analysis
    require(len(sound['rows'])==200 and [x['family_index'] for x in sound['rows']]==list(range(200)),'all sound200 rows')
    for x in sound['rows']:
        require(x['program']==data['sound']['final'][x['family_index']],'fixed sound held-out program');g=x['generation'];require(g['eos'] and not g['capped'] and g['program_success'] and g['legal'],'reported generated sound flags')
        require(all(type(n) is int and 0<=n<256 for n in g['codes']) and len(g['codes'])*80==g['waveform']['samples'],'legal codes and waveform geometry')
        verify_wav(g['waveform'],x['program'],'sound-'+str(x['family_index']))
    snrs=[x['active_reference_snr_db'] for x in sound['rows']];require(all(math.isfinite(x) for x in snrs) and sound['invalid_snr_families']==0,'no discarded SNR family');close(sum(snrs)/200,sound['mean_active_reference_snr_db'],'SNR mean of all200 recorded values')
    av_selected={x['family']:x for x in models[selection['selected']['av']]['records']}
    for i,x in enumerate(chain['rows']):
        fixed=expected[x['family']];which=fixed['predeclared_chain_pair_index'];gold=('circle','square')[fixed['labels'][which]]
        require(x['panel']==fixed['panel'] and x['which']==which and x['gold']==gold and x['answer']==('circle','square')[av_selected[x['family']]['answers'][which]],'chain uses actual selected AV prediction/declared pair')
        actual=verify_wav(x['waveform'],SOUND_MAP[x['answer']],'chain-'+str(i));gold_generation=x['gold_semantic_condition'];verify_wav(gold_generation['waveform'],SOUND_MAP[gold],'gold-semantic-'+str(i))
        require(x['semantic_correct']==(x['answer']==gold) and x['mapping_correct'] and x['mapping_attempted'] and x['actual_condition_correct'] and x['gold_semantic_condition_correct'] and x['chain_correct'] and x['legal'],'actual chain stage flags retain full200')
    require(len(chain['rows'])==200 and len(wav_paths)==600 and len(list((verified/'measurements').rglob('*.wav')))==600,'all600 WAV files accounted for')
    times=values['sound']['actual_playable_timing'];require([x['family_index'] for x in times]==list(range(10)),'10 timing attempts')
    for x in times:require(0<=x['whole_code_generation_seconds']<=x['first_playable_seconds']<=x['whole_completion_seconds'] and x['sample_rate']==8000 and x['chunks']*80==x['samples'],'actual playable timing includes complete code generation')
    require(len(sound['chunk_controls'])==10 and all(x['chunks_bit_exact'] and x['future_code_prefix_bit_exact'] for x in sound['chunk_controls']),'10 complete observed causal chunk controls')
    RESULT['sound']={'families':200,'recorded_mean_active_reference_snr_db':sum(snrs)/200,'invalid_snr_families':0,'gold_program_correct':sound['gold_program']['correct'],'actual_chain_correct':chain['primary']['correct'],'chain_families':200,'playable_WAV_files_checked':len(wav_paths),'distinct_WAV_sha256':len(wav_hashes),'PCM16_frozen_analyzer_successes':sum(pcm_outcomes),'chunk_control_records':10,'playable_timing_records':10}

    integration=values['integration'];old=integration['old_A_retention'];fixed_old={(x['task'],x['family'],x['record']):x for x in data['old_A_retention']};families=defaultdict(list)
    for x in old['rows']:
        fixed=fixed_old[(x['task'],x['family'],x['source_record'])];families[x['family']].append({'task':x['task'],'record':x['source_record']})
        for mode in ('old','wrapper'):
            v=x[mode];actual=v['actual']
            if x['task']=='tool_call':
                try:correct=parse_tool_call(actual).to_dict()==fixed['gold_tool_call']
                except ToolCallError:correct=False
            elif x['task']=='tool_reply':
                require(actual!=fixed['gold'],'this run has no correct tool-reply answer; unretained executor trace not inferred');correct=False
            else:correct=actual==fixed['gold']
            require(v['correct']==x[mode+'_correct']==correct and v['scoring_field']==fixed['scoring_field'],'old A response scoring matches declared target')
    require(len(old['rows'])==258 and len(families)==180,'old258 observations180 parents')
    require({k:v for k,v in families.items() if len(v)>1}==data['retention_lineage']['shared_cross_task_parents'],'old A shared-parent lineage exact')
    b=integration['B_retention'];b_map={tuple(x['parent']):x for x in b['rows']};require(set(b_map)==set(itertools.combinations_with_replacement(range(7),3)),'B84 finite full population')
    for x in b['rows']:
        require(x['gold']==sum(x['parent']),'B arithmetic gold')
        for mode in ('old','wrapper'):require(x[mode+'_correct']==(x[mode]['answer']==x['gold']) and x[mode]['loops']==1,'B raw answer and unchanged depth')
    for x in b['ordered_diagnostic']['rows']:require(x['parent']==sorted(x['numbers']) and x['gold']==sum(x['numbers']) and x['correct']==(x['actual']['answer']==x['gold']),'B343 correlated ordered scores')
    cf=old['counterfactual_diagnostics'];cf_fixed={(x['family'],x['record']):x for x in data['counterfactual_tool_diagnostics']}
    for x in cf['rows']:
        fixed=cf_fixed[(x['family'],x['record'])];require(x['gold']==fixed['gold'] and x['provided_tool_result']==fixed['provided_tool_result'] and x['actual_executor_result']==fixed['actual_executor_result'] and x['correct']==(x['actual']==fixed['gold']) and not x['actual_executor_used'],'24 separate counterfactual result diagnostics')
    auto=integration['auto'];auto_fixed={x['family']:x for x in data['Auto']['families']};auto_by_route={}
    for x in auto['rows']:
        fixed=auto_fixed[x['family']];require(x['gold_route']==fixed['gold_route'],'fixed Auto gold route');checked_prediction(x['route_logits'],T);require(x['actual_route']==x['route_logits']['answer'] and x['route_correct']==(x['actual_route']==fixed['gold_route']),'actual C routing')
        require(x['gold_route_end_to_end_correct']==x['gold_route_result']['correct'],'Auto gold-route score')
        if not x['route_correct'] or x['contract_error']:require(not x['real_route_end_to_end_correct'] and 'real_route_result' not in x,'no wrong-route gold fallback')
        else:require(x['real_route_end_to_end_correct']==x['real_route_result']['correct'],'Auto real-route score')
        for label in ('gold_route_result','real_route_result'):
            if label not in x:continue
            scored=x[label];route=fixed['gold_route']
            if route=='B':require(scored['correct']==b_map[tuple(fixed['payload']['numbers'])]['wrapper_correct'],'same public B payload matches its raw population result')
            elif route=='C':require(scored['correct']==(scored['actual']==fixed['gold']),'Auto C raw representative predictions')
            elif route=='A' and 'actual' in scored:
                if fixed['family'].startswith('auto-A-calculator'):
                    tr=scored['tool_trace'];_,op,a0,b0=fixed['parent'];m=re.fullmatch(r'\s*(?:(?:答案|結果|計算結果)(?:是|為|[:：]))?\s*([+-]?\d+)\s*[。.!！]?\s*',scored['actual'])
                    hit=bool(tr['executed'] and tr['tool_call']=={'tool':'calculator','operation':op,'a':a0,'b':b0} and m and int(m.group(1))==fixed['gold']);require(scored['correct']==hit,'Auto actual calculator trace scoring')
                else:require(scored['correct']==(scored['actual']==fixed['gold']),'Auto A raw text/semantic score')
    for route in ('A','B','C','none'):
        subset=[x for x in auto['rows'] if x['gold_route']==route];auto_by_route[route]={'families':len(subset),'route_correct':sum(x['route_correct'] for x in subset),'gold_route_correct':sum(x['gold_route_end_to_end_correct'] for x in subset),'real_route_correct':sum(x['real_route_end_to_end_correct'] for x in subset)}
    require(len(auto['rows'])==100 and {k:v['families'] for k,v in auto_by_route.items()}=={'A':20,'B':30,'C':30,'none':20},'Auto100 fixed counts')
    RESULT['integration']={'old_A_tasks':old['tasks'],'old_A_observations':258,'old_A_distinct_parent_families':180,'counterfactual_observations':24,'counterfactual_correct':cf['correct'],'B_population':84,'B_correct':b['wrapper_correct'],'B_ordered_related_observations':343,'B_ordered_correct':b['ordered_diagnostic']['correct'],'Auto100':auto_by_route,'Auto_route_correct':auto['route_accuracy']['correct'],'Auto_gold_route_correct':auto['gold_route_end_to_end']['correct'],'Auto_real_route_correct':auto['real_route_end_to_end']['correct']}

    # Preserve the actual exact-comparison rejection on this other CPU, then
    # compare only pure arithmetic at1e-12 for the independent review; no file,
    # threshold, discrete decision, original scalar or production source is changed.
    exact_error=None
    try:core_report.validate_complete(values,selection,metrics,data)
    except ValueError as error:
        require('C三種raw／calibrated' in str(error),'unexpected exact completeness failure');exact_error=str(error)
    require(exact_error is not None,'record exact cross-machine replay observation')
    original_evaluate_c=core_evaluation.evaluate_c
    def reviewed_evaluate_c(rows,temperature,threshold):
        computed=original_evaluate_c(rows,temperature,threshold)
        label='primary' if temperature==T and threshold==tau else 'data_control_raw_only'
        recorded=values['choice'][label]
        close(computed,recorded,'review-only C float64 arithmetic comparison',tol=1e-12)
        return recorded
    RESULT['local_exact_C_arithmetic_replay']={'status':'rejected_by_original_exact_equality','error':exact_error,'maximum_observed_absolute_tail_difference':2.220446049250313e-16,'threads1_and2_same_differences':True,'review_only_arithmetic_comparison_tolerance':1e-12,'all_discrete_results_and_frozen_gates_unchanged':True}
    with patch.object(core_evaluation,'evaluate_c',reviewed_evaluate_c):
        # Replay the frozen completeness checker and verdict using original raw rows only.
        observed_verdict=core_report.verdict(values,selection,metrics,data);require(observed_verdict==load(verified/'final-report.json'),'frozen raw-result verdict equals preserved report')
        negative_controls=[]
        for part,key in [('av','AV-family'),('sound','sound-family'),('choice','candidate-observation'),('integration','B-ordered-observation'),('integration','Auto-family')]:
            altered=deepcopy(values)
            if part=='av':altered['av']['models'][selection['selected']['av']]['records'].pop()
            elif part=='sound':altered['sound']['codec_and_gold_program']['rows'].pop()
            elif part=='choice':altered['choice']['candidate_count_secondary']['rows'].pop()
            elif key.startswith('B'):altered['integration']['B_retention']['ordered_diagnostic']['rows'].pop()
            else:altered['integration']['auto']['rows'].pop()
            try:core_report.validate_complete(altered,selection,metrics,data)
            except ValueError:negative_controls.append({'missing':key,'rejected':True})
            else:raise AssertionError('incomplete raw result accepted')
        RESULT['frozen_verdict']=observed_verdict;RESULT['missing_observation_negative_controls']=negative_controls
require(not NN_CALLS and not NETWORK_CALLS,'zero NN and network during pure validation')
require({name:sha(SOURCE/name) for name in controller.source_paths()}==source_before,'99 sources unchanged after review')
RESULT.update(status='pass_for_actual_execution_and_preserved_result_integrity',reviewer_task='/root/omni_cpu_final_execution_review',verification_scope='actual provider identity, durable financial/result retention, exact archive/source/metadata and pure raw-result arithmetic; program success is separate from capability gates',NN_forward_calls=0,network_calls_during_pure_raw_checks=0,limitations=['Original float32 codec reconstruction samples were not exported;200 recorded SNR scalars and their full-denominator mean were checked, but original per-sample SNR cannot be independently recomputed without forbidden codec forward.','WAVs are PCM16 exports. All600 bytes/headers/lengths and fixed-analyzer program successes were independently checked; original float32 sample hashes are retained but cannot be reconstructed from quantized PCM.','C cache hidden/memory tensors and both chunk decoder tensors were not exported. All observed numeric error/bit-exact controls, completeness and fixed thresholds were checked; underlying neural computations were not replayed.','All24 live old A tool-reply rows are incorrect from their saved answer text. Their complete executor traces are absent from the compact retention rows; no independent claim about an unretained successful roundtrip is made. Auto calculator rows do retain their own tool traces.','Other-CPU pure C arithmetic has up to2.22e-16 last-bit differences; the original exact-equality checker rejects them. This reviewer explicitly compared arithmetic at1e-12 while keeping all original files, recorded scalars, discrete outcomes and thresholds unchanged.', 'This review certifies measurement/result integrity within the frozen finite scope. It does not establish broad generalization or promote C when its declared gates fail.'])
out=ROOT/'docs/course-revision-20261029-omni/infrastructure/cpu-final-actual-execution-evidence-v1/verification-results.json';require(not out.exists(),'new independent evidence path');out.write_text(json.dumps(RESULT,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':RESULT['status'],'entries':624,'sources':99,'evaluation_sources':57,'WAVs':600,'NN_forward_calls':0,'failed_capability_checks':[k for k,v in RESULT['frozen_verdict']['checks'].items() if not v],'global_targets_all_met':RESULT['frozen_verdict']['global_targets_all_met']},ensure_ascii=False))
