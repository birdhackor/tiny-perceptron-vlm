"""Independent reserve/CAS, single run, partial preservation, child and upload probes; no NN/network."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
from unittest.mock import patch
import zipfile

from scripts.omni_revision import cpu_final_execution as r
from scripts.omni_revision import finance as f

BASE=Path('/workspace/work/omni-cpu-final-execution-review-v1')
ROOT=Path('/workspace/work/omni-models-v1')
RESULTS=[]
ENV={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':r.REPOSITORY,'GITHUB_REF':'refs/heads/main','GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'a'*40}

def record(name,**details):RESULTS.append({'name':name,'observed':'pass',**details})
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def snapshot(at):return {'source':f.SOURCE,'workspace':'birdhackor','observed_at':at.isoformat(),'summaries':[{'cycle':'2026-10','start':'2026-10-01T00:00:00+00:00','end':'2026-11-01T00:00:00+00:00','metered_cost_usd':'8.00'}]}
def inventory(at):
    report={'retained_bounds':[],'carry_usd':'0','historical_audit_total_usd':'48.4','historical_audit_excluded_from_carry':True,'month_gross_highwater_usd':{'2026-10':'8.00'}}
    recon={'schema':'omni-legacy-reconciliation-v1','reviewer_task':'/root/independent-fixture','ledger_sha256':'b'*64,'quota_sha256':'c'*64,'legacy_policy_sha256':'d'*64,'legacy_verification_sha256':f.digest(report),'existing_volumes_sha256':f.digest([]),'retained_bounds':[]}
    return {'workspace':'birdhackor','environment':'main','observed_at':at.isoformat(),'ledger_sha256':'b'*64,'quota_sha256':'c'*64,'legacy_policy_sha256':'d'*64,'legacy_verification':report,'existing_volumes':[],'reconciliation':recon,'unknown_attempts':[],'active_apps':[],'errors':[]}

for mode in ['normal','stale','conflict','prior_final']:
    with tempfile.TemporaryDirectory(prefix='reserve-',dir=BASE/'probes') as tmp:
        folder=Path(tmp);output=folder/'out';ghout=folder/'gh-output';ghout.touch()
        now=datetime.now(UTC);old=now-timedelta(seconds=30)
        state=f.new_state(snapshot(old),inventory(old),'a'*40,now=old)
        if mode=='prior_final':state['attempts']=[{'purpose':json.dumps({'job':r.JOB}),'status':'failed-stopped'}]
        events=[];saved=[]
        class Store:
            def read(self):events.append('read');return deepcopy(state),'original-blob'
            def write(self,value,blob):
                assert blob=='original-blob';events.append('CAS')
                if mode=='conflict':raise RuntimeError('explicit fixture HTTP409')
                saved.append(value);return {'commit':'fixture','blob':'fixture'}
        def core(revision,state):events.append('strict_core');return {'core_selection_sha256':'c'*64,'core_final_opening_sha256':'d'*64}
        def child(args,log,seconds,state):
            assert args[2]=='legacy' and seconds==900;events.append('legacy_child');state['all_training_processes_exited']=True
        modal=types.ModuleType('modal');modal.Workspace=types.SimpleNamespace(from_context=lambda:object())
        snap=snapshot(now-timedelta(seconds=121)) if mode=='stale' else snapshot(now)
        with patch.dict(os.environ,{**ENV,'GITHUB_OUTPUT':str(ghout)}),patch.dict(sys.modules,{'modal':modal}),patch.object(r,'GitHubLedger',Store),patch.object(r,'checked_sources',lambda:{'revision':'a'*40,'files':{}}),patch.object(r,'prepare_core',core),patch.object(r.transport,'bounded_child',child),patch.object(r.transport,'pricing',lambda:{'explicit':'pricing fixture'}),patch.object(r.preflight,'load_reconciliation',lambda *_:inventory(now)['reconciliation']),patch.object(r.preflight,'inventory',lambda *_:inventory(now)),patch.object(r.preflight,'official_snapshot',lambda *_:snap):
            try:r.reserve_final('cpu-final-123-1',output)
            except (ValueError,RuntimeError) as error:
                assert mode!='normal' and not saved and ghout.read_text()=='' and not (output/'reservation.json').exists()
                if mode=='prior_final':assert events==['read']
                record('reserve_'+mode+'_stops_before_run_enable',events=events,error_type=type(error).__name__)
            else:
                assert mode=='normal' and ghout.read_text()=='reserved=true\n' and len(saved)==1
                row=saved[0]['attempts'][-1]
                assert row['reserved_usd']=='0.10' and row['status']=='reserved' and json.loads(row['purpose'])['job']==r.JOB
                assert row['components_usd']==r.transport.COMPONENTS
                record('reserve_normal_fresh_billing_CAS_and_retained_bound',events=events,bound=row['reserved_usd'],core_protection=json.loads((output/'future-protection.json').read_text()))

# Real shared child helper with synthetic programs, stripped credential placeholders, and an actual kill.
with tempfile.TemporaryDirectory(prefix='child-',dir=BASE/'probes') as tmp:
    root=Path(tmp);state={'all_training_processes_exited':True}
    keys=('GH_TOKEN','GITHUB_TOKEN','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET')
    with patch.dict(os.environ,{key:'fixture-not-a-credential' for key in keys}):
        r.transport.bounded_child(['-c',"import os; assert all(k not in os.environ for k in ('GH_TOKEN','GITHUB_TOKEN','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET')); print('sanitized')"],root/'clean.log',5,state)
    assert state['all_training_processes_exited'] is True
    record('actual_child_exits_without_GH_or_Modal_credentials')
    try:r.transport.bounded_child(['-c','import time; time.sleep(5)'],root/'deadline.log',0.01,state)
    except TimeoutError:assert state['all_training_processes_exited'] is True;record('actual_child_absolute_deadline_kill_wait_observed')
    else:raise AssertionError('deadline ignored')

for failed in [None,'sound','report']:
    with tempfile.TemporaryDirectory(prefix='run-',dir=BASE/'probes') as tmp:
        root=Path(tmp);out=root/'out';out.mkdir();frozen=root/'frozen';frozen.mkdir()
        (frozen/'model-selection.json').write_bytes(b'explicit fixed selection')
        selected=r.file_sha(frozen/'model-selection.json')
        row={'attempt_id':'cpu-final-123-1','status':'reserved','revision':'a'*40,'purpose':json.dumps({'job':r.JOB}),'reserved_usd':'0.10','components_usd':r.transport.COMPONENTS}
        write(out/'reservation.json',{'attempt':row});write(out/'sources.json',{'fixture':True});write(out/'core-inputs.json',{'fixture':True})
        calls=[];uploads=[]
        class Store:
            def read(self):return {'attempts':[row]},'blob'
        def child(args,log,seconds,state):
            log.write_text('explicit synthetic child, no model calls\n');state['all_training_processes_exited']=True
            report=args[1].endswith('core_report')
            part='report' if report else args[2];calls.append(part)
            assert seconds==(120 if report else 1500)
            if not report:assert args[args.index('--selection-sha256')+1]==selected
            target=Path(args[args.index('--output')+1])
            if report:target.write_text('{"fixture_report":true}')
            else:
                target.mkdir(parents=True);(target/'partial.json').write_text(json.dumps({'part':part,'fixture':True}))
            if part==failed:raise RuntimeError('explicit synthetic failure')
        def api(endpoint,*args):return {'id':7}
        def upload(tag,body,rid,path):
            assert rid==7 and tag=='omni-final-candidate-cpu-final-123-1' and 'source '+'a'*40 in body
            with zipfile.ZipFile(path) as z:
                manifest=json.loads(z.read('package-file-manifest.json'))
                for name,binding in manifest.items():assert len(z.read(name))==binding['bytes'] and r.transport.hashlib.sha256(z.read(name)).hexdigest()==binding['sha256']
                uploads.append(sorted(z.namelist()))
            return {'id':9,'name':path.name,'bytes':path.stat().st_size,'sha256':r.file_sha(path),'roundtrip_verified':True}
        with patch.dict(os.environ,ENV),patch.object(r,'ROOT',root),patch.object(r,'FROZEN',frozen),patch.object(r,'GitHubLedger',Store),patch.object(r,'checked_sources',lambda:{'revision':'a'*40}),patch.object(r,'find_draft',lambda *_:None),patch.object(r,'api',api),patch.object(r.transport,'bounded_child',child),patch.object(r.transport,'upload_owned',upload):
            try:r.run_final('cpu-final-123-1',out)
            except RuntimeError:assert failed is not None
            else:assert failed is None
            assert len(uploads)==1
            receipt=(out/'cpu-final-receipt.json').read_bytes();parsed=json.loads(receipt)
            if failed=='sound':
                assert calls==['av','sound'] and parsed['completed_parts']==['av'] and parsed['success'] is False
                assert 'measurements/sound/partial.json' in uploads[0] and 'measurements/av/partial.json' in uploads[0]
            else:assert calls==['av','sound','choice','integration','report']
            assert parsed['success']==(failed is None)
            try:r.run_final('cpu-final-123-1',out)
            except FileExistsError:pass
            else:raise AssertionError('second run admitted')
            assert (out/'cpu-final-receipt.json').read_bytes()==receipt
        record('run_order_fixed_selection_no_retry_'+str(failed),calls=calls,package_members=uploads[0])

# Real upload helper validates before-write ownership and verifies downloaded bytes on return.
for mode in ['valid','foreign','existing','wrong_digest','wrong_readback']:
    with tempfile.TemporaryDirectory(prefix='upload-',dir=BASE/'probes') as tmp:
        root=Path(tmp);path=root/'final-results.zip';path.write_bytes(b'explicit owned synthetic result')
        before={'id':7,'draft':True,'tag_name':'tag','body':'body','assets':[]}
        if mode=='foreign':before['body']='foreign'
        if mode=='existing':before['assets']=[{'id':9,'name':path.name,'size':path.stat().st_size}]
        after={**before,'assets':[{'id':9,'name':path.name,'size':path.stat().st_size,'digest':'sha256:'+(r.file_sha(path) if mode!='wrong_digest' else 'f'*64)}]}
        apicalls=[];commands=[]
        def api(endpoint):apicalls.append(endpoint);return before if len(apicalls)==1 else after
        def command(args):
            commands.append(args[2])
            if args[2]=='download':(Path(args[-1])/path.name).write_bytes(path.read_bytes() if mode!='wrong_readback' else b'changed')
        with patch.object(r.transport,'api',api),patch.object(r.transport,'command',command):
            try:value=r.transport.upload_owned('tag','body',7,path)
            except ValueError:assert mode!='valid'
            else:assert mode=='valid' and value['roundtrip_verified'] is True
        if mode in ['foreign','existing']:assert commands==[]
        record('upload_'+mode,commands=commands)

final=(ROOT/'.github/workflows/omni-final-cpu.yml').read_text()
for file in ['omni-core-cpu.yml','omni-mature-research.yml']:
    text=(ROOT/'.github/workflows'/file).read_text()
    assert 'group: tiny-perceptron-course-gpu' in final and 'group: tiny-perceptron-course-gpu' in text and 'cancel-in-progress: false' in final and 'cancel-in-progress: false' in text
assert "python-version: '3.13.5'" in final and 'runs-on: ubuntu-latest' in final and 'timeout-minutes: 180' in final
assert "steps.reserve.outputs.reserved == 'true'" in final and 'retention-days: 7' in final and 'persist-credentials: false' in final
record('static_workflow_fixed_python_shared_finance_concurrency_and_preservation')
(BASE/'original-control-probe-results.json').write_text(json.dumps({'reviewer_task':'/root/omni_cpu_final_execution_review','no_network':True,'no_forward_or_training':True,'results':RESULTS},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'results':len(RESULTS),'issues':[]},ensure_ascii=False))
