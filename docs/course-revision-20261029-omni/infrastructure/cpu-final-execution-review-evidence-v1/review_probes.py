"""Independent CPU final controller probes. No network, model forward, training, or final opening."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
import zipfile

import torch
from scripts.omni_revision import cpu_final_execution as r
from scripts.omni_revision import research_execution as re
from tiny_perceptron.lesson_products import delivery
from tiny_perceptron.selftrained import inference as old_inference

BASE = Path('/workspace/work/omni-cpu-final-execution-review-v1')
ROOT = Path('/workspace/work/omni-models-v1')
INV = json.loads((BASE/'author-scope-original.json').read_text())
RESULTS = []
ENV = {'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':r.REPOSITORY,'GITHUB_REF':'refs/heads/main','GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'a'*40,'OMNI_ARTIFACT_OUTCOME':'success'}

def record(name, observed, **details):
    RESULTS.append({'name':name,'observed':observed,**details})

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

# Actual source discovery and bytes are independently pinned before further tests.
assert set(r.source_paths()) == set(INV['files']) and len(r.source_paths()) == 98
assert {p:r.file_sha(ROOT/p) for p in r.source_paths()} == INV['files']
record('exact_source_closure_98', 'pass', count=len(r.source_paths()))

# Test the real audit/source and committed checks using isolated bytes and explicit git responses.
with tempfile.TemporaryDirectory(prefix='sources-',dir=BASE/'probes') as tmp:
    root=Path(tmp); audit=root/'audit.json'
    for name in INV['files']:
        dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(BASE/'original-source'/name,dest)
    valid={'status':'pass_for_reviewed_scope','reviewer_task':'/root/independent-fixture','files':INV['files']}
    write(audit,valid)
    committed_bytes={n:(root/n).read_bytes() for n in INV['files']}
    committed_bytes['audit.json']=audit.read_bytes()
    calls=[]
    mismatch=[None]
    def git(arguments, **kwargs):
        calls.append(arguments)
        if arguments==['git','rev-parse','HEAD']: return ('a'*40+'\n').encode()
        assert arguments[:2]==['git','show']
        name=arguments[2].split(':',1)[1]
        return b'changed committed bytes' if name==mismatch[0] else committed_bytes[name]
    with patch.object(r,'ROOT',root), patch.object(r,'AUDIT',audit), patch.object(re,'ROOT',root), patch.object(r,'source_paths',lambda:sorted(INV['files'])), patch('subprocess.check_output',git), patch.dict(os.environ,ENV):
        r.checked_sources(); record('full_audit_and_all_99_committed_bytes','pass',git_show_calls=sum(c[:2]==['git','show'] for c in calls))
        for name,change in [('wrong_status',{'status':'author_claim'}),('root_not_independent',{'reviewer_task':'/root'}),('missing_file',{'files':{k:v for k,v in INV['files'].items() if k!='uv.lock'}}),('extra_file',{'files':{**INV['files'],'extra':'f'*64}}),('wrong_hash',{'files':{**INV['files'],'uv.lock':'f'*64}})]:
            write(audit,{**valid,**change})
            try:r.checked_sources()
            except (ValueError,KeyError):record('audit_reject_'+name,'pass')
            else:record('audit_reject_'+name,'BUG_ACCEPTED')
        write(audit,valid)
        mismatch[0]='uv.lock'
        try:r.checked_sources()
        except ValueError:record('uncommitted_bytes_rejected','pass')
        else:record('uncommitted_bytes_rejected','BUG_ACCEPTED')
        mismatch[0]=None
        (root/'uv.lock').unlink(); (root/'uv.lock').symlink_to(BASE/'original-source/uv.lock')
        try:r.checked_sources()
        except ValueError:record('source_symlink_rejected','pass')
        else:record('source_symlink_rejected','BUG_ACCEPTED')
        with patch.dict(os.environ,{'GITHUB_SHA':'b'*40}):
            try:r.checked_sources()
            except ValueError:record('checkout_sha_mismatch_rejected','pass')
            else:record('checkout_sha_mismatch_rejected','BUG_ACCEPTED')

# Actual known core ZIP bytes, full archive verifier, and actual strict selection.
# Disable all module calls to make accidental NN forward a hard failure.
core_archive=Path('/workspace/work/omni-core-completed-40988311100/core-checkpoints.zip')
assert core_archive.stat().st_size==54751619 and r.file_sha(core_archive)=='107c418e69f917294c7e2fd1a17b820285e8712d5ea4e1f1d62332285de17aee'
with tempfile.TemporaryDirectory(prefix='core-',dir=BASE/'probes') as tmp:
    r.unpack_verified(core_archive,Path(tmp)/'verified')
    with patch.object(torch.nn.Module,'_call_impl',side_effect=AssertionError('NN forward forbidden')):
        selected=r.core_final.checked_selection(Path(tmp)/'verified/core',r.file_sha(r.FROZEN/'model-selection.json'))
    record('genuine_core_zip_and_all_ten_strict_jobs','pass',archive_bytes=core_archive.stat().st_size,archive_sha256=r.file_sha(core_archive),jobs=sorted(selected['all_formal_jobs']),selected=selected['selected'],temperature=selected['temperature'],threshold=selected['threshold'],forward_calls=0)

# Real data unpacking after an anonymous URL mock; A/B download callers are explicit stubs.
# Existing exact manifest is retained; no teacher or legacy model weight is downloaded.
legacy_archive=Path('/workspace/tiny-perceptron-vlm/assets/training/selftrained-v2.tar.gz')
legacy_manifest=json.loads((ROOT/'docs/selftrained/v2-manifest.json').read_bytes())
assert legacy_archive.stat().st_size==legacy_manifest['package']['bytes']==67862431
assert r.file_sha(legacy_archive)==legacy_manifest['package']['sha256']
with tempfile.TemporaryDirectory(prefix='legacy-',dir=BASE/'probes') as tmp:
    root=Path(tmp); (root/'docs/selftrained').mkdir(parents=True)
    shutil.copyfile(ROOT/'docs/selftrained/v2-manifest.json',root/'docs/selftrained/v2-manifest.json')
    out=root/'control';out.mkdir(); calls=[]
    def urlopen(url,timeout):
        expected=f"https://media.githubusercontent.com/media/{r.REPOSITORY}/{legacy_manifest['package']['revision']}/{legacy_manifest['package']['path']}"
        assert url==expected and timeout==60
        calls.append({'kind':'anonymous_git_lfs','url':url})
        return legacy_archive.open('rb')
    def a(repo,revision,dest,**kw):
        assert repo==r.OLD_A['repo'] and revision==r.OLD_A['revision'] and kw=={'prefix':r.OLD_A['prefix'],'manifest_sha256':r.OLD_A['manifest_sha256']}
        calls.append({'kind':'A_fixed_identity_stub'}); return {'authentication':'disabled','fixture_only':True}
    def b(repo,revision,prefix,sha,dest,product):
        assert (repo,revision,prefix,sha,product)==(r.OLD_B['repo'],r.OLD_B['revision'],r.OLD_B['prefix'],r.OLD_B['manifest_sha256'],'b')
        calls.append({'kind':'B_fixed_identity_stub'})
    def legacy_preflight(data,a_root,b_root):
        rows=[*legacy_manifest['records'],*legacy_manifest['assets']]
        actual={p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
        assert actual=={x['path'] for x in rows}
        assert len(rows)==8962 and sum(x['bytes'] for x in rows)==127161811
        for row in rows:
            p=data/row['path']; assert p.stat().st_size==row['bytes'] and r.file_sha(p)==row['sha256']
        return None,None,{'no_model_predictions':True,'data_files':len(rows),'data_bytes':sum(x['bytes'] for x in rows),'legacy_loaders':'explicit_stub'}
    with patch.object(r,'ROOT',root),patch.object(r.urllib.request,'urlopen',urlopen),patch.object(old_inference,'fetch_public_export',a),patch.object(delivery,'download_product',b),patch.object(r.core_final,'legacy_preflight',legacy_preflight):
        r.legacy_inputs(out)
    receipt=json.loads((out/'legacy-preflight.json').read_text())
    assert not (root/'outputs/omni-final-legacy-v1/data.tar.gz').exists()
    record('genuine_8962_member_tar_unpack_after_anonymous_mock','pass',archive_bytes=legacy_archive.stat().st_size,archive_sha256=r.file_sha(legacy_archive),data_files=receipt['data_files'],data_bytes=receipt['data_bytes'],calls=calls,A_B_real_loaders_not_executed=True)

# Finish operation: real finance.finish, isolated ledger, fully explicit provider responses.
def finish_case(mutation=None):
    with tempfile.TemporaryDirectory(prefix='finish-',dir=BASE/'probes') as tmp:
        out=Path(tmp)
        row={'attempt_id':'cpu-final-123-1','revision':'a'*40,'purpose':json.dumps({'job':r.JOB}),'reserved_usd':'0.10','components_usd':r.transport.COMPONENTS,'status':'reserved'}
        receipt={'attempt':'cpu-final-123-1','source_revision':'a'*40,'success':True,'all_training_processes_exited':True,'no_modal_compute_created':True,'release_id':7,'release_tag':'omni-final-candidate-cpu-final-123-1','assets':[{'id':9,'name':'final-results.zip','bytes':10,'sha256':'c'*64,'roundtrip_verified':True}],'completed_parts':['av','sound','choice','integration']}
        ownership={'body':'Owned unpublished fixed core final measurements; cpu-final-123-1; source '+'a'*40,'tag':receipt['release_tag'],'release_id':7}
        artifact={'id':8,'name':'omni-final-123-1','size_in_bytes':100,'expired':False,'created_at':'2026-10-30T00:00:00Z','expires_at':'2026-11-06T00:00:00Z'}
        owned={'id':7,'draft':True,'body':ownership['body'],'tag_name':receipt['release_tag'],'target_commitish':receipt['source_revision'],'assets':[{'id':9,'name':'final-results.zip','size':10}]}
        record_api={'total_count':1,'artifacts':[artifact]}
        if mutation:mutation(row,receipt,ownership,owned,record_api)
        write(out/'reservation.json',{'attempt':row});write(out/'cpu-final-receipt.json',receipt);write(out/'release-ownership.json',ownership)
        saved=[]
        class Store:
            def read(self):return {'attempts':[deepcopy(row)]},'old'
            def write(self,state,blob):assert blob=='old'; saved.append(state);return {'commit':'fixture','blob':'fixture'}
        def api(endpoint,*args):
            if endpoint.startswith('releases/'):return owned
            if endpoint.startswith('actions/runs/123/artifacts?'):return record_api
            raise AssertionError(endpoint)
        with patch.dict(os.environ,ENV),patch.object(r,'GitHubLedger',Store),patch.object(r,'api',api):
            try:r.finish_final('cpu-final-123-1',out)
            except Exception as e:return 'rejected',type(e).__name__
        assert len(saved)==1
        return saved[0]['attempts'][0]['status'],None

for name,mutation,expected in [
 ('finish_true_complete',None,'completed'),
 ('failed_work_not_completed',lambda row,rec,owner,owned,api:rec.update(success=False),'failed-stopped'),
 ('children_unknown',lambda row,rec,owner,owned,api:rec.update(all_training_processes_exited=False),'unresolved'),
 ('foreign_release_target',lambda row,rec,owner,owned,api:owned.update(target_commitish='b'*40),'unresolved'),
 ('foreign_release_body',lambda row,rec,owner,owned,api:owned.update(body='foreign'),'unresolved'),
 ('extra_release_asset',lambda row,rec,owner,owned,api:owned['assets'].append({'id':99,'name':'foreign','size':1}),'unresolved'),
 ('missing_roundtrip',lambda row,rec,owner,owned,api:rec['assets'][0].update(roundtrip_verified=False),'unresolved'),
 ('provider_artifact_too_large',lambda row,rec,owner,owned,api:api['artifacts'][0].update(size_in_bytes=256*1024**2+1),'unresolved'),
 ('provider_artifact_too_long',lambda row,rec,owner,owned,api:api['artifacts'][0].update(expires_at='2026-11-07T00:00:00Z'),'unresolved'),
 ('provider_artifact_expired',lambda row,rec,owner,owned,api:api['artifacts'][0].update(expired=True),'unresolved'),
 ('provider_artifact_wrong_name',lambda row,rec,owner,owned,api:api['artifacts'][0].update(name='foreign'),'unresolved'),
 ('provider_artifact_missing',lambda row,rec,owner,owned,api:api.update(total_count=0,artifacts=[]),'unresolved'),
 ('provider_artifact_pagination',lambda row,rec,owner,owned,api:api.update(total_count=101),'unresolved'),
 ('foreign_receipt_attempt',lambda row,rec,owner,owned,api:rec.update(attempt='cpu-final-999-1'),'rejected'),
 ('foreign_receipt_source',lambda row,rec,owner,owned,api:(rec.update(source_revision='b'*40),owned.update(target_commitish='b'*40)),'rejected'),
 ('foreign_receipt_release',lambda row,rec,owner,owned,api:(rec.update(release_tag='omni-final-candidate-cpu-final-999-1'),owner.update(tag='omni-final-candidate-cpu-final-999-1',body='foreign-but-owned'),owned.update(tag_name='omni-final-candidate-cpu-final-999-1',body='foreign-but-owned')),'rejected'),
]:
    status,error=finish_case(mutation)
    record(name,'pass' if status==expected else 'BUG_ACCEPTED',expected=expected,status=status,error_type=error)

# Explicit provider outage must not claim a ledger completion; record whether it can reconcile unresolved.
# This path currently raises before finance.finish; a reserved ledger remains conservative.

# Download size/SHA counterexample is performed on a tiny private manifest, not real data.
with tempfile.TemporaryDirectory(prefix='legacy-bad-',dir=BASE/'probes') as tmp:
    root=Path(tmp); frozen=root/'frozen';frozen.mkdir()
    manifest={'package':{'kind':'git-lfs','repo_id':r.REPOSITORY,'revision':'a'*40,'path':'assets/training/selftrained-v2.tar.gz','bytes':4,'unpacked_bytes':1,'sha256':'f'*64},'records':[{'path':'x','bytes':1,'sha256':'f'*64}],'assets':[]}
    write(root/'docs/selftrained/v2-manifest.json',manifest)
    write(frozen/'data-definition.json',{'files_sha256':{'docs/selftrained/v2-manifest.json':r.file_sha(root/'docs/selftrained/v2-manifest.json')}})
    with patch.object(r,'ROOT',root),patch.object(r,'FROZEN',frozen),patch.object(r.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(b'12345')):
        try:r.legacy_inputs(root/'out')
        except ValueError as e:record('oversized_legacy_download_rejected','pass',error=str(e))
        else:record('oversized_legacy_download_rejected','BUG_ACCEPTED')

(BASE/'original-probe-results.json').write_text(json.dumps({'reviewer_task':'/root/omni_cpu_final_execution_review','no_network':True,'no_forward_or_training':True,'no_final_opening_created':True,'results':RESULTS},indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'results':len(RESULTS),'issues':[x for x in RESULTS if x['observed']!='pass']},ensure_ascii=False,indent=2))
