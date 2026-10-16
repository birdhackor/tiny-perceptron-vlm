"""最小draft權限修正：唯讀diff/log與離線403前置失敗、同包completed重跑守門。"""
import contextlib
import copy
import difflib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC,datetime
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
TREE=Path('/workspace/work/lesson-products-transport')
ART=Path(__file__).parent
sys.path.insert(0,str(TREE))
from scripts.lesson_products import publish
from scripts.selftrained import finance


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 start=datetime.now(UTC).isoformat()
 wf='.github/workflows/lesson-products-publish.yml';previous=subprocess.check_output(['git','show','dddf58cde65cc5aeb6ea24b67fd07eefcc0199a3:'+wf],cwd=TREE).decode();current=(TREE/wf).read_text()
 inserted='  # GitHub requires push permission to read a draft release and its transport asset.\n  contents: write\n'
 assert current.replace(inserted,'  contents: read\n')==previous
 spec=json.loads((TREE/'docs/lesson-products/jobs/release-v1.json').read_bytes())
 old=json.loads((ROOT/'docs/lesson-products/reviews/engineering-publication-actual-inputs-original.json').read_bytes())
 old_hashes={row['path']:row['sha256'] for row in old['checked_files']}
 unchanged=['scripts/lesson_products/publish.py','scripts/lesson_products/publication.py','scripts/lesson_products/modal_runner.py','scripts/lesson_products/checkpoints.py','scripts/selftrained/modal_runner.py','scripts/selftrained/finance.py',publish.POLICY_PATH,'docs/lesson-products/data-v2-manifest.json','docs/lesson-products/jobs/release-v1.json']
 for name in unchanged:assert sha(TREE/name)==old_hashes[str(TREE/name)]
 archive=ROOT/'outputs/lesson-products/lesson-products-v1-20261016-release.tar.gz';assert sha(archive)==spec['transport_sha256']
 log=Path('/workspace/work/curriculum-20261016/publication-first-failure.log');raw=log.read_bytes();lines=raw.decode(errors='replace').splitlines()
 selected=[]
 for number,line in enumerate(lines,1):
  body=line.split('\t')[-1]
  if ('gh: Resource not accessible by integration (HTTP 403)' in body or ('File ' in body and 'scripts/lesson_products/publish.py' in body) or 'parent = json.loads(subprocess.check_output' in body or "subprocess.CalledProcessError: Command '['gh', 'api', 'repos/birdhackor/tiny-perceptron-vlm/releases/414676375']'" in body or 'Process completed with exit code 1.' in body):
   selected.append({'line':number,'text':body})
 assert any('HTTP 403' in x['text'] for x in selected) and any('line 287' in x['text'] for x in selected)
 assert all('env:' not in x['text'] and 'TOKEN' not in x['text'] for x in selected)
 results=[]
 with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='draft-permission-audit-') as temporary:
  work=Path(temporary)
  for mode in ('403_before_modal','metadata_success_until_live_boundary'):
   folder=work/mode;folder.mkdir();bundle=folder/'transport.tar.gz';os.link(archive,bundle)
   counts={'metadata':0,'read_live':0,'reserve':0,'register':0,'download':0}
   class BoundaryReached(Exception):pass
   def metadata(command):
    counts['metadata']+=1
    assert command==['gh','api','repos/birdhackor/tiny-perceptron-vlm/releases/414676375']
    if mode=='403_before_modal':raise subprocess.CalledProcessError(1,command,stderr=b'HTTP403 fixture')
    return json.dumps({'draft':True,'target_commitish':spec['release_commit'],'assets':[{'id':spec['asset_id'],'name':spec['asset_name'],'size':spec['asset_bytes']}]}).encode()
   def live(*args):counts['read_live']+=1;raise BoundaryReached('stop before any real billing/Modal API')
   def reserve(*args):counts['reserve']+=1;raise AssertionError('no reserve allowed')
   def register(*args):counts['register']+=1;raise AssertionError('no register allowed')
   def download(*args,**kwargs):counts['download']+=1;raise AssertionError('cached exact bundle should not download')
   argv=['publish','--spec','docs/lesson-products/jobs/release-v1.json','--revision','a'*40,'--run-id','permission-audit-'+mode,'--bundle',str(bundle),'--release-repo','synthetic/unused']
   caught=None
   with patch.object(sys,'argv',argv),patch.dict(os.environ,{'GITHUB_REPOSITORY':'birdhackor/tiny-perceptron-vlm'}),patch.object(publish,'inputs',return_value=(spec,{})),patch.object(publish.subprocess,'check_output',side_effect=metadata),patch.object(publish.subprocess,'run',side_effect=download),patch.object(publish,'read_live_ledger',side_effect=live),patch.object(publish,'reserve_entry',side_effect=reserve),patch.object(publish,'register',side_effect=register),contextlib.redirect_stdout(io.StringIO()):
    try:publish.main()
    except (subprocess.CalledProcessError,BoundaryReached) as error:caught=type(error).__name__
   assert caught==('CalledProcessError' if mode=='403_before_modal' else 'BoundaryReached')
   assert counts=={'metadata':1,'read_live':0 if mode=='403_before_modal' else 1,'reserve':0,'register':0,'download':0}
   assert sha(bundle)==spec['transport_sha256']
   results.append({'id':mode,'passed':True,'counts':counts,'expected_stop':caught,'scope':'Actual current main with actual cached package, fixture GitHub metadata and explicit stop before any cloud/ledger provider.'})
 # Real duplicate-completed guard before new reservation; pure quota arithmetic is not
 # the subject here, so check_quota is a recorded in-memory stub, no official account API.
 book={'reserved_total_usd':'10.00','reservations':[{'run_id':'old-history','reserved_usd':'10.00','status':'completed'}]}
 policy=json.loads((TREE/publish.POLICY_PATH).read_bytes())
 live={'rates':{'cpu_hour_cost':'.04730','mem_gib_hour_cost':'.00800','egress_gib_cost':'0'},'gross_quota_snapshot':{}}
 with patch.object(finance,'check_quota',return_value=({'synthetic':True},{'synthetic':True})):
  entry,following=publish.reserve_entry(book,'earlier-attempt','lesson-products-release-v1','b'*40,spec['transport_sha256'],spec,publish.canonical_sha(spec),live,{},policy,publish.POLICY_SHA,None)
  entry['experiment_id']='selftrained_lesson_products_release'
  publish.finish_entry(book,'earlier-attempt','completed',{'status':'completed'})
  unchanged_book=copy.deepcopy(book);error=None
  try:publish.reserve_entry(book,'new-attempt','lesson-products-release-v1','c'*40,spec['transport_sha256'],spec,publish.canonical_sha(spec),live,{},policy,publish.POLICY_SHA,following)
  except ValueError as raised:error=str(raised)
  assert error and 'already completed' in error and book==unchanged_book
 results.append({'id':'same_immutable_completed_package_refuses_new_reservation','passed':True,'error':error,'ledger_unchanged':True,'scope':'Actual reserve_entry/finish_entry duplicate guard; fixture policy snapshot/check_quota, no account API.'})
 assert sha(archive)==spec['transport_sha256']
 report={'reviewer_task':'/root/tech_20261016_review_tools','context':'informed draft-permission callback','started_at':start,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},'source_worktree':str(TREE),'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip(),'checked_files':[{'path':str(TREE/name),'sha256':sha(TREE/name)} for name in [wf,*unchanged]],'workflow_diff':''.join(difflib.unified_diff(previous.splitlines(True),current.splitlines(True),fromfile='dddf58c',tofile='pending_contents_write')),'actual_failure_log':{'path':str(log),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'selected_events_only':selected,'environment_not_copied':True,'attribution':'root-provided Actions38870254102 failed-step log; reviewer personally inspected this local receipt, not producer of the run.'},'package_sha256':spec['transport_sha256'],'file_count':len(spec['files']),'results':results,'all_expected':True,'limits':['No dispatch, token permission test, Modal/register/reserve API, GPU, paid operation, HF/write or Git mutation. Positive metadata fixture proves program order, not actual new workflow token succeeds.','403 control flow establishes this failed invocation did not reach live ledger/provider/register; not a global assertion about unrelated attempts.','Workflow is pending one-permission change and comment; real inputs will need new committed HEAD before dispatch because old inputs correctly reject changed workflow bytes.']}
 p=ART/'engineering-publication-draft-permission-results-callback-01.json'
 with p.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'cases':len(results),'all_expected':True,'sha256':sha(p),'workflow_sha256':sha(TREE/wf),'same_package':spec['transport_sha256']}))


if __name__=='__main__':main()
