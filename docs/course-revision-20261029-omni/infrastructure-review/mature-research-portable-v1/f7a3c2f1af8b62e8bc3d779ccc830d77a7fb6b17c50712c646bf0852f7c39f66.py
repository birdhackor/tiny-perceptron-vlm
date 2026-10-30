"""Actual exact92-source audit gate; all commit/API context is an isolated labeled fixture."""
import hashlib,json,os,shutil,sys
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;R=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(R))
from scripts.omni_revision import research_execution as rx
scope=json.loads((BASE/'current-v3-execution-scope.json').read_bytes());root=BASE/'probes/current-v3-source-gate';root.mkdir(parents=True,exist_ok=False);fixture=root/'source'
for name in scope['files']:
 p=fixture/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/name,p)
audit_path=fixture/'docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v1.json';audit_path.parent.mkdir(parents=True,exist_ok=True);audit={'status':'pass_for_reviewed_scope','reviewer_task':'/root/review_fixture_only','files':scope['files'],'test_fixture':True,'scope':'Private source gate only, not permission to train or infer.'};cases=[]
def git(arguments,**kwargs):
 if arguments==['git','rev-parse','HEAD']:return ('1'*40+'\n').encode()
 assert arguments[:2]==['git','show'];name=arguments[2].split(':',1)[1];raw=(fixture/name).read_bytes()
 return raw+b'\n' if mode=='git_bytes_drift' and name=='scripts/omni_revision/research_model_files.py' else raw
with patch.object(rx,'ROOT',fixture),patch.object(rx,'FROZEN',fixture/'docs/omni-products/v1'),patch.object(rx,'AUDIT',audit_path),patch.object(rx.subprocess,'check_output',side_effect=git):
 for mode in ('stable92','model_helper_drift','qwen_loader_drift','missing_model_helper_binding','self_reviewer','git_bytes_drift','HEAD_mismatch'):
  value=json.loads(json.dumps(audit));changed=None;saved=None
  if mode=='missing_model_helper_binding':value['files'].pop('scripts/omni_revision/research_model_files.py')
  if mode=='self_reviewer':value['reviewer_task']='/root'
  if mode in ('model_helper_drift','qwen_loader_drift'):
   changed=fixture/('scripts/omni_revision/research_model_files.py' if mode=='model_helper_drift' else 'scripts/omni_revision/qwen_research.py');saved=changed.read_bytes();changed.write_bytes(saved+b'\n# explicit isolated source drift\n')
  audit_path.write_text(json.dumps(value))
  with patch.dict(os.environ,{'GITHUB_SHA':'2'*40 if mode=='HEAD_mismatch' else '1'*40}):
   try:source=rx.checked_sources();entry={'case':mode,'accepted':True,'files_checked':len(source['files'])}
   except Exception as e:entry={'case':mode,'accepted':False,'error_type':type(e).__name__,'reason':str(e)}
  if changed is not None:changed.write_bytes(saved)
  assert entry['accepted'] is (mode=='stable92'),entry;cases.append(entry)
record={'scope':'Copied exact92 source bytes, explicit fake audit/readonly git HEAD/show context. Actual checked_sources/committed code; no git commit or provider/model/paid call.','cases':cases,'all_expected_outcomes':True}
p=BASE/'current-v3-source-gate-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
