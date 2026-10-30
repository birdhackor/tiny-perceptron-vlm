"""真 checked_sources 的私有commit fixture＋逐96檔漂移；真工作樹仍缺permit/audit並拒絕。"""

import hashlib,json,os,sys
from datetime import UTC,datetime
from pathlib import Path
from unittest.mock import patch

ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
from scripts.omni_revision import research_execution as runner,research_recovery as recovery
revision=json.loads((HERE/'source-snapshot-manifest.json').read_bytes())['head']
assert (ROOT/recovery.PERMIT_NAME).exists() and not runner.AUDIT.exists()
try:
 with patch.dict(os.environ,{'GITHUB_SHA':revision}):runner.checked_sources()
except FileNotFoundError:actual_blocked=True
else:raise AssertionError('missing new audit admitted')
private=HERE/'sources';permit=private/recovery.PERMIT_NAME;assert permit.read_bytes()==(ROOT/recovery.PERMIT_NAME).read_bytes()
audit=private/'docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v3.json'
real_sha=runner.file_sha
calls=[]
def git(args,**kwargs):
 calls.append(args)
 if args==['git','rev-parse','HEAD']:return revision.encode()
 assert args[:2]==['git','show'] and args[2].startswith(revision+':')
 return (private/args[2].split(':',1)[1]).read_bytes()

with patch.object(runner,'ROOT',private),patch.object(runner,'FROZEN',private/'docs/omni-products/v1'),patch.object(runner,'AUDIT',audit):
 names=runner.source_paths();assert len(names)==96
 files={name:real_sha(private/name) for name in names}
 audit.write_text(json.dumps({'status':'pass_for_reviewed_scope','reviewer_task':'/root/omni_development_execution_review/private_fixture','files':files,'private_fixture_not_operative':True}))
 with patch.dict(os.environ,{'GITHUB_SHA':revision}),patch.object(runner.subprocess,'check_output',side_effect=git):
  result=runner.checked_sources();assert result['files']==files and len(result['files'])==96
  rejected=[]
  for name in names:
   before=len(calls)
   def changed(path):return '0'*64 if Path(path)==private/name else real_sha(path)
   with patch.object(runner,'file_sha',side_effect=changed):
    try:runner.checked_sources()
    except ValueError:rejected.append(name)
    else:raise AssertionError('source drift admitted '+name)
   assert len(calls)-before==1,'must reject mismatched source map before git show/any provider'
  with patch.dict(os.environ,{'GITHUB_SHA':'0'*40}):
   try:runner.checked_sources()
   except ValueError:pass
   else:raise AssertionError('wrong revision admitted')
assert len(rejected)==96
out={'schema':'omni-mature-runtime-current-source-admission-probe-v1','created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'actual_worktree_missing_new_audit_rejected':actual_blocked,'private_snapshot_source_count':96,'private_all96_committed_bytes_positive':True,'private_individual_source_hash_drift_rejected':rejected,'wrong_github_sha_rejected':True,'private_git_fixture_not_actual_commit':True,'real_ledger_writes':0,'real_modal_rpc_calls':0,'operative_permit_or_audit_created_in_repository':False,'limitation':'Private source admission fixture supplies matching git show bytes; it proves the guard logic only. The actual corrected source is uncommitted and must remain blocked until all96 plus audit are committed on main.'}
(HERE/'source-admission-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_assertions_passed':True,'actual_missing_new_audit_blocked':actual_blocked,'individual_drift_negative_cases':len(rejected),'operative_files_created':False}))
