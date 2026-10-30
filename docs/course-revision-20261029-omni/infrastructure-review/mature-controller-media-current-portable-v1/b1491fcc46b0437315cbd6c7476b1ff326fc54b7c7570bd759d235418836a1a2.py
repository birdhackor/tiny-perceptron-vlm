"""新current source/science/USED permit/ordinary Qwen once唯讀核對，不新發permit。"""
import copy,hashlib,json,subprocess,sys,zipfile
from datetime import UTC,datetime
from pathlib import Path
ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
from scripts.omni_revision import research_execution as r,research_recovery as recovery,core_final
from tiny_perceptron.omni_products.training_runtime import source_binding
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
snap=read(HERE/'source-snapshot-manifest.json');files={x['path']:x['sha256'] for x in snap['files']};assert len(files)==96 and set(files)==set(r.source_paths())
base=snap['head'];changed=[]
for name,h in files.items():
 assert sha(ROOT/name)==h
 if hashlib.sha256(subprocess.check_output(['git','show',base+':'+name],cwd=ROOT)).hexdigest()!=h:changed.append(name)
assert set(changed)=={'.github/workflows/omni-mature-research.yml','scripts/omni_revision/research_execution.py'}
assert (ROOT/'scripts/omni_revision/research_execution.py').read_bytes().replace(b'mature-research-audit-v4.json',b'mature-research-audit-v3.json')==subprocess.check_output(['git','show',base+':scripts/omni_revision/research_execution.py'],cwd=ROOT)
workers=r.worker_files();assert len(workers)==62
with zipfile.ZipFile('/workspace/work/omni-kev-first-preparation-failure-41004743884/input.zip') as z:
 for name in workers:assert z.read(name)==(ROOT/name).read_bytes()
evaluation=core_final.evaluation_sources();assert len(evaluation)==57
for name,h in evaluation.items():assert hashlib.sha256(subprocess.check_output(['git','show',base+':'+name],cwd=ROOT)).hexdigest()==h
recipe=read(ROOT/'docs/omni-products/v1/formal-recipes.json');assert len(recipe['execution_sources_sha256'])==25
for name,h in recipe['execution_sources_sha256'].items():assert sha(ROOT/name)==h
assert source_binding()==recipe['implementation_sha256'] and len(source_binding())==8
assert sha(ROOT/'scripts/omni_revision/research_cases.py')==hashlib.sha256(subprocess.check_output(['git','show',base+':scripts/omni_revision/research_cases.py'],cwd=ROOT)).hexdigest()
permit,permit_sha=recovery.checked_permit();assert permit_sha=='929b40a464f2cedea3021e33db3fcdabc6570552aa361dfc969ce17fd0be2fd8'
lr=read(HERE/'provider-ledger.json')['structuredContent'];state=json.loads(lr['content']);assert state['reserved_total_usd']=='10.90'
consumed=[row for row in state['attempts'] if recovery.purpose(row).get('preparation_recovery_permit_sha256')==permit_sha];assert len(consumed)==1 and consumed[0]['attempt_id']=='mature-kev-41011677439-1' and consumed[0]['status']=='completed'
assert not any(recovery.purpose(row).get('job')=='mature-qwen25' for row in state['attempts'])
assert recovery.eligibility(state,'qwen25') is None
try:recovery.eligibility(state,'kev')
except ValueError:kev_closed=True
else:raise AssertionError('used Kev permit reopened')
once=[]
for status in ('reserved','unresolved','failed-stopped','completed'):
 altered=copy.deepcopy(state);altered['attempts'].append({'attempt_id':'private-Qwen-once-row','status':status,'purpose':json.dumps({'job':'mature-qwen25'})})
 try:recovery.eligibility(altered,'qwen25')
 except ValueError:once.append({'prior_status':status,'rejected':True})
 else:raise AssertionError('Qwen repeat allowed')
out={'schema':'omni-mature-mr09-current-unchanged-scope-once-v1','created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'source_paths':96,'changed_only':changed,'controller_change_only_audit_filename':True,'generator_unchanged':True,'worker62_original_bytes':True,'final57_training25_model8_unchanged':True,'unchanged_permit_sha256':permit_sha,'permit_used':True,'consuming_attempt':consumed[0]['attempt_id'],'kev_remaining_attempts':0,'kev_guard_closed':kev_closed,'qwen_prior_paid_rows':0,'qwen_ordinary_original_once_available':True,'qwen_prior_any_status_blocks_repeat':once,'ledger_blob':lr['sha'],'retained_bounds_usd':'10.90','new_permit_issued':False,'real_modal_rpc_calls':0,'ledger_or_release_writes':0,'nn_predictions':0}
(HERE/'unchanged-scope-once-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_assertions_passed':True,'source_paths':96,'worker62_unchanged':True,'used_kev_permit_closed':True,'qwen_ordinary_once_only':True,'paid_calls':0}))
