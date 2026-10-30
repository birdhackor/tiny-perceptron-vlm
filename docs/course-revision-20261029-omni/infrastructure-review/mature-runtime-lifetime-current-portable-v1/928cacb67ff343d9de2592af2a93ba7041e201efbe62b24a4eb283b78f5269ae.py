"""全source與科學範圍唯讀核對；舊permit只驗未消耗、不另簽或消費。"""
import hashlib,json,subprocess,sys,zipfile
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
from scripts.omni_revision import research_execution as r,research_recovery as recovery,core_final
from tiny_perceptron.omni_products.training_runtime import source_binding
from scripts.omni_revision.finance import digest
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
snap=read(HERE/'source-snapshot-manifest.json');names=r.source_paths();expected={x['path']:x['sha256'] for x in snap['files']};assert len(names)==96 and set(names)==set(expected)
for name,h in expected.items():assert sha(ROOT/name)==h
base=snap['head'];different=[]
for name in names:
 old=subprocess.check_output(['git','show',base+':'+name],cwd=ROOT)
 if hashlib.sha256(old).hexdigest()!=expected[name]:different.append(name)
assert set(different)=={'scripts/omni_revision/research_execution.py','tests/test_omni_mature_modal_sdk.py'}
actual=(ROOT/'scripts/omni_revision/research_execution.py').read_bytes();old=subprocess.check_output(['git','show',base+':scripts/omni_revision/research_execution.py'],cwd=ROOT);assert actual.replace(b'mature-research-audit-v3.json',b'mature-research-audit-v2.json')==old
worker=r.worker_files();assert len(worker)==62
with zipfile.ZipFile('/workspace/work/omni-kev-first-preparation-failure-41004743884/input.zip') as z:
 for name in worker:assert z.read(name)==(ROOT/name).read_bytes()
evaluation=core_final.evaluation_sources();assert len(evaluation)==57
for name,h in evaluation.items():assert hashlib.sha256(subprocess.check_output(['git','show',base+':'+name],cwd=ROOT)).hexdigest()==h
recipe=read(ROOT/'docs/omni-products/v1/formal-recipes.json');assert len(recipe['execution_sources_sha256'])==25
for name,h in recipe['execution_sources_sha256'].items():assert sha(ROOT/name)==h
assert source_binding()==recipe['implementation_sha256'] and len(source_binding())==8
permit,permit_sha=recovery.checked_permit();assert permit_sha=='929b40a464f2cedea3021e33db3fcdabc6570552aa361dfc969ce17fd0be2fd8'
lr=read(HERE/'provider-ledger.json')['structuredContent'];state=json.loads(lr['content']);assert lr['sha']=='21c26a416535f9b863340ac1bf2ae2669d389b30' and state['reserved_total_usd']=='6.60'
assert not any('41009648132' in row['attempt_id'] for row in state['attempts'])
oldrow=next(row for row in state['attempts'] if row['attempt_id']==recovery.IDENTITY['attempt']);assert digest(oldrow)==recovery.IDENTITY['row_sha256'];assert digest(oldrow['receipt'])==recovery.IDENTITY['receipt_sha256']
assert not any(recovery.purpose(row).get('recovery_of')==recovery.IDENTITY['attempt'] or recovery.purpose(row).get('preparation_recovery_permit_sha256')==permit_sha for row in state['attempts'])
lineage=recovery.eligibility(state,'kev');assert lineage==recovery.lineage(permit_sha)
result={'schema':'omni-mature-mr08-unchanged-scope-permit-check-v1','created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'same_exact_source_paths96':True,'changed_only':different,'controller_change_only_audit_filename':True,'worker62_exact_original_input_bytes':True,'evaluation57_unchanged':True,'training25_unchanged':True,'implementation8_unchanged':True,'unchanged_permit_sha256':permit_sha,'max_new_attempts':permit['maximum_new_attempts'],'new_bound_usd':permit['new_bound_usd'],'ledger_blob':lr['sha'],'retained_total_usd':state['reserved_total_usd'],'preCAS_run_added_no_row':True,'permit_unconsumed':True,'actual_checked_permit_and_live_predecessor_asset_eligibility':True,'lineage':lineage,'source_approval_extension_required_in_new_audit':True,'real_modal_rpc_calls':0,'ledger_writes':0,'release_writes':0,'dispatches':0,'nn_predictions':0}
(HERE/'unchanged-scope-permit-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_assertions_passed':True,'source_paths':96,'worker_files':62,'unchanged_permit_unused':True,'no_model_or_data_changes':True,'paid_calls':0}))
