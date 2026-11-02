"""Verify the exact two-file source18 change and preserved raw execution receipts."""
from pathlib import Path
from datetime import datetime,UTC
import ast,difflib,hashlib,json
ROOT=Path('/workspace/work/omni-book-integrated-v1')
BASE='docs/course-revision-20261029-omni/reviews/P5-actual/wave04/'
EV=Path(__file__).parent
BIND=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(name):return {'path':name,'sha256':sha(ROOT/name)}
def doc(r):
 p=ROOT/r['path'];assert sha(p)==r['sha256'],r['path'];BIND.append(r);return json.loads(p.read_bytes())
finding_ref=ref(BASE+'author-windows-sync-source18-finding-v1.json');finding=doc(finding_ref)
before=doc(finding['before_snapshots']);patch_path=ROOT/finding['actual_patch']['path'];assert sha(patch_path)==finding['actual_patch']['sha256']
generated='';files=[]
for row in before['files']:
 old=ROOT/row['snapshot'];new=ROOT/row['current_path'];assert sha(old)==row['sha256']
 generated+=''.join(difflib.unified_diff(old.read_text().splitlines(keepends=True),new.read_text().splitlines(keepends=True),fromfile='before/'+row['current_path'],tofile='current/'+row['current_path']))
 files.append({'path':row['current_path'],'sha256':sha(new),'before_snapshot':{'path':row['snapshot'],'sha256':sha(old)}})
assert generated.encode()==patch_path.read_bytes(),'patch does not describe complete exact before/current bytes'
old_tests=ROOT/before['files'][1]['snapshot'];new_tests=ROOT/'tests/test_windows_student_commands.py'
def funcs(p):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert funcs(old_tests)==funcs(new_tests),'test function/assertion logic changed'
logs=[]
for row in finding['actual_executions']:
 r=row['report'];p=ROOT/r['path'];assert sha(p)==r['sha256'];BIND.append(r)
 logs.append({'reference':r,'tail':p.read_text().splitlines()[-4:],'scope':'Actual author execution log independently read and hashed; not rerun by this reviewer.'})
old_finding_ref={'path':'docs/course-revision-20261029-omni/reviews/author-integration-and-visuals-findings-v1.json','sha256':'6ad8f813282795f4c79ec6e36d8a380faec016b45304db4bfae0c270ef20de01'}
old_finding=doc(old_finding_ref)['issues'][2];assert old_finding['id']=='AUTHOR-INTEGRATION-WINDOWS-SYNC'
runner_ref=ref('docs/technical-reviews/artifacts/omni-p6-tools/fresh-tools-original-v1/run_cpu_contract_tests.py');runner=(ROOT/runner_ref['path']).read_text();assert "r['path'].startswith('tests/')" in runner
suite_ref=ref('docs/technical-reviews/artifacts/omni-p6-tools/fresh-tools-original-v1/cpu-contract-tests-v1.log');suite=(ROOT/suite_ref['path']).read_text();assert '"tests/test_windows_student_commands.py"' in suite and '908 passed, 1 skipped, 3 deselected' in suite
collect_ref=ref('docs/technical-reviews/artifacts/omni-p6-tools/fresh-tools-current-v1/windows-collection-only-v1.log');assert '192 tests collected' in (ROOT/collect_ref['path']).read_text()
result={'schema_version':1,'reviewer_task':'/root/omni_p6_technical_tools','created_at':datetime.now(UTC).isoformat(),'verdict':'pass','scope':'Exact current two Windows synchronization files and the prior root-owned integration issue, not a Windows PowerShell execution or generic whole-course review.','checked_files':files,'source18_finding':finding_ref,'exact_full_patch':finding['actual_patch'],'before_after_full_patch_reconstructed':True,'all_test_function_AST_bodies_unchanged':True,'personally_checked_changes':['Current most-direct-python anchor matches the source18 convention; quotation/noncoverage text names the same source18 clauses.','Fixed SFT download/run prerequisites join the existing explicit CUDA Skip-Page list before encoders/projector/vqa; no claim those GPU recipes ran.','Existing baseline style inference is now before DPO update and preferred inference remains after the update.','Module-level documentation/PREFIXES changes preserve every test function/assertion AST; no parser or assertion weakening.'],'own_executed_evidence':{'runner':runner_ref,'908_case_actual_log':suite_ref,'Windows_module_collection_count':collect_ref,'interpretation':'The original independent908-case offline CPU run actually included all192 current Windows-command integrity cases; only the unrelated three named optimizer-update cases were deselected. Collection-only now confirms192 without rerunning training.'},'preserved_author_logs_independently_inspected':logs,'old_root_issue':{'report':old_finding_ref,'json_pointer':'/issues/2','id':old_finding['id'],'current_scope_reason':'Current README/training command/quote integrity remains covered by the intact192 case module, previously executed in my independent908-case run; the new source18 correction adds the missing prerequisite/order/anchor synchronization. Earlier source12 pass is retained as its own version, not relabeled source18.'},'resolved_issue_ids':['AUTHOR-SOURCE18-WINDOWS-SYNC','AUTHOR-INTEGRATION-WINDOWS-SYNC'],'additional_bound_inputs':BIND,'limits':['No new package, Windows OS/PowerShell runtime, GPU recipe, workflow or publication was invoked.','The author2755 full-suite after-log was hash/tail-inspected only; it is not my fresh full-suite execution.','Exact source18 current/native structural checks are verified; Windows runtime remains for the genuine committed PR.']}
p=EV/'windows-source18-independent-verification-v1.json'
with p.open('x') as s:s.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'verdict':result['verdict'],'full_patch_reconstructed':True,'all_test_function_AST_bodies_unchanged':True,'resolved_issue_ids':result['resolved_issue_ids']},ensure_ascii=False,indent=2))
