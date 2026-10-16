"""Late narrow diagnostic-path formatting callback; no model/lesson reruns."""
import ast
import copy
import difflib
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC,datetime
from pathlib import Path,PureWindowsPath
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT))
from scripts.check_technical_reviews import _validate
ART=Path(__file__).parent
TASK='/root/backfill_tech_curriculum'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
prior_path='docs/course-revision-20261016/reviews/technical-order-curriculum-callback-02.json'
prior=json.loads((ROOT/prior_path).read_text());prior_sha=sha(ROOT/prior_path)
assert prior_sha=='fc89613ba1d77c73f145cf590b50f1e6fa21060bf79ba0d38a7f19a8a7041141'
guard='scripts/check_lesson_order.py'
old_sha=next(s['sha256'] for s in prior['sources'] if s['id']=='sOrderCheckCode')
old_run=subprocess.run(['git','show','HEAD:'+guard],cwd=ROOT,capture_output=True,check=True)
assert hashlib.sha256(old_run.stdout).hexdigest()==old_sha
old=old_run.stdout.decode();new=(ROOT/guard).read_text();new_sha=sha(ROOT/guard)
old_ast=ast.parse(old);new_ast=ast.parse(new)
class StripOnlyPathFormatting(ast.NodeTransformer):
    count=0
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='as_posix' and not node.args and not node.keywords:
            assert ast.unparse(node.func.value)=='path.relative_to(root)'
            self.count+=1
            return node.func.value
        return node
strip=StripOnlyPathFormatting();normalized=strip.visit(new_ast)
assert strip.count==1
assert ast.dump(old_ast,include_attributes=False)==ast.dump(normalized,include_attributes=False)
# Narrow PureWindowsPath representation check is cross-platformstdlib only, not WindowsCI.
windows_example=PureWindowsPath('course/chapters/19.md')
assert str(windows_example)=='course\\chapters\\19.md' and windows_example.as_posix()=='course/chapters/19.md'
freeze_path='docs/course-revision-20261016/freeze-19.json'
assert sha(ROOT/freeze_path)==prior['callback']['freeze']['sha256']
f=json.loads((ROOT/freeze_path).read_text());page=next(p for p in f['pages'] if p['page_id']=='curriculum')
assert page['source_sha256']==prior['source_sha256']==sha(ROOT/'docs/curriculum.md')
assert sha(ROOT/page['snapshot'])==page['source_sha256']
checks={}
for s in prior['sources']:
 if s['kind']=='repository_code':
  current=sha(ROOT/s['path']);checks[s['id']]={'path':s['path'],'previous_sha256':s['sha256'],'current_sha256':current,'unchanged':current==s['sha256']}
assert [k for k,v in checks.items() if not v['unchanged']]==['sOrderCheckCode']
artifact_checks={a['id']:sha(ROOT/a['path'])==a['sha256'] for a in prior['artifacts']}
assert all(artifact_checks.values())
for p,h in prior['figure_sha256'].items():assert sha(ROOT/p)==h
cli={'command':'.venv/bin/python scripts/check_lesson_order.py','exit_code':0,'stdout':'{\n  "status": "passed",\n  "chapters": 23,\n  "lessons": 312,\n  "failures": []\n}\n','result':{'status':'passed','chapters':23,'lessons':312,'failures':[]},'environment':{'python':platform.python_version(),'system':platform.system(),'device':'cpu'},'implementation_sha256':new_sha,'scope':'Actual positive CLI invocation performed independently once in this callback before archival. This is Linux, not native WindowsCI.'}
cli_path=ART/'callback-03-cli-positive.json';dump(cli_path,cli)
proof={'reviewer_task':TASK,'current_context':'informed','collection_phase':'continuity','inspected_at':datetime.now(UTC).isoformat(),'scope':'Only the changed guard diagnostic path formatting, existing positive CLI result, currentsource identity and prior evidence pins. No full-page reread, ML, formaltest, timing, notebookkernel, HF or Windowsnative execution.',
 'command':'.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/curriculum/inspect-callback-03.py','environment':{'python':platform.python_version(),'system':platform.system(),'device':'cpu'},'previous_callback':{'path':prior_path,'sha256':prior_sha},
 'guard':{'path':guard,'old_source':'git HEAD blob verified equal to callback02pin','previous_sha256':old_sha,'current_sha256':new_sha,'diff':list(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='guard matched prior pin',tofile='currentguard')),'only_ast_change':'path.relative_to(root)→path.relative_to(root).as_posix() inside rejection-message f-string','as_posix_calls_changed':strip.count,'other_ast_identical':True,'interpretation':'The diagnostic path is rendered with forward slashes. The predicate ids!=expected, numeric ordering checks, returnedfailureconditions and CLIexitstatus logic remain identical. Formatting a message does not change which lessonorders pass or fail.'},
 'pure_windows_path_example':{'native_style_string':str(windows_example),'as_posix':windows_example.as_posix(),'scope':'PythonPureWindowsPath representation check on Linux; does not establish nativeWindowsCI success.'},'positive_cli':cli,'freeze19_unchanged':True,'current_curriculum_page':page,'prior_repository_pins':checks,'prior_artifacts_unchanged':artifact_checks,'figure_unchanged':True,'claim_statements_unchanged':40,'necessary_or_optional_issues':[],'not_done':['nativeWindowsCI','fullpage rereading','modeltraining','formaltest','HFoperations','notebookkernel','timingestimates','broadbenchmark']}
proof_path=ART/'callback-03-inspection.json';dump(proof_path,proof)
r=copy.deepcopy(prior)
r['current_context']='informed'
r['callback_history']=copy.deepcopy(prior.get('callback_history',[]))+[{'path':prior_path,'sha256':prior_sha,'scope':'Previous continuity-phase source-link callback; unchanged.'}]
r['callback']={'kind':'late_implementation_diagnostic_path_informed_callback','collection_phase':'continuity','reviewed_at':datetime.now(UTC).isoformat(),'freeze':prior['callback']['freeze'],'original':prior['callback']['original'],'previous_callback':proof['previous_callback'],'current_source_snapshot':page['snapshot'],'current_source_sha256':page['source_sha256'],'scope':proof['scope'],'claim_disposition':'All40statementspreserved; source sOrderCheckCode renewed only after actual matched-oldblob/currentdiff/AST inspection and actualpositiveCLI. Other sources/artifacts inherited unchanged.','issues_or_optional_suggestions':[]}
r['scope_note']='Currentfreeze19 overall verdict is inherited with this narrow diagnostic-format implementation check. The source page and40claimstatements are unchanged. reviewer_context=fresh describes only the original independent review; current_context=informed applies now. This callback was collected during continuity phase; prior reports and seals remain intact. It does not claim a new full-page technical reading.'
r['verification_boundary']='Personally inspected the matched prior guard blob and current source diff: only rejection-message relativepath.as_posix() plus line wrapping changed. Normalized AST proves all other code identical. Current positive CLI passed on Linux. The PureWindowsPath example is a local standard-library representation check; native WindowsCI remains to be verified by CI. No ML/formaltest/HF/fullpage/notebook/timing rerun.'
for source in r['sources']:
 if source['id']=='sOrderCheckCode':
  source.update(sha256=new_sha,version='Current diagnostic-path .as_posix() revision inspected2026-10-16',inspection_note='Read current guard and actual diff against git HEAD whoseSHAequals callback02pin. Only rejection f-string path adds.as_posix(), withRufflinewrap. AfterremovingthisoneASTcall,ASTisidentical; predicates, sort checks andexitstatus unchanged. Actual currentpositiveCLIpassed onLinux. NativeWindowsCI not run locally.',late_callback_provenance='Callback03renewedthissourcepin afteractualdiff/AST/CLIverification; original earlierguard evidence remains historical.')
for c in r['claims']:
 c['implementation_callback_03']={'previous_report':prior_path,'previous_report_sha256':prior_sha,'disposition':'Unchanged statement and earlier evidence inherited; no full technical rerun.'}
 if c['id']=='c15':
  c['artifact_ids'].extend(['aGuard03Inspection','aGuard03Code','aGuard03CLI'])
  c['evidence'].append({'source_id':'sGuard03Exec','locator':'callback-03-inspection.guard;positive_cli','supports':'Currentguard differs only in diagnosticpathformat; currentpositiveCLIpasses. The ordering predicates used by the prior routeclaim are unchanged.'})
  c['verification']['details']+=' Callback03onlychecks guarddiagnosticformatdiff/ASTandcurrentpositiveCLI; nofullrouteorMLrevalidation asserted.'
r['sources'].append({'id':'sGuard03Exec','kind':'execution','title':'Current guard diagnostic-path check','verified':True,'artifact_id':'aGuard03Inspection'})
r['artifacts'].extend([
 {'id':'aGuard03Inspection','kind':'execution','path':str(proof_path.relative_to(ROOT)),'sha256':sha(proof_path),'description':'Actual matched-oldblob/currentguarddiff andoneASTchange proof; priorpins/currentfreeze preservation.','command':proof['command'],'result':'One.as_posix() call inside errorpathf-string; normalizedremainingASTidentical. Allotherrepositorysources/artifacts andfreeze19 unchanged; positiveCLIpassed onLinux.','environment':proof['environment']},
 {'id':'aGuard03Code','kind':'code','path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(Path(__file__)),'description':'Exact narrowcallback03inspectionandpreservationcode.'},
 {'id':'aGuard03CLI','kind':'execution','path':str(cli_path.relative_to(ROOT)),'sha256':sha(cli_path),'description':'Actual currentguard positiveCLI result onLinux; notWindowsCI.','command':cli['command'],'result':'Passed23chapters/312lessons/failures[];exit0.','environment':cli['environment']}])
r['checks']={
 'factual_accuracy':{'status':'pass','details':'All40curriculumstatementscarriedunchanged. Only a dependencyguard diagnosticpath formatting changed; actualdiff/ASTinspection confirms sorting/rejection logic unchanged, with currentpositiveCLIpass. No full-page technical reread claimed.','claim_ids':[c['id'] for c in r['claims']]},
 'numeric_verification':{'status':'pass','details':'No numericalsource or modelexperiment changed. Earlier arithmetic/denominator verification inherited; currentguardCLI reports23chapters/312lessons. No ML or timing rerun.','claim_ids':prior['checks']['numeric_verification']['claim_ids']},
 'figure_consistency':{'status':'pass','details':'Original figure SHA and currentfreeze19map unchanged; prior personalrender/view evidence inherited. No newrender.','claim_ids':['c31']},
 'source_verification':{'status':'pass','details':'Only prior repositorypin sOrderCheckCode changed. Renewed to actualcurrentSHA after matched-oldblob diff, uniqueas_posixASTchange and positiveCLI. All other pinnedrepositorysources/artifacts unchanged; sBody remains currentfreeze19snapshot. Oldguard-bearingartifacts remain historical and were not rewritten.','claim_ids':[c['id'] for c in r['claims']]},
 'limitations':{'status':'pass','details':'Narrow informed continuity-phase implementation callback. LinuxpositiveCLI andPureWindowsPathformatcheck do not establish nativeWindowsCI success. Sortinglogic unchanged; no newcurriculumcapability claim, fullpageread, ML/formaltest, HF, kernels or estimates.','claim_ids':prior['checks']['limitations']['claim_ids']}}
r['issues']=[]
out=ROOT/'docs/course-revision-20261016/reviews/technical-order-curriculum-callback-03.json'
assert not out.exists(),'Refusing to overwrite callback03.'
assert [c['statement'] for c in r['claims']]==[c['statement'] for c in prior['claims']]
dump(out,r)
actual=json.loads(out.read_text());body=(ROOT/page['snapshot']).read_text()
errors=_validate(ROOT,ROOT/'docs/curriculum.md','curriculum',body,actual,{TASK:['curriculum']},set())
assert not errors,errors
assert sha(ROOT/prior_path)==prior_sha
for x in r['callback_history']:assert sha(ROOT/x['path'])==x['sha256']
origin=r['callback']['original'];assert sha(ROOT/origin['path'])==origin['sha256']
seal={'reviewer_task':TASK,'current_context':'informed','collection_phase':'continuity','completed_at':datetime.now(UTC).isoformat(),'format_validation_errors':errors,'report':{'path':str(out.relative_to(ROOT)),'sha256':sha(out)},'claims':len(actual['claims']),'sources':len(actual['sources']),'artifacts':len(actual['artifacts']),'source_sha256':actual['source_sha256'],'updated_guard_sha256':new_sha,'prior_reports_unchanged':True,'scope':'Narrowdiagnosticpathdiff/currentLinuxpositiveCLI; nativeWindowsCI notclaimed; all40statementsandfreeze19unchanged.'}
dump(ART/'callback-03-validation-seal.json',seal)
print(json.dumps(seal,ensure_ascii=False,indent=2))
