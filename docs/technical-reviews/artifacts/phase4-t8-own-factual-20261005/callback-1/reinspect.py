"""Original T.8 reviewer's real revision callback: source/version assertions and original aggregation execution."""
from pathlib import Path
from datetime import datetime,UTC
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).parent.parent;OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
import torch
started=datetime.now(UTC).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
initial=BASE/'T.8.initial-revise.json';assert sha(initial)=='003e715fac0a3b39a43d2d1672e37b0c2877867ba5901c2c053f315a9ff157ec';report=json.loads(initial.read_text())
spec=importlib.util.spec_from_file_location('sf',ROOT/'docs/review-tools/section_facts.py');sf=importlib.util.module_from_spec(spec);spec.loader.exec_module(sf)
body,_,first=sf.original_section(ROOT/'course/training.md','T.8');assert hashlib.sha256(body).hexdigest()=='09a876bc8eb23ae2bd3f7cfc8ecb48cfcc74444424f38446a6f88880357f77a6'
(OUT/'current-section.md').write_bytes(body)
old=(BASE/'section.md').read_bytes()
# All substantive changes are confined to the final paragraph. All four bash fences are unchanged.
old_start=old.index('它只測固定Q/K/V'.encode());new_start=body.index('它只測固定Q/K/V'.encode());assert old[:old_start]==body[:new_start]
old_f=sf.fences(old,354);new_f=sf.fences(body,first);assert len(old_f)==len(new_f)==4
assert all(a['raw']==b['raw'] for a,b in zip(old_f,new_f))
version_checks=[]
for artifact in report['artifacts']:
 p=ROOT/artifact['path'];assert sha(p)==artifact['sha256'];version_checks.append({'id':artifact['id'],'path':artifact['path'],'sha256':artifact['sha256'],'unchanged':True})
for source in report['sources']:
 if source['kind']=='repository_code':assert sha(ROOT/source['path'])==source['sha256']
frozen=json.loads((BASE/'frozen-input-manifest.json').read_text())
for file,m in frozen['files'].items():
 if file=='course/training.md':continue # Original whole-file bytes are historical; T.5/T.6/T.8 changes do not invalidate this retained input.
 assert sha(ROOT/file)==m['sha256'],file
for name in ['efficiency','precision','flash_probe']:
 p=ROOT/f'docs/course-experiments/results/{name}.json';assert sha(p)==sha(BASE/f'raw-results/{name}.json')
context={}
for file,ids in [('course/training.md',['T.1','T.4']),('course/chapters/05.md',['5.9','5.10']),('course/chapters/16.md',['16.1','16.7','16.8','16.9'])]:
 for lesson in ids:
  raw,_,_=sf.original_section(ROOT/file,lesson);assert raw==(BASE/f'read-scope/{lesson}.md').read_bytes();context[lesson]={'raw_sha256':hashlib.sha256(raw).hexdigest(),'unchanged':True}
# Read original current methods via AST; all relevant source bytes still match the personally inspected original.
src=(ROOT/'scripts/course_experiments/architecture.py').read_text();tree=ast.parse(src);fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_flash_probe')
lines=src.splitlines(keepends=True);(OUT/'original-aggregation-1360-1390.txt').write_text(''.join(lines[1359:1390]))
module=ast.Module(body=copy.deepcopy([n for n in fun.body if 1373<=n.lineno<=1388]),type_ignores=[]);ast.fix_missing_locations(module);code=compile(module,'original-run_flash_probe-aggregation-1373-1388','exec')
cases=[]
plans=[('all_completed',{'fp16':{'status':'completed'},'bf16':{'status':'completed'}},['fp16','bf16']),('partial_unsupported',{'fp16':{'status':'completed'},'bf16':{'status':'unsupported'}},['fp16']),('partial_runtime_error',{'fp16':{'status':'completed'},'bf16':{'status':'runtime_error'}},['fp16']),('partial_backend_unverified',{'fp16':{'status':'completed'},'bf16':{'status':'backend_unverified'}},['fp16']),('supported_numeric_mismatch',{'fp16':{'status':'completed'},'bf16':{'status':'numerical_mismatch'}},['fp16','bf16']),('all_unsupported',{'fp16':{'status':'unsupported'},'bf16':{'status':'unsupported'}},[]),('budget_exhausted',{'fp16':{'status':'completed'},'bf16':{'status':'budget_exhausted'}},['fp16'])]
for label,routes,supported in plans:
 r={'routes':routes,'supported_routes':supported};exec(code,{'report':r});assert r['verification_passed']==(bool(supported) and all(routes[k]['status']=='completed' for k in supported));cases.append({'case':label,'report':r})
assert cases[2]['report']['verification_passed'] and not cases[2]['report']['schedule_completed'] and cases[2]['report']['status']=='failed_verification'
assert cases[3]['report']['verification_passed'] and not cases[3]['report']['schedule_completed']
assert cases[1]['report']['schedule_completed'] and cases[1]['report']['status']=='completed_with_unsupported_routes'
assert not cases[5]['report']['verification_passed'] and cases[5]['report']['schedule_completed'] and cases[5]['report']['status']=='unsupported'
# Confirm the published report's exact pointer versions and original aggregation independently.
raw=json.loads((BASE/'raw-results/flash_probe.json').read_text())['results'];r={'routes':copy.deepcopy(raw['routes']),'supported_routes':list(raw['supported_routes'])};exec(code,{'report':r})
assert all(raw[k]==r[k] for k in ['verification_passed','schedule_completed','status'])
new_text=body.decode();assert '`supported_routes`至少有一條已核驗Flash後端的路線，而且這個集合內的路線都已完成；其他路線仍可能失敗' in new_text
assert '`results.schedule_completed`和`results.status`判讀整次探針是否完成，以及是否有不支援的路線' in new_text
result={'reviewer_task':'/root/phase4_factual_coordinator/factual_t_8','started_at_utc':started,'completed_at_utc':datetime.now(UTC).isoformat(),'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git':str(torch.version.git_version),'cuda_build':str(torch.version.cuda),'device':'cpu'},'initial_report_sha256':sha(initial),'current_section_sha256':sha(OUT/'current-section.md'),'original_historical_whole_input':{'path':str((BASE/'frozen/course/training.md').relative_to(ROOT)),'sha256':sha(BASE/'frozen/course/training.md'),'bytes':(BASE/'frozen/course/training.md').stat().st_size,'meaning':'True retained initial whole-file input, not a current whole chapter hash.'},'read_scope':'Personally read entire current T.8, original aggregation1360-1390, own original AST countercases and all13unchanged-claim evidence versions. T.5/T.6 are not necessary dependencies and were not read for this callback. Original authority support was retained/version-checked, not replaced by coordinator administrative text.','unchanged_context':context,'original_evidence_version_checks':version_checks,'unchanged_fences':4,'figures':0,'original_ast_reexecuted_cases':cases,'published_raw_aggregation_recomputed':{k:r[k] for k in ['verification_passed','schedule_completed','status']},'finding':'PASS: New wording exactly restricts verification_passed to a nonempty set of backend-verified supported_routes all completed, permits failure outside that subset, and separately requires per-route status/backend/tolerance plus schedule_completed and results.status to interpret complete/unsupported outcomes. No remaining substantive issue in T.8.','execution_scope':'No training, GPU, dataset/model download, or new quality score; only original aggregation statements and byte/version assertions executed.'}
(OUT/'reinspection-results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
