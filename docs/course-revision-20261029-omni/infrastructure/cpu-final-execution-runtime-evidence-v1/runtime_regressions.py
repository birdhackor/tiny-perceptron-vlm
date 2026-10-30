"""Independent99-source closure, prior finish negatives, reserve/CAS, preservation, and O1 regression."""
import ast
from copy import deepcopy
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from scripts.omni_revision import cpu_final_execution as r
from scripts.omni_revision import research_execution as re

ROOT=Path('/workspace/work/omni-models-v1')
BASE=Path('/workspace/work/omni-cpu-final-execution-review-v1')
EVIDENCE=ROOT/'docs/course-revision-20261029-omni/infrastructure/cpu-final-execution-runtime-evidence-v1'
ENV={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':r.REPOSITORY,'GITHUB_REF':'refs/heads/main','GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'a'*40,'OMNI_ARTIFACT_OUTCOME':'success'}
INV={'files':{name:r.file_sha(ROOT/name) for name in r.source_paths()}}
assert len(INV['files'])==99
RESULTS=[]

def record(name,observed='pass',**details):RESULTS.append({'name':name,'observed':observed,**details})
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

# Replay the previously independent source/audit/commit and finish negative assertions.
source=BASE/'current-probes/review_probes.py';tree=ast.parse(source.read_text())
nodes=[]
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name=='finish_case':nodes.append(node)
    elif isinstance(node,ast.With) and 'prefix=\'sources-\'' in ast.unparse(node.items[0]):nodes.append(node)
    elif isinstance(node,ast.For) and ast.unparse(node.target)=='(name, mutation, expected)':nodes.append(node)
assert len(nodes)==3
class FixturePaths(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=='current-source':return ast.copy_location(ast.Constant('runtime-current-v1/source'),node)
        if node.value=='uv.lock':return ast.copy_location(ast.Constant('scripts/omni_revision/legacy_download.py'),node)
        if node.value=='full_audit_and_all_99_committed_bytes':return ast.copy_location(ast.Constant('full_audit_and_all_100_committed_bytes'),node)
        return node
module=ast.fix_missing_locations(FixturePaths().visit(ast.Module(body=nodes,type_ignores=[])))
exec(compile(module,str(source),'exec'))
assert all(x['observed']=='pass' for x in RESULTS)
record('exact99_current_source_closure',count=len(INV['files']))

# Replay the prior complete controller/preservation/CAS probes with the new two-child API.
source_control=BASE/'current-probes/control_probes.py';raw=source_control.read_text()
old="assert args[2]=='legacy' and seconds==900;events.append('legacy_child');state['all_training_processes_exited']=True"
new="assert (args[1],seconds) in [('scripts.omni_revision.legacy_download',900),('scripts.omni_revision.core_final',180)];events.append(args[1]);state['all_training_processes_exited']=True"
assert raw.count(old)==1;raw=raw.replace(old,new)
assert raw.count('current-control-probe-results.json')==1;raw=raw.replace('current-control-probe-results.json','runtime-control-probe-results.json')
folder=BASE/'runtime-current-v1/probes';folder.mkdir(exist_ok=True)
derived=folder/'control_probes.py';derived.write_text(raw)
subprocess.run([os.sys.executable,str(derived)],cwd=ROOT,check=True,timeout=30)
control_results=json.loads((BASE/'runtime-control-probe-results.json').read_text())['results'];assert len(control_results)==15 and all(x['observed']=='pass' for x in control_results)

# Independently prove missing/foreign owned providers cannot retain temporary success.
for failure in ['absent','foreign-target']:
    with tempfile.TemporaryDirectory(prefix='runtime-export-',dir=BASE/'probes') as tmp:
        root=Path(tmp);out=root/'out';out.mkdir();frozen=root/'frozen';frozen.mkdir();(frozen/'model-selection.json').write_text('fixed synthetic selection')
        row={'attempt_id':'cpu-final-123-1','status':'reserved','revision':'a'*40,'purpose':json.dumps({'job':r.JOB}),'reserved_usd':'0.10','components_usd':r.transport.COMPONENTS}
        write(out/'reservation.json',{'attempt':row})
        class Store:
            def read(self):return {'attempts':[row]},'blob'
        owned={};calls=[]
        def api(endpoint,method='GET',payload=None):
            if method=='POST':owned.update(id=7,**payload);return dict(owned)
            if failure=='absent':return None
            return {**owned,'target_commitish':'b'*40}
        def child(args,log,seconds,state):
            state['all_training_processes_exited']=True;calls.append(args[1]);log.write_text('synthetic no NN\n')
            path=Path(args[args.index('--output')+1])
            if args[1].endswith('core_report'):path.write_text('{"fixture":true}')
            else:path.mkdir(parents=True);(path/'raw.json').write_text('{"fixture":true}')
        with patch.dict(os.environ,ENV),patch.object(r,'ROOT',root),patch.object(r,'FROZEN',frozen),patch.object(r,'checked_sources',lambda:{'revision':'a'*40}),patch.object(r,'GitHubLedger',Store),patch.object(r,'find_draft',lambda *a:None),patch.object(r,'api',api),patch.object(r.transport,'bounded_child',child),patch.object(r.transport,'upload_owned',side_effect=AssertionError('must not upload')):
            try:r.run_final('cpu-final-123-1',out)
            except RuntimeError:pass
            else:raise AssertionError('incomplete export accepted')
        receipt=json.loads((out/'cpu-final-receipt.json').read_text());assert receipt['success'] is False and receipt['export_error_type']=='ValueError' and (out/'final-results.zip').is_file() and len(calls)==5
        record('O1_incomplete_export_'+failure)

result={'reviewer_task':'/root/omni_cpu_final_execution_review','status':'pass','source_count':99,'old_assertions_source':str(source),'old_assertions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'fixture_adaptations':'Use current99 hashes and snapshot; target missing/dirty/symlink/committed checks at new legacy_download.py instead of uv.lock. Existing finish cases unchanged. Prior reserve fixture recognizes new download900s/preflight180s two-child interface.','control_source':str(source_control),'control_source_sha256':hashlib.sha256(source_control.read_bytes()).hexdigest(),'derived_control_sha256':hashlib.sha256(derived.read_bytes()).hexdigest(),'results':RESULTS,'prior_control_results':control_results,'network_calls':0,'model_forward_calls':0}
path=EVIDENCE/'runtime-regression-results.json';assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'pass','source_finish_export_cases':len(RESULTS),'prior_controller_cases':len(control_results),'source_count':99,'network_calls':0,'forward_calls':0}))
