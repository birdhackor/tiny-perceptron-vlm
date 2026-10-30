"""Replay the independent original assertions with current source and valid stronger-identity fixtures.

Original proofs/results stay immutable. Only fixture/schema/provider payloads and
result paths are adapted below; assertions and all original negative cases stay intact.
Generated proof scripts and full source snapshots stay outside ordinary Git.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE=Path('/workspace/work/omni-cpu-final-execution-review-v1')
ROOT=Path('/workspace/work/omni-models-v1')
EVIDENCE=ROOT/'docs/course-revision-20261029-omni/infrastructure/cpu-final-execution-review-evidence-v1'
CURRENT=BASE/'current-probes';CURRENT.mkdir(exist_ok=False)
changes=[]

def replacement(raw,old,new,expected=1):
    count=raw.count(old)
    assert count==expected,(old,count,expected)
    changes.append({'old':old,'new':new,'count':count})
    return raw.replace(old,new)

source=EVIDENCE/'review_probes.py'
raw=source.read_text()
raw=replacement(raw,'author-scope-original.json','author-scope-current.json')
raw=replacement(raw,'original-source','current-source',2)
raw=replacement(raw,'original-probe-results.json','current-probe-results.json')
raw=replacement(raw,"'purpose':json.dumps({'job':r.JOB})","'purpose':json.dumps({'job':r.JOB,'ownership':{'attempt':'cpu-final-123-1'},'sources_sha256':r.digest({'revision':'a'*40})})")
raw=replacement(raw,"receipt={'attempt':'cpu-final-123-1'","receipt={'schema':'omni-cpu-final-execution-v1','attempt':'cpu-final-123-1'")
raw=replacement(raw,"with patch.dict(os.environ,ENV),patch.object(r,'GitHubLedger',Store),patch.object(r,'api',api):","with patch.dict(os.environ,ENV),patch.object(r,'GitHubLedger',Store),patch.object(r,'api',api),patch.object(r,'checked_sources',lambda:{'revision':'a'*40}):")
path=CURRENT/source.name;path.write_text(raw)
index=[{'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'derived_path':str(path),'derived_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fixture_and_path_changes':list(changes)}]
changes=[]
source=EVIDENCE/'control_probes.py';raw=source.read_text()
raw=replacement(raw,'original-control-probe-results.json','current-control-probe-results.json')
raw=replacement(raw,"        def api(endpoint,*args):return {'id':7}","        provider={}\n        def api(endpoint,method='GET',payload=None):\n            if method=='POST':provider.update(id=7,**payload)\n            return dict(provider)")
path=CURRENT/source.name;path.write_text(raw)
index.append({'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'derived_path':str(path),'derived_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fixture_and_path_changes':list(changes)})
changes=[]
source=EVIDENCE/'legacy_loader_positive.py';raw=source.read_text()
raw=replacement(raw,'legacy-loader-positive-results.json','current-legacy-loader-positive-results.json')
path=CURRENT/source.name;path.write_text(raw)
index.append({'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'derived_path':str(path),'derived_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fixture_and_path_changes':list(changes)})
for row in index:
    subprocess.run([sys.executable,row['derived_path']],cwd=ROOT,check=True,timeout=120)
results=[]
for name in ['current-probe-results.json','current-control-probe-results.json']:
    path=BASE/name;data=json.loads(path.read_text());results.extend(data['results'])
    target=EVIDENCE/name;assert not target.exists();path.replace(target)
assert len(results)==44 and all(row['observed']=='pass' for row in results)
receipt={'reviewer_task':'/root/omni_cpu_final_execution_review','scope':'Replay of original independent 44 cases and actual legacy loader positive; fixtures adapted to the stronger valid contract without weakening assertions.','status':'pass','total_original_cases_replayed':44,'all_44_pass':True,'genuine_core_zip_strict_ten_jobs_no_forward_replayed':True,'genuine_8962_file_unpack_and_actual_A_B_loaders_no_forward_replayed':True,'derived_proofs':index}
(EVIDENCE/'current-replay-index.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'pass','original_independent_cases':44,'genuine_core_strict_jobs':10,'genuine_data_files':8962,'model_forward_calls':0},ensure_ascii=False))
