"""Current-v2 exact-source closure negative controls in fresh isolated processes."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'current-v2-source';probe=BASE/'probes/current-v2-source-closure';probe.mkdir(parents=True,exist_ok=False)
child=probe/'child.py';child.write_text('''import json,os,sys
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]);mode=sys.argv[2];sys.path.insert(0,str(root))
from scripts.omni_revision import cpu_execution as cpu,formal_local as fl,core_selection as cs
from tiny_perceptron.omni_products import reliability
os.environ['GITHUB_SHA']='1'*40 if mode!='head_mismatch' else '2'*40
def git(arguments,**kwargs):
 if arguments==['git','rev-parse','HEAD']:return ('1'*40+'\\n').encode()
 if arguments[:2]==['git','show']:
  name=arguments[2].split(':',1)[1];raw=(root/name).read_bytes()
  return raw+b'\\n' if mode=='git_bytes_mismatch' and name=='scripts/omni_revision/cpu_execution.py' else raw
 raise AssertionError(arguments)
value={'case':mode,'fresh_process':True}
try:
 with patch.object(cpu.subprocess,'check_output',side_effect=git):bound=cpu.checked_sources()
 value.update(accepted=True,files_checked=len(bound['files']),reliability_source_listed='tiny_perceptron/omni_products/reliability.py' in bound['files'],legacy_finance_source_listed='scripts/selftrained/finance.py' in bound['files'])
 if mode=='stable':
  rows=[{'family':str(i),'tasks':{k:{'logits':[0.,1.],'ids':['a','b'],'gold':'a'} for k in ('choice','info','rubric')}} for i in range(80)]
  value.update(loaded_grid_size=len(reliability.TEMPERATURES),selected_temperature=cs.select_temperature(rows)['temperature'])
except Exception as error:value.update(accepted=False,error_type=type(error).__name__,reason=str(error))
print(json.dumps(value,ensure_ascii=False))
''')
cases=[]
for mode in ('stable','reliability_drift','legacy_finance_drift','package_init_drift','missing_measurement_audit_binding','git_bytes_mismatch','head_mismatch'):
 root=probe/mode;shutil.copytree(SOURCE,root);frozen=root/'docs/omni-products/v1';recipe=json.loads((frozen/'formal-recipes.json').read_bytes());metrics=json.loads((frozen/'metrics.json').read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 required={str((frozen/n).relative_to(root)):sha(frozen/n) for n in ('data-definition.json','metrics.json','formal-recipes.json')};required.update(recipe['execution_sources_sha256']);required.update(metrics['measurement_sources_sha256']);required.update({'tiny_perceptron/omni_products/'+n:d for n,d in recipe['implementation_sha256'].items()})
 if mode=='missing_measurement_audit_binding':required.pop('tiny_perceptron/omni_products/reliability.py')
 (root/'docs/course-revision-20261029-omni/infrastructure/formal-core-audit-v1.json').write_text(json.dumps({'status':'pass_for_reviewed_scope','reviewer_task':'/root/review_fixture_only','reviewed_files_sha256':required,'test_fixture':True,'scope':'Private gate fixture only, never formal permission.'}))
 name={'reliability_drift':'tiny_perceptron/omni_products/reliability.py','legacy_finance_drift':'scripts/selftrained/finance.py','package_init_drift':'tiny_perceptron/omni_products/__init__.py'}.get(mode)
 if name:
  path=root/name;raw=path.read_bytes();path.write_bytes(raw.replace(b'TEMPERATURES = tuple(0.25 * 2 ** (k / 8) for k in range(33))',b'TEMPERATURES = (1.75,)  # private review drift') if mode=='reliability_drift' else raw+b'\n# isolated review-only source drift\n');assert path.read_bytes()!=raw
 r=subprocess.run([sys.executable,str(child),str(root),mode],capture_output=True,text=True,timeout=30);(probe/(mode+'.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr;value=json.loads(r.stdout);cases.append(value)
assert cases[0]['accepted'] is True
assert all(x['accepted'] is False for x in cases[1:]),cases
result={'scope':'Seven fresh processes, copied sources and labeled private audit/Git HEAD/show fixtures. No commit, remote mutation, formal training, paid or final calls.','cases':cases,'all_required_negative_controls_rejected':True}
(BASE/'current-v2-source-closure-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
