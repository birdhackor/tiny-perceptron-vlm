"""Private, labeled source-freeze fixture; readonly Git API responses only are substituted."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'current-source';probe=BASE/'probes/current-source-closure';probe.mkdir(parents=True,exist_ok=False);root=probe/'fixture-root';shutil.copytree(SOURCE,root)
frozen=root/'docs/omni-products/v1';recipe=json.loads((frozen/'formal-recipes.json').read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
required={str((frozen/n).relative_to(root)):sha(frozen/n) for n in ('data-definition.json','metrics.json','formal-recipes.json')};required.update(recipe['execution_sources_sha256']);required.update({'tiny_perceptron/omni_products/'+n:d for n,d in recipe['implementation_sha256'].items()})
(root/'docs/course-revision-20261029-omni/infrastructure/formal-core-audit-v1.json').write_text(json.dumps({'status':'pass_for_reviewed_scope','reviewer_task':'/root/review_fixture_only','reviewed_files_sha256':required,'test_fixture':True,'scope':'Not a formal training permission or real audit.'}))
p=root/'tiny_perceptron/omni_products/reliability.py';raw=p.read_bytes();p.write_bytes(raw.replace(b'TEMPERATURES = tuple(0.25 * 2 ** (k / 8) for k in range(33))',b'TEMPERATURES = (1.75,)  # isolated review-only behavior drift'))
assert p.read_bytes()!=raw
child=probe/'child.py';child.write_text('''import json,os,sys
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from scripts.omni_revision import cpu_execution as cpu,formal_local as fl,core_selection as cs
from tiny_perceptron.omni_products import reliability
os.environ['GITHUB_SHA']='1'*40
real=cpu.subprocess.check_output
def git(arguments,**kwargs):
 if arguments==['git','rev-parse','HEAD']:return ('1'*40+'\\n').encode()
 if arguments[:2]==['git','show']:return (root/arguments[2].split(':',1)[1]).read_bytes()
 raise AssertionError(arguments)
with patch.object(cpu.subprocess,'check_output',side_effect=git):
 bound=cpu.checked_sources()
families=[{'family':str(i),'tasks':{k:{'logits':[0.,1.],'ids':['a','b'],'gold':'a'} for k in ('choice','info','rubric')}} for i in range(80)]
result={'actual_checked_sources_accepted':True,'measurement_source_listed':str(reliability.__file__).removeprefix(str(root)+'/') in bound['files'],'loaded_reliability_source':str(reliability.__file__),'loaded_grid':list(reliability.TEMPERATURES),'actual_selected_temperature':cs.select_temperature(families)['temperature'],'git_fixture':'All readonly HEAD/show responses describe immutable fixture bytes; no git commit or remote mutation.','files_checked':len(bound['files'])}
print(json.dumps(result))
''')
r=subprocess.run([sys.executable,str(child),str(root)],capture_output=True,text=True,timeout=30);(probe/'child.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(scope='Fresh process in isolated source tree, synthetic 80-family logits, no model training or final prediction.',declared_reliability_sha256=json.loads((frozen/'metrics.json').read_bytes())['measurement_sources_sha256']['tiny_perceptron/omni_products/reliability.py'],actual_reliability_sha256=sha(p),frozen_grid_size=33,unbound_legacy_finance='scripts/selftrained/finance.py' not in required,formal_success_claimed=False)
(BASE/'current-source-closure-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
