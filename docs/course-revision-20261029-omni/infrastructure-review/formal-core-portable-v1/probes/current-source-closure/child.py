import json,os,sys
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from scripts.omni_revision import cpu_execution as cpu,formal_local as fl,core_selection as cs
from tiny_perceptron.omni_products import reliability
os.environ['GITHUB_SHA']='1'*40
real=cpu.subprocess.check_output
def git(arguments,**kwargs):
 if arguments==['git','rev-parse','HEAD']:return ('1'*40+'\n').encode()
 if arguments[:2]==['git','show']:return (root/arguments[2].split(':',1)[1]).read_bytes()
 raise AssertionError(arguments)
with patch.object(cpu.subprocess,'check_output',side_effect=git):
 bound=cpu.checked_sources()
families=[{'family':str(i),'tasks':{k:{'logits':[0.,1.],'ids':['a','b'],'gold':'a'} for k in ('choice','info','rubric')}} for i in range(80)]
result={'actual_checked_sources_accepted':True,'measurement_source_listed':str(reliability.__file__).removeprefix(str(root)+'/') in bound['files'],'loaded_reliability_source':str(reliability.__file__),'loaded_grid':list(reliability.TEMPERATURES),'actual_selected_temperature':cs.select_temperature(families)['temperature'],'git_fixture':'All readonly HEAD/show responses describe immutable fixture bytes; no git commit or remote mutation.','files_checked':len(bound['files'])}
print(json.dumps(result))
