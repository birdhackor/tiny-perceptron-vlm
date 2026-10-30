import json,os,sys
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]);mode=sys.argv[2];sys.path.insert(0,str(root))
from scripts.omni_revision import cpu_execution as cpu,formal_local as fl,core_selection as cs
from tiny_perceptron.omni_products import reliability
os.environ['GITHUB_SHA']='1'*40 if mode!='head_mismatch' else '2'*40
def git(arguments,**kwargs):
 if arguments==['git','rev-parse','HEAD']:return ('1'*40+'\n').encode()
 if arguments[:2]==['git','show']:
  name=arguments[2].split(':',1)[1];raw=(root/name).read_bytes()
  return raw+b'\n' if mode=='git_bytes_mismatch' and name=='scripts/omni_revision/cpu_execution.py' else raw
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
