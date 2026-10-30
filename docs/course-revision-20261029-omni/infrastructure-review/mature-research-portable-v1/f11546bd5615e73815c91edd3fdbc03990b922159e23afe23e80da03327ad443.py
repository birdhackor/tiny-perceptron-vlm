"""Compile the unchanged test-harness replacement under a real Windows-form path, without IO."""
import hashlib,json,sys
from pathlib import Path,PureWindowsPath
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'current-source'))
from scripts.omni_revision.staged_execution import staged_bootstrap
payload=b'explicit compile-only fixture';code=staged_bootstrap(payload);root=PureWindowsPath(r'C:\Users\reviewer\AppData\Local\Temp\pytest-fixture')
cases=[]
for mode in ('current_test_str','private_as_posix_control'):
 fixed=code
 for old,name in (('/tmp/omni-input.zip','input.zip'),('/tmp/omni-input.sha256','input.sha256'),('/tmp/omni-job','root')):
  value=str(root/name) if mode=='current_test_str' else (root/name).as_posix();fixed=fixed.replace(old,value)
 try:compile(fixed,'review-test-harness-'+mode,'exec');accepted=True;error=None
 except SyntaxError as e:accepted=False;error={'type':type(e).__name__,'reason':e.msg,'line':e.lineno}
 cases.append({'case':mode,'compiled':accepted,'error':error,'fixture_code_sha256':hashlib.sha256(fixed.encode()).hexdigest()})
assert not cases[0]['compiled'] and cases[1]['compiled']
record={'scope':'Unmodified production bootstrap and exact test str() replacement technique, simulated PureWindowsPath. compile only; no worker, filesystem IO, model, provider or final calls.','cases':cases,'product_bootstrap_unchanged':True,'product_workflow_host':'ubuntu-latest'}
p=BASE/'current-bootstrap-windows-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
