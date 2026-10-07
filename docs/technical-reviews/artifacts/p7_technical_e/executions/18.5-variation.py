import torch,json
from pathlib import Path
old=torch.tensor([4.,1.,0.]);new=torch.tensor([8.,2.,0.])
for T in [1.,2.,4.]:print('new',T,(new/T).softmax(0).tolist())
assert torch.equal((new/2).softmax(0),(old/1).softmax(0));assert torch.equal((new/4).softmax(0),(old/2).softmax(0));print('identical scaled pairs True')
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']
for name,t in r['tasks'].items():
 for n,v in t['runs'].items():
  tr=v.get('training',{})
  if tr.get('alpha',0)>0: print('config',name,n,tr['alpha'],tr['temperature'],tr['temperature_squared_applied_once']);assert (tr['alpha'],tr['temperature'],tr['temperature_squared_applied_once'])==(.5,2.,True)
