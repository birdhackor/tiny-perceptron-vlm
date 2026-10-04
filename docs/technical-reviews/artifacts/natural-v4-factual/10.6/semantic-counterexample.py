from pathlib import Path
import json,sys,torch
from torch import nn
# Both endpoints have width 2, but a width-compatible projector can reverse meanings.
# Hypothetical code: red = [1,0], blue = [0,1], score coordinate 0 = red.
x=torch.tensor([[1.,0.],[0.,1.]])
identity=nn.Linear(2,2,bias=False)
swap=nn.Linear(2,2,bias=False)
with torch.no_grad():
 identity.weight.copy_(torch.eye(2))
 swap.weight.copy_(torch.tensor([[0.,1.],[1.,0.]]))
a=identity(x); b=swap(x)
assert a.tolist()==[[1.,0.],[0.,1.]]
assert b.tolist()==[[0.,1.],[1.,0.]]
assert a.shape==b.shape
assert not torch.equal(a,b)
result={'prediction':'Both outputs preserve width 2; swap reverses the hypothetical red/blue score coordinates.', 'input':x.tolist(),'identity_output':a.tolist(),'swap_output':b.tolist(),'shape_equal':list(a.shape)==list(b.shape),'interpretation':'An explicit counterexample disproves any implication that matching feature width alone establishes shared meanings. The hypothetical coordinate meanings do not describe learned dimensions of an actual vision or language model.', 'environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu'},'limits':'No training or measured model quality; a mathematical counterexample with deliberately specified coordinates.'}
Path(__file__).with_name('semantic-counterexample-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
