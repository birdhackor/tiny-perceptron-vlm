from pathlib import Path
import sys,json
import torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.course_experiments.modalities import _Contrastive
from tiny_perceptron.multimodal import scene
torch.set_num_threads(1);torch.manual_seed(51)
m=_Contrastive();old={n:p.detach().clone() for n,p in m.named_parameters()}
s=m.scores(torch.stack([scene('red','circle'),scene('blue','square')]),['red circle','blue square'])
rows=[]
for i,j in [(0,0),(0,1),(1,0),(1,1)]:
 v,t=torch.autograd.grad(s[i,j],(m.vision.projection.weight,m.text.weight),retain_graph=True)
 rows.append({'score_entry':[i,j],'vision_projection_jacobian_norm':float(v.norm()),'text_embedding_jacobian_norm':float(t.norm())})
assert all(r['vision_projection_jacobian_norm']>0 and r['text_embedding_jacobian_norm']>0 for r in rows)
assert all(torch.equal(old[n],p.detach()) for n,p in m.named_parameters())
r={'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-10_10-independent/execution/check_shared_parameters.py','torch':torch.__version__,'python':sys.version,'device':'cpu','scores':s.detach().tolist(),'per_score_parameter_derivatives':rows,'parameters_unchanged':True,'scope':'Derivative checks on four scores show the same parameter tensors influence diagonal and off-diagonal entries; no optimizer, update, training or evaluation of existing checkpoints.'}
p=ROOT/'docs/technical-reviews/artifacts/phase4-10_10-independent/execution/shared-parameters-result.json';p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
