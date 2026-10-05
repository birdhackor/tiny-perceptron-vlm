"""Constructive check of 'may preserve detail', not a learned-model result."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.multimodal import patchify
assert torch.version.cuda is None
base=torch.full((1,3,16,16),.5)
detail=base.clone()
detail[:,:,0:2,0:2]=1
detail[:,:,0:2,2:4]=0
results=[]
for p in [4,2]:
    # A channelwise average is an allowed linear entry map, fixed here by hand.
    a=patchify(base,p).reshape(1,-1,3,p*p).mean(-1)
    b=patchify(detail,p).reshape(1,-1,3,p*p).mean(-1)
    count=int((a!=b).any(-1).sum())
    assert count==({4:0,2:2}[p])
    results.append({'patch_size':p,'visual_positions':a.shape[1],'changed_feature_positions':count,'first_patch_rgb_mean_before':a[0,0].tolist(),'first_patch_rgb_mean_after':b[0,0].tolist()})
print(json.dumps({'environment':{'torch':str(torch.__version__),'python':sys.version,'device':'cpu'},'input':'0.5 RGB field; two adjacent 2x2 areas in one 4x4 patch become 1 and 0. Values remain in [0,1].','fixed_entry':'channelwise mean, representable by a linear projection; hand-set weights, zero training','results':results,'scope':'A possibility proof of local detail surviving smaller grouping under a particular compressive entry map. It does not claim trained VisionEncoder improvement, small-object accuracy, small-text accuracy, or universal superiority; patchification itself retains all pixels.'},indent=2))
