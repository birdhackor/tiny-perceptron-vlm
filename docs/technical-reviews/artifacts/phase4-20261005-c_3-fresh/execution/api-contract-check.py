import json,math,sys
import torch
assert torch.version.cuda is None and not torch.cuda.is_available()
weights=torch.tensor([[0.,1.,0.]]).repeat(4,1)
picks=torch.multinomial(weights,1)
assert tuple(picks.shape)==(4,1) and torch.equal(picks,torch.ones((4,1),dtype=torch.long))
scores=torch.tensor([[0.,1.,2.]])
tempered=(scores/0.7).softmax(-1)
assert tuple(tempered.shape)==(1,3) and torch.allclose(tempered.sum(-1),torch.ones(1))
argmax=scores.argmax(-1)
assert argmax.tolist()==[2]
print(json.dumps({'python':sys.version,'torch':str(torch.__version__),'device':'cpu','torch_multinomial_single_draw_per_row_shape':list(picks.shape),'unit_mass_rows_output':[1,1,1,1],'temperature_0_7_preserves_probability_sum':True,'fixed_argmax_result':2,'scope':'API/shape and deterministic branch contract only; no model generation, inference benchmark, sampling-independence test or performance claim'}))
