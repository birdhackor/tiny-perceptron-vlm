import inspect,json,platform,torch
from pathlib import Path
from tiny_perceptron.model import TinyLM,ModelConfig
torch.manual_seed(42);m=TinyLM(ModelConfig(vocab_size=10,width=8))
base=m(torch.tensor([[1,2,3]]))['logits'];ids=torch.tensor([[0,0,1,2,3]]);valid=ids!=0;positions=torch.tensor([[0,0,0,1,2]])
actual=m(ids,valid=valid,positions=positions)['logits'][:,2:]
assert torch.allclose(base,actual,atol=1e-6,rtol=0.)
only_valid=m(ids,valid=valid)['logits'][:,2:];only_positions=m(ids,positions=positions)['logits'][:,2:]
src=Path('docs/technical-reviews/artifacts/p7_technical_b/sources/torch-mha-forward.py');assert not src.exists();src.write_text(inspect.getsource(torch.nn.MultiheadAttention.forward))
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','seed':42,'base_shape':list(base.shape),'both_correct_max_diff':float((base-actual).abs().max().detach()),'only_valid_max_diff':float((base-only_valid).abs().max().detach()),'only_positions_max_diff':float((base-only_positions).abs().max().detach()),'true_positions':positions[valid].tolist()},indent=2))
