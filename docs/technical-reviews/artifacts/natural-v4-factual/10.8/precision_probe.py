from pathlib import Path
import copy,json,sys,time
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
start=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(0)
lm=TinyLM(ModelConfig(width=8)).eval()
ids=torch.tensor([[1,20,30]])
manual=lm(ids)["logits"]
other=copy.deepcopy(lm)
for block in other.blocks: block.attention.backend="sdpa"
sdpa=other(ids)["logits"]
# Attention formulas match: softmax(Q K^T / sqrt(D) + causal mask) V.
# Their floating-point implementations need not be identical.
same_parameters=all(torch.equal(v,other.state_dict()[k]) for k,v in lm.state_dict().items())
a,b,c=[torch.tensor(x,dtype=torch.float32) for x in (1e20,-1e20,3.0)]
left=(a+b)+c;right=a+(b+c)
record={"environment":{"python":sys.version.split()[0],"torch":torch.__version__,"device":"cpu","dtype":"torch.float32","threads":str(torch.get_num_threads())},"manual_vs_sdpa":{"seed":0,"ids":ids.tolist(),"shape":list(manual.shape),"same_weights":same_parameters,"manual_finite":bool(torch.isfinite(manual).all()),"sdpa_finite":bool(torch.isfinite(sdpa).all()),"torch_equal":bool(torch.equal(manual,sdpa)),"max_absolute_difference":float((manual-sdpa).abs().max().detach()),"number_different":int((manual!=sdpa).sum()),"same_argmax":bool(torch.equal(manual.argmax(-1),sdpa.argmax(-1)))},"associative_real_expression":{"a":float(a),"b":float(b),"c":float(c),"real_arithmetic_expected":3,"left_float32":float(left),"right_float32":float(right),"torch_equal":bool(torch.equal(left,right))},"elapsed_seconds":time.perf_counter()-start,"scope":"No training; 792 logits on one sequence. Demonstrates that mathematical equivalence is insufficient for exact equality. The lesson's actual direct-forward path remains exact."}
assert same_parameters and not record["manual_vs_sdpa"]["torch_equal"]
assert not torch.equal(left,right)
Path("docs/technical-reviews/artifacts/natural-v4-factual/10.8/precision-results.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
