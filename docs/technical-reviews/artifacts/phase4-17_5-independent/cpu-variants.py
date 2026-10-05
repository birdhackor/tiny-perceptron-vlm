import json, sys, hashlib, platform
from fractions import Fraction
from pathlib import Path
import torch
from tiny_perceptron.quantization import quantize_symmetric
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
w = torch.tensor([[0.01, 0.03, 0.05, 0.08], [10.0, 30.0, 50.0, 80.0]])
rows=[]
for per_row in (False, True):
    q, scale=quantize_symmetric(w,bits=4,per_channel=per_row)
    restored=q.float()*scale
    error=(w-restored).abs().mean(-1)
    assert q.dtype==torch.int8 and scale.dtype==torch.float32
    assert tuple(scale.shape)==((2,1) if per_row else ())
    assert q[0].tolist()==([1,3,4,7] if per_row else [0,0,0,0])
    assert torch.allclose(error,torch.tensor([0.0025 if per_row else 0.0425,2.5]),rtol=0,atol=1e-5)
    rows.append({'per_row':per_row,'q':q.tolist(),'scale':scale.tolist(),'scale_shape':list(scale.shape),'scale_bytes':scale.numel()*scale.element_size(),'q_container_bits':q.element_size()*8,'restored':restored.tolist(),'row_mae':error.tolist(),'row_denominator':w.shape[-1]})
variant=w.clone();variant[0]*=1000
q0,s0=quantize_symmetric(variant,bits=4,per_channel=False)
q1,s1=quantize_symmetric(variant,bits=4,per_channel=True)
e0=(variant-q0.float()*s0).abs().mean(-1)
e1=(variant-q1.float()*s1).abs().mean(-1)
assert torch.equal(q0,q1) and torch.allclose(e0,e1,rtol=0,atol=1e-5)
assert torch.allclose(e0,torch.tensor([2.5,2.5]),rtol=0,atol=1e-5)
# Independent exact rational calculation: round here is nearest-even, but no fixture is at a half integer.
small=[Fraction(1,100),Fraction(3,100),Fraction(5,100),Fraction(8,100)]
big=[Fraction(10),Fraction(30),Fraction(50),Fraction(80)]
exact=[]
for name,values,scale in [('global-small',small,Fraction(80,7)),('row-small',small,Fraction(8,700)),('big',big,Fraction(80,7))]:
    codes=[round(x/scale) for x in values]
    err=sum(abs(x-code*scale) for x,code in zip(values,codes))/len(values)
    exact.append({'name':name,'scale':str(scale),'codes':codes,'mae_fraction':str(err),'mae_decimal':float(err),'denominator':len(values)})
assert [x['mae_fraction'] for x in exact]==['17/400','1/400','5/2']
# Bound on nearest-grid error and FP32 scale storage are structural, not a trained model score.
print(json.dumps({'rows':rows,'multiply_first_row_by_1000':{'input':variant.tolist(),'global_q':q0.tolist(),'row_q':q1.tolist(),'global_mae':e0.tolist(),'row_mae':e1.tolist(),'global_scale_bytes':s0.numel()*s0.element_size(),'row_scale_bytes':s1.numel()*s1.element_size()},'exact_rational':exact,'storage_bit_accounting':{'original_fp32_per_weight':32,'int4_plus_fp32_scale_per_weight':4+32,'group_size_64_scale_overhead_bits_per_weight':32/64},'environment':{'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'torch_git_version':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda)}},ensure_ascii=False,indent=2))
