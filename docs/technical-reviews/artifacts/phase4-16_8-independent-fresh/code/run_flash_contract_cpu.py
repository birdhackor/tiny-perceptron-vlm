import ast,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
OUT=Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-16_8-independent-fresh')
torch.set_num_threads(1)
assert torch.version.cuda is None
source=OUT/'inputs/flash_probe__scripts__course_experiments__architecture.py'
tree=ast.parse(source.read_bytes())
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_flash_manual_attention','_flash_sdpa_attention','_flash_error'}]
ns={'torch':torch,'F':F};exec(compile(ast.Module(body=selected,type_ignores=[]),'recorded_flash_methods','exec'),ns)
results={}
for name,dtype in [('fp16',torch.float16),('bf16',torch.bfloat16)]:
 generator=torch.Generator().manual_seed(42)
 inputs=tuple(torch.randn(2,4,8,32,generator=generator).to(dtype).requires_grad_() for _ in range(3))
 upstream=(torch.randn(2,4,8,32,generator=generator)*0.125).to(dtype)
 with sdpa_kernel(backends=[SDPBackend.MATH]):
  manual=ns['_flash_manual_attention'](*inputs);actual=ns['_flash_sdpa_attention'](*inputs)
  manual_grad=torch.autograd.grad(manual,inputs,upstream)
  actual_grad=torch.autograd.grad(actual,inputs,upstream)
 output=ns['_flash_error'](actual,manual,0.01 if name=='fp16' else 0.06,0.01 if name=='fp16' else 0.04)
 grads=[ns['_flash_error'](a,b,0.03 if name=='fp16' else 0.08,0.03 if name=='fp16' else 0.08) for a,b in zip(actual_grad,manual_grad)]
 assert all(c['finite'] and c['within_declared_tolerance'] for c in [output]+grads)
 results[name]={'shape':[2,4,8,32],'output':output,'gradients':dict(zip(['q','k','v'],grads)),'qk_matmul_dtype':str((inputs[0]@inputs[1].transpose(-2,-1)).dtype),'scaled_scores_dtype':str((inputs[0]@inputs[1].transpose(-2,-1)).float().dtype),'identical_upstream_object_for_both_routes':True}
result={'scope':'Only bounded CPU method-contract execution, force MATH, reduced T=8; not original fixture, original GPU error reproduction, Flash dispatch or model training.','torch':str(torch.__version__),'device':'cpu','backend':'SDPBackend.MATH','original_method_locators':[{ 'name':n.name,'first_line':n.lineno,'last_line':n.end_lineno} for n in selected],'results':results}
(OUT/'flash_cpu_contract.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'flash_cpu_stdout.txt').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
