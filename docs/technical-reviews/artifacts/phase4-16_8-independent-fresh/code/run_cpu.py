import contextlib,hashlib,io,json,os,sys,platform
from pathlib import Path
import torch
import torch.nn.functional as F

OUT=Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-16_8-independent-fresh')
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
environment=dict(python=sys.version,python_executable=sys.executable,torch=str(torch.__version__),torch_git_version=torch.version.git_version,device='cpu',cuda_build=str(torch.version.cuda),cuda_available=str(torch.cuda.is_available()),platform=platform.platform(),threads=str(torch.get_num_threads()))
(OUT/'cpu_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
raw=(OUT/'inputs/fence-1.py').read_bytes()
checks={}
transcript=io.StringIO()
for name,code,expect_failure in [('original',raw,False),('inverted_sdpa_mask',raw.replace(b'attn_mask=allowed',b'attn_mask=~allowed'),True),('seed_one',raw.replace(b'torch.manual_seed(0)',b'torch.manual_seed(1)'),False)]:
 namespace={}
 with contextlib.redirect_stdout(transcript):
  print('CASE',name)
  try:
   exec(compile(code,name,'exec'),namespace)
   failed=False
  except AssertionError:
   failed=True
   print('AssertionError as expected:',expect_failure)
 assert failed==expect_failure
 codepath=OUT/'code'/f'{name}.py';codepath.write_bytes(code)
 observed={k:list(namespace[k].shape) for k in ['q','k','v','allowed','scores','weights','manual','optimized']}
 observed.update(assert_failed=failed,max_abs_error=float((namespace['manual']-namespace['optimized']).abs().max()),allowed_counts=namespace['allowed'].sum(-1).flatten().tolist(),row_weight_sums=namespace['weights'].sum(-1).flatten().tolist(),blocked_max_weight=float(namespace['weights'].masked_select(~namespace['allowed']).abs().max()),code_sha256=hashlib.sha256(code).hexdigest(),dtype=str(namespace['q'].dtype))
 checks[name]=observed
 if name=='original':original=namespace

q,k,v=(original[x] for x in ['q','k','v'])
double_scaled=F.scaled_dot_product_attention(q/(3**0.5),k,v,attn_mask=original['allowed'],dropout_p=0.0)
checks['double_scaling']={'max_abs_difference':float((double_scaled-original['manual']).abs().max()),'differs':not torch.allclose(double_scaled,original['manual'],atol=1e-6)}
assert checks['double_scaling']['differs']

class DropoutProbe(torch.nn.Module):
 def forward(self,q,k,v,p):return F.scaled_dot_product_attention(q,k,v,attn_mask=original['allowed'],dropout_p=p)
module=DropoutProbe().eval()
torch.manual_seed(12)
a=module(q,k,v,0.5);b=module(q,k,v,0.5)
checks['eval_dropout']={'module_training':module.training,'nonzero_p_outputs_differ':not torch.equal(a,b),'zero_p_matches_manual':bool(torch.allclose(module(q,k,v,0.0),original['manual'],atol=1e-6))}
assert checks['eval_dropout']['nonzero_p_outputs_differ'] and checks['eval_dropout']['zero_p_matches_manual']

assert checks['original']['allowed_counts']==[1,2,3,4]
assert checks['original']['blocked_max_weight']==0
assert all(abs(x-1)<1e-6 for x in checks['original']['row_weight_sums'])
(OUT/'cpu_stdout.txt').write_text(transcript.getvalue())
(OUT/'cpu_results.json').write_text(json.dumps(checks,indent=2)+'\n')
print(transcript.getvalue())
print(json.dumps(checks,indent=2))
