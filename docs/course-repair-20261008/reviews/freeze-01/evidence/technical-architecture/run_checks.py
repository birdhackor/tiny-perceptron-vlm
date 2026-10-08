from pathlib import Path
import json,re,contextlib,io,hashlib,sys,platform
from datetime import datetime,timezone
sys.path.insert(0,'/workspace/tiny-perceptron-vlm')
import torch
b=Path('/workspace/tiny-perceptron-vlm/docs/course-repair-20261008/reviews/freeze-01');m=json.loads((b/'manifest.json').read_text());out={'at':datetime.now(timezone.utc).isoformat(),'command':'.venv/bin/python /workspace/work/tutorial-repair-20261008/technical-architecture-sources/run_checks.py','environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},'pages':{}}
torch.set_num_threads(1)
for p in m['groups']['architecture']['pages']:
 path=b/'freeze/sources'/f'{p}.md';s=path.read_text();blocks=re.findall(r'```python\n(.*?)```',s,re.S);outputs=[]
 for index,code in enumerate(blocks):
  stream=io.StringIO()
  try:
   with contextlib.redirect_stdout(stream):exec(compile(code,str(path), 'exec'),{})
   outputs.append({'block':index,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'status':'executed','stdout':stream.getvalue()})
  except Exception as e:outputs.append({'block':index,'status':'failed','error':repr(e),'stdout':stream.getvalue()})
 out['pages'][p]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'blocks':outputs,'shell_blocks_executed':False}
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.modern import rope,MoEFFN
from tiny_perceptron.quantization import QuantizedLinear
from scripts.course_experiments.compression import _QATLinear
small={}
x=torch.randn(1,2,3,8);y=torch.randn_like(x);pos=torch.arange(3)
small['rope_shared_shift_dot_max_error']=float(((rope(x,pos)*rope(y,pos)).sum(-1)-(rope(x,pos+10)*rope(y,pos+10)).sum(-1)).abs().max())
small['rope_lengths_max_error']=float((rope(x,pos).norm(dim=-1)-x.norm(dim=-1)).abs().max())
for kv in (4,1):
 model=TinyLM(ModelConfig(width=16,heads=4,kv_heads=kv)).eval();ids=torch.tensor([[1,2,3,4,5,6]])
 with torch.no_grad():
  c=model(ids)['cache'];full=model(ids)['logits'][:,-1];pre=model(ids[:,:5]);step=model(ids[:,5:],cache=pre['cache'])['logits'][:,0]
 small[f'kv{kv}']={'k_shape':list(c[0][0].shape),'cache_bytes':sum(t.numel()*t.element_size() for pair in c for t in pair),'full_cached_max_error':float((full-step).abs().max())}
layer=torch.nn.Linear(3,2);fake=_QATLinear(layer);packed=QuantizedLinear(layer,4);inputs=torch.randn(4,3)
small['qat_packed_random_layer_max_error']=float((fake(inputs)-packed(inputs)).abs().max())
small['float_limits']={str(t):{'max':torch.finfo(t).max,'eps':torch.finfo(t).eps,'tiny':torch.finfo(t).tiny} for t in [torch.float32,torch.float16,torch.bfloat16]}
small['yarn_softmax_equal_scores']=(torch.tensor([0.,0.])*2).softmax(-1).tolist()
out['small_checks']=small
path=b/'evidence/technical-architecture/cpu-execution.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
