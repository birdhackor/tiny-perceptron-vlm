import json,math,hashlib,torch
from pathlib import Path
print('torch',torch.__version__);r=json.load(open('docs/course-experiments/results/flash_probe.json'));p=r['results'];assert hashlib.sha256(Path('scripts/course_experiments/architecture.py').read_bytes()).hexdigest()==r['code_sha256']['scripts/course_experiments/architecture.py']
shape=p['configuration']['shape_B_H_T_D'];print('shape',shape,'numel_each_output_or_QKV',math.prod(shape),'requested',p['configuration']['requested_backend'])
for n,d in p['routes'].items():
 c=d['correctness'];o=c['output'];gs=c['gradients'];gm=max(v['max_absolute_error'] for v in gs.values())
 for z in [o,*gs.values()]:assert z['finite'] and z['within_declared_tolerance'] and z['max_absolute_error']<z['atol']
 print(n,'outputmax',o['max_absolute_error'],'gradmax',gm,'out_tolerance',o['atol'],o['rtol'],'gradient_tolerances',[(k,v['atol'],v['rtol']) for k,v in gs.items()])
 for mode,x in d['profiles'].items():
  assert x['flash_cuda_verified'] and x['flash_forward_operator_observed'] and x['cuda_kernel_names']
  if x['includes_backward']:assert x['flash_backward_operator_observed']
  print(n,mode,'actual_flash_ops',[k for k in x['operator_names'] if 'scaled_dot_product_flash' in k],'CUDA_kernel_count',len(x['cuda_kernel_names']))
