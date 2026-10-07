import json,torch,hashlib,math
from tiny_perceptron.model import TinyLM,ModelConfig
r=json.load(open('docs/course-experiments/results/modern.json'))
for p in ['scripts/course_experiments/architecture.py','tiny_perceptron/modern.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/data.py']:
 cur=open(p,'rb').read();h=hashlib.sha256(cur).hexdigest();historical_path='docs/technical-reviews/artifacts/p7_technical_d/modern-historical-'+p.replace('/','_');old=open(historical_path,'rb').read();oh=hashlib.sha256(old).hexdigest();print('codehash',p,'current_match',h==r['code_sha256'][p],'historical_match',oh==r['code_sha256'][p]);assert oh==r['code_sha256'][p]
 if p.endswith('architecture.py'):
  import ast
  current_functions={n.name:ast.get_source_segment(cur.decode(),n) for n in ast.parse(cur.decode()).body if isinstance(n,ast.FunctionDef)}
  old_functions={n.name:ast.get_source_segment(old.decode(),n) for n in ast.parse(old.decode()).body if isinstance(n,ast.FunctionDef)}
  for fn in ['_text_dataset','_train','_nll','_clone_config','_heldout','run_modern']:
   same=current_functions[fn]==old_functions[fn];print('same_function',fn,same);assert same
for name in ['baseline','rope']:
 v=r['results']['variants'][name];m=TinyLM(ModelConfig(**v['model']['config']));n=sum(x.numel() for x in m.parameters());print(name,'parameters',n,'steps',v['training']['steps'],'max',m.config.max_length);assert n==v['model']['parameters']
 for split in ['validation','test']:
  x=v['heldout'][split];calc=x['nll_sum']/x['effective_tokens'];print(name,split,calc,x['nll'],x['effective_tokens'],abs(calc-x['nll']));assert abs(calc-x['nll'])<1e-12
 if name=='rope':
  rows=[x for x in v['heldout']['test']['samples'] if x['prompt']=='Once upon a time, there '];print('sample',[(x['prompt'],x['generated']) for x in rows])
print('parameter_difference',141568-133376,128*64)
print('variation22_26',math.cos((26-22)*math.pi/4));print('utf8sizes',len('a'.encode()),len('猫'.encode()));print('torch',torch.__version__)
