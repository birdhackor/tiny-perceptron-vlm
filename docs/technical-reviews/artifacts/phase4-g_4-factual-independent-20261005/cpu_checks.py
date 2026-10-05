from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json,platform,re,sys,textwrap
import torch
A=Path(__file__).resolve().parent;R=Path.cwd();torch.set_num_threads(1);assert torch.version.cuda is None and not torch.cuda.is_available()
# Execute the original official method, without a model or external calls.
p=A/'sources/hf-prepare-length.py';ns={};exec(compile(textwrap.dedent(p.read_text()),str(p),'exec'),ns)
length_cases=[]
for prompt in [2,7]:
 g=SimpleNamespace(max_new_tokens=3,max_length=20,min_new_tokens=None,min_length=0)
 actual=ns['_prepare_generated_length'](SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False)),g,True,True,'input_ids',prompt,torch.zeros(1,prompt))
 assert actual.max_length==prompt+3 and actual.max_new_tokens==3;length_cases.append({'prompt_length':prompt,'max_new_tokens':3,'total_max_length':actual.max_length})
# Execute only personally inspected original functions, keeping author summaries out.
ns={'torch':torch,'nn':torch.nn};p=A/'code/tiny_perceptron/posttraining.py';exec(compile(p.read_bytes(),str(p),'exec'),ns)
adv=ns['bandit_advantage'](torch.tensor([1.,0.]),torch.tensor([.4,.4],requires_grad=True));assert torch.allclose(adv,torch.tensor([.6,-.4])) and not adv.requires_grad
new=torch.tensor([.4,.1],requires_grad=True);old=torch.tensor([.2,.2]);d=ns['ppo_clipped_objective'](new.log(),old.log(),adv)
assert torch.allclose(d['ratio'],torch.tensor([2.,.5]));assert torch.allclose(d['surrogate'],torch.tensor([.72,-.32]));d['policy_loss'].backward();assert torch.allclose(new.grad,torch.zeros(2))
# Negative-surrogate is a training objective, never interpreted as quality.
ns2={'MODES':('number','explain','missing')};p=A/'code/scripts/course_experiments/posttraining.py';exec(compile(p.read_bytes(),str(p),'exec'),ns2);records=ns2['build_records']();assert records and all(len(r['candidates'])==4 and isinstance(r['candidates'][0],str) for r in records)
ns3={'json':json};p=A/'code/tiny_perceptron/retrieval.py';exec(compile(p.read_bytes(),str(p),'exec'),ns3)
assert ns3['call_tool']('{"name":"add","arguments":{"a":2,"b":3}}')==5
rejected=[]
for text in ['{"name":"delete_all","arguments":{"a":2,"b":3}}','{"name":"add","arguments":{"a":true,"b":3}}']:
 try:ns3['call_tool'](text)
 except ValueError:rejected.append(text)
 else:raise AssertionError('must reject')
# Check each linked local start point exists, without using their correctness claims.
body=(A/'inputs/G.4.md').read_text();links=[]
for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',body):
 file,fragment=target.split('#');p=(R/'course'/file).resolve();assert p.is_file();assert re.search(r'^## '+re.escape(fragment)+r' ',p.read_text(),re.M);links.append(target)
result={'status':'passed','scope':'Bounded definitions and original methods only; no language generation, training, downloaded weights, timings or mature-method quality measurement.','environment':{'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'platform':platform.platform()},'max_new_tokens_cases':length_cases,'bandit_advantage':adv.tolist(),'ppo_ratio':d['ratio'].tolist(),'ppo_surrogate':d['surrogate'].tolist(),'clipped_case_gradient':new.grad.tolist(),'template_record_count':len(records),'response_options_each':4,'tool_call_success':5,'tool_call_rejected_count':len(rejected),'verified_local_links':links}
print(json.dumps(result,ensure_ascii=False,indent=2))
