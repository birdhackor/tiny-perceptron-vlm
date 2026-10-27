import ast,json,math,sys,hashlib,urllib.request
from pathlib import Path
import torch,torch.nn.functional as F
from tiny_perceptron.lesson_products.data import service_worlds,c_rows,split_of,choose_salt
from tiny_perceptron.lesson_products.inference import rule_depth
B=Path('/workspace/work/tokenizer-dynamic-20261027/curriculum-callback/sources')
A=Path('docs/technical-reviews/artifacts/tokenizer-20261027/curriculum-callback')
def helper(file,name,cls=None):
 tree=ast.parse((B/file).read_text());nodes=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
 fn=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name);fn.decorator_list=[]
 ns={'torch':torch,'math':math,'F':F,'dist':torch.distributed};exec(compile(ast.Module(body=[fn],type_ignores=[]),file,'exec'),ns);return ns[name]
x={'command':'uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/curriculum-callback/helper-audit.py','environment':{'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}}
f=helper('nanochat-loss_eval.py','evaluate_bpb')
class Fake:
 def get_device(self):return torch.device('cpu')
 def __call__(self,x,y,loss_reduction):return torch.tensor([[10.,1.,2.,100.]])
y=torch.tensor([[0,1,2,-1]]);bpb=f(Fake(),[(y,y)],1,torch.tensor([0,2,3]));assert abs(bpb-3/(5*math.log(2)))<1e-8
x['nanochat_bpb']={'ids':y.tolist(),'nats_per_position':[10,1,2,100],'token_bytes':[0,2,3],'normal_scored_nats':3,'normal_scored_bytes':5,'expected':3/(5*math.log(2)),'observed':bpb,'scope':'Original helper isolated by AST, synthetic return-loss model; both special token bytes=0 and ignored target excluded. Does not test any trained model.'}
f=helper('minimind-v-lm_dataset.py','generate_labels','VLMDataset');o=type('Stub',(),{'bos_id':[10,11],'eos_id':[12],'max_length':20})();ids=[1,2,10,11,6,7,12,3,10,11,9,12,0];labels=f(o,ids);expected=[-100,-100,-100,-100,6,7,12,-100,-100,-100,9,12,-100];assert labels==expected
x['minimind_v_labels']={'ids':ids,'expected':expected,'observed':labels,'scope':'Unchanged helper isolated by AST, synthetic tokens; confirms assistant content/end markers only, not data loading or VLM training.'}
f=helper('minimind-train_dpo.py','dpo_loss');a=torch.tensor([[.1,.2],[.3,.4]]);loss=f(a,a,torch.ones_like(a),.1).item();assert abs(loss-math.log(2))<1e-6;x['dpo_reference_identity']={'expected':math.log(2),'observed':loss,'tolerance':1e-6}
f=helper('minimind-train_distillation.py','distillation_loss');a=torch.tensor([[.1,.3,.5]]);loss=f(a,a,2.0).item();assert abs(loss)<1e-6;x['distillation_identity']={'expected':0,'observed':loss,'temperature':2,'tolerance':1e-6}
salt=choose_salt();found={}
for world in service_worlds():
 split=split_of(world['family_id'],salt)
 if split in found:continue
 for row in c_rows(world,split):
  if row['state']=='卡片進度;申請編號=?' and row['question']=='choice':found[split]={k:row[k] for k in ['family_id','split','state','question','gold']}
 if 'train' in found and 'test' in found:break
assert found['train']['family_id']!=found['test']['family_id'] and found['train']['state']==found['test']['state'] and found['train']['gold']==found['test']['gold']
x['duplicate_render']={'split_salt':salt,'examples':found,'scope':'Enumerated service-world metadata only, not rebuilding training data or running model inference.'}
policy=json.loads(Path('docs/lesson-products/results/b-auto-policy.json').read_text());x['rule_depth']={'sum':rule_depth(policy,'sum'),'boxes_0':rule_depth(policy,'boxes',0),'boxes_4':rule_depth(policy,'boxes',4),'boxes_5':rule_depth(policy,'boxes',5)}
(A/'helper-audit.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps(x,ensure_ascii=False,indent=2))
