import io,json,math,platform,sys
import torch
from torch.nn import functional as F
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
x=torch.tensor([2.,1.,0.],dtype=torch.float64,requires_grad=True)
p=torch.softmax(x,dim=0)
hand=[math.exp(v)/sum(math.exp(v) for v in [2.,1.,0.]) for v in [2.,1.,0.]]
assert max(abs(a-b) for a,b in zip(p.tolist(),hand))<1e-14
rounded=[round(a,3) for a in p.tolist()]
assert rounded==[.665,.245,.090] and abs(p.sum().item()-1)<1e-14
loss=F.cross_entropy(x.unsqueeze(0),torch.tensor([2]))
assert abs(loss.item()+math.log(hand[2]))<1e-14
grad=torch.autograd.grad(loss,x)[0]
analytic=p.detach()-torch.tensor([0.,0.,1.],dtype=torch.float64)
assert torch.max(torch.abs(grad-analytic)).item()<1e-14
eps=1e-5
finite=[]
for j in range(3):
 plus=x.detach().clone();minus=x.detach().clone();plus[j]+=eps;minus[j]-=eps
 finite.append(((F.cross_entropy(plus[None],torch.tensor([2]))-F.cross_entropy(minus[None],torch.tensor([2])))/(2*eps)).item())
assert max(abs(a-b) for a,b in zip(grad.tolist(),finite))<1e-9
updated_loss=F.cross_entropy((x.detach()-.1*grad)[None],torch.tensor([2]))
assert updated_loss<loss
correct_loss=F.cross_entropy(x.detach()[None],torch.tensor([0]))
assert x.argmax().item()==0 and correct_loss.item()>0
masked_x=torch.tensor([[2.,1.,0.],[0.,1.,2.]],dtype=torch.float64)
masked_loss=F.cross_entropy(masked_x,torch.tensor([0,-100]),ignore_index=-100)
unmasked_loss=F.cross_entropy(masked_x,torch.tensor([0,0]))
assert abs(masked_loss.item()-correct_loss.item())<1e-14 and unmasked_loss>masked_loss
# Bounded checkpoint check: one scalar-parameter SGD step, bytes in memory only.
parameter=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
optimizer=torch.optim.SGD([parameter],lr=.1,momentum=.9)
(parameter.square().sum()).backward();optimizer.step();optimizer.zero_grad()
state={'parameter':parameter.detach().clone(),'optimizer_state_dict':optimizer.state_dict(),'step':1}
buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0);loaded=torch.load(buffer,weights_only=True)
assert loaded['step']==1 and torch.equal(loaded['parameter'],parameter)
newparam=torch.nn.Parameter(loaded['parameter'].clone());newoptim=torch.optim.SGD([newparam],lr=.1,momentum=.9);newoptim.load_state_dict(loaded['optimizer_state_dict'])
assert torch.equal(newoptim.state[newparam]['momentum_buffer'],optimizer.state[parameter]['momentum_buffer'])
# A correct answer and high confidence are different properties: top class 0, truth class 2.
result={'environment':{'python':platform.python_version(),'python_executable':sys.executable,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available())},'softmax':{'logits':[2,1,0],'probabilities':p.tolist(),'rounded_3dp':rounded,'sum':p.sum().item(),'normalization_denominator':sum(math.exp(v) for v in [2.,1.,0.]),'candidate_count':3,'probability_units':'dimensionless'},'loss':{'target_index':2,'target_convention':'zero-based third candidate','nll':loss.item(),'nll_units':'nats (natural log)','correct_argmax_target_index':0,'correct_argmax_nll':correct_loss.item(),'after_one_logits_gradient_step':updated_loss.item(),'step_size':.1,'gradient':grad.tolist(),'finite_difference_gradient':finite,'finite_difference_step':eps,'finite_difference_max_error':max(abs(a-b) for a,b in zip(grad.tolist(),finite))},'loss_scope_example':{'samples':2,'included_target_positions':1,'ignored_target_positions':1,'masked_mean':masked_loss.item(),'unmasked_mean':unmasked_loss.item(),'conclusion_scope':'same predictions, changed evaluated target positions; not a model capability evaluation'},'checkpoint':{'optimizer':'SGD with momentum 0.9','scalar_parameter_count':1,'steps':1,'roundtrip_weights_optimizer_progress_exact':True,'storage':'memory buffer only; no trained weights retained'},'confidence_example':{'argmax_class':0,'truth_class':2,'max_probability':p.max().item(),'correct':False,'scope':'logical counterexample only; not an empirical calibration estimate'},'status':'passed'}
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
