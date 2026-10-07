import torch
q=torch.tensor([.5,.4,.1],dtype=torch.float64)
p=torch.tensor([.7,.2,.1],dtype=torch.float64)
raw=q.log()
detached=raw.detach()
logits=detached.clone().requires_grad_()
assert not q.requires_grad and not raw.requires_grad
assert detached.data_ptr()==raw.data_ptr() and logits.data_ptr()!=raw.data_ptr() and logits.is_leaf and logits.requires_grad
loss=-(p*logits.log_softmax(0)).sum();loss.backward()
assert torch.allclose(logits.grad,q-p,atol=1e-15,rtol=0)
before=logits.detach().clone();after=before-.1*logits.grad
assert after[2].item()==before[2].item()
after_q=after.softmax(0)
assert abs(after_q[2].item()-q[2].item())>1e-6
print("zero_direct_gradient",logits.grad[2].item(),"same_third_score",after[2].item()==before[2].item(),"third_probability",q[2].item(),"->",after_q[2].item(),"sum",after_q.sum().item())
q_upstream=torch.tensor([.5,.4,.1],dtype=torch.float64,requires_grad=True)
z=q_upstream.log().detach().clone().requires_grad_()
(-(p*z.log_softmax(0)).sum()).backward()
assert q_upstream.grad is None and z.grad is not None
print("detach_upstream_grad",q_upstream.grad,"fresh_leaf_gradient",z.grad.tolist())
qc=torch.tensor([.6,.2,.2],dtype=torch.float64)
for name,target in [("gold_cat",[1.,0.,0.]),("teacher_cat",[.8,.15,.05]),("teacher_wrong_dog",[.15,.8,.05])]:
 p_c=torch.tensor(target,dtype=torch.float64);z_c=qc.log().detach().clone().requires_grad_()
 (-(p_c*z_c.log_softmax(0)).sum()).backward()
 assert torch.allclose(z_c.grad,qc-p_c,atol=1e-15,rtol=0)
 print(name,"gradient",z_c.grad.tolist(),"analytic_q_minus_p",(qc-p_c).tolist())
print("no_model_training_or_accuracy_claim")
