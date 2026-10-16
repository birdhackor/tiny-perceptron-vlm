import random

loss_by_depth = {1: 1.2, 2: .8, 3: .7, 4: .9, 5: .6}
rng = random.Random(42)
R = rng.choice([1, 3, 5])
sampled_loss = loss_by_depth[R]
deep_loss = sum(loss_by_depth[r] for r in range(1, 6))/5
print("本次終止", R, "末端loss", sampled_loss, "逐輪平均", deep_loss)

# 審閱者補充的局部 autograd 核對；只查計算圖，不更新參數或評估品質。
import json, platform, sys, torch
import torch.nn.functional as F
print(json.dumps({"python":sys.version,"torch":torch.__version__,"device":"cpu","platform":platform.platform()},ensure_ascii=False))
print(json.dumps({"R":R,"sampled_loss":sampled_loss,"deep_loss":deep_loss,"uniform_sampled_expectation":sum(loss_by_depth[r] for r in [1,3,5])/3},ensure_ascii=False))
assert R == 5 and sampled_loss == 0.6 and abs(deep_loss-0.84) < 1e-14
logits=torch.tensor([[0.3,-0.2,0.5]],dtype=torch.float64,requires_grad=True)
target=torch.tensor([2],dtype=torch.long)
loss=F.cross_entropy(logits,target)
value=loss.item()
loss.backward()
expected_grad=torch.softmax(logits.detach(),dim=-1)
expected_grad[0,2]-=1
assert torch.allclose(logits.grad,expected_grad,atol=1e-14,rtol=1e-14)
assert isinstance(value,float) and not hasattr(value,"backward")
print(json.dumps({"cross_entropy_tensor":{"value":loss.item(),"requires_grad":loss.requires_grad,"grad_fn":type(loss.grad_fn).__name__,"logits_gradient":logits.grad.tolist(),"manual_softmax_minus_onehot":expected_grad.tolist()},"extracted_value":{"python_type":type(value).__name__,"has_backward":hasattr(value,"backward")}},ensure_ascii=False))
try:
 value.backward()
except AttributeError as error:
 print("float backward expected failure:",type(error).__name__,str(error))
else:
 raise AssertionError("Unexpected Python float backward")
# 平均多個 Tensor loss 仍保留可回傳的圖。
x=torch.tensor(2.0,dtype=torch.float64,requires_grad=True)
losses=[(x-k)**2 for k in (1.0,2.0,3.0)]
combined=sum(losses)/len(losses)
combined.backward()
assert combined.requires_grad and abs(combined.item()-2/3)<1e-14 and x.grad.item()==0.0
print(json.dumps({"averaged_tensor_losses":{"value":combined.item(),"requires_grad":combined.requires_grad,"x_grad":x.grad.item(),"expected_x_grad":"[2*(2-1)+2*(2-2)+2*(2-3)]/3 = 0"}},ensure_ascii=False))
print("All local mechanism checks passed; no training or accelerator benchmark was run.")
