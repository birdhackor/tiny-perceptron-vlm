import math,torch
print('torch',torch.__version__)
scores=torch.tensor([0.,math.log(2),math.log(4)]);values=torch.tensor([10.,20.,30.]);m=torch.tensor(float('-inf'));denominator=torch.tensor(0.);numerator=torch.tensor(0.)
for start in range(0,3,1):
 block=scores[start:start+1];new_m=torch.maximum(m,block.max());factor=torch.exp(m-new_m);weight=torch.exp(block-new_m)
 denominator=denominator*factor+weight.sum();numerator=numerator*factor+(weight*values[start:start+1]).sum();m=new_m;print("分母/分子",denominator.item(),numerator.item())
print("分塊結果",round((numerator/denominator).item(),4));print("完整結果",round((scores.softmax(0)*values).sum().item(),4));assert torch.allclose(numerator/denominator,(scores.softmax(0)*values).sum())
print('8192 squared',8192**2)
