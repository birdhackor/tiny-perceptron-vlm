import math
print('hand loss', math.log1p(math.exp(-4)))
print('hand gradient',1/(1+math.exp(4)))
print('hand step',2+0.1/(1+math.exp(4)),-0.1/(1+math.exp(4)))
wrong_scores=scores.detach().clone().requires_grad_(True)
wrong_loss=F.cross_entropy(wrong_scores/.5,torch.tensor([1,0])); wrong_loss.backward()
print('reversed',wrong_loss.item(),wrong_scores.grad.tolist())
