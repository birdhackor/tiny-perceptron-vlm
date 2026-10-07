swapped=F.normalize(image,dim=-1)@F.normalize(text.flip(0),dim=-1).T
print('swapped text',swapped.tolist(),swapped.argmax(-1).tolist())
print('normalize 3,4',F.normalize(torch.tensor([3.,4.]),dim=-1).tolist())
ties=F.normalize(torch.tensor([[1.,1.]]),dim=-1)@F.normalize(image,dim=-1).T
print('two same scores sum',ties.tolist(),ties.sum().item())
