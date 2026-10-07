import torch
for scale in [1,2,.5]:
 scores=torch.tensor([1.,0.])*scale
 print(scale,scores.tolist(),scores.softmax(0).tolist(),float(scores.softmax(0).sum()))
