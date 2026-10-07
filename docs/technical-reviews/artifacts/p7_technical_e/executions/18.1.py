import torch
candidates=['4','3','100'];probabilities=[torch.tensor([.7,.2,.1]),torch.tensor([.4,.35,.25])]
for prob in probabilities:
 answer=candidates[prob.argmax().item()];print('回答',answer,'分布',prob.tolist())
