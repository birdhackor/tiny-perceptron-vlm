import torch
from tiny_perceptron.attention import attention_mask
print('torch',torch.__version__)
position = torch.arange(4)
segment = torch.tensor([[0, 0, 1, 1]])
allowed = attention_mask(position, position, segments=segment)
print("隔離文件", allowed[0, 0].int().tolist())
causal_only = attention_mask(position, position)
print("只用因果的最後一列", causal_only[0, 0, -1].int().tolist())
print('variation_all_one_document',attention_mask(position,position,segments=torch.zeros_like(segment))[0,0].int().tolist())
assert allowed[0,0].int().tolist()==[[1,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,1,1]]
