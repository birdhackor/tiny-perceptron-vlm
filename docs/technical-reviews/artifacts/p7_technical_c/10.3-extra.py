other_embedding = nn.Linear(48,16)
print('out16',tuple(other_embedding(patches).shape),tuple(other_embedding.weight.shape))
print('two input collision',(torch.tensor([[1.,0.],[0.,1.]]) @ torch.tensor([0.5,0.5])).tolist())
