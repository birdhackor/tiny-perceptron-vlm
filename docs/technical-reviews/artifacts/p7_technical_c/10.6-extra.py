changed = nn.Linear(8,10)
print('out10',tuple(changed(vision).shape),tuple(changed.weight.shape))
toy=torch.tensor([1.,2.]);transform=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
print('recombine', (transform@toy).tolist())
