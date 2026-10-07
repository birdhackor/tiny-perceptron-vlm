print('stored difference after zero',difference.item(),difference.item()>0)
new_difference=(projector(a)-projector(b)).norm()
print('new difference after zero',new_difference.item(),new_difference.item()>0)
print('manual row outputs',float(a[0]@torch.tensor([1.,2.,0.,0.])+.5),float(b[0]@torch.tensor([1.,2.,0.,0.])+.5))
