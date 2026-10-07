print('zero new difference',(projector(a)-projector(b)).norm().item(),'old difference',difference.item())
w=torch.tensor([1.,2.,0.,0.]);print('manual affine',float(w@a[0]+.5),float(w@b[0]+.5))
