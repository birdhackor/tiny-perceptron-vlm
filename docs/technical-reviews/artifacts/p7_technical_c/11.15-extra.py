from torch.nn import functional as F
x=torch.zeros(1,1,32,32);x[:,:,:,15]=1;y=torch.zeros_like(x);y[:,:,:,14]=1
print('different original/same small',torch.equal(x,y),torch.equal(F.interpolate(x,(4,4),mode='area'),F.interpolate(y,(4,4),mode='area')))
x[:,:,:,14]=1
print('two columns max',F.interpolate(x,(4,4),mode='area').max().item(),'finer max',F.interpolate(x,(8,8),mode='area').max().item())
