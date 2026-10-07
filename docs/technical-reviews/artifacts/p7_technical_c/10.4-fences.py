import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch

patches = torch.ones(1, 4, 3)
position = torch.arange(4.0)[None, :, None]
x = patches + position
print("形狀", tuple(x.shape))
print("位置0", x[0, 0].tolist())
print("位置3", x[0, 3].tolist())
print("仍相同", torch.equal(x[:, 0], x[:, 3]))


# Owner supplied proportional check
print('without position',patches[0,0].tolist(),patches[0,3].tolist(),torch.equal(patches[:,0],patches[:,3]))

