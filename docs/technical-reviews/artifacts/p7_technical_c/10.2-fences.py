import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from tiny_perceptron.multimodal import scene, patchify, unpatchify

image = scene("red", "square")[None]
patches = patchify(image, 4)
restored = unpatchify(patches, 3, 16, 16, 4)
print("小塊序列", tuple(patches.shape))
print("拼回相同", torch.equal(image, restored))


# Owner supplied proportional check
cards = torch.tensor([[[[0., 1.], [2., 3.]]]])
print('four cards row order', patchify(cards, 1)[0, :, 0].tolist())
coordinate_image = torch.arange(3*16*16).reshape(1,3,16,16)
coordinate_patches = patchify(coordinate_image, 4)
print('second patch exact right upper C-row-col', torch.equal(coordinate_patches[0,1], coordinate_image[0,:,0:4,4:8].reshape(-1)))
changed = patchify(image, 8)
print('patch size 8', tuple(changed.shape), 'restored', torch.equal(image,unpatchify(changed,3,16,16,8)))

