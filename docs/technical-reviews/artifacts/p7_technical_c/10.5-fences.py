import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from torch import nn
from torch.nn import functional as F
from tiny_perceptron.multimodal import VisionEncoder, scene

torch.manual_seed(0)
encoder = VisionEncoder(width=8)
classifier = nn.Linear(8, 2)
images = torch.stack([scene("red", "square"), scene("blue", "circle")])
features = encoder(images)
logits = classifier(features.mean(1))
loss = F.cross_entropy(logits, torch.tensor([0, 1]))
loss.backward()
print("特徵", tuple(features.shape), "分數", tuple(logits.shape))
print("入口收到梯度", encoder.projection.weight.grad.norm().item() > 0)


# Owner supplied proportional check
before = {k:v.detach().clone() for k,v in encoder.state_dict().items()}
original_grad = encoder.projection.weight.grad.detach().clone()
encoder.zero_grad();classifier.zero_grad()
swapped_loss = F.cross_entropy(classifier(encoder(images).mean(1)),torch.tensor([1,0]))
swapped_loss.backward()
print('swapped targets gradient positive',encoder.projection.weight.grad.norm().item()>0)
print('swapped targets gradient changed',not torch.equal(original_grad,encoder.projection.weight.grad))
print('no optimizer encoder weights unchanged',all(torch.equal(v,before[k]) for k,v in encoder.state_dict().items()))

