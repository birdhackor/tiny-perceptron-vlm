import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from tiny_perceptron.multimodal import mel_filter_bank

bank = mel_filter_bank(bands=16)
power = torch.ones(201, 3)
mel_power = bank @ power
print("權重表", tuple(bank.shape))
print("合成後", tuple(mel_power.shape))
print("權重非負", bool((bank >= 0).all()))


# Owner supplied proportional check
import math
print('mel-formula',[(f,2595*math.log10(1+f/700)) for f in [0,700,2100]])
p=torch.tensor([4.,2.,1.]);w=torch.tensor([[1.,.5,0.],[0.,.5,1.]])
print('small-weighted',w@p)
b32=mel_filter_bank(bands=32);print('bands32',tuple(b32.shape),tuple((b32@power).shape),bool((b32>=0).all()))
print('ones-output-first-last',mel_power[0,0].item(),mel_power[-1,0].item())

