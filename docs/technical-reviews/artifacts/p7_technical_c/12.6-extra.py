import math
print('mel-formula',[(f,2595*math.log10(1+f/700)) for f in [0,700,2100]])
p=torch.tensor([4.,2.,1.]);w=torch.tensor([[1.,.5,0.],[0.,.5,1.]])
print('small-weighted',w@p)
b32=mel_filter_bank(bands=32);print('bands32',tuple(b32.shape),tuple((b32@power).shape),bool((b32>=0).all()))
print('ones-output-first-last',mel_power[0,0].item(),mel_power[-1,0].item())
