import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
from tiny_perceptron.natural_concepts import thin_stroke_report

report = thin_stroke_report()
print("原圖與縮圖尺寸", report["original_shape"], report["small_shape"])
print("最亮的數字", report["original_brightest"], report["small_brightest"])
print("至少半亮的數字個數", report["pixels_at_least_half_bright_before"], report["pixels_at_least_half_bright_after"])


# Owner supplied proportional check
from torch.nn import functional as F
x=torch.zeros(1,1,32,32);x[:,:,:,15]=1;y=torch.zeros_like(x);y[:,:,:,14]=1
print('different original/same small',torch.equal(x,y),torch.equal(F.interpolate(x,(4,4),mode='area'),F.interpolate(y,(4,4),mode='area')))
x[:,:,:,14]=1
print('two columns max',F.interpolate(x,(4,4),mode='area').max().item(),'finer max',F.interpolate(x,(8,8),mode='area').max().item())

