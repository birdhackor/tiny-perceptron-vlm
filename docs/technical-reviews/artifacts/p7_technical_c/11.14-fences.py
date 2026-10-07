import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
from tiny_perceptron.natural_concepts import picture_order_report

report = picture_order_report()
print("不同的像素數", report["different_pixels"])
print("4乘4平均摘要相同", report["same_4_by_4_summary"])
print("完整像素序列相同", report["same_pixel_sequence"])


# Owner supplied proportional check
from tiny_perceptron.natural_concepts import swapped_picture
from torch.nn import functional as F
a,b=swapped_picture();print('spatial changed',int((a!=b).any(0).sum()),'changed channel scalars',int((a!=b).sum()),'cell mean',a[:,8:16,0:8].mean((1,2)).tolist())
moved=a.clone();moved[0,8:16,0:4]=0;moved[0,8:16,8:12]=1
p=F.adaptive_avg_pool2d(a[None],(4,4));q=F.adaptive_avg_pool2d(moved[None],(4,4));print('red moved cell summary equal',torch.equal(p,q),'red old/new',p[0,0,1,:2].tolist(),q[0,0,1,:2].tolist())

