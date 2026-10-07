import torch
import torch.nn.functional as F
content = torch.tensor([2.0, 2.0, 2.0]);gate = torch.tensor([-2.0, 0.0, 2.0]);factor=F.silu(gate)
print("閘門係數", factor.round(decimals=4).tolist());print("調節內容", (content * factor).round(decimals=4).tolist())
d,h=8,16;print("普通MLP權重數",d*h+h*d);print("SwiGLU權重數",d*h+d*h+h*d);print('torch',torch.__version__)
