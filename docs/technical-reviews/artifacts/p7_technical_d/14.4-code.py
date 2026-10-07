import torch
import torch.nn.functional as F
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("輸入", x.tolist());print("GELU", F.gelu(x).round(decimals=4).tolist());print("ReLU平方", F.relu(x).square().tolist());print('torch',torch.__version__)
