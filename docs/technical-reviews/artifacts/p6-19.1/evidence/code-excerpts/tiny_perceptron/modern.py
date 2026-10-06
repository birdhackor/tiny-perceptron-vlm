tiny_perceptron/modern.py:L8-L16
8: class RMSNorm(nn.Module):
9:     def __init__(self, width, eps=1e-5):
10:         super().__init__()
11:         self.weight = nn.Parameter(torch.ones(width))
12:         self.eps = eps
13: 
14:     def forward(self, x):
15:         scale = (x.float().square().mean(-1, keepdim=True) + self.eps).rsqrt()
16:         return x * scale.to(x.dtype) * self.weight
