import torch
from torch import nn

torch.manual_seed(0)
teacher = nn.Linear(2, 3).eval().requires_grad_(False)
student = nn.Linear(2, 3)
optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
x = torch.tensor([[1.0, 2.0]])
teacher_before = teacher.weight.detach().clone()
student_before = student.weight.detach().clone()
with torch.no_grad():
    target = teacher(x)
output = student(x)
optimizer.zero_grad()
(output - target).square().mean().backward()
optimizer.step()
print("教師梯度", teacher.weight.grad)
print("教師最大改動", (teacher.weight - teacher_before).abs().max().item())
print("學生有改動", bool((student.weight - student_before).abs().max() > 0))
probe = x.clone().requires_grad_()
print("輸入可微時教師輸出記圖", teacher(probe).requires_grad)
