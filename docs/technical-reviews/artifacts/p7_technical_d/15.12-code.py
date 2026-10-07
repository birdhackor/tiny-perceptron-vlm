import math,torch
chosen=torch.tensor([0,0,0,1,0,0]);tokens,experts,k=len(chosen),2,1;capacity_factor=2/3;capacity=math.ceil(capacity_factor*tokens*k/experts);counts=torch.bincount(chosen,minlength=experts);overflow=(counts-capacity).clamp(min=0)
print('每位容量',capacity);print('工作次數',counts.tolist());print('超出筆數',overflow.tolist());print('總超出',overflow.sum().item())
