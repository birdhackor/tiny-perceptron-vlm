import torch
from tiny_perceptron.alignment import distillation_kl
teacher_prefix,student_prefix,answer_tokens=16,6,3
t_rows=list(range(teacher_prefix-1,teacher_prefix+answer_tokens-1));s_rows=list(range(student_prefix-1,student_prefix+answer_tokens-1))
t=torch.zeros(1,teacher_prefix+answer_tokens,2);s=torch.zeros(1,student_prefix+answer_tokens,2);t[:,t_rows]=torch.tensor([.8,.2]).log();labels=torch.zeros(1,3,dtype=torch.long)
print(t_rows,s_rows,tuple(t[:,t_rows].shape),tuple(s[:,s_rows].shape),round(distillation_kl(s[:,s_rows],t[:,t_rows],labels,temperature=1).item(),4))
