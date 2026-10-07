import torch
teacher_vocab=['貓','狗'];student_vocab=['貓','狗'];p=torch.tensor([.8,.2]);q=torch.tensor([.8,.2]);order=[student_vocab.index(token) for token in teacher_vocab];print('order',order,'direct',(p-q).abs().mean().item(),'reordered',(p-q[order]).abs().mean().item())
