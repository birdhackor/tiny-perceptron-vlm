import torch
teacher_vocab=['貓','狗'];student_vocab=['狗','貓'];p=torch.tensor([.8,.2]);q=torch.tensor([.2,.8]);order=[student_vocab.index(token) for token in teacher_vocab];reordered=q[order]
print('重排索引',order);print('錯配MAE',round((p-q).abs().mean().item(),4));print('對齊後',reordered.tolist(),'MAE',round((p-reordered).abs().mean().item(),4))
