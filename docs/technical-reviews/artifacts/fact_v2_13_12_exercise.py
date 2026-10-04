import torch

old_probability = torch.tensor([0.2, 0.2])
new_probability = torch.tensor([0.3, 0.1])
old_log_probability = old_probability.log().detach()
new_log_probability = new_probability.log()
ratio = (new_log_probability - old_log_probability).exp()
print("更新前後的機率比", [round(value, 4) for value in ratio.tolist()])
