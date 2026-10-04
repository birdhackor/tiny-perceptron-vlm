import torch

from tiny_perceptron.capstone import CapstoneModel, build_dataset, frozen_reference, preference_loss, preference_pairs

torch.manual_seed(42)
splits, _ = build_dataset()
pair = preference_pairs(splits["train"])[0]
print("同一問題", pair["row"]["user"])
policy = CapstoneModel()
reference = frozen_reference(policy)
loss, _ = preference_loss(policy, reference, [pair])
print("較喜歡", pair["chosen"], "較不喜歡", pair["rejected"])
print("相同起點的DPO誤差", round(loss.item(), 4), "參考可更新", any(p.requires_grad for p in reference.parameters()))
