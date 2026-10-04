import torch

from tiny_perceptron.capstone import CapstoneModel, build_dataset, default_config, prepare_batch
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss

torch.manual_seed(42)
splits, _ = build_dataset()
row = next(row for row in splits["train"] if row["task"] == "style")
print("問題", row["user"], "示範回答", row["answer"])
model = CapstoneModel(default_config(dense=True))
for pretrain in (True, False):
    batch, labels = prepare_batch([row], pretrain=pretrain)
    result = model(**batch)
    loss = masked_loss(result["logits"], labels)
    print("預訓練" if pretrain else "SFT", "有效目標", int((labels != IGNORE).sum()), "誤差有限", bool(loss.isfinite()))
