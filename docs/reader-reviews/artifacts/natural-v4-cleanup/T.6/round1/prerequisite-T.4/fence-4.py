from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import save_checkpoint, seed_everything

seed_everything(42)
model = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
save_checkpoint(
    "checkpoints/start.pt",
    model,
    step=0,
    metadata={"seed": 42, "note": "untrained baseline"},
)
print("已保存起始模型；沒有求導或更新")
