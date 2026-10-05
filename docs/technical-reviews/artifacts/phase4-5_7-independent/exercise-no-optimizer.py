import tempfile
from pathlib import Path
import torch
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.training import save_checkpoint, load_checkpoint

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
masked_loss(model(x)["logits"], y).backward()
optimizer.step()
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "state.pt"
    save_checkpoint(path, model, optimizer, step=1, metadata={"chars": ["。", "狗", "看", "貓", "，"]})
    loaded, payload = load_checkpoint(path, restore_rng=True)
    restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.01)
    # Exercise: omit optimizer-state restoration.
    assert torch.equal(model(x)["logits"], loaded(x)["logits"])
    for candidate, opt in [(model, optimizer), (loaded, restored_optimizer)]:
        opt.zero_grad()
        masked_loss(candidate(x)["logits"], y).backward()
        opt.step()
    assert not torch.equal(model(x)["logits"], loaded(x)["logits"])
    print("exercise_after_update_max_logit_difference", float((model(x)["logits"] - loaded(x)["logits"]).detach().abs().max()))
    print("保存步數", payload["step"], "字表", payload["metadata"]["chars"])
