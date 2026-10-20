"""以 CPU 小算例核對流程圖中的分工與更新順序。"""

import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(42)
dtype = torch.float64
q = torch.ones((1, 1, 3, 1), dtype=dtype)
k = q.clone()
v = torch.tensor([1., 2., 3.], dtype=dtype).view(1, 1, 3, 1)
attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
changed = v.clone()
changed[:, :, -1, :] = 300
attended_changed = F.scaled_dot_product_attention(q, k, changed, is_causal=True)
assert torch.equal(attended[:, :, :2], attended_changed[:, :, :2])
scores = torch.tensor([[2., -1.], [0., 1.], [-2., 3.]], dtype=dtype, requires_grad=True)
labels = torch.tensor([-100, 1, -100])
loss = F.cross_entropy(scores, labels)
expected_loss = -scores[1].log_softmax(-1)[1]
assert torch.equal(loss, expected_loss)
loss.backward()
assert torch.equal(scores.grad[[0, 2]], torch.zeros_like(scores.grad[[0, 2]]))
weight = torch.nn.Parameter(torch.tensor(1., dtype=dtype))
opt = torch.optim.SGD([weight], lr=.1)
weight.grad = torch.tensor(99., dtype=dtype)
before = float(weight.detach())
opt.zero_grad()
cleared = weight.grad is None
objective = (weight-3).square()
objective.backward()
gradient = float(weight.grad)
opt.step()
after = float(weight.detach())
assert cleared and gradient == -4 and abs(after-1.4) < 1e-12
opt.zero_grad()
((weight-3).square()/2).backward()
first = float(weight.grad)
((weight-3).square()/2).backward()
accumulated = float(weight.grad)
assert accumulated == first*2
raw_text = "中文A"
raw_bytes = len(raw_text.encode("utf-8"))
total_nats = 3.5
bpb = total_nats / (math.log(2)*raw_bytes)
assert raw_bytes == 7
help_command = ["uv", "run", "--no-sync", "python", "scripts/selftrained/train_local_stage.py", "--help"]
help_result = subprocess.run(help_command, capture_output=True, text=True, env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
assert help_result.returncode == 0, help_result.stderr
assert "Trainer arguments follow" in help_result.stdout
trainer_command = ["uv", "run", "--no-sync", "python", "scripts/selftrained/train.py", "--help"]
trainer_help = subprocess.run(trainer_command, capture_output=True, text=True, env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
sys.path.insert(0, str(Path.cwd()))
from scripts.selftrained.train import parser

trainer_options = sorted(parser()._option_string_actions)
assert "--resume" in trainer_options and "--init-checkpoint" in trainer_options
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-curriculum/verify_mechanics.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "platform": platform.platform(), "dtype": str(dtype)},
    "causal": {"values": v.flatten().tolist(), "output": attended.flatten().tolist(), "changed_last_value_output": attended_changed.flatten().tolist(), "past_outputs_equal": True},
    "loss_positions": {"labels": labels.tolist(), "loss": float(loss.detach()), "single_position_loss": float(expected_loss.detach()), "gradients": scores.grad.tolist()},
    "update": {"before": before, "old_gradient": 99., "zero_grad_cleared": cleared, "gradient": gradient, "learning_rate": .1, "after": after},
    "accumulation": {"first": first, "twice": accumulated},
    "bpb_derivation": {"formula": "sum scored negative log probabilities / (ln(2) * scored UTF-8 byte count)", "text": raw_text, "utf8_bytes": raw_bytes, "total_nats": total_nats, "bpb": bpb, "scope": "相同計分原文、相同邊界與上下文條件；不直接比較不同 tokenizer 的平均 token loss。"},
    "local_training_help": {"command": help_command, "exit_code": help_result.returncode, "stdout": help_result.stdout, "stderr": help_result.stderr},
    "trainer_help": {"command": trainer_command, "exit_code": trainer_help.returncode, "stdout": trainer_help.stdout, "stderr": trainer_help.stderr},
    "trainer_parser_options": trainer_options,
    "result": "機制斷言通過，本地 wrapper help 可用，trainer parser 登記新階段與 exact resume 參數；額外探查 trainer --help 因 help 字串格式失敗，完整 stderr 已保留。未執行訓練。"
}
target = Path(__file__).with_name("mechanics-evidence.json")
target.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"result": output["result"], "update": output["update"], "bpb": bpb}, ensure_ascii=False))
