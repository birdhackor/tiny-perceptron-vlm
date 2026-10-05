import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.alignment import dpo_loss

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
rows = []
for margin in (2.0, -2.0):
    for beta in (0.1, 1.0, 5.0):
        chosen = torch.tensor([margin - 4.0], dtype=torch.float64, requires_grad=True)
        rejected = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
        refc = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
        refr = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
        loss = dpo_loss(chosen, rejected, refc, refr, beta=beta)
        loss.backward()
        expected_loss = math.log1p(math.exp(-beta * margin))
        expected_gradient = -beta / (1.0 + math.exp(beta * margin))
        assert abs(loss.item() - expected_loss) < 1e-12
        assert abs(chosen.grad.item() - expected_gradient) < 1e-12
        assert abs(rejected.grad.item() + expected_gradient) < 1e-12
        assert refc.grad is None and refr.grad is None
        lr = 0.1
        p = torch.nn.Parameter(torch.tensor([margin - 4.0], dtype=torch.float64))
        opt = torch.optim.SGD([p], lr=lr, momentum=0, weight_decay=0)
        dpo_loss(p, rejected.detach(), refc.detach(), refr.detach(), beta=beta).backward()
        before = p.detach().clone()
        opt.step()
        update = float((p.detach() - before).item())
        assert abs(update + lr * expected_gradient) < 1e-12
        rows.append(dict(margin=margin, beta=beta, loss=loss.item(), chosen_gradient=chosen.grad.item(),
                         analytic_loss=expected_loss, analytic_gradient=expected_gradient,
                         sigmoid=1 / (1 + math.exp(-beta * margin)), scalar_sgd_update_lr_0_1=update))

c = torch.tensor([-2.0], requires_grad=True)
for _ in range(2):
    dpo_loss(c, torch.tensor([-4.0]), torch.tensor([-3.0]), torch.tensor([-3.0]), beta=1).backward()
assert abs(c.grad.item() - 2 * rows[1]['chosen_gradient']) < 2e-8

batch = torch.tensor([-2.0, -2.0], dtype=torch.float64, requires_grad=True)
batch_loss = dpo_loss(batch, torch.tensor([-4., -4.]), torch.tensor([-3., -3.]), torch.tensor([-3., -3.]), beta=1)
batch_loss.backward()
assert abs(batch_loss.item() - rows[1]['loss']) < 1e-12
assert all(abs(x - rows[1]['chosen_gradient'] / 2) < 1e-12 for x in batch.grad.tolist())

rejections = []
for beta in (0.0, -0.1):
    try:
        dpo_loss(torch.tensor([-2.]), torch.tensor([-4.]), torch.tensor([-3.]), torch.tensor([-3.]), beta=beta)
    except ValueError as e:
        rejections.append({'beta': beta, 'exception': str(e)})
    else:
        raise AssertionError('nonpositive beta accepted')

print(json.dumps({'environment': {'python': sys.version, 'torch': torch.__version__,
      'torch_git_version': torch.version.git_version, 'device': 'cpu', 'cuda_build': str(torch.version.cuda)},
      'rows': rows, 'reused_leaf_two_backward_gradient': c.grad.item(),
      'mean_batch_gradients': batch.grad.tolist(), 'nonpositive_beta_rejected': rejections,
      'tolerance_float64': 1e-12, 'tolerance_float32_accumulation': 2e-8}, ensure_ascii=False, indent=2))
