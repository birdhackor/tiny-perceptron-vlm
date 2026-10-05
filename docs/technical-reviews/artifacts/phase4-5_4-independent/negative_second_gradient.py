import torch

g = torch.tensor([1.0, -100.0])
m = torch.zeros(2)
v = torch.zeros(2)
m = 0.9 * m + 0.1 * g
v = 0.999 * v + 0.001 * g.square()
m_hat = m / (1 - 0.9)
v_hat = v / (1 - 0.999)
update = 0.001 * m_hat / (v_hat.sqrt() + 1e-8)
print("方向平均", m, "平方平均", v)
print("原梯度", g, "應減去的更新量", update)
