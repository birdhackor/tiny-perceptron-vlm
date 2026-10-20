"""核對加法算例與 clipped surrogate 的單步更新範圍。"""

import json
import platform

import torch


torch.set_default_device("cpu")
theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
old_probability = theta.detach().sigmoid()
probability = theta.sigmoid()
ratio = probability / old_probability
advantage = torch.tensor(1.0, dtype=torch.float64)
epsilon = 0.2
objective = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)
gradient = torch.autograd.grad(objective, theta)[0]
learning_rate = 10.0
new_theta = theta.detach() + learning_rate * gradient
new_probability = new_theta.sigmoid()
new_ratio = new_probability / old_probability

print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "dtype": "float64"},
    "arithmetic": {"one_plus_two": 1 + 2, "wrong_answer": 4, "correctness_reward": int(4 == 1 + 2), "counting_two_steps_from_one": [1 + 1, 1 + 1 + 1]},
    "ppo_single_sample": {
        "formula": "maximize min(r*A, clip(r, 1-epsilon, 1+epsilon)*A), r=p_new/p_old",
        "initial_theta": theta.item(), "old_probability": old_probability.item(),
        "advantage": advantage.item(), "epsilon": epsilon, "objective_at_start": objective.item(),
        "gradient": gradient.item(), "learning_rate": learning_rate, "new_theta": new_theta.item(),
        "new_probability": new_probability.item(), "new_probability_ratio": new_ratio.item(),
        "ratio_exceeds_clip_threshold": bool(new_ratio > 1 + epsilon),
        "scope": "一個合法 SGD ascent 更新的反例：clipped objective 並不投影更新後的機率。此學習率用於否定硬保證，不是訓練建議，也不是量測模型品質。",
    },
}, ensure_ascii=False, indent=2))
