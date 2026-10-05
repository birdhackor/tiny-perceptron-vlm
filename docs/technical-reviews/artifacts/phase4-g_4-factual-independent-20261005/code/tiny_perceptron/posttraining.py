def bandit_advantage(rewards, old_values):
    """單一步驟任務 A = R - V_old；本輪更新時兩份採樣資料保持固定。"""
    if rewards.shape != old_values.shape or not rewards.numel():
        raise ValueError("回饋與舊 value 必須同形狀且不可為空")
    return (rewards - old_values).detach()

def ppo_clipped_objective(new_log_prob, old_log_prob, advantage, clip_range=0.2):
    """PPO 論文式 (7)；回傳最小化的負 surrogate 與可檢查的各項。"""
    if new_log_prob.shape != old_log_prob.shape or new_log_prob.shape != advantage.shape:
        raise ValueError("新舊 log 機率與 advantage 必須同形狀")
    if not new_log_prob.numel() or not 0 < clip_range < 1:
        raise ValueError("至少需要一筆採樣，且 clip_range 必須在 0 與 1 之間")
    ratio = (new_log_prob - old_log_prob.detach()).exp()
    fixed_advantage = advantage.detach()
    unclipped = ratio * fixed_advantage
    clipped = ratio.clamp(1 - clip_range, 1 + clip_range) * fixed_advantage
    surrogate = torch.minimum(unclipped, clipped)
    return {
        "ratio": ratio,
        "unclipped": unclipped,
        "clipped": clipped,
        "surrogate": surrogate,
        "policy_loss": -surrogate.mean(),
        # 超出區間不一定是有害方向，也不代表每項都已失去梯度。
        "clip_fraction": ((ratio - 1).abs() > clip_range).float().mean(),
    }

class FiniteResponsePolicy(nn.Module):
    """輸入四個人工結構特徵；輸出四個事先寫好的回答的 logits。"""

    def __init__(self, features=4, width=16, actions=4):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, width), nn.Tanh(), nn.Linear(width, actions))

    def forward(self, features):
        return self.network(features)

class FiniteRewardModel(nn.Module):
    """對 context 與候選編號評分；不把候選中文當自然語言輸入。"""

    def __init__(self, features=4, width=24, actions=4):
        super().__init__()
        self.register_buffer("candidate_identity", torch.eye(actions))
        self.network = nn.Sequential(nn.Linear(features + actions, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, features):
        contexts = features[:, None, :].expand(-1, len(self.candidate_identity), -1)
        identities = self.candidate_identity[None, :, :].expand(len(features), -1, -1)
        return self.network(torch.cat((contexts, identities), -1)).squeeze(-1)

class FiniteValueModel(nn.Module):
    """從 context 預測目前策略會拿到的 RM 回饋；不是另一個偏好評審。"""

    def __init__(self, features=4, width=16):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)
