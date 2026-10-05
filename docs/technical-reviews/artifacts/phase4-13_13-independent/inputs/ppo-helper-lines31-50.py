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
