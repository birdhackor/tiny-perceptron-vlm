    policy = FiniteResponsePolicy().to(ctx.device)
    sft_optimizer = torch.optim.Adam(policy.parameters(), lr=CONFIG["sft_learning_rate"])
    sft_indices = [index for index, row in enumerate(train) if row["mode"] == "number"]
    sft_history = []
    stage_started = time.perf_counter()
    for step in range(CONFIG["sft_steps"]):
        indices = torch.tensor(sampler.choices(sft_indices, k=CONFIG["sft_batch_size"]), device=ctx.device)
        loss = F.cross_entropy(
            policy(x[indices]),
            torch.zeros(len(indices), dtype=torch.long, device=ctx.device),
            label_smoothing=CONFIG["sft_label_smoothing"],
        )
        _update(sft_optimizer, policy.parameters(), loss)
        if step in (0, CONFIG["sft_steps"] - 1):
            sft_history.append({"step": step + 1, "loss_before_update": float(loss.detach())})
    sft_seconds = time.perf_counter() - stage_started
    reference = copy.deepcopy(policy).requires_grad_(False).eval()
