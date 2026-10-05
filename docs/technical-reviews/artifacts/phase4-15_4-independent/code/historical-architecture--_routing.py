def _routing(model, examples, ctx):
    if not model.config.experts:
        return None
    counts = [torch.zeros(model.config.experts, dtype=torch.long) for _ in model.blocks]
    probability_sum = [torch.zeros(model.config.experts, dtype=torch.float64) for _ in model.blocks]
    aux_sum, input_tokens = [0.0 for _ in model.blocks], 0
    current_valid = None
    handles = []

    def make_hook(index):
        def collect(module, args, result):
            probability = module.router(args[0][current_valid]).float().softmax(-1)
            chosen = result[2][current_valid.reshape(-1)]
            count = torch.bincount(chosen.flatten(), minlength=model.config.experts).cpu()
            counts[index].add_(count)
            probability_sum[index].add_(probability.double().sum(0).cpu())
            load = count.to(probability.device).float() / count.sum()
            aux_sum[index] += float(model.config.experts * (load * probability.mean(0)).sum()) * len(probability)

        return collect

    model.eval()
    for i, block in enumerate(model.blocks):
        handles.append(block.ffn.register_forward_hook(make_hook(i)))
    try:
        for start in range(0, len(examples), 16):
            x, _, current_valid = (t.to(ctx.device) for t in pad_batch(examples[start : start + 16]))
            input_tokens += int(current_valid.sum())
            model(x, valid=current_valid)
    finally:
        for handle in handles:
            handle.remove()
    layers = []
    for count, probability, aux in zip(counts, probability_sum, aux_sum, strict=True):
        fraction = count.double() / count.sum()
        entropy = -float((fraction * fraction.clamp_min(1e-12).log()).sum())
        layers.append(
            {
                "dispatch_counts": count.tolist(),
                "dispatch_denominator": int(count.sum()),
                "load_fraction": fraction.tolist(),
                "load_entropy_nats": entropy,
                "normalized_load_entropy": entropy / math.log(model.config.experts),
                "mean_router_probability": (probability / input_tokens).tolist(),
                "mean_batch_auxiliary": aux / input_tokens,
            }
        )
    return {"effective_input_tokens": input_tokens, "padding_excluded": True, "layers": layers}
