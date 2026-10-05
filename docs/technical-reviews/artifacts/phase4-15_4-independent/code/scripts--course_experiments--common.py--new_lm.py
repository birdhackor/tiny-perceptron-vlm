def new_lm(ctx, width=64, layers=2, max_length=128, **kwargs):
    seed(ctx.seed)
    return TinyLM(ModelConfig(width=width, layers=layers, max_length=max_length, **kwargs)).to(ctx.device)
