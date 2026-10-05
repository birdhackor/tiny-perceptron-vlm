def _no_grad_normal_(
    tensor: Tensor,
    mean: float,
    std: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)
