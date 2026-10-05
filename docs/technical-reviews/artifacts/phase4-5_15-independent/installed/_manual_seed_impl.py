def _manual_seed_impl(seed) -> torch._C.Generator:
    seed = int(seed)
    import torch.cuda

    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    import torch.mps

    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    import torch.xpu

    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    import torch.mtia

    if not torch.mtia._is_in_bad_fork():
        torch.mtia.manual_seed_all(seed)

    _seed_custom_device(seed)

    return default_generator.manual_seed(seed)
