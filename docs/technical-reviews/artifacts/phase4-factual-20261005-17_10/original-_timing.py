@torch.no_grad()
def _timing(model, record, device):
    """先暖機，再各量 prompt 與固定 8 次 decode；不以 EOS 長短掩蓋成本。"""
    if "text" in record:
        tok = ByteTokenizer()
        prompt = torch.tensor([[tok.bos_id] + tok.encode(record["text"])[:24]], device=device)
    else:
        prompt = _prompt(record, model.config.max_length)[None].to(device)
    if prompt.shape[1] + 8 > model.config.max_length:
        return {"measured": False, "reason": "prompt 沒留下 8 個 decode 位置"}
    model.eval()
    model(prompt)
    _sync(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
    else:
        base = None
    prefill, decode = [], []
    for _ in range(3):
        _sync(device)
        start = time.perf_counter()
        output = model(prompt)
        _sync(device)
        prefill.append(time.perf_counter() - start)
        ids = prompt
        start = time.perf_counter()
        for _ in range(8):
            token = output["logits"][:, -1].argmax(-1, keepdim=True)
            ids = torch.cat((ids, token), -1)
            output = model(ids)
        _sync(device)
        decode.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if base is not None else None
    return {
        "measured": True,
        "repetitions": 3,
        "prompt_tokens": prompt.shape[1],
        "decode_tokens": 8,
        "prefill_seconds": sum(prefill) / 3,
        "decode_seconds": sum(decode) / 3,
        "decode_seconds_per_token": sum(decode) / 24,
        "cuda_allocated_before": base,
        "cuda_peak_allocated": peak,
        "cuda_additional_peak_bytes": None if peak is None else peak - base,
        "note": "CUDA allocator peak includes co-resident models; additional peak is this probe, not a low-bit RAM claim",
    }
