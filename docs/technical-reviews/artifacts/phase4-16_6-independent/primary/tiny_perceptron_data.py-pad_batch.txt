def pad_batch(examples, pad_id=0, max_length=None):
    """右側 padding；attention 的有效位置和 loss 的有效位置各自保留。"""
    if not examples:
        raise ValueError("batch 不能是空的")
    length = max(len(x) for x, _ in examples)
    if max_length is not None:
        length = min(length, max_length)
    x = torch.full((len(examples), length), pad_id, dtype=torch.long)
    y = torch.full_like(x, IGNORE)
    valid = torch.zeros_like(x, dtype=torch.bool)
    for row, (inputs, labels) in enumerate(examples):
        n = min(length, len(inputs))
        x[row, :n], y[row, :n], valid[row, :n] = inputs[:n], labels[:n], True
        if not (y[row] != IGNORE).any():
            raise ValueError(f"第 {row} 筆截斷後没有有效答案；增加 max_length 或移除這筆")
    return x, y, valid
