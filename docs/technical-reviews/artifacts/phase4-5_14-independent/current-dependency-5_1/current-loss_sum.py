def loss_sum(logits, labels):
    """先求和再數有效 token，才能正確合併長短不同的 micro-batch。"""
    count = (labels != IGNORE).sum()
    if count.item() == 0:
        raise ValueError("所有 labels 都被忽略：没有可學習的答案")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE, reduction="sum"
    )
    return loss, count
