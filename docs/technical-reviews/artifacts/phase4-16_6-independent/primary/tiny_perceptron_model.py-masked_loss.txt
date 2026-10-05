def masked_loss(logits, labels):
    total, count = loss_sum(logits, labels)
    return total / count
