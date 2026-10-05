# ORIGINAL tiny_perceptron/alignment.py:29-35
def sequence_log_probability(logits, labels):
    valid = labels != IGNORE
    safe_labels = labels.masked_fill(~valid, 0)
    selected = logits.log_softmax(-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    if not valid.any(-1).all():
        raise ValueError("偏好回答不能没有有效 token")
    return (selected * valid).sum(-1)

