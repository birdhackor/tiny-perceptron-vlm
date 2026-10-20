# 節錄自 tiny_perceptron/multimodal.py：expand_modalities
def expand_modalities(ids, labels, embedding, features, marker_ids):
    """一筆無 padding 資料：marker 展開成多向量，新增位置不直接算 loss。

    ids 與 labels 在此都尚未 shift。展開後才 shift，第一個答案的對齊才正確。
    """
    pieces, targets = [], []
    for token, label in zip(ids.tolist(), labels.tolist(), strict=True):
        if token in marker_ids:
            modal = features[token]
            pieces.append(modal)
            targets += [IGNORE] * len(modal)
        else:
            pieces.append(embedding(torch.tensor([token], device=embedding.weight.device)))
            targets.append(label)
    expanded = torch.cat(pieces)
    target = torch.tensor(targets, device=expanded.device, dtype=torch.long)
    return expanded[:-1].unsqueeze(0), target[1:].unsqueeze(0)
