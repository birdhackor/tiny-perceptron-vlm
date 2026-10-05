def _audio_records():
    splits = {}
    frequencies = {
        "train": (140, 180, 220, 260, 340, 380, 420, 460),
        "validation": (160, 240, 360, 440),
        "test": (200, 280, 290, 300, 310, 320, 400),
    }
    for split, values in frequencies.items():
        splits[split] = [
            {
                "modality": "audio",
                "frequency": f,
                "amplitude": amplitude,
                "seconds": seconds,
                "question": "pitch?",
                "answer": "high" if f > 300 else "low",
                "family": str(f),
            }
            for f in values
            for amplitude, seconds in ((0.25, 0.1), (0.5, 0.12))
        ]
    return splits

def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def confidence_metrics(logits, labels, temperature):
            probabilities = (logits / temperature).softmax(-1)
            # 正溫度保持 logits 的排名；用 logits 避免 softmax 捨入造成假平手。
            predicted = (logits / temperature).argmax(-1)
            correct = predicted == labels
            confidence = probabilities.max(-1).values
            one_hot = F.one_hot(labels, num_classes=len(classes)).to(probabilities.dtype)
            bins, ece = [], 0.0
            # [0,.2), [.2,.4), ..., [.8,1]；每題只進一格，空格也保存。
            bin_ids = (confidence * 5).floor().long().clamp(max=4)
            for bin_id in range(5):
                selected = bin_ids == bin_id
                count = int(selected.sum())
                mean_confidence = float(confidence[selected].mean()) if count else None
                accuracy = int(correct[selected].sum()) / count if count else None
                if count:
                    ece += count / len(labels) * abs(accuracy - mean_confidence)
                bins.append(
                    {
                        "lower": bin_id / 5,
                        "upper": (bin_id + 1) / 5,
                        "upper_inclusive": bin_id == 4,
                        "count": count,
                        "mean_confidence": mean_confidence,
                        "accuracy": accuracy,
                    }
                )
            return {
                "temperature": temperature,
                "count": len(labels),
                "correct": int(correct.sum()),
                "accuracy": int(correct.sum()) / len(labels),
                "mean_confidence": float(confidence.mean()),
                "nll": float(F.cross_entropy(logits / temperature, labels)),
                "brier": float((probabilities - one_hot).square().sum(-1).mean()),
                "brier_definition": "mean over examples of sum over classes (probability - one_hot_label)^2",
                "ece": ece,
                "ece_definition": "sum over five equal-width confidence bins: count/N * abs(bin accuracy - bin mean confidence)",
                "bins": bins,
                "probabilities": probabilities.tolist(),
                "predicted_labels": predicted.tolist(),
                "confidence": confidence.tolist(),
            }
