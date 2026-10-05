def _evaluate_tokenizer(model, rows, tokenizer):
    model.eval()
    device = next(model.parameters()).device
    total, count, raw_bytes = 0.0, 0, 0
    samples = []
    for index, row in enumerate(rows):
        ids = [tokenizer.bos_id] + tokenizer.encode(row["text"]) + [tokenizer.eos_id]
        x, y = shifted(ids)
        nll, effective = loss_sum(model(x[None].to(device))["logits"], y[None].to(device))
        total += float(nll)
        count += int(effective)
        raw_bytes += len(row["text"].encode())
        prompt = row["text"][:4]
        prefix = [tokenizer.bos_id] + tokenizer.encode(prompt)
        output = generate(model, torch.tensor([prefix], device=device), max_new_tokens=32, eos_id=tokenizer.eos_id)
        samples.append(
            {"row": index, "prompt": prompt, "generated": tokenizer.decode(output[0, len(prefix) :].tolist())}
        )
    return {
        "records": len(rows),
        "effective_tokens": count,
        "raw_utf8_bytes": raw_bytes,
        "mean_token_nll": total / count,
        "bpb_including_eos_boundary_targets": total / (raw_bytes * math.log(2)),
        "samples": samples,
        "skipped": [],
        "note": "same held-out raw text, vocabulary differs; EOS counted in total NLL",
    }
