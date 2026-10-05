@torch.no_grad()
def _sample(model, messages, ctx, count=1, tokens=32, temperature=0.0):
    """實際 batch 抽樣，保留 EOS、非法特殊 token 與測得的生成成本。"""
    tokenizer = ByteTokenizer()
    prompt = _prompt_ids(messages, tokenizer)
    if len(prompt) + tokens > model.config.max_length:
        raise ValueError(f"輸入 {len(prompt)} + 答案預留 {tokens} > {model.config.max_length}")
    was_training = model.training
    model.eval()
    current = torch.tensor([prompt] * count, device=ctx.device)
    active = torch.ones(count, dtype=torch.bool, device=ctx.device)
    outputs = [[] for _ in range(count)]
    state = None
    forward_tokens = 0
    _sync(ctx)
    started = time.perf_counter()
    try:
        for _ in range(tokens):
            result = model(current, cache=state)
            forward_tokens += current.numel()
            scores = result["logits"][:, -1]
            chosen = (
                scores.argmax(-1)
                if temperature <= 0
                else torch.multinomial((scores / temperature).softmax(-1), 1).squeeze(-1)
            )
            was_active = active.tolist()
            for row, (keep, token) in enumerate(zip(was_active, chosen.tolist(), strict=True)):
                if keep:
                    outputs[row].append(token)
            active = active & (chosen != tokenizer.eos_id)
            if not active.any():
                break
            current = chosen.masked_fill(~active, tokenizer.pad_id).unsqueeze(1)
            state = result["cache"]
    finally:
        model.train(was_training)
    _sync(ctx)
    seconds = time.perf_counter() - started
    samples = []
    for output in outputs:
        raw = output[:-1] if output and output[-1] == tokenizer.eos_id else output
        samples.append(
            {
                "generated": tokenizer.decode(raw),
                "generated_ids": output,
                "eos": bool(output and output[-1] == tokenizer.eos_id),
                "invalid_special_tokens": [token for token in raw if token < 8],
                "generated_tokens": len(output),
            }
        )
    return {
        "messages": messages,
        "input_ids": prompt,
        "input_tokens": len(prompt),
        "samples": samples,
        "candidate_count": count,
        "temperature": temperature,
        "max_new_tokens": tokens,
        "generated_tokens": sum(len(output) for output in outputs),
        "forward_input_tokens": forward_tokens,
        "seconds": seconds,
    }
def _reasoning_samples(model, rows, ctx, mode):
    budgets = []
    for candidate_count in (1, 2, 4, 8):
        samples = []
        for row in rows:
            generation = _sample(
                model,
                [{"role": "user", "content": row["question"]}],
                ctx,
                count=candidate_count,
                tokens=8 if mode == "direct" else 48,
                temperature=0.7,
            )
            candidates = [{**sample, **_verify_reasoning(sample, row, mode)} for sample in generation["samples"]]
            answers = [candidate["final_answer"] for candidate in candidates if candidate["final_answer"] is not None]
            majority = Counter(answers).most_common(1)[0][0] if answers else None
            verified = next(
                (candidate["final_answer"] for candidate in candidates if candidate["fully_verified"]), None
            )
            samples.append(
                {
                    **row,
                    "generation": {key: value for key, value in generation.items() if key != "samples"},
                    "candidates": candidates,
                    "oracle_coverage": any(candidate["final_correct"] for candidate in candidates),
                    "majority_answer": majority,
                    "majority_correct": majority == row["truth"],
                    "verified_answer": verified,
                    "verifier_correct": verified == row["truth"],
                    "selection_failed_despite_coverage": any(candidate["final_correct"] for candidate in candidates)
                    and majority != row["truth"],
                }
            )
        candidates = [candidate for sample in samples for candidate in sample["candidates"]]
        covered = [sample for sample in samples if sample["oracle_coverage"]]
        budgets.append(
            {
                "candidate_count": candidate_count,
                "oracle_coverage": _rate(len(covered), len(samples)),
                "majority_accuracy": _rate(sum(sample["majority_correct"] for sample in samples), len(samples)),
                "verifier_accuracy": _rate(sum(sample["verifier_correct"] for sample in samples), len(samples)),
                "majority_accuracy_given_coverage": _rate(
                    sum(sample["majority_correct"] for sample in covered), len(covered)
                ),
                "parsed_candidates": _rate(sum(candidate["parsed"] for candidate in candidates), len(candidates)),
                "candidate_final_accuracy": _rate(
                    sum(candidate["final_correct"] for candidate in candidates), len(candidates)
                ),
                "fully_verified_candidates": _rate(
                    sum(candidate["fully_verified"] for candidate in candidates), len(candidates)
                ),
                "final_correct_but_steps_invalid": _rate(
                    sum(candidate["final_correct"] and not candidate["fully_verified"] for candidate in candidates),
                    len(candidates),
                ),
                "generated_tokens": sum(sample["generation"]["generated_tokens"] for sample in samples),
                "forward_input_tokens": sum(sample["generation"]["forward_input_tokens"] for sample in samples),
                "generation_seconds": sum(sample["generation"]["seconds"] for sample in samples),
                "samples": samples,
            }
        )
    return budgets
