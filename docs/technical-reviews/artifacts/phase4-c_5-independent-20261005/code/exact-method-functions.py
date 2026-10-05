# Original scripts/course_experiments/applications.py:28-30
def _sync(ctx):
    if str(ctx.device).startswith("cuda"):
        torch.cuda.synchronize(ctx.device)

# Original scripts/course_experiments/applications.py:33-38
def _prompt_ids(messages, tokenizer):
    roles = {"system": tokenizer.system_id, "user": tokenizer.user_id, "assistant": tokenizer.assistant_id}
    ids = [tokenizer.bos_id]
    for message in messages:
        ids += [roles[message["role"]]] + tokenizer.encode(message["content"]) + [tokenizer.eos_id]
    return ids + [tokenizer.assistant_id]

# Original scripts/course_experiments/applications.py:41-103
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

# Original scripts/course_experiments/applications.py:106-111
def _rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }

# Original scripts/course_experiments/applications.py:679-695
def _reasoning_records():
    rows = []
    for a in range(6):
        for b in range(6):
            for c in range(6):
                rows.append(
                    {
                        # 相同數字的所有排列共用 family，測試沒有訓練題的重新排列。
                        "family": ":".join(map(str, sorted((a, b, c)))),
                        "a": a,
                        "b": b,
                        "c": c,
                        "question": f"({a}+{b})+{c}=?",
                        "truth": a + b + c,
                    }
                )
    return rows

# Original scripts/course_experiments/applications.py:707-747
def _verify_reasoning(sample, row, mode):
    text = sample["generated"].strip()
    clean = not sample["invalid_special_tokens"]
    if mode == "direct":
        parsed = int(text) if clean and re.fullmatch(r"-?[0-9]+", text) else None
        return {
            "final_answer": parsed,
            "parsed": parsed is not None,
            "final_correct": parsed == row["truth"],
            "fully_verified": parsed == row["truth"],
        }
    matched = re.fullmatch(
        r"(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)", text
    )
    if not matched or not clean:
        # 結尾格式可獨立解析，讓「最終對但過程錯」保留在資料中。
        final_match = re.search(r";answer=(-?[0-9]+)$", text) if clean else None
        parsed = int(final_match[1]) if final_match else None
        return {
            "final_answer": parsed,
            "parsed": False,
            "final_correct": parsed == row["truth"],
            "equations_valid": False,
            "linked": False,
            "task_operands_valid": False,
            "fully_verified": False,
        }
    a, b, subtotal, previous, c, total, final = map(int, matched.groups())
    equations = a + b == subtotal and previous + c == total
    linked = previous == subtotal and total == final
    task = (a, b, c) == (row["a"], row["b"], row["c"])
    return {
        "final_answer": final,
        "parsed": True,
        "steps": [[a, b, subtotal], [previous, c, total]],
        "final_correct": final == row["truth"],
        "equations_valid": equations,
        "linked": linked,
        "task_operands_valid": task,
        "fully_verified": equations and linked and task and final == row["truth"],
    }

# Original scripts/course_experiments/applications.py:750-811
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

