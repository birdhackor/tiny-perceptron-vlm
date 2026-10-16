# Historical source excerpts for C.3; original whole-file SHA-256 8fbb7e1c1a5ee33c50884a3cb6658352aacd80d2131ac6561c272e8c75b2ade0
# git show 910aebc6419c9fc6217279a27fde9851c5cfad30:scripts/course_experiments/applications.py

# Original lines 42-103
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

# Original lines 679-695
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

# Original lines 698-704
def _reasoning_sft(rows, mode):
    result = []
    for row in rows:
        a, b, c, truth = row["a"], row["b"], row["c"], row["truth"]
        answer = str(truth) if mode == "direct" else f"{a}+{b}={a + b};{a + b}+{c}={truth};answer={truth}"
        result.append({**row, "messages": _conversation(row["question"], answer)})
    return result

# Original lines 707-747
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

# Original lines 750-811
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

# Original lines 1071-1123
def run_reasoning(ctx):
    """同初始權重的短答／可驗證步驟，真實多候選，有限策略 REINFORCE。"""
    started = time.perf_counter()
    seed(ctx.seed)
    ctx.output.mkdir(parents=True, exist_ok=True)
    splits = split_records(_reasoning_records(), ctx.seed)
    write_json(ctx.output / "dataset.json", splits)
    base = new_lm(ctx, width=64, layers=2, max_length=128, heads=2, backend="sdpa")
    base_state = copy.deepcopy(base.state_dict())
    save_checkpoint(
        ctx.output / "same-base.pt",
        base,
        step=0,
        metadata={"seed": ctx.seed, "purpose": "identical direct/steps initialization"},
    )
    base_hash = hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in base_state.values())
    ).hexdigest()
    models = {}
    for mode in ("direct", "steps"):
        seed(ctx.seed)
        model = new_lm(ctx, width=64, layers=2, max_length=128, heads=2, backend="sdpa")
        model.load_state_dict(base_state)
        mode_splits = {name: _reasoning_sft(rows, mode) for name, rows in splits.items()}
        write_json(ctx.output / f"{mode}-dataset.json", mode_splits)
        training = _fit_lm(model, mode_splits["train"], ctx, mode="sft", steps=900, batch_size=24, name=mode)
        models[mode] = {
            "base_state_sha256": base_hash,
            "training": training,
            "validation": _sft_nll(model, mode_splits["validation"], ctx),
            "test": _sft_nll(model, mode_splits["test"], ctx),
            "budgets": _reasoning_samples(model, splits["test"], ctx, mode),
        }
    result = {
        "task": "C: controlled two-addition arithmetic, candidate selection, and finite-policy RLVR",
        "seed": ctx.seed,
        "split": _split_summary(splits),
        "model_config": asdict(base.config),
        "comparison": models,
        "reinforce": _reinforce(splits, ctx),
        "gsm8k_training_pilot": _gsm8k_pilot(ctx),
        "seconds": time.perf_counter() - started,
        "limits": [
            "Direct and step models start from identical random weights and receive equal optimizer steps and sampled example counts, not equal supervised-token budgets.",
            "Each N budget really samples a separate batch; candidate generations and measured batch times are preserved.",
            "Oracle coverage is scored using known arithmetic answers; it is not a deployable general selector.",
            "The step verifier checks both equations, linking, original operands, and final answer in this narrow task.",
            "Visible valid steps and self-consistency do not establish faithfulness of an internal chain of thought.",
            "Weak arithmetic generalization or GSM8K loss reduction must not be presented as natural-language complex reasoning.",
        ],
    }
    write_json(ctx.output / "generations.json", models)
    return result
