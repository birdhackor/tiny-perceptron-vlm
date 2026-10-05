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
