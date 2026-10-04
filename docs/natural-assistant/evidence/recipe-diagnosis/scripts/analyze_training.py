"""CPU-only descriptive analysis; only train/validation rows are selected."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
manifest_path = ROOT / "docs/natural-assistant/manifest.json"
manifest = json.loads(manifest_path.read_text())
train = [r for r in manifest["rows"] if r["split"] == "train"]
validation = [r for r in manifest["rows"] if r["split"] == "validation"]
train_by_id = {r["id"]: r for r in train}
record = json.loads((ROOT / "docs/natural-assistant/evidence/train/training.json").read_text())
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", revision=record["model_revision"],
    cache_dir=ROOT / "outputs/natural-extension/student-base-cache/hf", local_files_only=True,
)


def supervision_count(row):
    # Image placeholders affect the prefix, not the assistant target suffix. No image is loaded.
    messages = [{"role": t["role"], "content": t["content"]} for t in row.get("history", [])]
    if row.get("system"):
        messages.insert(0, {"role": "system", "content": row["system"]})
    content = [{"type": "image", "image": row["image"]}] if row.get("image") else []
    content.append({"type": "text", "text": row["user"]})
    messages.append({"role": "user", "content": content})
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}],
                                         tokenize=False, add_generation_prompt=False)
    prefix = processor.tokenizer(prompt)["input_ids"]
    tokens = processor.tokenizer(full)["input_ids"]
    assert tokens[:len(prefix)] == prefix
    return len(tokens) - len(prefix)


target_tokens = {r["id"]: supervision_count(r) for r in train}
history = record["history"]
assert all(sum(target_tokens[rid] for rid in h["row_ids"]) == h["supervised_tokens"] for h in history)
visits = collections.Counter(rid for h in history for rid in h["row_ids"])


def history_summary(group):
    return {
        "updates": len(group),
        "token_weighted_loss": sum(h["answer_token_loss_before_update"] * h["supervised_tokens"] for h in group)
                               / sum(h["supervised_tokens"] for h in group) if group else None,
        "median_step_loss": statistics.median(h["answer_token_loss_before_update"] for h in group) if group else None,
        "supervised_tokens": sum(h["supervised_tokens"] for h in group),
        "clipped_updates": sum(h["gradient_norm_before_clip"] > 1 for h in group),
    }


task_stats = {}
for task in sorted({r["task"] for r in train}):
    rows = [r for r in train if r["task"] == task]
    pairs = {(r["user"], r["answer"], json.dumps(r.get("history", []), sort_keys=True)) for r in rows}
    tokens_once = sum(target_tokens[r["id"]] for r in rows)
    tokens_seen = sum(target_tokens[r["id"]] * visits[r["id"]] for r in rows)
    pure = [h for h in history if all(train_by_id[rid]["task"] == task for rid in h["row_ids"])]
    task_stats[task] = {
        "rows": len(rows), "unique_user_prompts": len({r["user"] for r in rows}),
        "unique_prompt_answer_history": len(pairs), "unique_answers": len({r["answer"] for r in rows}),
        "row_visits": sum(visits[r["id"]] for r in rows), "supervised_tokens_once": tokens_once,
        "supervised_tokens_seen": tokens_seen,
        "answer_tokens_median": statistics.median(target_tokens[r["id"]] for r in rows),
        "pure_task_updates": history_summary(pure),
        "pure_task_epoch1": history_summary([h for h in pure if h["step"] <= 136]),
        "pure_task_epoch2": history_summary([h for h in pure if h["step"] > 136]),
    }

base = json.loads((ROOT / "docs/natural-assistant/evidence/validation/generations-base.json").read_text())
adapter = json.loads((ROOT / "docs/natural-assistant/evidence/validation/generations-adapter.json").read_text())
assert all(r["split"] == "validation" for r in base + adapter)


def repeated_ngrams(row, n=4):
    tokens = [t for t in row["generated_token_ids"] if t not in row["eos_token_ids"]]
    counts = collections.Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))
    return max(counts.values(), default=0)


generation_stats = {}
repeat_examples = []
for variant, rows in [("base", base), ("adapter", adapter)]:
    generation_stats[variant] = {}
    for task in sorted({r["task"] for r in rows}):
        group = [r for r in rows if r["task"] == task]
        generation_stats[variant][task] = {
            "count": len(group), "truncated": sum(r["truncated"] for r in group),
            "ended_with_eos": sum(r["ended_with_eos"] for r in group),
            "generated_tokens_median": statistics.median(r["generated_tokens"] for r in group),
            "fourgram_repeated_at_least_3_times": sum(repeated_ngrams(r) >= 3 for r in group),
            "fourgram_repeated_at_least_10_times": sum(repeated_ngrams(r) >= 10 for r in group),
            "proxy_passed": sum((r.get("score") or {}).get("passed") is True for r in group),
            "proxy_scored": sum((r.get("score") or {}).get("passed") is not None for r in group),
        }
    for name, predicate in [("scene_descriptions", lambda r:r["task"] == "scene" and r["id"].endswith("/scene")),
                            ("scene_facts", lambda r:r["task"] == "scene" and "/fact-" in r["id"])]:
        group = [r for r in rows if predicate(r)]
        generation_stats[variant][name] = {
            "count": len(group), "truncated": sum(r["truncated"] for r in group),
            "generated_tokens_median": statistics.median(r["generated_tokens"] for r in group),
            "generated_tokens_mean": statistics.mean(r["generated_tokens"] for r in group),
            "max_fourgram_occurrences": max(repeated_ngrams(r) for r in group),
            "fourgram_repeated_at_least_10_times": sum(repeated_ngrams(r) >= 10 for r in group),
        }
    for r in rows:
        if r["task"] == "scene" and repeated_ngrams(r) >= 3:
            repeat_examples.append({"variant": variant, "id": r["id"], "user": r["user"],
                                    "max_fourgram_occurrences": repeated_ngrams(r),
                                    "generated_tokens": r["generated_tokens"], "truncated": r["truncated"],
                                    "prediction": r["prediction"]})

report = {
    "scope": "CPU tokenizer/descriptive analysis; no inference, no test output/image inspected; no independent semantic grading",
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "training_config": {k: record[k] for k in ["model", "model_revision", "learning_rate", "completed_steps",
        "gradient_accumulation", "trained_rows", "trainable_parameters", "total_parameters", "peak_cuda_memory_allocated_bytes"]},
    "exact_recorded_supervised_tokens_reproduced": True,
    "target_token_count_scope": "Exact template suffix count, including EOS/template terminal token; no image processing",
    "train_tasks": task_stats,
    "validation_task_counts": dict(collections.Counter(r["task"] for r in validation)),
    "scene_training_prompt": sorted({r["user"] for r in train if r["task"] == "scene"}),
    "scene_validation_prompts": sorted({r["user"] for r in validation if r["task"] == "scene"}),
    "all_updates": history_summary(history),
    "visit_distribution": dict(collections.Counter(visits.values())),
    "epochs": {"first_136_updates": history_summary(history[:136]), "next_44_updates": history_summary(history[136:])},
    "blocks_30_updates": {f"{i+1}-{min(i+30,len(history))}": history_summary(history[i:i+30])
                          for i in range(0, len(history), 30)},
    "loss_attribution_limit": "Recorded loss is token-weighted over two rows. Only pure-task updates identify a task-specific loss; epoch subsets are different examples and cannot establish per-example learning curves.",
    "generation_stats": generation_stats,
    "fourgram_metric_limit": "Raw generated-token ngram recurrence is an objective repetition screen, not a semantic completeness or grounding score.",
    "repeat_examples": repeat_examples,
}
(OUT / "training-and-output-analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ["scene_validation_prompts", "repeat_examples"]}, ensure_ascii=False, indent=2))
