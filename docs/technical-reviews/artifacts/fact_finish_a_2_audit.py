"""A.2 independent CPU replay and exhaustive record/configuration arithmetic.

The default downloads the fixed public HF revision anonymously and verifies its
checksum and every tensor against the tracked metadata/hash inventory.
No optimizer update or paid compute is performed.
"""

import argparse
import copy
import hashlib
import io
import json
import platform
import random
import re
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download

from scripts.course_experiments.applications import (
    _icl_records,
    _rag_splits,
    _sample,
)
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[3]
ART = Path(__file__).resolve().parent
PREFIX = "fact_finish_a_2_"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save(name, value):
    path = ART / (PREFIX + name)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path


def score(generated, ids, docs, expected, fact_answer):
    raw = ids[:-1] if ids and ids[-1] == 2 else ids
    clean = all(token >= 8 for token in raw)
    parsed = re.fullmatch(r"([A-E][0-9])\[(D[0-9])\]", generated)
    address, citation = parsed.groups() if parsed else (None, None)
    supported = any(d["id"] == citation and f"address={address}" in d["text"] for d in docs)
    return {
        "exact_match": clean and generated == expected,
        "fact_answer_correct": clean and generated == fact_answer,
        "citation_valid": clean and citation in {d["id"] for d in docs},
        "supported_by_cited_source": clean and parsed is not None and supported,
        "unknown": clean and generated == "UNKNOWN",
        "parsed_address": address,
        "parsed_citation": citation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--label", default="hf_replay")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = json.loads((ART / (PREFIX + "formal_rag.json")).read_text())
    result = report["results"]
    seed = report["seed"]
    assert seed == result["seed"] == 42
    assert report["step_scale"] == result["training"]["step_scale"] == 1.0
    code_inventory = {}
    relevant_code = {
        "scripts/course_experiments/applications.py",
        "scripts/course_experiments/common.py",
        "scripts/course_experiments/run.py",
        "tiny_perceptron/__init__.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/model.py",
        "tiny_perceptron/training.py",
        "tiny_perceptron/retrieval.py",
        "tiny_perceptron/attention.py",
        "tiny_perceptron/modern.py",
    }
    for path, expected_hash in report["code_sha256"].items():
        observed_hash = digest(ROOT / path)
        if path in relevant_code:
            assert observed_hash == expected_hash, path
        code_inventory[path] = {
            "current_sha256": observed_hash,
            "formal_sha256": expected_hash,
            "required_by_rag_replay": path in relevant_code,
        }

    public = json.loads((ROOT / "docs/course-experiments/public-models.json").read_text())
    public = next(model for model in public["models"] if model["id"] == "rag")
    exported = json.loads((ART / (PREFIX + "export-manifest.json")).read_text())
    private_checkpoint = next(item for item in report["artifacts"] if item["path"] == "model.pt")
    assert exported["files"][0]["source_sha256"] == private_checkpoint["sha256"]
    expected = next(f for f in public["files"] if f["output"] == "model.pt")
    checkpoint = args.checkpoint or Path(
        hf_hub_download(repo_id=public["repo"], revision=public["revision"], filename=expected["path"], token=False)
    )
    assert digest(checkpoint) == expected["sha256"] == exported["files"][0]["sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    inventory = {
        "origin": {
            "repo": public["repo"],
            "revision": public["revision"],
            "path": expected["path"],
            "sha256": expected["sha256"],
            "bytes": expected["bytes"],
            "private_source_sha256": private_checkpoint["sha256"],
        },
        "config": payload["config"],
        "metadata": payload["metadata"],
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256_raw_contiguous_bytes": tensor_digest(tensor),
            }
            for name, tensor in payload["model"].items()
        },
        "scope": "Tensor identity/shape/dtype/raw-contiguous-byte SHA only. Model values remain in pinned public HF storage and ignored local downloads.",
    }
    previous_inventory = json.loads((ART / (PREFIX + "model_tensors.json")).read_text())
    assert inventory == previous_inventory
    for name, tensor in payload["model"].items():
        assert tensor_digest(tensor) == inventory["tensors"][name]["sha256_raw_contiguous_bytes"]
    assert payload["config"] == result["model"]["config"]
    assert payload["metadata"]["revision"] == report["revision"]
    model = TinyLM(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    assert model.description() == result["model"]
    before = {name: tensor_digest(tensor) for name, tensor in model.state_dict().items()}

    splits = _rag_splits(seed)
    icl_splits = split_records(_icl_records(), seed)
    original_splits = copy.deepcopy(splits)
    original_icl = copy.deepcopy(icl_splits)
    record_checks = []
    split_summary = {}
    all_family_sets = []
    for name, records in splits.items():
        keys = {record["key"] for record in records}
        all_family_sets.append(keys)
        actual = {"records": len(records), "families": len(keys), "sha256": records_sha256(records)}
        assert actual == result["split"][name]
        split_summary[name] = actual
        assert Counter(row["key"] for row in records) == Counter(dict.fromkeys(keys, 8))
        for row in records:
            docs = row["documents"]
            assert row["key"] == row["family"]
            assert all(document["text"].split()[0] in keys for document in docs)
            answer = row["messages"][1]["content"]
            expected = f"{row['address']}[{row['source']}]" if row["variant"] < 6 else "UNKNOWN"
            assert answer == expected
            context = "\n".join(f"[{d['id']}] {d['text']}" for d in docs) or "EMPTY"
            question = f"Docs:\n{context}\nFind:{row['key']}\nReply:address[source] or UNKNOWN"
            assert row["messages"][0]["content"] == question
            record_checks.append({"split": name, "key": row["key"], "variant": row["variant"], "answer": answer})
    assert not any(all_family_sets[i] & all_family_sets[j] for i in range(3) for j in range(i))
    icl_summary = {}
    for name, rows in icl_splits.items():
        observed = {"records": len(rows), "families": len({r["family"] for r in rows}), "sha256": records_sha256(rows)}
        assert observed == result["icl_split"][name]
        icl_summary[name] = observed
    for filename, data in (("dataset.json", splits), ("icl-dataset.json", icl_splits)):
        path = save(filename, data)
        official = next(a for a in report["artifacts"] if a["path"] == filename)
        assert digest(path) == official["sha256"]

    training_rows = splits["train"] + icl_splits["train"]
    assert len(training_rows) == result["training"]["records"] == 894
    assert records_sha256(training_rows) == result["training"]["records_sha256"]
    examples = text_examples(training_rows, mode="sft", max_length=160)
    target_counts = []
    for row, (inputs, labels) in zip(training_rows, examples, strict=True):
        assert inputs.dtype == labels.dtype == torch.int64
        count = int((labels != IGNORE).sum())
        expected = len(row["messages"][1]["content"].encode("utf-8")) + 1
        assert count == expected
        target_counts.append(count)
        answer_ids = ByteTokenizer().encode(row["messages"][1]["content"]) + [2]
        assert labels[labels != IGNORE].tolist() == answer_ids
    sampler = random.Random(seed)
    step_targets = []
    for _ in range(1000):
        batch = sampler.choices(examples, k=24)
        step_targets.append(sum(int((labels != IGNORE).sum()) for _, labels in batch))
    assert sum(step_targets) == result["training"]["effective_tokens"] == 157361
    assert result["training"]["steps"] == result["training"]["planned_steps"] == 1000
    save(
        "record_checks.json", {"rag": record_checks, "train_target_counts": target_counts, "step_targets": step_targets}
    )

    rng = random.Random(seed + 100)
    facts = []
    for key in sorted(all_family_sets[2]):
        source = f"D{rng.randrange(10)}"
        address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
        facts.append({"family": key, "key": key, "address": address, "source": source})
    assert records_sha256(facts) == result["test_facts_sha256"]
    corpus = [{"id": f"doc-{row['key']}", "text": f"{row['key']} address={row['address']}"} for row in facts]
    corpus += [
        {"id": f"noise-{index}", "text": f"Z{index:03d} address={rng.choice('ABCD')}{rng.randrange(10)}"}
        for index in range(60)
    ]
    assert records_sha256(corpus) == result["corpus_sha256"]
    corpus_path = save("retrieval-corpus.json", corpus)
    assert digest(corpus_path) == next(a for a in report["artifacts"] if a["path"] == "retrieval-corpus.json")["sha256"]
    metrics = defaultdict(Counter)
    seen = set()
    replay = []
    tokenizer = ByteTokenizer()
    modes = {
        "without_context",
        "correct_context",
        "retrieved_context",
        "distractor_only",
        "correct_with_distractor",
        "changed_address_context",
        "changed_source_context",
    }
    for row in result["samples"]:
        key, mode = row["family"], row["mode"]
        assert (key, mode) not in seen and mode in modes
        seen.add((key, mode))
        fact = next(f for f in facts if f["key"] == key)
        assert row["fact"] == fact
        assert key not in all_family_sets[0]
        hits = retrieve(row["query"], corpus, k=1)
        assert row["retrieved_ids"] == [hit["id"] for hit in hits]
        assert row["retrieval_hit"] == any(hit["id"] == f"doc-{key}" for hit in hits)
        assert row["citation_id_map"] == {hit["id"]: fact["source"] for hit in hits}
        docs = row["documents"]
        context_fact = row["context_fact"]
        if mode == "changed_address_context":
            assert context_fact == {**fact, "address": fact["address"][0] + str((int(fact["address"][1]) + 1) % 10)}
        elif mode == "changed_source_context":
            assert context_fact == {**fact, "source": "D" + str((int(fact["source"][1]) + 1) % 10)}
        else:
            assert context_fact == fact
        fact_answer = f"{context_fact['address']}[{context_fact['source']}]"
        expected = (
            "UNKNOWN"
            if mode in {"without_context", "distractor_only"} or (mode == "retrieved_context" and not hits)
            else fact_answer
        )
        assert expected == row["expected"]
        if mode in {"correct_context", "changed_address_context", "changed_source_context"}:
            assert docs == [{"id": context_fact["source"], "text": f"{key} address={context_fact['address']}"}]
        elif mode == "without_context":
            assert docs == []
        elif mode == "retrieved_context":
            assert docs == [{"id": fact["source"], "text": hit["text"]} for hit in hits]
        context = "\n".join(f"[{d['id']}] {d['text']}" for d in docs) or "EMPTY"
        prompt = f"Docs:\n{context}\nFind:{key}\nReply:address[source] or UNKNOWN"
        assert row["generation"]["messages"] == [{"role": "user", "content": prompt}]
        ids = [1, 3] + tokenizer.encode(prompt) + [2, 4]
        assert ids == row["generation"]["input_ids"]
        assert len(ids) == row["generation"]["input_tokens"]
        assert len(ids) + row["generation"]["max_new_tokens"] <= 160
        assert row["generation"]["candidate_count"] == 1
        assert row["generation"]["max_new_tokens"] == 16
        assert row["generation"]["temperature"] == 0.0
        sample = row["generation"]["samples"][0]
        raw = sample["generated_ids"][:-1] if sample["eos"] else sample["generated_ids"]
        assert sample["eos"] == (sample["generated_ids"][-1] == 2)
        assert sample["generated"] == tokenizer.decode(raw)
        assert sample["invalid_special_tokens"] == [token for token in raw if token < 8]
        scores = score(sample["generated"], sample["generated_ids"], docs, expected, fact_answer)
        assert all(row[field] == value for field, value in scores.items())
        observed = _sample(model, row["generation"]["messages"], SimpleNamespace(device="cpu"), tokens=16)
        observed_sample = observed["samples"][0]
        assert observed["input_ids"] == ids
        exact_ids = observed_sample["generated_ids"] == sample["generated_ids"]
        replay.append(
            {
                "family": key,
                "mode": mode,
                "documents": docs,
                "prompt": prompt,
                "input_ids": ids,
                "expected": expected,
                "formal_generated_ids": sample["generated_ids"],
                "cpu_generated_ids": observed_sample["generated_ids"],
                "cpu_generated": observed_sample["generated"],
                "identical_ids": exact_ids,
                "scores": scores,
            }
        )
        for field in ("exact_match", "fact_answer_correct", "citation_valid", "supported_by_cited_source", "unknown"):
            metrics[mode][field] += int(scores[field])
        metrics[mode]["denominator"] += 1
    assert len(seen) == 84
    assert seen == {(f["key"], mode) for f in facts for mode in modes}
    for mode, counts in metrics.items():
        assert counts["denominator"] == 12
        for field in ("exact_match", "fact_answer_correct", "citation_valid", "supported_by_cited_source", "unknown"):
            original = result["metrics"][mode][field]
            assert original == {"numerator": counts[field], "denominator": 12, "rate": counts[field] / 12}
    baseline = {r["family"]: r for r in replay if r["mode"] == "correct_context"}
    paired = {}
    for mode in ("changed_address_context", "changed_source_context"):
        rows = [row for row in replay if row["mode"] == mode]
        both = sum(
            row["scores"]["fact_answer_correct"] and baseline[row["family"]]["scores"]["fact_answer_correct"]
            for row in rows
        )
        changed = sum(row["scores"]["fact_answer_correct"] for row in rows)
        paired[mode] = {"both_answers_correct": both, "changed_answer_correct": changed, "denominator": len(rows)}
        assert both == result["paired_counterfactuals"][mode]["both_answers_correct"]["numerator"]
        assert changed == result["paired_counterfactuals"][mode]["changed_answer_correct"]["numerator"]
    assert before == {name: tensor_digest(tensor) for name, tensor in model.state_dict().items()}
    assert splits == original_splits and icl_splits == original_icl
    assert all(row["identical_ids"] for row in replay)

    body = (ART / (PREFIX + "section_A.2.md")).read_text()
    code = re.search(r"```python\n(.*?)```", body, re.S)[1]
    exercise_outputs = {}
    for label, snippet in (("original", code), ("red_address_exercise", code.replace("青街8號", "紅街9號"))):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(compile(snippet, "A.2.md", "exec"), {})
        exercise_outputs[label] = buffer.getvalue()
        assert len(buffer.getvalue().splitlines()) == 3
    assert exercise_outputs["original"].splitlines()[-1] == "由資料確定的答案 青街8號"
    assert exercise_outputs["red_address_exercise"].splitlines()[-1] == "由資料確定的答案 紅街9號"
    execution = {
        "reviewer_task": "/root/fact_finish_a_2",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": str(torch.get_num_threads()),
        },
        "source_snapshot_sha256": digest(ART / (PREFIX + "section_A.2.md")),
        "code_inventory": code_inventory,
        "model_origin": inventory["origin"],
        "model_config": payload["config"],
        "model_parameters": result["model"]["parameters"],
        "model_tensor_hashes_before": before,
        "model_tensor_hashes_unchanged": True,
        "split": split_summary,
        "icl_split": icl_summary,
        "training_arithmetic": {
            "seed": seed,
            "steps": 1000,
            "batch_size": 24,
            "learning_rate_from_inspected_fit_lm_default": 0.003,
            "rag_train_records": 768,
            "icl_train_records": 126,
            "records": len(examples),
            "effective_target_tokens": sum(step_targets),
            "label_dtype": "torch.int64",
            "ignore_index": IGNORE,
            "assistant_eos_included": True,
            "optimizer_updates_in_this_audit": 0,
        },
        "metrics": metrics,
        "paired": paired,
        "replayed_rag_records": len(replay),
        "identical_formal_cpu_raw_id_records": sum(r["identical_ids"] for r in replay),
        "rag_input_token_range": [min(len(r["input_ids"]) for r in replay), max(len(r["input_ids"]) for r in replay)],
        "exercise_outputs": exercise_outputs,
        "limits": "No retraining; original L4 timing not replayed or validated by this CPU inference/arithmetic check. Training steps are checked against stored formal logs and inspected loop; token denominator is reconstructed exactly from seeded sampling.",
    }
    save(args.label + "_generations.json", replay)
    path = save(args.label + "_execution.json", execution)
    print(
        json.dumps(
            {
                "output": str(path.relative_to(ROOT)),
                "identical_raw_ids": f"{sum(r['identical_ids'] for r in replay)}/84",
                "training_targets_recomputed": sum(step_targets),
                "paired": paired,
                "metrics": metrics,
                "exercise_outputs": exercise_outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
