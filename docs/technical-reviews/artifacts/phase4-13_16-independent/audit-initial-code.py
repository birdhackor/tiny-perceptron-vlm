"""Independent bounded audit: saved measurements only; no model training or weights."""
import ast
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path

A = Path(__file__).resolve().parent
original = A / "posttraining-original.json"
data = json.loads(original.read_bytes())
public = json.loads((A / "posttraining-public-1df3353.json").read_bytes())
evaluations = data["results"]["evaluations"]
source = A / "posttraining-original.py"
tree = ast.parse(source.read_bytes())
namespace = {"random": random}
selected = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"MODES", "RESERVED_TEST_FAMILY"} for t in node.targets):
        selected.append(node)
    if isinstance(node, ast.FunctionDef) and node.name in {"rule_best_action", "build_records", "split_records"}:
        selected.append(node)
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
generated = namespace["split_records"](namespace["build_records"](), data["seed"])
pointers = ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/code_sha256"]
result = {"source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "public_sha256": hashlib.sha256((A / "posttraining-public-1df3353.json").read_bytes()).hexdigest(), "recorded_revision": data["revision"], "public_commit": "1df335318bda03fd771807f66976953231d5a00b", "environment": {"python": platform.python_version(), "device": "cpu; JSON arithmetic, no weights"}, "splits": {}}
all_families = {}
row_fields = ["family", "operands", "mode", "prompt", "candidates", "features", "preference_pairs", "expected_action", "candidate_content_or_clarification_correct", "candidate_meets_full_request", "reward_model_raw_scores", "policies"]
for split in ["train", "validation", "test"]:
    evaluation = evaluations[split]
    rows = evaluation["rows"]
    assert len(rows) == len(generated[split]) == evaluation["contexts"]
    all_families[split] = {row["family"] for row in rows}
    expected = {"number": 0, "explain": 1, "missing": 3}
    counts = Counter(row["mode"] for row in rows)
    wins = pairs = 0
    successes = {name: Counter() for name in ["sft", "ppo", "dpo"]}
    for index, (row, built) in enumerate(zip(rows, generated[split], strict=True)):
        for field in row_fields:
            pointers.append(f"/results/evaluations/{split}/rows/{index}/{field}")
        for field in row_fields[:11]:
            assert row[field] == built[field], (split, index, field)
        ideal = expected[row["mode"]]
        assert row["expected_action"] == ideal
        assert row["candidate_meets_full_request"] == [i == ideal for i in range(4)]
        # The finite explanation contract is equation plus result, not semantic depth.
        if row["mode"] == "explain":
            a, b = row["operands"]
            assert row["candidates"][ideal] == f"{a} 加 {b} 是 {a + b}。"
        rewards = row["reward_model_raw_scores"]
        for winner, loser in row["preference_pairs"]:
            pairs += 1
            wins += rewards[winner] > rewards[loser]
        assert max(range(4), key=rewards.__getitem__) == ideal
        for name in ["sft", "ppo", "dpo"]:
            p = row["policies"][name]
            probs = p["probabilities"]
            assert len(probs) == 4 and all(0 <= value <= 1 for value in probs)
            assert abs(sum(probs) - 1) < 2e-7
            chosen = max(range(4), key=probs.__getitem__)
            assert chosen == p["chosen_action"]
            assert p["chosen_response"] == row["candidates"][chosen]
            good = chosen == ideal
            assert good == p["full_request_success"]
            successes[name][row["mode"]] += good
    assert pairs == evaluation["preference_pairs"]
    assert evaluation["reward_model_pairwise_accuracy"]["numerator"] == wins
    assert evaluation["reward_model_pairwise_accuracy"]["denominator"] == pairs
    for name in successes:
        summary = evaluation["policies"][name]
        assert summary["greedy_full_request_success"]["numerator"] == sum(successes[name].values())
        assert summary["greedy_full_request_success"]["denominator"] == len(rows)
        for mode in counts:
            assert summary["by_mode"][mode]["numerator"] == successes[name][mode]
            assert summary["by_mode"][mode]["denominator"] == counts[mode]
    # Read-only comparison of raw samples and measurement summaries at the linked immutable version.
    public_eval = public["results"]["evaluations"][split]
    for index, row in enumerate(rows):
        for field in row_fields:
            assert row[field] == public_eval["rows"][index][field], ("public mismatch", split, index, field)
    for field in ["contexts", "preference_pairs", "reward_model_pairwise_accuracy", "policies"]:
        assert evaluation[field] == public_eval[field]
        pointers.append(f"/results/evaluations/{split}/{field}")
    result["splits"][split] = {"contexts": len(rows), "families": len(all_families[split]), "counts_by_mode": dict(counts), "reward_pairwise": [wins, pairs], "policy_successes": {name: {mode: [successes[name][mode], counts[mode]] for mode in counts} for name in successes}}
for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not all_families[first] & all_families[second]
assert [len(all_families[s]) for s in ["train", "validation", "test"]] == [44, 5, 6]
sample_index, sample = next((i, row) for i, row in enumerate(evaluations["test"]["rows"]) if row["family"] == "pair:1:2" and row["mode"] == "explain")
values = {"sentence_reward": sample["reward_model_raw_scores"][1], "short_reward": sample["reward_model_raw_scores"][0], "ppo_short_probability": sample["policies"]["ppo"]["probabilities"][0], "ppo_sentence_probability": sample["policies"]["ppo"]["probabilities"][1]}
for key, rounded in [("sentence_reward", 13.5565), ("short_reward", 8.7679), ("ppo_short_probability", 0.7928), ("ppo_sentence_probability", 0.2014)]:
    assert abs(values[key] - rounded) <= 0.00005
result["sample"] = {"pointer": f"/results/evaluations/test/rows/{sample_index}", **values, "chosen_response": sample["policies"]["ppo"]["chosen_response"]}
result["scope"] = "Recomputed existing raw samples; no training, replayed model evaluation, or new model performance. All observed reward winners agree with fixed task rules; failed policies choose a lower-scoring short answer on explanation rows. This does not rule out reward hacking outside this finite evidence."
result["inspected_json_pointers"] = pointers
(A / "measurement-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "inspected_json_pointers"}, ensure_ascii=False, indent=2))
print("PASS: raw sample recomputation, saved aggregates, disjoint family split, and immutable public measurement equality")
