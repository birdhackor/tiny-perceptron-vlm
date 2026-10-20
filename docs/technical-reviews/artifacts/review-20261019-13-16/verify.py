"""重算有限候選實驗紀錄與課文短程式；不重新訓練。"""

import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments.posttraining import CONFIG, build_records, split_records
from scripts.course_experiments.common import records_sha256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tolerance=1e-7):
    assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


def main():
    report_path = ROOT / "docs/course-experiments/results/posttraining.json"
    report = json.loads(report_path.read_bytes())
    result = report["results"]
    assert report["device"] == "cpu" and report["seed"] == 42
    assert result["config"] == CONFIG
    code_hashes = {path: sha(ROOT / path) for path in report["code_sha256"]}
    assert code_hashes == report["code_sha256"]
    assert result["schedule_completed"] is True
    assert result["reference_state_sha256_before"] == result["reference_state_sha256_after"]
    assert result["ppo"]["initial_state_sha256"] == result["reference_state_sha256_before"]
    assert result["dpo"]["initial_state_sha256"] == result["reference_state_sha256_before"]
    splits = split_records(build_records(), report["seed"])
    aggregate = {}
    families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
    assert all(not families[a] & families[b] for a, b in (("train", "test"), ("train", "validation"), ("test", "validation")))
    sample = None
    for name, generated in splits.items():
        ev = result["evaluations"][name]
        rows = ev["rows"]
        assert len(rows) == len(generated) == ev["contexts"]
        assert records_sha256(generated) == result["splits"][name]["sha256"]
        counts = Counter()
        pair_counts = Counter()
        pair_wins = 0
        rm_agrees = 0
        success = {policy: Counter() for policy in ev["policies"]}
        failed_below_rm_best = Counter()
        for row, template in zip(rows, generated, strict=True):
            assert all(row[key] == value for key, value in template.items())
            mode = row["mode"]
            a, b = row["operands"]
            assert 1 <= a <= b <= 10
            best = {"number": 0, "explain": 1, "missing": 3}[mode]
            if mode == "missing":
                expected_candidates = [str(a + b), f"單價加運費是 {a + b} 元，所以總價是 {a + b} 元。", str(a + b + 1), "請補充購買數量。"]
                pairs = [[3, other] for other in range(3)]
            else:
                expected_candidates = [str(a + b), f"{a} 加 {b} 是 {a + b}。", str(a + b + 1), "請再提供更多資訊。"]
                ranking = [0, 1, 3, 2] if mode == "number" else [1, 0, 3, 2]
                pairs = [[winner, loser] for i, winner in enumerate(ranking) for loser in ranking[i + 1:]]
            assert row["candidates"] == expected_candidates
            assert row["preference_pairs"] == pairs
            assert row["expected_action"] == best
            assert row["candidate_meets_full_request"] == [i == best for i in range(4)]
            scores = row["reward_model_raw_scores"]
            assert len(scores) == 4 and all(math.isfinite(v) for v in scores)
            rm_best = max(range(4), key=scores.__getitem__)
            assert rm_best == row["reward_model_best_action"]
            rm_agrees += rm_best == best
            counts[mode] += 1
            pair_counts[mode] += len(pairs)
            pair_wins += sum(scores[winner] > scores[loser] for winner, loser in pairs)
            for policy, saved in row["policies"].items():
                probs = saved["probabilities"]
                assert len(probs) == 4 and all(math.isfinite(v) and 0 <= v <= 1 for v in probs)
                close(sum(probs), 1, 2e-7)
                chosen = max(range(4), key=probs.__getitem__)
                assert chosen == saved["chosen_action"]
                assert row["candidates"][chosen] == saved["chosen_response"]
                correct = chosen == best
                assert correct == saved["full_request_success"]
                assert saved["content_or_clarification_correct"] == row["candidate_content_or_clarification_correct"][chosen]
                success[policy][mode] += correct
                if not correct:
                    failed_below_rm_best[policy] += scores[chosen] < scores[best]
            if name == "test" and row["operands"] == [1, 2] and mode == "explain":
                sample = row
        for policy, actual in success.items():
            saved = ev["policies"][policy]
            assert saved["greedy_full_request_success"]["numerator"] == sum(actual.values())
            assert saved["greedy_full_request_success"]["denominator"] == len(rows)
            close(saved["greedy_full_request_success"]["rate"], sum(actual.values()) / len(rows))
            for mode, denominator in counts.items():
                entry = saved["by_mode"][mode]
                assert entry["numerator"] == actual[mode] and entry["denominator"] == denominator
                close(entry["rate"], actual[mode] / denominator)
        total_pairs = sum(pair_counts.values())
        assert ev["preference_pairs"] == total_pairs
        assert ev["reward_model_pairwise_accuracy"] == {"numerator": pair_wins, "denominator": total_pairs, "rate": pair_wins / total_pairs}
        assert rm_agrees == len(rows)
        aggregate[name] = {
            "contexts": len(rows), "families": sorted(families[name]),
            "counts_by_mode": dict(counts), "pairs_by_mode": dict(pair_counts),
            "reward_pairwise": {"numerator": pair_wins, "denominator": total_pairs},
            "rm_best_matches_rule": rm_agrees,
            "success_by_policy_and_mode": {p: dict(v) for p, v in success.items()},
            "full_success_by_policy": {p: sum(v.values()) for p, v in success.items()},
            "failed_choices_strictly_below_rule_best_rm_score": dict(failed_below_rm_best),
        }
    assert aggregate["test"]["contexts"] == 18
    assert aggregate["test"]["counts_by_mode"] == {"number": 6, "explain": 6, "missing": 6}
    assert aggregate["test"]["reward_pairwise"] == {"numerator": 90, "denominator": 90}
    assert aggregate["test"]["full_success_by_policy"] == {"sft": 6, "ppo": 12, "dpo": 12}
    assert all(aggregate["train"]["success_by_policy_and_mode"][p]["explain"] == 0 for p in ("sft", "ppo", "dpo"))
    assert aggregate["train"]["counts_by_mode"]["explain"] == 44
    assert sample is not None
    rounded = {
        "rm_sentence": round(sample["reward_model_raw_scores"][1], 4),
        "rm_short": round(sample["reward_model_raw_scores"][0], 4),
        "ppo_short": round(sample["policies"]["ppo"]["probabilities"][0], 4),
        "ppo_sentence": round(sample["policies"]["ppo"]["probabilities"][1], 4),
    }
    assert rounded == {"rm_sentence": 13.5565, "rm_short": 8.7679, "ppo_short": 0.7928, "ppo_sentence": 0.2014}
    answers = ["3", "我已經仔細思考過，答案是4。"]
    length_scores = [len(answer) for answer in answers]
    length_chosen = length_scores.index(max(length_scores))
    equality_scores = [int(answer == str(1 + 2)) for answer in answers]
    equality_chosen = equality_scores.index(max(equality_scores))
    assert length_scores == [1, 14] and length_chosen == 1
    assert answers[length_chosen] != str(1 + 2)
    assert equality_scores == [1, 0] and equality_chosen == 0
    paraphrases = ["3", "答案是3。", "1 加 2 是 3。"]
    exact_matches = [answer == str(1 + 2) for answer in paraphrases]
    assert exact_matches == [True, False, False]
    source = ROOT / "docs/course-review-20261019/freeze-01/sources/13.16.md"
    raw = (ROOT / "course/chapters/13.md").read_bytes()
    match = re.search(rb"(?m)^## 13[.]16 [^\n]*\n", raw)
    assert match is not None
    rest = raw[match.start():]
    following = re.search(rb"(?m)^## ", rest[3:])
    body = rest[:following.start() + 3] if following else rest
    assert body == source.read_bytes()
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-16/verify.py",
        "result": "所有驗證通過；僅重算既有逐題紀錄與短程式，未重新訓練。",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu; 記錄重算，未載入權重", "recorded_python": report["python_version"], "recorded_torch": report["torch_version"]},
        "input_sha256": {str(report_path.relative_to(ROOT)): sha(report_path), str(source.relative_to(ROOT)): sha(source), **code_hashes},
        "recorded_config": result["config"],
        "recorded_training_budgets": {"sft_steps": 60, "reward_steps": 300, "ppo_updates": result["ppo"]["policy_updates"], "dpo_updates": result["dpo"]["policy_updates"], "sft_demonstrations": result["sft"]["demonstrations"]},
        "aggregation": aggregate, "explanation_sample": sample, "rounded_sample": rounded,
        "exercise": {"answers": answers, "length_scores": length_scores, "length_chosen": length_chosen, "length_task_check": answers[length_chosen] == str(1 + 2), "equality_scores": equality_scores, "equality_chosen": equality_chosen, "equality_response": answers[equality_chosen], "paraphrases": paraphrases, "exact_matches": exact_matches},
        "reference_reward_counterexample": {"reference_probabilities": [0.5, 0.5], "rewards": length_scores, "kl_coefficient": 1, "optimum_wrong_probability": 1 / (1 + math.exp(-13)), "derivation": "最大化 E[r] - beta KL(pi||ref) 的解為 pi(a) 與 ref(a) exp(r(a)/beta) 成正比；固定參考與 KL 不會把獎勵的錯誤排序改成任務正確性。"},
        "limits": "單一 seed 的 CPU 有限候選紀錄；沒有重跑訓練、沒有驗證自然語言生成能力，也沒有把未觀察到漏洞外推成不存在漏洞。",
    }
    (ARTIFACT / "verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
