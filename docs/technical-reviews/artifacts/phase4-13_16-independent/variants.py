"""Literal section fence, prescribed exercise, and a bounded counterexample to exact match."""
from pathlib import Path
import json
import platform
import math
import torch
from tiny_perceptron.posttraining import ppo_clipped_objective

A = Path(__file__).resolve().parent
code = (A / "fence-1.py").read_text()
ns = {}
exec(compile(code, "original-section-fence", "exec"), ns)
assert ns["scores"] == [1, 14] and ns["chosen"] == 1
assert ns["answers"][ns["chosen"]] != str(1 + 2)
exercise = code.replace("len(answer)", "int(answer == str(1 + 2))")
assert exercise.count("int(answer == str(1 + 2))") == 1
(A / "exercise-literal.py").write_text(exercise)
exercise_ns = {}
exec(compile(exercise, "prescribed-exercise-fence", "exec"), exercise_ns)
assert exercise_ns["scores"] == [1, 0] and exercise_ns["chosen"] == 0
paraphrases = ["1 加 2 是 3。", "把 1 與 2 相加，結果是 3。"]
assert all(answer != str(1 + 2) for answer in paraphrases)
print(json.dumps({"environment": {"python": platform.python_version(), "device": "cpu"}, "literal_scores": ns["scores"], "exercise_scores": exercise_ns["scores"], "exact_match_rejects_valid_open_explanations": paraphrases}, ensure_ascii=False))
# DPO paper Eq. 4: uniform fixed reference, beta=1, rewards [1,14].
# Analytic optimum of reward minus reference KL still favors the wrong answer.
wrong_probability = 1 / (1 + math.exp(-(14 - 1)))
assert wrong_probability > 0.99999
ratio = torch.tensor([1.0], requires_grad=True)
old_log = torch.tensor([0.5]).log()
terms = ppo_clipped_objective(old_log + ratio.log(), old_log, torch.tensor([1.0]), 0.2)
terms["policy_loss"].backward()
assert ratio.grad.item() == -1.0
print(json.dumps({"fixed_reference_bad_reward_optimum": wrong_probability, "beta": 1, "reference_probabilities": [0.5, 0.5], "rewards": [1, 14], "ppo_bad_reward_positive_advantage_ratio_gradient": ratio.grad.item(), "torch": str(torch.__version__), "device": "cpu", "interpretation": "A descent update increases the ratio for the incorrectly rewarded answer; clipping and KL add no task truth checker."}))
