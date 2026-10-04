"""檢查偏好／PPO 的數學與採样邊界，不把訓練收斂寫成單元測試。"""

import pytest
import torch

from scripts.course_experiments.posttraining import build_records, clipping_probe, split_records
from tiny_perceptron.posttraining import bandit_advantage, exact_kl, ppo_clipped_objective, preference_loss


def test_bradley_terry_is_shift_invariant_and_prefers_larger_winner():
    chosen = torch.tensor([2.0, 0.5], requires_grad=True)
    rejected = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = preference_loss(chosen, rejected)
    torch.testing.assert_close(loss, preference_loss(chosen + 19, rejected + 19))
    loss.backward()
    assert (chosen.grad < 0).all() and (rejected.grad > 0).all()


@pytest.mark.parametrize(
    "advantage,ratio,gradient", [(1.0, 1.3, 0.0), (1.0, 0.7, -1.0), (-1.0, 0.7, 0.0), (-1.0, 1.3, 1.0)]
)
def test_clipping_removes_only_the_apparent_improvement_incentive(advantage, ratio, gradient):
    probability_ratio = torch.tensor([ratio], requires_grad=True)
    terms = ppo_clipped_objective(probability_ratio.log(), torch.zeros(1), torch.tensor([advantage]))
    terms["policy_loss"].backward()
    assert float(probability_ratio.grad) == pytest.approx(gradient)


def test_old_policy_and_advantage_are_fixed_sampling_records():
    new = torch.tensor([-1.0], requires_grad=True)
    old = torch.tensor([-1.1], requires_grad=True)
    advantage = torch.tensor([0.5], requires_grad=True)
    ppo_clipped_objective(new, old, advantage)["policy_loss"].backward()
    assert new.grad is not None and old.grad is None and advantage.grad is None


def test_bandit_advantage_does_not_backpropagate_into_reward_or_critic():
    advantage = bandit_advantage(
        torch.tensor([1.0, 0.0], requires_grad=True), torch.tensor([0.4, 0.4], requires_grad=True)
    )
    torch.testing.assert_close(advantage, torch.tensor([0.6, -0.4]))
    assert not advantage.requires_grad


def test_exact_kl_has_the_declared_direction_and_frozen_reference():
    logits = torch.tensor([[0.8, 0.2]]).log().requires_grad_()
    reference = torch.tensor([[0.5, 0.5]]).log().requires_grad_()
    result = exact_kl(logits, reference)
    expected = 0.8 * torch.log(torch.tensor(1.6)) + 0.2 * torch.log(torch.tensor(0.4))
    torch.testing.assert_close(result, expected.unsqueeze(0))
    result.sum().backward()
    assert logits.grad is not None and reference.grad is None
    torch.testing.assert_close(exact_kl(logits.detach(), logits.detach()), torch.zeros(1))


def test_all_task_conditions_and_preference_descendants_stay_with_family():
    splits = split_records(build_records())
    owners = {}
    for name, rows in splits.items():
        for row in rows:
            owners.setdefault(tuple(row["operands"]), set()).add(name)
    assert all(len(names) == 1 for names in owners.values())
    assert owners[(1, 2)] == {"test"}
    assert [len(rows) for rows in splits.values()] == [132, 15, 18]
    assert [sum(len(row["preference_pairs"]) for row in rows) for rows in splits.values()] == [660, 75, 90]


def test_same_operands_change_preference_with_request_and_missing_information():
    rows = {row["mode"]: row for row in build_records() if row["family"] == "pair:1:2"}
    assert [rows[mode]["expected_action"] for mode in ("number", "explain", "missing")] == [0, 1, 3]
    assert [0, 1] in rows["number"]["preference_pairs"]
    assert [1, 0] in rows["explain"]["preference_pairs"]
    assert rows["missing"]["candidate_content_or_clarification_correct"] == [False, False, False, True]
    assert rows["missing"]["preference_pairs"] == [[3, 0], [3, 1], [3, 2]]


def test_probe_includes_both_advantage_signs_and_preserves_adverse_direction():
    probes = clipping_probe()
    assert len(probes) == 6
    by_case = {(row["advantage"], row["ratio"]): row for row in probes}
    assert by_case[(1.0, 1.3)]["loss_derivative_wrt_ratio"] == 0
    assert by_case[(-1.0, 0.7)]["loss_derivative_wrt_ratio"] == 0
    assert by_case[(-1.0, 1.3)]["loss_derivative_wrt_ratio"] == pytest.approx(1)


def test_invalid_shapes_and_empty_rollouts_fail_before_silent_broadcast():
    with pytest.raises(ValueError):
        preference_loss(torch.ones(2), torch.ones(1))
    with pytest.raises(ValueError):
        ppo_clipped_objective(torch.ones(2), torch.ones(2), torch.ones(1))
    with pytest.raises(ValueError):
        ppo_clipped_objective(torch.empty(0), torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError):
        exact_kl(torch.ones(2, 4), torch.ones(2, 3))
