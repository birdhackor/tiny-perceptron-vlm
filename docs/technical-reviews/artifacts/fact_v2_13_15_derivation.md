# 13.15 independent hand calculations and figure inspection

Reviewer: /root/integration_technical_coordinator/fact_v2_13_15. Date: 2026-10-04.

These are hand calculations, separate from the CPU executions in fact_v2_13_15_audit.json.

- With integers 1 through 10 and unordered pairs allowing equality, the number of families is 10 + 9 + ... + 1 = 10*11/2 = 55. The split 44 + 5 + 6 = 55 keeps entire families together. Three conditions per family give 132, 15, and 18 contexts.
- A dense layer from d inputs to h outputs has d*h weights and h biases. Policy: 4*16 + 16 + 16*4 + 4 = 148. Reward: (4 context features + 4 candidate-identity features)*24 + 24 + 24*1 + 1 = 241. Its 4-by-4 identity buffer has no trainable parameters. Critic: 4*16 + 16 + 16*1 + 1 = 97.
- Each number/explain condition has all unordered comparisons of four ranked cards, 4*3/2 = 6. Missing-quantity condition only has clarification against each other card, 3 comparisons. Each family contributes 6 + 6 + 3 = 15 pairs, so training contains 44*15 = 660 distinct preference pairs.
- SFT: 60 updates * 32 sampled examples = 1,920 example draws from 44 distinct number-only demonstrations. Reward: 300 * 64 = 19,200 pair draws from 660 distinct pairs. Sampling is with replacement; draws are not unique examples.
- PPO: 120 rollout batches * 64 newly sampled actions = 7,680 actions. Three reuse epochs per batch give 120*3 = 360 policy updates and 360 critic updates, and 7,680*3 = 23,040 processed action records. Reuse creates no new action sample.
- For four classes and smoothing epsilon=0.2, the target is (1-epsilon)*one_hot + epsilon/4. The chosen target is 0.8+0.05=0.85 and each other target is 0.05. This supplies an incentive to retain other probabilities; it does not prove a universal probability lower bound throughout training.
- Before the first SGD update, both copies of the policy give identical logits for the same context. Thus log p_new - log p_old = 0, ratio = exp(0) = 1, inside [0.8,1.2]. For KL against an identical reference, each term p_i*(log p_i-log q_i)=0. This is a first-step statement, not a hard constraint on later ratios.
- The demonstrated SGD has zero momentum and weight decay: theta_next = theta - 0.1*gradient. The 0.1 coefficient of KL and 0.5 coefficient of value error change the objective before differentiation; learning rate multiplies the resulting gradients afterwards.
- Greedy full-request success is sum(action==rule_best_action)/18. The original and rerun contain SFT 6/18 (all number cases), PPO 12/18 (all number and missing cases, no explain cases). Evaluation chooses argmax over four probabilities, not a fresh stochastic answer sample.

The target section contains no SVG. I also inspected XML and personally viewed Inkscape renders of its required prerequisites:

- ppo_clip.svg: the common ratio positions follow x=60+250*(r-0.4) on the left and x=450+250*(r-0.4) on the right. Blue positive-advantage curve rises until r=1.2 (x=260), then flattens; orange negative-advantage curve is flat until r=0.8 (x=550), then declines. Dashed lines show the unclipped branches. These positions and directions implement min(r,clip(r)) and min(-r,-clip(r)). No vertical numerical scale is claimed. The text explicitly says probabilities are not locked into the interval. Rendered Chinese labels are legible and do not conflict with the curves.
- ppo_roles.svg: reward learns comparisons before freezing, value learns from observed rewards, old selected log probability remains fixed within a rollout and is recollected next rollout, and reference remains the SFT starting state throughout. This matches the inspected code and original PPO/DPO sources; the actual software stores old log probabilities rather than requiring a separate persistent old-policy network.

Actual rendering commands (both exited 0; only nonfatal GTK/Pango warnings):

```text
inkscape course/figures/ppo_clip.svg --export-type=png --export-filename=docs/technical-reviews/artifacts/fact_v2_13_15_ppo_clip.png
inkscape course/figures/ppo_roles.svg --export-type=png --export-filename=docs/technical-reviews/artifacts/fact_v2_13_15_ppo_roles.png
inkscape --version
```

Observed renderer: Inkscape 1.4 (e7c3feb100, 2024-10-09). Font match: NotoSansCJK-Regular.ttc, Noto Sans CJK TC Regular.
