# 13.16 independent hand calculation and scope checks

The literal example has two fixed Unicode strings. `3` has one character.
`我 已 經 仔 細 思 考 過 ， 答 案 是 4 。` contains 14 characters.
Therefore the lengths are [1, 14], and argmax chooses index 1. The independent
answer for 1+2 is the exact string `3`; index 1 fails both this equality and the
only-number format. No policy was trained or sampled in this example.

Changing the scores to int(answer == str(1+2)) yields [1, 0]: the first equality
is true, the second false. Argmax now selects index 0, response `3`. This is
correct for these exact fixed candidates; different valid explanations cannot
generally be assessed by equality to a single number string.

The test families are pair:1:1, pair:1:2, pair:1:7, pair:2:8, pair:6:10,
pair:7:8. Each has three conditions, so 6*3=18 contexts, six per condition.
Each number and explain context has all C(4,2)=6 ordered preference comparisons;
each missing-quantity context has only the three clarification-vs-other pairs.
The reward-model denominator is consequently 6*(6+6+3)=90 comparisons, not 90
answers and not 90 independent families. The full-request denominator is 18
greedy policy choices. SFT passes 6+0+0=6; PPO and DPO pass 6+0+6=12 each.
Train has 44 families and 44 explain contexts; all three policies choose the
short answer in all 44. This describes training-side failure, without identifying
which hyperparameter, optimization issue or objective term caused the failure.

For test pair:1:2/explain, raw RM scores are [8.767887115478516,
13.556549072265625, -2.3207294940948486, 3.4358766078948975]. Rounding the
first two to four decimal places gives 8.7679 and 13.5565. The sentence wins
their comparison because 13.556549072265625-8.767887115478516 is positive.
PPO probabilities are [0.7928321957588196, 0.2014310657978058,
0.0046974108554422855, 0.0010393551783636212]; the first two round to 0.7928
and 0.2014. Their maximum is index 0, `3`; it has correct arithmetic content
but fails the explanation request. The published rounded values use an absolute
rounding tolerance of 0.00005. Checkpoint recomputation gave exactly the stored
float32 values in this environment; a fresh fixed training run gave identical
model state fingerprints and every recorded evaluation row.

The original configuration has seed 42 and CPU threads 2. SFT has 60*32=1,920
demonstration draws from 44 distinct examples, RM has 300*64=19,200 draws from
660 distinct pairs, PPO collects 120*64=7,680 actions and reuses each three
times, giving 23,040 update reads and 120*3=360 policy and critic updates.
DPO has 360*64=23,040 pair draws. These units differ: policy update parity is
not data or compute parity. There are no autoregressive answer tokens, so the
experiment's effective_tokens=0 is consistent with nonzero bandit training.

The original 2.122668108 seconds is one main() wall-time observation, with
0.657001449 seconds inside the PPO rollout/update loop. It includes DPO as
well as SFT, RM, evaluation and checkpoint saves, and excludes installation.
The new replay timing is separately stored in fact_v2_13_16_audit.json; neither
observation is a warmed repeated throughput benchmark or a general speed claim.

PPO paper v2 Eq.7 clips a surrogate formed from reward-derived advantage.
DPO paper v3 Eq.3 adds a policy/reference KL penalty to expected learned reward;
Eq.7 uses labeled relative preferences. Neither formula contains an independent
truth oracle or changes an incorrect preference label. At an unchanged policy
ratio=1 and KL=0, even an incorrect positive advantage still rewards increasing
its action probability. This supports the limited statement that reference/KL
and clipping cannot automatically correct a misspecified reward standard.

All 825 stored pair comparisons across 165 contexts agree with the fixed rules.
The formal run supplies no observed learned-RM exploitation: its explanation
failures select a lower-scoring card despite RM preferring the independently
correct card. This is a finite four-card outcome under one seed, not a universal
claim that exploitation is absent on other prompts, models or rewards.
