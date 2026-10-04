# 13.10 independent derivation

Read the actual DPO arXiv:2305.18290v3 PDF, p.3, equations (1)–(2), and p.5, Definition 1 / Lemma 1. With the same prompt x and reward scores c=r(x,y_w), d=r(x,y_l), Bradley–Terry assumes

P(y_w ≻ y_l | x) = exp(c)/(exp(c)+exp(d)) = 1/(1+exp(d-c)) = σ(c-d).

For this lesson's explicit toy prompt, 1+2=3. Both `3` and `1+2等於3。` state the same result; only `3` satisfies the instruction to return a number only. This label follows the task rule, not a universal rule against explanations. Equation (1) conditions both candidates on the same x; its probability model does not itself decide this task's label.

For an observed chosen winner the negative log-likelihood is

L(c,d) = -ln σ(c-d) = ln(1+exp(-(c-d))).

A batch of N labeled comparisons has L_batch = (1/N) Σ_i L(c_i,d_i), as implemented by the elementwise `logsigmoid` followed by `.mean()`. Let Δ=c-d. Then ∂L/∂c=σ(Δ)-1<0 and ∂L/∂d=1-σ(Δ)>0 for finite real scores. Increasing the gap decreases the loss. This is a statement about scores in this loss; one neural-network parameter update can also affect other comparisons.

Numerical substitutions (natural logs; rounded to four decimals):

| c | d | Δ | exp(-Δ) | P | L |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 0.5000000000 → 0.5000 | ln 2 = 0.6931471806 → 0.6931 |
| 2 | 0 | 2 | 0.1353352832 | 0.8807970780 → 0.8808 | ln(1.1353352832) = 0.1269280110 → 0.1269 |
| 0 | 1 | -1 | 2.7182818285 | 0.2689414214 → 0.2689 | ln(3.7182818285) = 1.3132616875 → 1.3133 |
| 2 | 1 | 1 | 0.3678794412 | 0.7310585786 → 0.7311 | ln(1.3678794412) = 0.3132616875 → 0.3133 |

For a prompt-dependent common offset k(x), (c+k)-(d+k)=c-d, so P and L are unchanged. A raw score 3 therefore cannot determine an absolute correctness probability: it can be shifted to 13 with identical comparisons. Multiplying every score by a non-unit factor generally changes the probability; no scale-invariance claim is made. No absolute factual correctness objective occurs in these formulas.

The CPU audit independently recomputes all values with Python `math` and float64 PyTorch. The lesson itself executes with default float32 tensors and prints four decimals. Float64 analytic agreement uses absolute tolerance 1e-15; float32 four-decimal outputs are checked literally. The common integer +10 offset is exactly representable in these toy inputs, and equal outputs are observed. This is not a claim of bitwise equality for arbitrary floating-point values or large offsets. The batch mean for gaps [-1,0,1,2] is the arithmetic mean of the four losses above.

Experiment denominators are distinct: 55 canonical families × three modes = 165 contexts. Each number/explain context gives six comparisons from a four-card full ranking, while each missing-quantity context gives three comparisons with clarification beating the three unsupported totals. Thus each family gives 6+6+3=15 labeled comparisons; 44/5/6 families give 660/75/90 train/validation/test comparisons and 132/15/18 contexts. Training makes 300 updates × 64 sampled pairs = 19,200 pair draws, with replacement, from 660 unique train pairs. Each comparison has one effective binary preference target; there are zero token targets. The model's 241 parameters are (4 context features+4 one-hot candidate features)×24+24 +24×1+1 = 241. Context features and card identities, not candidate strings, are input to the network.

Scope: pairwise test accuracy means comparison agreement in this fixed synthetic ranking task, using strict score comparison. It does not measure Chinese reading, arithmetic, open-ended generation, calibrated truth probability, or human preference quality. All contexts are canonical a≤b pairs; swapped operands are not a separate generalization evaluation. Family separation prevents cross-split membership of a canonical pair.
