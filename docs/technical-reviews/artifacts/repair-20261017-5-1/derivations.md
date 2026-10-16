# 5.1 independent transparent checks

Reviewer task: `/root/repair_tech_5_1`. Source: frozen `5.1.md`, SHA-256
`6061c991835eb4cd66d4b3397a75b83f42c36c890ec7aa38ff14f2b571e61637`.

## Hand calculation: the explicit local character IDs

The lesson defines `。→0, 狗→1, 看→2, 貓→3, ，→4`. Thus the four-character
training string `狗看貓，` is `[1,2,3,4]`. Shift exactly once:

| Position | Input | Desired next character |
| --- | --- | --- |
| 0 | 狗 / 1 | 看 / 2 |
| 1 | 看 / 2 | 貓 / 3 |
| 2 | 貓 / 3 | ， / 4 |

Therefore `x=[[1,2,3]]`, `y=[[2,3,4]]`. Three labels differ from `IGNORE=-100`,
so `masked_loss = summed cross entropy / 3`. This is a local explicit ID
convention; the different cat/dog row ordering in prerequisite 2.1 does not
change this calculation. No automatic tokenizer or byte encoding is used here.

## Definitions separately evaluated by the bounded program

For real 5×8 gradient matrix G, the displayed default Frobenius norm is
`sqrt(sum(i=0..4,j=0..7, G[i,j]^2))`. The scalar calculated independently from
the actually observed 40 entries is 0.3323640354617846; the float32 `.norm()`
result is 0.332364022731781 (absolute difference < 1e-7). Positivity establishes
a nonzero derivative of this loss with respect to this table. It does not
establish a parameter update or behavior on a new input. `micro_check.json`
records the separately observed backward/update boundary and fixed-input loss.

The AdamW recurrence with weight_decay=0 used in the two-scalar-step check is
`m_t=.9*m_(t-1)+.1*g_t`, `v_t=.999*v_(t-1)+.001*g_t^2`,
`theta_t=theta_(t-1)-.03*(m_t/(1-.9^t))/(sqrt(v_t/(1-.999^t))+1e-8)`.
Start theta=1, m=v=0 and explicitly supply gradients 2 then -1. Hand substitution
gives first m=.2, v=.004 and theta≈.97000000015. Second m=.08, v=.004996,
theta≈.9620098890107295, still decreasing despite the changed gradient sign.
These prescribed gradients illustrate history; they are not a loss-optimization
experiment. The program's float64 result agrees to <1e-14.

## Historical result arithmetic, without replaying training

The existing raw result's validation sum is 33.69866943359375 over 35 effective
targets. Division gives 0.9628191266741072 and rounding to five decimal places
gives 0.96282. The existing train means 5.792432553300234 and
0.0635230875460901 round to 5.79243 and 0.06352. The fresh input-only regeneration
matches all three saved split SHA-256 values and shows 321 training targets
per full re-evaluation, distinct from 342462 sampled targets over 600 updates.
