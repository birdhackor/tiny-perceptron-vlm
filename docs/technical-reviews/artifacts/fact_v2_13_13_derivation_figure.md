# Independent 13.13 calculation and figure inspection

Reviewer: /root/integration_technical_coordinator/fact_v2_13_13
Date: 2026-10-04
Read scope: complete docs/technical-review-guide.md, course/chapters/13.md#13.13,
and explicit prerequisite #13.12. No previous review conclusions or author history read.

## Hand calculation (distinct from the executed CPU result)

For fixed advantage A and epsilon e, let
f(r,A,e) = min(r*A, clamp(r,1-e,1+e)*A).
With A=+1, f=min(r,1+e): below 1-e the unclipped branch wins;
above 1+e the constant clipped branch wins.
With A=-1, f=-max(r,1-e): below 1-e the constant clipped branch wins;
above 1+e the decreasing unclipped branch wins.
Thus clipping only the ratio without the outer minimum discards the unfavorable
direction beyond the interval; it would wrongly stop that correction.

Each entry is an independently selected action from its own sample. These six
entries do not constitute a single probability distribution. Old selected probability
is 0.5; new selected probabilities 0.5*r are 0.35, 0.5, 0.65 in each group,
all in [0,1]. log(p_new)-log(p_old)=log(r), so exp recovers r mathematically;
the executed float32 path has errors around 1e-7, not exact real arithmetic.

For e=0.2, bounds are [0.8,1.2]:

| r | A | r*A | clamp(r)*A | f | derivative of -f w.r.t. r |
|---|---|---|---|---|---|
|0.7|+1|0.7|0.8|0.7|-1|
|1.0|+1|1.0|1.0|1.0|-1|
|1.3|+1|1.3|1.2|1.2|0|
|0.7|-1|-0.7|-0.8|-0.8|0|
|1.0|-1|-1.0|-1.0|-1.0|+1|
|1.3|-1|-1.3|-1.2|-1.3|+1|

Sum f=-0.2, hence the lesson's negative sum is 0.2. The API also returns a
negative mean, 0.2/6, but the lesson deliberately differentiates the negative sum.
The reported gradients are ratio partial derivatives, not network-parameter gradients.

For e=0.1, bounds are [0.9,1.1]:

| r | A | r*A | clamp(r)*A | f | derivative of -f w.r.t. r |
|---|---|---|---|---|---|
|0.7|+1|0.7|0.9|0.7|-1|
|1.0|+1|1.0|1.0|1.0|-1|
|1.3|+1|1.3|1.1|1.1|0|
|0.7|-1|-0.7|-0.9|-0.9|0|
|1.0|-1|-1.0|-1.0|-1.0|+1|
|1.3|-1|-1.3|-1.1|-1.3|+1|

Sum f=-0.4; negative sum is 0.4. None of these six inputs lies on the
piecewise breakpoints, so no boundary derivative convention is being asserted.

## XML and rendered SVG inspection

I read the complete XML of course/figures/ppo_clip.svg and personally inspected
fact_v2_13_13_ppo_clip.png rendered by Inkscape 1.4. I also inspected PPO v2
PDF page 3 rendered in fact_v2_13_13_ppo_page3.png, including formula 7 and Figure 1.
The Chinese font rendered successfully (Noto Sans CJK TC); both graph labels,
the legend and the no-hard-bound caveat are readable.

SVG dimensions: viewBox 0 0 820 470. Left ratio map: x=250*r-40;
right ratio map: x=250*r+350. Every printed tick at r=0.4,0.8,1,1.2,1.6
matches these maps (left x=60,160,210,260,360; right x=450,550,600,650,750).
Dashed vertical boundaries are x=160/260 and x=550/650, hence r=0.8/1.2.

Left value map is y=310-125*f. For A=+1, the blue solid path
M60 260 L260 160 H360 equals f=min(r,1.2): increasing to (r,f)=(1.2,1.2)
then constant to r=1.6. The gray dashed path M60 260 L360 110 equals f=r.

Right value map is y=130-125*f. For A=-1, the orange solid path
M450 230 H550 L750 330 equals f=-max(r,0.8): constant f=-0.8 to r=0.8,
then decreases to f=-1.6 at r=1.6. The gray dashed path M450 180 L750 330
equals f=-r. The right-panel horizontal origin lies above all negative values,
consistent with their signs. No numerical y ticks are claimed by the figure;
the constant 125-pixel scale in both panels still matches the formulas.

Solid and dashed branches coincide where clipping is inactive. Both rendered
panels reproduce the direction of the paper's Figure 1, and the legend's
interpretation is about a per-sample surrogate term, not a bound on probabilities.

## Primary-source boundaries and scope

The independently fetched PPO arXiv:1707.06347v2 (28 August 2017) PDF has SHA
e78feadadbdbb0b601b3c2bcc81404722cd431a489b307545f9b7bea1e8c4f5b,
identical to the candidate PDF. Page 3 formula 7 and its following explanation
justify clipping only favorable changes and retaining unfavorable changes.
Page 5 formula 9 explicitly combines surrogate/value/entropy terms for shared
architectures; Algorithm 1 collects old-policy data, optimizes K epochs,
then updates old policy for the next iteration.

The independently fetched OpenAI train_policy.py at commit
cbfd210bb8b08f6bc5c26878c10984b90f516c66 agrees with the candidate bytes.
Lines 350-354 implement exp(new-old) and max of negative clipped/unclipped
losses, algebraically equal to -min of positive surrogate branches.
Lines 357,360,367 combine a value loss and record approximate KL.
I did not run its historical TensorFlow trainer and make no compatibility or
training-performance claim about it.

The independently fetched OpenAI Spinning Up PPO page, footer revision
038665d6, HTML lines 299-333, states the formula, the positive/negative
cases, and the explicit limitation: policy updates can still move too far;
its implementation stops taking gradient steps when mean KL exceeds a threshold.
This supports monitoring displacement, not a claim that clipping guarantees
improvement or fixes every implementation. The fetched HTML preserves math
in image alt text, unlike the candidate plain-text extraction.

No empirical quality, speed, memory, seed-comparison or long-training result
is asserted in 13.13. The audit checks deterministic CPU math, six valid samples
per example, 121 ratio-grid points per sign, and one shared-scalar counterexample.
It neither validates an entire PPO trainer nor supplies a final-quality guarantee.
