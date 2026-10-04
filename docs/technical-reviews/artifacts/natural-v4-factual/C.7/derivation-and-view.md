# C.7 independent derivation and personally viewed figure

Reviewer: `/root/v4_review_coordinator/factual_v4_c_7`; current section SHA-256
`ce058edc74867d084607cceefd6789c1be5a6b2eba479b878f9bf263bf9b3a72`.

For free scores z, p_i = exp(z_i)/sum_j exp(z_j), log p_a =
z_a - log sum_j exp(z_j). Consequently d log p_a/d z_i = 1[i=a]-p_i.
For the fixed scalar advantage A=r-b, loss L=-A log p_a has gradient
A(p_i-1[i=a]). At z=[0,0], p=[0.5,0.5], a=1 and A=0.5, this is
[0.25,-0.25]. Subtracting 0.1 times the gradient gives [-0.025,0.025].
p_1'=1/(1+exp(-0.05))=0.5124973964842103; p_0'=0.4875026035157897.
The selected score increases by 0.025; its probability increases by
0.0124973964842103. These are different units. Float32 execution differs
from the double formula by at most 3.1009509404711366e-8 and prints the
six-decimal values stated in the lesson.

With only reward changed to 0, A=-0.5, gradient=[-0.25,0.25], updated
scores=[0.025,-0.025], and selected p_1'=0.4875026035157897. A nonnegative
raw reward is therefore insufficient to determine the direction; the
comparison with the baseline determines this particular update.

For a sampled action and a baseline independent of that current action,
E[b grad log p_a] = b sum_a p_a grad log p_a = b grad sum_a p_a = 0.
This preserves the expected score-function gradient. It does not ensure
lower variance for every baseline. In the initial two-action example,
with reward 0 for action 0 and 1 for action 1, b=0 gives update-estimator
values [0,0] and [-0.5,0.5], mean [-0.25,0.25] and variance 0.0625 per
coordinate. b=0.5 gives [-0.25,0.25] for both actions, the same mean and
zero variance. The executed enumeration confirms this limited example.

Three operands have six categories each, so three concatenated one-hot
vectors have 3*6=18 features. The allowed sums cover 0..15 (16 values),
and enumeration adds one action, giving 17. H(p)=-sum p_i log p_i.
At uniform p_i=1/17, H=log(17)=2.833213344056216. Because
H=log(17)-KL(p||uniform), and KL>=0, this is the maximum, attained only
at uniform p. At a point mass H=0 using the continuous convention
0 log 0=0. For a loss minimized by gradient descent, subtracting 0.005H
rewards entropy; the uniform contribution is -0.014166066720281081.
The entropy term supplies no answer-correctness guarantee.

The fixed schedule draws 1,200*64=76,800 actions per branch, while the
evaluation draws 24*16=384 actions across only 24 distinct questions.
The logged final strict training reward 0.609375 is exactly 39/64.

I rendered the current SVG with Inkscape and personally viewed
`policy-update-render.png` through `view_image`. The two horizontal arrows
run from original scores, through gradient subtraction, to updated scores.
The left and right downward arrows independently apply softmax to their
respective score lists. Neither is a probability-space update arrow. Both
lists retain answer 3 then answer 4. All gradients, learning rate, score
values and probability values match the derivation. Text labels are visible
and the rendered layout contains no numeric or directional contradiction.
No necessary prerequisite section read for this review references an SVG.

The exact fixed-record audit validates all 1,536 recorded sample actions
(two before evaluations and two after evaluations). The before records
are identical, hence they are not independent extra experimental evidence.
Existing exported-checkpoint forwards validate final probabilities on CPU;
the maximum CPU/L4 absolute differences are 5.066394805908203e-7 (strict)
and 2.091837814077735e-11 (weak). This is not a new GPU training or timing
replication. Only one actual repo policy update per branch was executed.
