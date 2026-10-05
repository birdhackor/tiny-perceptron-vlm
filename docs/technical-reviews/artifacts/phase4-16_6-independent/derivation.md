# Independent 16.6 derivation

For scalar w and fixed inputs x_i, with a zero target,
L(w) = (sum_i (w*x_i)^2)/N = w^2 (sum_i x_i^2)/N.
dL/dw = 2w (sum_i x_i^2)/N. At w=1, x=[1,2,3], N=3,
L=14/3 and dL/dw=28/3=9.333333333333333...

The split [1] and [2,3] contributes losses 1/3 and 13/3;
its gradients are 2/3 and 26/3. Sum: 28/3.
Giving each micro-batch mean one-half assigns example weights
[1/2,1/4,1/4], yielding 2*(1/2+4/4+9/4)=7.5.
The exercise split [1,2] and [3] with equal micro-batch weights
assigns [1/4,1/4,1/2], yielding 2*(1/4+4/4+9/2)=11.5.

For token sums S_j and positive token counts n_j, the intended mean is
sum_j S_j / sum_j n_j = sum_j [n_j/sum_k n_k] * (S_j/n_j).
The arithmetic mean of micro-batch means is sum_j (S_j/n_j)/m.
Equal counts guarantee equality of these weightings. Unequal counts
need not produce distinct scalar values for every dataset: all eleven
losses equal to 1, split into counts 1 and 10, give both means equal to 1.
The intended token weights still differ from equal micro-batch weights.
Linearity of differentiation applies when evaluating all pieces at the
same parameters and when the per-example computation is preserved.

Reduction axes in the scalar fence: x is a one-dimensional example axis;
square is elementwise, sum/mean reduce all its elements, and backward
starts from a scalar. The token variant has logits [batch,sequence,vocabulary]
and labels [batch,sequence], flattened to [batch*sequence,vocabulary]
and [batch*sequence] respectively for cross entropy. IGNORE=-100 positions
contribute neither direct target loss nor direct logits gradient.

Existing empirical units: milliseconds = seconds*1000;
MiB = bytes/2^20. Training additional allocated peak = full allocated
peak minus that variant's pre-training allocated baseline. This is a
CUDA tensor allocator measurement, not total device memory or an isolated
deployment weight size. The original median calculation excludes the first
three training steps. Its individual latency observations are not retained
in the inspected raw JSON, so this audit checks the saved median's units
and rounding rather than recreating its observations.
