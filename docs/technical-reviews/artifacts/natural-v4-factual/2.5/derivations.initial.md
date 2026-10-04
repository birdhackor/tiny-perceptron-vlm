# Independent factual derivations for 2.5

Reviewer: `/root/v4_review_coordinator/factual_v4_2_5`, fresh factual-only task.

## Visible information and identifiability

The original two strings have five characters: 紅/色/物/體/是 and 藍/色/物/體/是. For a positive C, Python's `prefix[-C:]` starts at `len(prefix)-C`, ends at the end, and uses the default forward step of 1. Thus C=1 gives 是; C=3 gives 物體是; C=4 gives 色物體是; C=5 includes the distinguishing first character. The actual source block and all four cases were executed by `probe.py`.

Fix the model parameters and suppose the model receives only this visible suffix, with no other variable carrying the color. If both examples map to the same visible input v, a deterministic function g(v) has the same output for both. A conditional probability model also has one q(.|v) for both. If we choose the two examples equally often, a color-independent randomized prediction is correct with probability (q(圓|v)+q(方|v))/2, at most 1/2. Randomness cannot correlate a choice with color that it never receives. Distinct inputs at C=5 remove this obstruction; they do not prove the model will learn the required association or generalize.

This proof concerns just the two prefix/answer examples. It is not a loss bound for the later 12-document training experiment, whose targets cover many different positions and characters.

The exercise strings have eight characters: 紅/色/的/小/小/物/體/是 and 藍/色/的/小/小/物/體/是. Their last seven characters coincide, 色的小小物體是; their full eight-character strings differ. The probe checks both counts and slices.

## Dense mixing size and parameter counts

Concatenating C vectors of dimension D produces C*D input coordinates. For H outputs, each output has a row of C*D weights. The matrix therefore has H*C*D entries; optional bias adds H entries, not another H*C*D. The dense forward calculation uses H*C*D coordinate multiplications and corresponding additions per example (embedding lookup and the remaining layers are separate work). At D=2, C=3 gives 6 input coordinates and 6H weights; C=6 gives 12 input coordinates and 12H weights. The first matrix's weights double. This does not assert that total-model parameters, wall-clock time, or hardware memory exactly double.

The project's `ContextMLP` has a vocabulary of V=17 IDs, with D=H=16:

* Embedding: V*D = 17*16 = 272.
* Hidden weights: H*C*D = 16*C*16 = 256C.
* Hidden bias: H = 16.
* Output weights: V*H = 17*16 = 272.
* Output bias: V = 17.

Total = 272 + 256C + 16 + 272 + 17 = 577 + 256C.
At C=1: 577+256=833; C=3: 577+768=1345; C=5: 577+1280=1857. Moving 1 to 3 or 3 to 5 adds 512 parameters each. The probe independently compares these hand calculations with `sum(p.numel())` from constructed models.

## MLP and nonlinearity

Bengio et al. 2003, Section 2, printed pages 1141–1143, describes a shared word-feature table followed by a feed-forward probability network. Its equation (1), with optional direct matrix W=0, is `y=b+U*tanh(d+H*x)`, where x concatenates the context's word features. This directly supports the conceptual architecture here. The project implements embedding -> flatten -> hidden Linear -> tanh -> output Linear; it emits logits and `cross_entropy` applies the relevant normalized loss. The paper's word units, optimization, data and experiments are different from the project's character experiment.

For the explicit 2.4 prerequisite's mechanism, composing two affine maps without a nonlinearity gives B(Ax+a)+b=(BA)x+(Ba+b). The specified two-feature hidden representation (x,-x), with ReLU followed by sum, yields max(0,x)+max(0,-x)=|x|; at x=-2,0,2 this gives 2,0,2. An affine kx+b matching the middle point requires b=0; matching the two outer points would require k=-1 and k=1 simultaneously. This is a transparent example of representational difference, not evidence that any training run succeeds.

## Mean next-character loss

For logits z and a class-index target y, p_y=exp(z_y)/sum_j exp(z_j), and negative log probability is log(sum_j exp(z_j))-z_y. For N equally weighted nonignored targets, mean loss is L=(1/N)*sum_i[-log(p_yi)]. Thus exp(-L) is the geometric mean of the correct targets' probabilities. A lower L means a higher geometric mean; it need not increase every individual target probability, arithmetic mean probability, or argmax accuracy. This is how the section's average-probability statement is read.

The current original run code averages over all target positions and includes one end marker for each document, rather than averaging per-document losses. In the independently reconstructed seed-42 split, the 9 training documents contain 103 targets, the 1 validation document contains 11, and the 2 test documents contain 22. The actual split includes four eleven-character 三角 strings and five ten-character strings, so the hand count is 4*(11+1)+5*(10+1)=48+55=103. Validation has 10+1=11; both test strings have 10+1=11, giving 22.

The original stored post-update losses, rounded to five decimals, are:

| C | train | validation |
|---|---:|---:|
| 1 | 0.33352 | 0.43103 |
| 3 | 0.21302 | 0.30577 |
| 5 | 0.19883 | 0.51252 |

Using the displayed values: train change 1->3 is -0.12050; validation change is -0.12526. Train change 3->5 is -0.01419; validation change is +0.20675. These directions also hold for unrounded original values and the fresh CPU rerun. Same seed, split and width do not equalize parameter count: C=5 has 1024 more total parameters than C=1. One validation document and one seed support only this particular comparison, not a general best window.

## Evidence scope

`probe-results.json` distinguishes the existing project record's CPU run (torch 2.14.1+cu126, Python 3.13.3) from this reviewer's own CPU rerun (torch 2.14.1+cpu, Python 3.13.5). The probe performs all 200 updates for each of the three tiny MLPs, with seed reset per model, full training batches, AdamW lr=0.01/default weight decay=0.01, and gradient norm clipping at 1.0. It uses float32 parameters and long targets, two CPU threads, no target masking, no class weighting and no label smoothing. For every target it independently recomputes the stable scalar log-sum-exp loss in Python float64 and checks the mean against PyTorch within 2e-6. The existing record is read, not represented as this reviewer's historical execution. This review makes no timing, GPU or natural-language quality claim.
