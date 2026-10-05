# 1.7 independent mathematical and numerical check

Reviewer: `/root/phase4_factual_coordinator/factual_1_7`; context: fresh.
Read original current 1.7 in `course/chapters/01.md`, lines 239–273, and complete
preceding 1.6 to confirm that the new candidate order is cat/look/dog (貓/看/狗).
The 1.7 original bytes have SHA-256
`860731c9ecb2f8590e83fa539f4245fa7e162b887917acc7ba0202d2de2dd7ce`.
No old technical report or old review-history conclusions were read.

For a finite real candidate vector z, define w_i=exp(z_i) and D=sum_j w_j.
Every w_i>0 and D>0. Hence p_i=w_i/D is positive and sums to 1. The exponential
is strictly increasing and the same positive denominator is used for each
candidate, so the ranking of scores is preserved. The vector [1,2,-1] itself
is not a probability distribution: a probability cannot be negative, and this
vector sums to 2. These scores, weights and probabilities are dimensionless.

For a shared finite real constant c,

    exp(z_i+c) / sum_j exp(z_j+c)
    = exp(c)*exp(z_i) / (exp(c)*sum_j exp(z_j))
    = exp(z_i) / sum_j exp(z_j).

Choosing c=-max(z) makes every exponent <=0 and one exponent 0, so the
exponential weights are <=1 and at least one equals 1. This avoids exponential
overflow for the finite values used here. It does not imply that every
floating-point shift preserves tiny differences exactly, that underflow is
impossible, or that infinite/NaN inputs are covered. The official v2.9.0 CPU
SoftMaxKernel.cpp implementation, lines 64–94, uses the same max subtraction.

The independent Python math.exp calculation in phase4-1_7-probe.py gives:

| vector | exponential weights | denominator | probabilities |
| --- | --- | --- | --- |
| [1,2,-1] | [2.718281828459045,7.389056098930650,0.367879441171442] | 10.475217368561138 | [0.2594964603424191,0.7053845126982412,0.03511902695933972] |
| [-1,0,-3] | [0.367879441171442,1,0.049787068367864] | 1.4176665095393062 | same probabilities within absolute 1e-12 |
| [3,2,-1] | [20.085536923187668,7.389056098930650,0.367879441171442] | 27.842472463289760 | [0.7213991842739687,0.26538792877224193,0.013212886953789416] |

Displayed three-decimal approximations are checked within absolute 0.0005;
four-decimal approximations within absolute 0.00005 (half of the last displayed
decimal place). In the exercise, look's numerator exp(2) stays fixed while D
increases, so look's probability decreases. No count, token or dataset
denominator is involved: D is the sum over exactly three candidate weights.

In the installed torch float32 CPU execution, the manual p sums to
1.0000001192092896, a rounding difference of 1.1920928955078125e-7. This is the
exact stdout; mathematical normalization and the section's floating-point
allclose explanation supply the context for its prose 'sum 1'. The literal
stdout is retained without changing it to 1. The original default allclose
uses rtol=1e-5 and atol=1e-8, i.e. elementwise abs(a-b)<=atol+rtol*abs(b).
The manual and built-in softmax values in this original run agree exactly;
independent float64 expectations are compared using rtol=1e-6, atol=1e-7.

The original shape is [3]; its only candidate axis is dim=0. For the bounded
variation [2,3], each row is a separate question and dim=-1 (equivalently 1)
is the candidate axis. Manual max/sum reductions also use dim=-1, keepdim=True.
The observed row sums are [1.0000001192092896,0.9999999403953552]. A deliberately
shared global denominator instead gives [0.2733781337738037,0.7266219258308411],
and dim=0 gives row sums [1.1192028522491455,1.880797028541565]. These comparisons
check the stated [batch,candidates] contract; they do not imply that dim=0 is
wrong for a differently arranged input.

For float32 [101,102,99], naive exponentiation produces infinity in all three
positions and its normalized ratios are NaN; built-in softmax remains finite
and agrees with the original scores. This bounded check illustrates overflow
avoidance without making a universal guarantee about all possible floats.

The original source has no targets, sampling, backward call or optimizer; it
constructs z with requires_grad=False and computes new tensors. z is exactly
unchanged in the bounded check and z.grad is None. When dog's score is highest
([1,2,3]), its probability is highest (approximately 0.6652409434), regardless
of which answer would be correct. A normalized distribution is not evidence
of correct predictions, learned parameters or model quality.

Official sources personally inspected: PyTorch v2.9.0 activation.py
Softmax definition/axis (1755–1819), functional.py softmax signature/body
(2096–2139), CPU SoftMaxKernel.cpp (64–155), _torch_docs.py torch.tensor
(9223–9276), torch.exp (4115–4138), torch.max (6598–6667), torch.sum
(10829–10884), torch.allclose (764–795), torch.add (354–389), torch.sub
(10788–10818), torch.div (3880–3929); _tensor_docs.py method aliases exp
(1852–1859), max (3224–3231), sum (5102–5109), softmax (5495–5502), item
(2821–2838); CPython v3.13.5 simple_stmts.rst assert (378–415).

Source version differs from installed PyTorch 2.14.1+cpu (git
5c4886908584029761b579af026dcfb627c84070). The pinned v2.9.0 original official
source supplies mathematical/API contracts, while execution checks behavior
in the actually installed version. The hosted PyTorch 2.9 documentation URLs
returned HTTP 403; the original official repository files containing their
docstrings and implementation were successfully retrieved and read instead.
The helper contract and bootstrap were fully read before execution: a CPU
offline child process, original bytes/sha checks, a shared namespace for
selected Python fences, and no external commands or writes outside the run
directory. No figure is referenced by 1.7, so no render or visual inspection
is applicable. No training, model/data download, upload or GPU work ran.
