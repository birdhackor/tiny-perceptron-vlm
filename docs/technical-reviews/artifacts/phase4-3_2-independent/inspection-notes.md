# Independent factual inspection of 3.2

Reviewer: `/root/phase4_factual_coordinator/factual_3_2`; fresh context; 2026-10-05.

I read the current `course/chapters/03.md` lines 1–220, including the chapter introduction and 3.1–3.6; this report judges only 3.2, lines 36–68. I also read `course/chapters/01.md` lines 120–260, including the cited 1.7 softmax introduction, `scripts/build_course.py` bootstrap lines 26–61, and the current `tiny_perceptron/attention.py` (attention implementation and roles). I read the complete factual-reviewer instructions, checker schema and extraction/execution helper. No prior technical-report body, conclusions, history, or another reviewer's inspection notes were read. Original external source files and retrieval receipts were reused only as allowed by the task.

## Original sources personally read

The immutable `Attention Is All You Need` arXiv v7 PDF is SHA-256 `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`; its HTTPS retrieval receipt is independently referenced in `sources/receipts.json`. I read §3.2 across pp. 3–4 and §3.2.1, Eq. (1), and footnote 4 on p. 4. I rendered and personally viewed p. 4 with Poppler `pdftoppm` 26.05.0. The printed equation is `softmax(QK^T/sqrt(d_k))V`; the paragraph explicitly identifies ordinary dot-product attention as the same construction without the scaling factor. The opening states that each value's weight is computed from compatibility of a query with its corresponding key. Thus the query/key demand/book-index analogy is pedagogical shorthand for compatibility between vectors; it is not evidence of semantic understanding. The complete attention formula additionally multiplies by V, which this section correctly postpones.

I retrieved and personally read official PyTorch raw documentation/source pinned to the installed wheel's git commit `5c4886908584029761b579af026dcfb627c84070` (installed `2.14.1+cpu`):

- `_torch_docs.py` lines 7909–7923: `torch.matmul`'s 1-D × 2-D rule temporarily prepends a size-1 dimension and removes it afterwards; lines 9583–9611: `torch.tensor` constructs data tensors, has `requires_grad=False` by default and no autograd history.
- `_tensor_docs.py` lines 5428–5434: `Tensor.softmax` is an alias of `torch.nn.functional.softmax`; lines 6770–6783: `.T` reverses dimensions as a view. Here K is exactly 2-D, so the higher-dimensional deprecation warning does not apply.
- `nn/functional.py` lines 2176–2216: softmax formula `exp(x_i)/sum_j exp(x_j)`, selected `dim`, and `[0,1]` entries summing to one mathematically; lines 5895–5915: cosine similarity divides the dot product by the two vector norms. A raw dot product has no such denominator.

These sources support precise vector compatibility and API contracts. They do not imply that an arbitrary hand-selected feature corresponds to color, that the largest attention weight indicates conceptual understanding, or that this toy was trained.

## Own calculation and execution

The helper extracted raw UTF-8 bytes without stripping or newline normalization: section SHA-256 `0ec167d309ad75e1d205886807372037c5c27d95bb18668b1219f4950b100bfb`. The original single Python fence ran in `.venv` CPU with exit 0 and printed scores `[1,0,-1]`, weights `[0.6652,0.2447,0.0900]`. Permanent copies retain the original code, bootstrap, helper, commands, environment, execution metadata, stdout and stderr.

My independent CPU probe used a separate Python `sum(a*b)` path for the dot products and `math.exp` fractions for probabilities. For each case the denominator includes all three candidate positions, not the two feature dimensions. Features, scores and proportions are dimensionless; no feature has assigned semantic units.

- Original query `[1,0]`: scores `[1,0,-1]`; probabilities `[0.665240955775,0.244728471055,0.090030573170]`. Equivalent stable denominator is `1+exp(-1)+exp(-2)=1.5032147244080551`. All three entries are positive, ranked first > second > third. The lesson's three-decimal weights round correctly.
- Change only query to `[0,1]`: scores `[0,1,0]`; probabilities `[0.211941557617,0.576116884766,0.211941557617]`; stable denominator `exp(-1)+1+exp(-1)=1.7357588823428847`. The second candidate is highest with unchanged keys.
- Continue with query `[0,1]` and change only third key to `[-1,1]`: scores `[0,1,1]`; probabilities `[0.155362403497,0.422318798252,0.422318798252]`; denominator `exp(-1)+1+1=2.3678794411714423`. The two trailing weights round to 0.422.
- Return to query `[1,0]`, lengthen the first key from `[1,0]` to `[2,0]`: scores `[2,0,-1]`; probabilities `[0.843794734481,0.114195199385,0.042010066134]`. The first dot product doubles although direction is unchanged. A separate norm-normalized cosine control returns 1 for both keys. This directly distinguishes raw matching scores from a direction-only comparison.

All dot-product values are exact small representable integers. Each float32 weight differs from the independent Python fraction by at most `3.4751687972e-8`, below `1e-7`; the largest three-decimal rounding error is `0.000318798252`, below `0.0005`. Three-term weight sums use absolute tolerance `3*eps(float32)=3.5762786865234375e-7`; maximum observed sum error is `1.1920928955078125e-7`.

The first version of my probe used a sum tolerance of `1e-7`, too tight for the normal one-epsilon float32 accumulation in the changed-query case, and exited 1. I retained its code and stderr, diagnosed the sum as `1.0000001192092896`, then used the three-term accumulation bound above. I did not change any lesson, source data, algorithm, individual-weight tolerance or claim. The corrected probe passed all four cases and cosine control with exit 0; this is a verification-harness adjustment, not a textbook correction.

API/shape checks confirm q `[2]`, keys `[3,2]`, keys.T `[2,3]`, scores and weights `[3]`; the actual `@` result equals `torch.matmul` and `softmax(dim=0)` normalizes the candidate axis. `requires_grad=False` and no grad functions are observed for all data and results. No optimizer or parameter update is present.

## Figures and boundaries

3.2 has zero image/SVG references, independently confirmed by the current raw section and extraction metadata; there is no course figure to render or judge. The source paper page was actually rendered and viewed only to verify its equation and source context.

This is matching and proportion calculation only. It does not read V, divide by `sqrt(d_k)`, mask future positions, learn a semantic query, or train a model. The lesson explicitly states these boundaries and links later sections; all current claims are consistent within that scope. No technical correction is requested.
