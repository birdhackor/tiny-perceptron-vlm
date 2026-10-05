# 3.5 independent derivation, 2026-10-05

Read scope: current course/chapters/03.md sections 3.4–3.6 (3.5 is the reviewed scope),
current course/chapters/01.md#1.15 for the temperature reference, the whole
tiny_perceptron/attention.py, the extracted BOOTSTRAP, the complete
docs/review-tools/factual-reviewer-instructions.md and section_facts.py, and the
complete scripts/check_technical_reviews.py schema. No old report/history body or
another reviewer's inspection notes or conclusions were read. Task:
/root/phase4_factual_coordinator/factual_3_5; fresh one-section context.

Let P_i be a coordinate product. For independent P_i with E[P_i]=0 and
Var(P_i)=1, S=sum_i P_i has E[S]=0 and
Var(S)=sum_i Var(P_i)+2 sum_{i<j} Cov(P_i,P_j)=D. Independence makes the covariance
terms zero. Std(S)=sqrt(D). Var(S/sqrt(D))=Var(S)/D=1, so its standard deviation is
1 in the probability model. This controls statistical scale under the stated
assumptions, not every sampled magnitude and not every learned query/key.

Independent fair q_i,k_i in {-1,+1} have four equiprobable sign pairs. Two pairs
give +1 and two give -1; their product is a fair sign, with expectation 0 and
variance E[P_i^2]-(E[P_i])^2=1. Independent coordinate draws give independent
products. Two product coordinates have sums -2,0,0,2. Their complete distribution
has mean 0 and variance (4+0+0+4)/4=2, standard deviation sqrt(2). The denominator
4 is the whole probability distribution, not an estimate from four observations.
The independent verification also enumerates all 16 four-coordinate sums;
their probability variance is exactly 4.

In the code, q and k have axes [pair,matching_feature]=[1000,d]. Multiplication
is elementwise. sum(dim=-1) reduces the matching_feature axis and leaves one
dot product per pair, shape [1000]; std() then reduces that one pair axis.
The installed Tensor.std default is correction=1, so the sample variance divides
by N-1=999. Its correction=0 alternative divides by N=1000. Both are sample
statistics and differ from the exact probability variance of the fair-sign
model. Hand calculation uses the actual 1000 values and their sample mean,
divides squared deviations by 999, and agrees with PyTorch within 1e-6 absolute.
All demonstration scores are dimensionless; standard deviation has the same
dimensionless scale as those scores. D counts features, not positions, pairs,
value width, or the full model width across heads.

Scaling both q and k by 2 gives (2q) dot (2k)=4(q dot k). For the same sampled
data, std(4S)=4 std(S), before and after division by sqrt(D). Scaling only q
instead gives factor 2. These identities are exact here for the stored integer
valued scores; printed float32 standard deviations are checked with 1e-5
absolute tolerance. A seed-7 rerun is a meaningful sampling change, not a claim
that arbitrary seeds must give the same samples. The 8% finite-sample check is
a bounded demonstration tolerance, not a statistical confidence guarantee.

The independence boundary check repeats one fair sign across all D products in
each row, with exactly 500 positive and 500 negative rows. Each product still
has mean 0 and probability variance 1, but all coordinate products are perfectly
correlated. The unscaled probability standard deviation is D and the scaled
one sqrt(D), so the divisor does not remove this change in covariance. This
supports the text's conditional limit and does not describe trained Q/K data.

For two logits [a,0], softmax probabilities are
[1/(1+exp(-a)), 1/(1+exp(a))]. For a=1 they are
[0.7310585786300049,0.2689414213699951]; for a=8 they are
[0.9996646498695336,0.0003353501304664781]. The displayed decimals are correctly
rounded (3 and 4 decimal places). A positive denominator sqrt(D) has the same
algebraic role as a positive temperature before softmax; attention distributes
weight across key positions, whereas generation temperature transforms next-token
vocabulary scores. The source TemperatureLogitsWarper was inspected, not executed
with its model/download example.

Source inspection: personally read Vaswani et al. arXiv:1706.03762v7 PDF page 4,
section 3.2.1, Eq (1), footnote 4, and section 3.2.2 including page-5 per-head
dimension definitions in the immutable original PDF's text extraction. The PDF's
SHA-256 was independently verified. Rendered page 4 with Poppler pdftoppm
26.05.0 and viewed the PNG; the equation clearly puts sqrt(d_k) beneath QK^T
before softmax and the footnote states the independence/mean/variance assumptions.
The current 3.5 section contains no image references, so section figure
consistency is not applicable. No Chromium or Inkscape limitation occurred.

API inspection: personally read pinned upstream PyTorch commit
5c4886908584029761b579af026dcfb627c84070 matching installed 2.14.1+cpu,
_torch_docs.py randint, mul, div, sum and std; _tensor_docs.py float, item, std
and sum; nn/functional.py softmax and scaled_dot_product_attention; random.py
manual_seed and its implementation. Read CPython v3.13.5 math.rst sqrt,
and Transformers v4.57.1 logits_process.py TemperatureLogitsWarper docstring,
positive-temperature validation and scores/temperature implementation. Each
source's exact HTTPS URL and SHA-256 is in sources/receipts.json. The installed
PyTorch version is stated explicitly; no assertion about latest-release identity
or cross-release random sequence reproducibility is made.

No model training, optimizer, gradient computation, model/data download, GPU,
or model-quality evaluation was performed. The original is a sample-scale
demonstration and the textbook explicitly says so. No revision is required.
