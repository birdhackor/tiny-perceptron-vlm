# Original authority inspection

Reviewer `/root/v4_review_coordinator/factual_v4_c_3`, accessed 2026-10-04.
Full external files reside only in ignored `outputs/natural-v4/factual-research/C.3/`.
The public receipts preserve URLs, actual versions and SHA-256; this file is
the reviewer's own summary, with only necessary short quotations.

1. Mark Chen et al., *Evaluating Large Language Models Trained on Code*,
   https://arxiv.org/pdf/2107.03374v2, arXiv v2 dated 14 July 2021 as printed on
   the first page. Read section 2.1 on pages 2–3, Equation (1) and Figure 3.
   Definition uses k code samples per problem, any test-passing sample, then
   fraction of problems solved. Its lower-variance estimator uses n>=k,
   n=200 and k<=100 in the paper. Figure 3's n-c<k guard agrees with the
   binomial-zero convention. Necessary short quote: “computing pass@k in
   this way can have high variance.” The section's arithmetic record is an
   n=k special case, not a reproduction of this code benchmark. Also read
   the paragraph distinguishing selecting a tested sample from selecting
   highest mean log probability; oracle availability differs from deployment.

2. Python Language Reference, https://docs.python.org/3.13/reference/expressions.html,
   retrieved page identifies itself as Python 3.13.16 documentation.
   Read section 6.5 `the-power-operator`: two-argument exponentiation raises
   the left argument to the right argument's power. Read section 6.10.2
   `membership-test-operations`: for built-in containers including list,
   membership means some contained element is identical or equal.
   This supports the lesson's ** and in semantics. Runtime here is 3.13.5;
   actual snippet/exercise execution independently confirms these specific
   semantics on that patch version. Membership neither samples candidates
   nor selects one answer.

3. PyTorch official source, release tag v2.14.1,
   https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_torch_docs.py,
   read lines 8087–8130, `torch.multinomial` docstring. Each matrix row supplies
   its own sampling distribution; num_samples=1 selects one index per row.
   Nonnegative, finite, nonzero weights are required. replacement=False's
   within-row restriction does not ban equal answers in different candidate
   rows. Installed runtime's API was exercised by the actual `_sample` CPU
   probe at temperature 0.7. A finite mechanics probe is not an empirical
   proof of iid candidate independence.

4. PyTorch official source, release tag v2.14.1,
   https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/utils/benchmark/utils/timer.py,
   read `Timer` docstring lines 66–90. It explains warmups for lazy
   initialization, fixed threadpool size, synchronization of asynchronous
   accelerator operations and replicates to quantify run-to-run variation.
   This supports the section's timing qualifications. The formal report has
   recorded batch times; neither those unbalanced first-call timings nor my
   four-token mechanics probe establishes a general speed improvement.
   Batching in the inspected `_sample` code provides a concrete way that
   candidate count differs from serial call count. No timing ratio is claimed
   in C.3, and no new CPU/GPU latency benchmark is claimed by this review.

5. Alec Radford et al., *Language Models are Unsupervised Multitask Learners*,
   OpenAI author-hosted 2019 technical report,
   https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.
   Read section 2.2 `Input Representation` on page 3: byte, character and word
   units, BPE interpolation between character and word-level symbols,
   multi-symbol vocabulary entries and UTF-8 byte representations.
   This supports the general contextual explanation of variable token size.
   The arithmetic experiment uses this repository's ByteTokenizer instead,
   whose actual implementation is one UTF-8 byte plus offset 8 per text ID.
   General token terminology does not make the course's byte counts universal.

The initially attempted PyTorch rendered documentation endpoints returned
HTTP 403. The actual read authoritative alternatives are release-tagged
official source files above. Preliminary v2.8.0 source reads are retained only
in the ignored research workspace; the report cites the successfully fetched
v2.14.1 originals matching the current API version.
