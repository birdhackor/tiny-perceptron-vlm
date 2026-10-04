# Additional original probability authority inspection

Reviewer `/root/v4_review_coordinator/factual_v4_c_3`, read 2026-10-04.
This evidence supplements the independently preserved first report after the
official schema check requested an original authority for concept-classed
elementary probability statements. The mathematical calculations and factual
conclusion are unchanged; no textbook or checker was edited.

Actually retrieved and read Grinstead and Snell's *Introduction to Probability*,
CHANCE Project version dated 4 July 2006, edited by Peter G. Doyle from the
authors' AMS 2003 second edition, hosted by Dartmouth's author/project site:
https://math.dartmouth.edu/~prob/prob/prob.pdf.
Retrieved bytes SHA-256:
`763eab9894983ddfd6cd7f84685548d1515a9326a2d9fd015474534460551a5e`.
The full PDF and text remain only in ignored research.

Read section 3.2, Definition 3.5, pp.96–97: Bernoulli trials have success and
failure outcomes, identical success probability p for each experiment, and
that probability is unaffected by knowledge of previous outcomes. Failure
probability is q=1-p. Read Theorem 3.6, p.98: probability of j successes in
n Bernoulli trials is binom(n,j)*p^j*q^(n-j). For j=0 the result is q^n;
complementing gives the lesson's 1-(1-p)^n. This is exact under the stated
assumptions, not an empirical success-rate estimator from a made-up list.

Read section 4.1, Definition 4.1 and Theorem 4.1, pp.139–140: event
independence is equivalent to the joint event probability being the product
of marginal event probabilities. Read Definition 4.2, p.141: mutual
independence is required for every relevant subset, equivalently arbitrary
events or their complements. The text distinguishes mutual independence
from pairwise independence. These definitions support independently checking
the changed-prompt/shared-random-variate caveats rather than treating repeated
answers as evidence of dependence. The p=0 boundary follows directly from
the binomial distribution and from deterministic failure, while reusing one
random variate can produce perfectly dependent success indicators.

Applied to each question j separately, the binomial zero-success result is
(1-p_j)^n. Averaging questions therefore averages the individual nonlinear
coverage results. My exact p=[0,0.6], n=2 calculation in derivation.md
establishes 0.42 versus plugging the average p=0.3 to obtain 0.51. The textbook
supports the per-question probability law; this counterexample is my own
calculation, not a quotation or claimed textbook example.

Additionally read Chen et al. arXiv:2107.03374v2, Appendix A, p.13,
`Estimating pass@k`. It distinguishes empirical plug-in bias from the
unbiased binomial/combinatorial estimator, explicitly assumes c~Binom(n,p),
and expands its expectation to 1-(1-p)^k. Its plug-in-estimator bias concerns
finite sampled p-hat, a separate issue from mixing different question p_j.
The section does not conflate those issues, and the review keeps them separate.
