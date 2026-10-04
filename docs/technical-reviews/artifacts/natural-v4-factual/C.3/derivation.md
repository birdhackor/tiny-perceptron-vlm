# Independent C.3 mathematical verification

Reviewer: `/root/v4_review_coordinator/factual_v4_c_3`; fresh factual-only review.
Read source: `course/chapters/0C.md#C.3`, raw UTF-8 including terminal blank lines.

Let S_i be success of trial i for one fixed question. If P(S_i)=p for every
trial and the trials are mutually independent, then
P(no success among n) = product_i P(not S_i) = (1-p)^n.
Its complement therefore has probability C_n(p)=1-(1-p)^n.
Independence plus a common per-question p are conditions, not conclusions from
a finite list of observed answers. For 0<p<1, C_(n+1)-C_n=p(1-p)^n>0.
At p=0 and p=1 this difference is zero; repeated deterministic failure remains
failure. The section's increasing statement uses its interior value p=0.3.

With p=3/10, failure probability is 7/10. Two failures have probability
49/100; two-trial coverage is 51/100. Exact values for n=1,2,4,8 are:

| n | (7/10)^n | 1-(7/10)^n | round(...,6) |
|---|---|---|---|
|1|7/10|3/10|0.3|
|2|49/100|51/100|0.51|
|4|2401/10000|7599/10000|0.7599|
|8|5764801/100000000|94235199/100000000|0.942352|

The Python float calculation prints the displayed rounded values. It does not
measure a model's p. In [3,4,3,5], 4 occurs at index 1 (the second item), so
membership is True. Selecting index 0 returns 3 and fails against truth 4.
Changing the assigned list to [3,3,3,3] changes membership to False and leaves
the independently assigned p and the four formula outputs unchanged.

Repeated answers do not establish dependent sampling: two independent trials
that return the same wrong answer with probability 0.7 each will both return
it with probability 0.49. If the same random variate U is instead reused in
each Bernoulli decision [U<p], all failure indicators are identical and joint
failure is 1-p rather than (1-p)^n. Changing a later prompt can change the
conditional output distribution; the identical-distribution condition must
then be reexamined. PRNG draws are a computational approximation to random
sampling; the CPU probe does not prove mathematical independence.

For heterogeneous questions with success probabilities p_j, population mean
coverage is mean_j[1-(1-p_j)^n], generally not 1-(1-mean_j p_j)^n.
For two equally weighted questions p_1=0 and p_2=0.6, n=2 coverage is
(0 + (1-0.4^2))/2 = 0.42. The average p is 0.3, whose plug-in coverage is
1-0.7^2=0.51. For n>=2, the coverage function is concave in p; the gap is a
question-mixture effect even when every individual question is iid sampled.

Chen et al., arXiv:2107.03374v2, section 2.1, Equation (1), uses
mean_problems[1 - binom(n-c,k)/binom(n,k)], where n>=k and c counts successful
samples. Its unit-test success criterion is a property of that code benchmark;
this arithmetic course instead uses its declared parsed-final-answer criterion.
For n=k and c=0: binom(n,n)/binom(n,n)=1, giving 0. For c>0:
n-c<n, so binom(n-c,n)=0, giving 1. The average estimator in this special case
is the fraction of questions whose actual k-candidate set has a success.
The paper explicitly warns that directly reporting k-sample hits has high
variance and uses larger n=200 candidate pools; this lesson explicitly confines
itself to n=k observations and makes no HumanEval benchmark claim.

Separately resampled sets for k=1,2,4,8 need not contain one another. The
observed hit indicator for a question, and the aggregate over just 24 questions,
can therefore decrease when k grows even though expected iid coverage is
nondecreasing. No contradiction follows from direct coverage 2/24 at k=4 and
1/24 at k=8. Each branch has 24*(1+2+4+8)=360 candidate draws, but still only
24 distinct questions. Eight table cells have denominators 24, not 360 or 720.

Arithmetic target (1+2)+3=6; its supervised steps 1+2=3 and 3+3=6 are valid.
Equal 900 updates at 24 sampled examples/update gives 21,600 example draws
per branch. Replaying the recorded seed-42 sampler and counting nonignored
assistant labels (including EOS) gives direct 48,803 and steps 466,322;
466,322/48,803 is approximately 9.5552. Equal examples and updates thus do not
mean equal supervised-token budgets. This is a replay of data selection and
mask counting, not a replay of optimizer updates or GPU model training.

Independent record audit decodes each raw candidate token ID array, checks
the question's operands and arithmetic truth, rejects invalid special tokens
for scoring, and parses only the declared final-answer format. It then checks
every stored final_answer, final_correct and oracle_coverage flag against that
independent result. Coverage does not require all intermediate equations to
be correct and does not by itself define a deployable selector.

No SVG is referenced by C.3 or its actually read W.4/1.15/C.1 prerequisites.
The complete figure map is empty; there is no figure to render or view.
