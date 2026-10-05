# W.4 independent factual inspection

Reviewer: `/root/phase4_factual_coordinator/factual_w_4`.
Date: 2026-10-05 UTC. No previous technical/reader report, history,
review-dispatch body, extra author result interpretation, or correction answer
was read. The locator cache was read only at top-level key/type and matching
`/locators/{46,58,63,71,119,131,228,263,95,96}` URL/version/path/hash records;
their descriptions were not used as evidence. Actual PyTorch authority was
downloaded from its immutable upstream commit and read independently.

Actual course reading: current `course/first-steps.md` lines 78–140 (W.3 as
necessary tensor/axis context, all of W.4, and the visible opening of W.5).
Only W.4 is judged. The exact W.4 UTF-8 bytes are `execution/original-section.md`.
The whole Markdown copy in `inputs/course/first-steps.md` is explicitly the
initial frozen input; its hash is not represented as the current whole chapter.
The W.3 figure was outside this technical review; W.4 itself has no image,
SVG, graphical datum, or spatial label/arrow claim. Its card counts and scalar
probabilities are fully specified in words. Figure consistency is concretely NA.

I read the full current factual instructions, section extraction/execution helper,
technical checker schema, clear-tutorial SKILL, and review protocol. The helper's
BOOTSTRAP was extracted by its AST selection and read at `execution/original-bootstrap.py`.
The actual fence uses only `math`, a for-loop, negation, `round`, and `print`;
it performs no gradient calculation, parameter update, model training, or scoring.

## Authority actually inspected

* NIST/SEMATECH e-Handbook, §1.3.6.1, `nist-probability.html` lines 90–128:
  discrete p(x), nonnegative probabilities, sum over all outcomes = 1,
  and the resulting 0 ≤ p(x) ≤ 1. The bag/candidate context is finite and discrete;
  this review does not generalize probability-zero wording to continuous
  sample spaces or claim probability calibration of a trained language model.
* NIST DLMF 1.2.8, released 2026-09-15, §4.2 (`nist-log.html`): E2 and its
  positive-real restriction paragraph; E11–E14; E21; E25; E27–E28; E32.
  These supply the natural logarithm, base e, inverse exponential, negative
  powers and noninteger real powers. §4.4 (`nist-logidentities.html`) E1
  states ln(1)=0. I read the equations and surrounding conditions, not summaries.
* CPython v3.13.5 original documentation: `python-math.rst` lines 480–486
  (single-argument log is natural), 767–770 (e), 816–825 (log(0.0) raises
  ValueError); `python-functions.rst` lines 1760–1785 (round precision and
  ties-to-even); `python-expressions.rst` lines 1223–1250 (power operator).
* MIT OCW 6.041/6.431, Fall 2010, original Lecture 3 PDF, printed page 1,
  “Independence of two events”: P(A∩B)=P(A)P(B), conditional-probability
  interpretation, and symmetric definition. Printed page 2, “Independence
  vs. pairwise independence”, gives two fair coin tosses with probabilities
  1/2. I read the original PDF-derived text, including printed page numbers.
* PyTorch upstream commit 5c4886908584029761b579af026dcfb627c84070:
  `pytorch-functional.py` AST-located functions softmax lines 2176–2216,
  cross_entropy lines 3478–3569; `pytorch-loss.py` AST-located classes NLLLoss
  lines 136–194 and CrossEntropyLoss lines 1200–1302. I read the formula,
  reduction, class-index, input, shape, weighting and ignore-index contracts.
  My CPU script independently established byte identity of downloaded
  functional.py and the installed 2.14.1+cpu source. Class-index targets,
  no weight, no ignored items, label smoothing zero, and default mean are
  the scope supporting this section's introductory description of CE.
* Vaswani et al., Attention Is All You Need, arXiv:1706.03762v7,
  2023-08-02: personally checked title/authors/version on the original first
  page and §3.4, printed page 5, in the original PDF-derived text.
  The decoder's linear transformation and softmax produce predicted next-token
  probabilities. This supports the introductory language-model candidate
  probability statement; it is not used to assert a capability of this repo.

I used the NIST DLMF integral definition on positive real values to derive,
rather than assume, the log properties used by the section. For x,y>0,

ln(xy) = ∫[1,x] dt/t + ∫[x,xy] dt/t
       = ln(x) + ∫[1,y] du/u = ln(x)+ln(y), using t=xu.

For 0<p≤1, -ln(p)=∫[p,1] dt/t ≥ 0. Its derivative is -1/p < 0;
ln(1)=0; as p→0+, the integral diverges. Thus the penalty increases
when the correct-answer probability falls. This is a real-positive argument;
there is no complex-log branch identity being asserted. A finite NLL value
requires p>0. The nonfinite p=0 endpoint is a limit, not a successful
`math.log(0)` call. The original fence's round expression leaves its `cost`
variable at full float precision; only the value passed to print is rounded.

## Independent checks and scope

The original fence executed once through the provided helper, exit 0,
with 0.9 0.105 / 0.5 0.693 / 0.1 2.303. Its full raw code, bootstrap,
commands, stdout/stderr, extraction metadata, and CPU environment are preserved.
No blocked guard event occurred.

`execution/independent_cpu_checks.py` was authored by this reviewer and actually
executed in the same .venv on CPU, exit 0. Its assertions compare binary math.log
with independently computed 60-digit Decimal logarithms, check rounding with
Decimal quantization and absolute error ≤0.0005, enumerate the four equally
likely cards and four independent two-question outcomes, and contrast perfect
dependence (joint 1/2) with independence (joint 1/4). It checks `log(0)`'s
ValueError, the p=1 endpoint, fractional exponents, and three positive
probability pairs for the multiplication identity. It also validates the
0.25 exercise (1.3862943611198906 → 1.386), e≈2.718281828459045, and
e^-1≈0.36787944117144233 → 0.368.

For API coverage I independently use float64 logits [2 rows, 2 classes],
class axis 1 and targets [0,0]. softmax yields [0.9,0.1] and [0.1,0.9];
per-target losses are 0.10536051565782635 and 2.302585092994046.
Default CE gives 1.2039728043259361 with denominator 2; sum gives
2.4079456086518722. Passing probabilities as the predicted input instead
gives 0.7711006659477777, demonstrating the input contract. A uniform +500
logit shift preserves CE to 1e-13. These are arithmetic/API checks; no
model capability result or new training result is produced.

Two Penn State candidate URLs and a Yale candidate URL returned HTTP 503;
an initially guessed MIT resource name returned 404. I resolved independence
from the MIT course's own lecture list and original lecture PDF. NIST §4.3
was discovered to be Graphics and was not used; §4.4's title is Special Values
and Limits (the local filename is merely descriptive). No failed candidate
is cited as evidence. NIST binomial pages were inspected but not used for
independence because they did not explicitly state the needed definition.

No substantive unresolved issue was found. The subsection is a hand arithmetic
demonstration and preparatory API explanation, not an empirical trained-model
claim. Its independence example explicitly states the condition; CE averaging
is supported in the section's single-correct-answer, default unweighted setting.
