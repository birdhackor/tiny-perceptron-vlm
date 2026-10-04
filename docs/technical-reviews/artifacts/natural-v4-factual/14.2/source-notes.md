# Independent factual review notes: 14.2

Reviewer: `/root/v4_review_coordinator/factual_v4_14_2`, fresh factual owner.
Date: 2026-10-04. No prior verdict, author note or coordinator conclusion was read.
Prior `docs/technical-reviews/14.2.json` was copied as raw bytes, unread, before replacement.

The complete raw UTF-8 section, including trailing blank lines, was read and saved as
`original-section.md`: SHA-256 `a5be41502a2ec0d15790becd4bd18c13ba75200abcf61d82bcce314e89a99b0d`.
Explicit prerequisites 4.3 and T.8 were fully read and saved with hashes in the report.
Only their normalization and modern-comparison evidence is used for this section;
this review does not claim to review all other experiments described in T.8.
Neither prerequisite has an SVG reference. The sole direct figure is normalization.svg.

## Original authorities actually inspected

Full original PDFs and official HTML responses were retrieved into ignored
`outputs/natural-v4/factual-research/14.2/originals/`. Public `source-receipts.json`
contains URLs, version, bytes and retrieval SHA-256, not complete third-party documents.
The notes here are the reviewer's own summaries. Papers were read using their
extracted original text, and the official documentation's main articles were read.

- Ba, Kiros and Hinton, *Layer Normalization*, arXiv:1607.06450v1 (21 July 2016):
  abstract and pp. 2–4, §3 Eq. (3), §3.1 Eq. (4), §5.1 Eqs. (5)–(7).
  Statistics are over hidden units within a case/time step; gain and bias are
  feature-specific and shared across time. The normalization subtracts the mean.
  For the concrete epsilon and API convention, the current official PyTorch
  definition is decisive, rather than the paper's idealized denominator.
- Zhang and Sennrich, *Root Mean Square Layer Normalization*, arXiv:1910.07467v1
  (16 October 2019): pp. 3–5, §3 Eqs. (2)–(3), §4 Eq. (4), §4.1 Eqs. (5)–(7),
  §6 measurement setup; pp. 6–9, §6.1–6.4 and §7 speed limitations.
  Eq. (4) divides by sqrt(sum of squared features/n), with gain and without
  centering. §4.1 expressly says it is not invariant to all recenterings.
  The paper's broader layer formula also has an optional downstream bias;
  therefore the chapter's “common RMSNorm has only gamma” is checked as a common
  implementation convention, not as a universal prohibition on a bias.
  §7 explicitly makes actual speed depend on framework, hardware, architecture
  and the relative cost of other components. The paper's experiment speeds were
  not independently benchmarked here and are not imported as project speeds.
- PyTorch 2.14 official `torch.nn.LayerNorm` article: defining formula,
  last-D-dimension statistics, biased variance (`correction=0`), `eps`,
  `elementwise_affine`, `bias`, initial weight/bias, and input/output shape.
- PyTorch 2.14 official `torch.nn.RMSNorm` article: defining equation with epsilon
  inside the square root, last-D dimensions, affine weights only and same shape.
  Its default epsilon is dtype-dependent; this chapter explicitly uses 1e-5,
  so no claim of equality to the unspecified library default is made.
- PyTorch 2.14 official `torch.nn.functional.layer_norm`: actual signature is
  `(input, normalized_shape, weight=None, bias=None, eps=1e-05)` and it refers to
  LayerNorm for semantics. The chapter's absent weight/bias act as identity/zero.
- PyTorch 2.14 official `torch.mean`: `dim` reduction and `keepdim=True` retain
  size 1 on reduced dimensions. On a [2,3] table the denominator is [2,1].
- PyTorch 2.14 official *Reproducibility*: introduction and “Controlling sources
  of randomness / PyTorch random number generator / Python”, plus the rest of
  the article for its determinism boundaries. Resetting seeds repeats random
  sequences in a fixed environment; it does not promise bitwise results across
  releases, platforms, CPU/GPU, or nondeterministic algorithms.

## Independent arithmetic and actual CPU verification

For a vector with n=3 features, m=sum(x)/3 and v=sum((x-m)^2)/3.
LN(x)_j=(x_j-m)/sqrt(v+eps) with gamma=1 and beta=0.
RMS(x)_j=x_j/sqrt(sum(x^2)/3+eps), gamma=1.

For [2,2,2], m=2, v=0, so LN is [0,0,0]. With no epsilon,
RMS=sqrt((4+4+4)/3)=2 and its output is [1,1,1]. With eps=1e-5,
each entry is 2/sqrt(4.00001)=0.9999987500023438; the FP32 observation
is 0.9999988079071045. “Approximately all ones” is the proper epsilon reading.

For [1,2,3], m=2, centered values [-1,0,1], v=2/3.
sqrt(2/3+1e-5)=0.8165027046291191 gives +/-1.22473568590839.
The ideal RMS is sqrt(14/3)=2.160246899469287, and adding epsilon makes
the divisor 2.1602492140182963. Dividing [1,2,3] yields approximately
[0.4629095539,0.9258191078,1.3887286617]. The four-decimal textbook
roundings are correct; scalar math vs CPU tensors has max errors below 7e-8.

Adding c to every feature gives m'=m+c and x'-m'=x-m; v is unchanged,
so LN translation invariance holds algebraically even with fixed epsilon.
RMS instead uses sum((x_j+c)^2)/3, so it generally changes. Its noncentering
preserves a common component, not its original absolute amplitude or recoverability.
The actual exercise changes the second row to [11,12,13] and produces approximately
[0.9146,0.9977,1.0808]; the constant positive row still rounds to all ones.

The exact source code block was extracted and executed, then executed again with
`x=x+10` inserted after construction. `cpu_verify.py`, `main-excerpt.py`,
`cpu.stdout.txt`, `cpu.stderr.txt`, `cpu.exit-code.txt` and `cpu-results.json`
preserve the code, pre-check predictions, exact outputs and environment.
Actual runtime: Python 3.13.5, torch 2.14.1+cpu, one CPU thread, FP32,
CUDA unavailable. Checks also covered all-zero inputs (finite zero output),
same shapes, [2,1] broadcasting denominators, another row's independence,
agreement of the project and official RMSNorm at explicit eps=1e-5, and a
bounded [2,4] TinyLM forward/backward for each rule with 8 valid targets,
finite outputs/gradients and zero optimizer updates. These are mechanism checks.

Two blocks each have two normalization modules, and final_norm adds one: five.
Each LN has 64 gamma and 64 beta parameters; each project RMSNorm has 64 gamma
parameters. Removal is exactly 5*64=320. Instantiated original configurations
contain 141568 and 141248 parameters, agreeing with the original report.

## Original project run accounting

The original record actually inspected is `docs/course-experiments/results/modern.json`,
SHA-256 `f14577a67074ea059361eb1da2fc43dd5d06966b813230bb8d29d9953a22ae9f`.
The review compared only baseline and rmsnorm for 14.2. It records revision
`4ef6555710e9915179a4a69738cdfabe80c299cf`, NVIDIA L4,
torch 2.14.1+cu126, Python 3.13.3, seed 42, FP32 (`amp_dtype=None`),
TF32 disabled, width 64, two layers, four heads, max length 128,
batch 16, AdamW learning rate .003, 240 completed and successful updates,
zero skipped updates, 452102 sampled valid targets per variant.

The actual original 512 JSONL records already in `data/training/text-initial/`
were used without downloading more data. Every text's SHA was recalculated.
An independent sorted-family/shuffle-by-42 split reconstructed 409/51/52 stories
and exactly matched the run's three split hashes. No full-text fingerprint family
crosses sides. Near duplicates were not clustered, consistent with T.8's caveat.
Every story has len(UTF8 bytes)+1 supervised next-byte/EOS targets; BOS is
input only. Nonoverlapping 128-target windows share only preceding context.
Independent lengths give training 329581 corpus targets/2789 windows,
validation 39256/334, and test 41914/352. Replaying only the random integer
window sampler for 240 batches of 16 gives exactly 452102 valid train targets.
No model was trained in that replay.

Independent original-record divisions:

- Baseline validation: 88848.587890625 / 39256 = 2.2633123061602047 -> 2.26331.
- RMS validation: 88684.2216796875 / 39256 = 2.2591252720523616 -> 2.25913.
- Baseline test: 96051.251953125 / 41914 = 2.291626949303932 -> 2.29163.
- RMS test: 95828.59106445312 / 41914 = 2.2863146219509742 -> 2.28631.

These are natural-log cross-entropy nats per valid next-byte/EOS target, not
character loss, accuracy, or a replicate of the original GPU run.

Actual seconds-to-milliseconds conversions of the saved step medians are
0.013062208999999214*1000 = 13.062208999999214 -> 13.062 and
0.01395199199999908*1000 = 13.95199199999908 -> 13.952.
RMS is slower by this recorded median, so this run contains no measured speed benefit.
Original `_train` starts timing after zero_grad and a device synchronization;
it includes pad/device input transfer, forward/loss, backward, finite-gradient
checks, clipping, optimizer/scaler update and ending synchronization. It excludes
the preceding sampler, initial/heldout NLL and checkpoint saves from each step.
It computes statistics.median(latencies[3:]), 237 timed updates after three warmups.
The original report does not retain all 240 latency samples. I checked its saved
value/units and the actual median implementation; I did not rebuild the median
from missing raw timings or claim a GPU timing reproduction.

Current architecture.py differs from the run's full-file hash because it adds the
later Flash probe. The original committed file was independently fetched into
ignored research, and SHA `1ad0b0789318236d5e1a8fc1117cb65758bbd6e3ad62dc2d646a111ae1ea37a3`
matches the original record. The eight relevant current/original functions have
identical parsed ASTs, verified by executed code. The full current file SHA and
the other inspected code files are recorded in the report; modern.py, model.py,
data.py, common.py and training.py still match the run's registered hashes.

Single seed and short small-model training support only the recorded results.
The small loss gap gives no variance estimate or stable quality guarantee.
Reset seed 42 repeats model initialization and Python sample indices in the
actual CPU environment, whereas seed 43 changes both; this does not prove
cross-platform or complete training determinism. Changing a normalization rule
changes activations before later weights, so suitability of a trained checkpoint
and speed/quality must be measured rather than inferred from fewer operations.

## Figure actually rendered and personally viewed

Original SVG SHA is `a0f27f008b6e2325e12753b55d985b753364c54c851b3ba66678a4520fa2929c`.
Inkscape 1.4 generated `normalization-render.png`; the reviewer used view_image
on that actual PNG. All five cards are visible: one token [1,2,3] ->
mean/scale (only feature) -> LN or RMS (two different rules) -> scale/shift
(gamma; LN has beta) -> same-shape output. Four arrows point right.
The footer prohibits cross-time normalization. There is no numerical output,
quality scale or speed promise in the figure. This is consistent with the
chapter's formulas, axes and explicit common RMS gain-only convention.
The mean/scale card covers the alternatives; it does not assert RMS subtracts a mean.
Inkscape's CSS-animation and Pango/GTK warnings were preserved in render.stderr.txt;
its static render shows all content. XML inspection shows pulse changes only
stroke/color, not the equations, text or arrow direction.
Poppler's nonfatal LayerNorm PDF xref reconstruction warning was preserved with
its actual exit code 0; the inspected equations and sections remain readable.

No substantive contradiction or unresolved factual claim was found. Official
checker output is recorded separately; it establishes schema/current files,
not the truth of these independently reviewed claims.
