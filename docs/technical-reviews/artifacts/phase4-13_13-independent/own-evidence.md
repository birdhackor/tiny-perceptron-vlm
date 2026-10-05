# 13.13 independent factual review: own evidence

Reviewer: /root/phase4_factual_coordinator/factual_13_13, fresh context, 2026-10-05.
Scope: current course/chapters/13.md#13.13 and required predecessor #13.12 only.
I read all of both sections, the cited SVG source and real rendered images. I did
not read prior technical/reader verdicts, histories or author result/correction summaries.
No subagents, model/data download, GPU, full training, model remeasurement, text/figure editing or commit.

## Frozen input and actual original-source reading

- Section SHA: 63aad406add8fde6187114a390fe7e70ca6aa9b3a8e8cbcba081caaadd746ca6. Original UTF-8 bytes, including heading and trailing newlines.
- Entire chapter is explicitly frozen input snapshot inputs/13.frozen.md:
  c33e12dd7e61e73f11a45e3f7ee8c099dd130d23133c5a15426adfe2ca24adbe. This does not assert current whole-chapter identity.
- Prerequisite 13.12 snapshot SHA: 4bf5bd63724d7b04fdd158402c7123d8e3e0d05d36d33540d2e2dafd5f7f39ed.
- Original repo helper tiny_perceptron/posttraining.py full SHA:
  3d0e2ae3b29f95abebde9b10d0cd106639a131fab1b2c7a509e732ba8daf295b; AST function ranges first, then own read of lines31–50.
  Shape/nonempty/epsilon guards, detached old and advantage, ratio exp(log difference),
  clamp, minimum, negative mean policy_loss and clip_fraction inspected. No auxiliary result strings read.
- Paper pointer was /records/56 of immutable original-paper-locators.json, fields
  path, sha256, first_page_arxiv_identifiers only; it located original PDF, never supplied a verdict.
- I independently read PDF page1 title/authors/arXiv identifier and version date, and
  page3 §3 equations6–7, accompanying explanation and Figure1/caption. The original
  PDF explicitly identifies 1707.06347v2, 28 Aug 2017, authors Schulman et al. (OpenAI).
  Authority URL https://arxiv.org/pdf/1707.06347v2, accessed2026-10-05. Local original
  SHA e78feadadbdbb0b601b3c2bcc81404722cd431a489b307545f9b7bea1e8c4f5b.
  Paper Fig1 was independently rendered and viewed (sources/ppo-page3.png).
- Official PyTorch sources were retrieved over HTTPS from pytorch/pytorch at actual
  installed commit 5c4886908584029761b579af026dcfb627c84070; original raw hashes/URLs in sources/official-source-manifest.json.
  Own API reads: _torch_docs.py clamp2936–2977, exp4310–4333, log6289–6316,
  mean7195–7261, minimum7628–7653, tensor9582–9635, sum11266–11321,
  full12856–12882; _tensor_docs.py Tensor.clamp1071–1078, exp1826–1833,
  log2933–2940, mean3245–3252, sum5027–5034, tolist5445–5465;
  _tensor.py backward566–625, detach798–814. First selector did not recognize
  add_docstr_all or detach assignment; second AST selector added those genuine reads.
  This grouped coverage accounts for original fence and helper API contracts.

## Independent derivation (dimensionless ratio and objective units)

For positive probabilities exp(log p_new-log p_old)=p_new/p_old.
Old is the sampled selected-card probability, not a six-card distribution; each
of the six samples may have old0.5. New0.5r gives0.35,0.5,0.65, all within[0,1].

With A=+1 and r>=0, f(r)=min(r,1+epsilon): clipping at the lower endpoint does
not remove the unfavorable slope. With A=-1, f(r)=-max(r,1-epsilon): clipping
at the upper endpoint does not remove the unfavorable slope. Minimization
uses -f, so slopes away from clipping corners are (-1,0) for positive A and
(0,+1) for negative A. No claim about the nondifferentiable endpoints is made.

epsilon0.2 gives f=[0.7,1,1.2,-0.8,-1,-1.3], d(-sum f)/dr=[-1,-1,0,0,1,1].
The original fence uses SUM, so there is no 1/6 derivative factor. The helper's
separate policy_loss is the MEAN over6: sum_loss0.2/6=0.0333333333333.
epsilon0.1 gives[0.7,1,1.1,-0.9,-1,-1.3] and the same six derivatives because
all six points remain on the same respective sides; sum_loss0.4, mean0.4/6.
The values have advantage units; ratio and probability are dimensionless.

I executed original fence unchanged via CPU offline section_facts worker:
exit0, output exactly the printed six values and gradients at2-decimal formatting.
The separately saved float64 variant checks assert abs error<=1e-12 (rtol0),
old and advantage receive no gradient, helper sum/mean denominator consistency,
epsilon0.1 and shape/nonempty/epsilon guards. Original float32 numerical
rounding is assessed at displayed2-decimal precision (raw tolerance1e-6 sufficient).
No optimizer is used in the original fence; backward only computes gradients.

For incorrect clamp-only A=[+1,-1], r=[0.7,1.3], f=[0.8,-1.2] has zero gradients
in both unfavorable directions. This demonstrates why the minimum cannot be omitted.
A bounded two-sample shared-parameter counterexample sets first ratio1.3 (clipped)
and second ratio0.8125 (not clipped). First loss derivative0; total loss derivative
-0.284375. One synthetic gradient step changes first ratio to1.423301235719266.
This is a mathematical counterexample to a hard bound, not model training/evaluation.
Figure1 caption explicitly says the objective sums many terms; clipping one term's
incentive does not project parameters or all probabilities into a feasible interval.

## Real visual check and limits

Chromium151 actual command timed out after30s with exit124; stderr retained.
Inkscape1.4 rendered640×850 and390×518 PNGs successfully; I viewed both.
All six numerical entries, gradient axis/sign convention, upper/lower clipping
labels and shared-parameter caveat match original fence and own calculation.
The figure gives a table, not a measured curve. Full browser-page responsive
layout remains unverified; this does not block technical consistency of the SVG.
No empirical performance claim or existing experiment results occur in13.13.
The demonstration supports objective/gradient mechanism only, not learning,
policy improvement, KL magnitude, generalization or guaranteed ratio confinement.
