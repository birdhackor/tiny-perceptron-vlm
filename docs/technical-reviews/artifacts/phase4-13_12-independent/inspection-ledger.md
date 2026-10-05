# Independent factual review 13.12

Reviewer task: `/root/phase4_factual_coordinator/factual_13_12`.
This is a fresh independent review. No older technical, reader, history,
dispatch conclusions, or author correction summaries were read.
The initial broad filename inventory printed paths only; no contents from
those paths were opened. No exposure to a prohibited conclusion occurred.

## Input and actual reading

Read the complete current section 13.12, extracted from original UTF-8 bytes
at chapter lines 373 onward. Read 13.11 for the contextual bandit and fixed
advantage definition and 13.4 for the policy/reference definition.
The frozen whole chapter is `frozen-input/chapter13-frozen-input.md`, SHA-256
`c33e12dd7e61e73f11a45e3f7ee8c099dd130d23133c5a15426adfe2ca24adbe`.
That fingerprint means this review's frozen input snapshot, not the current
whole chapter. Section SHA-256 is
`4bf5bd63724d7b04fdd158402c7123d8e3e0d05d36d33540d2e2dafd5f7f39ed`.
This is not a chapter-first section, so an introduction review is not required.

Read the factual reviewer instructions, checker schema, section extraction
helper, clear-tutorial SKILL.md and its review protocol. The cloud setup skill
was also read; no installation or configuration mutation was needed.

## Original authority personally inspected

PPO paper: Schulman et al., *Proximal Policy Optimization Algorithms*,
arXiv:1707.06347v2, 28 August 2017,
https://arxiv.org/pdf/1707.06347v2 . The PDF itself identifies this version on
page 1. Its SHA agrees with the immutable locator, independently recomputed
from the actual local PDF. Read abstract, sections 1–3, section 5 and
Algorithm 1, PDF pages 1–5. Formula 6 defines the selected action ratio with
the old sampling policy as denominator and multiplies it by the advantage;
formula 7 defines the clipped surrogate and explains that only changes that
improve the objective lose their incentive outside the clip range.
Algorithm 1 collects under pi_old, performs K epochs, then assigns theta_old
from theta before another collection iteration. No empirical PPO benchmark
score is asserted by the lesson or by this review.

InstructGPT paper: Ouyang et al., *Training language models to follow
instructions with human feedback*, arXiv:2203.02155v1, 4 March 2022,
https://arxiv.org/pdf/2203.02155v1 . Personally checked page-1 identity and
section 3.2, the reinforcement-learning paragraph and equation 2 (PDF page 9;
extracted-text lines 484–507). This learned RL policy is initialized from SFT
and regularized against pi_SFT. Pi_SFT is a distinct fixed starting policy in
this equation; it is not the PPO sampling-policy snapshot refreshed after
each rollout. This supports the reference's role in the stated SFT-to-RLHF
context, not a claim that every possible posttraining method needs a fixed
reference.

PyTorch official 2.9 API originals:
https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.detach.html ,
https://docs.pytorch.org/docs/2.9/generated/torch.log.html ,
https://docs.pytorch.org/docs/2.9/generated/torch.exp.html .
Read each complete API article from the fetched original HTML.
Detach disconnects both reverse/forward autograd and shares storage;
log is elementwise natural logarithm; exp is elementwise exponential.
The installed CPU validation version is 2.14.1+cpu, not the cited 2.9
documentation release. Relevant operations were executed on the installed
version and matched those contracts. Fetch URL, resolved URL, date and hashes
are in `official-fetch-provenance.json`; original HTML and API article text
are saved, not a prior reviewer summary.

OpenAI original PyTorch PPO implementation in Spinning Up:
https://github.com/openai/spinningup/blob/20921137141b154454c0a2698709d9f9a0302101/spinup/algos/pytorch/ppo/ppo.py .
Downloaded this exact commit from OpenAI's repository and inspected AST
function names/ranges first. Read PPOBuffer lines 12–40 and 71–84,
compute_loss_pi lines 227–243, and collection/update loop lines 295–316 and
335–354. Log probabilities are stored at collection time in a dedicated
buffer; loss recomputes current selected log probabilities and exponentiates
their difference. Its objective is the negative mean of the min surrogate
for minimization. No environment was run.

## Repository contracts read after AST location

`tiny_perceptron/posttraining.py`, SHA
`3d0e2ae3b29f95abebde9b10d0cd106639a131fab1b2c7a509e732ba8daf295b`:
read lines 1–53, including bandit_advantage 24–28 and
ppo_clipped_objective 31–50. The latter detaches old log probability and
advantage, computes ratio and min-clipped surrogate, and returns its negative
mean as policy loss. Only this small helper was executed, never training.

`scripts/course_experiments/posttraining.py`, SHA
`862dc12680fee8374679d52a1cc55a2dd325a7aebbf634668731fa7def08aa04`:
AST function inventory and selected assignment ranges first; actually read
285–290 and 324–369. Reference is an independent frozen deepcopy after SFT.
Within each rollout, no_grad captures old log probabilities and clone saves
the selected ones before epochs. Each epoch uses new policy log probabilities
with those saved old values. Avoided the result-construction dictionary at
461–533. No output JSON is required by the current section because it claims
no existing model measurement. No result notes or author conclusions read.

## Independent arithmetic, axes and execution

There are two samples (two selections), not two entries along one action
distribution axis. The bounded CPU check explicitly constructed two
normalized categorical rows with shapes (batch=2, action=2), gathered action
0 separately from each row, and recovered ratios 2 and 0.5.

For positive recorded probabilities, exp(log(new)-log(old))=new/old.
0.4/0.2=2; 0.1/0.2=0.5; changing the first new value to 0.3 gives 1.5.
Using new/new removes the measured difference, giving 1 in both entries.
Ratio is dimensionless. The target term is ratio times advantage in reward
units: 2*0.6=1.2. This is a surrogate term, not a measured doubling of quality.
Ratio 1 concerns the selected probability at the same context and action;
it does not prove the entire policy distribution is unchanged.

Original fence executed without alteration using the helper on the existing
CPU `.venv`; exit 0, printed [2.0, 0.5]. It is arithmetic only: its tensors do
not require gradient, and it calls neither backward nor an optimizer.

The independent variant enables gradient on both new/old log values and uses
the actual helper: old gradient is None, new gradient of the sum of unclipped
terms is [1.2,-0.2]. Float64 tolerance 1e-12; original float32 display is checked
after the lesson's four-decimal rounding. A detached alias changes under an
in-place edit of its live tensor, while a saved clone retains the old values.
This verifies why graph detachment alone is not historical persistence.
Log values -1000 and -999 produce exp(1)=2.718281828459045, while conversion
to probabilities first underflows to 0/0 in float64. This is a bounded
numerical illustration, not a promise of universal numerical stability.
No model was trained, evaluated, downloaded, uploaded or remeasured.

## Visual inspection and limitations

The section contains no figure reference. The two sampled entries and ratio
calculation can be understood from the stated numbers and code; no spatial
diagram or material asset is needed for this factual claim.

Actually rendered the frozen section at 1280x800 and 390x844, then opened both
PNG files with view_image. Text, minus signs, division, code and numerical
output are visible; no horizontal page overflow. This is an isolated Markdown
HTML render with a simple review stylesheet, not verification of the final
course theme. The supplementary details are collapsed in the screenshots;
their full source text was read independently.

The first Chromium attempt failed with ERR_BLOCKED_BY_ADMINISTRATOR for a
file URL. Its exact script/stderr/stdout are retained under attempt1 names.
A second actual attempt used page.set_content and completed successfully.
There was no Chromium timeout and no Inkscape fallback was used.

The review supports this finite-card/contextual-bandit example and the PPO
sampling/surrogate contracts. It makes no token-level RLHF implementation,
quality improvement, deployment, complete-training or GPU-performance claim.
