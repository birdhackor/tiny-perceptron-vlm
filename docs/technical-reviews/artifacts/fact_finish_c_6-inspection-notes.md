# C.6 personal inspection record

Reviewer: `/root/fact_finish_c_6`, fresh technical review, read on 2026-10-04.

I first read the entire `outputs/integration-runs/root-technical-instructions.txt`
and `docs/technical-review-guide.md`. I then personally read all of C.6 and the
explicitly linked prerequisite C.2. I did not read prior review conclusions,
author history or dispatch records. The previous C.6 report was preserved as
opaque bytes in a history file before replacement.

The source sections were extracted using `scripts.check_technical_reviews.sections`
and saved exactly as UTF-8. C.6 read SHA is
`720bb4198892489dbd56f2aa7ca0db5acaca9532fd7cbb2c266a9cbfc51c6edf`;
C.2 read SHA is
`6abf752dfad6dff873d0d90bdc85648865e7e713330ebc6a0965330bc3d69167`.
The first `##` in the chapter is C.1, so C.6 does not require an introductory
section review. Neither C.6 nor this prerequisite has an image reference; there
is no figure XML or rendered image to inspect for this review.

## Personally inspected original sources

* DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning*, arXiv `2501.12948v1`, PDF pages 5–6, sections
  2.2.1 and 2.2.2. Downloaded from the version-specific original arXiv PDF
  using `curl -fL --max-time 45 https://arxiv.org/pdf/2501.12948v1`, saved the
  original PDF and `pdftotext -layout` output, and read the objective and reward
  modeling paragraphs. Section 2.2.1 samples outputs then optimizes a policy;
  section 2.2.2 distinguishes correctness and format rewards, describing final
  answers in specified formats for deterministic mathematics and predefined
  code tests. This supports the general reward/optimization distinction and
  the relevance of parsing criteria. It does not establish this project's
  binary reward, results, REINFORCE implementation, or general reasoning quality.
  C.6 makes no historical claim about R1; I have not added one.
* Python 3.13 official `re` and built-in-types HTML, obtained with `curl -fL`
  from `https://docs.python.org/3.13/library/re.html` and
  `https://docs.python.org/3.13/library/stdtypes.html`. I read the actual
  `re.fullmatch`, `str.strip` and `typesnumeric` paragraphs. They specify
  whole-string matching, a None result on failure, whitespace stripping when
  chars is omitted, and booleans as an integer subtype with numeric constructors.
  The executed interpreter was Python 3.13.5. ASCII `[0-9]` in this code also
  deliberately excludes full-width digits; the exercise uses short strings,
  rather than claiming a general parser for unlimited untrusted inputs.
* `scripts/course_experiments/applications.py`, lines 128–144, 679–748,
  813–951 and 1081–1122: personally read the step scaling, arithmetic data and
  verifier, 18→48→17 finite policy, both reward rules, multinomial evaluation,
  sampled REINFORCE loss and training entrypoint. `common.py`, lines 50–78
  and 205–241: family split, record SHA encoding, actual optimizer updates,
  and checkpoint saving. Full current-file SHAs equal those recorded in the
  original reasoning run; the replay asserts this identity.

## Direct calculations and execution

The original C.6 fenced code was executed from the read snapshot, first unchanged,
then with exactly `truth = 4` changed to `truth = 5`. It yielded the stated
parsed values and reward vectors. Extra short inputs test whitespace, negative
numbers, leading zeroes, plus signs, full-width digits, empty strings and extra
words. The C.2 counterexample independently evaluates to equations
`[False, True]`, linked `True`, final `True`. Final-answer equality therefore
does not inspect or repair steps. A long erroneous explanation still cannot
become correct merely by having more characters.

I checked all 17 actions for all truths 0–15, giving 272 strict/proxy score pairs.
The three original operands are each 0–5, so `0 <= a+b+c <= 15`. Enumeration
action 16 contains every supported truth as a space-separated token and always
earns proxy reward 1, while strict fullmatch rejects it and gives 0.

I read the original CUDA run configuration and complete weak-policy after-training
records, and saved the complete finite-policy subsection, not merely its totals.
I inspected the 24 after-training rows, their 17 probabilities and every sampled
action ID/reward triple. The CPU audit reconstructs all 216 arithmetic questions,
the seed-42 family split, each split SHA, all test truths and all 384 candidate
strings, IDs and scores. Counts match the original report: enumeration 384/384,
proxy 384/384, strict 0/384. The training record is complete: 1200 steps × 64
action draws = 76800, with no autoregressive token count. Test families are
disjoint from train and validation families.

The first pinned private-checkpoint request returned HTTP 401. Its command and
result are preserved in a separate receipt. Subsequently I personally obtained
the anonymous public inference checkpoint at immutable revision
`ac5ac599faabcb026a159af09433ec0490a397f7`, checked its 10561-byte size and SHA
`3247f10643a85f9c121ecad47240f80833317d859efa34d207cae6b89842d650`, and checked
the original export-manifest bytes/SHA. The export manifest links this file to
the original private checkpoint SHA
`31801a24084b1e4053f4ff12bb24f6805d9c709a2baf3582670c92949155d356`, matching
the original raw run's artifact. I loaded only this public export using
`torch.load(..., map_location='cpu', weights_only=True)`, checked all four tensors
for shape, dtype, finite values and individual SHA, and loaded them strictly
into the personally inspected `_FinitePolicy` class. Metadata matches the
original code revision, and inputs are exactly three six-way one-hot operands.

CPU probabilities differ from the recorded CUDA probabilities by at most
`1.1920928955078125e-7`, within a stated absolute tolerance `2e-7` for float32
backend rounding. Probability of enumeration is about 0.999883–0.999964, not
mathematically 1. Fresh CPU seed-42 multinomial draws also produced 384
enumerations, 384 proxy successes and zero strict successes. This is an
independent sample, not a claim to reproduce the original CUDA RNG stream.
The raw run's observed 384/384 is a finite count; it is not a guarantee that all
future samples will select enumeration. No training was repeated, no GPU
speed was measured, and no language-model reasoning claim follows.

The replay code and execution JSON preserve the actual command, environment,
outputs, input identities, all reward pairs, candidate audit and checkpoint
metadata/tensor receipts. Binary weights are not saved to review artifacts;
the code redownloads the exact public pinned file and validates it before use.
