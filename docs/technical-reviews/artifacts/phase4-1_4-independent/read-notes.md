# Independent phase 4 factual review of 1.4

Reviewer: `/root/phase4_factual_coordinator/factual_1_4`, fresh context.
Read the current `course/chapters/01.md` lines 1–260, with the complete target section at lines 125–158. Read the complete review method and checker schema; read the extraction/CPU helper and the build helper BOOTSTRAP contract. No old technical report, history, reader conclusion, or source-candidate summary was used. The target has no figure reference, recipe, repository helper import, training, or measured evaluation result.

## Original authorities actually inspected

- Jurafsky and Martin, *Speech and Language Processing*, chapter 3, author-hosted Stanford PDF, draft August 19, 2026. Read extracted lines 1–181, 219–233, 588–628, 735–785. Pages 2–4 define an n-gram model using n−1 previous units and count normalization; with n=1 the context is empty. Pages 14–15 explicitly give unsmoothed unigram probability `P(w_i)=c_i/N`, with N the total number of tokens. Pages 11–12 define sampling and unigram interval widths proportional to frequency. The source uses words/tokens; this section deliberately takes each Chinese character as a unit. It supports the method and sampling principle, not an accuracy claim for this five-character demonstration.
- Python 3.13 documentation, currently labelled 3.13.16. `collections.Counter`, extracted lines 734–812: an iterable supplies elements; counts are stored as dictionary values; the documented string example counts characters; insertion order is preserved. The local interpreter is CPython 3.13.5. The relevant Counter contract predates both patch versions, and the local actual implementation at `/usr/lib/python3.13/collections/__init__.py` lines 599–611, 618–620, and 687–705 was also inspected and snapshotted.
- Python 3.13 built-ins, `abs` lines 617–624 and `sum` lines 3880–3908: absolute value and aggregate total. The improved float-sum algorithm is documented as introduced in Python 3.12; the review makes no assertion that a particular probability sum must exhibit a nonzero rounding error.
- Python 3.13 floating-point tutorial, lines 255–269, 314–390, 417–488: binary storage approximates most decimal fractions; exact decimal equality can fail; approximate comparisons are appropriate. This supports the rationale for the displayed absolute tolerance, not a guarantee that checking the total catches every possible omitted or miscounted item.
- Python 3.13 `random.choices`, lines 592–628: selection according to relative weights. Used to verify the sampling concept only; the chapter fence does not call this API.

All source URLs, final URLs, access timestamps, ETags, byte counts, and SHA-256 values are in `source-acquisition.json`. Raw HTTPS responses are retained beside the text extractions. Full documents were fetched; the paragraph ranges above were actually inspected and have the supporting scope listed here independently.

## Actual bounded CPU verification

Executed the exact original fence bytes with `.venv/bin/python` (CPython 3.13.5), then its one-literal exercise variant. The output is preserved in `verify.stdout.txt`, with the command and environment in `environment.json`. Base counts are 3,1,1 over 5 tokens, probabilities .6,.2,.2; the exercise counts are 3,3,1 over 7 tokens and cat probability 3/7, rounded .429. Exact `Fraction` arithmetic checks the denominator and total mass. Additional bounded checks permute input order and duplicate the entire bag, preserving probabilities while changing counts as expected. Greedy max returns cat; exact intervals [0,3/5), [3/5,4/5), [4/5,1) establish nonzero dog/bird sampling chances without inventing sampled accuracy. `abs`, `1e-6`, and a decimal rounding example are executed.

The chapter fence only calculates a dictionary. It consumes no prefix, samples no character, computes no gradient, updates no model, and measures no prediction accuracy. The sentence that guessing cat may happen to succeed is a possibility statement, not evidence that this model works on actual language. No smoothing or unseen-character handling is claimed. The assertion verifies total mass in the supplied three-candidate dictionary; the reviewer separately checks membership and counts.

## Independent decision

Pass. Every substantive statement is supported within the explicitly defined candidate bag and unsmoothed character unigram example. No factual correction or unresolved issue was identified. No textbook, figure, training artifact, or environment configuration was changed.
