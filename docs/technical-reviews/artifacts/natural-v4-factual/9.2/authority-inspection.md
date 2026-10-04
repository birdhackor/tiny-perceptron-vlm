# Original authorities actually read for 9.2

Access date: 2026-10-04. These are my own summaries. Complete third-party PDFs
and HTML were retrieved only into ignored `outputs/natural-v4/factual-research/9.2/`.
The small public `authority-retrieval-receipts.json` records original URLs,
resolved URLs, HTTP status, byte size, version where needed, and raw retrieval SHA.
No candidate-source summaries or previous reviews served as authority.

1. **Rajpurkar, Jia and Liang, ACL 2018 published version**, *Know What You Don't
   Know: Unanswerable Questions for SQuAD*, P18-2124, published pp.784–789.
   Original PDF: <https://aclanthology.org/P18-2124.pdf>.
   Read the abstract and §1 (p.784), §2 Desiderata (p.785), §4.1 (p.786),
   §§4.2, 5.1–5.2 and footnote 3 (p.787), and §6 (p.788). The motivating
   distinction is an answer supported by the given context versus a plausible
   unsupported guess. §2 explains why word overlap and answer-type cues are weak
   substitutes for determining answerability. §4.1 combines answerable and
   unanswerable questions within article-based splits. §5.1 evaluates an
   abstention prediction with a development-set threshold, separate from answer
   extraction. §5.2 explicitly exposes the always-abstain baseline; footnote 3
   gives abstention credit on negatives and zero to other answers. These support
   the lesson's evaluation design and information-relative notion of support.
   This is extractive text QA, not evidence of our ball model's numerical scores,
   image recognition, dialogue training success, or internal self-awareness.

2. **Ouyang et al., arXiv:2203.02155v1**, *Training language models to follow
   instructions with human feedback*, revision v1 dated 2022-03-04.
   Original PDF: <https://arxiv.org/pdf/2203.02155v1>.
   Read §3.1 step 1 (PDF p.6), §3.5 SFT (p.8), §3.6 (pp.9–10), and
   appendix B.2/Table 10 (pp.36–37). Prompt-specific demonstrations are used
   for supervised fine-tuning; this supports creating dialogues with alternative
   intended answer behaviors, without promising that two examples alone produce
   robust generalization. §3.6 distinguishes measured truthfulness from private
   model beliefs, which the authors cannot infer. B.2 notes unwanted over-refusal;
   Table 10 describes clarification and explaining confusion, and avoiding
   unsupported extraneous context. Our lesson uses an explicit finite operational
   rule for honesty; it does not adopt an assertion about a model's private beliefs.
   No InstructGPT experimental score is imported into the lesson.

3. **Python Software Foundation, exact release 3.13.5 language reference**.
   <https://docs.python.org/release/3.13.5/reference/expressions.html>.
   Actually read §6.3.2 Subscriptions (`#subscriptions`), §6.10.3 Identity
   comparisons (`#is`), and §6.13 Conditional expressions
   (`#conditional-expressions`). Dictionary subscription selects the named key;
   `is not` negates object identity. The conditional evaluates its condition
   first and evaluates only the chosen result expression. `is not None` does
   not reject zero. The wording 'has a value' in the lesson is valid within its
   specified integer-or-None data contract, rather than a generic validator.

4. **Python Software Foundation, exact release 3.13.5 built-in constants**.
   <https://docs.python.org/release/3.13.5/library/constants.html#None>.
   Read the `None` entry: absence-of-value use and the sole instance of NoneType.
   Here absence is the author's explicit missing-observation convention, not
   evidence that a photograph was actually analyzed.

5. **Python Software Foundation, exact release 3.13.5 built-in functions**.
   <https://docs.python.org/release/3.13.5/library/functions.html#func-str> and
   `#print`. Read both actual entries: `str` returns a string version of the
   object; `print` converts arguments to strings, uses default one-space
   separators, and follows the result with a newline. The actual CPython 3.13.5
   execution independently confirms the exact integer and None output here.

6. **Original project safety code at the recorded run revision**.
   <https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/a864a60bbf72583afc9bbaf45e052bd4fe076c62/scripts/course_experiments/behavior.py>.
   Actual retrieval SHA `94ab92aa1b8017edbb8f5524ee598dce93756cf295377ac21515bf3c0a05bdc0`
   matches the run record. Read `_safety_records` (original lines 363–418),
   `_safety_evaluations` (421–451), and `run_safety` (490–521). The unknown-count
   labels ignore hidden quantities. The mixed variant adds arithmetic train
   records, then wording evaluation runs after both training loops with no new
   fitting. Direct whole-file diff shows current changes at imports and the
   unrelated style-metric function. The four relevant function ASTs match.

7. **Original project public model bytes, already present locally**.
   Manifest: `docs/course-experiments/public-models.json`, safety entry,
   public revision `308aa207d6917d5dc8f9c819f2d7257d5cad7f8e`.
   Pinned authority URL:
   <https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e/course/course-v1/safety/model.pt>.
   This task did not retrieve or download weights; it read the already present
   original checkpoint bytes, verified SHA `0271ddaf1320059110cfaf0e10d0477613f510b741f4299b7976250b36cf09a8`
   and size 575558 against that pinned manifest, inspected config/metadata,
   loaded using `weights_only=True`, and ran frozen CPU inference. The checkpoint
   metadata points to the same experiment and original code revision as the
   recorded report. All 23 examined output token sequences match the report.

Registered current local sources are the independently read relevant regions of
`behavior.py`, all of `common.py`, `text.py` split/save/arithmetic functions,
all of `data.py`, all of `model.py`, `training.py` seed/save/load functions,
the original `results/safety.json` metadata and mixed evaluation samples, and
the safety entry of `public-models.json`. The report records each current full
file SHA. Reading a relevant region is not a claim that every unrelated function
in a long repository module received a complete technical review.
