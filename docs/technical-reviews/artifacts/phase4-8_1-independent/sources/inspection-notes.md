# Actual original-source inspection by factual_8_1

No previous factual or reader verdicts, report prose, or interpretation notes
were read. The shared locator index was used only to identify the cached
InstructGPT PDF. A fresh HTTPS download from arxiv.org/pdf/2203.02155v1 had
exactly the same SHA-256 as that cache, verifying the original version bytes.
The paper's first page says arXiv:2203.02155v1, 4 Mar 2022, and identifies the
original authors and OpenAI affiliation. The source is the researchers' own
paper. The downloaded PDF and pdftotext extraction are archived here.

I read InstructGPT sections 3.1 (PDF page 6: Step 1, demonstration data and
supervised policy), 3.2 (page 6: deduplication and train/validation/test split),
3.5 (page 8: supervised fine-tuning on demonstrations), 3.6 (page 10: following
user intentions and distinct truthfulness assessment; held-out prompt evaluation),
and Appendix B.1's labeling-instruction box (page 37). Those passages support
specific desired demonstrations, explicit task assessment, and separate tests
of task helpfulness and factual truth. They do not establish that a chopstick
analogy model was trained here, or measure creativity. Held-out evaluation
supports the need to test beyond repeated training responses; it is not a
sufficiency theorem that one new arithmetic question proves broad competence.

I fetched the official Carnegie Mellon Eberly Center page “Grading and
Performance Rubrics,” and read “What are Rubrics?” and “Advantages of Using
Rubrics” (extracted text lines 17-23). It defines a rubric as an explicit
scoring tool for performance expectations, divided into component criteria
with described mastery levels, and recommends recording component scores.
The two binary criteria in 8.1 are a simple instance. The page gives no
automatic NLP grader and does not supply these particular chopstick labels.
The site has no published immutable release tag, so the exact downloaded
HTML SHA and access date identify this inspected official-page snapshot.

I fetched RFC 8259 (December 2017) from the RFC Editor and read the Abstract,
sections 1-4, and section 6's number grammar. It is the Internet Standards
Track JSON specification from the IETF. It defines JSON as textual structured
data serialization and an object as string name/value pairs. The example
{"answer": 4} is an object with name answer and number 4. Section 2 also
permits scalar JSON values: the introductory sentence describes the object
example rather than requiring every JSON document to have named fields.

I fetched CPython's own documentation source tagged v3.13.5, matching the
installed Python 3.13.5 release. I read the dictionaries and looping sections
(Doc/tutorial/datastructures.rst lines 499-579). The basic contract is storing
and retrieving supplied values by dictionary key and traversing a sequence.
This documentation and actual fence execution support the single program
coverage claim; separate claims per dictionary key or print call are unnecessary.

For the saved empirical statement I read the original style.json metadata,
after validation/test samples and rubric objects, default-style comparisons,
and same-weight prompt probes as structured data. I extracted and read the
historical sources using git show at its recorded revision
ae7bbbf95537d228a44810041d2a9e978360d369. I read behavior.py lines 20-162,
common.py split_records and evaluate_lm, text.py arithmetic_records and
_save_splits, and data.py ByteTokenizer plus model.py generate. The archived
metric at that revision has no invalid-control-token guard; current code has
an added guard. The audit therefore uses the historical metric and separately
checks saved raw tokens, which have no hidden answer control tokens and agree
with the visible UTF-8 text. Exact regenerated JSONL hashes independently
bind the arithmetic, style and date row order to the saved result. This is
verification of archived samples; no model weights or GPU computation ran.
