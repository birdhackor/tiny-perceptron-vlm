Independent factual review of 20.10
=================================

I read all raw bytes of 20.10, including the blank lines before 20.11, and all of
the linked necessary 11.16 and 20.13. 20.10 is not the first section, so the chapter
introduction is not applicable. These three sections contain no course figures.
No notebook/browser preview was necessary: the current raw lesson code was
extracted and executed directly in the actual project virtual environment.

Hand calculation and unit boundaries
-----------------------------------

Reference positions are 今 / 天 / 去 / 臺 / 北, hence N = 5. The prediction differs
only at position 4 (台). It cannot have distance zero because the strings differ;
one substitution suffices, so the minimum distance is exactly 1. CER = 1/5 = 0.2.
The five characters each occupy three UTF-8 bytes (15 bytes total), but the
implementation counts five Python Unicode code points.

For the exercise 今天去臺, 北 was omitted from the prediction. Repairing the
prediction to the reference requires insertion of 北; describing the prediction
as having deleted 北 uses the opposite transformation direction. Both distances
are 1. The reference denominator remains 5 and CER remains 0.2. 今天去台 needs
one replacement and one insertion, distance 2 and CER 0.4. An extra 北 also costs
one edit. A reference of one character with three added characters gives CER
3/1 = 3; this metric is not universally bounded by one. An empty reference gives
division by zero and the code explicitly returns None, including empty versus
empty. That does not prevent an exact comparison or a separate false-positive
test.

Python strings count code points, not bytes and not necessarily visible grapheme
clusters. For example e + combining acute has two code points. The lesson's five
Han characters each happen to be one code point. OCR-D defines its own character
unit as grapheme clusters in NFC and its benchmark uses a separately normalized
CER; I used its explicit standard (i+s+d)/GT-length formula and edit definition,
not those alternative OCR-D choices, to verify this lesson's disclosed metric.

Executed checks
---------------

The exact lesson fence produced False, 1, 5 and 0.2. The actual environment is
Python 3.13.5, Torch 2.14.1+cpu, UCD 15.1.0; CUDA is unavailable. A breadth-first
graph of individual insertion/deletion/substitution operations independently
checked all 961 ordered pairs of strings of lengths 0 through 4 over 甲/乙,
with no discrepancy against the project dynamic programming implementation.
This checks short, nontrivial omission/insertion/reordering cases without merely
copying its recurrence. There was no model loading, training, installation or GPU
use. The initial successful probe is retained. Its only later correction was an
inaccurate hardcoded scope sentence about family metadata; the actual difference
list was empty in both runs.

NFKC changed Ａ to A and fullwidth comma to comma, but left 臺, 台, 絕味鴨脖 and
绝味鸭脖 distinct. It does not perform Chinese script conversion. The production
OCR scorer ignores Unicode whitespace for the single-region task but preserves
internal newlines for the ordered task. It preserves case and punctuation after
NFKC. The actual comparisons confirm those rules. Raw versus normalized results
are distinct measurements; UAX #15 explicitly cautions that compatibility
normalization erases distinctions, supporting retention of original text. The
policy to transcribe the pictured script rather than translating it is a task
specification, consistent with the frozen simplified-script target 绝味鸭脖.

Task and source boundaries
--------------------------

A nonempty response to an actually empty reference inserts unsupported content.
Presence detection, legible transcription and indeterminate/blurred text are
different conditions. Neither this toy code nor a benchmark presence score
establishes that a model handles blank or unreadable images reliably.

I read the original CC-OCR v3 Introduction, sections 3.1–3.3, Appendix A.1 and
A.4.1: they distinguish scene, document, short-line and full-image tasks, include
blurred and differently oriented text, and document semantically plausible
insertions/content changes. I did not import its numbers or metrics into the
course. I read the original SynthText paper pages 1–4, especially sections 2.1–2.3:
font/text selection, local geometry, perspective, colour/background and lighting
are controlled separately. This supports the lesson's domain distinction, not
an assertion that synthetic data can never generalize. I rendered and personally
viewed original paper page 3 / Figure 3 in ignored research: text-free RGB,
predicted depth, segmentation and filtered regions appear above four renderings
with generated text and bounding boxes, consistent with its caption. This is an
external source figure, not a figure referenced by the lesson or prerequisites.
No complete original paper/HTML/image dataset is copied into public evidence.
The pinned NVIDIA original README confirms synthetic document generation and
separate full-page and line/word annotations; no data shards were downloaded.

The necessary capability-card link was checked narrowly. I independently matched
31 test targets, prompts, image paths and family fields to fixed original GPU
prediction records and recomputed exact normalized comparisons plus recorded EOS
completion: presence 18/18 (10 positive, 8 negative), transcription 8/10 and
ordered transcription 1/3. All 31 records end with a recorded EOS token within
the 384-token cap. They use 18 unique real photographs, reuse positive images
across questions, and include no synthetic final-test tasks. The presence query
asks about clearly visible Chinese text, so its negative examples are not a
claim of completely blank backgrounds. These are conditional counts using
frozen reference labels, not my independent visual re-annotation, my GPU
inference, or a claim of whole-page competence. The saved GPU runtime identifies
the base model, seed, split, software versions, device and token budget in the
probe result. No speed/quality generalization is inferred from these records.

The first CC-OCR PDF retrieval was deliberately bounded at 5 MB and failed that
bound; the genuine attempt is retained in the source receipt. Its original v3
HTML then succeeded, so this is not an unresolved source blocker.
