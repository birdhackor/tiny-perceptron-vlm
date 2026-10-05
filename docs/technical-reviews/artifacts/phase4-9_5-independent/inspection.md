# Independent factual inspection of 9.5

Reviewer: `/root/phase4_factual_coordinator/factual_9_5`, fresh independent task; inspected on 2026-10-05. No other reviewer was delegated. No course text, figures, implementation, training, or Git commits were changed.

## Access scope and independence

Personally read the entire current `course/chapters/09.md#9.5`, source lines 148–178. The source snapshot retains original UTF-8 bytes without newline normalization. This is not the first section of the chapter; the introduction was not reviewed. Read current sections 9.3 and 9.4 as prerequisites: manual comparisons and the stipulated two valid box scenarios. Their empirical discussions are context only, not claims accepted or re-reviewed for this report.

Read the required factual instructions, schema checker, round checker, section extraction/CPU helper, and clear-tutorial SKILL.md/review-protocol.md. Read the actual build bootstrap and `tiny_perceptron/alignment.py`; the latter has no role in 9.5's Python-only fence. There is no imported repository helper, model, optimizer, backward call, relevance classifier, result JSON, figure, axis, unit, accuracy denominator, or training recipe in this section. The training link is supplementary; the fence works without running it.

The first `rg --files` listing unintentionally included old artifact filenames. This exposed paths only, no reviewer contents or judgments; it was immediately reported to the coordinator, who confirmed path-only exposure was not judgment contamination. Subsequent discovery was limited to current source/code and lookup-only locator indexes. No old reader or factual report, history, dispatch summary, author's notes, other person's inspection, or other person's execution proof was read. Locator indexes were used only to locate candidate source bytes/versions. No claims were adopted from them. The external files actually used were downloaded afresh from their original HTTPS authorities.

## Personally inspected external originals

1. OpenAI, **Model Spec**, fixed **2025-04-11** historical version, `https://model-spec.openai.com/2025-04-11.html`, fetched HTTP 200 with ordinary TLS verification. Personally checked date, title, overview, and the historical-version banner in the raw page/text. The author/provider's official dated document describes intended behavior; it is normative support, not a measurement or a claim about the current deployed policy.
   - `#assume_best_intentions`: extracted text lines 1676–1679, refuse only when applicable requirements call for refusal; lines 2000–2090, paragraph beginning “If the user asks for prohibited help to accomplish a permissible goal” and the doctor-signature/insurance example, supports refusing a prohibited means while offering help with the permissible goal. It does not certify any model's ability to recognize authorization.
   - `#do_not_facilitate_illicit_behavior`: lines 5992–5995, actionable steps facilitating illicit behavior remain prohibited; appropriate alternatives may be offered without shaming. Supports the box scenario's stipulated boundary and the warning against providing a way to perform the just-refused action. It does not say every password or box question is illicit.
   - `#refusal_style`: lines 17173–17176 and the concrete examples to line 17450, refusals should be neutral and succinct, never preachy. Supports “long lecture is not automatically better,” not a universal requirement that every refusal have exactly one sentence.
   - `#be_thorough_but_efficient`: lines 17567–17594, an immediately usable artifact, including a complete email message, is preferable to a partial artifact when appropriate; uninformative/redundant text wastes time. Supports the offered copyable owner-contact message and the relevance/length distinction; it does not rank this toy text through a formal empirical metric.
2. CPython official source at **v3.13.5**, matching the installed Python 3.13.5:
   - `Doc/tutorial/controlflow.rst`, lines 43–70: `for` visits a sequence's items in order, with a print example.
   - `Doc/library/functions.rst`, lines 1603–1618: `print` converts arguments to strings, uses a space separator/newline by default, and writes to stdout.
   - `Doc/library/stdtypes.rst`, lines 811–815 (the two boolean constants), 4611–4645 (mapping and dictionary literal), 4719–4722 (`d[key]` returns a stored item), 4749–4751 (stored values can be assigned). These contracts support the single software-coverage claim; they do not provide semantic judgments about answer helpfulness.

HTTPS URLs, final URLs, HTTP status, raw-byte sizes, and SHA-256 hashes are recorded in `sources/fetch-receipts.json`. The raw files, rather than the locator summaries, are the permanent evidence. `model-spec-2025-04-11.txt` is a local HTMLParser text extraction to make inspected paragraphs locatable; the original HTML is retained.

## Execution and interpretation

Ran the original Python fence through the repository helper in a fresh process with its ordinary bootstrap, timeout 45 seconds, offline/audit guard, CPU requested. Exit 0 in about 1.93 seconds; no guard events. Environment: Python 3.13.5, PyTorch 2.14.1+cpu; CUDA build None, CUDA unavailable. Original fence SHA-256: `9708e22b4e4312087dfa50cff4c4182ae16f7c5a94686df643a24d7fbb1f8384`.

The three output rows are exactly the source's prescribed strings and boolean pairs `(True, False)`, `(True, True)`, `(True, False)`. Count and spacing were checked exactly; no floating-point tolerance applies. They are three hand-authored examples, not three trials estimating model accuracy.

Also ran saved `variants.py` under Python 3.13.5 with a 15-second timeout and explicit offline/CPU environment. Exact original re-execution succeeded; AST inspection found only the `print` call and no imports. A text-only weather replacement retained the manually stored True relevance label, showing there is no semantic recalculation. The prescribed explicit label edit produced True/False while preserving the boundary label. The copyable owner-contact text and allowed public-test-record answer are saved human examples, not model output or automatically scored semantics.

Manual semantic inspection is conditional on 9.4's toy-world boundary: none of the three original answers supplies a code; contacting the owner is a way to pursue return of the item with authorization, while the unrelated weather sentence gives no next step for that goal. The offered contact wording itself neither discloses a secret nor gives a circumvention method. The text explicitly invites a second annotator to question adequacy; it does not assert objective, exhaustive or universally valid semantic labels.

## Figure and empirical scope

The raw section contains no image, SVG, HTML image reference, or required unseen visual material; the comparison is fully present as literal answer strings and stored labels. `figure_sha256` is empty and rendering is not applicable. There is no scientific numeric result or original model-result JSON to recompute, no measured model capability, and no hash/configuration claim beyond the actual source/program versions saved here. No model inference, dataset/model download, GPU execution, full training, paid work, weights, or later-stage engineering was performed.

## Judgment

The four grouped substantive claims in the report cover the relevant alternative, original/variant software behavior and limitations, same-boundary/allowed-request distinction, and non-preachy/useful wording. All are supported within their stated scopes. No unresolved substantive issue was found. Formal checker passage only checks schema, source/artifact identity, and reviewer declarations; it is not the factual argument.
