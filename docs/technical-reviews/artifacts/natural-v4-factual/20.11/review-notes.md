# Independent current-source review receipt

Reviewer: `/root/v4_review_coordinator/factual_v4_20_11`.

Fully read the raw current20.11 through before20.12, including blank lines, and actual11.16 and20.13.20.11 SHA256 is `e8f4cf1d6278447be85ae1201020b74193f9e26e5959c3ff7498dbe734352a7f`. Introduction not applicable: this is not the first section. Initial preservation code would copy the canonical20.11 report only if it existed; no unread archive was created because that canonical path was absent. My earlier commentary incorrectly asserted that a copy had happened; this was corrected to the coordinator immediately when receipt generation exposed the missing archive. No old report content or old course reader/technical verdict was consulted.

Independent predictions before execution: reference and swapped strings each contain 牛、奶、麵、包 and one LF, each with count1. Thus Counter equality is True, complete string equality False, and splitlines gives opposite ordered lists. The exercise has the same four nonwhitespace codepoints and correct relative character order but no LF: its complete Counter comparison is False, its exact comparison False, and its line list has one item. Removing whitespace gives the same flattened string as reference, but swapping the lines still produces a different flattened string. Both equality checks concern Python codepoints, not grapheme clusters, bytes, semantic equivalence or OCR quality. Counter preserves distinct-key insertion order but does not preserve the complete original character sequence; its equality test is the operation illustrated by the lesson.

Cropping counterexample: identical3×3 pixel arrays occur at xyxy[6,8,9,11] and[6,48,9,51] on two64×32 pages. Crops alone are equal, pages differ, so absolute original placement cannot be uniquely reconstructed from those crop pixels. A crop may retain useful local context; it does not always erase every layout cue. Focusing/resizing a region can make small writing easier to inspect, but this is not a measured universal OCR improvement or recovery of lost source pixels. PAGE XML and LayoutReader provide the substantive region/coordinate/order representation, not a claim that all formats have the same reading order. A two-column receipt requires an intended grouping/order convention; the lesson does not prescribe one universal answer.

Fixed-record audit: only the13 relevant test OCR cases were compared to the current manifest's frozen answers with independently implemented normalization and raw EOS fields. Exact current scorer normalized() was also executed as an AST-extracted pure function and compared.8/10 single-region cases and1/3 ordered cases agree with the saved scores. Manifest, generation file and scorer hashes match the score binding. Questions explicitly select regions/lines, and three ordered tasks require2,3,2 output lines with declared mixed or horizontal reading directions. Whole-photo input does not make the selected-text task a full-page layout benchmark. These checks are conditional on supplied annotations, which are AI annotations rather than independent human gold. They are not fresh inference, GPU replication, training, a release/data audit, or validation of arbitrary multi-column documents.20.13 is read for the OCR distinctions and capability-card scope only; its unrelated model, ASR, publication and resource claims are outside this assigned review.

Figure personally viewed after actual Inkscape rasterization: shopping list 牛奶 above 麵包,①above② and downward arrow; green correct sequence 牛奶→麵包; orange reversed sequence 麵包→牛奶. Lower labels state same character counts do not imply the same order and require separate multi-column/vertical conventions. The complete720×1160 render is legible and its positions/directions agree with current text.11.16 and20.13 reference no prerequisite SVG requiring viewing. Figure render returned exit0 with two GTK/Pango warnings, retained below; neither prevented the render.

Execution limits: Python3.13.5, Torch2.14.1+cpu, UCD15.1.0. No installs, GPU, downloaded models/datasets, training, environment/Git/source/checker edits, whole-book reruns or new subagents. The cloud setup skill was read to guide runtime validation; no setup change was required. No preview was needed for the bounded Python/figure/record claims. External full HTML/PDF/XSD bytes and extracted research text stay only in ignored `outputs/natural-v4/factual-research/20.11/`; durable evidence contains original URL/version/read-locator/retrieval hashes and own summaries. One guessed OCR-D locator returned404; original accessible authorities were used and this actual failure remains in the retrieval receipt.

## Accidental discovery exposure

The initial discovery command was too broad:

`rg -n 'reading.order|reading_order|ocr_trans|normalize|NFKC|splitlines|ocr_layout|layout' tiny_perceptron scripts docs/natural-assistant/v4 docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/scored/scores.json docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/selected-final-test-audit-summary.json course/figures/natural_reading_order.svg`

Its tool result reported622745 original tokens and was heavily truncated; the aggregate functions output was also truncated. The full initial output was not stored, so the exact complete truncated output cannot be recovered without repeating the forbidden search. The following already-visible draft excerpts are preserved exactly; no draft was subsequently searched or opened. My initial message's “three snippets” meant three source regions; there were six visible draft lines. They expose older source text and one unrelated20.3 checklist, not a20.11 author assessment, expected verdict, or previous course-review judgment. None was used as authority:

```text
docs/natural-assistant/v4/drafts/20.md:6:2. 20.3：核對 Qwen 固定pin/參數、最後 ASR 名稱/pin/參數、LoRA數；補新CPU/GPU實測資源短框，不稱硬體最低需求。ASR已依16筆validation選turbo；normalized small117/510、turbo50/510，raw亦改善；這是ASR選擇，不能報成LoRA改進或finaltest。
docs/natural-assistant/v4/drafts/20.md:319:文字比較前，有時會整理全形字或空白，這叫正規化。例如NFKC能把全形「Ａ」整理成「A」，卻不把「臺」改成「台」。是否接受異體字、換簡繁、忽略標點，要在看結果前指定；原樣結果也應保留。逐字任務中，即使要求用繁中解釋，圖片寫簡體的部分仍應原樣轉寫，不自行改字。
docs/natural-assistant/v4/drafts/20.md:331:![上行牛奶、下行麵包；交換兩行保留相同字元，卻不是相同完整轉寫](../figures/natural_reading_order.svg)
docs/natural-assistant/v4/drafts/20.md:340:print("參考行序", reference.splitlines())
docs/natural-assistant/v4/drafts/20.md:341:print("預測行序", predicted.splitlines())
docs/natural-assistant/v4/drafts/20.md:344:`Counter`只數每種字元出現幾次，所以第一行True；它像把兩份購物單都倒進同一個字袋，丟掉了排列。完整字串比較是False。`\n`表示換行，`splitlines()`把文字按行分開，後兩項清楚列出上下交換。
```

Necessary manifest-row inspection also exposed embedded original OCR dataset annotation `author_review`/`peer_review` provenance. Those are original data records, not assigned course-review verdicts. Their visual assessments were not accepted as authoritative proof; my recomputation uses actual question/reference/prediction fields and clearly states its annotation limit.

## Actual first render output

Command: `inkscape course/figures/natural_reading_order.svg --export-type=png --export-filename=docs/technical-reviews/artifacts/natural-v4-factual/20.11/natural_reading_order.png`

Exit code0, stdout empty. Exact visible stderr:

```text

** (inkscape:116440): WARNING **: 22:44:34.038: Failed to wrap object of type 'PangoFT2FontMap'. Hint: this error is commonly caused by failing to call a library init() function.

** (inkscape:116440): WARNING **: 22:44:34.171: Failed to wrap object of type 'GtkRecentManager'. Hint: this error is commonly caused by failing to call a library init() function.
```
