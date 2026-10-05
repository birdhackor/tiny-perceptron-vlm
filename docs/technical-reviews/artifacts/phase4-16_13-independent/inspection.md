# Independent factual inspection of 16.13

Reviewer: `/root/phase4_factual_coordinator/factual_16_13`; date: 2026-10-05.

I read the current raw section 16.13, its referenced SVG, the necessary prior sections 16.9 and 16.12, the factual reviewer instructions, checker schema, `section_facts.py` interface, clear-tutorial skill and review protocol. The frozen full chapter is an input snapshot, not a claim about the latest entire chapter. The formal section fingerprint uses original UTF-8 section bytes. There are no Python, shell or other fences in 16.13 and no pre-existing experimental score JSON to verify for this section. I did not read another review report or an author's results/revision summary. The initial filename locator returned artifact filenames but not their content; subsequent source inspection was limited to exact files. No substantive review contamination event occurred.

I personally downloaded and read these original sources, with exact version URLs and original-file SHA-256 preserved in the source snapshots/manifest:

- FlashAttention, arXiv 2205.14135v2: section 2.2 (row-wise softmax and materialized scores/probabilities), sections 3.1–3.2, Algorithm 1, Theorems 1–2; section 3.3 (a distinct block sparsity mask and skipping zero blocks); Appendix B.3, Algorithm 2 (explicit mask applied within each tile). PDF text locators: lines 207–319, 362–416 and 1087–1140 in the pdftotext derivative.
- Transformers official `masking_utils.py`, tag v4.57.1: AST-located `and_masks` lines 46–57, causal predicate lines 74–78, sliding overlay lines 81–90, chunk overlay lines 93–102, legacy overlay lines 105–114, sliding and chunked factories lines 117–130; I also read `eager_mask` lines 475–522, whose contract constructs a dense 4D mask. AST selection preserves the exact official function definitions. CPU execution uses the torch>=2.6 branch and zero left padding. I do not claim the whole Transformers package or a production attention backend ran.
- FlashAttention official README, tag v2.8.3: `flash_attn_qkvpacked_func` and `flash_attn_func` contracts, lines 218–266, document separate causal and local window settings. `Tests`, lines 521–526, explicitly uses numerical tolerance rather than bitwise equality.
- BigBird, arXiv 2007.14062v2: section 2, global token construction, text lines 202–221; Appendix D block implementation, lines 1860–1879, connects local and global/random key blocks. This supports the existence of other cross-group sparse architectures, not a property of the simplified causal mask in this lesson. pdftotext emitted an xref reconstruction message while succeeding; I read the recovered source passages and did not use any model results from this paper.
- Swin Transformer, arXiv 2103.14030v1: section 3.2, shifted window partitioning in successive blocks, text lines 234–279. Consecutive blocks alternate partitions to introduce cross-window links. This is a vision architecture existence example; it is not represented as a language-model causal mask or a result of this lesson.

Independent math and CPU checks:

- For six query positions and six key positions, each grid contains 36 possible cells. Rows are queries and columns are keys, not reversed. All three masks prohibit keys with index greater than the query.
- Full causal row lengths are 1,2,3,4,5,6, giving 6×7/2 = 21 allowed pairs. Sliding width three includes self, giving 1+2+4×3 = 15. Fixed groups 0–2 and 3–5 give 2×(1+2+3) = 12. Counts are exact integer pairs, not timing, accuracy, memory bytes or a proportion with an unstated denominator.
- The original official mask factories were executed on CPU and agree with separate direct predicates for all 36 cells per rule. Position 3 yields [0,1,2,3], [1,2,3], [3]. Position 4 yields [0,1,2,3,4], [2,3,4], [3,4]. The sliding rule crosses the fixed group boundary.
- SVG parsing checked every one of the 108 data cells against these independently obtained masks. I rendered and viewed the actual SVG at 640×1175. Its axes, 0–5 labels, blue/gray legend, position 3 captions and visible patterns agree. No arrows convey a conflicting dataflow. This is a figure content check, not a whole webpage or mobile usability acceptance.
- The actual referenced 16.9 Python fence was extracted byte-for-byte from the frozen chapter and executed: its printed numerator/denominator pairs are 25/1.5 and 42.5/1.75, and both final answers are 24.2857. It is a forward mathematical example, with no gradient or parameter update.
- A separate six-row float64 CPU attention calculation uses the same full causal mask and changes the candidate tile size to 1,2,3,4,6. All variants visit 21 allowed pairs; maximum absolute errors against the complete row softmax are 3.55e-15 to 7.11e-15, below the stated 1e-12 tolerance. This checks normalization and schedule invariance, not FlashAttention hardware or speed.

The mathematical distinction is supported independently by the original algorithm: exact tiling combines all allowed block contributions while retaining row statistics; a sparse visibility mask changes which contributions are allowed. The original Theorem 1 retains O(N²d) arithmetic while changing additional storage, and Theorem 2 concerns HBM accesses. A causal triangle retains quadratic order despite roughly half the pairs. These quantities must not be equated with the 15 or 12 toy-mask counts. The FlashAttention interface can select local attention, while a dense eager mask can still be materialized; the name of a core or the existence of a local mask does not itself determine either visibility or actual implementation cost.

Execution environment: Python 3.13.5; PyTorch 2.14.1+cpu, git 5c4886908584029761b579af026dcfb627c84070; CUDA build None, CUDA unavailable; one torch CPU thread. Chromium 151.0.7922.173 rendered the figure via `page.set_content` and the installed Noto font query succeeded. Initial browser attempts encountered a missing bundled browser and then administrator-blocked file navigation; using the existing system Chromium and direct SVG content succeeded. Those were renderer-environment limitations, not textbook defects.

Actual successful commands:

```text
.venv/bin/python docs/review-tools/section_facts.py course/chapters/16.md#16.13 --output /tmp/factual16_13_20261005
pdftotext -layout docs/technical-reviews/artifacts/phase4-16_13-independent/sources/flashattention-2205.14135v2.pdf docs/technical-reviews/artifacts/phase4-16_13-independent/sources/flashattention-2205.14135v2.txt
pdftotext -layout docs/technical-reviews/artifacts/phase4-16_13-independent/sources/bigbird-2007.14062v2.pdf docs/technical-reviews/artifacts/phase4-16_13-independent/sources/bigbird-2007.14062v2.txt
pdftotext -layout docs/technical-reviews/artifacts/phase4-16_13-independent/sources/swin-2103.14030v1.pdf docs/technical-reviews/artifacts/phase4-16_13-independent/sources/swin-2103.14030v1.txt
.venv/bin/python docs/technical-reviews/artifacts/phase4-16_13-independent/verify_masks.py
.venv/bin/python docs/technical-reviews/artifacts/phase4-16_13-independent/render_figure.py
```

The scripts persist their serialized result/inspection/environment and print that result. The referenced original fence stdout is separately preserved. No GPU execution, training, model/dataset download, full model scoring, corpus-wide content search, textbook/figure edit, commit or push was performed. No model-weight files or copied directory trees/symlinks are included in this evidence directory. All permanent evidence files are regular files; the hash manifest records each original snapshot and generated result.

Judgment: the substantive claims of the inspected 16.13 are verified within their explicitly stated scope. No unresolved substantive factual issue was identified.
