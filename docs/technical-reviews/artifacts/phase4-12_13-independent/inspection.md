# 12.13 independent factual inspection

Reviewer: `/root/phase4_factual_coordinator/factual_12_13`, fresh one-section identity.
Date: 2026-10-05. This is an independent factual review, not an author or reader review.

## Actual reading and isolation

I read `docs/review-tools/factual-reviewer-instructions.md`, the whole checker schema and
`section_facts.py`, `.agents/skills/clear-tutorial/SKILL.md`, and its review protocol.
The source read was all current 12.13 (initial chapter lines 426–455), plus necessary preceding
12.8 and 12.11. I preserved the complete chapter as the initially frozen input; the
formal section digest is the original UTF-8 byte section, without newline normalization.
The initially frozen complete chapter SHA is
`97250b5340e72b9b1b05c9cae2e0aed167904e5d3ad3ebe9fa14f7c210c37616` at
`inputs/course/chapters/12.md`. This is an input snapshot, not a claim that the entire
chapter remains unchanged during other reviewers' work.

The coordinator later notified me that only 12.9 had changed. I neither read nor depend
on 12.9 for this review; no scientific conclusion accompanied that notice. I retain the
initial chapter snapshot and digest as historical frozen input, and compare current
12.13 bytes independently before writing the report.

I first located originals using `rg --files`. For repository implementation I first
listed AST function spans, then read natural_concepts.py imports/module contract lines
1–11 and `audio_order_report` lines 70–85; multimodal.py imports lines 1–11 and
`tone`, `mel_filter_bank`, `log_mel`, `AudioEncoder` lines 44–81. These are computational
contracts and ordinary representation-example explanations. I also AST-located and
read build_course.py BOOTSTRAP lines 26–61 before running the helper. No old review
body, reader result, author correction, extra result summary, or another inspection was
read. No contamination event occurred. No children were spawned.

12.13 cites no historical experimental result JSON and reports no trained-model
accuracy. The two historical experiment links in prerequisite prose were not needed to
support this section and their contents were not read. Locator JSON was used only to
find immutable original paper bytes: original-source-locators `/locators/0` and
original-paper-locators `/records/64` (path/hash/identifier fields). I inspected top-level
keys/types and record field names/types first; no scope/error summary values were read.
Copied PDF bytes were compared with my own HTTPS fetch of the original arXiv URLs and
were identical; see `sources/copied-source-url-validation.json`.

## Own calculation and CPU checks

Each tone has round(0.1×16000)=1600 samples. Concatenation has L=3200 samples, 0.2 s.
STFT uses a 400-sample Hann window, hop 160 samples, default centered reflection padding.
Thus T=1+floor(L/hop)=21 frames. The mel bank has 16 rows, so log_mel is (band,time)
=(16,21). Transpose makes (time,band)=(21,16). `flip(0)` reverses time; `mean(0)`
reduces exactly 21 frames and leaves 16 band values. The values are averages of natural
log mel-band power after clipping to 1e-8, not physical energy totals in joules.

I executed the original single Python fence using the repository helper on CPU with its
offline/artifact-write guard: exit 0, attempted fence [1], no guard events, empty stderr.
Observed stdout is `時間框與頻帶 21 16`, `先後序列相同 False`,
`平均摘要近似相同 True`. Only explicit small files were copied from the helper run;
the symlinked helper workspace was neither copied nor followed recursively.

My independent `execution/bounded_check.py` reconstructed the two 0.1-second tones and
checked shape, time-axis reversal, inverse reversal, exact reversed index equivalence,
a second seeded frame permutation, and the different band-axis permutation. It made
no training, model loading, dataset download, or optimizer update. All three
`requires_grad` flags are false. Installed Python is 3.13.5, PyTorch 2.14.1+cpu,
device CPU and float32, with one CPU thread; torch.version.cuda is None.

Actual per-band mean maximum absolute difference is 9.5367431640625e-7. Bitwise
`torch.equal` of means is false, whereas default `allclose` is true. Its condition is
|a_b-b_b| ≤ 1e-8 + 1e-5|b_b| for every band; the maximum observed fraction of this
tolerance is 0.007810839917510748. The independently permuted frame means also pass.
Reversing the band axis reverses band-summary positions and changes the summary.

For exact arithmetic, [low-frame, high-frame]=[[1,0],[0,1]] and its reversal both
have mean [1/2,1/2], but different first-frame labels. More generally, for a permutation
π of T positions, mean_b=(1/T)∑_t x_(t,b)=(1/T)∑_t x_(π(t),b). A deterministic
rule fed only this common summary cannot reliably output both conflicting first-tone
labels. An order-independent presence target can sometimes use a summary (e.g. toy
high-band mean >0), but this example does not prove universal sufficiency for detecting
a tone in real speech or noisy recordings.

I also recomputed features for two short synthetic waveform changes only to check the
boundary of the example: wave.flip(0) is not exactly the reversal of the extracted
frames (max difference 0.16136151552200317), and re-synthesizing high then low is also
not exactly that frame reversal (max difference 2.754044532775879). These are this run's
transformations, not an assertion that all reversed waveforms must differ. They agree
with the explicit limit in the text and figure.

For my generated raw JSON I checked top-level keys/types before reading the named
`/environment`, `/parameters`, `/measurements`, `/samples`, `/original_helper_return`,
and `/provenance` pointers. There are no model scores or author notes in this own result.

## Original authority and support range

* Deep Sets, arXiv:1703.06114v3 (14 Apr 2018), original authors' arXiv PDF, accessed
  2026-10-05: personally read title/version/abstract and §2.1–2.2, Property 1 and
  Theorem 2 (text lines 1–98; PDF pp.1–2). Property 1 defines invariance under a
  permutation; the summed representation supplies the relevant invariant construction.
  I use only this invariance property and the sufficiency of sums, not the theorem's
  general necessity beyond its stated countable/fixed-size qualifications. The simple
  real-valued, fixed-T mean identity above is my own finite-sum derivation.
* Attention Is All You Need, arXiv:1706.03762v7 (2 Aug 2023), original authors' arXiv
  PDF: personally checked title/version and §3.5 (text lines 276–303; PDF p.6). It states
  that the nonrecurrent, nonconvolutional architecture needs relative or absolute
  position information to use sequence order. This supports retaining temporal
  information; it does not imply that preserving an array alone guarantees learning.
* Robust Speech Recognition via Large-Scale Weak Supervision, arXiv:2212.04356v1
  (6 Dec 2022), original OpenAI authors' arXiv PDF: personally checked title/version,
  §2.1 data processing and §2.2–2.3 model/task interface (text lines 129–191; PDF
  pp.2–3). Audio/transcript pairs target transcribed text, and different tasks on the
  same audio require task specification. Positional embeddings are used by its audio
  encoder. I do not import its benchmark numbers as evidence for this repository.
* SALMONN, arXiv:2310.13289v2 (8 Apr 2024), original authors' arXiv PDF: personally
  checked title/version/abstract; §4.2 instruction tuning (text lines 321–360) and §4.3,
  Table 2 (395–422; PDF pp.5–6). ASR, speech/audio QA and spoken-query QA/slot filling
  have different labels/instructions and evaluation metrics. This supports the lesson's
  task distinction; it does not establish performance of the untrained toy helper.
* PyTorch official source tag v2.9.0: personally read `_torch_docs.py` allclose
  lines 764–796, mean 6874–6940, flip 11110–11142, and functional.py stft
  lines 508–691. The mean reduction axis, flip-axis definition, allclose formula/defaults,
  STFT centering/padding/frame-count formula were checked as one connected API contract.
  This official source version differs from installed 2.14.1+cpu; installed behavior
  was separately observed in the original and bounded CPU runs. Official rendered API
  URLs returned HTTP 403, retained in retrieval.json; original source tag fetches returned
  200 and supplied the complete contract. The failed page access did not determine
  the verdict.

## Own visual inspection

I read the current source SVG and viewed its actual 640×630 Inkscape PNG. I then rendered
the actual lesson page in Chromium 151.0.7922.173 at 1280×800 and 390×844, saved full-page
screenshots plus figure screenshots, and personally viewed all four. The figure is
688×677.25 CSS pixels on desktop and 343×337.640625 on mobile. Both frequency-group
labels, direction arrows, reversed pair, near-equal average and waveform caveat are
visible without opening a zoom view. The two boxes are explicitly a frame-group
schematic, not a literal measurement of two STFT frames or of a reversed recording.
Paragraphs (5), code fence and served SVG bytes were checked against current source.

The first direct CLI Chromium screenshots stayed pending and were terminated by me;
their own process IDs and commands are recorded. A first Playwright selector timed out
because site JS expanded the image src into an absolute URL. I preserved its stderr,
inspected the DOM, corrected the selector, and completed both viewport renders. The
successful renderer blocked external font/MathJax/GitHub requests; local page/style/SVG
requests succeeded. This section has no rendered formula depending on MathJax. These
are tool attempts, not contradictory lesson results.

## Judgment and limitations

The current section and figure are factually consistent. The demonstration establishes
the many-to-one loss of ordering in a time-mean representation and the actual dimensions
and comparisons in this implementation. It does not train a classifier, ASR, or intent
model; it provides no accuracy, generalization, real-recording validation, or product
acceptance. Keeping temporal information is necessary for the illustrated conflicting
order labels, while task-relevant teaching material and actual evaluation remain separate
requirements. Transcript mismatch and requirement-answer mismatch are logically distinct:
e.g. reversing syllables can change their string, whereas an irrelevant transcription
typo need not change a demand label. No unresolved substantive claim was identified.
