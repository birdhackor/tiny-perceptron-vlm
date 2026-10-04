# Independent factual check: 11.15

Reviewer: `/root/natural_factual_11_15`; author task supplied by coordinator: `/root`.
Date: 2026-10-04. Scope: current 11.15, its linked 11.9 and 11.12 prerequisites,
the 20.1 forward link, and the implementation/source material cited below.

## Independence and procedure

An initial broad shell read accidentally displayed the existing reader report.
I told the coordinator immediately. The coordinator allowed this factual check to
continue provided that the accident was documented and the factual judgments
were independently established. No reader-report conclusion, output file,
render, exercise answer, wording, or hash was used as evidence here. I read the
current raw section, implemented the independent calculation/check below,
executed its original Python fence in a new CPU process, fetched and read the
primary sources myself, rendered the actual SVG, and viewed my own PNG.

## Exact numerical argument

The input is a float32 tensor shaped (1,1,32,32), initially zero. Setting column
15 to one gives 32 bright scalar pixels. In this exact integer downsampling,
each output bin covers 8 rows and 8 columns. Only output column 1 contains the
stroke: each of its four bins has eight ones and 56 zeros. Its average is
8/64 = 1/8 = 0.125. The other twelve bins average to zero. Thus max changes
from 1 to 0.125; count(value >= 0.5) changes from 32 to zero.
0.125 is exactly representable in float32; the reported values and manually
computed block means match with exact tensor equality.

PyTorch's original API docstring at installed commit
5c4886908584029761b579af026dcfb627c84070 documents N,C,H,W and area mode.
Its functional implementation sends a 4-D area input to adaptive_avg_pool2d.
AdaptivePooling.h defines start/end indices; for output index a, output size 4,
input size 32, the interval is [8a,8(a+1)). The CPU kernel sums the interval
and divides by height and width. The downloaded whole functional.py has the
same SHA-256 as the installed source; this is not an assumption based on a
different PyTorch release. A versioned documentation webpage request returned
403 and is recorded in the source manifest; that unavailable page is not used
as verified evidence. The successfully fetched original API docstring and
implementation supply the authoritative API evidence instead.

## Lost information versus residual information

The small image still contains a nonzero 0.125 signal. Crossing this artificial
threshold does not establish that an OCR system cannot detect that signal.
For an actual information-loss counterexample, I independently constructed a
second whole-column image with the bright column at 14 instead of 15. Its 64
different fine scalar values collapse to the identical 4x4 output. A decoder
given only that output and the same other input cannot uniquely recover both
originals, regardless of its parameter count. A learned prior or context may
make a useful guess; the lack of a guarantee is not a claim that every guess
fails. Resizing the pooled tensor back to 32x32 also does not restore its
original fine column.

## Original execution and limits

The original section fence was extracted from raw UTF-8 bytes and executed by
section_facts.py, using the actual natural_concepts.py and CPU bootstrap, with
OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. Exit code: 0. Observed stdout:

```text
原圖與縮圖尺寸 [32, 32] [4, 4]
最亮的數字 1.0 0.125
至少半亮的數字個數 32 0
```

The independent check additionally reconstructs the image, compares against
manual 8x8 means, checks the collision, and samples the rendered PNG. No
pretrained weights, OCR recognizer, training procedure, GPU, latency test, or
quality benchmark was run. The 0.5 comparison is present only in the reporting
code; it is not an asserted recognition rule. The separate earlier OCR producer
accepts ASCII digit strings and generates 16x16 RGB images; inspection of
prepare_ocr.py and run_ocr in modalities.py confirms the numeric-only scope.
That earlier experiment is not Chinese OCR evidence.

## Figure inspection

I ran Inkscape on the current practical_stroke.svg and inspected the resulting
1200x430 PNG with view_image. All text, the left-to-right arrow, the numeral
labels, and the bottom restriction are visible without clipping. The diagram
shows a representative expanded group of eight columns, not the entire 32x32
tensor or its literal column coordinates. The left rectangle is 320 pixels
wide and its one white column is 40 pixels wide: 1/8. Overlaid antialiased grid
strokes blend its two edges, leaving 38 pure-white interior pixels. The first
independent assertion expected 40 pure-white pixels and failed; sampling the
edge pixels diagnosed the grid overlay. The subsequent check distinguishes
40-pixel stripe extent from 38-pixel pure-white interior. This is a rendering
measurement correction, not a change to the SVG or the mathematical example.

The right interior is RGB(32,32,32), the nearest 8-bit grayscale to 255/8.
32/255 = 0.125490196, within one display quantization step of 0.125. The left
dark palette is illustrative, not calibrated tensor intensity. The numerical
claim is checked from the actual tensors, rather than averaging theme colors.
The caption explicitly disclaims a Chinese recognition experiment.

## Resolution, slices, positions, and resources

I read the original LLaVA-UHD v1 paper: section 3.1 divides native-resolution
images into slices and adds a low-resolution overview; section 3.2 explains
the extra visual-token computation and its compression; section 3.3 encodes
relative slice rows and columns. This directly supports the proposed practical
strategy and the need to retain spatial organization, without claiming it is
the only possible strategy or that this course implements LLaVA-UHD.

I also read Qwen2-VL v1 sections 2.1 and 3.3.1. They describe resolution-dependent
visual-token counts, spatial position IDs, packed-length memory limits, and
the fact that increasing size does not always improve results; excessive
enlargement can hurt. Under fixed per-slice processing, more slices usually
mean more visual positions and more work/storage. Token caps, compression,
batching and implementation can alter this relation. The course states
"usually" and gives no universal monotonic theorem or numerical time/memory
claim. Keeping source details is useful; interpolation of an already degraded
source is not guaranteed to recreate missing strokes. No benchmark numbers
from these papers are assigned to this course's models.

## Upstream model and new adaptation

The fixed official Qwen3-VL-2B-Instruct card describes pretraining and OCR and
loads both model and AutoProcessor with from_pretrained. I checked the local
load_core path: the model/processor share the fixed revision, the base is
frozen, LoRA targets selected language attention projections, and run_train
rejects non-LoRA trainable names. PEFT's official v0.17.1 adapter guide explains
new low-rank update matrices on frozen original weights. "Partial weights"
can refer to these update parameters; it does not establish that original
base weights changed. This verifies the conceptual distinction and inspected
route, not successful execution of a large-model fine-tuning run. The upstream
visual/text learning was already done by the model provider; this chapter's
new examples are an adaptation, not a fresh reproduction of that pretraining.

## My exercise answer

Increasing the response model's parameter count offers more capacity to learn
language and use context, subject to suitable data and training. It cannot
uniquely determine fine pixels when different originals have become the same
input. Retaining the original detail or properly positioned crops changes the
evidence available to the model, while increasing the processing budget in
typical settings. Chinese glyph images paired with correct text supply the
supervision for mapping visible shapes to characters and for the requested
response format. None of these changes alone proves reliable Chinese OCR;
that needs a matching held-out evaluation, including unclear images.
