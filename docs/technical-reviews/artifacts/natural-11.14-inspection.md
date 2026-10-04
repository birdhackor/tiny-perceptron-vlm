# Independent factual inspection of 11.14

Reviewer: `/root/natural_factual_11_14`; fresh context, separate from author `/root` and reader stage. Read the actual section first, then the whole required `tiny_perceptron/natural_concepts.py`, explicit prerequisites 11.4 and 10.2, linked 20.5, review instructions, helper, and checker. No previous reader report or technical conclusion was used. No lesson or SVG was edited.

## Executed code

Original command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/review-tools/section_facts.py 'course/chapters/11.md#11.14' --execute --timeout 120 --output outputs/reviewer-tools/runs/natural-11.14-first`. The helper executed the raw Python fence with the current CPU bootstrap, exit 0, no audit-guard events. Raw stdout, stderr, environment, extraction, bootstrap, fence, and section bytes were copied without rewriting into `natural-11.14-original/`.

Independent arithmetic command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/natural-11.14-verify.py > docs/technical-reviews/artifacts/natural-11.14-verify-output.json`. Exit 0, all assertions passed. Python 3.13.5, PyTorch 2.14.1+cpu at commit `5c4886908584029761b579af026dcfb627c84070`, CPU float32, OMP/MKL 1 thread. This run used the repository helper, inspected its entire 3x32x32 outputs, explicit 8x8 block means, 4x4 and 8x8 pooling, flattening, and a concrete version of the paper exercise. It loaded no model.

## Primary source inspection

The current online HTML documentation URLs for PyTorch adaptive pooling and torchvision Faster R-CNN returned HTTP 403. Those failures are retained in the download manifest. I then read their authoritative documentation directly in the official source repository, rather than treating an unavailable HTML page as verified. The exact installed PyTorch commit was available. The downloaded `torch/nn/functional.py` is byte-for-byte identical to the installed file (SHA-256 `95ff403085bb179477a01df63acfddf59dfac0fb58829e99a9c8b5c2fe98899b`).

Read exact-commit `functional.py` lines 1440-1459 (`adaptive_avg_pool2d`), `pooling.py` lines 1477-1520 (`AdaptiveAvgPool2d` documentation of N,C,H,W and channel preservation), `AdaptivePooling.h` lines 31-37 (bin boundaries), and CPU `AdaptiveAvgPoolKernel.cpp` lines 18-67 (per-channel region sum divided by its height and width). The exact 32-to-4 divisibility is essential to the simple disjoint 8x8 explanation. Arbitrary input/output sizes need not have that layout. Initial v2.11.0 source snapshots were also downloaded, but the review uses the matching installed commit.

Read Hudson and Manning, GQA, CVPR 2019 original publisher PDF, pages 6700-6703: introduction distinguishes object recognition, relations, grounded understanding and dataset shortcuts; section 2 distinguishes realistic photos from constrained synthetic images and mentions speculative events; sections 3 and 3.1 describe objects, attributes, actions, spatial relation edges and bounding-box position/size. These support task distinctions and evaluation limits, not any claim that this course's untrained tensor example demonstrates natural-photo capability.

Read Dosovitskiy et al., ViT, arXiv 2010.11929v2 PDF (ICLR 2021), Figure 1, section 3.1, equation (1), and the inductive-bias paragraph on pages 3-4: flattened image patches, added position embeddings, and spatial relations that must be learned. Read Kim, Son and Kim, ViLT, official ICML/PMLR 139 (2021) PDF, section 3.1 equations (1)-(6) and section 4.3: image patches and position embeddings join the text sequence; the resulting model is fine-tuned for image/question pairs on VQAv2. This is an existence example of patch-based VQA, not a promise that retaining raw pixels automatically learns relations.

Read official torchvision v0.25.0 `faster_rcnn.py`, `FasterRCNN` class documentation lines 44-75: detector outputs include boxes [x1,y1,x2,y2], labels and scores. Positions can supply spatial clues; bounding boxes alone do not establish holding, intention or scene history.

20.5 is a forward reference to natural-photo evaluation design. At the time of this review its actual-results block was still a placeholder. Section 11.14 uses future language and makes no numeric claim about completed natural-photo performance; this review does not certify that future experiment or promote the toy output into natural-photo evidence.

## Actual figure inspection

The ImageMagick command failed because its configured `rsvg-convert` executable was absent. Rendering then succeeded using the existing system librsvg through `rsvg_handle_render_document`, with Cairo 1.18.4: `.venv/bin/python docs/technical-reviews/artifacts/natural-11.14-render.py > docs/technical-reviews/artifacts/natural-11.14-render-output.json`. I viewed the resulting 1200x480 PNG.

Both displayed strips have equal area, 50x90 = 4500 SVG units squared. Red and blue are adjacent and reversed between A and B. The dashed boundary encloses both strips in each example. The arrow points from those examples to the same-summary text. The output says equal red and blue amounts; it does not display a numeric RGB result. Its pale purple panel is a diagram panel, not a color-calibrated rendering of the actual pooled [0.5,0,0.5]. The dashed 110x140 SVG rectangles and their white margins are schematic, not literal square pixel bins. The visible footer explicitly says only a local schematic is drawn and the actual program swaps within the same 8x8 averaging cell of a 32x32 image. No literal area fraction, pixel mapping, numerical color calibration, or full-image scale is inferred from this figure. All Chinese text is visible, with no clipping. The title, left/right captions, pooling direction and summary agree with the code and prose.

## Scope

There are no empirical model-quality claims in this section. All exact numeric results below apply to these two synthetic arrays and this CPU run. They do not demonstrate natural-photo reasoning, recover missing visual evidence, or show that a ViT architecture by itself understands objects/actions/relations. Appropriate task examples and separate natural-photo validation are needed to assess those behaviors. The opening also limits unseen before/after stories, which is consistent with visual evidence being insufficient to select a unique history.
