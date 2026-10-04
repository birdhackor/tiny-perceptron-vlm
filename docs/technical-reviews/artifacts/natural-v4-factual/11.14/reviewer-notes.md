# Independent 11.14 factual review

Reviewer: /root/v4_review_coordinator/factual_v4_11_14. Current complete raw section SHA-256: 8a8d6b6441b5e66b01424cbd3b8eeee7d6191ebba47b41afb3a187e7948ce12e. Read from heading 11.14 through every trailing blank line before 11.15. Not the first section; introduction is not applicable. Necessary raw prerequisites 11.4, 10.2, 20.9 and 20.13 were read and preserved. Any pre-existing assigned report was copied unread; no previous section-review conclusion informed this verdict.

## Original authorities actually read

Retrieval URL, resolved URL, raw byte SHA, extraction exit and genuine warnings/failures are in retrieval-receipts.json. Complete originals remain only in ignored outputs/natural-v4/factual-research/11.14/. No complete external publication is copied into this evidence directory.

- Dosovitskiy et al., *An Image Is Worth 16x16 Words*, arXiv:2010.11929v2, 3 June 2021, section 3.1, Figure 1 and equations (1)-(4), PDF pp. 3-4. Read flattened patch dimensions, projection, learned positional embeddings and the paragraph explaining that spatial relations must be learned. This is representation and architecture evidence, not a proof that merely retaining raw pixels gives semantic understanding.
- Antol et al., *VQA: Visual Question Answering*, ICCV 2015 publisher PDF, abstract and sections 1-2, proceedings pp. 2425-2426. Read image-plus-question-to-answer definition and examples distinguishing activity, object count and attributes. The original task includes common-sense questions; its definition alone does not establish that all answers are directly visible. The lesson's narrower visible-fact rule is consistent with the LLaVA prompt below and its own evaluation rubric.
- Carion et al., *End-to-End Object Detection with Transformers*, arXiv:2005.12872v3, 28 May 2020, abstract, section 1 and Figure 1, PDF pp. 1-2. Detection predicts a set of category labels and bounding boxes. Boxes localize objects and provide spatial clues; bounding boxes alone do not certify a holding action or eliminate ambiguity/occlusion.
- Liu et al., *Visual Instruction Tuning*, arXiv:2304.08485v2, 11 December 2023, section 3 (PDF p. 3), section 4.1 and equation (1) (p. 4), section 4.2 (pp. 4-5), Appendix F/Table 13 (p. 22). Read captions and box coordinates as symbolic inputs to the data generator; questions about actions, locations and relative positions; the restriction to confidently answerable visible content; and inference architecture using projected CLIP ViT grid features as a sequence. The boxes used by the text-only data generator are not a prerequisite object-detector stage of LLaVA's image inference architecture.
- Krishna et al., *Visual Genome*, arXiv:1602.07332v1, 23 February 2016, abstract/section 1 (p. 1), sections 3.2-3.5 (pp. 11-12), and section 6.2 (p. 37). Read objects, attributes, relationships as distinct annotations; scenes with the same man/bike objects but different riding/falling relations; and relationship-prediction supervision. Region/relationship annotations illustrate other training objectives. These papers motivate richer relation information; they do not prove an exclusive universal requirement for the lesson's two named example formats.
- PyTorch official source, tag v2.14.1, torch/nn/modules/pooling.py, AdaptiveAvgPool2d lines 1477-1514. Read input/output axes, target spatial size, channel preservation and dispatch to F.adaptive_avg_pool2d. Original version-specific docs endpoint returned HTTP403; version-matching official GitHub source succeeded. Also inspected the actual installed Torch 2.14.1+cpu functional implementation (full-file SHA 95ff403085bb179477a01df63acfddf59dfac0fb58829e99a9c8b5c2fe98899b), which dispatches to the native average-pool operation. The executed probe verifies actual behavior in this runtime.

## Independent arithmetic and actual CPU checks

Code inspected: tiny_perceptron/natural_concepts.py, including swapped_picture and picture_order_report; tiny_perceptron/multimodal.py patchify/unpatchify and VisionEncoder. Full file SHA values are recorded in the review JSON.

Each strip covers 8 rows by 4 columns = 32 RGB positions. Their union is 64 positions. Swapping red and blue changes red and blue channels at each position and leaves green unchanged: 64 x 2 = 128 scalar values, not 128 RGB positions. Red and blue channel sums are each 32 in each image. At output row 1, column 0, an 8x8 = 64-position cell has mean (32/64, 0/64, 32/64) = (0.5, 0, 0.5). All other output cells are zero. The sum is permutation-invariant, so the two pooled representations coincide while their full sequences differ. For an observer given only that identical representation, it is impossible to uniquely determine which of these two original images was supplied. Side information and learned prior guesses do not restore information present in the representation itself.

probe.py runs the actual unchanged helper and independent Python cell summations, verifies Torch equality against those sums, counts changed spatial positions/channel values, and directly checks known row/channel patch ordering from 10.2. It loads no model and runs no training.

Exercise: move the red 8x4 strip from x=0:4 to x=8:12, retaining y=8:16 and the blue strip. First cell becomes (0, 0, 0.5); second cell becomes (0.5, 0, 0). Per-cell summaries therefore change, although global channel totals stay (32, 0, 32). A suitable photo question is “杯子由誰拿著？”; a photo cannot by itself support “他稍後一定會喝咖啡” or why the person picked up the cup. If the visible hand/cup relation is occluded, the correct response should retain uncertainty.

## Relevant fixed-run records

This is record verification and independent arithmetic, not my own GPU inference/training replication or a new semantic regrading of 126 answers. The bound scores, generations, manifest and descriptive-subset files were inspected for only the relevant photo rows. The probe checks their original hashes, actual prompt/image/split matching, decoder completion, passed = rubric_passed AND decoder_complete, direct input qa_type subset membership, unique photo identities, and current image/family disjointness from train/validation.

Recomputed: photo summaries 25/42; factual QA 58/84; action 3/11; relation 21/29. All relevant photo outputs ended normally. There are 42 shared images with one summary and two QA per image, not 126 independent image situations; action and relation rows are subsets of the 84 QA. Fixed run gha-37221188153-1, execution revision 1e71b8abffb34eccd297747d39827135a627107a, base Qwen/Qwen3-VL-2B-Instruct revision 89644892e4d85e24eaac8bacfd4f463576704203, adapters empty. No foundation-pretraining exclusion is proved. No GPU, speed, parameter-count, release, ASR, OCR or chat claim is inferred from this bounded check.

## Actual figure viewing

render.py rendered current SVG bytes inline in actual Chromium 151.0.7922.173 with Playwright 1.63.0, reduced motion, device scale 1. I personally viewed all three resulting PNGs through view_image.

- practical_order.png: top red is left of blue; bottom blue is left of red. Equal-sized strips lie inside the dashed cell. The downward arrow leads to identical-average summary, not to successful recognition. The text explicitly says enlarged local schematic, 32x32 input, same 8x8 cell. It makes no misleading scale/count assertion.
- patchify.png: 0-3 first row, 4-7 second, 8-11 third, 12-15 fourth; arrow goes to row-major sequence and 3x4x4=48 per patch. Direct ramp checks supplement the round-trip result.
- natural_photo_evidence.png: the same person/bike categories appear in both drawings; top person sits over the bike and reaches the handlebar with bent legs, bottom stands alongside. It is explicitly an artificial concept diagram, not a measured natural-photo result.

The first file:// rendering attempt was blocked by administrator browser policy. Inline rendering succeeded without changing that policy; the real failure is preserved in failed-attempts.txt.

## Remaining substantive issue

The sentence “需要含整圖描述或關係問答的示範，才能教到……” is an unqualified necessity claim about specific supervision formats. The originals establish useful approaches, relation content and limitations of object/color labels, but not that these two formats are necessary across models, training objectives or prior learning. In particular, Visual Genome also studies direct region relationship prediction, while ViT learns spatial representations from image classification and pretrained encoders/language can transfer existing knowledge. Neither is by itself a new Chinese semantic relation-training experiment; they show why necessity cannot be justified by the read examples.

Suggested correction: “若要在本課的監督訓練中直接教到『拿著』『左邊』『正在做什麼』，可以加入整圖描述或關係問答示範；只增加顏色標籤不會保證學會這些關係。” This preserves the supported teaching recommendation without asserting universal necessity. The source is not edited by this reviewer. Verdict: revise for this wording; all numeric, software and figure checks pass.
