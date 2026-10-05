# R.3 original-owner corrective factual check, 2026-10-05

Reviewer: /root/v4_review_coordinator/factual_v4_r_3. This reuses my own completed original review and genuine original-authority/CPU/browser records. It is not a new whole-chapter reading or a full-book/release audit. The canonical pre-correction report was preserved before changes and byte-equals the supplied coordinator archive; first original judgment and first native checker output remain untouched.

## Actual changed-source inspection

I personally read complete current 20.6, 20.7 and 20.8, and compared current chapter20 bytes with the exact old chapter20 source matching my registered SHA. The whole-file difference is one 20.7 sentence: ambiguous “this right-shift relation” became explicit “input and next-target alignment.” All chapter20 headings and Python blocks are identical; all other chapter20 sections are byte-identical. This allows reuse of my previous actual chapter introduction/title reading and complete 20.1/20.3/20.12 plus the earlier genuine authority/code inspections. I do not claim newly reading the other ten sections.

R.3 itself exactly byte-equals my saved complete original source; SHA f74efc543c090759edf6bba9a8b7a95871cf39b195c20a45aa0326549f538be6. No separate introduction applies to this non-first section. Every registered Python source byte-equals revision25ca4bf5f8f8dc0a9e88aa6cd70caeb4c2f419e8 whose full-file SHA also matches my original source records. Original chapter07 source SHA still matches my record. byte-comparison.json retains each actual result, so preserved executions are reused as dated evidence rather than labeled new runs.

I reread current render_chat and encode_training_row implementations at their actual function bodies. render_chat returns ids[:-1] and original targets[1:] exactly once for the small byte example. The separate Qwen training encoder builds processor/template inputs, verifies prompt-prefix equality, ignores prefix/PAD labels, rejects excess encoded length, and returns full model inputs/labels. The tiny six-position figure must not be used to assign lengths or shifted labels to the actual Qwen multimodal input. This correction preserves that distinction.

## Actual current SVG render and personal viewing

I read the full current answer-mask SVG, compared its bytes with the archived SVG, rendered its actual img element in http://127.0.0.1:8790/20.7.html, and personally opened the saved PNG with view_image. The served SVG bytes exactly equal the local current SVG. SVG SHA ffc11a178aef0ba81ed1416c10ddeb90f8b233927cc5bdee4a7d5b620a04707b.

Only descriptive wording changed: the accessible description explicitly says per-input-position next targets; the visible heading is now “下一項目標 Y” without “已右移.” Six input boxes remain start/user/Q/EOS/assistant/A. Four gray target boxes mean no direct loss; blue targets A and EOS align vertically with assistant and A inputs. Two downward arrows and the first-answer caption agree with the next-token events; the question remains an input. The bottom statement explicitly excludes treating these six boxes as the full multimodal-model sequence length. I found no contradictory number, position, direction or support claim.

## Bounded new execution and claim reassessment

The corrective probe actually ran on Python3.13.5/torch2.14.1+cpu without CUDA; return code0, empty stderr. Q/A returns X=[1,3,89,2,4,73], Y=[-100,-100,-100,-100,73,2], length6, two active targets, first4, input4. Q/AB returns length7, targets[73,74,2], three active targets and first4/input4. This directly verifies current 20.7 and its single-edit exercise while retaining the earlier genuine 7.4/QQ cases; no model training was run. The current 20.7 Notebook Python block matches its current canonical section.

Current actual preview route: R.3 chapter20 link -> chapter20 entry, whose20.6/20.7/20.8 links exist -> clicked20.7 page. The updated alignment sentence, small-example boundary and Notebook links are present; the Colab link names source9e963cb0863c9f0139e02f00a4c9209741cec7f9. This is a focused current-route check, not another 23-chapter/266-page sweep.

c1 (curriculum coverage): unchanged chapter20 titles/other contents and the inspected clarification retain the advertised fine-tuning/acceptance topic. Old 23-chapter/266-Notebook receipts remain the actual original inspections; the correction records current chapter20 hash and current20.7 Notebook parity explicitly.

c10 (pretrained vision/text base, optional candidates, per-use acceptance and shared chat after ASR): full current20.6-20.8 still distinguishes adapter updates from candidate acceptance, keeps ASR fixed across vision/text candidates, and does not promise every candidate improves. The revised20.7 figure represents toy supervision alignment only; it neither changes the ASR->shared-chat route nor pretends to describe actual Qwen positions. My already-read original Qwen/Whisper authorities and byte-unchanged code support the same limited architecture/workflow claim.

c11 (chapter->section->Notebook route): the original complete route/parity receipt is retained as historical evidence; the changed chapter/section route was actually rechecked at8790, and20.7 still provides its matching Notebook. No new Colab-kernel execution is claimed.

c12 (7.4 target/first-answer alignment): original7.4 source and code remain unchanged, and new20.7 Q/A/AB outputs reinforce the same next-token boundary relation without making six positions universal. Current figure and wording agree with this claim.

My independent corrective judgment is pass, with no unresolved R.3 issue. This conclusion is limited to R.3’s advertised curriculum/routes/mechanisms; the unchanged2077/39436/78872 training totals and20.8 release scores were not freshly audited or adopted as new empirical claims in R.3. No peer report, author expected verdict, global journal or reading-time record was consulted or edited.
