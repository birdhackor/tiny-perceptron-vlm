# Independent R.3 factual inspection

Reviewer: /root/v4_review_coordinator/factual_v4_r_3. Assigned raw body SHA-256: f74efc543c090759edf6bba9a8b7a95871cf39b195c20a45aa0326549f538be6. The complete section, all 23 table rows, and its final navigation/exercise paragraph were read. No introductory text is required for this non-first section; R.3 has no direct SVG.

The preexisting assigned report was copied without reading it. No previous reader/technical verdict, expected outcome, or substantive author note was used. The shared original-only catalog was read as navigation; its two cached papers were not needed. A process listing accidentally returned unrelated Chromium runtime arguments; no prior substantive judgment or author assessment was visible in the retained tool output, and no follow-up investigation of other tasks was made.

## Curriculum correspondence

Every actual chapter introduction and every numbered section title was personally read. The durable CPU receipt records full-file hashes and complete titles. Table correspondence is substantive curriculum coverage, not a new claim that every downstream empirical result was replicated.

1: encoding, count baselines, learnable preference table, loss, gradients and generation.
2: embeddings, ordered context concatenation, feature mixing, nonlinear transformation and finite window.
3: weighted sum, content weights, separate queries/keys/values, scaling, causal mask and multiple heads.
4: position, residual information, normalization, feature processing, blocks and next-token alignment.
5: updates, batches, optimizers, checkpoint continuation, held-out evaluation, data overlap and budget.
6: character/byte/BPE costs, common merges, boundaries and UTF-8 partial tokens.
7: conversation boundaries, supervision selection/shift, padding/end, SFT and pre/post-training.
8: observable style, prompts/weights/adapters, instruction/format checks and scorer biases.
9: unknown information, premise correction, refusal/over-refusal and confidence calibration.
10: pixels, patches, vector projection, spatial position and image/text interfaces.
11: alignment versus actual image use, image swapping, resolution/crops, joint relations, strokes and reading order.
12: waveform, sampling, framing, frequency/log-mel, audio features, transcription versus chat, shared chat input.
13: comparisons and reward model, reference, DPO, PPO probability/advantage/clipping/value/update and limitations.
14: relative position, RMSNorm, Q/K scale, activation and gated features.
15: dense FFN versus independent experts, router/top-k, dispatch/load/capacity and cost comparisons.
16: cost measurement, prefill/decode, KV reuse, SDPA, attention intermediate matrices/online softmax and recomputation.
17: representation/storage, scales, reconstruction error, packing, calibration and quality versus speed.
18: smaller student, teacher texts/distributions, soft targets, temperature/KL, independent truth and inherited errors.
19: one small-world MoE language core with actual image/tone input, tool request/runtime/second generation, held-out capability checks.
20: pretrained Qwen vision/text starting point, optional course LoRA candidates, use-specific acceptance, and separate Whisper ASR transcript entering the same chat core.
A: retrieval, retrieved content use, citation lookup/support and context limitations.
B: explicit request data, parse/schema/allowlist, actual arithmetic execution, results returned to the conversation and completion.
C: intermediate steps, sampling alternatives, choosing/verifying answers and additional compute cost.

Actual contents additionally read as needed: 7.2, 7.3 and all 7.4; 1.3 and 4.7 for single-shift alignment; 6.7 and glossary G.1 for token meaning; 16.2/16.3/16.8/16.9 for cache/SDPA/Flash mechanism; 17.1/17.2/17.8 for storage/precision/packing; 18.1/18.11 for teacher signals/errors; A.3/A.4 and B.1/B.2; 19.2/19.6 and actual capstone code; 20.1/20.3/20.6/20.8/20.12 (workflow-relevant passages). The current small interface implementations were inspected at their reported function locators.

## Original sources personally read

Attention Is All You Need, arXiv 1706.03762v7: section 3.1 decoder causal masking/one-position offset, section 3.2 and equation (1) for softmax-scaled QK and weighted V, sections 3.3-3.5 for FFN/embedding/positions. These support the connected attention/Transformer topics, not general semantic truth or quality of a tiny model.

FlashAttention, arXiv 2205.14135v2: section 2.2/Algorithm 0 on materialized N-by-N scores/weights; section 3.1 tiling, rescaled softmax statistics, recomputation and Algorithm 1; Theorem 1 and section 3.2 Theorem 2. Complete mathematical attention remains quadratic in arithmetic, while HBM movement and additional stored intermediates can decrease under the stated SRAM assumptions. A CPU Python online sum is not the CUDA kernel, and this R.3 overview promises no universal speed ratio.

Jacob et al., arXiv 1712.05877v1: introduction, section 2.1 and equation (1), r=S(q-Z). Lower-bit discrete representations require scales/zero points and suitable hardware arithmetic; weight-only storage benefits do not themselves demonstrate runtime acceleration. The initially attempted v3 URL returned 404, preserved in retrieval receipts; the actual v1 original was then retrieved and read.

Hinton et al., arXiv 1503.02531v1: introduction and section 2/equation (1), teacher probability soft targets and temperature; the passage combining true labels with teacher targets. Teaching from a distribution transfers the teacher's tendencies, and matching a teacher does not logically imply correctness. The three-case counterexample was independently executed; it is not trained-model evidence.

Lewis et al., arXiv 2005.11401v4: introduction, Figure 1 explanation, section 2 and 2.1. Query-driven retrieval feeds documents to a generator. The course's lexical toy does not implement this paper's trainable DPR/BART marginalization; the comparison supports only the connected retrieval-plus-use topic. Citation existence alone is not entailment.

Toolformer, arXiv 2302.04761v1: section 2 API tuple/name/input serialization; Executing API Calls, Filtering API Calls and Inference paragraphs. A call is data until an external runtime executes it and returns a result. The course's local JSON/allowlist example is not Toolformer training or an arbitrary code execution system.

Qwen3-VL-2B-Instruct original weight-repository README at commit 89644892e4d85e24eaac8bacfd4f463576704203: model identity, visual/text pretraining description and Transformers Chat quickstart, including image-plus-text messages, processor template and model generation. No marketing benchmark or native audio capability was adopted.

Whisper-large-v3-turbo original release README at commit 41f01f3fe87f28c78e2fbf8b568835947dd65ed9: Whisper introduction, task=transcribe usage, Model details and Performance/Limitations. It is an ASR/transcription component; transcription quality varies and hallucinated text is possible. No model weights or datasets were downloaded.

Original URLs, versions, byte SHA-256 and retrieval failures are retained in the small original-retrieval-receipts.json; complete third-party originals remain only in ignored research.

## Execution and scope

CPU runtime was personally verified: Python 3.13.5, torch 2.14.1+cpu, no CUDA available. review_probe.py completed with return code 0 and empty stderr. Static Notebook title/Python-block parity covered 266 chapter sections and 23 chapters. This does not execute 266 lesson kernels.

7.4: Q/A gives X=[1,3,89,2,4,73], Y=[-100,-100,-100,-100,73,2], first=4; QQ/A moves first to 5 while input/target remain 4/73; QQ/B keeps first=5, changes target to 74. A=ASCII65+8=73, B=66+8=74; render_chat performs exactly one target shift and model.masked_loss pairs same positions.

The attention probe equates explicit float64 weights/value mixing, CPU SDPA and block-size-one online attention within 1e-12. KV logits differed by 2.384185791015625e-07 at atol=1e-6 for one seeded fixed-length example. Packed seven int4 integer values used four payload bytes and reconstructed exactly, while a float quantization example had nonzero reconstruction error. Toy retrieval found d2 and missed an unmatched synonym; multiply produced 5535 and unknown/bool requests were rejected. The actual random-weight CapstoneModel accepted real synthetic image/tone tensors with four experts/top-2, and logits matched the target position axes. Actual messages_for assembled equal histories/questions for hand-supplied typed/transcribed text. These are interface/mechanism checks, not trained capability, ASR, Qwen generation, GPU throughput, or full release/experiment replication.

The browser probe's first two failures came from my selectors (relative href assumption; h2 assumption where the section uses h1). Both original scripts/stdout/stderr/return codes are preserved. They are probe defects, not course defects. Only the probe selectors were corrected.

The third browser attempt was blocked by Chromium policy for file:// navigation. The actual SVG bytes were then inserted unchanged with page.set_content and screenshotted by Chromium. The third failure's code/stdout/stderr/return code is preserved; no browser policy or source was modified. The final browser probe completed with return code 0 and empty stderr: 23 chapter entries, 266 section pages and three SVG renders. The exercise route clicked the real chapter-07 entry link to 7.4.html, whose downloadable Notebook and Colab links are present; Colab references executed-preview source revision 25ca4bf5f8f8dc0a9e88aa6cd70caeb4c2f419e8.

## Personally viewed figures

All three saved PNGs were personally opened with view_image after actual Chromium SVG rendering.

foundations_answer_alignment.svg: six rows indexed 0-5; first four rows gray with Y=-100; green row 4 has assistant ID4 -> A ID73, green row 5 A ID73 -> EOS ID2. The input Q ID89 and sequence BOS/user/Q/EOS/assistant/A/EOS, X removal of last item and Y removal of first item match the executed Q/A result. The arrows encode next-target prediction, not answer copying; gray input is still readable.

cache.svg: old row contains K0/V0, K1/V1, K2/V2; new row says only compute position3; appended row contains positions0-3. This agrees with three-position prefill and one new query/key/value, and does not imply an answer is already cached. No scale or speed ratio is drawn.

natural_shared_chat.svg: typed question goes directly to history/current question; recording first goes down through ASR/transcript, then joins the same history/question. Image pixels enter the common vision/text base separately; optional course fine-tuning is inside that core; output is text and speech synthesis is explicitly absent. ASR can omit/mishear words, so actual transcript rather than a reference replacement is routed. These arrows match current messages_for/load_asr/transcribe and NaturalServer.operation, and the restrained R.3 chapter20 row.

The Transformers 4.57.1 official tokenizer summary was also retrieved and personally read at the introductory tokenization, character/subword and Byte-level BPE paragraphs. It supports token as a model processing unit, not necessarily an entire written character. Its original HTML receipt is retained. An optional bs4 parser was absent in system Python; the already-downloaded bytes were read using standard-library HTMLParser, without installation.
