# 12.13 independent factual analysis

Reviewer: `/root/v4_review_coordinator/factual_v4_12_13`, fresh factual task.

The original raw section SHA-256 is
`40e6e2a741e86f3a115b221a20c5cc83c8a1b9d13a6934b3440beda985dea89f`.
I read its complete UTF-8 text, including blank lines through EOF, then the
complete explicit 12.8 and 12.11 prerequisites. I inspected the linked 20.12
and 20.13 destinations for their actual separation of ASR and answers; I do
not independently re-review the final project's many score claims here.

ASR and replies have different targets. Whisper v1 §2.1 (PDF p.2) constructs
audio/transcript pairs, §2.2–2.3 (p.3) preserves positions and predicts
transcription/translation tokens, and §3.2 (p.5) compares recognized text to
a reference transcript. This objective supplies what the speaker said; it
does not supply a dinner recommendation. OpenAI's original SFT guide,
“How it works” and “Formatting your data,” describes prompts with desired
assistant responses. Its short phrase “examples of correct responses to
prompts” is the relevant distinction. These are examples of supervision,
not a claim that either supervision guarantees good generalization or that
ASR and answering must always live in two different physical models.

Order matters when the target is an ordered utterance. LibriSpeech §2.3,
Figure 2(a) explicitly treats a word transposition as a transcript mismatch.
Whisper §2.2 uses positional representations rather than an unordered mean.
For the fixed extracted vectors x[0],...,x[T-1], however, any permutation pi
has the same real-arithmetic mean:

    mean(x_pi) = (1/T) sum_j x[pi(j)] = (1/T) sum_i x[i] = mean(x).

This follows by a bijective change of index; no model-quality inference is
involved. Reversing these computed frames uses pi(j)=20-j. It is different
from reversing the waveform and re-running centered window analysis.
Floating point reductions need not be bitwise equal: the pinned official
PyTorch numerical-accuracy note says addition is not associative, and
`torch.allclose` checks abs(a-b) <= 1e-8 + 1e-5*abs(b) by default.

The transparent frame-count calculation is 0.1*16000=1600 samples per tone,
3200 after concatenation, and 1+floor(3200/160)=21 centered STFT frames.
The 400-point real FFT has 201 bins; the configured bank reduces these to
16 mel bands, then transpose gives [21,16]. Mean over time gives [16].
I ran the exact learner code and independently reconstructed centered
reflected frames, a periodic Hann window, a real FFT and triangular mel
bank. Both the production and independent sequence change under reversal;
their mean vectors agree within their stated tolerances. The production
float32 mean maximum difference is 9.5367431640625e-07; `math.fsum` on each
column yields identical means for the same stored frame values in either
order. This validates the reported 21,16,False,True in Python3.13.5 and
torch2.14.1+cpu. There is no speech recording, learned model or chat result.

For conventional supervised ASR construction and evaluation, recordings
with reference text, an order-sensitive recognizer, and held-out recordings
are an appropriate recipe. LibriSpeech §2 requires aligned utterance/text
segments; §3.1–3.2 separates speakers among training, development and test.
Whisper §3.1–3.3 and §3.7 show why held-out in-distribution recognition is
not enough to establish domain/noise robustness. Google Cloud Speech-to-Text
“Overview” and “Custom classes” identify rare proper names and domain words,
as well as noisy audio, as recognition challenges. Separate accent, noise
and vocabulary checks are therefore appropriate. This is not a universal
theorem that every ASR training algorithm must use paired transcripts:
Whisper's introduction itself mentions an unsupervised exception. The lesson
asks for reference transcripts as part of a practical training/evaluation
design; it does not claim such an impossibility result.

I directly inspected the fixed project originals `audio.json` and the FSDD
part of `real_modal.json`, including manifests, all held-out token outputs,
training target histories, source-code hashes and timing scopes. My short
CPU audit re-counts raw generated IDs before EOS, all denominators, actual
speaker splits, seeded batch target totals and rate/sample arithmetic.
The synthetic experiment only predicts low/high around 300Hz (16/8/14
records); FSDD only predicts a digit string (20/20/20 records across
jackson/nicolas/theo). FSDD's 5/20=25%,3/20=15%, EOS20/20,250 updates and
2000 targets agree with the needed prerequisite figure. Its depicted
4591/8000=9182/16000=0.573875 seconds also agrees. Upsampling is computed
from existing samples; it supplies no independently observed high-frequency
content. I did not regenerate any GPU outputs, re-train, download audio,
infer using FSDD weights, or replicate a speed/quality benchmark. Those
single-seed narrow tasks cannot establish sentence ASR or conversational
ability.

Both needed SVGs were actually rendered and personally viewed. The 12.8
figure's left-to-right train/validation/test labels, counts and resampling
arrow match the read prerequisite and independently re-counted records.
The 20.12 diagram branches from the same original question, passes the
recording route through ASR, and passes both textual inputs into the same
chat model before keeping two answers separate. There is no performance
claim in that diagram.

All external originals reside only in ignored research. Public evidence is
this analysis, precise original URLs and retrieved SHA receipts, own CPU
programs/output, and two small rendered figures. The 2.14 PyTorch web pages
returned 403; the exact installed commit's official raw source succeeded,
and its complete `_torch_docs.py` and `functional.py` bytes match the installed
files. The failed requests are retained. No factual issue remains unresolved
for this assigned section. The official checker will establish report schema
and current versions only, not factual truth.
