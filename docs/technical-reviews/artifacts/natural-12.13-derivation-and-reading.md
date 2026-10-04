# Independent 12.13 derivation and source reading

Reviewer: `/root/natural_factual_12_13`, fresh technical context; 2026-10-04.

Read the exact UTF-8 target section and its explicitly required 12.8 and 12.11
prerequisites, `natural_concepts.py`, and the `tone`, `mel_filter_bank`, and
`log_mel` implementations in `multimodal.py`. The target contains no SVG.
Rendered the prerequisite `fsdd_speaker_holdout.svg` with Inkscape and actually
viewed the PNG. Its panels depict separate speakers and single-digit tasks;
its numeric scores are not results newly measured or reviewed in this target.

## Transparent arithmetic

Each generated segment contains round(0.1 * 16000) = 1600 samples. Concatenating
440 Hz followed by 880 Hz gives 3200 samples, or 0.2 seconds. The installed
PyTorch STFT uses n_fft=400, hop_length=160, center=True, reflect padding, and
real-input one-sided spectra. There are 400//2+1=201 frequency bins. Centering
adds 200 samples at each end, so the frame count is
floor((3200 + 400 - 400)/160)+1 = 21. Each segment alone gives 11 frames, but
concatenating waveforms before extraction gives 21, not 22. Multiplication by
the 16-by-201 mel bank gives a 16-by-21 spectrogram. Transposition gives the
21-by-16 frame sequence; averaging dimension zero gives a 16-element vector.

For real-valued vectors x_0,...,x_20, define y_i=x_(20-i). Then
(1/21) sum(i=0..20) y_i = (1/21) sum(j=0..20) x_j by substituting j=20-i.
Thus the mathematical mean does not recover order. For example, the sequences
[(1,0),(0,1)] and [(0,1),(1,0)] both have mean (1/2,1/2) while their first
elements differ. This is a representation counterexample, not a measurement
of any speech recognizer or chat model.

Float32 reduction can depend on reduction order. The actual mean vectors are
not exactly equal: their largest absolute difference is
9.5367431640625e-07. The official PyTorch allclose contract is elementwise
abs(a_i-b_i) <= 1e-8 + 1e-5 * abs(b_i); every band satisfies this bound in the
saved CPU run. No universal promise that arbitrary floating point permutations
pass allclose is inferred. Reversing the waveform and recomputing log-mel is
a separate operation: in this run its features fail allclose with the reversed
extracted sequence, with maximum absolute difference 0.16136151552200317.

## Primary source reading and boundaries

Read the actual original PDF of Radford et al., arXiv:2212.04356v1,
6 December 2022, https://arxiv.org/pdf/2212.04356v1. Section 2.1, printed page 2,
describes audio/transcript pairs and aligned 30-second segments. Section 2.2,
printed page 3, uses encoder and decoder position embeddings; section 2.3
distinguishes transcription from translation and specifies transcript output.
Sections 3.7 and 3.8, printed pages 8–9, evaluate additive noise and long-form
recordings, including recordings with jargon. These are Whisper's methods and
evaluations, not evidence that this course's 16-band CPU example recognizes
human sentences. The paper's page-1 footnote notes an unsupervised recognition
exception: the target's paired-data requirement is read within the course's
supervised training/evaluation workflow, not as a theorem that every possible
ASR training approach requires paired transcripts.

Read OpenAI's official model-card.md at commit
fc5ded7d9045c693692f13853857c3f8baea3a7b, particularly Model Details (line 10),
Evaluated Use (line 44), Training Data (line 51), and Performance and Limitations
(lines 58–64). It defines ASR as transcription and explicitly reports unequal
performance across accents/dialects and speakers and recommends evaluation in
the intended domain. These support the need to test speakers, accents,
background sound, and specialized vocabulary separately; they supply no
Mandarin score for this section.

Read Ouyang et al., arXiv:2203.02155v1, 4 March 2022,
https://arxiv.org/pdf/2203.02155v1, section 3.1, printed page 4, Step 1.
Human demonstrations specify desired outputs for input prompts, unlike the
recording/transcript pairs in Whisper. This supports the exercise's distinction
between teaching transcription and teaching an appropriate response, without
claiming response correctness or requiring the course to use InstructGPT.

Both official PyTorch documentation hostnames returned HTTP 403. Instead,
downloaded and read upstream source at the exact installed wheel's Git revision
5c4886908584029761b579af026dcfb627c84070. torch/_torch_docs.py lines 835–853
specify allclose and defaults; lines 7196–7240 describe mean reductions.
torch/functional.py lines 507–518 specify STFT defaults; lines 572–585 describe
centering, reflect padding, and one-sided output; lines 642–647 give shape and
frame count; lines 674–679 implement centering. Raw downloaded bytes and failed
HTTP attempts are saved. TLS and checksums were not bypassed.

Read the existing course modality contracts: run_audio's explicit synthetic
tone scope and run_real_modal's FSDD digit questions/answers, and the matching
audio/real_modal result records. They identify two narrow tasks; this review
does not rerun their training or adopt their numerical quality claims. Read
20.7 only to check the cross-reference's planned ASR/chat separation. The
forward reference is not evidence of completed Mandarin evaluation.
