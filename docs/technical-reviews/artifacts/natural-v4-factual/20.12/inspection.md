# Independent inspection of 20.12

Reviewer: `/root/v4_review_coordinator/factual_v4_20_12`, fresh assigned context.
Read all current raw 20.12 bytes, including blank lines, SHA-256
`00264a7ccf7032048941b85e388bf978b0c2dbc9261175310e284b3e311443ef`.
Introduction is not applicable because this is not the first section. Read full
necessary sections 12.13, 20.1 and 20.13; exact originals are retained here.
The previous assigned report was copied without opening its contents.

## Original authorities actually read

Retrieval URLs, byte counts, versions/commit headers and SHA-256 are in
`original-retrievals.json`; complete downloaded sources remain only in ignored
`outputs/natural-v4/factual-research/20.12/`.

- Hugging Face Evaluate CER implementation, retrieved commit
  `76a87418a42c9c8e6e6832daed7979456774aba8`, `cer.py` lines 87–108
  (`_DESCRIPTION`) and 152–193 (`_compute`): CER = (S+D+I)/N; reference
  characters, rather than hypothesis length or recording count, form N.
  The implementation sums errors and reference characters over examples for
  micro CER. The section's explicit punctuation convention is its own raw
  codepoint convention; it is not described as the metric package's default.
- Ribeiro et al., *Beyond Accuracy: Behavioral Testing of NLP Models with
  CheckList*, ACL 2020 final PDF: pp. 4903–4904, §§2.1–2.2 (capabilities,
  minimum-functionality, invariance, directional expectations), and p. 4906
  Table 2, Negation row (a negated question need not retain the same intent).
  These support separately specified behavioral expectations, not a claim that
  CER measures semantic adequacy or a theorem that every negation reverses all
  meanings. In this specific Chinese pair, 不辣 excludes spicy food whereas
  辣 requests spicy food; one codepoint edit therefore changes the constraint.
- Bu et al., AISHELL-1, arXiv `1709.05522v1`: pp. 1–2, §§2–3, recording
  devices, selected raw text, transcription and quality checking. OpenSLR 33
  “About this resource”, plus pinned official AISHELL card
  `bbe295d530192a4cd41644b711c9aecd087df653`, specify human participants,
  quiet indoor recording, high fidelity microphone and 16 kHz audio.
  The source is read corpus utterances, not spontaneous assistant dialogue.
  Read the eight selected utterance-ID lines from the official pinned
  transcript's first 1 MiB, discarding the final partial line; independently
  verified space removal, all eight WAV hashes and SoundFile metadata.
- OpenAI Whisper turbo card, commit
  `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, “Whisper”, “Model details”,
  “Evaluated Use”, “Performance and Limitations”: ASR predicts transcriptions;
  released pretraining and optional subsequent fine-tuning are distinct.
  Domain, accent and hallucination limitations require specific evaluation.
- Radford et al., Whisper paper `2212.04356v1`, §6 pp. 13–14 and Appendix C
  pp. 20–21: long-form failures, missing words, repetition/hallucination, and
  normalization choices. Its non-English evaluation normalizer removes
  punctuation and lowercases. The section explicitly uses its own raw
  punctuation-preserving convention, so it does not inherit those choices.

## Independent arithmetic, implementation and original-record audit

`probe.py` ran in `.venv`, Python 3.13.5 / Torch 2.14.1+cpu, with no GPU.
The section outputs are 1, 9, 11.1, False. Length difference gives an edit-distance
lower bound of one and inserting 不 at index 3 attains it, proving the minimum.
The nine explicit characters include the final period. Restoring 不 gives
0/9 = 0; deleting only the period gives 1/9. An independent full edit matrix
also reproduces the distances. No waveform, ASR, or chatbot is run by this
section example.

Read `natural_concepts.py` in full, and the relevant runtime
`natural_assistant.py` APIs (`messages_for`, `asr_cer_metrics`, `generate`,
`load_asr`, `transcribe`, paired evaluation). Current runtime bytes match the
pretest recorded SHA-256. Actual evaluation takes the same row/model/options
twice, replacing only the spoken route's current user with the real ASR
prediction. System/history/image remain the same. Original final-run generation,
transcript and result copies match the original artifact-index hashes. All four
AISHELL test references total 42 characters; all four real transcripts match
them, hence 0/42. Both routes have identical raw generated token sequences and
complete EOS on these particular four pairs. This is a fixed-record audit,
not a fresh GPU/ASR inference or training replication.

The four-question test uses no photos and empty history, with the same declared
response system policy. It establishes the paired protocol for that condition;
it cannot establish arbitrary photographic or long-history speech accuracy.
Raw generation logs independently record current user/image/family, but do not
log rendered chat-template tokens: system/history are reconstructed from the
hash-bound manifest and runtime, rather than independently observed at the
token boundary.

The frozen dataset plan binds the current manifest and question source hashes
at 14:18:43 UTC; the pretest guard binds the same manifest/runtime/protocol at
17:36:35 UTC; generation provenance is 17:38:54 UTC. The per-question rubrics in
the source, preparation script and frozen manifest agree. No rubric was inferred
from the output. For `BAC009S0042W0480`, the correct transcript asks what one
would do about “such a thing”, while the frozen rubric requires asking what
actually happened. The complete answer gives a generic reaction without that
clarification. This directly demonstrates a second-stage failure despite
perfect transcription, without treating equal answers or EOS as correctness.

Read the actual browser client handlers and server operation. ASR fills the
editable `prompt` and separately displays its original text. Sending posts the
edited prompt together with the speech identifier. A capture-only runner probe
executed the real `/api/chat` operation: the corrected text reached the runner,
the original transcript remained unchanged, and the response reported
`submitted_text` plus `corrected: true`. This verifies protocol behavior using a
fixed transcript and injected generation boundary, not model recognition or
browser interaction with loaded weights. Evaluation code never uses this
manual-correction path to substitute gold text in the raw spoken route.

The necessary 12.13 synthetic-tone prerequisite executed as 21 frames, 16 bands,
False sequence equality and True mean equality within tolerance. It does not
demonstrate human ASR. No prerequisite figure is referenced in 12.13 or 20.13.

## Figure viewing and actual limitations

Rendered the current ASR two-routes SVG and 20.1 shared-chat SVG using Chromium
and personally viewed both PNGs. The left route goes directly from gold text to
the same chat model; the right route goes through true recording, ASR and raw
transcript before the same chat model. Both produce separate answers and the
same-weights/history/photo labels agree with the text. The shared-chat figure
joins typed and actual-ASR text in common history, adds image pixels, and ends
in text output. Neither figure displays CER as chat quality or shows an oracle
replacement in the raw speech route. Labels and arrowheads are visible and
unclipped; there are no numerical plot scales to verify.

Initial file-URL rendering failed with Chromium `ERR_BLOCKED_BY_ADMINISTRATOR`;
rendering the same bytes through `page.set_content` succeeded without changing
browser policy. The first bounded transcript request exceeded the 8 MiB read
cap; a 206 range request returned precisely the first 1 MiB, including all eight
needed IDs. Both failures are retained. `pdftotext` emitted two nonfatal
CheckList ligature warnings; the relevant prose/test descriptions were readable.

Procedural-scope disclosure: one navigation read opened
`docs/natural-assistant/v4/runtime-scoring.md`, whose text calls itself author
execution instructions. The visible material was scorer CLI/format, alias
blindness, denominators, binding and EOS protocol, not a 20.12 author conclusion
or a prior reader/technical verdict. It was excluded as factual authority.
No old reviewer verdict or author check was opened. No source, checker, Git,
environment or reading-time edit was made.

The review supports the declared nine-character demonstration, paired
diagnostic method, recorded human-read corpus protocol and correction behavior.
It does not support unrestricted spontaneous conversation, population error
rates, new ASR training, untested accents/noise/overlap, or a fresh model-quality
benchmark. No substantive contradiction or unresolved source blocker remains.
