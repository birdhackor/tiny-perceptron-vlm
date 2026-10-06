# MInDS-14 Chinese teaching subset

Audio and source transcripts attribution: PolyAI and the MInDS-14 authors (2021), CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ .

Gerz et al., Multilingual and Cross-Lingual Intent Detection from Spoken Data: https://arxiv.org/abs/2104.08524 . Official source: https://huggingface.co/datasets/PolyAI/minds14 revision `40ce77cb32a384e4d50a568e1ec39ac804019d33`. The original license statement is retained in `licenses/minds14/README.md`.

Original WAV bytes are retained unchanged. Changes: opaque filenames, selected intents, duplicate groups, deterministic splits, original project teaching responses and format-history examples. Two topic-continuation examples per selected recording add a text format rewrite after the audio response; heldout evaluation must use the model's actual first response as history, never the demonstrated topic response. Project-written replies are MIT-licensed teaching examples, not verified banking procedures. No relationship or endorsement by PolyAI is implied.

The source has no speaker or session identifiers. These splits cannot prove unseen-speaker capability. Source machine transcripts are audit labels only and are not listening gold or inference prompts. Automated decoding/quality checks ran; human listening did not.

Training-only derivatives retain CC BY4.0 attribution and their parent recording/group identity. Speed and deterministic synthetic-noise modifications are recorded in the V2 derivative provenance; these are not additional independent human recordings. No human listening audit is claimed.
