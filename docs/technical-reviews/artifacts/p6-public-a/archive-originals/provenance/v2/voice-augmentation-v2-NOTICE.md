# V2 training-only MInDS-14 audio modifications

Source attribution: PolyAI; Gerz et al., Multilingual and Cross-Lingual Intent Detection from Spoken Data (2021). https://huggingface.co/datasets/PolyAI/minds14 revision `40ce77cb32a384e4d50a568e1ec39ac804019d33`. CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . Original card and notices remain in the original data/manifests.

Changes: training audio only, fixed-point speed resampling, gentle deterministic synthetic background noise, leading/trailing padding without cutting speech, optional peak attenuation, PCM16 encoding and appended training rows. Original human recordings, source labels, group IDs, response/history templates and evaluation structure remain the same. No new independent utterances or speakers are claimed; no endorsement by the source authors is implied. The project-authored teaching replies retain the project's MIT license.

Validation/test JSONL and original media bytes remain unchanged. Other reviewed text/OCR training appenders may add rows while retaining their original exact byte prefix. No human listening was performed.
