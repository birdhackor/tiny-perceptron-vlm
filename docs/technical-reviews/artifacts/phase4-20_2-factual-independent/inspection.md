# 20.2 independent factual inspection

Reviewer: `/root/phase4_factual_coordinator/factual_20_2`; fresh single-section task.
Access date: 2026-10-05. No child agents, textbook edits, commits, model-weight downloads,
training, GPU, or new model quality evaluation.

## Actual reading and source boundaries

Read the entire current 20.2, chapter introduction and 20.1 as input-route context,
and 20.3. A bounded source slice also exposed the opening 20.4 task/caption;
it supplied no separate outcome judgment and was not reviewed. The complete frozen chapter
is preserved at `originals/course/chapters/20.md` with its then-current SHA in
`copy-provenance.json`; `section.md` and `extraction.json` identify the 20.2 bytes.
Read factual-reviewer-instructions, section_facts, checker schema, clear-tutorial
SKILL and review-protocol. Only locator keys/types and locator-only candidate rows
were consulted from the two source-locator indexes; none supplied a review answer.

Read source functions, not training/evaluation narrative constants:
`fetch_natural_release.py` lines 1–200 and 278–493;
`natural_assistant.py` messages_for/encode_messages (118–154), load_core (263–306),
generate (315–367), load_asr/transcribe (687–770);
`natural_ui.py` upload contracts and NaturalServer/Handler/create_server/serve/PAGE
(38–453). The fixed Git commit 59a1eda4ed7b6e8609892ec2b9013c821ac93e69 contains
identical fetch/core/UI/requirements/manifest bytes. Public access to its exact
requirements was also checked by a fresh HTTPS fetch.

Official primary-source inspection: PyTorch previous-versions HTML lines 650–672
states the 2.8.0/0.23.0 CPU pair and CPU index; Git LFS 3.6.1 config manpage
388–397 explains GIT_LFS_SKIP_SMUDGE; Python 3.12 venv documentation 210–251 and
334–345 explains isolation, python -m venv and pip bootstrap. Qwen's own pinned
model card 79–130 gives the class/processor, image+text message and generate/decode
contract. OpenAI's pinned Whisper card 124–172, 208–219 explains turbo, CPU fp32,
ASR, and same-language transcription; pinned config /architectures confirms its
class. These model cards establish supported routes, not this release's measured
accuracy. Transformers 4.57.6 modeling_utils 4384–4412 and 4451–4534 documents
from_pretrained revision/cache/local-files-only, sdpa and dtype override.
Hub 0.36.2 file_download 816–860 and 884–919 documents pinned revision and cache.
URLs, versions and exact acquired bytes are saved in source-acquisition.json.

## Original measurements inspected and independently recomputed

The existing selected-public-ui actual run is original measurement evidence, not
a previous technical verdict. Only top keys/types were first listed. The actual
read pointers are enumerated in verification.json. Did not read original report
status, infrastructure_checks, closure summaries, author review notes or correction
judgments. Original report and observer files remain complete and unmodified.
The original observer source 1–127 delegates each load/generate/transcribe/operation
once to the unchanged original function. The original smoke harness 42–135 sets
the public manifest/runtime verification gate, captures source bytes, removes token
bindings, fixes CPU threads and enforces offline loading. No rerun of that harness.

Own recomputation found one base load, one ASR load, four generated chat suffixes
with 3/41/32/29 tokens, one ASR suffix of 14 tokens, and EOS on all four chat turns.
The 3rd and 4th generation's raw histories each contain the previous image.
Reset has zero history/assets/transcriptions. Original manifest/source hashes
match the fixed Git bytes and all 11 original dependency versions match pins
after removing the CPU wheel local suffix. Original official cache has
5,889,111,977 unique file bytes, supporting the qualitative several-GB disk remark.
This checks one operational trial, not general visual/ASR accuracy or reliability.

## Bounded checks actually executed

All four original Bash fences were preserved and passed bash -n. Original --list,
anonymous metadata-only download and --verify exited 0. The only downloaded public
files were the 8,722-byte README and 540-byte provenance file; both match the
committed SHA/size. Custom release README content was kept opaque, not read as an
author-provided answer. The receipt itself matches the original manifest.

Current .venv is Python 3.13.5 / torch 2.14.1+cpu, so it is not a replay of the
prescribed Python 3.12 / torch 2.8 environment. check_runtime properly rejects it.
The check with expected version metadata mocked still validates the real local
requirements and source hashes. Loader stand-ins verified pinned revision and
float32/sdpa forwarding, frozen/eval base and adapter=None. CPU float16 and bfloat16
were rejected. UI stand-ins verified ASR loads only on the first transcribe call,
transcription alone does not generate a chat, edited submitted text is used,
previous images remain in history, reset deletes material, and server_close
cleans upload files. These stand-ins test contracts only and produce no model
capability measurements. The existing actual-model run supplies the operational
evidence separately.

## Figure applicability

20.2 contains no Markdown image, img tag, SVG or other figure. Its information is
a numbered interaction sequence and explicit shell commands; no axes, arrows,
spatial image content or plotted measurements require a figure. Figure render/view
is concretely not applicable. This review makes no new desktop/mobile visual
acceptance claim about the product UI.

## Outcome

The substantive claims match the fixed release implementation and the limited
original operational evidence. No unresolved factual issue was found. Installation,
full-weight loading and model capability evaluation were not rerun; the report
keeps these boundaries explicit and accepts no general accuracy claim.
