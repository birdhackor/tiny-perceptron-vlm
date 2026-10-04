# Independent whole STUDENT factual inspection

Reviewer: `/root/v4_review_coordinator/factual_guide_student`, fresh context.
Assigned document: `docs/natural-assistant/v4/STUDENT.md`, every line 1–150,
including opening and EOF. Raw UTF-8 snapshot is `STUDENT.initial-fullfile.md`;
SHA-256 `6732af69f9505ef1842c42d10eac86e60b1a54fb0c62a6c26a67798b8d1ddb7c`.
No previous review, author notes, or coordinator factual verdict was consulted.
No preexisting assigned report existed at first inspection.

## Reading and original authorities

Read order: supplemental factual contract and technical-review guide; complete
current STUDENT; requirements/public manifest and fetch script; current UI/core
functions needed by the operating commands; linked current 20.1, 20.3 and 20.13;
original structured operation/cache records and their actual measurement harness;
official original sources; actual rendered STUDENT in Chromium and its 20.1 link.
The source reading was substantive reading, not inferred from hashes or HTTP.

Original retrieval receipts are `authority-retrieval.json`,
`authority-retrieval-threads.json`, `authority-retrieval-extra.json`, and
`authority-retrieval-command.json`. They give
actual HTTPS URLs, retrieval dates, versions/pins via URL, byte counts and SHA.
Full original documents/source stay under ignored
`outputs/natural-v4/factual-research/student/`. The summaries below are mine.

- Git clone manual, current page last changed at 2.54.0, OPTIONS `--no-checkout`
  (readable lines 576–580): cloning does not check out HEAD. Git LFS v3.7.1
  `git-lfs-config.adoc` lines 396–403: `GIT_LFS_SKIP_SMUDGE=1` suppresses pointer
  conversion during checkout. Project `.gitattributes` identifies training
  archives as LFS. The pinned checkout's script, core, UI, requirements and
  v4 manifest are byte-identical to the current relevant files.
  The original git-checkout manual's `--detach [<branch>]`/commit form permits
  checking out the fixed commit without creating a branch. PyPA's version
  matching specification states local labels are ignored when the public
  equality specifier has no local label; the small packaging probe confirmed
  the requirement `==2.8.0` accepts both `2.8.0+cpu` and `2.8.0+cu128`.
- Python 3.12.15 `venv`, introduction, Creating virtual environments, and How
  venvs work (readable lines 55–89, 105–161, 246–260, 284–355): virtual
  environments isolate package sets by default; POSIX uses `bin`; explicitly
  naming its interpreter works without activation. This does not install an
  absent Python 3.12 or its distro venv support.
- PyTorch previous versions, `v2.8.0` → Linux and Windows (readable lines
  180–194): the documented pair is torch 2.8.0 / torchvision 0.23.0, with CPU
  and CUDA 12.8 indexes matching the guide. NVIDIA's original nvidia-smi
  SUMMARY OPTIONS and GPU ATTRIBUTES (readable lines 41–45, 521–546) describe
  connected GPUs and driver versions. A driver's CUDA capability is distinct
  from an installed toolkit. No minimum VRAM or NVIDIA execution is inferred.
- PyTorch v2.8.0 `torch/cuda/__init__.py` lines 163–216: `is_available()` tests
  CUDA availability; `is_bf16_supported()` includes emulation by default and
  is not a performance promise. Official 2.8 `set_num_threads` and
  `set_num_interop_threads` API pages describe intra-op and inter-op CPU
  parallelism and the before-work/once-only restrictions. The guide describes
  configuration; it does not guarantee every eager operation uses all threads.
- Qwen's pinned model card at `89644892e4d85e24eaac8bacfd4f463576704203`,
  README lines 11–18, 53, 77–130: image/text input reaches the same
  conditional-generation model and yields decoded text. The upstream card's
  obsolete source-install recommendation is not treated as this project's
  dependency pin or benchmark. The project's exact 4.57.6 pin is separate.
- OpenAI's pinned Whisper-turbo model card at
  `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, Usage and transcription vs
  translation paragraphs (lines 111–179, 206–231): an ASR model produces
  transcripts; known language and `task=transcribe` have the described meaning.
  The project explicitly uses Chinese transcription and keeps its ASR model
  on CPU. Neither official card establishes this project's answer quality.
- HF Hub v0.36.2 `file_download.py` lines 817–832, 906–913, 1092–1121,
  1517–1548: exact revisions address cache entries; local-files-only avoids
  download and missing cache fails. `_headers.py` lines 55–84 explicitly says
  `token=False` suppresses authentication. Project fetching supplies it.
- Transformers v4.57.6 `modeling_utils.py` lines 4384–4412, 4472–4481,
  4508–4532: `from_pretrained` accepts commit revisions/local-files-only and
  dtype overrides, and loads the pretrained weights rather than an optional
  adapter alone. The project separately loads processor, core and ASR pins.
- Python 3.12.15 `hashlib`, Usage/Hash objects: SHA-256 hashes bytes, `update`
  covers concatenated input, `hexdigest` is the full hexadecimal digest.
  File matching establishes the selected bytes, not response quality.
- RFC 1122 (October 1989), §3.2.1.3(g), text lines 1783–1786: 127/8 is
  internal host loopback and must not appear outside that host. This supports
  why a phone's 127.0.0.1 is not the desktop server.

## Actual verification and limits

`cpu_probe.py` really ran on Python 3.13.5 / Torch 2.14.1+cpu. Its other actual
versions are in `probe-library-versions.json`: HF Hub 1.33.0, Pillow 12.3.0,
SoundFile 0.14.0, NumPy 2.5.3. This is not a new installation or replay of
the guide's Python 3.12 / pinned runtime. The CLI `--list` executed unchanged.
The current environment correctly fails the product's Python 3.12 guard.
Small existing public files were injected into the actual fetch/verify
functions offline. This tests hash, size, directory preservation and extras;
it is not a fresh public download. A current-HF missing-cache probe blocked
every attempted request and observed zero requests plus LocalEntryNotFoundError;
the pinned HF source separately supplies the 0.36.2 contract.

Actual small PNG/JPEG/WebP samples passed; animated WebP failed. A 16 kHz
480,000-sample WAV (exactly 30 seconds) passed; 480,001 samples failed.
Injected inference callbacks exercise actual session operations: transcription
loads ASR lazily without generating a reply; the corrected prompt retains a
separate original transcript; the second chat retains the first photo in its
history; reset removes both uploads and state; server_close removes a later
upload directory. These callbacks prove wiring only. No real model inference,
new model/data download, installation, GPU work or training was performed.

Independently recomputed the two public file sizes, all 23 snapshot-file sizes,
GiB/KiB conversions, rounded startup/chat/ASR/workflow times, and 8 MiB ↔ MB.
All equal the guide. The server-only maximum RSS was checked against its
RUSAGE_SELF observer, not assumed from the runner's RUSAGE_CHILDREN. The timer
harness starts before subprocess launch, waits for the actual UI, measures each
browser action, and ends after service shutdown. These are original record
audits: one Linux CPU workflow, four chat calls, one ASR call, one photo and one
recording, no warmup run. The audio is the recorded AISHELL validation utterance;
the photo is a training-split fixture used for smoke operation, not quality
evaluation. The manifest declares seed 42; generation uses `do_sample=False`.
No new GPU or full benchmark replication is claimed.

The guide correctly excludes Python environment/browser overhead from the
relevant rows, distinguishes complete-workflow duration from answer latency,
and explicitly refuses minimum-memory, universal latency, cross-platform or
quality conclusions from this one run. Verbatim copying versus summarization
requires different comparisons: mentioning a familiar name cannot logically
prove every requested character is copied. This is task/metric reasoning,
not a model quality measurement.

## Browser and figure personally inspected

Chromium 151.0.7922.173 really visited the executed preview at 8782, navigated to
its observed STUDENT href, and exposed all seven guide sections. I personally
read the saved visible article text through EOF and viewed the top/measurement
screenshots. Its 20.1 content link was clicked and showed the expected section
heading at `/20.1.html`. Current guide bytes equal the literal `307c325` blob.
The preview predates local 10.8/11.2 revisions; I make no claim about those
revisions. Reading-time metadata is unfinished; this is not publication approval.

The guide has no direct SVG. Its necessary 20.1 prerequisite uses
`course/figures/natural_shared_chat.svg`, SHA-256
`76bb0c7b66623d3c1cafd595430a8b5fd7fe497720111caad28591d0303d46c1`.
I read its SVG source, personally retrieved the same bytes from the actual
preview, rendered them in Chromium and used view_image on the resulting PNG.
Typed input bypasses ASR; recording flows down through ASR; both join history
and the current question; the photo arrow enters the same core; output is text.
The optional fine-tuning box is a general route; the guide's selected variant
is explicitly base-only. No numeric labels or arrow direction conflict exists.
The diagram's actual-ASR route does not claim the guide forbids explicit manual
correction; guide and UI retain the original/correction distinction.

## Preserved real harness failures

First browser attempt selected a hidden collapsed sidebar link and timed out;
its initial code and actual observed failure description are preserved.
Second attempt expected h2 on an exported single-section h1 page; exact stderr
and code are preserved. Diagnostic browser inspection confirmed the link really
opened the correct page. Third attempt timed out taking a full-page screenshot
of a standalone SVG; exact stderr/code are preserved. Rendering the personally
retrieved SVG bytes in an ordinary Chromium HTML page succeeded. First CPU
probe tried `call_count` on a directly injected function; exact stderr/code
are preserved. Using a mock with the same side effect corrected this harness
error. The final browser and CPU executions both exit 0. These failures are
not disguised as product regressions and do not change the guide's source.
