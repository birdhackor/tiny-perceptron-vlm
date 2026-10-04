# Selected public BASE + turbo: actual anonymous retrieval and CPU browser trial

This completed trial used the published `assistant-2b-v4` manifest, not the earlier core-only infrastructure probe. Native `list`, `fetch` and `verify` all exited 0 against a new destination and new empty anonymous Hugging Face cache. Chromium then operated the unchanged product UI against the verified selected release. The models came from an existing official pinned cache and served offline; this was not a fresh model-weight download.

The actual run source commit was `59a1eda4ed7b6e8609892ec2b9013c821ac93e69`. The public manifest was 2,200 bytes, SHA-256 `1758ceb9f8859108fa3ad315fb8f6a61e0c8cef40ec49b0bc41361d9c70e5ccf`. The public repository was `birdhackor/tiny-perceptron-course-models` at exact commit `d3954d6900b3cf81e593d99d9b8b1a91e6f9741d`. Its downloaded provenance records release source commit `42c2516362340a140b68619bcb0f3025a0215f9a`, data manifest SHA-256 `0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60`, and approval SHA-256 `05229a7132e7c4045c5ffe4996135d27282218093ba998f0b3c48011878b7a38`. These are different identities with different purposes. Root resumed separate book/evidence work after the run capture; closure records that later HEAD independently.

The selected variant was BASE, adapter parameters 0. Qwen/Qwen3-VL-2B-Instruct was pinned to `89644892e4d85e24eaac8bacfd4f463576704203` (upstream Apache-2.0). Whisper-large-v3-turbo was pinned to `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` (upstream MIT). Public project documents were MIT. No upstream weights were copied into these evidence files or the public project release.

Actual anonymous traffic comprised four HEAD requests and two successful GET requests, without Authorization, Cookie or Proxy-Authorization headers. `token=False` was used and credential environment variables were removed. Native responses were observed and delegated unchanged; no HTTP or model response was mocked. The retrieved files were independently verified:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| README.md | 8,722 | c319a97d90aca2a074410e3ece8c6325515e0c3a336660e4ca1d00ba6fab65fd |
| release-provenance.json | 540 | b04dd5f22bc42c62fe9a57a858991ba4d256311442d7db474a90512001893ff4 |

Chromium `151.0.7922.173` actually loaded the page, typed and clicked send, selected a training DOCCI photo, selected an existing validation AISHELL recording, clicked transcribe, edited the copied transcript, sent that correction, sent a history follow-up, and clicked “開始新對話”. Four serial language-model generations and one ASR call ran, with the core and ASR each loaded once. The DOM contained the actual returned text. Historical photo bytes persisted into later requests; the original transcript and submitted correction remained separately displayed. Reset cleared eight rendered messages, server history, assets and stored transcriptions. Shutdown exited 0 and removed the uploaded temporary files. The trial used no test rows, scoring references or quality comparisons.

The five representative screenshots in [screenshots/](screenshots/) were actually viewed by the reviewing AI agent: ready page, transcript before editing, submitted correction, 358px mobile conversation, and reset page. Controls, image, original-transcript note and history were visible; the mobile browser measured no horizontal overflow and logged no console errors. This was an automated headless browser trial with AI screenshot review. `browser_manual_trial_complete:false` remains in the original report; no human manual click trial or human usability study is claimed.

| Actual resource observation | Value |
| --- | ---: |
| CPU settings | FP32; 5 Torch threads; 1 interop thread; offline serve |
| UI ready time | 24.24 s |
| Complete harness wall time | 170.33 s |
| Browser-observed four chat times | 3.42 / 42.87 / 36.58 / 37.11 s |
| Browser-observed ASR time, including lazy loading | 18.03 s |
| Model-server CPU time | 465.82 s user + 87.66 s system |
| Model-server maximum RSS | 13,496,104 KiB = 12.8709 GiB |
| Existing official cache, unique regular bytes | 5,889,111,977 bytes |
| Fresh public destination, including verified manifest receipt | 11,462 bytes |

These measured values describe this machine and these interactions, not minimum requirements or a speed benchmark. Server-only `RUSAGE_SELF` metrics exclude Chromium and the driver; the separate aggregate child-resource fields are preserved in the original report. All four language generations and the ASR decoder ended with EOS without hitting their token cap. EOS establishes decoder stopping, not semantic completeness or recognition accuracy. Limits were 1,800 seconds overall and 600 seconds per browser request.

The exact executed harnesses are available as inert `.py.source.txt` files in [executed-harness/](executed-harness/). Reproduction in the original execution workspace used:

```bash
.venv-natural/bin/python outputs/natural-v4/release-preflight/run_public_fetch.py \
  --manifest docs/natural-assistant/v4/public-release.json \
  --manifest-sha256 1758ceb9f8859108fa3ad315fb8f6a61e0c8cef40ec49b0bc41361d9c70e5ccf \
  --output outputs/natural-v4/release-preflight/anonymous-selected-release \
  --evidence outputs/natural-v4/release-preflight/anonymous-selected-fetch-001

.venv-natural/bin/python outputs/natural-v4/release-preflight/run_ui_smoke.py \
  --mode release --interaction browser \
  --manifest docs/natural-assistant/v4/public-release.json \
  --manifest-sha256 1758ceb9f8859108fa3ad315fb8f6a61e0c8cef40ec49b0bc41361d9c70e5ccf \
  --release-directory outputs/natural-v4/release-preflight/anonymous-selected-release \
  --cache-dir outputs/natural-v4/student-base-cache/hf \
  --image outputs/natural-v4/anonymous-data-download/data/vision/images/train_07076.jpg \
  --audio outputs/natural-v4/anonymous-data-download/data/voice/audio/validation/aishell1-BAC009S0005W0205.wav \
  --output outputs/natural-v4/release-preflight/selected-public-ui-run-001 \
  --max-seconds 1800 --request-timeout 600
```

The run IDs above already exist and must not be overwritten; reproduction needs new IDs. The `outputs/` helpers and raw paths are execution artifacts, not ordinary checkout files. To reproduce from checkout, materialize the archived inert harness copies under their original paths and provide the pinned verified official cache and downloaded data. [receipt.json](receipt.json), [source-index.json](source-index.json), `anonymous-fetch/`, `actual-ui/`, `fetched-public-files/` and the five copied screenshots are small checkout-readable evidence. The other indexed raw files require a separately published proof archive; this trial did not publish an archive. The source index marks these two availability scopes explicitly. It excludes model weights, HF caches, the original photo and original WAV.

Closure executed native `validate_manifest`, `verify_release` and `check_runtime` without model loading or network retrieval. It rechecked the six product source hashes, manifest hash, both input byte hashes, native anonymous request counts, actual load counts, history progression and reset state. The model process and uploaded files were absent. Its exact source and actual check outputs are preserved here. No book, guide, product runtime, model data, gold label or Git state was changed by this trial.
