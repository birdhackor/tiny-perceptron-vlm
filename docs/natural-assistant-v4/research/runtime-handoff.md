# v4 bounded runner handoff

This is an implementation handoff for the coordinator, not a lesson, dataset freeze, measured GPU result or permission request. No GPU job was started by this author.

The manual workflow remains `.github/workflows/natural-assistant.yml`. All manifests stay in the existing committed `docs/natural-assistant/` namespace; v4 uses `docs/natural-assistant/v4/manifest.json`. Release approvals stay in `docs/natural-assistant/releases/`. Private/public HF prefixes remain `natural-v3/`; use a distinct batch/release ID such as `natural-v4` / `assistant-2b-v4`.

Current bounded options:

| Input | Contract |
| --- | --- |
| `learning_rate` | Only `0.0001` or `0.00003`; legacy default remains `0.00003` |
| `asr_variant` | `small` or `turbo`; fixed model/revision pairs, legacy default small |
| `steps` | 1–3000 requested updates |
| `max_seconds` | GPU at most 3300, CPU stages at most 1620; leave hard-timeout cleanup margin |
| `checkpoint_steps` | Train only, at most two increasing completed-update boundaries, e.g. `500,1000` |
| `adapter_checkpoint` | One `step-NNNNNN` archived version from `adapter_run_id`; empty uses legacy latest adapter |
| `adapter_checkpoints` | Validation only, at most two distinct archived names; cannot also supply singular selection |

`small` is `openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d`; `turbo` is `openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Turbo metadata/card were anonymously fetched and saved at `docs/natural-assistant/evidence/v4-research/experiment/`; no turbo weights or quality test were performed by this author.

Reserve and execute receive identical options in the workflow. A canonical configuration SHA is saved in the durable reservation and checked against both the actual runner arguments and ledger before execution. The original cumulative budget remains US$40; the prior cumulative reservation is retained, not recreated. Source/manifest bytes, selected weight bytes and real validation result bytes retain their existing checks.

## Dispatch sequence

Use a committed ref that includes the runtime, frozen v4 manifest and protocol; do not dispatch an uncommitted local file. Set the variables below to actual coordinator values. These commands are examples, not commands executed by this author.

```bash
training_ref=YOUR_COMMITTED_REMOTE_REF
train_rows=YOUR_FROZEN_TRAIN_ROW_COUNT
first_update=$(((train_rows + 1) / 2))
last_update=$train_rows
printf -v first_name 'step-%06d' "$first_update"
printf -v last_name 'step-%06d' "$last_update"
```

With accumulation=2 and an odd row count, the first archive visits one extra record beyond the first full pass. `last_update=train_rows` gives exactly twice the row count in total visits; describe actual visits rather than claiming both archives land at exact epoch boundaries. `train_rows` must be at most 3000 for this two-pass schedule. If there are too few rows to produce distinct boundaries, declare a different bounded schedule before outputs.

Run `prepare` first with the same batch/manifest/ASR choice; it downloads/checksums matching data and fixed model caches. A GPU developer baseline/fit check can use the same workflow's `baseline` stage on validation only.

```bash
gh workflow run natural-assistant.yml --ref "$training_ref" \
  -f stage=train -f batch_id=natural-v4 \
  -f manifest=docs/natural-assistant/v4/manifest.json \
  -f steps="$last_update" -f max_seconds=3300 \
  -f learning_rate=0.0001 -f asr_variant=small \
  -f checkpoint_steps="$first_update,$last_update"
```

Each boundary saves the actual then-current adapter/config, complete training metadata, optimizer and CPU/CUDA RNG state. Archives are immutable at `checkpoints/step-NNNNNN/`; the normal rotating latest `adapter/` still exists. The private backup includes full archives and verifies binary files after HF upload. The Actions review copy contains inference weights/JSON metadata and raw outputs, excluding optimizer/RNG `.pt` files.

After the completed train run, replace `TRAIN_RUN_ID` with its actual `gha-...-...` ID:

```bash
gh workflow run natural-assistant.yml --ref "$training_ref" \
  -f stage=validation -f batch_id=natural-v4 \
  -f manifest=docs/natural-assistant/v4/manifest.json \
  -f adapter_run_id=TRAIN_RUN_ID -f max_seconds=3300 \
  -f learning_rate=0.0001 -f asr_variant=small \
  -f adapter_checkpoints="$first_name,$last_name"
```

Audio rows explicitly marked `speech_transcription` run ASR/CER only. `speech_chat` and legacy `asr` rows generate the paired typed-reference/actual-ASR replies; missing task retains legacy behavior, unknown task fails. Completion counts reflect chat-eligible audio rows and also require every requested transcription to finish. Transcription outputs retain original references and dataset source, with task/source CER groups.

One core model compares variants `base`, `adapter-step-NNNNNN`, `adapter-step-NNNNNN`, using native PEFT adapter switching. ASR is loaded/transcribed once for all variants; typed reference and actual ASR hypothesis remain distinct responses. Each variant has its own full raw generation file and completion flag. Do not count a partial run as completed validation.

## Selected checkpoint and release

The coordinator freezes a validation-only decision in `docs/natural-assistant/v4/selection.json`. For a selected archive, include:

```json
{
  "dataset_manifest_sha256": "ACTUAL_MANIFEST_SHA256",
  "validation_run_id": "ACTUAL_VALIDATION_RUN_ID",
  "validation_result_sha256": "ACTUAL_COMPLETED_RESULT_SHA256",
  "selected_variant": "adapter-step-000500",
  "adapter_run_id": "ACTUAL_TRAIN_RUN_ID",
  "adapter_checkpoint": "step-000500",
  "adapter_sha256": "ACTUAL_ARCHIVED_WEIGHT_SHA256",
  "criterion": "ACTUAL_PREDECLARED_CRITERION",
  "decision": "ACTUAL_VALIDATION_DECISION"
}
```

Use the actual selected step, not the example `000500`. If base wins, use `selected_variant=base`, no adapter identifiers/hashes/checkpoint, and no adapter argument at final evaluation. Latest-only old decisions remain compatible with `selected_variant=adapter` and empty checkpoint.

For final `evaluate`, supply the committed selection file, exact train ID and singular `adapter_checkpoint`; leave plural comparison empty. All settings affecting final evaluation must be predeclared. Both reservation and execution verify selected checkpoint bytes against the actual completed validation report. New test results do not reselect the checkpoint.

Release retains the explicit inference-only allowlist. To publish an earlier archive at the legacy public adapter paths, use the train-run private HF prefix and optional `source_path`:

```json
{
  "path": "adapter/adapter_model.safetensors",
  "source_path": "checkpoints/step-000500/adapter_model.safetensors",
  "bytes": 123,
  "sha256": "ACTUAL_ARCHIVED_WEIGHT_SHA256",
  "license": "Apache-2.0",
  "redistribution_approved": true
}
```

Add the actual `adapter_config.json` similarly. The coordinator provides actual byte counts, hashes, pinned private HF source commit, reviewed model card and approval fields. Mapping permits only the same inference adapter/config basename from latest or an explicit archive; optimizer/RNG/training state cannot be renamed into public metadata. Release downloads and verifies exact private bytes and anonymously checks the exact public bytes after upload. This mapping keeps the existing student fetch/serve public layout.

## Verification performed

- Actual existing Modal CLI `--help` in control phase lists all new options; no remote stage was invoked.
- Ruff lint/format and Python compile passed for the four runtime files/new test.
- 148 targeted natural runtime/data/UI/release tests passed after all runtime changes, including mixed audio task routing and partial-ASR completion checks.
- Tests use real CPU AdamW updates and a minimal valid safetensors fixture to prove archived first/last tensor values differ, final archive equals the live/final adapter, and optimizer step/RNG/metadata belong to the real update. They also reject overwritten/modified archives, configuration drift and selection of an unvalidated checkpoint; actual client/runner argument construction and paired adapter/ASR switching are exercised without remote calls.

Actual 2B GPU checkpoint creation, final holdout quality and public release are still to be performed by the coordinator. No inference-quality claim follows from the CPU fixture tests.
