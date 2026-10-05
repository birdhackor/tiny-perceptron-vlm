# 2.5 permanent evidence storage recheck, 2026-10-05

Reviewer: /root/phase4_factual_coordinator/factual_2_5, the original independent
2.5 reviewer. There is no manuscript or figure revision in this recheck.

The initial pass report was preserved as opaque bytes before repair at
docs/technical-reviews/history/phase4-2_5-own-initial-pass-b4f2685b62da833f3723d8c705f61d4874e0cfa33eb9f05fc6159e534adbdcdb.json
with SHA-256 b4f2685b62da833f3723d8c705f61d4874e0cfa33eb9f05fc6159e534adbdcdb.

The root audit found that checkpoint-mlp1/3/5 were mandatory artifact entries
whose paths pointed to ignored outputs/course-experiments/.../*.pt. They existed
and matched hashes during the original evaluation, but a fresh Git/CI checkout
does not retain those files. This was a storage-contract problem, not a new
judgment on the chapter's technical claims.

Correction: replace those three required file-artifact entries with the permanent
input-provenance.json. It preserves the exact original local checkpoint paths,
SHA-256, source run/revision/config, model contract, complete original recorded
per-split means/sums/logsoftmax sums and target denominators. Published 26f34eb
and already-existing local 5d60e35 CPU versions remain separate. The original
weights are identified as historical inputs only and are not report-required
artifacts, repository-code sources, committed snapshots, or fresh Git/CI inputs.
No unverified public URL is substituted for the local run.

All six substantive claims, original evidence locators/supports, numeric scope,
and original actual model-execution receipts are retained. The original probe
code, command/stdout/stderr, environment, raw historical/local results, dataset
bytes, source-code versions, official originals and actual figure renders remain
permanent. Replaying that historical forward evaluation requires separately
obtaining its identified weights; the repaired report does not claim to provide
those weights or to have newly repeated the forward pass.

Actual recheck: permanent-input-provenance-recheck.py read the current manuscript
and figures for byte identity, checked all retained permanent artifact/source
hashes, compared checkpoint identities and source/data/config against the stored
original result JSON and original probe JSON, and independently recounted the
9/1/2 documents and 103/11/22 targets from permanent data. It optionally hashed
the currently existing checkpoint bytes for identity only. It did not load a
model, import torch, run forward/evaluation, download a model, copy weights or
train. The result and stdout distinguish this storage audit from original CPU
model execution. The corrected report is then checked with the current schema.
