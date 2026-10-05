# Original reviewer formal recheck of 19.5

Reviewer task: `/root/phase4_factual_coordinator/factual_19_5`, the same independent factual reviewer who found I1. Date: 2026-10-05. The coordinator's writer receipt was treated as a proposed change; the reviewer independently read the actual current files before deciding this recheck.

## Actual reading and version

I personally reread **all** current 19.5, from its heading through the closing details marker, including the whole unchanged fence, all scores, limitations, input/output example and final links. Current original UTF-8 SHA-256 is `f07a4500b54927e9a2820ecdfa2e8892d98404c34818a45bcf66e284e4d1ecd3`. It has one unchanged Python fence and no referenced figure.

I personally reread the necessary **current** 19.12 target: section lines 1–15 and 29–51, particularly capability definitions in line 3, the criteria matrix lines 5–11, its explicit pending/no-score status in line 13, and old scope/criteria lines 38–42 and 51. The whole context section was saved without alteration, but only those specified ranges were read in this recheck. Its SHA-256 is `7d90da817d297f56bc68601d60efc18de5a1078ea3251ef28612e11fea9d439f`, identical to my initial target snapshot.

The complete current chapter was frozen solely as input provenance in `chapter-frozen-input.md`, SHA-256 `25432acb0964bad584a97846e0168b33374d2f75cfce8accb49faebf556cb4b1`. This is the actual chapter at the current extraction, not a declaration of later whole-chapter freshness. The initial complete chapter and initial issue/report remain preserved separately.

The only 19.5 change, independently checked at raw-byte level, replaces:

> 最終文字能力與各項分母統一見[19.12的能力表](19.md#19.12)

with:

> 整合成品的能力界定與驗收安排見[19.12](19.md#19.12)

Both fixed-commit joint/DPO validation links remain exactly unchanged. The new wording names what the target actually supplies and makes no claim that final text task scores have been tabulated there.

## Recheck of every substantive claim

- **C1**: Rechecked the original question specification: selected bag/boot are two items and two lines. At 14 currency units/item, quantities 1 and 2 give 14 and 28, so missing quantity prevents a unique total. This remains a hand example, not a model result.
- **C2**: Rechecked complete demonstrations, generic system conditions and EOS support. Personally reread original `prompt_ids`/`encode_record` source lines 299–318; reread original InstructGPT §3.1 Step 1 and §3.6, and official Transformers v4.57.1 role description lines 78–84. Nothing in the new lesson claims that system settings alone train the behavior. The generic system contains tool availability/style, not per-question answers.
- **C3**: Reread the entire original fence and its real CPU stdout from the initial execution. It still only selects one row/task from train and prints question/system/author answer. Raw fence bytes are identical. The initial exit-0 execution, saved version/device/guard record and code evidence remain applicable. No second run of an unchanged fence was necessary for this reference-only correction.
- **C4**: Rechecked my personally produced raw-record results and all four SFT scores/EOS evidence: style 3/3, missing 3/3, unavailable 10/10, safety 3/3. The original source and recorded measurement SHA identities remain unchanged. These are still validation counts, not final-test scores or new model inference.
- **C5**: Rechecked the other-text denominator decomposition 10 calculator + 5 tool_return + 5 concept + 3 rag = 23, all correct. Including the four text behaviors gives 42/42. The conclusion is only that this small fixed set was not entirely refused.
- **C6**: Rechecked my raw-label counts for missing, unavailable, safety and rag. Each task has one constant expected payload; rag's three true labels are all `DIRECT:書櫃`. The full current lesson explicitly preserves these limits. It does not claim general safety, calibrated confidence or varied-source QA.
- **C7**: Rechecked the same-ID question/system/generated example `691c9de656c01fa2df60`, my prior prompt/generated token identity results and source version evidence. All ten original implementation/raw files still have the personally inspected pinned-commit hashes. The IDs/raw strings and readable source questions remain distinct.
- **C8**: Rechecked the archived comparison: all three stages retain 42/42 text; joint-task performance goes from joint 18/18 to DPO 14/18. My prior record-level comparison lists four changed correctness outcomes, all in task joint. No new scores or general claim about DPO were introduced.
- **C9**: Reread the whole future-product criteria paragraph and the introduction. New phrasing/list/history/answerable-versus-insufficient checks remain planned acceptance criteria and a staged course design, not completed product acceptance. The current 19.12 matrix explicitly confirms its pending status.
- **C10 / I1**: The actual revised link wording is now supported by the reread target: 19.12 line 3 gives capability-scope definitions; lines 5–13 give acceptance criteria and pending status; lines 38–42,51 distinguish old synthetic limits and evaluation criteria. The prior promised final text score table is no longer asserted. I1 is resolved by this actual source change and independent reread.

The original `evaluate_rows` result aggregation / `parse_action` lines 537–561 were also reread to verify the full-action-plus-EOS criterion. The original complete-method inspection and raw JSON pointer log remain in the initial evidence bundle; they were not replaced by another reviewer's claims.

## Bounded execution and limits

Executed `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-19_5-independent-factual/recheck/verify_recheck.py > docs/technical-reviews/artifacts/phase4-19_5-independent-factual/recheck/stdout.txt 2> docs/technical-reviews/artifacts/phase4-19_5-independent-factual/recheck/stderr.txt`, exit 0. This checks current raw section identity, the exact single sentence replacement, unchanged fence/target bytes, ten original code/result hashes and exact hand arithmetic. Command code, stdout/stderr, environment and JSON result are permanent. It does not evaluate a model.

No GPU, training, new model score, data/model download, model engineering, figure editing or outside report editing occurred. Initial raw snapshots, initial `revise` report SHA `6559424bfc52bea5840a8a026c16ab972358c34e58b8171e8c0f75ffeadf5f5b`, initial issue quote and checker rejection remain preserved. No old external review or author outcome summary was read.

My independent recheck verdict is **pass**, with no unresolved factual issue. This remains an AI factual review and no broader product acceptance is inferred.
