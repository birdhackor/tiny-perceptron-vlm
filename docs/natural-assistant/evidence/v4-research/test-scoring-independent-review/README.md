Fresh independent review by actual task `/root/natural_v4_test_scorer_fresh_review`, distinct from author `/root/natural_v4_test_scoring`. No candidate validation outputs were inspected, no model was invoked, and no GPU or publication job was dispatched.

The final [read report](read-report.json) records the original sources, exact hashes, commands, outcomes, finding and limits. The new selected-model v4 test is unrun; these files contain no new test scores or semantic grades.

One medium finding was reproduced and closed. Initially the review binding omitted the grading code. Changing a copied scorer after blind export to casefold OCR changed its synthetic count from 9/10 to 10/10 with unchanged artifact binding. The author added actual SHA-256 and byte-count bindings for the scorer, imported validation helper and core, checked again at export and scoring. The fresh reproduction now rejects the changed CLI before creating scores.

- [Before-fix reproduction](reproduction-before-fix.json) and [original read report](read-report-before-fix.json) preserve the initial evidence.
- [After-fix reproduction](reproduction-after-fix.json) preserves independent closure. The [reproduction program](reproduce-scoring-code-gap.py) only creates temporary synthetic fixture copies under this directory; `--expect-rejected` checks the fixed behavior.
- [Updated test output](pytest-after-fix.stdout.txt): 129 passed in 39.91 seconds, exit 0; [stderr](pytest-after-fix.stderr.txt) is empty. The initial target suite had 115 passes. Reviewer-owned generated pytest fixtures were removed after completion.
- [Actual prior schema check](actual-prior-schema-check.json) checks old real v3 raw suffix and transcript fields. It is explicitly separate from any new v4 result.

The implementation retains 178 LM, 22 ASR and 147 manual-test denominators, strict EOS masks, frozen exact normalization, independently recomputed raw/normalized CER and rational test scoring. It writes no selection decision. Actual source assets were hash-checked by the core manifest loader; this review did not semantically grade photos or model answers.

Physical base/ASR weight SHAs are known from actual CPU prepare receipts, while the test loader checks the HF model/revision and runtime pins without per-forward full-file rehashing. System/history are reconstructed from committed manifest/source code; raw records do not independently preserve rendered chat-template tokens. Grader independence and source-image inspection still require actual external task/read evidence beyond Boolean attestations. These are accurately bounded limits, not claimed validations.
