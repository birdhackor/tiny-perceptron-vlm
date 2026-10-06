# Repository-card release engineering evidence

This archive records independent engineering approval of frozen patch
`10d06273609487db5c5d40e75affbdc5017f492e9dd77e81387de9e665b36ab4`.
The patch changes only the existing `modal_runner.py` and `hf_transport.py`.

The frozen author's proof, synthetic test source, independent actual proof,
test adapter, extra checks, raw pytest logs and pinned Hub SDK source excerpts
are copied without modification. [integrity-index.json](integrity-index.json)
records each original path, copied path, byte count and both SHA-256 values.

The independent candidate runs passed 40 author-test cases, 85 existing
infrastructure/gross-quota cases and 7 additional cases. The extra tests use
the actual pinned Hub 1.33.0 SDK control flow for a singleton README CAS,
SDK no-op race rejection and an ambiguous response without retry. The initial
extra-test fixture failure and its final passing log are both retained;
the reviewer corrected the fixture's missing regular upload mode, leaving
the frozen candidate unchanged.

The original [independent proof](independent/actualproof.json) records the
review before root applied the patch and remains byte-for-byte unchanged.
During archiving, the reviewer observed actual runner SHA
`ee7676871d1c24962905660775b869f50ac5c83818980934cef920f5ba1c2947`
and transport SHA
`500417c5f922cd1ce7ae0e6add8a31e2bc34bb7434a30bd845be93b47e831600`,
both matching the frozen candidate. This is a code-byte observation only;
the separate root regression run was still in progress at the archive request.

Approval covers the committed-source gates, single authorized root README
operation, pinned-parent CAS, immutable anonymous byte verification,
accurate no-op status, and preservation of the existing batch/resources,
Secret, serialization and gross-quota controls. It does not establish actual
HF publication or production financial readiness. No remote, paid, GPU,
model/checkpoint or held-out-gold operation was performed for this review.
The archived test sources retain their original execution paths; this archive
does not introduce a production execution workflow.
