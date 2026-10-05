# 15.12 independent factual inspection

Reviewer: /root/phase4_factual_coordinator/factual_15_12; fresh single-section task.
Date: 2026-10-05. This is technical verification, not a first-reading session.

The exact 15.12 UTF-8 bytes were extracted and read from course/chapters/15.md,
beginning at original line 453. SHA-256:
e70682d67c01d2d5d8ed9bd2271a2211c1a2fdd2fa2caa4057c682508e60fa5e.
The full chapter was frozen in inputs/chapter-15-frozen.md, SHA-256
d261089b88d4d274b09276f06322e527a7862492768c06a8236614f2efea75be.
That hash describes this actual frozen input snapshot, not a claim about future
versions of the full chapter. The chapter introduction was not read or reviewed.

Necessary original pretext actually read: 15.4 (top-k definition), 15.8
(counts versus router probabilities and what imbalance means), 4.2 (residual
y=x+f(x)). Exact raw section snapshots are retained. Only link destinations
from 15.13 were extracted to locate the primary moe.json file; the 15.13 body
was not substantively reviewed. This does not provide a technical verdict on
any other section.

Instruction files were personally read: factual-reviewer-instructions.md,
section_facts.py, check_technical_reviews.py (schema and CLI), clear-tutorial
SKILL.md, and review-protocol.md. Root/ancestor AGENTS.md checks found no file.
The cloud-environment-onboarding setup skill and its onboarding reference were
read. Existing .venv already supports this CPU workflow; no installation or
environment configuration change was needed.

Implementation reads were AST-located before reading source blocks:
current modern.py DenseFFN 35-53 and MoEFFN 56-84; current model.py ModelConfig
15-28 and Block 31-50; current architecture.py run_moe 430-493. The matching
experiment revision 48a4f3e912b483d70aee57c42c2aac226534a9a6 was retrieved with
git show (read-only), then its AST-located MoEFFN 56-84, ModelConfig 15-28,
Block 31-50, and run_moe 428-491 were personally read. Original modern/model
hashes equal current files; architecture's full-file hash differs from current,
so archival provenance is backed by the retained original revision and exact
recorded hash, not by pretending current architecture is the historical file.
Ordinary implementation comments and methodological limitations were read;
none contained an author-result correction or old review verdict.

Raw result handling: initial and branch key/type inspection only, followed by
named provenance, original code hashes, model/config, and raw requested/actual
step, update, skipped-update, and effective-token pointers. The full unmodified
raw JSON is retained with SHA-256 a95f60479ef6c51b018af486b1e42d3f8e6ea99bb90c88470184406adb8f9890.
The executed verify_provenance.py prints the exact value-pointer log; metric
key/type inspection is separately retained. No heldout-quality values, author
result notes, correction summaries, reader reports, or earlier technical-report
bodies were read. Broad initial filename/status listings exposed filenames only,
not review content or judgments. No contamination event occurred.

External originals personally checked: Switch Transformers arXiv2101.03961v3
(16 Jun 2022), title/version, section2.2 Equation3 and fixed-capacity/residual
paragraph on p7; preceding k=1 discussion on p6; AppendixB on p29 and Figure11
caption on p30. Compact raw text excerpts and full-download/extraction hashes
are retained. PyTorch official version2.14 bincount and clamp API descriptions
and Python3.13 math.ceil API description were read from primary official pages.
The stable PyTorch URLs were meta-redirects to version2.14; actual version2.14
pages were fetched and inspected. Retrieval URLs, commands, access date, versions,
full-download hashes, and raw HTML fragments are in primary/.

Execution scope: exact original Python fence executed under section_facts.py's
CPU/offline/artifact-only guard; exit0, no guard events. Independent bounded CPU
script ran original and only-factor variants 1/2, k=2 denominator counterexample,
and two MoEFFN forward checks with six tokens concentrated on one first expert.
No backward call, optimizer update, checkpoint, dataset/model download, full
training, GPU execution, or model-quality reevaluation occurred. Historical
training was CUDA; those archival records were inspected, not rerun.

Figure scope: section15.12 has no direct image or SVG reference; its capacity
example is specified fully by the chosen list, counts, formula, and prose.
There is no visual-material claim requiring a render PASS for this section.
The necessary 4.2 pretext's residual SVG was additionally rendered and viewed:
one Chromium attempt had a bounded20-second timeout and produced no image;
Inkscape then rendered an actual1180-wide PNG. Personally viewed x=[1,2],
identity=[1,2], correction2*x=[2,4], merging arrows/plus, y=[3,6], and sensitivity
1+2=3. That verifies the ancillary figure's content only. Desktop/mobile course
page browser layout was not verified and is not claimed as passed.

The saved runtime evidence excludes runtime/workspace symlinks and weights.
Only explicitly selected helper files and stdout/stderr/environment/commands
were copied from /tmp into the permanent artifacts directory. No old report
was used as an answer, no textbook or original figure was edited, and no commit
or push was performed. The new report is written independently before checker.
