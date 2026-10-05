"""Write this reviewer's independently checked W.3 report, without reading its predecessor."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path):
    return path.relative_to(ROOT).as_posix()
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

meta = json.loads((BASE / "extraction.json").read_bytes())
raw = (ROOT / "course/first-steps.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
i = next(i for i,h in enumerate(headers) if h[0].startswith(b"## W.3 "))
section = raw[headers[i].start():headers[i+1].start()]
assert section == (BASE / "section.md").read_bytes()
assert sha(BASE / "frozen-input/course/first-steps.md") == meta["source_file_sha256"]
assert json.loads((BASE / "execution.json").read_bytes())["exit_code"] == 0
variation = json.loads((BASE / "variations-stdout.txt").read_bytes())
assert variation["status"] == "all assertions passed"
runtime = json.loads((BASE / "environment.json").read_bytes())
env = {"python": runtime["python"], "torch": runtime["torch"], "torch_git_revision": runtime["torch_git_version"], "device": "CPU", "CUDA": "No CUDA build; unavailable"}
inspection = {
    "actual_read_scope": ["course/first-steps.md lines 1-110: introductory paragraph, W.1, W.2, entire W.3; W.1/W.2 only as necessary context, no technical verdict on those sections", "course/figures/rewrite-W-3-axes.svg: source plus actual PNG images viewed at 640 and 358 pixels", "all five required review-method files; checker and helper re-read separately after initial combined output truncation"],
    "original_source_ranges": {
        "torch/_torch_docs.py": [[50,122],[7195,7261],[7903,7926],[7925,7988],[9582,9635],[9833,9863]],
        "torch/_tensor_docs.py": [[1665,1672],[2788,2805],[3245,3252],[4163,4179],[4737,4778],[5445,5465]],
        "docs/source/tensors.md": [[1,82]],
        "aten/src/ATen/TensorIndexing.h": [[233,278],[454,476],[550,591],[647,703]],
        "c10/core/WrapDimMinimal.h": [[1,48]],
        "CPython v3.13.5 Doc/library/stdtypes.rst": [[1341,1374]]
    },
    "candidate_lookup": "Original-source-locators top-level keys/types and locator item keys/types, then only torch-related URL/version/path/SHA locator fields; candidates were not treated as answers. No prior report body or author result explanation was read.",
    "original_source_verification": "Personally fetched official HTTPS files at installed PyTorch git revision; _torch_docs.py at v2.14.1 is byte-identical to the commit snapshot. Read the recorded ranges, not only search matches.",
    "figures": {"native_render": "Inkscape 1.4, export-width 640, actual view_image inspection", "smaller_render": "Inkscape 1.4, export-width 358, actual view_image inspection", "observed": "Both display all six scores, correct student/quiz labels, dim=1 horizontal grouped triples to 70/90, dim=0 vertical grouped pairs to 70/80/90. No clipped label or wrong arrow was observed.", "optional_browser_attempt": "Chromium figure-only screenshots were attempted at 1280x900 and 390x844, yielded no files, and the owned commands were terminated. Full course-page browser layout is not claimed."},
    "scope_limit": "Existing bounded tensor examples only. No model training, GPU execution, model/data download, empirical ability score, or engineering change. No textbook or figure edits."
}
save(BASE / "inspection.json", inspection)
artifacts = []
def artifact(identifier, filename, kind, description, **extra):
    p = BASE / filename
    value = dict(id=identifier, path=relative(p), sha256=sha(p), kind=kind, description=description)
    value.update(extra)
    artifacts.append(value)
artifact("original-section", "section.md", "source_snapshot", "Raw UTF-8 W.3 bytes, including its heading and trailing blank lines")
artifact("frozen-whole-input", "frozen-input/course/first-steps.md", "source_snapshot", "Frozen whole Markdown captured for this read; its SHA agrees with the extraction whole-file fingerprint, not a claim about future whole-chapter bytes")
artifact("extraction", "extraction.json", "source_snapshot", "Raw-byte section and figure extraction, original code location and hashes")
artifact("original-code", "fence-1.py", "code", "Exactly the original W.3 Python fence")
artifact("bootstrap", "bootstrap.py", "code", "Actual helper bootstrap used by the original-code execution")
artifact("original-run", "execution.json", "execution", "Original fence actual CPU execution record; stdout, stderr and environment are separate permanent artifacts", command=".venv/bin/python docs/review-tools/section_facts.py course/first-steps.md#W.3 --output /tmp/phase4-W3-factual-original-20261005 --execute --timeout 30", result="exit_code=0; attempted fence [1]; all five original print outputs completed", environment=env)
artifact("original-stdout", "stdout.txt", "source_snapshot", "Actual original-code stdout: Size([2,3]), first row, 100, both arithmetic averages")
artifact("original-stderr", "stderr.txt", "source_snapshot", "Actual original-code stderr: empty")
artifact("original-environment", "environment.json", "source_snapshot", "Actual runtime and CPU/offline helper guard facts")
artifact("variation-code", "variations.py", "code", "Independent bounded assertions covering APIs, exact denominator calculations, exercise, ranks, transpose and failure contracts")
artifact("variation-run", "variations-execution.json", "execution", "Actual bounded CPU variation execution", command=json.loads((BASE / "variations-execution.json").read_bytes())["command"], result="exit_code=0; all assertions passed; no GPU or training", environment=env)
artifact("variation-stdout", "variations-stdout.txt", "source_snapshot", "Actual variation values, denominators, units and expected exceptions")
artifact("variation-stderr", "variations-stderr.txt", "source_snapshot", "Actual variation stderr: empty")
artifact("figure-native", "figure-native.png", "figure_render", "Actual Inkscape 1.4 render at 640x760; personally viewed")
artifact("figure-width358", "figure-width358.png", "figure_render", "Actual Inkscape 1.4 render at width 358; personally viewed; this is figure inspection, not full-page browser validation")
artifact("figure-original", "figures/course/figures/rewrite-W-3-axes.svg", "source_snapshot", "Exact original referenced figure bytes")
artifact("inspection", "inspection.json", "source_snapshot", "True read ranges, independent source inspection and visual findings")
artifact("commands", "commands.json", "source_snapshot", "Actual commands, source acquisition method, working directory, renderer version and permanent-copy provenance")
artifact("report-writer", "write-report.py", "code", "This reviewer's actual report generation code; it never reads predecessor reports")
artifact("acquisition", "sources/acquisition.json", "source_snapshot", "Actual HTTPS source acquisition URLs, versions, bytes and SHA-256")
artifact("tag-version", "sources/tag-version-verification.json", "source_snapshot", "Official v2.14.1 and installed git-revision document bytes match")
artifact("optional-browser", "optional-chromium-attempt.json", "source_snapshot", "Truthful optional browser attempt limitation; no browser screenshot or full-page verification claim")
for identifier, name, desc in [
    ("torch-docs", "torch--_torch_docs.py", "Original official tensor, mean and reshape API documentation at installed revision"),
    ("tensor-docs", "torch--_tensor_docs.py", "Original official shape/size/dim/item/tolist/reshape/mean contracts"),
    ("tensor-basics", "docs--source--tensors.md", "Original official tensor basics and indexing examples"),
    ("index-source", "aten--src--ATen--TensorIndexing.h", "Original official indexing implementation"),
    ("wrap-dim-source", "c10--core--WrapDimMinimal.h", "Original official negative dimension wrapping implementation"),
    ("python-tuples", "python-3.13.5-stdtypes.rst", "Original official Python v3.13.5 tuple constructor documentation")
]: artifact(identifier, "sources/" + name, "source_snapshot", desc)

commit = "5c4886908584029761b579af026dcfb627c84070"
sources = []
def authority(identifier, title, remote, snapshot, note, version=None):
    sources.append(dict(id=identifier, kind="official_source", title=title, url=remote,
        version=version or "PyTorch v2.14.1; immutable git revision " + commit,
        authority_reason="Official upstream project repository, source documentation and implementation, fetched directly over HTTPS",
        accessed_on="2026-10-05", verified=True, checked_original=True, inspection_note=note,
        snapshot_artifact_id=snapshot))
prefix = "https://raw.githubusercontent.com/pytorch/pytorch/" + commit + "/"
authority("torch-docs", "PyTorch tensor / mean / reshape original API docs", prefix + "torch/_torch_docs.py", "torch-docs", "Personally read scalar constructor, vector/matrix vocabulary, mean defaults and reduction contract, reshape data and numel preservation; read ranges are in inspection.json; verified v2.14.1 tag bytes match.")
authority("tensor-docs", "PyTorch Tensor method original API docs", prefix + "torch/_tensor_docs.py", "tensor-docs", "AST located add_docstr_all names, then personally read dim 1665-1672, item 2788-2805, mean 3245-3252, reshape 4163-4179, size/shape 4737-4778, tolist 5445-5465.")
authority("tensor-basics", "PyTorch tensor initializing and basic operations", prefix + "docs/source/tensors.md", "tensor-basics", "Personally read lines 1-82: multidimensional numbers, sequence construction, indexing and single-value Python conversion.")
authority("index-source", "PyTorch tensor integer and multidimensional indexing", prefix + "aten/src/ATen/TensorIndexing.h", "index-source", "Personally read applySelect, integer handleDimInMultiDimIndexing, applySlicing and get_item at recorded line ranges.")
authority("wrap-dim-source", "PyTorch negative dimension wrapping", prefix + "c10/core/WrapDimMinimal.h", "wrap-dim-source", "Personally read _maybe_wrap_dim lines 19-31: for rank N, negative valid dim maps to dim+N.")
authority("python-tuples", "CPython tuple constructor documentation", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst", "python-tuples", "Personally read Tuples, tuple([iterable]), lines 1341-1374: preserves iterable item order.", "CPython v3.13.5 official release tag; installed Python 3.13.5")
sources.extend([
    dict(id="original-execution", kind="execution", title="Actual W.3 original fence CPU execution", verified=True, artifact_id="original-run"),
    dict(id="variation-execution", kind="execution", title="Actual independent W.3 bounded CPU variations", verified=True, artifact_id="variation-run")
])
def evidence(identifier, locator, supports):
    return dict(source_id=identifier, locator=locator, supports=supports)
def claim(identifier, kind, statement, location, scope, refs, aids, verification=None):
    x=dict(id=identifier, kind=kind, statement=statement, location=location, scope=scope, status="verified", evidence=refs, artifact_ids=aids)
    if verification: x["verification"]=verification
    return x
def executed(expected, observed, details, **extras):
    return dict(method="executed", expected=expected, observed=observed, details=details, **extras)
claims = [
    claim("tensor-ranks", "concept", "PyTorch tensors hold structured numeric data; scalars, vectors and matrices are zero-, one- and two-axis forms, and shape records axis lengths.", "course/first-steps.md:80-82", "Dense regular score examples only; axis labels are author-assigned meanings, not inherent Tensor semantics.", [evidence("tensor-basics", "docs/source/tensors.md lines 9-24", "Multidimensional Tensor and construction from lists"), evidence("torch-docs", "torch/_torch_docs.py lines 9628-9632, 7918-7925 and 7961-7972", "Zero-dimensional scalar; one-dimensional vector and two-dimensional matrix vocabulary"), evidence("tensor-docs", "torch/_tensor_docs.py dim 1665-1672 and shape/size 4737-4778", "Dimension count and per-axis size")], ["original-section","variation-code","variation-run","variation-stdout"]),
    claim("construction-and-python-conversion", "software", "torch.tensor builds the 2x3 score table; indexing returns the first row or a selected scalar; shape gives torch.Size([2,3]); tuple(shape) and tolist preserve displayed order; item works only on one element.", "course/first-steps.md:84-98", "Exact fence and documented display/read conversions; conversion operations do not mutate scores. No claim about differentiability or arbitrary serialization.", [evidence("torch-docs", "torch/_torch_docs.py torch.tensor lines 9582-9635", "Sequence-to-Tensor construction"), evidence("index-source", "TensorIndexing.h get_item lines 647-703; applySelect 233-274; applySlicing 550-591", "Integer and tuple-of-integer indexing"), evidence("tensor-docs", "item 2788-2805; size/shape 4737-4778; tolist 5445-5465", "One-element item restriction, shape sizes and ordered nested-list conversion"), evidence("python-tuples", "Doc/library/stdtypes.rst lines 1352-1370", "tuple(iterable) keeps item order"), evidence("original-execution", "stdout.txt lines 1-3", "Original Size([2,3]), first row and 100.0 outputs"), evidence("variation-execution", "variations.py assertions before failure loop and failure_contracts in variations-stdout.txt", "Exact tuple/list/type/nonmutation and multi-element item failure")], ["original-code","original-run","original-stdout","variation-code","variation-run","variation-stdout"], executed("(2,3); [60,70,80]; selected item 100.0; nested lists unchanged; scores.item() rejects six elements", "All expected values asserted; RuntimeError: a Tensor with 6 elements cannot be converted to Scalar", "Original fence executed unchanged; independent assertions additionally checked exact values, Python float type and original tensor equality before/after conversion.")),
    claim("reduction-axes", "concept", "mean(dim=1) reduces the quiz axis and mean(dim=0) reduces students; dim=-1 is the last axis; these calls remove the reduced axis; axis numbers depend on layout.", "course/first-steps.md:100-104", "These calls use keepdim=False, its default. The section does not generalize axis removal to explicit keepdim=True.", [evidence("torch-docs", "torch/_torch_docs.py mean 7195-7261; multi_dim_common 62-85", "Mean by dimensions and omitted keepdim producing one fewer axis"), evidence("tensor-docs", "Tensor.mean lines 3245-3252", "Method shares torch.mean contract and default keepdim=False"), evidence("wrap-dim-source", "WrapDimMinimal.h _maybe_wrap_dim lines 19-31", "Negative dimension wrapping by rank"), evidence("variation-execution", "variations.py negative dim, output shapes and transposed assertions", "-1=1 for this 2D example; exchanging data axes exchanges mean meanings")], ["variation-code","variation-run","variation-stdout","figure-native","figure-width358"]),
    claim("arithmetic-and-exercise", "numeric", "Original per-student means are [70,90] and per-quiz means [70,80,90]; changing only score [0,2] from 80 to 110 gives [80,90] and [70,80,105].", "course/first-steps.md:100,110; referenced figure", "Hand-designed arithmetic examples in score points; denominators are 3 quizzes per student and 2 students per quiz, not an empirical model result.", [evidence("variation-execution", "variations.py Fraction expected_rows/expected_cols and changed assertions; variations-stdout.txt original/changed means and denominators", "Independent exact rational baseline calculation and actual changed-input verification"), evidence("original-execution", "stdout.txt lines 4-5", "Actual unchanged original-code means")], ["original-run","original-stdout","variation-code","variation-run","variation-stdout","figure-native","figure-width358"], executed("(60+70+80)/3=70; (80+90+100)/3=90; columns (60+80)/2=70, (70+90)/2=80, (80+100)/2=90; changed first row=80 and last column=105", "Exact agreement with all original and changed values; no rounding required", "Baseline expected means independently computed using Fraction. Score units remain score points. The changed row denominator is 3 and changed column denominator is 2.", tolerance="Exact equality; all stated integer averages are exactly representable in this float32 example.")),
    claim("three-axis-example", "software", "The author-defined [B,T,D] labels denote batch, text positions and per-position features; [2,4,3] has 24 entries and selecting first axis index 0 leaves [4,3].", "course/first-steps.md:106", "Illustrative regular tensor layout and naming convention; no claim that every text model always uses this axis order or that B,T,D are special Python syntax.", [evidence("index-source", "TensorIndexing.h get_item lines 662-665 and applySelect lines 233-274", "Single integer index selects the first axis"), evidence("tensor-docs", "shape/size lines 4737-4778", "Shape entries are ordered axis lengths"), evidence("variation-execution", "variations.py three_axes and shape/value assertions; variations-stdout.txt shape_after_first_axis_index", "Actual 2x4x3 tensor selection leaves 4x3")], ["variation-code","variation-run","variation-stdout"], executed("arange(24).reshape(2,4,3)[0] has shape (4,3) and first 12 ordered numbers", "Exact shape (4,3), rows [0,1,2] through [9,10,11]", "Bounded integer tensor substitutes for abstract features while checking the exact positional indexing contract. B/T/D meanings are explicit author notation.")),
    claim("reshape-contract", "software", "scores.reshape(6) keeps the six values in order and can be reshaped back to (2,3); reshaping six entries to eight entries is invalid.", "course/first-steps.md:108", "Same logical values and element count, not a guarantee of copy versus view or semantic labels being inferred by reshape.", [evidence("torch-docs", "torch/_torch_docs.py torch.reshape lines 9833-9863", "Same data and number of elements with new shape; view/copy behavior"), evidence("tensor-docs", "Tensor.reshape lines 4163-4179", "Method accepts desired shape tuple or positional integers"), evidence("variation-execution", "variations.py flat and reshape assertions plus failure loop; variations-stdout.txt flat/failure_contracts", "Six-element ordered flatten, inverse shape and invalid eight-element exception")], ["variation-code","variation-run","variation-stdout"], executed("Flat values [60,70,80,80,90,100], restored scores equal original; reshape(8) raises", "Exact flatten and round-trip; RuntimeError: shape '[8]' is invalid for input of size 6", "No parameter updates, gradients or training. Inferred view/copy memory behavior is outside the textbook claim."))
]
report = dict(schema_version=1, review_stage="technical", lesson_id="W.3", source="course/first-steps.md#W.3", source_sha256=sha(BASE / "section.md"), verdict="pass", reviewer_task="/root/phase4_factual_coordinator/factual_w_3", reviewer_context="fresh", author_tasks=[],
    figure_sha256=meta["figure_sha256"], artifacts=artifacts, sources=sources, claims=claims, issues=[],
    frozen_input=dict(path=relative(BASE / "frozen-input/course/first-steps.md"), sha256=sha(BASE / "frozen-input/course/first-steps.md"), meaning="Frozen whole Markdown bytes for this reviewer read; matches extraction whole-input hash; formal source_sha256 is only raw W.3 bytes"),
    actual_read_scope=inspection["actual_read_scope"], independence_record=inspection["candidate_lookup"],
    checks={
        "factual_accuracy": dict(status="pass", details="Personally verified all six substantive claim groups against original official source or actual CPU execution. No unresolved discrepancy.", claim_ids=[c["id"] for c in claims]),
        "numeric_verification": dict(status="pass", details="Independent Fraction calculations and bounded actual altered input confirm all means, units, denominators and exact equality.", claim_ids=["arithmetic-and-exercise"]),
        "figure_consistency": dict(status="pass", details="Exact referenced SVG rendered with Inkscape 1.4 and personally viewed at 640 and 358 pixels. Labels, six scores, two reduction directions and all five means match text and code. Optional Chromium sizing did not complete; no page-layout browser claim.", claim_ids=["reduction-axes","arithmetic-and-exercise"]),
        "source_verification": dict(status="pass", details="Fetched and read original official HTTPS source at installed PyTorch commit; checked v2.14.1 tag equivalence and CPython 3.13.5 tuple docs. Record exact ranges and permanent snapshots, not locator-cache conclusions.", claim_ids=[c["id"] for c in claims]),
        "limitations": dict(status="pass", details="Only structured numeric representation and arithmetic/API demonstrations are supported. Default keepdim=False scope is explicit in review, reshape copy/view behavior is not inferred, B/T/D are illustrative labels, and nothing is called trained ability or model evaluation.", claim_ids=[c["id"] for c in claims])})
save(BASE / "report.json", report)
save(ROOT / "docs/technical-reviews/W.3.json", report)
own = json.loads((ROOT / "docs/technical-reviews/W.3.json").read_bytes())
assert own["reviewer_task"] == "/root/phase4_factual_coordinator/factual_w_3"
assert own["source_sha256"] == hashlib.sha256(section).hexdigest()
assert own == report
assert (BASE / "report.json").read_bytes() == (ROOT / "docs/technical-reviews/W.3.json").read_bytes()
print(json.dumps({"own_canonical_report_written_and_asserted": True, "report_sha256": sha(ROOT / "docs/technical-reviews/W.3.json"), "source_sha256": own["source_sha256"], "figure_sha256": own["figure_sha256"], "claims": len(claims)}, ensure_ascii=False))
