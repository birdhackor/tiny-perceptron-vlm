"""Build only this reviewer's new report; never read the previous canonical report."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PROOF = Path(__file__).resolve().parents[1]
REL = PROOF.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_15_2"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def path(name):
    return REL + "/" + name

cpu_command = "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + path("checks/check_cpu.py")
environment = {"python": "3.13.5", "torch": "2.14.1+cpu", "device": "CPU", "cuda_build": "None", "threads": "1"}
artifacts = []
artifact_by_name = {}
descriptions = {
    "inputs/course/chapters/15.md": "Actual full-chapter frozen input snapshot; no full-chapter or introduction audit claimed.",
    "execution-original/section.md": "Original UTF-8 bytes of only 15.2, including optional details.",
    "inputs/14.6.md": "Necessary shared-parameter prerequisite; contextual reading only.",
    "inputs/docs/course-experiments/results/moe.json": "Complete unaltered original result; only named raw/provenance pointers and key/type inventories inspected.",
    "checks/raw-pointer-selection.json": "Actual named raw configuration, budget, dataset-count/hash and provenance pointer values inspected.",
    "inspection.md": "Reviewer's actual reading scope, original source support limits, CPU and visual observations, and real failures.",
    "render/desktop-set-content.png": "Personally viewed frozen HTML preview at viewport 1280x800; no published-site acceptance claimed.",
    "render/mobile-set-content.png": "Personally viewed frozen HTML preview at viewport 390x844; mobile code scrolls horizontally.",
    "render/render-execution.json": "Actual first stopped CLI attempt and two bounded file:// timeouts, all without screenshot.",
    "render/playwright-results.json": "Successful system-Chromium set_content screenshot dimensions and actual viewports.",
    "primary/download-manifest.json": "HTTPS official-source requests, raw hashes and actual HTTP 403 failures; failed pages are not evidence.",
}
for file in sorted(PROOF.rglob("*")):
    if not file.is_file() or file.is_symlink() or file.name in ("proof-manifest.json", "checker.stdout.txt", "checker.stderr.txt", "checker-execution.json"):
        continue
    name = file.relative_to(PROOF).as_posix()
    identifier = "A" + str(len(artifacts) + 1)
    kind = "code" if file.suffix == ".py" else "figure_render" if file.suffix == ".png" else "source_snapshot"
    item = {"id": identifier, "path": path(name), "sha256": sha(file), "kind": kind,
            "description": descriptions.get(name, "Preserved original bytes or exact inspected excerpt/metadata: " + name + "; actual inspection scope is in inspection.md.")}
    if name == "checks/cpu.stdout.txt":
        item.update(kind="execution", command=cpu_command,
                    result="Exit 0: exact fence, mutation, independent storage, Linear axis, four constructor-only formal parameter counts, and method/provenance equivalence assertions passed.",
                    environment=environment)
    elif name == "execution-original/stdout.txt":
        item.update(kind="execution", command=".venv/bin/python docs/review-tools/section_facts.py course/chapters/15.md#15.2 --output outputs/reviewer-tools/phase4-15_2-independent --execute --timeout 120",
                    result="Exit 0; exact original fence prints three expected outputs, False, 12, 4; CPU helper records no guard events.",
                    environment=environment)
    elif name == "render/playwright.stdout.txt":
        item.update(kind="execution", command=".venv/bin/python " + path("render/render_playwright.py"),
                    result="Exit 0; two set_content screenshots saved and personally viewed; desktop 1280x800/mobile390x844, Chromium151.0.7922.173.",
                    environment={"python": "3.13.5", "chromium": "151.0.7922.173", "device": "CPU", "method": "Playwright set_content"})
    artifacts.append(item)
    artifact_by_name[name] = identifier

def artifact(name):
    return artifact_by_name[name]

def official(identifier, file, title, locator, note):
    return {"id": identifier, "kind": "official_source", "title": title,
            "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/" + locator,
            "version": "PyTorch official tag v2.9.0; installed runtime checked separately as 2.14.1+cpu",
            "authority_reason": "PyTorch maintainers' original tagged implementation and API documentation, fetched directly over HTTPS.",
            "accessed_on": "2026-10-05", "verified": True, "checked_original": True,
            "inspection_note": note + "; preserved original " + path("primary/" + file)}

sources = [
    {"id": "shazeer", "kind": "paper", "title": "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
     "url": "https://arxiv.org/pdf/1701.06538v1", "version": "arXiv:1701.06538v1, 2017-01-23",
     "authority_reason": "Original authors' paper introducing the sparse MoE layer and its computation rule.",
     "accessed_on": "2026-10-05", "verified": True, "checked_original": True,
     "inspection_note": "Personally read first-page version, sections 1.2 and 2 (pp.2-3), Eq.(1), and Appendix C.1 (p.14), including expert input/output shapes, independent parameters, skip-zero gating, one hidden+output layer. Locator index used only to find immutable PDF."},
    {"id": "mixtral", "kind": "paper", "title": "Mixtral of Experts",
     "url": "https://arxiv.org/pdf/2401.04088v1", "version": "arXiv:2401.04088v1, 2024-01-08",
     "authority_reason": "Original model authors' architecture and direct routing-analysis method/results.",
     "accessed_on": "2026-10-05", "verified": True, "checked_original": True,
     "inspection_note": "Personally verified first-page version; read section2.1 pp.2-3 (distinct expert weight sets, sparse evaluation) and section5 p.7 (domain routing investigation with no obvious topic pattern). Supports architecture and need for evidence of specialization, not a universal outcome."},
    official("modulelist", "container-v2.9.0.py", "PyTorch ModuleList original source", "torch/nn/modules/container.py",
             "Read ModuleList lines332-400 and462-493: registered indexed/iterable child modules and construction/extend, no routing or forward"),
    official("parameters", "module-v2.9.0.py", "PyTorch Module parameter iteration original source", "torch/nn/modules/module.py",
             "Read _named_members, parameters and named_parameters lines2636-2711, named_modules2823-2870: recursive registered parameters and default identity deduplication"),
    official("linear", "linear-v2.9.0.py", "PyTorch Linear original source", "torch/nn/modules/linear.py",
             "Read Linear lines53-140: xA.T+b, weight out/in axes, independent Parameter allocation, bias=False, reset_parameters and forward"),
    official("nograd", "grad-mode-v2.9.0.py", "PyTorch no_grad original source", "torch/autograd/grad_mode.py",
             "Read no_grad lines21-85: thread-local reverse-autograd disable and context restoration; parameter factory exception"),
    official("tensorops", "tensor-docs-v2.9.0.py", "PyTorch Tensor operation original API documentation", "torch/_tensor_docs.py",
             "Read copy_1204-1222 (elements copied into self), mul_3413-3420 (in-place multiplication), tolist5513-5533 (nested Python list)"),
    official("tensorfactories", "torch-docs-v2.9.0.py", "PyTorch tensor factory/count original API documentation", "torch/_torch_docs.py",
             "Read eye4158-4186 (diagonal ones), numel8397-8417 (total elements), tensor9223-9276 (copy data, leaf), grouped ordinary API coverage"),
    {"id": "zip", "kind": "official_docs", "title": "Python 3.13 builtins: zip",
     "url": "https://docs.python.org/3.13/library/functions.html#zip", "version": "Official Python 3.13 documentation; resolved to builtins/functions.html",
     "authority_reason": "Python Software Foundation's official language-library documentation.",
     "accessed_on": "2026-10-05", "verified": True, "checked_original": True,
     "inspection_note": "Personally read zip description and examples: i-th elements paired, lazy tuples, default shortest-iterable termination; excerpt saved. Here both iterables have three elements."},
]
for identifier, relative, version, note in [
    ("moecode", "inputs/recorded-revision/tiny_perceptron/modern.py", "48a4f3e912b483d70aee57c42c2aac226534a9a6; full SHA matches raw recorded provenance and current file", "AST-located DenseFFN35-53 and MoEFFN56-84: up/down linear with biases, independent expert ModuleList, router construction and selected-only dispatch."),
    ("modelcode", "inputs/recorded-revision/tiny_perceptron/model.py", "48a4f3e912b483d70aee57c42c2aac226534a9a6; full SHA matches raw recorded provenance and current file", "Read ModelConfig15-28, Block31-50, TinyLM54-86 for width/layers and parameter registration; no trained forward executed."),
    ("experimentcode", "inputs/recorded-revision/scripts/course_experiments/architecture.py", "48a4f3e912b483d70aee57c42c2aac226534a9a6; full SHA matches original report provenance", "AST-selected parameter budget331-346 and run_moe428-477; current _text_dataset74-84, _heldout289-290 and _routing380-427 are byte-equivalent to original methods. Final return author annotations excluded."),
    ("scoringcode", "inputs/recorded-revision/scripts/course_experiments/common.py", "48a4f3e912b483d70aee57c42c2aac226534a9a6; full SHA matches original report provenance", "Read new_lm45-47, text_examples77-97 and evaluate_lm243-304, independently checked method byte-equivalence with current source; overall text evaluation, not per-domain expert specialization."),
    ("rawreport", "inputs/docs/course-experiments/results/moe.json", "experiment revision48a4f3e912b483d70aee57c42c2aac226534a9a6; original CUDA torch2.14.1+cu126/python3.13.3/seed42", "Values inspected only at explicitly saved /revision, runtime provenance, four MoE model/config and budget, dataset count/hash and necessary /code_sha256 pointers. Heldout/routing schema keys/types only. Complete raw JSON retained unchanged; no extra author result interpretations read."),
]:
    sources.append({"id": identifier, "kind": "repository_code", "title": identifier + " original repository input",
                    "path": path(relative), "sha256": sha(PROOF / relative), "version": version,
                    "verified": True, "inspection_note": note})
sources += [
    {"id": "original_execution", "kind": "execution", "title": "Exact original 15.2 fence CPU execution", "verified": True, "artifact_id": artifact("execution-original/stdout.txt")},
    {"id": "independent_execution", "kind": "execution", "title": "Bounded independent CPU variants and constructor-only budget check", "verified": True, "artifact_id": artifact("checks/cpu.stdout.txt")},
    {"id": "arithmetic", "kind": "derivation", "title": "Independent dimension and parameter arithmetic", "verified": True,
     "details": "Linear y=xW.T with W=I,2I,swap gives [1,2],[2,4],[2,1]; each bias-free2x2 table has4 parameters,3 independent tables12 and one shared table4. For width64 hidden256, up/down with biases has64*256+256+256*64+64=33088;2 layers*4 experts*33088=264704;2*64*4 routers=512;264704+512+75392=340608. No rounding or per-token quality denominator is involved."},
]

def ev(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}

claims = [
    {"id": "expert_structure", "kind": "concept", "statement": "MoE experts have separate parameters and matching input/output dimensions; conventional expert FFNs have an expansion/nonlinearity/output transform, while this section intentionally uses one Linear to isolate independent weights.",
     "location": "15.2 paragraphs1-2", "scope": "Architectural terminology and a typical FFN case; not a requirement that every MoE expert has exactly two ungated Linear modules.", "status": "verified",
     "evidence": [ev("shazeer", "section2 pp.3 Eq.(1); Appendix C.1 p.14", "Same input/output shapes, independent parameters, concrete one hidden+output layer expert."), ev("mixtral", "section2.1 pp.2-3", "Contemporary distinct expert weight sets in FFN/SwiGLU architecture."), ev("moecode", "DenseFFN35-53;MoEFFN59-65", "Local two-transform experts constructed independently.")],
     "artifact_ids": [artifact("primary/inspected-shazeer-1701.06538v1.txt.txt"), artifact("primary/inspected-mixtral-2401.04088v1.txt.txt")]},
    {"id": "api_and_independence", "kind": "software", "statement": "The grouped APIs create three registered bias-free Linear modules, pair tables by position, copy known values under no_grad, compute outputs explicitly, and count each distinct registered parameter once; repeating one shared module creates no new independent capacity.",
     "location": "15.2 Python fence and its three explanatory paragraphs", "scope": "Explicit coverage: ModuleList registration/index/iteration and absent forward; Linear constructor/weight axes/bias flag; torch.eye/tensor, zip; no_grad/copy_; Module.parameters and numel; Tensor.tolist; scalar in-place exercise. Counts apply to these trainable weights, not a generic promise that parameters() filters frozen parameters.", "status": "verified",
     "evidence": [ev("modulelist", "ModuleList332-400,462-493", "Registered children; explicit module calls are required."), ev("parameters", "2636-2711,2823-2870", "Recursive parameter iteration with default identity deduplication."), ev("linear", "Linear53-140", "Own Parameter allocations and bias=False input/output axes."), ev("nograd", "no_grad21-85", "Temporarily disables reverse-autograd recording while existing parameters remain parameters."), ev("tensorops", "copy_1204-1222;mul_3413-3420;tolist5513-5533", "Copy elements into existing storage, in-place multiplication, nested list output."), ev("tensorfactories", "eye4158-4186;numel8397-8417;tensor9223-9276", "Identity/table construction and total element counts."), ev("zip", "zip API description and examples", "Positional pairing for equal-length expert/table lists."), ev("independent_execution", "check_cpu.py exact fence and storage/ModuleList assertions", "Installed torch2.14.1+cpu behavior agrees with inspected official contract.")],
     "artifact_ids": [artifact("execution-original/stdout.txt"), artifact("checks/cpu.stdout.txt"), artifact("checks/check_cpu.py")],
     "verification": {"method": "executed", "expected": "3 unique2x2 weights, registered count12; repeated shared count4; ModuleList itself has no forward; copy preserves independent identity/storage.", "observed": "Exact fence and all independent identity, storage, shape, duplicate-parameter and no-forward assertions pass, exit0.", "details": "CPU-only; installed2.14.1+cpu vs official tagged2.9.0 explicitly distinguished. No optimizer or trained model used."}},
    {"id": "toy_numbers", "kind": "numeric", "statement": "The three outputs are [[1,2]], [[2,4]], [[2,1]], separate-weight identity comparison is False, parameter totals are12 and4, and multiplying only expert0 weights by3 changes only its output to[3,6].",
     "location": "15.2 expected outputs/counts and exercise", "scope": "Exact small integer toy arithmetic and manual independence test; not trained specialization or quality.", "status": "verified",
     "evidence": [ev("arithmetic", "2x2 Linear y=xW.T and3*4 versus1*4", "Independent numerical derivation."), ev("original_execution", "stdout four original lines", "Exact fence output."), ev("independent_execution", "Independent modification and asymmetric axis variation", "Exercise output and explicit xW.T=[5,11] orientation check.")],
     "artifact_ids": [artifact("execution-original/stdout.txt"),artifact("checks/cpu.stdout.txt")],
     "verification": {"method": "executed", "expected": "[[[1,2]],[[2,4]],[[2,1]]],False,12,4; exercise [[[3,6]],[[2,4]],[[2,1]]].", "observed": "All equalities asserted and stdout agrees. Asymmetric Linear axis test gives[[5,11]].", "details": "One token shape(1,2),3 experts,each2x2 no-bias table; no denominator beyond element counts.", "tolerance": "Exact equality: small integer-valued operations are exactly representable in float32; parameter counts and object identity exact."}},
    {"id": "sparse_computation", "kind": "concept", "statement": "Executing all three experts and then averaging does not save expert arithmetic; sparse MoE can save expert computations when a router selects a few and unselected experts are not executed.",
     "location": "15.2 paragraph beginning 目前程式把三位都算了一次", "scope": "Comparison with one same-size expert and with selected-only evaluation; not a hardware-speed or wall-clock performance guarantee.", "status": "verified",
     "evidence": [ev("shazeer", "section2 p.3 Eq.(1) and skip-zero paragraph", "Computation savings require not computing experts with zero gates."), ev("mixtral", "section2.1 pp.2-3", "Selected-only expert computation and distinct total versus active weights."), ev("original_execution", "exact fence [e(x) for e in experts]", "Toy explicitly executes every expert and contains no router.")],
     "artifact_ids": [artifact("execution-original/fence-1.py")]},
    {"id": "formal_parameter_counts", "kind": "numeric", "statement": "The recorded formal MoE comparison has2 layers with4 independent FFNs each:8 experts,264704 expert parameters, and340608 total parameters including router/shared parts.",
     "location": "15.2 optional details first sentence", "scope": "Existing experiment's structural counts, verified through named original configuration/budget pointers and CPU constructors only. No weights, training, performance reevaluation or changed result scores.", "status": "verified",
     "evidence": [ev("rawreport", "/results/variants/{top1_aux0,top1_aux0.01,top2_aux0,top2_aux0.01}/{model,budget};/revision;/code_sha256", "All four recorded MoE configs and budgets give2 layers,4 experts,264704/340608 at the preserved original revision."), ev("moecode", "DenseFFN36-43;MoEFFN59-65", "Bias-inclusive two-transform FFNs and independent module construction."), ev("modelcode", "ModelConfig15-28;Block32-41;TinyLM54-66", "Layer count and full-model registered structure."), ev("experimentcode", "original_parameter_budget331-346;run_moe428-477", "Structural count definition and four MoE conditions."), ev("arithmetic", "8*(64*256+256+256*64+64);plus512+75392", "Independent parameter arithmetic."), ev("independent_execution", "four constructor-only budget assertion lines", "Eight unique experts and32 distinct parameter storages, counts exactly match raw originals.")],
     "artifact_ids": [artifact("checks/cpu.stdout.txt"),artifact("checks/raw-pointer-selection.json"),artifact("checks/check_cpu.py")],
     "verification": {"method": "executed", "expected": "8 independent experts,each33088;expert264704;router512;shared75392;total340608 in all4 MoE configurations.", "observed": "All4 configurations exactly match recorded budgets and unique identities/storages; stdout confirms8/264704/340608.", "details": "2 layers*4 experts;64 input/output features,hidden256,up/down biases included. Count of stored parameter elements, not token-based empirical quality.", "tolerance": "Exact integer equality."}},
    {"id": "specialization_limits", "kind": "concept", "statement": "The name expert and the extra parameter count do not establish an innate English/math specialization; specialization requires trained-data/behavior investigation, and the inspected local short report's methods evaluate overall text and routing rather than language/subject expert competence.",
     "location": "15.2 paragraph1 and optional details final three sentences", "scope": "No universal denial of specialization; no local per-domain specialization claim inferred. Evidence limited to inspected raw measurement schema and original-equivalent scoring/routing methods, with author result interpretations excluded.", "status": "verified",
     "evidence": [ev("linear", "Linear102-126 allocation/reset_parameters", "New experts initially receive ordinary initialized weights, not domain labels."), ev("shazeer", "section1.2 pp.2-3", "Joint training and observed syntax/semantic specialization as measured behavior."), ev("mixtral", "section5 p.7", "Domain routing investigated explicitly; model expert labels alone do not establish subject division."), ev("experimentcode", "_routing original-equivalent body;_heldout;run_moe raw assembly", "Aggregated expert dispatch/probabilities and heldout text, no per-language or subject skill testing."), ev("scoringcode", "text_examples77-97;evaluate_lm243-304", "Whole-model text loss and samples; no expert-specific domain competence evaluation."), ev("rawreport", "/results/variants/top2_aux0.01/{heldout,validation_routing} key/type inventories", "Inspected raw measurement schema has validation/test and per-layer routing fields without domain specialization measures.")],
     "artifact_ids": [artifact("checks/cpu.stdout.txt"),artifact("inspection.md")]},
]

section = (PROOF / "execution-original/section.md").read_bytes()
current = (ROOT / "course/chapters/15.md").read_bytes()
start = current.index(b"## 15.2 ")
end = current.index(b"## 15.3 ", start)
assert current[start:end] == section, "Current selected section changed; do not publish stale review."
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "15.2",
    "source": "course/chapters/15.md#15.2", "source_sha256": hashlib.sha256(section).hexdigest(),
    "reviewer_task": TASK, "reviewer_context": "fresh", "author_tasks": [], "verdict": "pass", "figure_sha256": {},
    "frozen_input": {"meaning": "Actual full-chapter snapshot frozen at this inspection, not full-chapter/current-global-version audit.",
                     "path": path("inputs/course/chapters/15.md"), "sha256": sha(PROOF / "inputs/course/chapters/15.md")},
    "actual_scope": {"main": "Entire15.2 including optional details; exact original fence and independent shortCPU checks.",
                     "context": "Chapter15 current lines1-160 and raw14.6 snapshot; no other section or introduction verdict.",
                     "implementation": "AST-first selected original/current-equivalent construction, parameter, data, routing and overall-text scoring methods; final author result annotations excluded.",
                     "raw_pointers": path("checks/raw-pointer-selection.json"),
                     "visual": "No referenced figure; personally viewed own frozenHTML Playwright set_content desktop/mobile snapshots. CLI file:// timeouts preserved; actual published site unverified.",
                     "contamination": "No prior report/conclusion or extra author correction/result interpretation read."},
    "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "Independent experts, matching dimensions, grouped API behavior and selected-only computation agree with original papers, official tagged sources, executed toy and original model methods.", "claim_ids": ["expert_structure", "api_and_independence", "sparse_computation", "specialization_limits"]},
        "numeric_verification": {"status": "pass", "details": "Original outputs/counts and requested independent mutation checked exactly; four recorded structural configs verified by constructors and integer arithmetic with all biases and router/shared parts. No training/model scoring repeated.", "claim_ids": ["toy_numbers", "formal_parameter_counts"]},
        "figure_consistency": {"status": "not_applicable", "details": "15.2 has no figure references; explicit vectors/matrices suffice for this mechanism. Frozen standalone HTML actually rendered and personally viewed at1280x800/390x844 through set_content; first CLI stop and two timeouts retained. Not actual published-site verification; preview rawHTML details link limitation recorded.", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "Personally checked exact original arXiv versions/paragraphs, official PyTorchv2.9.0 source contracts and Python3.13zip; installed2.14.1+cpu separately executed. Original result code provenance matched Git revision bytes; current architecture full-file difference distinguished from byte-identical inspected methods. Indexes only locate originals.", "claim_ids": [c["id"] for c in claims]},
        "limitations": {"status": "pass", "details": "Manually filled one-layer toy demonstrates independent weights, not learning. Formal counts demonstrate stored capacity; inspected local evaluation does not establish per-domain specialization. Sparse arithmetic claim does not promise measured speed. No complete trained model evaluation/engineering work. HTTP403 and file:// timeout limitations preserved with successful original-source/set_content alternatives.", "claim_ids": ["expert_structure", "toy_numbers", "sparse_computation", "formal_parameter_counts", "specialization_limits"]},
    },
}
assert report["reviewer_task"] == TASK
destination = ROOT / "docs/technical-reviews/15.2.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
saved = json.loads(destination.read_bytes())
assert saved["reviewer_task"] == TASK and saved["source_sha256"] == hashlib.sha256(section).hexdigest()
manifest = [{"path": file.relative_to(ROOT).as_posix(), "sha256": sha(file), "bytes": file.stat().st_size}
            for file in sorted(PROOF.rglob("*")) if file.is_file() and not file.is_symlink() and file.name != "proof-manifest.json"]
(PROOF / "proof-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"reviewer_task": TASK, "report": str(destination), "report_sha256": sha(destination),
                  "source_sha256": saved["source_sha256"], "verdict": saved["verdict"],
                  "claims": len(claims), "artifacts": len(artifacts)}, indent=2))
