"""Write this reviewer's own 5.17 report from inspected original evidence."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
PREFIX = BASE.relative_to(REPO).as_posix()
REV = "5c4886908584029761b579af026dcfb627c84070"
ENV = {"python": "3.13.5", "torch": "2.14.1+cpu", "torch_git_version": REV,
       "device": "cpu", "cuda_build": "None", "cuda_available": "False"}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def artifact_id(path):
    return path.replace("/", "--").replace(".", "_")
artifacts = []
execution = {
    "original-run/execution.json": (
        ".venv/bin/python docs/review-tools/section_facts.py course/chapters/05.md#5.17 --output /tmp/phase4-5_17-fresh --execute --timeout 60",
        "Exit 0; both untouched Python fences executed; stdout and environment copied byte-for-byte into original-run."),
    "independent-exercise.stdout.txt": (
        ".venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_17-independent/original-run/fence-2.py",
        "Exit 0 in a fresh process; output.requires_grad True, W and x.grad printed identically, original allclose assertion passed."),
    "contracts.json": (
        ".venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_17-independent/code/check_contracts.py",
        "Exit 0; 24 Linear, 6 Dropout and 3 generator cases plus scoped AD/factory/inference boundary assertions all passed; preserved deprecation warning."),
}
for path in sorted(BASE.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(BASE).as_posix()
    if relative.startswith("checker") or relative.startswith("write-report"):
        continue
    kind = "code" if path.suffix == ".py" else "source_snapshot"
    if relative == "derivation.md":
        kind = "derivation"
    artifact = {"id": artifact_id(relative), "path": PREFIX + "/" + relative,
        "sha256": sha(path), "kind": kind,
        "description": "5.17 fresh independent review evidence: " + relative}
    if relative in execution:
        artifact.update(kind="execution", command=execution[relative][0], result=execution[relative][1], environment=ENV)
    artifacts.append(artifact)

inspections = {
    "docs/source/notes/autograd.md": ("official_docs", "Autograd mechanics: reverse AD, local grad controls and eval", "Personally read lines 12–34, 182–230, 234–320 and 328–348; graph/grad_fn, input/leaf conditions, three grad modes and eval/train orthogonality. Supports only documented AD contracts, not measured training claims."),
    "torch/nn/modules/module.py": ("official_source", "nn.Module mode and parameter APIs", "Personally read parameters() 2670–2697 and train/eval/requires_grad_ 2894–2955; training flag recursion, eval=train(False), iterator and per-parameter flag mutation."),
    "torch/nn/modules/linear.py": ("official_source", "nn.Linear formula, initialization and forward", "Personally read 53–134; y=xA^T+b, input/output/weight/bias axes, Parameter allocation, initialization and forward to F.linear without training flag."),
    "torch/nn/modules/dropout.py": ("official_source", "nn.Dropout module forward contract", "Personally read 35–73; zero mask and 1/(1-p) scale during training, identity evaluation, passes self.training. Regularization effectiveness is outside this section's claim scope."),
    "torch/autograd/grad_mode.py": ("official_source", "no_grad and inference_mode context APIs", "Personally read 22–86 and 213–295; no_grad flag save/restore, factory and forward-AD exceptions; inference extra restrictions and no automatic eval. Nested enable_grad checked by execution, not claimed as course coverage."),
    "torch/nn/parameter.py": ("official_source", "Parameter registration and requires_grad defaults", "Personally read 30–62; module registration, default requires_grad=True, Parameter creation factory exception under no_grad."),
    "torch/nn/functional.py": ("official_source", "Functional dropout and linear primitive wrappers", "Personally read 1473–1499 and 2382–2406; dropout training bool to primitive, F.linear y=xA^T+b and shape contract."),
    "torch/_tensor.py": ("official_source", "Tensor.backward implementation and contract", "Personally read 566–626; chain-rule graph differentiation, gradients accumulate in leaves, scalar default seed, no optimizer operation."),
    "torch/_torch_docs.py": ("official_docs", "torch.allclose, ones and sum original API documentation", "Personally read 835–867, 8837–8865 and 11267–11317; tolerance inequality, ones values/shape/default flag, sum all elements without dim."),
    "torch/_tensor_docs.py": ("official_docs", "Tensor gradient flags and sum original API documentation", "Personally read 4126–4144, 5028–5034 and 6614–6640; in-place requires_grad_, Tensor.sum mapping, requires_grad/grad_fn/leaf .grad distinction."),
}
source_id = {
    "docs/source/notes/autograd.md": "autograd", "torch/nn/modules/module.py": "module",
    "torch/nn/modules/linear.py": "linear", "torch/nn/modules/dropout.py": "dropout",
    "torch/autograd/grad_mode.py": "grad-mode", "torch/nn/parameter.py": "parameter",
    "torch/nn/functional.py": "functional", "torch/_tensor.py": "backward",
    "torch/_torch_docs.py": "torch-api", "torch/_tensor_docs.py": "tensor-api",
}
sources = []
for path, (kind, title, inspection) in inspections.items():
    sources.append({"id": source_id[path], "kind": kind, "title": title,
        "verified": True, "checked_original": True,
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/" + REV + "/" + path,
        "version": "PyTorch installed 2.14.1+cpu, immutable git revision " + REV,
        "accessed_on": "2026-10-05", "authority_reason": "Original documentation/implementation in the PyTorch organization's pytorch repository at the installed git revision; installed Python bytes independently SHA-compared.",
        "inspection_note": inspection,
        "snapshot_artifact_id": artifact_id("sources/" + path.replace("/", "--"))})
sources.extend([
    {"id": "math", "kind": "derivation", "title": "Affine scalar and input-sensitivity derivation", "verified": True,
     "details": "Personally derived y=xW^T+b for shapes (1,2),(1,2),(1); one output.sum() has no denominator. dL/dW=[[1,1]], dL/db=[1], dL/dx=W. Fixed W=[[2,-3]], b=.5 yields y=-.5. See derivation.md; allclose defaults rtol=1e-5,atol=1e-8; deterministic values additionally checked exactly."},
    {"id": "original-execution", "kind": "execution", "title": "Exact two-fence CPU execution", "verified": True, "artifact_id": artifact_id("original-run/execution.json")},
    {"id": "independent-execution", "kind": "execution", "title": "Exact second fence alone in a fresh CPU process", "verified": True, "artifact_id": artifact_id("independent-exercise.stdout.txt")},
    {"id": "boundary-execution", "kind": "execution", "title": "Bounded independent CPU gradient and mode contracts", "verified": True, "artifact_id": artifact_id("contracts.json")},
])
def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}
def verification(expected, observed, details, tolerance=None):
    v = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if tolerance:
        v["tolerance"] = tolerance
    return v
def claim(identifier, kind, statement, location, scope, citations, artifact_paths, verify=None):
    c = {"id": identifier, "kind": kind, "statement": statement, "location": location,
         "scope": scope, "status": "verified", "evidence": citations,
         "artifact_ids": [artifact_id(p) for p in artifact_paths]}
    if verify:
        c["verification"] = verify
    return c
claims = [
    claim("affine-shapes", "numeric", "Linear(2,1) maps a one-row [1,1] to y=w1+w2+b, with weight row [w1,w2] and one bias.", "05.md#5.17 opening and exercise interpretation, source lines 625, 670",
        "One sample, two dimensionless input features, one scalar output; W axes are out_features × in_features. No averaging denominator.",
        [evidence("linear", "Linear doc/constructor/forward, lines 53–134", "Affine formula and (1,2) weight, (1) bias, (1,1) output shapes."), evidence("functional", "F.linear doc, lines 2382–2406", "y=xA^T+b and axes."), evidence("math", "derivation.md paragraphs 2–4", "Expansion and exact y=-0.5 for W=[2,-3],b=.5,x=[1,1].")],
        ["derivation.md", "contracts.json"], verification("x=(1,2),W=(1,2),b=(1),y=(1,1); fixed example y=-.5", "All 24 cases y=[[-.5]], shape [1,1]", "Each case freshly creates CPU float32 layer with fixed W,b and no updates.", "Exact equality for selected binary-exact values; original random printed values are not fixed expected answers.")),
    claim("default-flags-ones", "software", "torch.ones(1,2) creates the two ones; Linear weight and bias default to requires_grad=True, while that ordinary input defaults to False.", "05.md#5.17 first fence and following explanation, source lines 633–645",
        "Fresh ordinary CPU tensors and nn.Parameter defaults; not an assertion that every tensor has gradients.",
        [evidence("torch-api", "torch.ones doc, lines 8837–8865", "Filled ones, shape arguments and requires_grad=False default."), evidence("parameter", "Parameter doc/__new__, lines 30–57", "Module parameter registration and requires_grad=True default."), evidence("linear", "Linear.__init__, lines 106–114", "Weight and bias are Parameter instances.")],
        ["original-run/fence-1.py", "original-run/execution.json", "contracts.json"], verification("Fresh parameter flags True, input flag False; new Linear output has graph in grad mode", "First fence prints True and AddmmBackward0; matrix checks parameter/input flags are retained across modes", "Exact original fence plus fresh 24-case instrumented checks, CPU; no optimizer.")),
    claim("eval-train-orthogonal", "concept", "eval controls module forward behavior and does not disable differentiation; eval is train(False), and train/no_grad are independent controls.", "05.md#5.17 opening, second paragraph and mode comparison, source lines 625–649",
        "nn.Module built-in mode contracts; effects depend on the specific forward implementation. Changing mode does not force every module's numeric output to change.",
        [evidence("autograd", "Evaluation Mode, lines 328–348", "eval/train are orthogonal to no_grad and inference; modules define mode-specific behavior."), evidence("module", "train/eval, lines 2894–2932", "Sets training flag recursively, eval delegates to train(False), with no grad-context change.")],
        ["contracts.json", "original-run/execution.json"]),
    claim("first-fence-backward", "software", "In the first fence output.requires_grad is True with a grad_fn, and output.sum().backward() propagates gradients to w1,w2,b after eval.", "05.md#5.17 first fence and grad_fn/backward explanation, source lines 635–645",
        "grad_fn marks the backward graph entry, not a name the reader must memorize. backward computes/accumulates derivatives; this code does not update parameters. x requires_grad is False here.",
        [evidence("autograd", "How autograd encodes history, lines 12–34; leaf accumulation 200–205", "Graph and grad_fn role, chain-rule propagation to eligible leaves."), evidence("backward", "Tensor.backward, lines 566–626", "Scalar seed and leaf gradient accumulation."), evidence("torch-api", "torch.sum, lines 11267–11275", "No dim sums all elements."), evidence("tensor-api", "Tensor.sum 5028–5034 and requires_grad/is_leaf 6614–6640", "Tensor.sum mapping and .grad scope."), evidence("boundary-execution", "contracts.json matrix training=false,default,param=true,input=false", "weight.grad=[[1,1]],bias.grad=[1],input.grad=None; parameters unchanged.")],
        ["original-run/execution.json", "original-run/stdout.txt", "contracts.json"], verification("True/non-null grad_fn; dL/dW=[[1,1]],dL/db=[1], no parameter update", "Original output True/AddmmBackward0 and backward succeeds; corresponding matrix gradients exact, parameters unchanged", "One-element output.sum scalar, eligible leaf gradients checked. No training quality claim.")),
    claim("no-grad-forward", "software", "The first fence's new Linear output under no_grad is requires_grad=False and grad_fn=None, and training mode also permits forwards with no new backward graph.", "05.md#5.17 second paragraph, first fence with block and follow-up, source lines 627, 639–647",
        "New Linear operations in default reverse-mode AD, with no nested re-enable. no_grad does not freeze input flags, clear previous .grad, set eval, disable forward AD, or alter a pre-existing tensor returned as identity.",
        [evidence("grad-mode", "no_grad doc and enter/exit, lines 22–86", "No new graph, factory/forward-AD exceptions and flag restoration."), evidence("autograd", "No-grad Mode, lines 277–297", "New operations excluded while tensors can be used later in grad mode."), evidence("boundary-execution", "contracts.json matrix no_grad rows; dropout; factory_exception; forward_ad", "Both train/eval Linear outputs false/None; context leaves module mode intact; scoped exceptions verified.")],
        ["original-run/execution.json", "original-run/stdout.txt", "contracts.json"], verification("other False/None; same result for newly computed train-mode Linear output under no_grad", "Original prints False None; all 8 no_grad matrix cases have false/None and no leaf gradients", "Scope limited to newly computed operations; boundary probes retain identity/factory/forward-AD behavior without misdescribing this example.")),
    claim("linear-dropout-modes", "concept", "Linear has no dropout or mode-specific numeric forward change; adding ordinary Dropout changes train/eval forward behavior while leaving grad controls independent.", "05.md#5.17 first/second paragraphs and Linear selection explanation, source lines 625–649",
        "Ordinary nn.Linear and nn.Dropout implementations; eval Dropout is identity, train masks/scales. No promise that each random train output differs, no BatchNorm derivation or regularization-quality claim.",
        [evidence("linear", "Linear.forward, lines 130–134", "Only F.linear(input,weight,bias), no self.training branch."), evidence("dropout", "Dropout doc/forward, lines 35–73", "Training random zeros and scaling; evaluation identity; self.training passed."), evidence("functional", "dropout wrapper, lines 1473–1499", "training controls primitive application."), evidence("autograd", "Evaluation Mode, lines 328–345", "Mode independence from AD contexts.")],
        ["contracts.json"]),
    claim("freeze-parameters", "software", "Iterating layer.parameters() and parameter.requires_grad_(False) freezes both Linear weight and bias as AD leaves while retaining a graph when an input requires gradients.", "05.md#5.17 third setting and exercise loop, source lines 649, 658–664",
        "Fresh parameter freezing excludes new parameter .grad accumulation; with no required input no graph remains. It does not clear stale .grad or prohibit manual value changes.",
        [evidence("module", "parameters, lines 2670–2697; requires_grad_,2934–2955", "Iterator includes registered weight/bias; module variant has same loop contract."), evidence("tensor-api", "Tensor.requires_grad_, lines 4126–4144", "In-place tensor flag change."), evidence("autograd", "Setting requires_grad, lines 194–230", "An operation records if at least one input requires grad; leaf gradients accumulate only for required leaves.")],
        ["original-run/fence-2.py", "original-run/execution.json", "contracts.json"], verification("Both parameter flags False; x=True gives recorded output and input gradient, parameter grads None", "Exact exercise succeeds; default,frozen,input=True matrix rows show x.grad=[2,-3],weight/bias grads None; frozen,input=False has no graph", "Fresh leaf conditions checked across all modes; separate stale-gradient probe confirms no implied .grad clearing.")),
    claim("frozen-input-derivative", "numeric", "For the independent frozen Linear exercise, backward gives x.grad with the same two entries as the fixed weight row w1,w2.", "05.md#5.17 exercise prediction, fence and explanation, source lines 651–671",
        "Single sample and one output, differentiated loss output.sum; x is a fresh leaf with requires_grad=True and forward occurs outside no_grad/inference. Bias contributes no derivative to x.",
        [evidence("linear", "Linear formula/shapes, lines 53–79", "Each output's coefficient for input coordinate is the matching W entry."), evidence("math", "derivation.md paragraphs 2–6", "d(output.sum)/dx=W with no averaging denominator and input/output axis match."), evidence("independent-execution", "independent-exercise.stdout.txt", "Printed W and x.grad match the same run and original assertion succeeds."), evidence("boundary-execution", "contracts.json default,frozen,input=True cases", "Exact fixed [2,-3] sensitivity verified independently.")],
        ["derivation.md", "independent-exercise.stdout.txt", "original-run/execution.json", "contracts.json"], verification("x.grad=W with shape (1,2), original exercise output flag True", "Independent untouched exercise prints W=x.grad=[[.6714,-.6219]] at display precision; matrix gives exact [[2,-3]]", "Different initialization values between invocations are expected; compare tensors within the same invocation, not four-decimal strings", "Original torch.allclose defaults atol=1e-8,rtol=1e-5; deterministic matrix torch.equal exact.")),
    claim("independent-exercise-allclose", "software", "The replacement exercise is a complete independent program and its final torch.allclose assertion checks the input-sensitivity relationship within that invocation.", "05.md#5.17 exercise replacement instruction and final assertion, source lines 651–669",
        "Untouched second fence executed alone in a new interpreter; no other variable or old no_grad comparison/first assertion dependency. allclose tests numerical closeness, not a universal exact-bit equality claim.",
        [evidence("torch-api", "torch.allclose, lines 835–867", "Elementwise |a-b|<=atol+rtol*|b| with defaults."), evidence("independent-execution", "bounded-run-receipts.json independent-exercise; independent-exercise.stdout.txt", "Exact original fence SHA 5bd51b67b5fb7d6f6b00770e438f508ca84268abe659b1f5f856b78f87392f8b run separately with -I, exit 0.")],
        ["original-run/fence-2.py", "independent-exercise.stdout.txt", "bounded-run-receipts.json"], verification("Exercise needs only its imports and fresh layer/input; final allclose succeeds", "New-process execution exit 0, stdout True and corresponding coefficients/gradients; empty stderr", "No bootstrap, original-fence-one state, old other or enclosing no_grad context was used.")),
    claim("fixed-reference-upstream", "concept", "Using eval, parameter freezing and no_grad can serve distinct purposes for a fixed reference, but its differentiable forward must retain the graph to train an upstream input generator through it.", "05.md#5.17 third-setting paragraph, source line 649",
        "Conditional common fixed-reference pattern when that reference branch needs no upstream derivative. For differentiable paths through a frozen reference, freezing preserves input gradients while no_grad/inference remove the new path. Not every frozen layer is differentiable, nor is inference_mode taught in this section.",
        [evidence("autograd", "Reverse AD lines 12–34; requires_grad194–230; no_grad277–297; eval328–348", "Chain rule, input eligibility and independent functions of the three controls."), evidence("grad-mode", "inference_mode, lines 213–245", "Additional checked boundary: no recorded computation and no automatic eval."), evidence("boundary-execution", "contracts.json generator default/no_grad/inference", "Generator gradient [[2,2],[-3,-3]] only in grad mode; reference grads always None; blocked branches cannot backward from y.")],
        ["derivation.md", "contracts.json"]),
]
snapshot = json.loads((BASE / "original-run/extraction.json").read_text())
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "5.17",
    "source": "course/chapters/05.md#5.17", "source_sha256": snapshot["source_sha256"],
    "figure_sha256": {}, "reviewer_task": "/root/phase4_factual_coordinator/factual_5_17",
    "reviewer_context": "fresh", "verdict": "pass", "reviewed_on": "2026-10-05",
    "actual_read_scope": "Current 5.17 raw source and both original fences, actual bootstrap and four requested instructions/schema files; ten newly fetched original official PyTorch files at installed commit, relevant ranges in inspection.md. No prior report/verdict read; no chapter introduction needed.",
    "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
    "coordinator_questions": [
        {"original_question": "新增獨立 frozen-parameter/input-gradient 練習尚未執行，請親自執行並驗證其條件/輸出", "resolution": "Untouched second fence executed in helper and independently with -I in a fresh CPU process, both exit 0; flags, weight/bias freezing and x derivative separately checked in bounded matrix.", "evidence": ["original-run/execution.json", "independent-exercise.stdout.txt", "contracts.json"]},
        {"original_question": "eval/train/no_grad/inference與requires_grad的各種契約及梯度範圍是否一致？", "resolution": "Official installed-revision contracts personally inspected; 24 Linear/6 Dropout/3 generator probes all consistent with scoped text. Preserve factory/identity/nested context/forward AD/inference and stale-grad boundaries; text does not overclaim those boundaries.", "evidence": ["inspection.md", "contracts.json", "sources/fetch-receipts.json"]},
    ],
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "10 material claims individually cover all substantive APIs, affine axes, graph/leaf scopes, mode controls and fixed-reference chain rule; no unresolved contradiction found.", "claim_ids": [c["id"] for c in claims]},
        "numeric_verification": {"status": "pass", "details": "Personally derived scalar affine and gradients with explicit axes and no denominator; bounded exact float32 W=[2,-3],b=.5 yields y=-.5 and matching input gradient. Random original exercise compares same-run tensors with documented tolerance.", "claim_ids": ["affine-shapes", "frozen-input-derivative"]},
        "figure_consistency": {"status": "not_applicable", "details": "No figures referenced. Scalar formula, explicit feature order and weight row suffice for this section; no visual rendering claimed.", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "Fetched and personally read original official docs/code at installed git commit; HTTPS receipts with hashes and UTC dates, and nine installed files SHA-identical. No search summaries or old review conclusions used.", "claim_ids": [c["id"] for c in claims]},
        "limitations": {"status": "pass", "details": "Only short CPU AD contracts, no optimizer/training/data/model work or speed benchmark. no_grad statement limited to newly computed Linear reverse-AD path; factories, identity returns, nested enable_grad, forward AD and inference restrictions checked. Fresh freezing does not imply clearing stale gradients or immutable storage. Inference probes are audit boundaries rather than new course claims; no existing measured result cited.", "claim_ids": ["no-grad-forward", "freeze-parameters", "fixed-reference-upstream"]},
    },
}
target = REPO / "docs/technical-reviews/5.17.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": str(target), "sha256": sha(target), "source_sha256": snapshot["source_sha256"], "claims": len(claims), "artifacts": len(artifacts), "verdict": report["verdict"]}))
