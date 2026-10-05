"""Generate this reviewer's complete new 13.4 report without reading an old report."""
import hashlib
import json
import re
from pathlib import Path

A = Path(__file__).resolve().parent
ROOT = A.parents[3]
PREFIX = A.relative_to(ROOT).as_posix()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def artifact(identifier, filename, kind, description, **extra):
    return {"id": identifier, "path": PREFIX + "/" + filename,
            "sha256": sha(A / filename), "kind": kind, "description": description, **extra}

environment = json.loads((A / "environment.json").read_text())
receipt = json.loads((A / "execution.json").read_text())
assert receipt["exit_code"] == 0
assert receipt["code_sha256"] == sha(A / "review_checks.py")
assert receipt["stdout_sha256"] == sha(A / "checks.stdout.txt")
assert receipt["stderr_sha256"] == sha(A / "checks.stderr.txt")
manifest = json.loads((A / "inputs/manifest.json").read_text())
chapter = (ROOT / "course/chapters/13.md").read_bytes()
match = re.search(rb"(?m)^## 13\.4 [^\r\n]+", chapter)
end = re.search(rb"(?m)^## ", chapter[match.end():])
body = chapter[match.start():match.end() + end.start()] if end else chapter[match.start():]
assert sha(A / "inputs/section.md") == hashlib.sha256(body).hexdigest() == manifest["source_sha256"]

artifacts = [
    artifact("section", "inputs/section.md", "source_snapshot", "Entire personally read original UTF-8 section, lines 102–137; no newline normalization."),
    artifact("frozen_chapter", "inputs/frozen-chapter-13.md", "source_snapshot", "Full chapter frozen input captured at first reading; this is a frozen snapshot fingerprint, not a declaration of current whole-chapter state."),
    artifact("input_manifest", "inputs/manifest.json", "source_snapshot", "Section byte hash, frozen chapter snapshot, original fence hash and source line ranges."),
    artifact("original_fence", "inputs/fence-1.python", "code", "Original section fence executed unchanged."),
    artifact("checks_code", "review_checks.py", "code", "Personal bounded checks; raw JSON pointer coverage, code-hash checks, arithmetic and original/alias/eval variants."),
    artifact("checks_stdout", "checks.stdout.txt", "execution", "Actual successful CPU stdout, including every assertion outcome and raw measurement recomputation.", command=receipt["command"], result="Exit 0; all assertions completed; no CUDA, checkpoint load, training or GPU run.", environment=environment),
    artifact("execution_receipt", "execution.json", "execution", "Actual argv, cwd, time, timeout, exit status and hashes.", command=receipt["command"], result="Exit 0 in " + str(receipt["elapsed_seconds"]) + " seconds.", environment=environment),
    artifact("checks_stderr", "checks.stderr.txt", "source_snapshot", "Actual successful execution stderr; empty."),
    artifact("environment", "environment.json", "source_snapshot", "Python/PyTorch version, exact PyTorch git revision, CPU device and seed."),
    artifact("measurements", "measurements.json", "derivation", "Personally selected raw measurement pointers and independently recalculated values; no author notes or scope summaries."),
    artifact("support", "personal-support.md", "derivation", "Personal source locators, read ranges, reasoning, provenance, tooling corrections and limits."),
    artifact("contract_comparison", "contract-comparison.json", "derivation", "Actual AST equality checks between personally read current operational functions and the immutable result-matched original functions, with both line ranges."),
    artifact("report_writer", "write_report.py", "code", "This reviewer's complete report generator; never reads an old canonical report."),
    artifact("alias_code", "alias-fence.py", "code", "Original exercise variation with only deepcopy(policy) changed to policy."),
    artifact("original_stdout", "original-fence.stdout.txt", "source_snapshot", "Actual original fence output True/False."),
    artifact("alias_stdout", "alias-fence.stdout.txt", "source_snapshot", "Actual alias variation output False/False."),
]
snapshot_items = [
    ("paper_pdf", "original/dpo-2305.18290v3.pdf", "Original authors' arXiv v3 PDF; version personally confirmed on the first page."),
    ("paper_text", "original/dpo-2305.18290v3.txt", "Personally read pdftotext extraction: Eq. (7), §4 outline and Appendix B."),
    ("copy_source", "original/cpython-copy-v3.13.5.py", "Personally fetched official CPython v3.13.5 copy implementation."),
    ("module_source", "original/pytorch-module.py", "Original immutable PyTorch module API source; eval/train/requires_grad personally read."),
    ("autograd_source", "original/pytorch-autograd.md", "Original immutable PyTorch autograd documentation personally read."),
    ("official_locators", "original/official-locators.json", "HTTPS versions and original cached-source hashes, personally rechecked rather than relying on old inspections."),
    ("dpo_raw", "original/docs/course-experiments/results/dpo.json", "Complete untouched original DPO raw result; only named measurement/provenance pointers inspected."),
    ("style_raw", "original/docs/course-experiments/results/style.json", "Complete untouched original style/content raw result; only named measurement/provenance pointers inspected."),
    ("model_code", "original/tiny_perceptron/model.py", "Snapshot of actual TinyLM source read and executed; matches the original DPO result's model code hash."),
    ("alignment_code", "original/tiny_perceptron/alignment.py", "Actual sequence-logprob and DPO-loss source read and executed; matches the original DPO result's hash."),
    ("data_code", "original/tiny_perceptron/data.py", "ByteTokenizer/EOS contract read; matches original result's code hash."),
    ("dpo_methods", "original/git-8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d-behavior.py", "Original DPO immutable revision; result /code_sha256 matched; operational DPO methods inspected."),
    ("common_methods", "original/git-8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d-common.py", "Original result-matched immutable common.py; optimizer, split and exact evaluation contracts."),
    ("text_methods", "original/git-8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d-text.py", "Original result-matched immutable text.py; arithmetic data construction and JSONL hash contract."),
    ("style_methods", "original/git-ae7bbbf95537d228a44810041d2a9e978360d369-behavior.py", "Original style result-matched immutable run_style content SFT contract."),
]
artifacts.extend(artifact(i, p, "source_snapshot", d) for i, p, d in snapshot_items)
for run in ["initial-run", "second-run", "third-run"]:
    for filename, kind in [("review_checks.py", "code"), ("execution.json", "source_snapshot"), ("checks.stdout.txt", "source_snapshot"), ("checks.stderr.txt", "source_snapshot"), ("environment.json", "source_snapshot")]:
        artifacts.append(artifact(run.replace("-", "_") + "_" + filename.replace(".", "_"), run + "/" + filename, kind, "Preserved actual " + run + " version; first two runs failed on reviewer instrumentation assumptions, third passed. See personal-support.md."))

def authority(identifier, kind, title, url, version, reason, note):
    return {"id": identifier, "kind": kind, "title": title, "url": url,
            "version": version, "authority_reason": reason, "accessed_on": "2026-10-05",
            "verified": True, "checked_original": True, "inspection_note": note}

sources = [
    authority("dpo_paper", "paper", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "https://arxiv.org/pdf/2305.18290v3", "arXiv:2305.18290v3, 29 July 2024, identified on original PDF first page", "Authors' original DPO paper.", "Personally inspected original PDF Eq. (7), §4 gradient/outline and Appendix B loss code and hyperparameters; copied named cached original and rechecked version/hash. No old reviewer interpretation used."),
    authority("cpython_copy", "official_source", "CPython copy.py deepcopy", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Lib/copy.py", "CPython v3.13.5", "Official Python implementation, version matching CPU Python 3.13.5.", "Personally fetched original source over HTTPS, read lines 1–45 and deepcopy 119–164; recursive copying with type-specific hooks. TinyLM storage independence verified separately by actual execution."),
    authority("torch_module", "official_source", "PyTorch Module.eval/train/requires_grad_", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/module.py", "Immutable PyTorch 5c4886908584029761b579af026dcfb627c84070; installed torch 2.14.1+cpu same git revision", "Original upstream PyTorch API implementation.", "Personally read lines 2894–2966 in the original cached bytes after checking source hash; train(False) eval mode and parameter-wise requires_grad flags."),
    authority("torch_autograd", "official_docs", "PyTorch Autograd mechanics: freezing and evaluation mode", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/autograd.md", "Immutable PyTorch 5c4886908584029761b579af026dcfb627c84070; installed torch 2.14.1+cpu same git revision", "Original upstream PyTorch autograd documentation.", "Personally read lines 184–230 and 331–347; requires_grad graph-recording condition, frozen parameters and eval orthogonality."),
    {"id": "cpu_checks", "kind": "execution", "title": "Personally executed original fence, variants and raw-evidence computations", "verified": True, "artifact_id": "checks_stdout"},
    {"id": "gap_derivation", "kind": "derivation", "title": "Difference-of-differences and probability ordering", "verified": True, "details": "Reference gap -4-(-3)=-1; policy gap -3-(-3)=0; relative gap 0-(-1)=1. Initial identical policy/reference cancels to zero regardless of each absolute ranking. Natural exp is strictly increasing: equal log scores tie; a negative chosen-minus-rejected score implies chosen probability is lower. Raw sums -13.649187088012695-(-5.2012939453125)=-8.447893142700195; subtract -10.893460392951965 gives 2.44556725025177."},
]
repo_specs = [
    ("tiny_model", "model_code", "Actual TinyLM model", "current CPU-read snapshot, same hash as original DPO /code_sha256", "ModelConfig/TinyLM/Block lines 1–89; integer token embedding, independent parameters and logits output contract."),
    ("alignment", "alignment_code", "Actual sequence_log_probability and dpo_loss", "result-matched original/current snapshot", "Personally read lines 29–42: gather label log-softmax, ignore -100, sum tokens; subtract detached reference gap, require beta>0 and -logsigmoid(beta*gap)."),
    ("token_contract", "data_code", "ByteTokenizer and EOS contract", "result-matched original/current snapshot", "Personally read lines 1–28: UTF-8 bytes plus eight and EOS ID 2; used to recompute raw exact-match flags."),
    ("dpo_contract", "dpo_methods", "Original DPO run and scoring implementation", "git 8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d; complete hash equals result /code_sha256", "AST located operational methods; run_dpo lines 697–718 loads style/content.pt, clones base for both beta settings and checks reference hashes; _preference_evaluate lines 609–641 defines raw gaps. _dpo_train policy optimizer contract read in current AST-matched operational methods; unrelated scope/pilot explanations excluded."),
    ("common_contract", "common_methods", "Original split/optimizer/exact evaluation implementation", "git 8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d; complete hash equals result /code_sha256", "Personally read split_records 50–70, records_sha256 73–74, fit 205–239 and evaluate_lm 243–304. Optimizer receives only trainable policy parameters. Raw token exact criterion and sample messages exclude the answer."),
    ("text_contract", "text_methods", "Original arithmetic and saved split provenance", "git 8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d; complete hash equals result /code_sha256", "Personally read _json_bytes/_digest 39–44, _save_splits 47–62 and arithmetic_records; executed only pure arithmetic/split reconstruction, checking exact JSONL SHA."),
    ("style_contract", "style_methods", "Original content.pt SFT construction", "git ae7bbbf95537d228a44810041d2a9e978360d369; complete hash equals original style /code_sha256", "Personally read original run_style 95–101: arithmetic split, new width64/layers2 content, mode=sft, 1000-step content checkpoint; read context without unrelated return scope explanations."),
    ("dpo_measurements", "dpo_raw", "Untouched original DPO raw measurements", "original GPU result revision 8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d; SHA f298491097b9dcd7f62b2619804e744b8ad660a5831f5bfccbf5f2eedc947735", "Top-level/nested keys/types first. Then only named measurement/provenance pointers in measurements.json: /revision, /code_sha256, /results/data, /results/before/{validation,test}/samples, /results/runs/{model,beta1}/beta,/training/steps,/preference/{validation,test}/samples, /results/reference_sha256,/results/reference_unchanged. Extra author scope/note summaries not read."),
    ("style_measurements", "style_raw", "Untouched original content baseline measurements", "original GPU result revision ae7bbbf95537d228a44810041d2a9e978360d369; SHA b534bf8062231c452c01b259ff3419df53e1da771ec19c339549ed736f0f4384", "Top-level/nested keys/types first. Then named /revision, /code_sha256, /results/arithmetic_data/{train,validation,test}, /results/content_evaluation/test/{records,examples,matches,exact_match,samples}, /results/content_checkpoint and /results/content_training/checkpoint. Raw token criteria independently recomputed; no weights loaded."),
]
artifact_map = {item["id"]: item for item in artifacts}
for identifier, artifact_id, title, version, note in repo_specs:
    item = artifact_map[artifact_id]
    sources.append({"id": identifier, "kind": "repository_code", "title": title,
                    "verified": True, "path": item["path"], "sha256": item["sha256"],
                    "version": version, "inspection_note": note})

def ev(source_id, locator, supports):
    return {"source_id": source_id, "locator": locator, "supports": supports}

def verification(expected, observed, details, **extra):
    return {"method": "executed", "expected": expected, "observed": observed, "details": details, **extra}

def claim(identifier, kind, statement, location, scope, evidence, artifact_ids, check=None):
    item = {"id": identifier, "kind": kind, "statement": statement, "location": location,
            "scope": scope, "status": "verified", "evidence": evidence, "artifact_ids": artifact_ids}
    if check is not None: item["verification"] = check
    return item

claims = [
    claim("manual_gap", "numeric", "Reference chosen/rejected logs -4/-3 yield gap -1; policy -3/-3 yields gap 0, improvement 1 and equal candidate probability.", "course/chapters/13.md:104", "Hand-selected same-prompt log-score arithmetic; no generated performance claim.", [ev("gap_derivation", "Explicit -4-(-3), -3-(-3), difference and exp monotonicity", "Signs, units (natural log), same-answer comparison and ties."), ev("cpu_checks", "checks.stdout.txt MANUAL_GAPS", "Independent executable recomputation.")], ["checks_stdout", "checks_code", "support"], verification("(-1,0,1), policy tie", "(-1,0,1), exp(-3)==exp(-3)", "Exact integer subtraction; probabilities restored from logs only for ordering.", tolerance="Exact gap equality; no approximation required.")),
    claim("policy_reference", "concept", "DPO updates the policy against a fixed reference and measures change in the chosen-minus-rejected log-score gap.", "course/chapters/13.md:106,125", "Standard original DPO policy/reference objective; does not declare a ground-truth scorer.", [ev("dpo_paper", "Eq.(7), §4 DPO outline, Appendix B pi_logratios/ref_logratios", "Optimized pi_theta, given reference pi_ref and difference of ratios."), ev("dpo_contract", "Original run_dpo 697–718 and _dpo_train", "Distinct fixed reference and two copied policies; reference never supplied to fit's policy optimizer."), ev("common_contract", "fit 205–239", "Optimizer holds trainable policy parameters only.")], ["paper_pdf", "paper_text", "dpo_methods", "common_methods", "support"]),
    claim("reference_sft", "concept", "The reference is commonly the SFT starting policy; it serves as behavior baseline rather than truth authority.", "course/chapters/13.md:125,132", "Original DPO convention whenever SFT is available; toy baseline counterexample is handled separately.", [ev("dpo_paper", "§4 DPO outline: initialize pi_ref=pi_SFT whenever available; Eq.(7)", "SFT reference convention; preference objective is based on labels, without a correctness-oracle condition."), ev("style_contract", "Original run_style 95–101", "This experiment's content.pt comes from arithmetic SFT."), ev("dpo_contract", "Original run_dpo 700–701", "Experiment uses that content checkpoint as fixed reference.")], ["paper_pdf", "style_methods", "dpo_methods", "support"]),
    claim("copy_freeze_fence", "software", "Original TinyLM fence makes an independent frozen eval reference; editing policy adds .1 without changing reference, and reference logits require no grad.", "course/chapters/13.md:109–123", "Actual integer-input random CPU TinyLM example. Eval changes mode; freezing disables parameter gradients; no preference training occurs.", [ev("cpython_copy", "copy.py 21–22, deepcopy 119–164", "Generic recursive deep-copy contract."), ev("torch_module", "Module.train/eval/requires_grad_ 2894–2966", "Eval mode and frozen parameter flags are separate API operations."), ev("torch_autograd", "184–230,331–347", "Frozen parameters plus integer token inputs give no autograd graph; eval alone does not do that."), ev("tiny_model", "TinyLM 53–86", "Embedding/forward/logits actual contract."), ev("cpu_checks", "ORIGINAL_FENCE and EVAL_ONLY stdout", "Unchanged original fence True/False, different data_ptr, frozen reference/unfrozen policy, no grads; eval-only output requires grad.")], ["original_fence", "original_stdout", "checks_stdout", "checks_code", "module_source", "autograd_source"], verification("Original outputs True/False; independent storage and no training", "True/False; distinct embedding data_ptr; reference params=False, policy=True; no parameter grad; eval-only autograd=True", "Original code executed unchanged under CPU .venv; no backward/optimizer step.")),
    claim("alias_exercise", "software", "Replacing deepcopy(policy) by policy makes names aliases; freezing also freezes policy, and manual .1 edit changes the reference so first result is False.", "course/chapters/13.md:127", "Exact stated exercise in this direct mutable TinyLM-reference implementation; copying and freezing have separate roles.", [ev("torch_module", "requires_grad_ mutates all parameters in place, 2934–2960", "Freezing the same module freezes the aliased policy too."), ev("cpu_checks", "ALIAS_EXERCISE stdout and alias-fence.py", "Policy is reference; all policy parameters frozen; manual no_grad add changes shared tensor; outputs False/False.")], ["alias_code", "alias_stdout", "checks_stdout", "checks_code"], verification("Policy is reference; policy frozen; first print False", "Same object, all policy flags False; output False/False", "Only deepcopy(policy) was replaced by policy; no inference using existing weights or model training.")),
    claim("initial_cancellation", "numeric", "Initially identical policy and reference have relative gap zero even when chosen is less probable.", "course/chapters/13.md:125", "Same parameter states and deterministic answer-score evaluation; relative is a difference of gaps, not absolute preferred-answer victory.", [ev("dpo_paper", "Eq.(7) and Appendix B", "Subtracting identical log ratios gives zero."), ev("gap_derivation", "Initial cancellation calculation", "Zero regardless of sign of original gap."), ev("cpu_checks", "DPO_MARGIN_FORMULA and ALL_RECORDED_REFERENCE_MARGINS", "Loss log(2) at initial margin0; all 15 raw initial held-out relative gaps are exactly0.")], ["checks_stdout", "checks_code", "measurements"], verification("Relative gap=0 for identical models, loss log(2)", "Gap0 and loss0.6931471805599453; all15 initial recorded relative gaps0", "Independent float64 DPO-loss call and checks of raw initial samples; no updates run.", tolerance="Initial raw gaps exact0; loss absolute tolerance 1e-14.")),
    claim("beta_scale", "concept", "Positive beta scales the relative gap inside the DPO log-sigmoid; it is distinct from learning rate.", "course/chapters/13.md:132", "Loss-setting description only; no monotonic learning-speed or better-beta performance guarantee.", [ev("dpo_paper", "Eq.(7), Appendix B code and hyperparameters", "beta times relative logratio; separate optimizer learning rate specified."), ev("alignment", "dpo_loss 38–42", "Enforces beta>0 and multiplies relative_margin by beta."), ev("dpo_contract", "_dpo_train fit lr=0.001 and run_dpo beta=.1,1", "These experiment settings are separate.")], ["paper_text", "alignment_code", "dpo_methods", "checks_stdout"]),
    claim("baseline_zero", "empirical", "The fixed content.pt SFT baseline answered zero of seven held-out arithmetic test questions correctly.", "course/chapters/13.md:132", "Seven original toy arithmetic test records under raw token exact-match criterion; original GPU evaluation, independently audited without model rerun.", [ev("style_measurements", "/results/content_evaluation/test/{records,examples,matches,exact_match,samples}; /results/content_checkpoint", "Recorded 0/7 with raw expected/generated token sequences."), ev("common_contract", "evaluate_lm 249–283", "Samples contain prompt messages only; exact score uses generated raw answer IDs before EOS."), ev("token_contract", "ByteTokenizer 14–28", "UTF-8 bytes plus8; EOS2."), ev("cpu_checks", "CONTENT_BASELINE_TEST and RECONSTRUCTED_SPLITS", "All seven arithmetic answers independently checked; token exact flags rederived; split SHA matches.")], ["style_raw", "common_methods", "data_code", "checks_stdout", "checks_code", "measurements"], verification("7 test examples; matches0; exact_match0; content.pt linked", "7 samples, each predicted number differs; raw token exact flags allFalse; matches0/7", "Read only named raw pointers; reconstructed original seed42 arithmetic split and its exact JSONL SHA; no checkpoint loaded.", denominators={"test_records":7,"test_families":4,"criterion":"raw answer token IDs before EOS equal encoded true answer; every number also independently computed from question"})),
    claim("experiment_reference", "empirical", "Both beta .1 and1 runs start from the same content.pt base and leave the reference numerical state fixed.", "course/chapters/13.md:132", "Original recorded comparison and original immutable code's numerical before/after hash guard; not a new state-hash measurement or training run.", [ev("dpo_contract", "run_dpo 697–718; _state_digest 163–169", "Single base dependency, independently copied policies, frozen reference, complete state hash checked after both policy runs."), ev("dpo_measurements", "/results/reference_sha256,/results/reference_unchanged,/results/runs/{model,beta1}/beta,/training/steps and named preference samples", "Recorded fixed-state digest, True, beta values,250 steps each and consistent reference margins."), ev("cpu_checks", "ALL_RECORDED_REFERENCE_MARGINS, RECORDED_REFERENCE_DIGEST, BETA_RUNS", "All15 raw held-out reference margins invariant across both beta branches; original split provenance independently reconstructed.")], ["dpo_methods", "dpo_raw", "checks_stdout", "checks_code", "measurements"], verification("Same copied base; beta.1/1; recorded reference unchanged and invariant margins", "Original code contract matches raw code SHA; unchangedTrue and digest4cf7b1e...; all15 reference margins identical in each run", "Audited original hash guard and recorded digest; did not load weights or independently recompute the original full reference state hash.", denominators={"train_records":49,"validation_records":8,"test_records":7,"beta_branches":2,"recorded_updates_each":250,"checked_reference_margin_rows_each_branch":15})),
    claim("margin_counterexample", "empirical", "At validation 3+5=?, beta.1 changes chosen8 minus rejected9 gap from about -10.89346 to -8.44789, an improvement2.44557 that still favors9.", "course/chapters/13.md:134", "One named validation sample, same full-answer sum/EOS convention and same reference; candidate probability ordering, not a generated-answer accuracy claim.", [ev("dpo_measurements", "/results/before/validation/samples/2 and /results/runs/model/preference/validation/samples/2", "Raw chosen/rejected scores, unchanged reference margin, relative gap and token counts."), ev("dpo_contract", "_preference_evaluate 609–641", "Same prompt, own candidate prefix, summed log probabilities; margin difference definitions."), ev("alignment", "sequence_log_probability 29–35", "Sums valid labels; no length averaging."), ev("gap_derivation", "Raw subtraction and exp monotonicity", "Improvement can coexist with negative absolute chosen-minus-rejected gap."), ev("cpu_checks", "RAW_3_PLUS_5_MARGINS", "Independent arithmetic reproduces every decimal and ordering.")], ["dpo_raw", "dpo_methods", "alignment_code", "checks_stdout", "checks_code", "measurements"], verification("-10.89346→-8.44789, relative2.44557, wrong candidate higher", "Exact -10.893460392951965→-8.447893142700195, relative2.44556725025177; rejected logp -5.2012939453125 exceeds chosen -13.649187088012695", "Five-decimal rounding; validation index2 in reconstructed seed42 split; each candidate2 valid targets includingEOS; all raw scores re-subtracted.", tolerance="Displayed five decimals: absolute rounding error at most 5e-6; raw arithmetic exact equality.", denominators={"selected_validation_samples":1,"validation_records":8,"validation_families":4,"chosen_answer_targets_including_EOS":2,"rejected_answer_targets_including_EOS":2})),
]

report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "13.4",
    "source": "course/chapters/13.md#13.4", "source_sha256": manifest["source_sha256"],
    "reviewer_task": "/root/phase4_factual_coordinator/factual_13_4", "reviewer_context": "fresh",
    "verdict": "pass", "figure_sha256": {}, "sources": sources, "artifacts": artifacts,
    "claims": claims, "issues": [],
    "read_scope": {"section": "Complete original section lines102–137, including details and fence", "prerequisites": ["13.2 complete", "13.3 complete", "8.3 complete"], "intro_required": False, "old_reports_read": False, "author_correction_summaries_read": False, "frozen_input_snapshot": artifact_map["frozen_chapter"]["path"], "frozen_input_sha256": artifact_map["frozen_chapter"]["sha256"]},
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "Every substantive role, fixed-baseline, copying/freezing, initial-cancellation, SFT-reference, beta and nontruth claim mapped to original authority plus own bounded checks. No unsupported learning claim or unresolved substantive error identified.", "claim_ids": [c["id"] for c in claims]},
        "numeric_verification": {"status": "pass", "details": "Executed manual -1/0/1 arithmetic, initial0/log2 cancellation, raw0/7 exact criteria, original split SHA49/8/7 and exact 3+5 margins/rounding/token denominators. Existing GPU data were audited, not rerun.", "claim_ids": ["manual_gap", "initial_cancellation", "baseline_zero", "experiment_reference", "margin_counterexample"]},
        "figure_consistency": {"status": "not_applicable", "details": "Complete original section has no image/SVG references. Scalar arithmetic and explicit aliasing code supply the necessary material; no visual rendering/viewing claimed.", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "Personally read original DPO v3 version/equation/outline/code, official versioned Python/PyTorch source contracts and exact result-matched git source; original JSON fingerprints and named pointers retained. Cached locators were used only to locate original sources.", "claim_ids": [c["id"] for c in claims]},
        "limitations": {"status": "pass", "details": "Random fence checks copying/freeze only; no backward/preference training. Existing raw GPU measurements are confined to toy arithmetic and candidate ordering. Full reference-state digest was not independently rerun; original hash guard and all recorded reference margins were audited. No data download, existing-weight inference, GPU work, full training, figures, author edits or commits.", "claim_ids": ["copy_freeze_fence", "alias_exercise", "baseline_zero", "experiment_reference", "margin_counterexample"]},
    },
}
for item in artifacts:
    assert (ROOT / item["path"]).is_file()
    assert sha(ROOT / item["path"]) == item["sha256"]
for run in ["initial-run", "second-run", "third-run"]:
    old = json.loads((A / run / "execution.json").read_text())
    assert old["code_sha256"] == sha(A / run / "review_checks.py")
    assert old["stdout_sha256"] == sha(A / run / "checks.stdout.txt")
    assert old["stderr_sha256"] == sha(A / run / "checks.stderr.txt")
target = ROOT / "docs/technical-reviews/13.4.json"
raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
target.write_bytes(raw)
assert target.read_bytes() == raw
assert json.loads(target.read_bytes())["reviewer_task"] == report["reviewer_task"]
(A / "generated-report.json").write_bytes(raw)
print(json.dumps({"generated": True, "canonical_confirmed": True, "verdict": report["verdict"], "source_sha256": report["source_sha256"], "report_sha256": hashlib.sha256(raw).hexdigest(), "report_path": str(target), "claims": len(claims), "artifacts": len(artifacts)}, indent=2))
