"""Write this review from its actually read source freeze and completed receipts."""
from pathlib import Path
import hashlib
import json

BASE = Path("docs/technical-reviews/artifacts/natural-v4-factual/19.10")
audit = json.loads((BASE / "audit-results.json").read_text())
env = {k: str(v) for k, v in audit["environment"].items()}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def artifact(identifier, kind, tail, description, **extra):
    path = BASE / tail
    return {"id": identifier, "kind": kind, "path": str(path), "sha256": sha(path), "description": description, **extra}
def repo(identifier, path, note):
    return {"id": identifier, "kind": "repository_code", "title": path, "path": path, "sha256": sha(path),
            "version": "Current full-file SHA; compared with original experiment code/record receipts in this fresh audit.",
            "verified": True, "inspection_note": note}
def paper(identifier, title, version, note):
    return {"id": identifier, "kind": "paper", "title": title, "url": "https://arxiv.org/pdf/" + version,
            "version": version, "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
            "authority_reason": "Original authors' research paper; actual original PDF read independently, with retrieval SHA in a_originals.",
            "inspection_note": note}
def evidence(identifier, locator, supports):
    return {"source_id": identifier, "locator": locator, "supports": supports}
def claim(identifier, kind, statement, location, supports, scope, verification=None, status="verified", artifacts=None):
    value = {"id": identifier, "kind": kind, "statement": statement, "location": location, "status": status,
             "evidence": supports, "artifact_ids": artifacts or [], "scope": scope}
    if verification is not None:
        value["verification"] = verification
    return value
def check(status, details, identifiers):
    return {"status": status, "details": details, "claim_ids": identifiers}

sources = [
    paper("p_quant", "A White Paper on Neural Network Quantization", "2106.08295v1", "Read introduction p.2; §§2.2–2.3 pp.4–6, Eqs.(4)–(8), per-output-channel granularity and floating-point simulation versus fixed-point inference. Actual v2 request returned 404; v1 was retrieved and read."),
    paper("p_kd", "Distilling the Knowledge in a Neural Network", "1503.02531v1", "Read §§1–2 pp.1–3, Eq.(1), same-temperature soft targets, mixed hard-label/soft-target objective and T² gradient scaling. Independently derived fixed-teacher forward-KL equivalence to soft CE up to teacher entropy."),
    paper("p_moe", "Mixtral of Experts", "2401.04088v1", "Read §§2–2.1 pp.2–3, weighted expert outputs, G(x)=Softmax(TopK(xWg)), total/active distinction and specialized kernels. No transfer of Mixtral performance figures to this project."),
    paper("p_rms", "Root Mean Square Layer Normalization", "1910.07467v1", "Read §4 Eq.(4) pp.3–4 and explicit removal of the mean statistic. RMSNorm rescales without mean subtraction; this contradicts c4's wording for this RMSNorm capstone."),
    paper("p_dpo", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "2305.18290v3", "Actually read independently rehashed original cached PDF, §§3–4 pp.3–5, preferred/dispreferred response pairs, Eq.(7), gradient interpretation and DPO outline. Source lineage verified separately using actual project SHA records."),
    repo("r_quant", "tiny_perceptron/quantization.py", "Read complete file: quantize_symmetric:8–17, signed +8 nibble packing:29–44, QuantizedLinear:47–64, ordinary F.linear reconstruction and buffer storage accounting."),
    repo("r_ptq", "tiny_perceptron/capstone_quantization.py", "Read complete file: nonrouter Linear selection:21–26, FP32 state with per-channel values/scale:37–114, serialized reload:96–101, validated shape/range/scales and FP32 restoration:117–189."),
    repo("r_model", "tiny_perceptron/model.py", "Read configuration and model/blocks:15–106, embedding and untied output, RMS versus LayerNorm selection, Dense/MoE selection and answer-only masked CE denominator."),
    repo("r_core", "tiny_perceptron/capstone.py", "Read required model/config/projectors:24–121; full data builder:139–282; encoding/masking:299–339; evaluation and action/runtime/expected answer:499–605; load/export:644–704. Matched original run's full-file SHA and recomputed all relevant original data and raw-record scores."),
    repo("r_modern", "tiny_perceptron/modern.py", "Read complete file: RMSNorm:8–16 rescales without subtracting mean; DenseFFN:35–53; feature-dependent top-k router and separately stored experts:56–84."),
    repo("r_alignment", "tiny_perceptron/alignment.py", "Read distillation_kl/distillation_loss:45–63, teacher detach, shared T, vocabulary sum then mean over valid labels, T² once, and convex CE/KL combination; independently evaluated literal logits."),
    repo("r_student_runner", "scripts/course_experiments/capstone_student.py", "Read complete original-matching runner:35–147 identical deep-copied Dense origin, AdamW lr0.003, seed+5000 sampler, batch24, alpha0.5/T2, no_grad teacher, effective-answer count;150–227 DPO teacher validation/freeze, width48,350 steps, validation/test, KD serialized PTQ reloading."),
    repo("r_deploy_runner", "scripts/course_experiments/capstone_deployment.py", "Read complete original-matching runner; relevant §§run:272–305 export/reload and joint/DPO PTQ evaluation on same frozen test;368–399 recommendation, quantization provenance and interpretation. Benchmarks concern main teacher, not student speed."),
    repo("r_sampling", "scripts/course_experiments/capstone.py", "Read _balanced_sample:62–67 and deployment:312 onward. Balanced sampling chooses a sorted task then a row; reconstructed exact prescribed 350×24 answer-only budgets without training or claiming historical per-batch GPU logs."),
    repo("r_deploy_record", "docs/course-experiments/results/capstone_deployment.json", "Inspected original environment/config, results.data_manifest, public_stage_exports/ptq and exact binary/test artifact receipts. All necessary local raw evaluation hashes matched original receipts; independently recomputed scores and ID differences."),
    repo("r_student_record", "docs/course-experiments/results/capstone_student.json", "Inspected original GPU config/seed, teacher/initial SHA, recipe_frozen_before_test, branches with350 steps145163 valid positions, timings, identical architecture, exports/ptq, raw evaluation artifact hashes and file bytes. CE/KD scores and ID differences independently recomputed."),
    repo("r_joint_record", "docs/course-experiments/results/capstone_joint.json", "Read exact joint inference_export SHA and shared data manifest; establishes original joint source used for deployment/PTQ and DPO parent."),
    repo("r_dpo_record", "docs/course-experiments/results/capstone_preference.json", "Read original DPO parent_checkpoint_sha256, completed stage, inference_export SHA and data manifest. Parent is joint; student's teacher SHA is this DPO inference export."),
    repo("r_data_record", "docs/course-experiments/capstone-evidence/deployment/data.json", "Read frozen original data through CPU audit: all726 input rows/specifications, original seed42 split manifest. Current build_dataset exactly equals saved splits and manifest, split counts552/84/90 and zero family intersections. Independently looked up quoted failure questions by id."),
    {"id": "s_cpu", "kind": "execution", "title": "Fresh bounded CPU probe and original raw-record audit", "verified": True, "artifact_id": "a_execution"},
]
torch_receipts = json.loads((BASE / "torch-original-source-receipt.json").read_text())
for api in ("Linear", "Embedding", "no_grad", "manual_seed", "RMSNorm"):
    receipt = next(item for item in torch_receipts if item["api"] == api)
    sources.append({"id": "t_" + api, "kind": "official_source", "title": "PyTorch " + api + " original installed source",
                    "url": receipt["url"], "version": "2.14.1+cpu; git " + receipt["torch_git_version"],
                    "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
                    "authority_reason": "Original installed PyTorch implementation/docstring, independently inspected with version, full-file SHA and commit URL retained in a_torch.",
                    "inspection_note": "Actually read API docstring and implementation at starting line " + str(receipt["class_or_function_start_line"]) + "; inspected forward/context behavior, exact full package-file SHA in a_torch. For seed, claim is fixed environment rather than cross-platform reproducibility."})

artifacts = [
    artifact("a_source", "source_snapshot", "section.raw.md", "Own complete raw UTF-8 section freeze, including all blank lines; SHA matches dispatched current source."),
    artifact("a_prerequisites", "source_snapshot", "prerequisite-shas.json", "Actual necessary linked sections were read in full; source paths and unnormalized section SHA receipts. No introduction belongs to19.10."),
    artifact("a_code", "code", "audit.py", "Own fresh CPU verification code and independent decoding/scoring/denominator/ID-difference audit of original GPU raw records."),
    artifact("a_execution", "execution", "audit.stdout.txt", "Actual stdout: exact lesson block/exercise, CPU mechanisms and all original record recomputations; assertions completed.",
             command="PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/19.10/audit.py > docs/technical-reviews/artifacts/natural-v4-factual/19.10/audit.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-factual/19.10/audit.stderr.txt",
             result="Exit0. All mechanism and original-record assertions completed; RMSNorm wording counterexample recorded. No own GPU training/inference replication.", environment=env),
    artifact("a_results", "source_snapshot", "audit-results.json", "Own detailed computed results and exact full-file hashes of every actually inspected raw evaluation/input record."),
    artifact("a_stderr", "source_snapshot", "audit.stderr.txt", "Actual stderr for the completed bounded CPU audit."),
    artifact("a_analysis", "derivation", "original-source-analysis.md", "Own original-paper/API analysis with precise locators, KD algebra, normalization contradiction and empirical audit limits."),
    artifact("a_originals", "source_snapshot", "original-retrieval-receipts.json", "Actual original HTTPS URL/version/byte-SHA receipts, including failed nonexistent-v2 request and original cached DPO bytes independently rehashed; full papers ignored."),
    artifact("a_torch", "source_snapshot", "torch-original-source-receipt.json", "Actual installed PyTorch original package-file SHAs, implementation locators, version and git-source URLs."),
    artifact("a_render_receipt", "source_snapshot", "render-receipts.json", "Actual Inkscape commands, exit codes, stdout/stderr and source/render SHAs for five needed prerequisite figures."),
    artifact("a_figure_inspection", "derivation", "figure-inspection.md", "Own personal visual inspection after viewing each of five actual prerequisite renders; numbers, arrows and scope checked."),
]
render_receipts = json.loads((BASE / "render-receipts.json").read_text())
figures = {}
for i, receipt in enumerate(render_receipts, 1):
    figures[receipt["source"]] = receipt["source_sha256"]
    artifacts.append(artifact("a_figure" + str(i), "figure_render", Path(receipt["render"]).name,
                              "Actually rendered and personally viewed prerequisite figure " + receipt["source"] + "; inspection in a_figure_inspection."))

claims = [
    claim("c1", "concept", "Quantization approximates stored numbers with fewer-bit codes; PTQ follows training. Distillation trains a separately chosen student, so a student label alone does not shrink its architecture. Serialized artifacts should be reloaded and task-evaluated.",
          "19.10 lines477–479, opening quantization/distillation and PTQ paragraphs", [
              evidence("p_quant", "Introduction p.2; §§2.2–2.3 pp.4–6 Eqs.(4)–(8)", "PTQ follows a trained network and introduces numerical rounding/clipping; floating-point simulation differs from actual low-bit hardware."),
              evidence("p_kd", "§1 pp.1–2; §2 pp.2–3", "Knowledge transfer to a separately selected small model; distillation objective alone does not remove parameters."),
              evidence("r_deploy_runner", "run:282–304; r_ptq quantize_capstone:95–101", "Actual reload then task evaluation, rather than evaluating only an in-memory substitute.")],
          "This project uses weight-only PTQ and separate Dense students. Neither method guarantees preserved tasks or faster ordinary floating-point CPU inference.", artifacts=["a_analysis"]),
    claim("c2", "numeric", "The seeded Linear(8,4) has32 weights4 biases and FP32 payload144 bytes; packed4 stores16 weight+16 scale+16 bias=48 bytes. The single input produces(1,4), no gradients, and max absolute output difference0.0463463068 here; bits8 uses64 bytes and error0.0037038326 here.",
          "19.10 lines483–504 code and explanation; line542 bits8 exercise", [
              evidence("t_Linear", "nn.Linear class:53, weight/bias shapes, forward:130–134", "Linear output dimensions and FP32 parameter layout."),
              evidence("t_no_grad", "no_grad:22, __enter__/__exit__:81–86", "Disables gradient graph construction in the shown inference block."),
              evidence("t_manual_seed", "torch.random.manual_seed:49–60", "Fixes random-number seed in this same code/environment."),
              evidence("r_quant", "QuantizedLinear:50–64", "Per-output-row scale/bias buffers and dequantized F.linear; storage_bytes counts only buffer tensor payload.")],
          "Exact storage sums; error is one input's maximum output error, not task accuracy. FP32 buffer bytes exclude object/file/header/runtime peak. Seed reproducibility is for this fixed code/environment.",
          {"method": "executed", "expected": "144 FP32 bytes; packed4 48=16+16+16; bits8 64=32+16+16; shapes(1,4), nonzero error and8-bit smaller error for this seed/input.",
           "observed": "144,48,64 bytes; shapes(1,4); error4=0.046346306800842285, error8=0.0037038326263427734; no parameter gradients.",
           "tolerance": "Exact integer counts; floating errors are direct FP32 observations, not predetermined universal values.",
           "details": "Executed exact current lesson block from a_source and its sole bits=8 edit, reseeding42 for each branch; checked each buffer's shape,dtype,bytes and four-element maximum."}, artifacts=["a_code", "a_execution", "a_results"]),
    claim("c3", "software", "Full capstone PTQ quantizes every nonrouter Linear weight including experts, attention, media projectors and output; embeds, normalization weights, bias and router remainFP32. It uses oneFP32 scale per output row, and loading restores all weights toFP32 for ordinary inference.",
          "19.10 lines503–509 storage, projector/embedding roles, PTQ and deployment paragraphs", [
              evidence("r_ptq", "quantizable_weights:21–26; quantize_capstone:44–86; restore_quantized_payload:147–184", "Exact nonrouter Linear selection, retained floats and per-output scale shapes; integer decode multiplies FP32 scales and loads ordinary model."),
              evidence("r_core", "CapstoneModel:84–107; default_config:34–47", "Projectors map pooled48-image/16-audio features to model width; model owns embedding and FP32 norm choice."),
              evidence("t_Embedding", "Embedding class:14, forward:188–197", "IDs look up stored embedding vectors."),
              evidence("r_quant", "QuantizedLinear.forward:58–61", "Reference layer calls ordinary F.linear after reconstruction, not an integer kernel.")],
          "Storage/download reduction does not imply packed execution memory, one-eighth totalRAM or CPU speed. Bias/norm/router retention isolates direct quantization effects but not changed upstream router inputs.",
          {"method": "executed", "expected": "MoE FP32/packed4/packed8 tensor bytes1312512/248864/402720; Dense48 FP32/packed4 bytes319680/91680; all reconstructed parameters FP32, unquantized tensors exact.",
           "observed": "All expected payload counts matched; nonrouter lists matched; every restored parameter wasFP32; all retained embeddings,norms,biases,routers were exactly equal.",
           "details": "Built random models only, saved/quantized/reloaded under ignored research, independently summed integer values, scales and unquantized tensors; no original model weights or task inference loaded."}, artifacts=["a_code", "a_execution", "a_results"]),
    claim("c4", "concept", "The section describes normalization in this full capstone as adjusting both the numerical center and scale of a feature group.",
          "19.10 line505: ‘正規化則調整一組特徵的數值中心與尺度’", [
              evidence("p_rms", "§4 Eq.(4), pp.3–4, paragraph immediately following Eq.(4)", "Contradicts centering forRMSNorm: the original explicitly removes the mean statistic and divides byRMS."),
              evidence("t_RMSNorm", "RMSNorm class:343–427, RMS formula and forward", "Official current source independently documents no mean subtraction."),
              evidence("r_core", "default_config:41; CapstoneModel.language:85", "Actual capstone configuration is norm='rms'."),
              evidence("r_modern", "RMSNorm.forward:14–16", "Project implementation divides by sqrt(mean(x²)+eps), with no x−mean(x).")],
          "LayerNorm can subtract a feature mean; this capstone's actual RMSNorm does not. Correct the text to distinguish normalization methods or specifically explain RMS-only rescaling.", status="contradicted", artifacts=["a_analysis", "a_code", "a_execution", "a_results"]),
    claim("c5", "concept", "Dense applies one shared FFN at each position; MoE routing selects separately stored experts based on current input features. Preserving router weights does not guarantee the same experts after upstream quantization changes those features.",
          "19.10 lines505–509 per-channel/router and smaller Dense paragraphs; prerequisite19.2", [
              evidence("p_moe", "§§2–2.1 pp.2–3, G(x)=Softmax(TopK(xWg)) and weighted expert sum", "Sparse expert selection depends onx and is distinct from a Dense FFN and total storage."),
              evidence("r_modern", "DenseFFN:35–53; MoEFFN:59–80", "Actual Dense has one FFN; router computes probabilities/top-k from current features."),
              evidence("r_ptq", "quantizable_weights:21–26; retained FP32 state:44–53", "Router weights are retained while other upstream weights are quantized.")],
          "Possible route changes, not a claimed count of route changes in this run. Fewer stored/active parameters do not specify wall-clock speed.", artifacts=["a_analysis", "a_figure3", "a_figure_inspection"]),
    claim("c6", "empirical", "The original deployment containers/tensor payloads are joint1345023/1312512 bytes, joint4 275381/248864, joint8 429365/402720; CE andKD student342451/319680 each, KD4 106229/91680.",
          "19.10 lines513–518 and533–537 both delivery tables", [
              evidence("r_deploy_record", "results.public_stage_exports[joint,joint-int4,joint-int8] and matching artifacts entries", "Original export file/tensor sizes with binary SHA receipts."),
              evidence("r_student_record", "results.branches[ce,kd].export, results.ptq and artifacts[ce/model.pt,kd/model.pt,model-int4.pt]", "Original student container byte receipts; same student topology and KD-source PTQ."),
              evidence("s_cpu", "a_results.original_container_and_payload_bytes, original_binary_container_receipts_compared and architecture_and_ptq_mechanisms", "Checked receipt consistency and independently recomputed payload bytes from actual model shapes/packed tensors.")],
          "Original container counts are record audits, not fresh stat measurements of original binaries. The weights were not downloaded. Current CPU random-model containers differ; public metadata removal may change later release sizes/SHA. Payload excludes runtime peaks.",
          {"method": "executed", "expected": "The six file/payload pairs shown in both section tables.", "observed": "All six exact byte pairs matched original export and binary artifact receipts; independent packed payload counts matched.",
           "details": "Compared metadata byte/SHA receipts independently and executed CPU structural storage recomputation. Original GPU model-container bytes remain attributed to the original run.",
           "denominators": {"joint_parameters": 328128, "student_parameters": 79920, "bytes_per_unquantized_FP32_element": 4, "delivery_variants": 6, "packed_bits": [4,8], "scope": "Original file receipts plus independent CPU payload counts"}}, artifacts=["a_execution", "a_results"]),
    claim("c7", "empirical", "Joint andDPO FP32/4-bit/8-bit each score78/90 with identical per-task score counts. Joint4 changes one generated-ID sequence (1+0 readback111→11), DPO4 changes two; both8-bit branches show zero sequence changes. Changed rows were already wrong.",
          "19.10 lines511–524 joint/DPO PTQ scores, specific failure, linked raw records and scope", [
              evidence("r_deploy_runner", "run:286–305", "PTQ models actually serialized/reloaded before same test evaluation; joint andDPO kept distinct."),
              evidence("r_deploy_record", "results.stages and joint_ptq; artifacts[test-joint.json,test-joint-ptq4.json,test-joint-ptq8.json,test-dpo.json,test-ptq4.json,test-ptq8.json]", "Original fixed90-row raw outputs with SHA receipts and seed/split metadata."),
              evidence("s_cpu", "a_results.record_scores_independently_recomputed and ptq_generation_id_differences[joint/joint4,joint/joint8,dpo/dpo4,dpo/dpo8]", "Fresh decode/parse/runtime/score recomputation from raw IDs, all task totals and direct ID comparisons including the quoted failure.")],
          "Audit of all original raw records, not own GPU generation replication. Equal scores on this fixed test do not mean numerical equivalence, all outputs identical or identical performance on unseen inputs; cache consistency is a different comparison.",
          {"method": "executed", "expected": "Six90-row evaluations all78 correct; joint4/joint8/dpo4/dpo8 changed rows1/0/2/0; exact joint failure111→11.",
           "observed": "All six recomputed78/90; all per-task dictionaries matched within each PTQ comparison; ID changes1/0/2/0; question 請算1加0。 is the sole joint4 change, both answers incorrect.",
           "details": "Independently decode every generated byte ID, checkEOS, expected input-answer pairing, runtime arithmetic and final answer; recompute saved booleans/totals and compare action/final-ID arrays, not decoded text only.",
           "denominators": {"seed":42, "data_version":"capstone-small-world-v2", "test_rows_per_variant":90, "variants":6, "calculator_rows":12, "tasks":12, "generation_max_new_tokens_per_turn":64, "original_device":"NVIDIA L4"}}, artifacts=["a_code", "a_execution", "a_results"]),
    claim("c8", "software", "The portable student is a genuinely smaller random Dense model with2 layers,width48,vocab264,79920 parameters, not a pruned/renamed MoE. Its preselected teacher is theDPO branch, whose parent isjoint; both student branches copy the same initial state and freeze the teacher.",
          "19.10 lines509 and525–527 student architecture, teacher identity and training setup", [
              evidence("r_student_runner", "run:160–175,179–193; _train_student:38,101–106", "DPO teacher validation/freezing; Dense48 from scratch, same deep-copied origin; teacherno_grad only forKD."),
              evidence("r_student_record", "results.initial_student_state_sha256, teacher_checkpoint_sha256 and branches.parameters", "Exact recorded initialization, teacher identity and architecture."),
              evidence("r_joint_record", "results.inference_export.sha256", "Original joint fingerprint8f7e… isDPO's parent."),
              evidence("r_dpo_record", "results.parent_checkpoint_sha256 and inference_export.sha256", "Original DPO fingerprintb242… matches student teacher, not recommendedjoint."),
              evidence("s_cpu", "a_results.architecture_and_ptq_mechanisms and original_run_provenance", "Actual CPU parameter count79920 and random initialization SHA exactly match original report.")],
          "A structural/source-lineage check with original run receipts; student quality measured separately. Student is not described as distilled recommendedjoint, and its smaller parameter count alone gives no speed figure.",
          {"method":"executed", "expected":"Dense config2×48/vocab264,79920 parameters; exact shared initial SHA3c1018…; DPO teacher SHA equals preference export, DPO parent equalsjoint export.",
           "observed":"All architecture/count and full64-character SHA comparisons passed, including exact CPU initial student-state match to original GPU record.",
           "details":"Instantiated actual configuration onCPU with seed42; independently hashed ordered state tensors; compared full recorded teacher/parent/source fingerprints, without original weight loading or training."}, artifacts=["a_execution", "a_results", "a_figure4", "a_figure_inspection"]),
    claim("c9", "concept", "KD combines0.5 answer-only CE with0.5T² forwardKL(p_T||q_T), T=2, p teacher/q student on matched known answer contexts. The helper suppliesT² internally once, detaches teacher targets and averages only effective answer positions.",
          "19.10 lines525–527 KD formula, KL and fixed-teacher explanation", [
              evidence("p_kd", "§2 Eq.(1) and weighted two-objective/T² paragraph, pp.2–3", "SameT soft distributions and a hard/soft mixed objective; fixed-teacher forwardKL equals softCE minus constant teacher entropy."),
              evidence("r_alignment", "distillation_kl/distillation_loss:45–63", "Actualp teacher/q student direction, teacher detach, valid-label mask, T² once and alpha mixing."),
              evidence("r_student_runner", "_train_student:97–108; run:169–170", "Same batch/context for teacher/student, answer-onlyCE, fixed teacher and no secondT² outsidehelper."),
              evidence("s_cpu", "a_results.distillation_formula", "Independent Python-math mixture agrees within1e-6; teacher gradientNone and ignored-position gradientzero.")],
          "Fixed alpha/T are this experiment's recipe, not generally optimal. Different objectives' training-loss values cannot rank model quality. Teacher forward compute remains an extra training cost.", artifacts=["a_analysis", "a_code", "a_execution", "a_results"]),
    claim("c10", "empirical", "TheCE/KD recipe fixes seed42, same random start and prescribed balanced sampler/batch24 for350 updates each; each reports145163 effective answer/EOS positions. Same prescribed sequence is reconstructible, but GPU per-batch IDs were not logged. Teacher forward compute is included forKD.",
          "19.10 line527 fixed350-step recipe, budgets, batch-log and training-cost limitations", [
              evidence("r_student_runner", "_train_student:35–115; run:169–193", "Same deepcopy origin, AdamWlr0.003, same sampler seed+5000, batch24, valid-label counts, KD teacher forward."),
              evidence("r_sampling", "_balanced_sample:62–67", "Exact uniform task-then-row sampling rule."),
              evidence("r_student_record", "results.branches[ce,kd].steps,effective_tokens,seconds,history; results.recipe_frozen_before_test", "Both completed350 updates and recorded145163 positions; actual time scope includes each branch training."),
              evidence("s_cpu", "a_results.student_budget_and_sampler", "Independent prescribed sampler reconstruction8400 sampled rows and145163 answer-byte/EOS positions per branch; sequenceSHA identical.")],
          "Recomputed prescribed sampling rule and aggregate positions, not a historical GPU batch-log replay or own training. OriginalCE/KD training times10.931293079/12.063111923s are retained with original device/timing scope and not generalized. No student inference speed/memory peak was measured.",
          {"method":"executed", "expected":"350×24 sampled rows,145163 effective answer positions per branch; identical prescribed sequence SHA and original shared initialization.",
           "observed":"8400 sampled rows perbranch,145163 effective positions each; sequenceSHA480758a3e018e6735cc8d1da9ff4e2288d8e1a06a86d2a0e5469ea9bbe8d485d identical; original completed-update records matched.",
           "details":"Used exact original-matching runner/sampler source andseed5042 to reconstruct selections without model forward or optimizer; counted each answer's UTF8bytes plusEOS, and compared original aggregate counts/configuration.",
           "denominators":{"seed":42,"sampler_seed":5042,"train_rows":552,"validation_rows":84,"test_rows":90,"updates_per_branch":350,"batch_size":24,"sampled_rows_per_branch":8400,"effective_answer_positions_per_branch":145163,"branches":2,"temperature":2,"alpha":0.5,"optimizer":"AdamW lr0.003","original_device":"NVIDIA L4"}}, artifacts=["a_code", "a_execution", "a_results"]),
    claim("c11", "empirical", "Both students validate59/84; testCE62/90,KD61/90,KD4 61/90. CE/KD calculator scores1/12 and0/12; both tool-return1/6,style0/3,image-shape0/9. The sole correctness difference is1+8: CE requests1+8→runtime9→answer9, KD requests1+9→runtime10→answer11. KD4 changes6/90 action/final-ID sequences, all already wrong.",
          "19.10 lines529–540 student results, subtask denominators, failure, table, PTQ and scope", [
              evidence("r_student_record", "results.evaluations and raw artifacts[validation-ce.json,validation-kd.json,test-ce.json,test-kd.json,test-kd-ptq4.json]", "Original student raw outputs and completed frozen recipe are direct authority for the finite experiment."),
              evidence("r_student_runner", "run:197–209", "Same validation/test rows forboth studentbranches; KD PTQ serialized/reloaded then evaluated."),
              evidence("r_core", "evaluate_rows:499–547 and expected_final:601–605", "Exact action+EOS, runtime, generated final content and expected answer protocol; each subtask is a disjoint subset of the same90 rows."),
              evidence("s_cpu", "a_results.record_scores_independently_recomputed, ce_kd_one_score_difference and ptq_generation_id_differences[kd/kd4]", "Fresh all-record scoring and raw-ID comparison reproduce totals, every task denominator, sole CE/KD correctness difference and6 already-wrong PTQ changes.")],
          "Single fixed350-step seed42 small-world experiment: KD did not beat same-sizeCE here and did not fully preserveDPO teacher capability. Not a conclusion about all Dense/KD methods. Task rows belong within90 and cannot be added again; no student speed or inference-memory result.",
          {"method":"executed", "expected":"Validation59/84 each; test62/90,61/90,61/90; calculator1/12 vs0/12, toolreturn1/6,style0/3,shape0/9 each; named1+8 failure; KD4 six already-wrong IDchanges.",
           "observed":"Every listed total/subtotal matched independent raw-ID scoring; CE/KD differing correctness set contains onlyidffe914bb5e9abc53f39f (1+8); KD4 exactly6 changed rows withfalse correctness before/after.",
           "details":"Recomputed every raw record from byte-ID decoding/EOS, known input-answer expected values, parsed request arithmetic and generated answer; task counters recomputed and compared; source hashes checked against original artifactreceipts.",
           "denominators":{"seed":42,"validation_rows_per_branch":84,"test_rows_per_variant":90,"student_test_variants":3,"updates_per_branch":350,"calculator_rows":12,"tool_return_rows":6,"style_rows":3,"image_shape_rows":9,"effective_answer_positions_per_branch":145163,"generation_max_new_tokens_per_turn":64,"original_device":"NVIDIA L4"}}, artifacts=["a_code", "a_execution", "a_results", "a_figure5", "a_figure_inspection"]),
]

report = {"schema_version":1,"review_stage":"technical","lesson_id":"19.10","source":"course/chapters/19.md#19.10",
          "reviewer_task":"/root/v4_review_coordinator/factual_v4_19_10","reviewer_context":"fresh",
          "source_sha256":sha(BASE / "section.raw.md"),"figure_sha256":figures,"verdict":"revise",
          "prerequisite_sections":json.loads((BASE / "prerequisite-shas.json").read_text()),
          "direct_section_figure_sha256":{},"introduction_read":"not_applicable:19.10 is not the first section; no owned introduction",
          "claims":claims,"sources":sources,"artifacts":artifacts,
          "issues":[{"claim_id":"c4","status":"contradicted","details":"Line505 says normalization adjusts numerical center and scale, but this capstone's norm='rms' implementsRMSNorm without mean subtraction. OriginalRMSNorm §4 Eq.(4), actual code and CPU counterexample agree.",
                     "required_correction":"Explain that this capstone usesRMSNorm to rescale byRMS without subtracting the mean; or distinguish LayerNorm's centering fromRMSNorm. Re-read revised current section and affected dependencies after actual corrective reader closure before factual pass."}],
          "checks":{
              "factual_accuracy":check("revise","All other claims matched original authorities/records; c4 incorrectly implies centering by the actualRMSNorm capstone.",[c["id"] for c in claims]),
              "numeric_verification":check("pass","Executed exact block/bits8 exercise, actual packed/reloaded structural counts, independent KD math and raw-ID recomputation of all scores/budgets/sizes; no ownGPUtraining claim.",["c2","c3","c6","c7","c8","c10","c11"]),
              "figure_consistency":check("pass","19.10 has no directSVG; personally rendered/viewed five actually needed prerequisite figures and checked their numerals/arrows/lineage/tool-loop scope against original sources and text.",["c1","c5","c8","c11"]),
              "source_verification":check("pass","Read original PDFs with exact versions/locators/retrievalSHAs and original installedPyTorch source; inspected original raw records and code whose fullSHA matches original run receipts; sources for c4 directly expose the contradiction.",[c["id"] for c in claims]),
              "limitations":check("pass","Kept payload/file/runtime costs separate, one-input error and fixed-seed results scoped, originalGPU record audit distinguished from replication, prescribed sampler from unrecorded per-batch logs, student/DPO teacher from recommendedjoint, scores from raw-ID equivalence; no student speed/memory extrapolation.",["c1","c2","c3","c5","c6","c7","c8","c9","c10","c11"])} }
Path("docs/technical-reviews/19.10.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"report":"docs/technical-reviews/19.10.json","verdict":report["verdict"],"source_sha256":report["source_sha256"],"claims":len(claims),"unresolved_issues":len(report["issues"])},ensure_ascii=False))
