"""Materialize this reviewer's own report without reading an earlier report."""
import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_18_12"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

original = json.loads((HERE / "inputs/extraction.json").read_bytes())
raw = (ROOT / "course/chapters/18.md").read_bytes()
heads = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
i = next(i for i, x in enumerate(heads) if x[0].startswith(b"## 18.12 "))
current = raw[heads[i].start():heads[i+1].start() if i+1<len(heads) else len(raw)]
assert hashlib.sha256(current).hexdigest() == original["source_sha256"]
observed = json.loads((HERE / "verification.stdout.json").read_bytes())
assert (HERE / "verification.stderr.txt").read_bytes() == b""
command = "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + BASE + "/verify.py"
(HERE / "commands.txt").write_text(
    ".venv/bin/python docs/review-tools/section_facts.py course/chapters/18.md#18.12 --output /tmp/factual-18_12-original\n"
    ".venv/bin/python docs/review-tools/section_facts.py course/chapters/18.md#18.12 --output /tmp/factual-18_12-executed --execute --timeout 40\n"
    + command + " > " + BASE + "/verification.stdout.json 2> " + BASE + "/verification.stderr.txt\n"
    "git show 5af615e5d7c9642afee800390fa072257f895d0c:scripts/course_experiments/compression.py\n"
    "git show 5af615e5d7c9642afee800390fa072257f895d0c:scripts/course_experiments/common.py\n"
    "git show 5af615e5d7c9642afee800390fa072257f895d0c:scripts/course_experiments/architecture.py\n"
    "git show 48a4f3e912b483d70aee57c42c2aac226534a9a6:scripts/course_experiments/architecture.py\n",
    encoding="utf-8",
)
save(HERE / "verification-execution.json", {
    "command": command, "cwd": str(ROOT), "exit_code": 0,
    "environment": observed["environment"], "stdout": "verification.stdout.json",
    "stdout_sha256": sha(HERE / "verification.stdout.json"),
    "stderr": "verification.stderr.txt", "stderr_sha256": sha(HERE / "verification.stderr.txt"),
    "scope": "CPU toy forward/backward and arithmetic over existing records only. No optimizer, model load, new model score, GPU, or training/data download.",
})
save(HERE / "inspection.json", {
    "reviewer_task": TASK, "date": "2026-10-05", "independence": "New section-specific reviewer; assignment supplied target/method/locator-only indices. No author, reader or old technical report body/conclusion read.",
    "actual_manuscript_scope": [
        {"source": "course/chapters/18.md#18.12", "lines": [original["section_first_line"], original["section_first_line"]+len(current.splitlines())-1], "read": "Entire section including optional detail and original Python fence"},
        *json.loads((HERE / "context-scope.json").read_bytes()),
    ],
    "frozen_whole_chapter_input": {"path": BASE+"/inputs/course/chapters/18.md", "sha256": sha(HERE / "inputs/course/chapters/18.md"), "meaning": "Actual original UTF-8 frozen input copied at initial reading; not a claim about later whole-chapter edits."},
    "original_code_inspected": {
        "model.py": "ModelConfig 15-28; Block 31-50; TinyLM 53-89; loss_sum/masked_loss 92-105",
        "alignment.py": "distillation_kl 45-55; distillation_loss 58-63",
        "modern.py": "DenseFFN 35-53; MoEFFN 56-84, including ordinary dispatch comments and method limits",
        "data.py": "IGNORE=-100; ByteTokenizer 14-28",
        "attention.py": "CausalAttention.__init__ 32-46 (projection contracts used for parameter arithmetic)",
        "compression-current.py and compression-provenance.py": "AST-located _teacher 55-67, _example 79-97, _examples 100-103, _dataset 110-128, _evaluate 216-284, _cache_text 337-347, _fit_text 350-417; _distill_case non-Return computations 721-828 and return field keys only 829-844; run_distillation non-Return computations 848-854. Extra results/limitations interpretation constants in final return not read.",
        "common.py at recorded revision": "text_examples 77-97: raw story BOS/bytes/EOS shifting and chunk boundary method",
        "architecture.py and architecture-teacher-provenance.py": "run_moe AST statement boundaries; source computations 461-477, fixed model.pt copy at 476-477; teacher_variant field 484 only; other result interpretation values not read. The teacher record's revision is 48a4f3e912b483d70aee57c42c2aac226534a9a6; its archived full file SHA matches teacher /code_sha256. The necessary loop AST exactly equals that in the distillation revision's file. Own initial report used the later revision label for raw teacher JSON; this metadata was corrected and that own draft preserved.",
    },
    "raw_json_inspection": {
        "inputs/distillation.json": ["top-level key/types; /revision; /seed; /torch_version; /python_version; /assets/0 raw asset provenance; /code_sha256; /results/tasks/moe_to_dense key/types", "/results/tasks/moe_to_dense/teacher_provenance", "/results/tasks/moe_to_dense/data", "/results/tasks/moe_to_dense/teacher_frozen_and_unchanged", "/results/tasks/moe_to_dense/teacher_test/generated_samples/family", "/results/tasks/moe_to_dense/runs/w32_ce/training", "/results/tasks/moe_to_dense/runs/w32_ce_kl/training", "/results/tasks/moe_to_dense/runs/{w32_ce,w32_ce_kl}/{validation,test} measurement scalars and all raw generated_samples"],
        "inputs/moe.json": ["top-level key/types; /results/dataset; /results/teacher_variant", "/results/variants/top2_aux0.01/model", "/results/variants/top2_aux0.01/training", "/artifacts checkpoint/dataset provenance", "/public_exports"],
        "outputs/course-control/37066555979/batch/moe/result.json": "Top-level key/types only; not used as source evidence",
    },
    "paper_inspection": {"distillation-v1.pdf": "Personally read version/authors on page 1, Introduction pages 1-2, Section 2 and Eq.1-2 pages 2-3", "switch-v3.pdf": "Personally read version/authors on page 1, Section 2.1 equations 1-2 and routing pages 5-6, Section 4.2 and Table 6 pages 16-17, parallelism cost trade-off page 22"},
    "official_source_inspection": "Fetched immutable PyTorch git 5c4886908584029761b579af026dcfb627c84070 over HTTPS with TLS verification. functional.py softmax 2176-2216, log_softmax 2293-2324, kl_div 3401-3475; module.py eval 2916-2932 and requires_grad_ 2934-2955; grad_mode.py no_grad 22-86; _tensor.py backward 566-625. Fetched functional.py equals installed bytes; module.py equals original locator snapshot bytes.",
    "figure_scope": "18.12 has no referenced figure. Context 18.6 SVG was not evidence for the target's technical claims and was not rendered. The target has no spatial/numeric diagram requiring visual inference.",
    "safety_and_limits": "Filename-only search emitted paths of other review files but none of their content was read. Permitted ordinary method docstrings/comments, objectives and raw results inspected. No prior verdict or author correction summary encountered. No permanent symlinks, workspace tree or .pt file added. Original full raw inputs retained unmodified; ignored locator indices only locate primary files.",
})

artifacts = []
ids = {}
for n, path in enumerate(sorted(HERE.rglob("*")), 1):
    if not path.is_file(): continue
    assert not path.is_symlink() and path.suffix != ".pt"
    relative = path.relative_to(HERE).as_posix()
    identifier = "a" + str(n)
    ids[relative] = identifier
    item = {"id": identifier, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
        "kind": "code" if path.suffix==".py" else "source_snapshot",
        "description": "Original unmodified source/input or this reviewer's reproducible record: " + relative}
    if relative in {"verification.stdout.json", "original-execution/stdout.txt"}:
        item.update(kind="execution", command=command if relative.startswith("verification") else ".venv/bin/python docs/review-tools/section_facts.py course/chapters/18.md#18.12 --output /tmp/factual-18_12-executed --execute --timeout 40",
            result="exit 0; bounded assertions passed" if relative.startswith("verification") else "exit 0; shape (1,2,264), parameters 18048/6104, student gradient True; empty stderr, zero guard events",
            environment={"python": "3.13.5", "torch": "2.14.1+cpu", "device": "CPU", "threads": "1", "torch_git_version": "5c4886908584029761b579af026dcfb627c84070"})
    artifacts.append(item)

sources = []
def local(identifier, filename, title, version, note):
    sources.append({"id": identifier, "kind": "repository_code", "title": title, "path": BASE+"/"+filename,
        "sha256": sha(HERE/filename), "version": version, "verified": True, "inspection_note": note})
for ident,file,title,note in [
    ("model","model.py","TinyLM/ModelConfig output and parameter contract","ModelConfig, Block, TinyLM, loss_sum and masked_loss personally read"),
    ("alignment","alignment.py","KL validity and shape contract","AST-located distillation_kl 45-55 and distillation_loss 58-63 personally read"),
    ("modern","modern.py","Stored MoE experts and Dense FFN","DenseFFN 35-53, MoEFFN 56-84; top-k dispatch, stored ModuleList, affine sizes personally read"),
    ("attention","attention.py","Attention projection dimensions","CausalAttention.__init__ four bias-free projections personally read"),
    ("data","data.py","Byte tokenizer and IGNORE ID","IGNORE constant and ByteTokenizer 14-28 personally read"),
]: local(ident,"inputs/tiny_perceptron/"+file,title,"Frozen current repository bytes, SHA-256 pinned",note)
revision="5af615e5d7c9642afee800390fa072257f895d0c"
local("compression","inputs/compression-provenance.py","Original recorded distillation experiment implementation",revision,"git show bytes exactly match original results /code_sha256; AST functions/branches recorded in inspection.json; explanatory final return strings skipped")
local("common","inputs/common.py","Original raw story target construction",revision,"text_examples 77-97 read; SHA exactly matches recorded /code_sha256")
teacher_revision=json.loads((HERE / "inputs/moe.json").read_bytes())["revision"]
local("architecture","inputs/architecture-teacher-provenance.py","Original fixed MoE teacher export",teacher_revision,"run_moe 461-477 computation and fixed teacher_variant literal at 484 read; SHA matches original teacher /code_sha256; necessary loop AST equals distillation-revision version")
local("raw-distillation","inputs/distillation.json","Original raw distillation measurements and provenance",revision,"Named raw pointers in inspection.json only; no final results interpretation, notes, review or correction summaries read; original whole JSON retained")
local("raw-moe","inputs/moe.json","Original MoE teacher provenance and data counts",teacher_revision,"Named raw pointers in inspection.json; fixed teacher provenance and checkpoint digest cross-matched; original whole JSON retained")

def official(identifier, kind, title, url, version, snapshot, locator, authority):
    sources.append({"id":identifier,"kind":kind,"title":title,"url":url,"version":version,"accessed_on":"2026-10-05",
        "authority_reason":authority,"checked_original":True,"verified":True,
        "inspection_note":locator+"; personally inspected unmodified original snapshot "+BASE+"/"+snapshot+" (SHA "+sha(HERE/snapshot)+")"})
official("hinton","paper","Distilling the Knowledge in a Neural Network","https://arxiv.org/pdf/1503.02531v1","arXiv:1503.02531v1, 2015-03-09","sources/distillation-v1.pdf","Pages 1-3, Introduction, Section 2, equations 1-2","Original Hinton, Vinyals and Dean paper, confirmed from title/authors/version on page 1")
official("switch","paper","Switch Transformers","https://arxiv.org/pdf/2101.03961v3","arXiv:2101.03961v3, 2022-06-16","sources/switch-v3.pdf","Sections 2.1, 4.2 and parallelism cost discussion; pages 5-6,16-17,22","Original Fedus, Zoph and Shazeer paper, title/authors/version confirmed on page 1")
for identifier,file,locator in [
    ("torch-functional","torch/nn/functional.py","softmax 2176-2216; log_softmax 2293-2324; kl_div 3401-3475; identical to installed file bytes"),
    ("torch-module","torch/nn/modules/module.py","eval 2916-2932; requires_grad_ 2934-2955"),
    ("torch-no-grad","torch/autograd/grad_mode.py","no_grad 22-86"),
    ("torch-backward","torch/_tensor.py","Tensor.backward 566-625"),
]: official(identifier,"official_source",file,"https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/"+file,"Immutable git 5c4886908584029761b579af026dcfb627c84070; installed torch 2.14.1+cpu","sources/official-"+file.replace("/","--"),locator,"Fetched from the PyTorch project's official upstream immutable commit, matching torch.version.git_version")
sources += [{"id":"cpu","kind":"execution","title":"Independent bounded CPU verification", "verified":True,"artifact_id":ids["verification.stdout.json"]},
    {"id":"original-fence","kind":"execution","title":"Unmodified original section fence execution","verified":True,"artifact_id":ids["original-execution/stdout.txt"]},
    {"id":"param-derivation","kind":"derivation","title":"Independent affine parameter sum","verified":True,"details":"For width w, untied embedding/output=528w; position=128w; attention=4w²; three LayerNorms=6w; each default FFN=8w²+5w. Teacher has three FFNs and router=3w; student one FFN and no router. Gives 18048 and 6104, exactly agreeing with executed parameter enumeration."}]

def ev(source, locator, supports): return {"source_id":source,"locator":locator,"supports":supports}
def verified(method, expected, seen, details, **kw): return {"method":method,"expected":expected,"observed":seen,"details":details,**kw}
proof=[ids["verification.stdout.json"],ids["verify.py"]]
claims=[
 {"id":"cross-architecture","kind":"concept","statement":"Output-distribution distillation can teach a Dense student from an MoE teacher without importing its experts, provided next-token candidates and conditioning positions share the same meanings.","location":"18.12 opening and paragraph 2","scope":"Output-based distillation; no hidden-feature subtraction or equality of teacher/student capability is claimed.","status":"verified","evidence":[ev("hinton","Introduction pp.1-2 and Section 2 pp.2-3","Knowledge viewed as input-to-output mapping; small student trained on teacher soft class distributions"),ev("switch","Section 4.2 pp.16-17, Table 6","Direct original-paper example of sparse-to-dense language-model distillation"),ev("model","TinyLM.__init__/forward 54-86","Internal width/depth differ while output last axis is vocab_size"),ev("alignment","distillation_kl 45-55","Only logits and validity labels passed; no expert identities are consumed")],"artifact_ids":proof},
 {"id":"shape-parameters","kind":"numeric","statement":"The toy teacher width16/3 experts/top2 and Dense student width8 both output (1,2,264), with total parameter counts 18048 and 6104; all three expert sets remain stored.","location":"18.12 paragraph 2 and reported fence output","scope":"Default single-layer, one-head, untied TinyLM configurations, two input IDs; parameters are stored count, not activation cost/time.","status":"verified","evidence":[ev("param-derivation","verify.py parameter_formula and verification.stdout.json /toy/*_parameter_formula","Independent affine counts including biases, embeddings, positions, norms, router and all experts"),ev("model","ModelConfig 15-28 and description 88-89","Default vocabulary 264 and exact total-parameter enumeration"),ev("modern","DenseFFN 35-53; MoEFFN 59-84","Hidden width4w, all experts stored, chosen expert axis top_k=2"),ev("attention","CausalAttention.__init__ 32-46","Four bias-free width-square projections")],"artifact_ids":proof+[ids["original-execution/stdout.txt"]],"verification":verified("executed","Both shapes (1,2,264); exact counts 18048/6104; three stored experts, two chosen per token","Original output matches; independent sums match; chosen shape (2,2)","Axes: batch1, positions2, candidates264. Entire ModuleList counted irrespective of selected experts.",tolerance="Exact integer and shape equality; no floating-point rounding involved")},
 {"id":"original-api-gradient","kind":"software","statement":"The original fence freezes the teacher and computes student KL gradients; backward alone performs no parameter update.","location":"18.12 Python fence and following two paragraphs","scope":"Grouped ordinary APIs covered: manual_seed, tensor creation, eval, requires_grad_, no_grad, module forward/indexing, description/parameters/numel, backward, shape/tuple, abs/sum/item and print. Toy gradient path only.","status":"verified","evidence":[ev("original-fence","original-execution/stdout.txt, execution.json and environment.json","Original unmodified fence completed on CPU with exact displayed output"),ev("torch-module","eval 2916-2932; requires_grad_ 2934-2955","Evaluation mode and explicit teacher parameter freezing are separate operations"),ev("torch-no-grad","no_grad 22-86","Teacher computation gradient recording disabled"),ev("torch-backward","Tensor.backward 566-625","Backward accumulates leaf gradients; this fence has no optimizer.step"),ev("cpu","verification.stdout.json /toy","Student gradient positive, teacher grads absent, every student parameter unchanged after backward")],"artifact_ids":proof+[ids["original-execution/stdout.txt"]],"verification":verified("executed","Student output gradient present, teacher gradients absent, no weight update","Gradient absolute sum 3.3217830657958984; all student weights unchanged; teacher requires_grad=False and grads None","Original fence executed first. Separate CPU assertions compare cloned parameters before/after backward; no optimizer or checkpoint.")},
 {"id":"kl-mask-and-exercise","kind":"software","statement":"distillation_kl compares teacher||student across candidates and averages valid positions; 265 student vocabulary causes a shape error and 264 restores the route. Artificial labels only mark both positions valid.","location":"18.12 fence explanation and vocabulary exercise","scope":"This helper's shape/validity contract, temperature1; not a complete conversation mask or automatic candidate semantic remapping.","status":"verified","evidence":[ev("alignment","distillation_kl 45-55","Teacher detached; probabilities/log probabilities across last axis; per-token sum, valid mask and mean; rejects shape mismatch"),ev("torch-functional","kl_div 3401-3475, softmax 2176-2216, log_softmax 2293-2324","PyTorch argument order and reduction='none' semantics"),ev("cpu","verification.stdout.json /toy/kl, /manual_kl, /one_valid_position_kl, /vocab_265_error, /restored_shape, /labels_are_validity_only","Independent formula, one-position mask, arbitrary nonignore label and exercise actually executed")],"artifact_ids":proof,"verification":verified("executed","KL equals sum p(logp-logq) averaged over two valid positions; 265 rejected and 264 accepted","Both manual/helper KL=0.3357211947441101; masked KL=0.36398816108703613; 265 raises ValueError; restored (1,2,264)","Numerical formula atol1e-7/rtol1e-6. Labels 999 give unchanged KL because labels only select validity. Additional width12/layers2 variant still outputs (1,2,264).")},
 {"id":"capacity-and-cost-limits","kind":"concept","statement":"Same output shape does not establish teacher skill, successful distillation, equality on all inputs or a faster Dense student; evaluation must compare a real teacher, baseline quality and measured costs.","location":"18.12 random-initialization limits and final main-text paragraph","scope":"Qualified method restrictions and experiment design recommendations, not new empirical claims about this random teacher.","status":"verified","evidence":[ev("hinton","Introduction p.2, paragraph before Section 2","Small model typically cannot exactly match all teacher soft targets"),ev("switch","Section 4.2/Table6 pp.16-17 and parallelism costs p.22","Sparse-to-dense transfer measured against baselines; FLOPs, communication and memory trade-offs affect runtime"),ev("cpu","verification.stdout.json /toy/student_parameters_unchanged_after_backward","The displayed toy path has no training updates, so cannot establish learned task ability")],"artifact_ids":proof},
 {"id":"original-experiment-design","kind":"empirical","statement":"The optional experiment uses the fixed 15.13 MoE teacher and two Dense width32/layer1 students with the same initialization and batch plan; students use first32 training stories while the teacher training split contains409.","location":"18.12 optional detail, paragraphs 1-2","scope":"Verification of recorded single-seed experiment/provenance and source operations; teacher-versus-student gap also includes data-budget differences. No historical training re-execution.","status":"verified","evidence":[ev("raw-distillation","/results/tasks/moe_to_dense/{teacher_provenance,data,runs/w32_ce/training,runs/w32_ce_kl/training}","Teacher config/checkpoint and students' matching initialization/data/steps/batch digests"),ev("raw-moe","/results/{teacher_variant,dataset}; /results/variants/top2_aux0.01/model; /artifacts model.pt","Fixed teacher export and original409-record split SHA cross-match"),ev("architecture","run_moe 461-477; teacher_variant return literal","Fixed top2_aux0.01 file export independent of score ranking"),ev("compression","_distill_case 732-747 and 770-787; run_distillation 851-852","Selects first32 original documents, creates one Dense initial model, deep-copies into CE/CE+KL branches; original MoE case steps300")],"artifact_ids":proof+[ids["inputs/distillation.json"],ids["inputs/moe.json"]],"verification":verified("executed","Same Dense initialization/data/batch plan, 32 vs409 recorded story budgets and fixed teacher provenance","Matching initialization SHA492b04... and batch-plan SHA4955cd...; both300 updates,213 chunks,559651 supervised tokens; teacher checkpoint SHA2840bb... matches fixed export","Cross-checked original full JSONs, recorded code SHA, immutable code revision and raw teacher provenance. No teacher weights loaded; data counts checked against original recorded manifest, not rereading all training stories.",denominators={"teacher_training_stories":409,"student_training_stories_each":32,"student_updates_each":300,"student_sequence_chunks_each":213,"student_effective_supervised_tokens_each":559651,"seed":42})},
 {"id":"original-objective","kind":"empirical","statement":"The CE branch targets the original story's next token; CE+KL retains gold CE and adds teacher distribution on matching gold-prefix positions.","location":"18.12 optional detail first paragraph","scope":"Recorded raw-story objective; distinct from 18.8's demonstration of soft-target CE/KL algebra. The experiment changes mixture objective weights, not only adds an unweighted term.","status":"verified","evidence":[ev("common","text_examples 77-97","BOS + actual story byte IDs + EOS form shifted gold targets and chunks"),ev("compression","_cache_text 337-347; _fit_text 354-386","Shared gold-prefix examples; cached valid teacher rows; ce=masked_loss, ce_kl=(1-alpha)*ce+alpha*kl"),ev("raw-distillation","/results/tasks/moe_to_dense/runs/{w32_ce,w32_ce_kl}/training/{objective,alpha,temperature,loss_trace}","CE alpha0 versus CE+KL alpha0.5/T2 with T² applied once"),ev("hinton","Section2 p.3","Weighted combination of correct-label and teacher soft objectives supported")],"artifact_ids":proof,"verification":verified("executed","Recorded objective/trace weights agree with original code: CE vs0.5CE+0.5KL(T2)","All recorded trace loss values match their weighted CE/KL components within1e-6","Checks existing trace arithmetic and raw target method; no re-training.",denominators={"student_stories_each":32,"student_updates_each":300,"batch_size":16,"student_supervised_tokens_each":559651,"temperature":2,"alpha_ce_kl":0.5})},
 {"id":"recorded-generations-and-nll","kind":"empirical","statement":"Recorded student continuations still repeat fragments; lower average gold prediction cost does not establish story-writing ability. The linked raw result includes complete recipe metadata, raw per-record generation IDs and held-out evaluation.","location":"18.12 optional detail final paragraph","scope":"Existing51-validation/52-test continuations of at most24 generated tokens; lower NLL evaluated with teacher-forced gold text. No new model score and no general claim that all tasks improve.","status":"verified","evidence":[ev("raw-distillation","/results/tasks/moe_to_dense/runs/{w32_ce,w32_ce_kl}/{validation,test}","Raw sums/denominators and every sample's generated_ids, generated, expected and flags"),ev("compression","_evaluate 216-284","Defines gold target NLL, byte denominator including EOS NLL, fixed24-token reference continuation and EOS criteria"),ev("cpu","verification.stdout.json /existing_evidence_only/summaries","All raw ID strings and flags rechecked; sums/denominators recomputed; examples show repeated 'the'/'and'")],"artifact_ids":proof+[ids["inputs/distillation.json"]],"verification":verified("executed","Raw sample counts and exact/EOS flags agree; lower average NLL coexists with repetitive generation","Test NLL2.460136601788036→2.388968066028803; both0/52 exact and0/52 EOS. Validation2.4199832211158805→2.3559079991354825; both0/51 exact/EOS; raw generations include 'the t the the the the th'","NLL=nll_sum/supervised_tokens; bpb=nll_sum/(answer_bytes*ln2), atol1e-12. Reconstructed exact flags from stored reference/ID arrays and decoded all samples; no checkpoint or model inference used.",denominators={"validation_stories_each":51,"validation_targets_each":39256,"validation_bytes_each":39205,"test_stories_each":52,"test_targets_each":41914,"test_bytes_each":41862,"max_generated_tokens_per_sample":24})},
]

report={"schema_version":1,"review_stage":"technical","lesson_id":"18.12","source":"course/chapters/18.md#18.12",
    "source_sha256":original["source_sha256"],"figure_sha256":{},"reviewer_task":TASK,"reviewer_context":"fresh","author_tasks":[],
    "verdict":"pass","reviewed_on":"2026-10-05","actual_scope_artifact":BASE+"/inspection.json",
    "frozen_inputs":[{"path":BASE+"/inputs/course/chapters/18.md","sha256":sha(HERE/"inputs/course/chapters/18.md"),"meaning":"Initial actual UTF-8 whole-chapter snapshot; only named section/context content read."}],
    "artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","details":"All substantive main/optional/exercise claims separately covered by original papers, personally read code contracts and recorded raw result pointers.","claim_ids":[x["id"] for x in claims]},
        "numeric_verification":{"status":"pass","details":"Exact tensor axes/parameters/storage counts, KL denominator/mask arithmetic, original trace arithmetic and existing raw generation/nll denominators independently checked on CPU.","claim_ids":["shape-parameters","original-experiment-design","original-objective","recorded-generations-and-nll"]},
        "figure_consistency":{"status":"not_applicable","details":"18.12 contains no referenced image. Text and executed tensors suffice for this interface/gradient example; no target figure was rendered or claimed inspected.","claim_ids":[]},
        "source_verification":{"status":"pass","details":"Original Hinton v1 and Switch v3 papers personally inspected; immutable official PyTorch sources retrieved and matched to installed commit; raw recorded implementation SHA and teacher result/data/checkpoint provenance cross-matched. Locator indices were only navigation.","claim_ids":[x["id"] for x in claims]},
        "limitations":{"status":"pass","details":"Toy proves only a differentiable interface, not learned ability. Existing result is seed42, teacher409/student32 stories and matched student objective comparison; no GPU/training/checkpoint reevaluation or original full training-data reread. No guaranteed equal capability or latency inferred.","claim_ids":["capacity-and-cost-limits","original-experiment-design","recorded-generations-and-nll"]},
    },
    "unresolved":[],
}
assert report["reviewer_task"] == TASK
save(ROOT/"docs/technical-reviews/18.12.json",report)
print(json.dumps({"verdict":report["verdict"],"reviewer_task":report["reviewer_task"],"source_sha256":report["source_sha256"],"report_sha256":sha(ROOT/"docs/technical-reviews/18.12.json"),"artifacts":len(artifacts),"claims":len(claims)},ensure_ascii=False))
