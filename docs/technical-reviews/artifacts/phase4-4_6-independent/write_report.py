"""Assemble only this reviewer's fresh section 4.6 report from inspected evidence."""
import hashlib
import json
import shlex
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
PREFIX = ART.relative_to(ROOT).as_posix()


def read(name):
    return json.loads((ART / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aid(name):
    return "artifact-" + name.replace("/", "-").replace(".", "-")


def ev(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


extraction = read("original/extraction.json")
original_env = read("original/environment.json")
original_run = read("original/execution.json")
probe_env = read("probe-environment.json")
probe_run = read("probe-receipt.json")
probe = read("probe-result.json")
original_environment = {"python": original_env["python"], "torch": original_env["torch"],
                        "torch_git_version": original_env["torch_git_version"], "device": "cpu",
                        "cuda_build": original_env["cuda_build"], "cuda_available": original_env["cuda_available"]}
excluded = {"manifest.json", "checker.stdout.txt", "checker.stderr.txt", "checker-receipt.json"}
files = sorted(path for path in ART.rglob("*") if path.is_file() and path.name not in excluded)
artifacts = []
for path in files:
    relative = path.relative_to(ART).as_posix()
    kind = "source_snapshot" if relative.startswith("sources/") else "code" if path.suffix == ".py" else "source_snapshot"
    description = "Permanent original input/source/receipt snapshot: " + relative
    if relative == "inspection.md":
        kind, description = "derivation", "Reviewer-authored scope, primary-source inspection and numerical derivation; not a prior report."
    artifact = {"id": aid(relative), "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                "kind": kind, "description": description}
    if relative == "original/stdout.txt":
        artifact.update(kind="execution", description="Actual stdout of both original Python fences, in order.",
                        command=original_run["command"], environment=original_environment,
                        result="exit=0; both fences attempted; [1,3,20], argmax [[10,17,7]], mean CE=3.335638999938965; no guard events.")
    if relative == "probe-result.json":
        artifact.update(kind="execution", description="Independent bounded CPU assertions and full literal-tensor results.",
                        command=shlex.join(probe_run["command_argv"]), environment=probe_env,
                        result="exit=0; all shape/dot-product/causal/generation/reshape/CE/vocabulary/boundary/parameter-unchanged assertions passed.")
    artifacts.append(artifact)

receipts = {item.get("path"): item for item in read("sources/fetch-receipts.json")}
source_specs = [
    ("pt-linear", "pytorch-linear.py", "PyTorch Linear original API and implementation",
     "Personally read lines 53–134: affine formula, last-axis shape contract, learnable [out,in] weights, bias=False and F.linear forwarding. Supports vocabulary projection, not causal language-model training."),
    ("pt-embedding", "pytorch-sparse.py", "PyTorch Embedding original API and implementation",
     "Personally read lines 15–44: index lookup, dictionary size and [num_embeddings,embedding_dim] learnable table. Valid-ID boundary additionally read in functional.py and executed locally."),
    ("pt-functional", "pytorch-functional.py", "PyTorch functional embedding and cross_entropy original API",
     "Personally read embedding lines 2509–2557 and cross_entropy lines 3478–3569: max-index+1 table size; unnormalized scores, class targets (N,C)/(N), default mean/ignore_index=-100/label_smoothing=0, call into native CE. No model training implied by a loss call."),
    ("pt-loss", "pytorch-loss.py", "PyTorch CrossEntropyLoss exact class-index formula",
     "Personally read lines 1200–1242: raw logits, target class range, -log(exp(target)/sum(exp(classes))), weighted nonignored denominator and LogSoftmax+NLL equivalence. Only the unweighted nonignored three-target specialization is applied here."),
    ("pt-tensor-docs", "pytorch-torch-docs.py", "PyTorch original argmax and reshape documentation source",
     "Personally read lines 7106–7156 (argmax across a dimension; keepdim=False) and 9833–9863 (same data/element count, -1 inference). Used with execution to establish last candidate axis and exact row-order preservation."),
    ("gpt2-model", "gpt2-model.py", "OpenAI GPT-2 original causal language-model implementation",
     "Personally read model.py lines 1–174, especially 58–112 inclusive causal attention and 147–174 token/position tables, blocks, final norm and [batch,sequence,vocab] logits. This original implementation ties output weights, so it does not support TinyLM's untied default."),
    ("gpt2-sample", "gpt2-sample.py", "OpenAI GPT-2 original autoregressive sampler",
     "Personally read lines 1–95, especially 43–95: final-position logits, one sampled token and one-token output concatenation per iteration. It caches prior states; the source supports final-row selection, while full recomputation in the lesson is confirmed by the local uncached generate implementation."),
    ("hf-head", "transformers-gpt2-modeling.py", "Hugging Face official GPT-2 language modeling head implementation",
     "Personally read lines 156–178, 614–628, 975–994 and 1072–1103: attention head count is a separate module setting; language modeling head is explicitly a linear projection; hidden_states feed lm_head and logits are before SoftMax. Used only for terminology/functional distinction, not for installed-library execution or TinyLM weight tying."),
]
sources = []
for identifier, filename, title, note in source_specs:
    item = receipts[filename]
    sources.append({"id": identifier, "kind": "official_source", "title": title,
                    "url": item["url"], "version": item["version"], "accessed_on": item["accessed_on"],
                    "authority_reason": "Direct original file from the official maintainer repository, version pinned in URL; downloaded raw bytes and personally inspected.",
                    "checked_original": True, "verified": True, "inspection_note": note,
                    "snapshot_artifact_id": aid("sources/" + filename)})
for identifier, filename, note in [
    ("repo-model", "tiny_perceptron/model.py", "Personally read complete lines 1–131: width/vocab/layers/tied defaults, separate Embedding and bias-free output, final_norm before logits, dict return, and uncached generate's final-row argmax/concat/recompute loop."),
    ("repo-attention", "tiny_perceptron/attention.py", "Personally read complete lines 1–74: key_position<=query_position inclusive allowed mask, blocked scores replaced by -inf before softmax, and same-width attention output."),
    ("repo-ffn", "tiny_perceptron/modern.py", "Personally read lines 1–53; relevant DenseFFN up width->4*width, GELU, down back to width. Later branches were not reviewed or exercised."),
]:
    path = ROOT / filename
    sources.append({"id": identifier, "kind": "repository_code", "title": filename,
                    "path": filename, "sha256": sha(path), "version": "Repository HEAD ae8b2795c918b70773e664b6d6731c054ecff74e; current file bytes frozen for this review.",
                    "verified": True, "inspection_note": note,
                    "snapshot_artifact_id": aid("original/" + filename)})
sources += [
    {"id": "original-fences", "kind": "execution", "title": "Both original section 4.6 Python fences on CPU", "verified": True, "artifact_id": aid("original/stdout.txt")},
    {"id": "independent-cpu", "kind": "execution", "title": "Independent literal-ID shape, causality, generation, loss and boundary checks", "verified": True, "artifact_id": aid("probe-result.json")},
    {"id": "loss-derivation", "kind": "derivation", "title": "Independent three-question cross entropy derivation", "verified": True,
     "details": "For each position q, l_q=logsumexp_c(z_qc)-z_q,target_q in natural-log nats. From raw saved float32 scores evaluated in float64: [3.2427700458769175,3.528132585015193,3.236014399990441]; sum/3=3.3356390102941837. No class weights/ignored labels; denominator=3 questions, classes=20. F.cross_entropy differs by 1.0355e-8 and rounds to 3.336."},
]

claims = [
    {"id": "output-head", "kind": "concept", "location": "course/chapters/04.md:212–214",
     "statement": "The eight internal features at each position are a hidden representation; the output/language-model head maps them to vocabulary candidate scores using one weight vector per candidate and is a different function from an attention head.",
     "scope": "The eight features are preserved through the local block and final LN; the actual last computation is final_norm then bias-free Linear. 'Language modeling head' describes the output role, not an attention subdivision; no universal claim about tied weights.",
     "status": "verified", "evidence": [
         ev("pt-linear", "linear.py:53–80,130–134", "One learnable output row weights H_in features and maps only the last axis; bias=False gives a weighted dot product."),
         ev("hf-head", "modeling_gpt2.py:156–178,975–987,1084–1092", "Official wording language modeling head, implemented Linear(hidden,vocab), versus attention.num_heads; hidden_states feed the projection."),
         ev("repo-model", "model.py:43–50,60–66,81–86", "The local residual block preserves width and final_norm precedes output Linear."),
         ev("repo-ffn", "modern.py:35–53", "The local feedforward branch returns the original width after expansion."),
         ev("independent-cpu", "probe-result.json:stage_shapes,output_mapping", "Hooked block/final LN both [1,3,8]; final projection equals hidden@weight.T with exactly zero error.")],
     "artifact_ids": [aid("probe-result.json"), aid("probe.py"), aid("inspection.md")]},
    {"id": "tiny-api-scores", "kind": "software", "location": "course/chapters/04.md:216–232,240,260",
     "statement": "TinyLM(ModelConfig(vocab_size=20,width=8)) accepts one [1,2,3] ID row and returns dict['logits'] with shape [1,3,20] through token/position embedding, block, final LN and output. These are raw scores; argmax(dim=-1) returns one candidate ID per position with shape [1,3]. Its input/output weights are independent trainable tables, and this code only initializes/forwards/calculates a loss.",
     "scope": "Exact current repository default config, seed 42 and CPU environment. No tokenizer semantics, trained quality or generic guarantee about all language models. The forward creates an autograd graph but does not itself backward or update weights.",
     "status": "verified", "evidence": [
         ev("repo-model", "model.py:14–28,53–86", "Default tied=False/layers=1; separate tables; embeddings plus positions; blocks and final norm; raw output projection stored under logits."),
         ev("pt-linear", "linear.py:66–80,108–134", "Linear preserves batch/position axes and has learnable [20,8] weights."),
         ev("pt-embedding", "sparse.py:15–44", "Embedding adds feature axis with learnable table; supports the [1,3,8] token representation."),
         ev("pt-tensor-docs", "_torch_docs.py:7132–7143", "argmax selects maxima indices across supplied dimension, keepdim=False."),
         ev("pt-loss", "loss.py:1209–1213", "Unnormalized logits need not be positive or sum to one."),
         ev("original-fences", "original/stdout.txt:1–4; original/environment.json:attempted_fences", "Both original code fences run in order: [1,3,20], guesses [[10,17,7]], loss printed."),
         ev("independent-cpu", "probe-result.json:stage_shapes,output_mapping,argmax,raw_score_range,all_original_model_parameters_unchanged,all_original_model_parameter_gradients_none", "Shapes and axes; separate requires_grad storage; negative/unnormalized scores; no update or populated parameter gradients.")],
     "artifact_ids": [aid("original/stdout.txt"), aid("original/fence-1.py"), aid("original/fence-2.py"), aid("probe-result.json"), aid("probe.py")],
     "verification": {"method": "executed", "expected": "[1,3,8] hidden then [1,3,20] raw logits; argmax [1,3]; distinct trainable [20,8] tables; no parameter update.",
                      "observed": "All expected shapes and storage checks passed. Raw scores range -1.1452488899230957 to 1.211037516593933; argmax [[10,17,7]]; parameters exactly unchanged and grads None.",
                      "details": "Original two fences used their actual bootstrap and repository imports. Independent hooks and dot-product checks used literal IDs, with no optimizer/backward/checkpoint/dataset."}},
    {"id": "causal-position-generation", "kind": "software", "location": "course/chapters/04.md:234–240",
     "statement": "Rows 0,1,2 can read prefixes [1],[1,2],[1,2,3] and serve different next-ID questions with prescribed targets 2,3,4. Continuation chooses a new ID using only the last row, appends it and repeats the forward computation, rather than appending all three earlier-position guesses.",
     "scope": "Inclusive causal attention and this repository's default generate(use_cache=False), with no padding/cache/packing. Target 4 is a prescribed synthetic continuation, not a semantic fact inferred from arbitrary ID numbers. Efficient implementations may cache instead of recomputing full prefixes.",
     "status": "verified", "evidence": [
         ev("gpt2-model", "model.py:58–66,83–99,169–173", "Original causal mask excludes future positions and output logits have per-position vocabulary predictions."),
         ev("gpt2-sample", "sample.py:62–74,79–95", "Original autoregressive sampler uses only logits[:, -1, :] then appends one new token each loop, with a cache."),
         ev("repo-attention", "attention.py:10–28,64–74", "Local inclusive causal visibility mask, applied before score softmax."),
         ev("repo-model", "model.py:108–131", "Uncached loop forwards entire current sequence, selects last row, argmaxes and appends one token."),
         ev("independent-cpu", "probe-result.json:causal_check,generation", "Triangular mask and future-token perturbation/prefix equivalence; captured actual uncached inputs and manual last-row generation comparison.")],
     "artifact_ids": [aid("probe-result.json"), aid("probe.py"), aid("probe-receipt.json"), aid("inspection.md")],
     "verification": {"method": "executed", "expected": "Mask rows allow only positions <= row; changing future ID leaves earlier outputs unchanged; two steps append only two tokens.",
                      "observed": "Mask [[T,F,F],[T,T,F],[T,T,T]]; prior-position perturbation error=0; separate prefixes max error<=1.79e-7; forward inputs [1,2,3] then [1,2,3,7]; output [1,2,3,7,6].",
                      "details": "Prefix float32 comparisons use absolute tolerance 1e-6. Generation max_new_tokens=2,use_cache=False,eos_id=-1 prevents an early synthetic EOS; it exactly matches two manual final-row argmax steps."}},
    {"id": "reshape-three-question-loss", "kind": "numeric", "location": "course/chapters/04.md:242–253",
     "statement": "Reshaping [1,3,20] to [3,20] and targets [1,3] to [3] preserves row/label order, yields three cross-entropy questions with labels 2,3,4, and the initialized mean cost is about 3.336.",
     "scope": "Only the seed-42 random model and exact repo code on the recorded CPU version. Mean is unweighted over three nonignored labels in natural-log nats/question; it does not demonstrate learned answers or model accuracy.",
     "status": "verified", "evidence": [
         ev("pt-tensor-docs", "_torch_docs.py:9833–9863", "reshape retains data/element count and infers the -1 dimension."),
         ev("pt-functional", "functional.py:3478–3522,3561–3569", "Cross entropy accepts (N,C)/(N) class-index inputs and defaults to mean, no smoothing, ignore_index=-100."),
         ev("pt-loss", "loss.py:1218–1242", "Class-index negative-log probability and weighted nonignored mean; specializes to sum of three losses divided by 3 here."),
         ev("original-fences", "original/stdout.txt:4; original/fence-2.py:3–6", "Exact original initialized CE=3.335638999938965."),
         ev("independent-cpu", "probe-result.json:loss", "Exact row identity, preserved labels, independent float64 logsumexp calculation, changed-label variation and round-to-three-decimals test."),
         ev("loss-derivation", "inspection.md numerical derivation; probe.py:scores64 and manual_mean", "Three independent per-question calculations produce 3.3356390102941837 nats/question.")],
     "artifact_ids": [aid("original/stdout.txt"), aid("original/fence-2.py"), aid("probe-result.json"), aid("probe.py"), aid("inspection.md")],
     "verification": {"method": "executed", "expected": "Exactly [3,20]/[3], label order [2,3,4], denominator=3 and mean rounds to 3.336.",
                      "observed": "F.cross_entropy=3.335638999938965; float64 manual mean=3.3356390102941837; per-question manual values [3.2427700458769175,3.528132585015193,3.236014399990441]; changed labels [4,3,2] give 3.198777437210083.",
                      "details": "Every flattened row is exactly equal to its original position. -1=(1*3*20)/20=3; answers flatten to 3 labels. No ignored labels, weights or smoothing; 20 is candidate count, not loss denominator.",
                      "tolerance": "Shape/row/label checks exact; absolute CE versus float64 tolerance 1e-6 (observed 1.0355e-8); display 3.336 checked by rounding to three decimals (0.0005 rounding unit)."}},
    {"id": "vocabulary-and-index-boundary", "kind": "software", "location": "course/chapters/04.md:255,260",
     "statement": "Changing vocab_size from 20 to 25 with the same input gives [1,3,25], five additional candidate scores at each position and possibly different initialized argmax IDs; for vocab_size=20 the legal input IDs are 0–19 and ID20 is out of range.",
     "scope": "This repo uses integer embedding lookup and separately initializes the model for each configuration. The output shape increase is guaranteed by configuration; the observed changed guesses are a local example, not a guarantee of change for every initialization.",
     "status": "verified", "evidence": [
         ev("pt-embedding", "sparse.py:21–23,37–44", "num_embeddings determines dictionary rows; embeddings preserve input shape plus feature axis."),
         ev("pt-functional", "functional.py:2536–2539,2552–2556", "Embedding row count V equals maximum possible index+1."),
         ev("pt-linear", "linear.py:66–76", "out_features determines last output axis, including vocabulary size."),
         ev("repo-model", "model.py:60,64–66,71–85", "Both input table and output projection use configured vocab_size; forward starts with input lookup."),
         ev("independent-cpu", "probe-result.json:vocab_25,boundary_ids", "Executed vocab=25 shape/count/argmax variation and legal/illegal indices.")],
     "artifact_ids": [aid("probe-result.json"), aid("probe.py")],
     "verification": {"method": "executed", "expected": "vocab 25 gives [1,3,25] versus [1,3,20]; 5 more scores/question. 0,19 succeed; 20 and negative -1 fail for vocab20.",
                      "observed": "All shape/count assertions passed; output weights 160->200; vocab25 guesses [[18,19,17]], versus vocab20 [[10,17,7]]. IDs20/-1 raise IndexError('index out of range in self').",
                      "details": "Reset seed42 before allocating the larger random model. The total added logits are 3*5=15, with unchanged number of positions; synthetic boundary checks load no model/data files."}},
]

report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "4.6",
    "source": "course/chapters/04.md#4.6", "source_sha256": extraction["source_sha256"],
    "figure_sha256": {}, "verdict": "pass",
    "reviewer_task": "/root/phase4_factual_coordinator/factual_4_6", "reviewer_context": "fresh",
    "reviewed_on": "2026-10-05",
    "read_scope": {"section": "Current original section 4.6 bytes, original lines 210–263; no other subsection or chapter introduction reviewed.",
                   "implementation": "model.py 1–131, attention.py 1–74, modern.py 1–53, build_course.py BOOTSTRAP 26–61; complete factual instructions/helper/schema/protocol.",
                   "prior_reports_read": False, "official_sources": "Direct pinned official sources personally inspected; details in each source and inspection.md.",
                   "figures": "No referenced or embedded figure; no SVG/image rendered or viewed. Table data/shape checked. Browser typography not checked.",
                   "existing_measurements": "None cited; no existing measurement JSON, checkpoint, dataset, or training output required."},
    "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "Five material groups personally inspected: output role; TinyLM API/shape/raw scores/untied behavior; causal prefixes/continuation; row alignment/CE; vocabulary boundaries. No unresolved contradiction or uncertainty."},
        "numeric_verification": {"status": "pass", "claim_ids": ["reshape-three-question-loss", "vocabulary-and-index-boundary"], "details": "Original CE=3.335638999938965; independent float64 mean=3.3356390102941837, error1.04e-8; denominator3, candidate axis20, units nats/question, rounded3.336. Exact shapes/counts verified for vocab20 and25."},
        "figure_consistency": {"status": "not_applicable", "claim_ids": ["causal-position-generation"], "details": "Original extraction finds zero SVG/image references. No figure was available to render/view. The in-text prefix table was checked against the inclusive triangular mask and separate-prefix forwards; browser/mobile typography was not verified."},
        "source_verification": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "Direct original official PyTorch files at exact installed commit, OpenAI original GPT2 files at pinned commit, official Transformers4.57.1 terminology source. Every group has explicit locator/support scope and own inspection; URLs/version/date/hash/download receipts are permanent."},
        "limitations": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "Bounded CPU interface/loss checks only; no training, backward, model/download/data or quality claim. IDs2,3,4 are prescribed synthetic labels. Full-recompute continuation is the local uncached method; GPT2 official sampler caches. GPT2/HF tied weights are not attributed to default untied TinyLM. No diagrams present and no browser typography checked."},
    },
}
(ROOT / "docs/technical-reviews/4.6.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
manifest = {"scope": "Own permanent evidence for fresh section4.6 review; excludes this manifest and later checker receipt/streams to avoid circular hashes.",
            "source_sha256": extraction["source_sha256"], "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in files]}
(ART / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print("Wrote docs/technical-reviews/4.6.json: pass, 5 material claim groups,", len(artifacts), "permanent artifacts")
