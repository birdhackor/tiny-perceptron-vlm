"""Build this reviewer's complete new report without opening any prior report."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK = "/root/phase4_factual_coordinator/factual_15_11"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aid(path):
    return "proof-" + re.sub(r"[^a-zA-Z0-9]+", "-", str(path.relative_to(HERE))).strip("-")


provenance = json.loads((HERE / "input-provenance.json").read_bytes())
inspection = json.loads((HERE / "primary-source-inspection.json").read_bytes())
commands = {x["id"]: x for x in json.loads((HERE / "command-record.json").read_bytes())["commands"]}
cpu = json.loads((HERE / "cpu-verification.json").read_bytes())
raw = json.loads((HERE / "raw-measurement-verification.json").read_bytes())
original_env = json.loads((HERE / "original-run/environment.json").read_bytes())
environment = {k: str(v) for k,v in cpu["environment"].items()}
artifacts = []
execution_files = {
    "original-run/stdout.txt": (commands["original-fence"], {k:str(original_env[k]) for k in ("python", "torch", "cuda_build", "cuda_available", "device_requested")}),
    "cpu.stdout.txt": (commands["bounded-cpu"], environment),
    "cpu-verification.json": (commands["bounded-cpu"], environment),
    "raw.stdout.txt": (commands["raw-measurement-check"], {"python":"3.13.5", "device":"CPU JSON arithmetic only"}),
    "raw-measurement-verification.json": (commands["raw-measurement-check"], {"python":"3.13.5", "device":"CPU JSON arithmetic only"}),
}
for path in sorted(HERE.rglob("*")):
    assert not path.is_symlink(), f"No symlinks retained: {path}"
    if not path.is_file() or path.name.startswith("checker"):
        continue
    rel = path.relative_to(HERE).as_posix()
    kind = "code" if path.suffix == ".py" else "figure_render" if path.suffix == ".png" else "source_snapshot"
    entry = {"id":aid(path), "path":path.relative_to(ROOT).as_posix(), "sha256":sha(path),
             "kind":kind, "description":"15.11 independent permanent proof: " + rel}
    if rel in execution_files:
        command, env = execution_files[rel]
        entry.update(kind="execution", command=command["command"], environment=env,
                     result="Exit0; actual stdout/JSON retained. " + command["scope"])
    artifacts.append(entry)


def source_code(identifier, path, version, note):
    return {"id":identifier, "kind":"repository_code", "title":path.name,
            "path":path.relative_to(ROOT).as_posix(), "sha256":sha(path), "version":version,
            "verified":True, "inspection_note":note}


sources = [
    source_code("modern-code", ROOT / "tiny_perceptron/modern.py", "full source SHA " + sha(ROOT / "tiny_perceptron/modern.py"), "AST-located then personally read DenseFFN35-53 and MoEFFN56-84; selected experts only, bias-free router, reshape/topk/index_add/auxiliary paths."),
    source_code("recorded-architecture", HERE / "primary/architecture-recorded-revision.py", "Git48a4f3e912b483d70aee57c42c2aac226534a9a6; exact recorded code_sha256", "Personally read original _sync40-44, _clone_config66-69, _forward115-135, _train152-284, run_moe428-477; timer includes padding/device movement/forward/loss/backward/finite checks/clipping/scaler.step/update and synchronization. Original median formula latencies[3:]; peak reset before training. Did not read final author result explanations."),
    source_code("common-code", ROOT / "scripts/course_experiments/common.py", "SHA matches recorded experiment code_sha256", "AST-located then personally read new_lm45-47, text_examples77-97. Same seed, width-independent byte tokenization/chunking, max_length128; no asset extraction or training execution."),
    {"id":"original-fence-execution", "kind":"execution", "title":"Original15.11 Python fence on CPU", "verified":True, "artifact_id":aid(HERE / "original-run/stdout.txt")},
    {"id":"bounded-cpu-execution", "kind":"execution", "title":"Bounded inference variants, two sequence lengths", "verified":True, "artifact_id":aid(HERE / "cpu-verification.json")},
    {"id":"raw-record-check", "kind":"execution", "title":"Original L4 record pointer and arithmetic verification", "verified":True, "artifact_id":aid(HERE / "raw-measurement-verification.json")},
    {"id":"matrix-derivation", "kind":"derivation", "title":"FFN matrix and bias counts", "verified":True, "details":"Dense:8*16+16*8=256 matrix entries, +16+8 bias=280. Two experts:2*(8*4+4*8)=128, router8*4=32. Total MoE:4*(64+4+8)+32=336. Sequence lengths8/64 with batch2 flatten to16/128 token rows; this counts entries, not FLOPs."},
]
for record in inspection["records"]:
    kind = "paper" if record["id"] == "megablocks" else "official_source"
    locator = record.get("personally_inspected_original_line_ranges", record.get("personally_inspected_original_locators"))
    sources.append({"id":record["id"], "kind":kind, "title":record["id"] + " original source",
        "verified":True, "checked_original":True, "url":record["url"], "version":record["version"],
        "accessed_on":record["accessed_on"], "authority_reason":record["authority_reason"],
        "inspection_note":"Personally fetched and read original at " + str(locator) + "; exact excerpts/full paper and hashes in primary-source-inspection.json. " + inspection["version_scope"]})


def ev(identifier, locator, supports):
    return {"source_id":identifier, "locator":locator, "supports":supports}


def claim(identifier, kind, statement, location, scope, evidence, proof, verification=None):
    value = {"id":identifier, "kind":kind, "statement":statement, "location":location,
             "scope":scope, "status":"verified", "evidence":evidence,
             "artifact_ids":[aid(HERE / x) for x in proof]}
    if verification:
        value["verification"] = verification
    return value


claims = [
claim("cost-vs-speed", "concept", "Sparse expert selection reduces matrix-weight use per token without guaranteeing shorter elapsed time; routing/permutation/indexing, workload size and specialized GPU primitives affect cost.", "15.11 opening; paragraphs after CPU fence; final limitations paragraph", "Mechanism and possible overhead only. The small Python CPU and recorded L4 results do not establish general MoE/GPU ranking or isolate a single operation.",
    [ev("megablocks", "PDF p1 §1; PDF p5 sufficiently-large-block discussion; PDF p6 §5.1.1-5.1.2", "Dynamic routing/load imbalance can map poorly to existing software; block sizes/arithmetic intensity and specialized kernels affect efficiency."),
     ev("modern-code", "MoEFFN.forward67-84", "Python expert loop, topk/where/indexing/index_add and auxiliary calculation exist in this implementation."),
     ev("bounded-cpu-execution", "cpu-verification.json /lengths", "In this bounded CPU environment all five repeats at each size observed higher MoE elapsed time.")],
    ["primary/megablocks-inspected-excerpt.txt", "cpu-verification.json"]),
claim("shape-and-counts", "numeric", "Input axes are batch2 × length8 × width8; Dense uses256 matrix entries per token and280 total parameters; two width4 experts use128 entries plus32 router entries, with336 total MoE parameters.", "15.11 input-size paragraph and parameter explanation", "Count matrix entries and all stored learnable biases separately. Counts are exact for the stated GELU FFNs; not operation counts or fair quality budgets.",
    [ev("torch-nn-linear.py", "Linear53-116", "Weight shape is out_features×in_features; bias defaults true and has out_features entries; false removes additive bias."),
     ev("modern-code", "DenseFFN35-53; MoEFFN59-65", "Dense two Linear modules; four experts; router is bias=False."),
     ev("matrix-derivation", "explicit products in details", "256/128/32 and280/336 derived independently."),
     ev("bounded-cpu-execution", "cpu-verification.json /matrix_weight_derivation and /lengths/*/layers/*/parameter_tables", "Executed exact named parameter counts, tensor axes, output shapes and changed-length flattening.")],
    ["cpu-verification.json", "original-run/stdout.txt"],
    {"method":"executed", "expected":"Dense280, MoE336; [2,8,8]→16×8 and [2,64,8]→128×8; unchanged feature/layer widths.", "observed":"Exact counts280/336 at both sizes; exact output shapes and chosen shape[token_rows,2].", "details":"Separate matrices/biases; flatten only batch/sequence axes. Formula256=8*16+16*8; MoE336=4*(8*4+4*8+4+8)+8*4.", "tolerance":"Exact integer equality; manual expert merge max absolute errors2.3841858e-7 and1.1920929e-7, threshold1e-6."}),
claim("cpu-apis-and-timer", "software", "manual_seed fixes this environment's random input/initial weights; randn creates the stated shape; parameters/numel count stored parameters; one CPU intraop thread, eval and no_grad are used; 5 warmups then30 timed forwards averaged and converted seconds→milliseconds.", "15.11 original Python fence and two following API-explanation paragraphs", "CPU forward timing only, no backward/optimizer update or quality evaluation. eval sets evaluation flags; it does not itself disable autograd. Same-environment seed reproducibility was exercised; cross-version/hardware identity is not claimed.",
    [ev("torch-random.py", "manual_seed32-70", "Seed controls default RNG and device RNG initialization."),
     ev("torch-docs.py", "randn9119-9173; numel8396-8416; set_num_threads9814-9825", "randn shape and normal distribution, total-element count, CPU intraop thread count."),
     ev("torch-tensor-docs.py", "Tensor.numel3698-3705", "Tensor method has same count contract as torch.numel."),
     ev("pytorch-api.py", "parameters2666-2689; train/eval2886-2924", "Recursive parameter iterator; eval is train(False), distinct from gradient disabling."),
     ev("torch-autograd-gradmode.py", "no_grad21-85", "Disable reverse-mode graph recording for these forward results; restore grad state afterward."),
     ev("python-time", "Doc/library/time.rst321-348", "perf_counter supplies fractional seconds; elapsed difference is meaningful."),
     ev("original-fence-execution", "original-run/stdout.txt; environment.json attempted_fences[1]", "Actual unmodified fence executed on torch2.14.1+cpu/Python3.13.5 without guard events."),
     ev("bounded-cpu-execution", "cpu-verification.json /lengths/*", "Seed recreation exact; eval flags false; outputs need no grad, parameter gradients None and state hashes unchanged.")],
    ["original-run/fence-1.py", "original-run/stdout.txt", "cpu-verification.json", "primary-source-inspection.json"],
    {"method":"executed", "expected":"Original fence runs; two parameter counts280/336; forward-only outputs and unchanged parameter states; no fixed timing required.", "observed":"Original Dense0.0419ms and MoE0.5979ms per call; helper exit0. Repeated variants confirmed unchanged weights/no gradients and seed identity.", "details":"Original35 forwards per layer, divided timed duration by30 and multiplied1000. Bounded variants five timing windows per size retain every observed value. Official PyTorch contracts are v2.11.0, while local wheel is2.14.1+cpu; no equality of versions asserted."}),
claim("length-exercise-and-limits", "software", "Changing only sequence length8→64 increases token rows16→128 without changing parameter counts; measured runtime ratios may vary and are not a quality or universal speed result.", "15.11 exercise paragraph and quality/scope paragraph", "Observed workload grows eightfold in token rows; no assertion that wall time must grow eightfold, that a specific layer wins, or that fixed overhead was measured separately.",
    [ev("modern-code", "DenseFFN.forward45-53; MoEFFN.forward67-84", "Last feature width fixed; batch/sequence flattening and per-expert grouping scale with token rows."),
     ev("bounded-cpu-execution", "cpu-verification.json /lengths[0,1] and /moe_dense_time_ratios", "Both lengths executed with fixed feature/layer sizes, five repeats, exact counts and no training."),
     ev("megablocks", "PDF p5 sufficiently-large-block paragraph; PDF p6 §5.1.2", "Arithmetic intensity and parallelism depend on matrix/block sizes; hardware performance is workload dependent.")],
    ["verify_cpu.py", "cpu-verification.json"],
    {"method":"executed", "expected":"Shapes[2,8,8]/[2,64,8], same counts280/336; measured times are observations, not mandated outcomes.", "observed":"Both sizes executed, unchanged states and counts, all ten timing windows showed MoE slower in this environment. Length change did not yield a fixed timing multiplier.", "details":"Token count ratio128/16=8; no separate dispatcher timing or quality measurements performed."}),
claim("recorded-l4-table", "empirical", "The recorded NVIDIA L4 five-variant table contains the printed median update times and allocator before/peak MiB; same16-sequence, max128-position sampling schedule, 180 completed updates each, and auxiliary multiplier0.01 for the two MoE rows.", "15.11 L4 example, details sampling/timing paragraphs and five-row table", "Verification of existing raw reported measurements and producing implementation, not a new L4 run. Table medians are recorded summaries; the180 latency samples are absent and cannot be independently recomputed. These observations do not identify a single source of slowdown or establish quality ranking.",
    [ev("raw-record-check", "raw-measurement-verification.json /inspected_pointers and /converted_table", "Every table value is recalculated from named original seconds/bytes; configurations, raw runtime, batch/update/token denominators and code hashes checked."),
     ev("recorded-architecture", "_train152-284, especially sampler173/188, timer193-221, median275; run_moe428-477; _forward115-135", "Same records and RNG seed/sampler per variant; synchronized timer includes forward/backward/finite checks/clipping/AdamW update. Median excludes first3; aux0.01 multiplies loss. No latency isolation."),
     ev("common-code", "new_lm45-47; text_examples77-97", "Default max_length128 and width-independent chunking/tokenization preserve the sampling pool."),
     ev("megablocks", "p3 §2.2 balancing-loss discussion and §2.4", "Balancing objectives encourage even assignments; selecting multiple experts entails weighted sum, without claiming this experiment demonstrates balancing success.")],
    ["primary/moe-original.json", "primary/architecture-recorded-revision.py", "raw-measurement-verification.json", "raw.stdout.txt"],
    {"method":"executed", "expected":"Rows(ms,beforeMiB,peakMiB):[12.676,35.693,112.648];[11.965,67.945,119.091];[14.054,68.413,129.432];[27.630,68.454,120.711];[27.242,68.454,133.961].", "observed":"All15 displayed numerical cells equal raw-derived values after3-decimal rounding. Each row reports180 requested/completed successful updates,0 skipped, batch16 and337761 effective training tokens.", "details":"Seconds×1000, bytes/2^20; each displayed value within0.0005 unit. Exact source revision48a4f3e9 architecture SHA matches raw code_sha256. Raw JSON stores a median and not the latency array; method formula inspected, no replacement series invented.", "denominators":raw["denominators"]}),
claim("cuda-timing-memory-and-update", "concept", "CUDA wall timing requires synchronization; allocator allocated-byte peaks are reset per branch and include currently resident tensors, not reserved unused or driver/external allocations; gradient clipping limits the global norm and AdamW uses gradient moments to update weights.", "15.11 GPU synchronization sentence; details update and allocator explanations", "Official API meanings and recorded original timer/reset code only. Not per-model isolated deployment memory; distinct baselines prohibit treating the peak table as a memory-efficiency ranking. PyTorch docs/source checked atv2.11.0; recorded GPU runtime2.14.1+cu126 was not rerun.",
    [ev("torch-cuda-notes.rst", "Asynchronous execution277-307; Memory management465-478", "Queued GPU execution makes unsynchronized timings inaccurate; allocated tensors differ from total reserved cache."),
     ev("torch-cuda.py", "synchronize1152-1162", "Wait for all kernels in all streams on selected device."),
     ev("torch-cuda-memory.py", "reset_peak_memory_stats360-376; memory_allocated513-527; max_memory_allocated530-548", "Allocator peak reset, tensor occupied bytes, peak since reset; excludes unused cache/context from allocated metric."),
     ev("torch-nn-clipgrad.py", "clip_grad_norm_186-233", "Global concatenated-gradient norm, modifies gradients in place, handles nonfinite norm."),
     ev("torch-optim-adamw.py", "AdamW.__doc__60-106", "First/second gradient moments, decoupled decay and parameter update equation."),
     ev("recorded-architecture", "_train177-182,193-221,235-236,275-278", "Actual reset before training, both timing-boundary synchronization and gradient check/clip/update; returned peak and incremental bytes."),
     ev("raw-record-check", "raw-measurement-verification.json /converted_table", "Baselines differ and additional bytes exactly equal peak minus baseline.")],
    ["primary-source-inspection.json", "primary/architecture-recorded-revision.py", "raw-measurement-verification.json"]),
]
claim_ids = [x["id"] for x in claims]
report = {"schema_version":1, "review_stage":"technical", "lesson_id":"15.11",
    "source":"course/chapters/15.md#15.11", "source_sha256":sha(HERE / "original-run/section.md"),
    "reviewer_task":TASK, "reviewer_context":"fresh", "author_tasks":[], "verdict":"pass",
    "figure_sha256":{}, "artifacts":artifacts, "sources":sources, "claims":claims, "issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass", "details":"Six claim groups explicitly cover mechanism, sizes/counts, all fence APIs/timer, length exercise, original L4 records, CUDA/update/allocator semantics. Each has scoped original sources or bounded proof.", "claim_ids":claim_ids},
        "numeric_verification":{"status":"pass", "details":"Exact parameter/axis calculation280/336 and256/128/32; executed raw seconds/bytes conversions for15 table cells with3-decimal rounding, dataset/batch/update/token denominators checked. Median's missing underlying180-element series expressly recorded rather than synthesized.", "claim_ids":["shape-and-counts","recorded-l4-table"]},
        "figure_consistency":{"status":"not_applicable", "details":"15.11 references no images and makes no screen-state claim; no15.11 render PASS claimed. For necessary15.6 dispatch/combine context its SVG was rendered once with Inkscape after one20s Chrome timeout and personally viewed; separate render-record.json preserves this bounded scope.", "claim_ids":[]},
        "source_verification":{"status":"pass", "details":"Personally read original versioned MegaBlocks paper and official tagged PyTorch/CPython originals with precise locators, HTTPS URLs, access date, authority reason and retained excerpts. Locator bank used only to find original URLs. Experiment commit source retrieved and exact recorded SHA verified. Local2.14.1 vs official2.11 source gap recorded; APIs exercised locally.", "claim_ids":claim_ids},
        "limitations":{"status":"pass", "details":"Inference is untrained/forward-only; counts are matrix-use proxies, not FLOPs/quality. L4 timer contains entire update and additional host work, no isolated dispatch or specialized GPU core benchmark. Per-branch resident-memory baselines differ. Raw latency array absent, no new GPU/full-model evaluation, no training/data/model download. Full page desktop/mobile visualization untested, this section has no visual-dependent claim.", "claim_ids":claim_ids}},
    "actual_read_scope":{"source_section":"15.11 complete raw59lines (chapter394-452)",
        "necessary_previous_content":"15.6 raw text/code/method explanation and its referenced row-restore figure, used only to understand dispatch/combine; not a formal re-review of15.6 or chapter introduction.",
        "code_read_scope":provenance["inspected_repo_code"],
        "raw_json_pointer_record":"docs/technical-reviews/artifacts/phase4-15_11-independent/raw-measurement-verification.json /inspected_pointers",
        "forbidden_reads":"No old technical/reader reports, author-review/fix summaries, heldout scores or raw result comparison/auxiliary_note/limitations values read. Ordinary code method comments/control/limitations inspected and independently checked; no author empirical verdict exposure.",
        "no_other_section_verdicts":True},
    "frozen_full_chapter_input":provenance["frozen_full_chapter_input"],
    "source_version_note":"Full chapter SHA refers only to actually saved frozen raw input; source_sha256 is the exact15.11 section bytes. Introduction is not reviewed by this non-first reviewer.",
    "unresolved_questions":[],
    "verification_limits":[raw["underlying_latency_limit"], "No independent GPU/hardware replay or full model score reevaluation. No quality causal conclusion.", "Chrome attempt timed out; prerequisite standalone SVG verified with Inkscape; no full lesson desktop/mobile page check."]}
assert report["reviewer_task"] == TASK
(ROOT / "docs/technical-reviews/15.11.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"report":"docs/technical-reviews/15.11.json", "reviewer_task":TASK, "verdict":report["verdict"],
                  "source_sha256":report["source_sha256"], "report_sha256":sha(ROOT / "docs/technical-reviews/15.11.json"),
                  "claim_groups":len(claims), "proof_files":len(artifacts)},ensure_ascii=False))
