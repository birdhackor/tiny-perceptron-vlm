"""Write only this reviewer's complete new report, without reading its predecessor."""
import hashlib
import json
from pathlib import Path
import shlex

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
REL = BASE.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_18_14"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aid(name):
    return "a_" + name.replace("/", "__").replace(".", "_").replace("-", "_")


metadata = json.loads((BASE / "extraction.json").read_bytes())
checks = json.loads((BASE / "cpu-checks.json").read_bytes())
commands = {x["name"]: x for x in json.loads((BASE / "commands.json").read_bytes())}
environment = checks["environment"]
artifacts = []
executions = {"original-fence.stdout.txt": "original-fence", "bits8-variant.stdout.txt": "bits8-variant", "cpu-checks.json": "bounded-cpu-checks"}
purposes = {"section.md": "18.14 original UTF-8 section bytes actually read",
            "frozen-input-chapter-18.md": "Full frozen input corresponding to initial chapter SHA; byte copy, not a whole-chapter reading claim",
            "context-18_10.md": "18.10 current manuscript context actually read for fixed initialization/data/token-budget conditions",
            "evidence/distillation-original.json": "Unmodified full original run record; only named raw/provenance/method pointers were read",
            "inspection-pointers.json": "Expanded exact JSON pointers actually inspected; original result SHA retained",
            "source-inspection.json": "Original authority URLs, versions, SHA, actual reading locators and copy verification",
            "inspection.md": "True reading scope, task identity, provenance and execution record",
            "derivation.md": "Independent parameter/buffer arithmetic, units, denominators and MAE example",
            "cpu-checks.json": "Actual CPU tensor enumeration, formula/API assertions, raw saved-ID reaggregation and environment",
            "commands.json": "Actual argv/cwd/timeout/exit status and code/stdout/stderr hashes for all three short commands"}
for p in sorted(BASE.rglob("*")):
    if not p.is_file():
        continue
    name = p.relative_to(BASE).as_posix()
    kind = "execution" if name in executions else "derivation" if name == "derivation.md" else "code" if p.suffix == ".py" and not name.startswith("sources/") else "source_snapshot"
    artifact = {"id": aid(name), "path": f"{REL}/{name}", "sha256": sha(p), "kind": kind,
                "description": purposes.get(name, "Original authority snapshot independently inspected" if name.startswith("sources/") else "Original method code/fence or current-run supporting record; scope detailed in inspection.md")}
    if name in executions:
        job = commands[executions[name]]
        artifact.update(command=shlex.join(job["command_argv"]), result=f"Actual exit code {job['exit_code']}; all expected outputs/assertions passed. See commands.json and associated stdout/stderr.", environment=environment)
    artifacts.append(artifact)

authority = json.loads((BASE / "source-inspection.json").read_bytes())
ids = ["hinton", "torch-module", "torch-loss", "torch-numel", "torch-element-size", "torch-quantization", "torch-serialization"]
titles = ["Distilling the Knowledge in a Neural Network", "PyTorch Module registration and iteration source", "PyTorch L1Loss mean absolute error source", "torch.numel", "Tensor.element_size", "PyTorch Quantization", "PyTorch Serialization semantics"]
sources = []
for identifier, title, item in zip(ids, titles, authority, strict=True):
    kind = "paper" if identifier == "hinton" else "official_source" if identifier in {"torch-module", "torch-loss"} else "official_docs"
    sources.append({"id": identifier, "kind": kind, "title": title, "verified": True,
                    "url": item["url"], "version": item["version"], "authority_reason": item["authority_reason"],
                    "checked_original": True, "accessed_on": item["accessed_on"],
                    "inspection_note": item["inspection_locator"] + "; original bytes independently read after SHA-equal copying. " + ("Installed source equality independently asserted." if kind == "official_source" else "Version verified from original first page/title."),
                    "snapshot_path": item["snapshot"], "snapshot_sha256": item["sha256"]})

code_sources = [
    ("model", "code/tiny_perceptron--model.py", "TinyLM ModelConfig/Block/architecture", "Current original source SHA also matches the experiment record", "AST then lines 15-89, default vocab=264, max_length=128, untied, LayerNorm and FFN architecture"),
    ("quantizer", "code/tiny_perceptron--quantization.py", "Weight-only QuantizedLinear and packed4 implementation", "Current original source SHA also matches the experiment record", "AST then lines 8-74; per-output scale, int4 packing, float bias, registered buffers, float reference forward, recursive Linear-only replacement"),
    ("attention", "code/tiny_perceptron--attention.py", "CausalAttention initialization", "Current original source SHA also matches the experiment record", "AST then lines 31-48; four linear weight matrices, no biases"),
    ("ffn", "code/tiny_perceptron--modern.py", "DenseFFN initialization", "Current original source SHA also matches the experiment record", "AST then lines 35-48; hidden=4*width and two biased Linear layers"),
    ("tokenizer", "code/tiny_perceptron--data.py", "ByteTokenizer UTF-8 byte and control IDs", "Current original source SHA also matches the experiment record", "AST source for ByteTokenizer lines 14-28; content byte ID offset=8 and EOS=2, used for saved-token exactness reaggregation"),
    ("checkpoint", "code/tiny_perceptron--training.py", "Save/load checkpoint method contracts", "Current original source SHA also matches the experiment record", "AST then save_checkpoint lines 30-66 and load_checkpoint 79-103; metadata/RNG/training-state overhead, quantized-v1 restoration without optimizer updates"),
    ("original-experiment", "code/compression-at-experiment-revision.py", "Original distillation experiment methods", "Git revision 5af615e5d7c9642afee800390fa072257f895d0c; exact SHA matches /code_sha256/scripts~1course_experiments~1compression.py", "AST first; method spans individually recorded in inspection.md. Read _dataset, _storage, _packed, _evaluate, _cache_text, _fit_text, _distill_case and run_distillation calls; avoided result explanation values"),
    ("original-results", "evidence/distillation-original.json", "Original distillation run raw measurements and provenance", "Full original JSON SHA 53ed77dc35fce33163da45bf06916636a17ddacbbf1ccd54139ea8089f9a0b8d; /revision=5af615e5d7c9642afee800390fa072257f895d0c; recorded CUDA PyTorch 2.14.1+cu126, seed 42", "Repository-local raw run evidence, represented with the schema's repository_code kind. Inspected only method/provenance/storage/test/training/sample pointers listed in inspection-pointers.json after key/type inspection; no author result commentary or review conclusions read")]
for identifier, name, title, version, note in code_sources:
    sources.append({"id": identifier, "kind": "repository_code", "title": title, "verified": True,
                    "path": f"{REL}/{name}", "sha256": sha(BASE / name), "version": version, "inspection_note": note})
sources += [{"id": "original-run", "kind": "execution", "title": "Exact unmodified 18.14 fence CPU execution", "verified": True, "artifact_id": aid("original-fence.stdout.txt")},
            {"id": "bits8-run", "kind": "execution", "title": "bits=8-only original-fence variant CPU execution", "verified": True, "artifact_id": aid("bits8-variant.stdout.txt")},
            {"id": "cpu-audit", "kind": "execution", "title": "Independent bounded CPU method/arithmetic/raw-record checks", "verified": True, "artifact_id": aid("cpu-checks.json")},
            {"id": "arithmetic", "kind": "derivation", "title": "Independent TinyLM payload and MAE derivation", "verified": True,
             "details": "derivation.md: FP32 elements 658d+L(12d²+9d); packed totals include retained float embedding/norm, biases, channel scales and packed linear weights; quantized/teacher ratio uses byte/byte. MAE averages 4 elementwise absolute errors, independently checked."}]


def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


def claim(identifier, kind, statement, location, scope, evidence_items, artifact_names, verification=None):
    item = {"id": identifier, "kind": kind, "statement": statement, "location": location, "scope": scope, "status": "verified",
            "evidence": evidence_items, "artifact_ids": [aid(x) for x in artifact_names]}
    if verification is not None:
        item["verification"] = verification
    return item


claims = [
    claim("compression-goals", "concept", "較小架構減少權重數字，較低儲存位元縮小部分數字；蒸餾改變學生目標，並不單靠目標改變同架構學生的權重儲存形狀。", "course/chapters/18.md:560-562", "Mechanism and intended goal; no claim that this random toy distills or that real student quality must improve.", [
        evidence("hinton", "Original PDF pp.1-3, introduction and §2", "Teacher predictions provide additional training targets/objectives for a deployable student; improvement is task-dependent, not guaranteed here."),
        evidence("torch-quantization", "Official 2.8 Introduction, text lines 731-760", "Lower precision represents some or all tensors more compactly; does not entail architecture reduction."),
        evidence("model", "ModelConfig/TinyLM lines 15-89", "Widths and block count determine actual weight shapes independent of which loss trained them.")], ["sources/hinton-1503.02531v1.pdf", "sources/torch-quantization-2.8.html", "code/tiny_perceptron--model.py"]),
    claim("four-controls", "concept", "教師、普通同架構學生、蒸餾學生、同蒸餾權重的量化版應分開比較，配合固定訓練條件與同一獨立真值可辨別學習訊號與量化的影響。", "course/chapters/18.md:562,599; linked 18.10 context", "Controlled-comparison recommendation. The linked 18.10 explicitly supplies same init/data/token conditions; architecture equality alone is not a general causal guarantee. The original CE vs CE+KL protocol matches those controls.", [
        evidence("hinton", "Original PDF p.2 normal small-model comparison and p.3 weighted hard/soft objectives", "The learning-target change has to be assessed against ordinary small-model training; no four-arm performance guarantee is assumed."),
        evidence("original-experiment", "_dataset lines 110-128; _fit_text 350-417; _distill_case 720-826", "Family-disjoint same test set, copied same initialization, seed/batch plan/update budget, and direct packed conversion of post-training CE+KL student."),
        evidence("cpu-audit", "cpu-checks.json /matched_training and /existing_result_reaggregation", "Rechecks CE/CE+KL common hashes, 400 updates, 45 examples, 44985 effective tokens and 10 shared test records.")], ["context-18_10.md", "code/compression-at-experiment-revision.py", "cpu-checks.json"]),
    claim("payload-api-and-code", "software", "原程式隨機建立 teacher/student，deepcopy 學生再只替換 Linear；parameters+buffers 的 numel*element_size 和計算的是 tensor payload，轉換後的值、scale、bias buffer 仍會被計入。", "course/chapters/18.md:564-588", "Grouped ordinary API coverage: ModelConfig/TinyLM creation, copy.deepcopy preservation, replace_linear_layers recursion, parameters/buffers iteration, numel, element_size, sum, ratio and round. No backward, optimizer step, checkpoint write or distillation occurs. Quantized reference outputs stay FP32.", [
        evidence("torch-module", "register_buffer 528-556; parameters 2670-2695; buffers 2731-2755", "Buffers differ from parameters but persist in state; recursive default iteration includes all relevant submodule tables."),
        evidence("torch-numel", "torch.numel API, text lines 731-750", "Total element count, verified for scalar, empty and multidimensional tensors on installed 2.14.1."),
        evidence("torch-element-size", "Tensor.element_size API, text lines 731-770", "Bytes per element, independently checked FP32=4 and uint8=1."),
        evidence("quantizer", "QuantizedLinear lines 47-64; replace_linear_layers 67-74", "Packed values and FP32 scale/bias register as buffers; only Linear is replaced and forward materializes floating weights."),
        evidence("original-run", "Untouched fence SHA 95c859a9cb7e437b21b87d212887d0228b1357e6027ac4910638f416fa677109; stdout lines 1-4", "Actually ran the exact source fence, not a hand-simplified substitute."),
        evidence("cpu-audit", "check_structures.py structure loop and ordinary_api_cases; cpu-checks.json /structure_payloads", "Checks original student equality after deepcopy conversion, seven replaced layers, float retained parameters/scales/biases, and a short random forward shape/dtype.")], ["fence-1.py", "original-fence.stdout.txt", "cpu-checks.json", "check_structures.py", "commands.json"],
        {"method": "executed", "expected": "Exact fence prints three byte totals and rounded ratio; no original student mutation, only Linear replaced, all buffer tables counted; ordinary count/size APIs match shape/dtype.", "observed": "Exit 0; 67840,24416,15680 and 0.2311. Independent checks passed, with FP32 random output shape [1,3,264].", "details": "Current .venv Python 3.13.5, PyTorch 2.14.1+cpu, no CUDA build/availability, one CPU thread; official Module source SHA matches installed bytes. Official 2.8 stable numel/element_size definitions were explicitly versioned and tested on installed 2.14.1."}),
    claim("displayed-numbers", "numeric", "教師 67840 bytes，學生 24416，packed4 學生 15680，學生/教師比例 round(...,4)=0.2311；只把 bits 改成8後學生原版與教師不變，量化學生為17120。", "course/chapters/18.md:586,594", "Exact structural bytes in default untied FP32 TinyLM; ratio is quantized student bytes divided by teacher bytes. Whole-student eightfold reduction does not follow because retained floats, scales and biases remain.", [
        evidence("arithmetic", "derivation.md complete element/byte decomposition", "Independent total-element formula and separate packed/float/bias/scale terms verify each integer value, ratio denominator and no odd-nibble padding."),
        evidence("original-run", "original-fence.stdout.txt lines 1-4", "Exact original program verifies the four displayed outputs."),
        evidence("bits8-run", "fence-1-bits8.py changes only bits=4 to bits=8; stdout lines 1-4", "Actually executed the exercise variant: 67840,24416,17120 and ratio 0.2524."),
        evidence("cpu-audit", "cpu-checks.json /structure_payloads, /ratio, /rounded_ratio", "Independent named tensor enumeration equals the formulas; per-tensor FP32 and integer packing byte sizes independently checked.")], ["derivation.md", "original-fence.stdout.txt", "bits8-variant.stdout.txt", "fence-1-bits8.py", "cpu-checks.json"],
        {"method": "executed", "expected": "Exact byte integers 67840/24416/15680; byte/byte ratio rounds to 0.2311; bits8 total17120>15680 without altering student24416 or teacher67840.", "observed": "All numbers match exactly. Unrounded packed4/teacher ratio 0.23113207547169812; retained float parameter bytes12736 plus packed4 buffers2944 gives15680.", "details": "Teacher FP32 elements16960, student6104; packed4 weights1440 + scale1344 + bias160 + retained floats12736. bits8 weights2880 replaces1440 only. Each count is numel times element_size in bytes; no file/RAM denominator used.", "tolerance": "All bytes/counts exact integer equality; ratio rounding error below 0.00005 at four decimals."}),
    claim("existing-four-storage", "empirical", "既有屬性任務寬64教師 payload566272；寬32 CE/CE+KL 學生皆134528；從同一CE+KL學生直接 packed4為64160 bytes，最後一步没有新增更新。", "course/chapters/18.md:590,599", "Original version's saved measurements, not retraining. Compression is topology/storage only; CE/CE+KL results are not claimed as same quality and trained tensor contents were not rescored.", [
        evidence("original-results", "/results/tasks/attributes/teacher_provenance/config; /teacher_storage/{parameter_tensor_bytes,buffer_tensor_bytes,tensor_bytes}; /runs/w32_ce/storage; /runs/w32_ce_kl/storage; /runs/w32_ce_kl_packed4/storage (all under /results/tasks/attributes)", "Original raw measurements 566272,134528,134528,64160; exact expanded pointers retained in inspection-pointers.json, with original SHA."),
        evidence("original-experiment", "_storage 149-177; _packed 188-212; _distill_case 770-826; run_distillation 847-854", "Storage adds parameter and buffer bytes; _packed copies the same post-training model, saves/reloads quantized state, and contains no update; width32/layers1 and width64/layers2 provenance."),
        evidence("checkpoint", "load_checkpoint 79-103", "quantized-v1 reconstruction loads saved state strictly and does not train or update it."),
        evidence("cpu-audit", "check_structures.py raw-record/storage assertions; cpu-checks.json /structure_payloads and /existing_result_reaggregation", "Independent formula/random-topology enumeration matches all four saved totals, including packed4 50944 parameter +13216 buffer bytes.")], ["evidence/distillation-original.json", "inspection-pointers.json", "code/compression-at-experiment-revision.py", "cpu-checks.json", "derivation.md"],
        {"method": "executed", "expected": "Each saved tensor_bytes equals parameter_tensor_bytes+buffer_tensor_bytes and independently derived architecture bytes; CE and CE+KL shapes/cost match; packed4 follows same CE+KL model without added training.", "observed": "All assertions passed: teacher566272; CE134528; CE+KL134528; packed4 50944+13216=64160. Original method's direct-copy flow and quantized loader contain no optimizer update.", "details": "Original result revision5af615e5d7c9642afee800390fa072257f895d0c; code SHA verified. Only random matching structures instantiated on CPU; no original checkpoint loaded or written, no trained-model score recomputation.", "denominators": {"teacher_width": 64, "teacher_layers": 2, "student_width": 32, "student_layers": 1, "teacher_fp32_elements": 141568, "student_fp32_elements": 33632, "packed4_float_parameter_bytes": 50944, "packed4_buffer_bytes": 13216, "storage_unit": "byte", "original_seed": 42}}),
    claim("existing-blue-to-ble", "empirical", "packed4 在原對照的一題把蒸餾學生原答 blue 改成 ble，提供量化後可見品質退步的實例。", "course/chapters/18.md:590", "A specific attribute-task response and existing 10-record test only. No general claim that packed4 or distillation always worsens quality. Original saved IDs reaggregated; no fresh trained-model inference.", [
        evidence("original-results", "/results/tasks/attributes/runs/w32_ce_kl/test/generated_samples/2 and /results/tasks/attributes/runs/w32_ce_kl_packed4/test/generated_samples/2; teacher_test and three selected test generated_samples arrays", "Same family/question/reference blue; CE+KL exact blue versus packed4 nonexact ble. All raw saved samples and exact/EOS denominators remain available in original-byte copy."),
        evidence("original-experiment", "_evaluate 228-283; _distill_case 789-813", "Each same held-out record is greedily generated with24-token cap; exact raw IDs are compared before EOS, completion and EOS separate; same test set before/after packing."),
        evidence("tokenizer", "ByteTokenizer 14-28", "UTF-8 bytes map to IDs+8; EOS=2, independently used to verify reported strings and exact flags."),
        evidence("cpu-audit", "cpu-checks.json /changed_sample and /existing_result_reaggregation; check_structures.py saved ID loop", "Recomputes token exactness/EOS per saved sample; teacher5/10, CE4/10, CE+KL4/10, packed4 3/10; blue-to-ble identity assertions pass.")], ["evidence/distillation-original.json", "inspection-pointers.json", "cpu-checks.json", "check_structures.py"],
        {"method": "executed", "expected": "Same color question expects blue; prepacking saved raw response blue exact, postpacking ble nonexact. Reaggregated counts equal the original stored fields with identical test denominator.", "observed": "Exact and EOS flags match all40 saved responses; counts5,4,4,3 of10, all10 per-version responses EOS-ended; sample index2 proves blue→ble with same question/family/target.", "details": "Two held-out families, original seed42, CUDA run. 69 teacher-forced supervised target tokens including EOS, 24-new-token generation cap. This is a raw evidence consistency audit and not renewed score generation.", "denominators": {"test_records_per_version": 10, "heldout_families": 2, "teacher_forced_supervised_tokens_per_version": 69, "generation_cap_new_tokens_per_record": 24, "versions_checked": 4, "saved_responses_checked": 40, "original_seed": 42}}),
    claim("metrics-separate", "concept", "蒸餾損失、逐格權重 MAE 與答案品質是不同指標，較小 byte 或 MAE 本身不足以宣布任務成功。", "course/chapters/18.md:592,601", "MAE averages absolute errors over weight elements, not answers. Loss measures distribution/training objectives; saved generation exactness measures target-token correctness. No numeric accuracy-MAE threshold asserted.", [
        evidence("hinton", "Original PDF §2, p.3 hard/soft weighted training objectives", "Distillation objectives measure hard/soft target distribution costs, not generated answer identity."),
        evidence("torch-loss", "L1Loss class66-133 original docstring", "Mean absolute error equals mean elementwise |x-y| over all tensor elements."),
        evidence("original-experiment", "_fit_text 361-382; _evaluate 243-258", "Training CE/KL differs from per-response greedy token-exact evaluation."),
        evidence("cpu-audit", "cpu-checks.json /mae_hand_example and /changed_sample; derivation.md MAE denominator", "Four-element independent MAE example returns0.5 and agrees with L1Loss; observed packed sample changes despite storage savings.")], ["sources/torch-loss-original.py", "sources/hinton-1503.02531v1.pdf", "derivation.md", "cpu-checks.json"]),
    claim("ptq-and-calibration-scope", "concept", "本例為 Linear 權重-only，嵌入、norm、bias與中間特徵保留浮點；正式PTQ要對已蒸餾驗證的同一權重轉換再評測。8bit或保留敏感層浮點可另比較，bits改動並不完成中間特徵的校準。", "course/chapters/18.md:588,592,594", "Proposed experiments/workflow, not completed training or demonstrated quality improvement. Activation-range calibration refers to static activation quantization; this weight-only example does not require that calibration. Dynamic quantization is a separate approach.", [
        evidence("torch-quantization", "Original official2.8 PTQ Static lines1049-1058; module selection1610-1647; weight-only example1660-1750; types2110-2117", "Posttraining conversion, distinct weight-only/static/dynamic activation types, representative calibration for static activations, and selectively leaving modules float are supported; no quality guarantee."),
        evidence("quantizer", "quantize_symmetric8-17; QuantizedLinear47-64; replacement67-74", "Only weights quantized; per-output scales from weight maxabs. Activations and reconstructed weights stay floating; changing bits selects weight storage resolution only."),
        evidence("bits8-run", "bits8-variant.stdout.txt and fence-1-bits8.py", "8-bit variant demonstrates only cost change, without architecture change or learning."),
        evidence("cpu-audit", "check_structures.py float parameter/buffer and output dtype assertions", "Actually verifies remaining FP32 coefficients and FP32 outputs in random bounded CPU forward.")], ["sources/torch-quantization-2.8.html", "code/tiny_perceptron--quantization.py", "bits8-variant.stdout.txt", "cpu-checks.json"]),
    claim("cost-and-limitations", "concept", "tensor payload不等於檔案大小、執行RAM或速度；教師訓練/訊號生成成本應另算，部署只留學生不能抹去它；原 packed4 forward完整反量化到FP32，未證實低位元kernel速度或隔離執行RAM。", "course/chapters/18.md:586,588,590,599-601", "Explicit measurement boundaries. No private checkpoint file size treated as a public export, no dedicated low-bit speed/RAM gain inferred, no new mainline product capability claimed.", [
        evidence("torch-serialization", "Official2.8 Serialized file format for torch.save, text1306-1350", "Serialized objects have ZIP/pickle metadata, alignment and storage entries; payload-only count cannot predict final exported file size."),
        evidence("hinton", "Original PDF introduction p.1", "Large teacher training and lightweight deployment serve different computational requirements; teacher computation still exists."),
        evidence("original-experiment", "_cache_text337-347; _distill_case732-747; _packed188-212; _storage149-177; _evaluate216-284", "Teacher logits generation and saving happen outside student inference; converted reference forward is FP32; storage versus file overhead separate and no specialized low-bit or isolated-memory benchmark in this distillation branch."),
        evidence("original-results", "/timing_scope; /peak_memory_scope; /results/tasks/attributes/teacher_cache; selected storage/{file_bytes,file_overhead_bytes,forward}", "Scopes are method descriptions, not review conclusions: combined CUDA allocator scope isn't isolated packed inference RAM; metadata/file bytes distinct; teacher cache has its own generation time/file storage."),
        evidence("cpu-audit", "cpu-checks.json /structure_payloads and checked saved storage overhead", "Checks measured file_bytes-tensor_bytes equals stored overhead and random packed forward remains float; does not benchmark speed or peak memory.")], ["sources/torch-serialization-2.8.html", "sources/hinton-1503.02531v1.pdf", "code/compression-at-experiment-revision.py", "evidence/distillation-original.json", "cpu-checks.json"])
]

report = {"schema_version": 1, "review_stage": "technical", "lesson_id": "18.14", "source": "course/chapters/18.md#18.14",
          "source_sha256": metadata["source_sha256"], "reviewer_task": TASK, "reviewer_context": "fresh", "reviewed_on": "2026-10-05",
          "verdict": "pass", "figure_sha256": {}, "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
          "read_scope": [{"source": "course/chapters/18.md#18.14", "sha256": sha(BASE / "section.md"), "frozen_path": f"{REL}/section.md", "range": "original source lines558-603, full current section"},
                         {"source": "course/chapters/18.md#18.10", "sha256": sha(BASE / "context-18_10.md"), "frozen_path": f"{REL}/context-18_10.md", "range": "full linked section, context for controls only; its claims/fences are not certified by this report"}],
          "frozen_input": {"path": f"{REL}/frozen-input-chapter-18.md", "sha256": sha(BASE / "frozen-input-chapter-18.md"), "meaning": "Actual initial complete chapter byte snapshot corresponding to extractor source_file_sha256; does not claim present entire-chapter identity or whole-chapter reading"},
          "independence": {"old_review_content_read": False, "author_result_commentary_read": False, "locator_indices_only": True, "reused_author_or_reader_identity": False},
          "limits": "Short CPU random-structure/API checks and reaggregation of existing saved outputs only; no original trained-model score rerun, new training, GPU, data/model download, new weight file, symlink, manuscript edit or figure render. Main section has no figure. Original result is single-seed CUDA; current verification uses CPU. See inspection.md for exact scope.",
          "checks": {"factual_accuracy": {"status": "pass", "details": "All nine substantive claim groups have individual original-source/method support; linked18.10 conditions and original matched protocol confirm control wording.", "claim_ids": [x["id"] for x in claims]},
                     "numeric_verification": {"status": "pass", "details": "Exact original fence and bits8 variant ran; independent byte formulas, ratio rounding, MAE element denominator, raw storage sums and saved-response exact/EOS aggregation passed.", "claim_ids": ["displayed-numbers", "existing-four-storage", "existing-blue-to-ble"]},
                     "figure_consistency": {"status": "not_applicable", "details": "18.14 has no image/SVG references or spatial label mapping requiring a graphic; cost structures and tables are specified in text/code. No rendering/viewing claimed.", "claim_ids": []},
                     "source_verification": {"status": "pass", "details": "Original Hinton arXivv1, official versioned2.8 docs, immutable installed2.14.1 Module/L1Loss source independently read. Copies SHA-equal; installed source equality, raw result SHA and original experiment code commit/hash checked. Exact locators in source-inspection.json and JSON inspection-pointers.json.", "claim_ids": [x["id"] for x in claims]},
                     "limitations": {"status": "pass", "details": "Random structure vs learned behavior; whole-model floats/scales/bias overhead; payload vs export/RAM/speed; weight-only vs static activation calibration; teacher cost; original10-record two-family single-seed evidence; proposed alternatives vs completed experiment all kept distinct.", "claim_ids": ["compression-goals", "four-controls", "existing-four-storage", "existing-blue-to-ble", "metrics-separate", "ptq-and-calibration-scope", "cost-and-limitations"]}}}
assert report["reviewer_task"] == TASK
assert report["source_sha256"] == "d567bd6de852bfe2a2b8ed178beb89e45c937fc288ce03e955463e11a586d980"
(ROOT / "docs/technical-reviews/18.14.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print("Written own new report", sha(ROOT / "docs/technical-reviews/18.14.json"))
