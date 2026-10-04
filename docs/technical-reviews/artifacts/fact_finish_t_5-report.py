"""Persist this review's claims against the exact section snapshot already read."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_t_5"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads((BASE / f"{PREFIX}-audit-output.json").read_text())
    smoke = json.loads((BASE / f"{PREFIX}-cli-smoke-output.json").read_text())
    weights = json.loads((BASE / f"{PREFIX}-weight-identities-output.json").read_text())
    artifacts = []
    for id, filename, content in (
        ("cpu", f"{PREFIX}-audit-output.json", audit),
        ("cli", f"{PREFIX}-cli-smoke-output.json", smoke),
        ("weights", f"{PREFIX}-weight-identities-output.json", weights),
    ):
        path = BASE / filename
        artifacts.append(
            {
                "id": id,
                "kind": "execution",
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "description": content["scope"],
                "command": content["command"],
                "result": content["result"],
                "environment": content["environment"],
            }
        )
    registered = {a["path"] for a in artifacts}
    for path in sorted(BASE.glob(f"{PREFIX}*")):
        if path.name.startswith(f"{PREFIX}-prior-report-") or path.name.endswith(("-first-stderr.txt", "-report.py")):
            continue
        relative = str(path.relative_to(ROOT))
        if relative in registered or not path.is_file():
            continue
        artifacts.append(
            {
                "id": f"snapshot{len(artifacts)}",
                "kind": "code" if path.suffix == ".py" else "source_snapshot",
                "path": relative,
                "sha256": digest(path),
                "description": "Personally inspected source/data snapshot or durable raw output from this fresh T.5 audit: "
                + path.name,
            }
        )
    sources = [
        {
            "id": "sft-paper",
            "kind": "paper",
            "title": "Training language models to follow instructions with human feedback",
            "url": "https://arxiv.org/pdf/2203.02155v1",
            "version": "arXiv:2203.02155v1; original PDF bytes and extracted text saved in this review",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "Original InstructGPT researchers' paper defines demonstration-based supervised fine-tuning and explicitly distinguishes following instructions from safety.",
            "inspection_note": "Personally read §3.1 Step 1 (demonstrations of desired behavior, supervised policy) and §5.3 Models (neither fully safe nor fully aligned; harmful instructions still followed). Supports method and limitations only, not this tiny model's performance.",
        },
        {
            "id": "lora-paper",
            "kind": "paper",
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "url": "https://arxiv.org/pdf/2106.09685v2",
            "version": "arXiv:2106.09685v2; original PDF bytes and extracted text saved",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "Original LoRA paper defines the frozen base plus trainable low-rank update.",
            "inspection_note": "Personally read §4.1 eq.(3): W0+BA, A Gaussian/B zero, alpha/r scale; explicit merging. §4.2 permits dense-layer adaptation in principle but its own empirical study focuses on attention, freezes MLP, and lists memory/inference limitations. The repository's MLP/output placements are its own recipe, not the paper's empirical recipe.",
        },
        {
            "id": "pku-card",
            "kind": "official_docs",
            "title": "PKU-SafeRLHF original dataset card",
            "url": "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/raw/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md",
            "version": "dataset commit 9421ffafec3fa40a1f1a7d567b4d525079477ecb; complete README SHA 77ce394a7406b94e7ebe04b63e25850733286369db2d038936a6114502923579",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "Dataset creators' pinned documentation supplies label definitions and license.",
            "inspection_note": "Personally read complete original README including YAML cc-by-nc-4.0, Dataset Summary, separate helpfulness/harmlessness rankings, per-response safety categories, and warning. Ranking is relative and does not make a both-unsafe winner a safe SFT positive. The public 100-row snapshot preserves these source labels.",
        },
        {
            "id": "cc-license",
            "kind": "official_docs",
            "title": "Creative Commons Attribution-NonCommercial 4.0 International legal code",
            "url": "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en",
            "version": "CC BY-NC 4.0; retrieved 2026-10-04, HTML SHA 3307f50ad87c95b551b6a9c4e30c7fc4408f15e5b0e957542b2883f24df74474",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "Creative Commons publishes the actual legal text.",
            "inspection_note": "Personally read §1 NonCommercial definition, §2(a)(1) permission to reproduce/share original and adapted material for noncommercial purposes, and §3(a) attribution/notice/change conditions. Private publication is this project's policy; CC-BY-NC does not itself prohibit all public sharing. Initial .txt URL returned 403; .en original HTML retrieved with TLS verification.",
        },
        {
            "id": "cpu-source",
            "kind": "execution",
            "title": "This review's own CPU replay, denominator reconstruction and strict output audit",
            "verified": True,
            "artifact_id": "cpu",
        },
        {
            "id": "cli-source",
            "kind": "execution",
            "title": "This review's six-command CPU interface smoke",
            "verified": True,
            "artifact_id": "cli",
        },
        {
            "id": "weight-source",
            "kind": "execution",
            "title": "This review's public weight provenance and per-tensor inspection",
            "verified": True,
            "artifact_id": "weights",
        },
    ]
    descriptions = {
        "data": (
            "tiny_perceptron/data.py",
            "ByteTokenizer, render_chat and pad_batch: byte IDs, role masking, assistant bytes plus EOS targets, input/target shift.",
        ),
        "generator": (
            "scripts/prepare_data.py",
            "generate_records style/safety and main family split: three styles of the same arithmetic prompt; True/False permission examples only; generated validation row order.",
        ),
        "train": (
            "scripts/train.py",
            "parser, prepare_examples and main: --task sft, --train, --steps, max_length, masked_loss and optimizer.step guarded by --train.",
        ),
        "evaluate": (
            "scripts/evaluate.py",
            "answer_sample/evaluate/main: exact raw content IDs, normal-EOS completed exact, row, target, generated, skipped and separate actual denominators; default limit20/all.",
        ),
        "infer": (
            "scripts/infer.py",
            "main --chat encoding, --tokens cap, default temperature0, --adapter native-checkpoint requirement and JSON raw IDs/EOS.",
        ),
        "adapters": (
            "tiny_perceptron/adapters.py",
            "base_state_sha256 and load_lora_adapter: raw FP32 named tensor bytes, exact config, base fingerprint and tensor equality, finite A/B shape checks before mutation; rejects merged or different base.",
        ),
        "alignment": (
            "tiny_perceptron/alignment.py",
            "LoRALinear.forward/merged_weight: frozen wrapped Linear, A/B paths, alpha/rank and BA merge.",
        ),
        "behavior": (
            "scripts/course_experiments/behavior.py",
            "Personally read complete style/LoRA/safety definitions through run_safety; run_style/run_lora/run_safety, 11 LoRA placements, all-base freeze, sampling and scoring; safe-source filter and 120-byte crops.",
        ),
        "common": (
            "scripts/course_experiments/common.py",
            "Context/new_lm/split_records/text_examples/fit_lm/evaluate_lm: family dedup and disjoint splits, lr0.003/batch16 constant training, effective-label sum, every held-out record generation.",
        ),
        "text": (
            "scripts/course_experiments/text.py",
            "_save_splits, _steps, _asset_rows, _utf8_prefix and arithmetic_records: exact JSONL bytes, exchanged-addend family, source target scoping, UTF-8-character-safe byte prefix.",
        ),
        "fetch-models": (
            "scripts/fetch_course_models.py",
            "safe_relative/fetch_model: fixed40hex HF revision, token=False public requests, per-file SHA and byte count; no implicit all-model fetch.",
        ),
        "fetch-assets": (
            "scripts/fetch_training_assets.py",
            "main: selected asset, actual LFS pointer handling and manifest-verified unpack_asset.",
        ),
        "asset-integrity": (
            "tiny_perceptron/assets.py",
            "unpack_asset: archive and every member SHA/size, attribution files retained, validates before writing.",
        ),
    }
    for id, (path, note) in descriptions.items():
        sources.append(
            {
                "id": id,
                "kind": "repository_code",
                "title": path,
                "path": path,
                "sha256": digest(ROOT / path),
                "version": "Personally read current source bytes; snapshot and hash saved in fact_finish_t_5-input-identities.json",
                "inspection_note": note,
                "verified": True,
            }
        )
    for name in ("style", "safety"):
        path = BASE / f"{PREFIX}-formal-{name}-scripts_course_experiments_behavior.py.txt"
        sources.append(
            {
                "id": f"formal-{name}",
                "kind": "repository_code",
                "title": f"Original formal {name} experiment behavior source",
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "version": audit["formal_config"][name]["revision"],
                "inspection_note": "Retrieved exact original source with git show at the record's named revision and verified against that record's code_sha256. Personally inspected relevant functions and diff to current source. Style/LoRA code differences add illegal-ID safeguards; style's older safety generator is outside the style run. Current replays preserve every generated ID and rechecked rubric score.",
                "verified": True,
            }
        )
    claims = []

    def claim(
        id, kind, statement, location, scope, evidence, expected=None, observed=None, details=None, denominators=None
    ):
        value = {
            "id": id,
            "kind": kind,
            "statement": statement,
            "location": location,
            "status": "verified",
            "scope": scope,
            "evidence": [
                {"source_id": source, "locator": locator, "supports": supports}
                for source, locator, supports in evidence
            ],
            "artifact_ids": ["cpu", "cli", "weights"] if kind != "concept" else [],
        }
        if kind != "concept":
            value["verification"] = {
                "method": "executed",
                "expected": expected,
                "observed": observed,
                "details": details,
                "tolerance": "Exact counts, raw IDs and hashes; CPU/GPU NLL tolerance 2e-5; FP32 merge max-absolute tolerance 1e-4.",
            }
            if kind == "empirical":
                value["verification"]["denominators"] = denominators
        claims.append(value)

    narrow = "Only the declared fixed toy tasks and scoring rules; no general arithmetic, creativity, instruction following or natural-world safety claim."
    claim(
        "c1",
        "concept",
        "SFT teaches desired assistant answers from supervised demonstrations; instruction-following and safety are separate evaluation goals.",
        "T.5 opening and SFT definition",
        narrow,
        [
            (
                "sft-paper",
                "§3.1 Step1; §5.3 Models",
                "Demonstrations define supervised targets, while following user instructions can still be unsafe.",
            ),
            (
                "pku-card",
                "Human-Preference on Harmlessness and Helpfulness / Ranking of Responses",
                "Safety and helpfulness are judged separately; relative preference is not an absolute safety label.",
            ),
        ],
    )
    claim(
        "c2",
        "software",
        "The style/safety generator supplies three arithmetic answer styles and binary permission-policy demonstrations, with no unspecified-permission class.",
        "T.5 first recipe and permission/extra new-question paragraphs",
        "The offline generator recipe differs from the formal exchanged-addend and rule-family split; artificial missing-permission probes are additional manual data.",
        [
            (
                "generator",
                "generate_records(style/safety)",
                "Concise/vivid/one-field JSON targets and exactly True/False permission examples.",
            ),
            (
                "cli-source",
                "checks.style/checks.safety.manifest",
                "Both actual generator commands produced the documented data types.",
            ),
        ],
        "Same arithmetic content across three styles; True/False box15 target strings as written.",
        "Six-command smoke succeeded; generated source records and manifests agree.",
        "Read generated JSONL targets and family order; compare source targets and byte-plus-EOS target counts.",
    )
    claim(
        "c3",
        "software",
        "--task sft/--train/--steps select supervised updating; --max-length caps input units and --tokens caps generated units.",
        "T.5 initial 500-step commands and max-length/tokens explanation",
        "500 steps is the unexecuted full recipe in this review; one-step width16 CPU smoke verifies the interface without promising quality or 500-step convergence.",
        [
            ("train", "parser/main optimizer.step and pad_batch(max_length)", "Flags and update condition."),
            (
                "infer",
                "main generate args.tokens",
                "Generated unit cap; ByteTokenizer units are not visible-character counts.",
            ),
            ("cli-source", "commands and checks", "Both short training/evaluation interfaces actually ran."),
        ],
        "Valid positive SFT updates and capped generation.",
        "One CPU update for style and safety; all six commands exit0 and generation lengths <=2 in the limited smoke.",
        "Width16, layer1, batch2, max_length256, steps1, generation cap2; this intentionally shortened smoke is distinct from full formal runs.",
    )
    claim(
        "c4",
        "software",
        "SFT evaluation maps row0 to the first input row, preserves raw-ID exactness, separately requires EOS for completed exactness, and counts only effective answer targets.",
        "T.5 samples, skipped, effective_tokens and mean_token_nll paragraphs",
        "Only current evaluate.py/TinyLM byte-tokenizer behavior; unselected/skipped records are reported separately and are never counted correct.",
        [
            (
                "evaluate",
                "answer_sample/evaluate/main",
                "Raw ID identity, whitespace/control retention, EOS completion, row numbering, actual denominators and default20/all.",
            ),
            ("data", "render_chat", "Only assistant content and EOS are labels; user/context positions are -100."),
        ],
        "Normal '5'+EOS completed exact; '5' without EOS only exact; space/control variants fail.",
        "All four strict cases match; smoke rows and targets agree with original validation JSONL; effective target sums equal manifests.",
        "Count len(answer.encode('utf-8'))+1 and compare all non--100 label counts; no question positions are counted.",
    )
    claim(
        "c5",
        "empirical",
        "The formal arithmetic content baseline scores 0/8 validation and 0/7 test.",
        "T.5 paragraph beginning 上面500步的離線入口之外",
        narrow,
        [
            (
                "formal-style",
                "run_style content_training/content_evaluation",
                "New arithmetic base, 1000 updates, width64/layers2.",
            ),
            (
                "cpu-source",
                "accounting.arithmetic; cpu_replay content/*",
                "Independently reconstructed splits and replayed every held-out generation.",
            ),
        ],
        "49 training rows/28 families, 8 and7 held-out rows; 141568 parameters; zero correct.",
        "CPU exact raw IDs equal formal CUDA records; 0/8 and0/7, both EOS1.",
        "Complete records, no filtering. Formal config seed42, lr0.003, batch16, context128; report retains original source revision and training history.",
        {
            "train_records": 49,
            "train_families": 28,
            "validation_records": 8,
            "test_records": 7,
            "validation_answer_targets": 16,
            "test_answer_targets": 14,
            "updates": 1000,
            "batch": 16,
            "seed": 42,
        },
    )
    claim(
        "c6",
        "software",
        "The formal default-style branches use 450 updates each and the conditional style/date branch uses 1000, with common lr0.003/batch16/seed42/context128.",
        "T.5 paragraph beginning 從這份相同起點",
        "This is the recorded formal recipe and exact sampling reconstruction, not a new CPU long-training run; all three start from the declared arithmetic content base.",
        [
            (
                "formal-style",
                "run_style",
                "copy.deepcopy(content), reset seed, fixed450/1000 schedules and conditional dates.",
            ),
            ("common", "fit_lm/new_lm", "Fixed batch/lr and effective target accumulation."),
        ],
        "Conditional train/validation/test 185/28/27, common141568 model parameters.",
        "Reconstructed all manifests/hashes; counts185/28/27; target sums3783/564/546; default cumulative targets16591/318991, conditional324973.",
        "Replayed Python Random(42).choices for each scheduled batch, byte-answer+EOS lengths; all cumulative totals equal formal records.",
    )
    claim(
        "c7",
        "empirical",
        "The 27 conditional-style test records have only three full matches, all missing-date clarifications, and all terminate with EOS.",
        "T.5 conditional model performance paragraph",
        narrow,
        [
            (
                "cpu-source",
                "cpu_replay conditional/test",
                "All27 raw generated ID arrays match formal run and strict exactness is recomputed.",
            ),
            ("behavior", "_style_metrics/run_style", "Rubric checks each requested style separately."),
        ],
        "3/27 full matches; EOS27/27; matching rows are missing-date queries.",
        "CPU replay reproduces3/27 with27/27 EOS; supplied-date3/3 fail, while missing-date3/3 pass.",
        "Three styles each7 arithmetic cases plus6 paired date records; check complete expected/generated ID sequences rather than only decoded text.",
        {
            "arithmetic_test_per_style": 7,
            "date_test_records": 6,
            "total_test_records": 27,
            "effective_answer_targets": 546,
            "seed": 42,
            "conditional_updates": 1000,
        },
    )
    claim(
        "c8",
        "software",
        "The formal box data is deduplicated and split by eight complete arithmetic-rule families into102/17/17 rows, and the mixed branch adds49 arithmetic rows.",
        "T.5 formal-box recipe and9.8 split reference",
        "The primary templates remain shared across sides; rule-family disjointness does not establish arbitrary new-language generalization.",
        [
            (
                "formal-safety",
                "_safety_records/run_safety",
                "identifier%8 family, split before training and mixed training composition.",
            ),
            ("common", "split_records", "Dedup by full serialized record and keep family descendants together."),
        ],
        "Six/one/one families, 102/17/17 rows,466 targets on each held-out side.",
        "All reconstructed JSONL fingerprints equal formal manifests; no families cross sides; mixed training151 rows.",
        "Source24 boxes repeat eight arithmetic rules; dedup168 generated rows to136 unique records before split.",
    )
    claim(
        "c9",
        "numeric",
        "The two900-update safety schedules accumulate394888 and275389 effective answer targets.",
        "T.5 paragraph beginning 兩支訓練共讀到",
        "Counts are batch-sampled assistant-byte-plus-EOS targets; equal steps and batch size do not mean equal supervised text volume.",
        [
            (
                "cpu-source",
                "accounting.training.safety-only/safety-mixed",
                "Transparent reconstruction of every sampled answer length.",
            ),
            ("common", "fit_lm sampler and effective_tokens", "Counts all nonignored labels in sampled batches."),
        ],
        "394888 versus275389.",
        "Exact reconstruction gives394888 versus275389.",
        "For each900 updates, Random(42).choices draws16 rows; sum UTF-8 answer-byte length+EOS, with102 versus151 training records.",
    )
    claim(
        "c10",
        "empirical",
        "The box test matches15/17 for safety-only and16/17 for mixed; all three denied requests contain the fixed refusal phrase and no normal record does.",
        "T.5 final safety results",
        narrow
        + " Refusal detector is a literal template substring, while normal completion requires exact expected IDs.",
        [
            (
                "cpu-source",
                "cpu_replay safety-only/test and model/test",
                "Replays every record, recomputes exact and refusal-group counts.",
            ),
            (
                "formal-safety",
                "_safety_evaluations",
                "Three denied versus14 normal records, separate exact correctness and refusal audit.",
            ),
        ],
        "15/17 and16/17; appropriate refusal3/3 both; over-refusal0/14 both.",
        "Every CPU generated ID matches formal CUDA records; all17 terminate for each branch, normal exact12/14 and13/14.",
        "Explicitly recomputed refusal groups from same disjoint test data and literal phrase 無法提供.",
        {
            "test_records_each": 17,
            "test_answer_targets_each": 466,
            "denied_records": 3,
            "normal_records": 14,
            "updates_each": 900,
            "batch": 16,
            "seed": 42,
        },
    )
    claim(
        "c11",
        "empirical",
        "The mixed safety model fails all six paraphrase probes, including the specified unknown-count query yielding6.",
        "T.5 safety inference command and performance caveat",
        "Six probes cover only missing-count and document-color wording changes; toy template failure is not a broad language/safety benchmark.",
        [
            ("cpu-source", "cpu_replay safety/paraphrases", "All six generated ID sequences reproduced exactly."),
            (
                "formal-safety",
                "run_safety paraphrases",
                "Only wording of three unknown and three injection cases changes; model stays fixed.",
            ),
        ],
        "0/6 with6/6 EOS; 盒子1 unknown-count paraphrase ->6.",
        "CPU replay0/6, all EOS, exact example output6.",
        "Read every changed prompt, expected response, generated IDs and termination; answers including red/rn are preserved.",
        {
            "unknown_probes": 3,
            "document_probes": 3,
            "total_probes": 6,
            "effective_answer_targets": 117,
            "seed": 42,
            "updates_before_frozen_eval": 900,
        },
    )
    claim(
        "c12",
        "concept",
        "LoRA freezes base weights but still computes through them, trains low-rank corrections and can merge the correction into the original matrix.",
        "T.5 LoRA introduction and merge/switch explanation",
        "Reduced trainable parameter count does not imply proportional total memory, runtime, model-size reduction or equivalent quality; this project's11 layer placements extend beyond the paper's attention-only empirical scope.",
        [
            (
                "lora-paper",
                "§4.1 eq.(3), alpha/r paragraph, merge; §4.2 benefits/limits",
                "W0+alpha/r BA retains the base forward path and permits one explicit merge.",
            ),
            ("alignment", "LoRALinear.forward/merged_weight", "Same additive/scaled formula in this implementation."),
        ],
    )
    claim(
        "c13",
        "numeric",
        "Each formal adapter trains9504 added parameters while the141568 original model parameters remain frozen.",
        "T.5 paragraph beginning 每套adapter各更新450次",
        "Training path _add_lora freezes the whole base; inference loader's requires_grad flags alone are not evidence of the training freeze.",
        [
            (
                "behavior",
                "_add_lora/_fit_adapter/_base_state",
                "Whole-model freeze before inserting eleven trainable rank4/alpha4 paths; fingerprint check after training.",
            ),
            (
                "cpu-source",
                "lora_mechanism/base_tensor_identities",
                "Counts and two-step freeze probe plus public adapter compatibility.",
            ),
        ],
        "9504 trainable corrections,141568 base parameters.",
        "Exactly9504 and141568; two CPU optimizer updates leave all base tensor bytes unchanged.",
        "Six attention matrices contribute6*4*(64+64)=3072, four64<->256 FFN matrices4*4*(64+256)=5120, output64->264 adds4*(64+264)=1312; sum9504. Base total directly counted, not inferred from file bytes.",
    )
    claim(
        "c14",
        "empirical",
        "The vivid LoRA test rubric passes6/7 versus7/7 for full SFT, while both remain0/7 arithmetic correct.",
        "T.5 LoRA seven-test comparison",
        narrow
        + " Style correctness means presence of the fixed building-block phrase, not free-form apt metaphor quality.",
        [
            (
                "cpu-source",
                "cpu_replay lora-vivid/test and full-sft/test",
                "CPU replays every raw ID and independently applies current rubric matching the original counts.",
            ),
            (
                "behavior",
                "_style_metrics/run_lora",
                "Same base, style data and450 updates; all original weights frozen only in LoRA.",
            ),
        ],
        "Fixed-vivid style6/7 versus7/7; arithmetic0/7 both; base also0/7.",
        "All original generated IDs and rubric totals reproduced, including malformed UTF-8 replacement output causing the single style failure.",
        "Each49/8/7 arithmetic split; vivid450updates reads318991 targets for both LoRA and full SFT. EOS7/7 is separately checked.",
        {
            "train_records": 49,
            "validation_records": 8,
            "test_records": 7,
            "test_answer_targets": 308,
            "updates_each": 450,
            "effective_training_targets_each": 318991,
            "rank": 4,
            "alpha": 4,
            "seed": 42,
        },
    )
    claim(
        "c15",
        "software",
        "Adapter loading requires the actual original FP32 base fingerprint; matching shape is insufficient, and merged weights already contain the correction.",
        "T.5 adapter files, base fingerprint and omit--adapter for merged explanation",
        "Applies to this repository's native/lora-v1 loader and fixed adapters; numerical merge/switch agreement is a forward consistency check, not a quality test.",
        [
            (
                "adapters",
                "base_state_sha256/load_lora_adapter",
                "Hashes names/shapes/FP32 bytes and checks actual model against original state before mutation.",
            ),
            ("alignment", "merged_weight", "Exactly adds scaledBA once."),
            (
                "cpu-source",
                "lora_mechanism and cli",
                "Valid adapters accepted; same-shape modified base rejected; forward merge and A/B/A tested.",
            ),
        ],
        "Valid base accepted, changed same-shape base rejected; A/B/A returns exactly; FP32 merge tolerance1e-4.",
        f"Valid fingerprint4cf7b1e...b658d98; changed base rejected. A/B/A max error0; CPU merge max error{audit['lora_mechanism']['merge_max_difference']}, below1e-4.",
        "Public original base is loaded separately for each adapter. Source alpha/rank scaling compared with original LoRA eq3; merging then adding again represents an additional nonzero delta.",
    )
    claim(
        "c16",
        "empirical",
        "The public CPU2+2 raw-adapter examples emit3 and3，像把兩組積木合在一起再數。 with EOS and valid control IDs.",
        "T.5 public style/lora download and CPU example paragraph",
        "One fixed greedy example, not the held-out seven-item denominator; both outputs are wrong arithmetic despite style change.",
        [
            (
                "cpu-source",
                "cli",
                "Personally executed both complete infer.py commands using manifest-pinned public downloads.",
            ),
            (
                "weight-source",
                "files/public per-tensor records",
                "All11 public weights match SHA and original-source export lineage.",
            ),
            ("fetch-models", "fetch_model token=False/revision/SHA", "Public-only pinned download source."),
        ],
        "Concise3; vivid fixed phrase starting3; normalEOS and base-fingerprint check.",
        "Both outputs exactly match original student-check media/raw-adapter record; EOS true, no illegal special markers, fingerprint verified.",
        "Actual HF pinned style23b58c... and lora1bfb0e...; tokens96/defaulttemperature0, CPU. 2+2=4 independently confirms both answers wrong.",
        {
            "examples_per_adapter": 1,
            "adapters": 2,
            "generation_cap": 96,
            "decoding": "greedy temperature0",
            "device": "cpu",
        },
    )
    claim(
        "c17",
        "software",
        "The100-row PKU source archive is public CC-BY-NC-4.0 material; this project's derived pilot samples/weights are excluded from its public student package and its report is aggregate only.",
        "T.5 PKU source/publication paragraphs",
        "Public source data and private derived artifacts are distinct. This review reconstructed source selection and verified the original execution receipt's fingerprints/program; it did not retrieve or infer private pilot weights.",
        [
            (
                "pku-card",
                "YAML license and Dataset Summary",
                "Pinned source's noncommercial dataset license and label scope.",
            ),
            (
                "cc-license",
                "§1 NonCommercial; §2(a)(1); §3(a)",
                "Attribution/noncommercial sharing permissions and conditions; does not require all sharing to be private.",
            ),
            ("fetch-assets", "main", "Existing selected source archive entry point."),
            ("asset-integrity", "unpack_asset", "Archive contains hash-verified source and attribution files."),
            (
                "cpu-source",
                "pku_source",
                "Raw100 SHA c4a88... matches manifest; selected60 and exact48/6/6 JSONL hashes match original receipt; public weight list contains no pku pilot.",
            ),
        ],
        "Public original100 rows and attribution, distinct private pilot exports.",
        "Original README re-fetch hash equals archive source; raw100 and reconstructed derived split hashes match source manifest/formal record/original CPU receipt; public weights contain only toy safety versions.",
        "Inspect original receipt, read-only probe source, formal result/probe hashes,4 file hashes and support scope. No private checkpoint access or private-text generation in this review; existing receipt is retained as original execution provenance, not a review verdict.",
    )
    claim(
        "c18",
        "numeric",
        "The two exercise JSON answers both equal0+5=5 and satisfy the extra answer/explanation schema; only the vivid one adds the stated analogy.",
        "T.5 closing exercise",
        "Exercise's manually added two-field schema differs from generator's one-field JSON targets; metaphor usefulness is the described human rubric.",
        [
            ("cpu-source", "exercise", "Parsed both written examples, checked exact field set and integer answer."),
            (
                "generator",
                "generate_records style=json",
                "Generator only produces answer, keeping the exercise a separately stated target.",
            ),
        ],
        "5 and5, both answer/explanation keys.",
        "Both parse and pass exact schema and arithmetic checks; vivid explanation relates empty plusfive-block boxes tofive blocks.",
        "Compute0+5=5; JSON parsing and field equality executed, manual text inspection verifies useful metaphor rather than response length alone.",
    )
    ids = [item["id"] for item in claims]
    report = {
        "schema_version": 1,
        "review_stage": "technical",
        "lesson_id": "T.5",
        "source": "course/training.md#T.5",
        "reviewer_task": "/root/fact_finish_t_5",
        "reviewer_context": "fresh",
        "source_sha256": digest(BASE / f"{PREFIX}-source.txt"),
        "figure_sha256": {},
        "verdict": "pass",
        "claims": claims,
        "sources": sources,
        "artifacts": artifacts,
        "issues": [],
        "checks": {
            "factual_accuracy": {
                "status": "pass",
                "details": "Personally read complete T.5 source and its explicit necessary T.3/T.4/8.1/8.3/8.4/8.6/8.8/9.1/9.6/9.8 prerequisites. Every scoped claim is supported; independent public CPU replay agrees with original records.",
                "claim_ids": ids,
            },
            "numeric_verification": {
                "status": "pass",
                "details": "389 complete persisted synthetic generated records inspected;219 held-out records personally replayed on CPU. Every manifest hash, effective target total, count, refusal split, style rubric and raw generated ID checked; exercise and9504 count independently recomputed.",
                "claim_ids": [id for id in ids if id not in ("c1", "c12")],
            },
            "figure_consistency": {
                "status": "not_applicable",
                "details": "T.5 contains no embedded SVG or other figure; linked prerequisite illustrations are not evidence claimed in this review.",
                "claim_ids": [],
            },
            "source_verification": {
                "status": "pass",
                "details": "Read original InstructGPT/LoRA PDFs, pinned PKU dataset README and CC legal code; saved all read bytes. Retrieved original formal-code versions, verified their report hashes, inspected changes and rescored current rubric. Public weights traced to fixed HF commit and export/formal source hashes; private execution receipt provenance checked without accessing private weights.",
                "claim_ids": ids,
            },
            "limitations": {
                "status": "pass",
                "details": "Distinguishes offline500-step proposed recipe, formal single-seed toy results, per-example CPU CLI, strict EOS/content scoring and narrow fixed style/refusal rubric. No CPU-to-GPU speed claim, universal quality/safety guarantee, direct private-weight retrieval claim or parameter-count-to-total-memory inference.",
                "claim_ids": ids,
            },
        },
        "source_read_receipt": {
            "read_complete": True,
            "snapshot_path": f"docs/technical-reviews/artifacts/{PREFIX}-source.txt",
            "snapshot_sha256": digest(BASE / f"{PREFIX}-source.txt"),
            "first_numbered_section": "T.1",
            "intro_required_for_this_section": False,
            "inspection_note": "T.5 is not the source's first numbered section; read-time snapshot is immutable and report SHA comes from those already-read bytes, never refreshed from the latest source by this builder.",
        },
    }
    (ROOT / "docs/technical-reviews/T.5.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Wrote independent T.5 pass: {len(claims)} claims; {len(sources)} sources; {len(artifacts)} artifacts; source SHA {report['source_sha256']}"
    )


if __name__ == "__main__":
    main()
