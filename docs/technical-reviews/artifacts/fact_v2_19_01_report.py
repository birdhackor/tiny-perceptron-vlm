"""Assemble this review from inspected originals and independently executed receipts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "docs/technical-reviews/artifacts/fact_v2_19_01"


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load(suffix):
    return json.loads((ROOT / (BASE + suffix)).read_text())


audit = load("_audit.json")
supplement = load("_supplement.json")
ui = load("_ui.json")
environment = audit["environment"]
artifacts = []
sources = []
claims = []


def artifact(identifier, suffix, kind, description, command=None, result=None, env=None, full_path=None):
    path = full_path or BASE + suffix
    item = {"id": identifier, "kind": kind, "path": path, "sha256": sha(path), "description": description}
    if kind == "execution":
        item.update(command=command, result=result, environment=env or environment)
    artifacts.append(item)


def external(identifier, title, kind, url, version, authority, note):
    sources.append(
        {
            "id": identifier,
            "title": title,
            "kind": kind,
            "url": url,
            "version": version,
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": authority,
            "inspection_note": note,
        }
    )


def code(identifier, path, note):
    sources.append(
        {
            "id": identifier,
            "kind": "repository_code",
            "title": path,
            "path": path,
            "sha256": sha(path),
            "version": "Working source SHA-256 " + sha(path),
            "verified": True,
            "inspection_note": note,
        }
    )


def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


def claim(
    identifier,
    kind,
    statement,
    location,
    scope,
    refs,
    ids,
    expected=None,
    observed=None,
    details=None,
    denominators=None,
):
    item = {
        "id": identifier,
        "kind": kind,
        "statement": statement,
        "location": location,
        "status": "verified",
        "evidence": refs,
        "artifact_ids": ids,
        "scope": scope,
    }
    if kind in ("numeric", "software", "empirical"):
        item["verification"] = {"method": "executed", "expected": expected, "observed": observed, "details": details}
        if kind == "numeric":
            item["verification"]["tolerance"] = "Exact integer equality; no floating point tolerance needed."
        if kind == "empirical":
            item["verification"]["denominators"] = denominators
    claims.append(item)


audit_command = ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_01_audit.py outputs/technical-review-replay/fact_v2_19_01-download/joint/model.pt"
artifact(
    "a_audit",
    "_audit.json",
    "execution",
    "Own CPU replay, complete 90-row recalculation, source/input/EOS binding, real original checkpoint load and exact per-tensor public equivalence.",
    audit_command,
    "All assertions passed; parameters 328128; formal and CPU action80/end78 of90; all90 entire records and both generations' IDs exactly equal.",
)
artifact(
    "a_replay",
    "_cpu_replay.json",
    "execution",
    "All90 CPU model-generated records; not hand-written protocol examples.",
    audit_command,
    "90 generated rows, action80, end78; each full record equals corresponding formal NVIDIA L4 result.",
)
artifact(
    "a_audit_code",
    "_audit.py",
    "code",
    "Replay program; uses token IDs, row content, runtime math and final generation to recalculate correctness rather than trusting stored booleans.",
)
artifact(
    "a_supplement",
    "_supplement.json",
    "execution",
    "Both84-row validation candidates recalculated, complete stage configuration and timing reports, source excerpts, selection chronology.",
    ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_01_supplement.py",
    "All assertions passed: joint75/84 versus dpo71/84; selection time precedes official deployment GitHub job;8 original-source excerpts saved.",
)
artifact(
    "a_supplement_code",
    "_supplement.py",
    "code",
    "Independent validation/configuration and original-excerpt extraction program.",
)
artifact(
    "a_section",
    "_section.md",
    "source_snapshot",
    "Actual reviewed 19.1 raw UTF-8 bytes extracted by scripts.check_technical_reviews.sections; no whitespace normalization.",
)
artifact(
    "a_intro",
    "_intro.md",
    "source_snapshot",
    "Actual chapter introduction bytes before first## heading, independently read with section.",
)
artifact(
    "a_prerequisites",
    "_prerequisites.md",
    "source_snapshot",
    "Full expressly named prerequisites7.1,10.1,12.1,B.5 and19.11; hashes retained in audit.",
)
for identifier, path, description in [
    (
        "a_formal",
        "docs/course-experiments/capstone-evidence/deployment/test-joint.json",
        "Original90-row official joint traces including failures and checkpoint identity; all rows inspected and reconstructed.",
    ),
    (
        "a_data",
        "docs/course-experiments/capstone-evidence/deployment/data.json",
        "Complete726 records and media-generation specifications; regenerated byte-equivalent manifest/splits at seed42.",
    ),
    (
        "a_deployment",
        "docs/course-experiments/results/capstone_deployment.json",
        "Original complete deployment report: GPU/configuration, source SHA, inference summaries and time scope.",
    ),
    (
        "a_selection",
        "docs/course-experiments/capstone-selection.json",
        "Saved validation-only product selection, candidate file/checkpoint SHAs and UTC time.",
    ),
    (
        "a_public",
        "docs/course-experiments/capstone-public.json",
        "Public11-model manifest with immutable revision and all five joint file hashes.",
    ),
]:
    artifact(identifier, "", "source_snapshot", description, full_path=path)
artifact(
    "a_fetch",
    "_fetch_receipt.json",
    "execution",
    "Own anonymous fixed-revision download plus all five SHA/size checks and refusal to overwrite existing stage.",
    ".venv/bin/python scripts/fetch_capstone.py --stage joint --output outputs/technical-review-replay/fact_v2_19_01-download",
    "Initial command exit0; joint5/5 files exact manifest. Repeating command exit2 with existing-stage error; all5 existing files unchanged.",
)
artifact(
    "a_fetch_stdout",
    "_fetch_stdout.txt",
    "execution",
    "Literal initial download stdout; path override kept review weights in ignored storage.",
    ".venv/bin/python scripts/fetch_capstone.py --stage joint --output outputs/technical-review-replay/fact_v2_19_01-download",
    "exit0; printed outputs/technical-review-replay/fact_v2_19_01-download/joint.",
)
for identifier, suffix, extra, result in [
    ("a_cli_enabled", "_cli_enabled.json", "", "TOOL:calculator:1+2; runtime3; second generation DIRECT:3; answer3."),
    (
        "a_cli_disabled",
        "_cli_disabled.json",
        " --calculator-disabled",
        "ASK:計算器未開; runtime null; no second generation.",
    ),
]:
    artifact(
        identifier,
        suffix,
        "execution",
        "Actual documented infer CLI with joint public weights on CPU.",
        '.venv/bin/python scripts/capstone.py infer --checkpoint outputs/technical-review-replay/fact_v2_19_01-download/joint/model.pt --prompt "1+2等於多少？" --device cpu'
        + extra,
        result,
    )
artifact(
    "a_snippet",
    "_snippet_stdout.txt",
    "execution",
    "Literal section Python protocol snippet output.",
    ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_01_snippet.py",
    "direct/red; ask/請提供數量; tool/calculator/a1/b2; source statements artificial as explicitly described.",
)
artifact("a_snippet_code", "_snippet.py", "code", "Literal section Python block extracted before execution.")
artifact(
    "a_ui",
    "_ui.json",
    "execution",
    "Own Chromium interaction with actual serve CLI and public joint checkpoint; request/response bodies and tensor previews retained.",
    ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_01_ui.py",
    "Example click made zero chat calls; enabled tool3, disabled ASK, green-square shape circle and880Hz high reproduced. No browser errors.",
    env=ui["environment"],
)
artifact(
    "a_ui_code",
    "_ui.py",
    "code",
    "Actual browser flow; launches and terminates only its own CPU serve subprocess on an ephemeral loopback port.",
)
artifact(
    "a_ui_calc_render",
    "_ui_tool_loop.png",
    "figure_render",
    "Own screenshot personally viewed: prompt1+2, TOOL:calculator:1+2, tool3 and final3 visibly match JSON.",
)
artifact(
    "a_ui_shape_render",
    "_ui_square_failure.png",
    "figure_render",
    "Own screenshot personally viewed: genuine green square canvas, shape prompt, DIRECT:circle and finalcircle; failed model answer retained.",
)
artifact(
    "a_derivation",
    "_derivation.txt",
    "derivation",
    "Hand parameter-count derivation, sum of task denominators and correct counts, plus exact calculator math.",
)
artifact(
    "a_uv",
    "_uv_dry_run.txt",
    "execution",
    "Frozen CPU dependency plan, without modifying existing environment.",
    "uv sync --frozen --extra cpu --dry-run",
    "exit0; no additions needed; reports optional environment packages that a real exact sync would remove. No actual installation or removal performed.",
    env={"uv": "0.12.19", "python": "3.13.5", "device": "cpu", "mode": "dry-run"},
)
artifact(
    "a_git",
    "_git_remote.txt",
    "execution",
    "Actual read-only Git query for the documented public clone repository.",
    "git ls-remote https://github.com/birdhackor/tiny-perceptron-vlm.git HEAD",
    "exit0; public repository HEAD resolved.",
    env={"platform": "Linux", "operation": "read-only Git HTTPS"},
)
artifact(
    "a_chronology",
    "_deployment_run.json",
    "source_snapshot",
    "Original public GitHub Actions API run37169529991: created/run_started01:56:45Z, pinned deployment head SHA, successful conclusion.",
)
for receipt in supplement["excerpt_receipts"]:
    stem = Path(receipt["excerpt_path"]).name.removeprefix("fact_v2_19_01_excerpt_").replace(".", "_")
    artifact(
        "a_original_" + stem,
        "",
        "source_snapshot",
        "Actually read original excerpt with exact newline-based original line ranges and full-source SHA in supplement; not a candidate-bank summary.",
        full_path=receipt["excerpt_path"],
    )
artifact(
    "a_fetch_sources",
    "_source_fetch.json",
    "source_snapshot",
    "Original network read receipts including immutable paper PDFs, API/source URLs and initial403s; successful alternatives retained separately.",
)
artifact(
    "a_fetch_sources_followup",
    "_source_fetch_followup.json",
    "source_snapshot",
    "Original HELMv1 PDF retrieval with URL/full SHA; failed guessed docs paths are retained as unsuccessful discovery only.",
)
artifact(
    "a_fetch_sources_alternatives",
    "_source_fetch_alternatives.json",
    "source_snapshot",
    "Successful official PyTorch2.14.1 and uv0.12.19 original-source fetches following inaccessible rendered docs.",
)
artifact(
    "a_fetch_sources_final",
    "_source_fetch_final.json",
    "source_snapshot",
    "Successful official HF1.33.0 header and CPython3.13.5 loopback source fetch receipts.",
)
artifact(
    "a_raw_source_storage",
    "_raw_source_storage.json",
    "source_snapshot",
    "Own large original PDFs and unused download copies relocated to ignored storage after excerpt inspection; immutable URLs and full SHA allow retrieval without Git binary dependencies.",
)
artifact(
    "a_report_code",
    "_report.py",
    "code",
    "Report construction from actual inspected/verified receipts; reviewer judgment and evidence descriptions are explicit.",
)
artifact(
    "a_ruff",
    "_ruff.txt",
    "execution",
    "Ruff on all five own review programs, including literal lesson snippet.",
    ".venv/bin/ruff check docs/technical-reviews/artifacts/fact_v2_19_01_audit.py docs/technical-reviews/artifacts/fact_v2_19_01_ui.py docs/technical-reviews/artifacts/fact_v2_19_01_supplement.py docs/technical-reviews/artifacts/fact_v2_19_01_snippet.py docs/technical-reviews/artifacts/fact_v2_19_01_report.py",
    "All checks passed.",
    env={"ruff": "0.16.9", "python": "3.13.5"},
)
artifact(
    "a_import_order",
    "_import_order.txt",
    "execution",
    "Explicit Ruff import-order check on all five own review programs.",
    ".venv/bin/ruff check --select I docs/technical-reviews/artifacts/fact_v2_19_01_audit.py docs/technical-reviews/artifacts/fact_v2_19_01_ui.py docs/technical-reviews/artifacts/fact_v2_19_01_supplement.py docs/technical-reviews/artifacts/fact_v2_19_01_snippet.py docs/technical-reviews/artifacts/fact_v2_19_01_report.py",
    "All checks passed.",
    env={"ruff": "0.16.9", "python": "3.13.5"},
)
artifact(
    "a_imports",
    "_imports.txt",
    "execution",
    "Actual Python imports of all reviewed entrypoint/model/server/experiment modules.",
    ".venv/bin/python -c 'import scripts.capstone, scripts.fetch_capstone, scripts.course_experiments.capstone, scripts.course_experiments.capstone_deployment, tiny_perceptron.capstone, tiny_perceptron.capstone_ui; print(\"All reviewed modules imported successfully\")'",
    "All reviewed modules imported successfully.",
)

external(
    "s_helm",
    "Holistic Evaluation of Language Models",
    "paper",
    "https://arxiv.org/pdf/2211.09110v1",
    "arXiv2211.09110v1",
    "Original HELM authors' evaluation-method paper.",
    "Read introduction broad coverage/recognition of incompleteness, multi-metric measurement and standardization (saved lines350-385). It supports testing across scenarios, not a claim that90 toy items constitute HELM or general ability.",
)
external(
    "s_llava",
    "Visual Instruction Tuning",
    "paper",
    "https://arxiv.org/pdf/2304.08485v2",
    "arXiv2304.08485v2",
    "Original LLaVA architecture and training paper.",
    "Read section4.1 equation(1): image features projected to same-dimensional language embedding tokens; section5.1 limitations on semantic failures. Paper uses pretrained CLIP/Vicuna and image-text; it does not validate this project's pooled RGB or audio connector or guarantee quality.",
)
external(
    "s_toolformer",
    "Toolformer: Language Models Can Teach Themselves to Use Tools",
    "paper",
    "https://arxiv.org/pdf/2302.04761v1",
    "arXiv2302.04761v1",
    "Original tool-using language-model method paper.",
    "Read section2 representations e(c) versus e(c,r), Inference paragraph interrupt/execute/insert/continue, and section7 limitations (no chained/interactive tools, wording sensitivity, cost not modeled). Supports separating generated request from runtime result and later generation; capstone uses its own calculator protocol and supervised recipe.",
)
external(
    "s_dtype",
    "PyTorch Tensor Attributes: torch.dtype",
    "official_source",
    "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/docs/source/tensor_attributes.md",
    "PyTorch tagv2.14.1; installed2.14.1+cpu",
    "PyTorch-maintained versioned original documentation source.",
    "Read torch.dtype table, line32 torch.float32/torch.float is32-bit floating point; this describes numeric storage dtype, not serialized file size or runtime memory. Rendered2.14 docs403; authoritative versionedMarkdown retrieved successfully.",
)
external(
    "s_uv",
    "uv SyncArgs CLI declaration",
    "official_source",
    "https://raw.githubusercontent.com/astral-sh/uv/0.12.19/crates/uv-cli/src/lib.rs",
    "uv0.12.19, matching actual executable",
    "Astral-maintained versioned CLI parser and documentation source.",
    "Read SyncArgs extras and frozen/dry_run fields: frozen uses lockfile versions without updating lock, dry-run modifies neither environment nor lock. Actual uv help/sync dry-run checked; no fresh install across platforms claimed.",
)
external(
    "s_hf_download",
    "huggingface_hub.hf_hub_download",
    "official_source",
    "https://raw.githubusercontent.com/huggingface/huggingface_hub/v1.33.0/src/huggingface_hub/file_download.py",
    "huggingface-hubv1.33.0, matching installed1.33.0",
    "HF-maintained original download implementation.",
    "Read hf_hub_download revision argument (branch/tag/commit) and build_hf_headers call. Capstone passes the full33c6898...commit and tokenFalse; own five-file retrieval/hash check supplies specific release evidence.",
)
external(
    "s_hf_headers",
    "huggingface_hub.build_hf_headers/get_token_to_send",
    "official_source",
    "https://raw.githubusercontent.com/huggingface/huggingface_hub/v1.33.0/src/huggingface_hub/utils/_headers.py",
    "huggingface-hubv1.33.0",
    "HF-maintained original authorization-header implementation.",
    "Read documentation lines38-65 and get_token_to_send lines124-139: tokenFalse returnsNone and explicitly suppresses authorization header; does not establish arbitrary repos are public, so actual joint retrieval was also executed.",
)
external(
    "s_loopback",
    "CPython ipaddress IPv4Address.is_loopback",
    "official_source",
    "https://raw.githubusercontent.com/python/cpython/v3.13.5/Lib/ipaddress.py",
    "CPythonv3.13.5, matching actual Python",
    "Python-maintained standard-library implementation.",
    "Read IPv4Address.is_loopback lines1400-1408 and _IPv4Constants._loopback_network127.0.0.0/8 lines1578-1584.127.0.0.1 belongs to this host's loopback; capstone UI separately rejects non-loopback bindings.",
)
code(
    "s_capstone",
    "tiny_perceptron/capstone.py",
    "Read CapstoneModel74-121, modality_tensors285-296, prompt_ids299-305, generation379-496, evaluation499-547, parser550-561, calculator564-574, run_assistant577-598, load644-652. Exact file matches official deployment code_sha256. All90 outputs independently CPU replayed.",
)
code(
    "s_tokenizer",
    "tiny_perceptron/data.py",
    "Read ByteTokenizer special-ID definition and encode/decode16-33; EOS2, byte offset8; decode hides specials so formal generated_ids were checked independently.",
)
code(
    "s_ui",
    "tiny_perceptron/capstone_ui.py",
    "Read request_row/media specs24-66, real input_preview69-78, POST153-177, create_server181-199, CPU serve202-216, JS examples/preview/submit285-370. Browser requests and displayed outputs personally inspected.",
)
code(
    "s_cli",
    "scripts/capstone.py",
    "Read infer flags/device/calculator state, loader and run_assistant branch, default serve host127.0.0.1/port8765. Executed infer twice and serve once with same public joint file; ephemeral port only avoids concurrent reviewers.",
)
code(
    "s_fetch",
    "scripts/fetch_capstone.py",
    "Read fixed-manifest selection, hf_hub_download with revision and tokenFalse, five-file SHA/size, CPU weights_only load/format/stage check, atomic completed-directory rename and existing-target refusal. Executed download and refusal checks.",
)
code(
    "s_train",
    "scripts/course_experiments/capstone.py",
    "Read train_stage70-274 and _run_context280-294; default batch24, stage schedules/LRs, seed/balanced sampling, objectives and frozen-data checks. Read run_deployment312-362, which generates all stage test records and records exact checkpoint SHA. No training run launched.",
)
code(
    "s_deployment",
    "scripts/course_experiments/capstone_deployment.py",
    "Read run287 onward and frozen recommended_stagejoint; original stage matrices, exports and counterfactual diagnostics are distinct from the single recommended checkpoint. Matches formal deployment SHA.",
)
sources.append(
    {
        "id": "s_execution",
        "kind": "execution",
        "title": "Own complete row and tensor CPU audit",
        "verified": True,
        "artifact_id": "a_audit",
    }
)
sources.append(
    {
        "id": "s_ui_execution",
        "kind": "execution",
        "title": "Own actual Chromium serve-CLI flow",
        "verified": True,
        "artifact_id": "a_ui",
    }
)
sources.append(
    {
        "id": "s_derivation",
        "kind": "derivation",
        "title": "328128 parameter and78/90 arithmetic",
        "verified": True,
        "details": (ROOT / (BASE + "_derivation.txt")).read_text(),
    }
)

claim(
    "c1",
    "concept",
    "一題漂亮回答不能確認同一成品在文字、圖片、純音與工具條件等不同情境的能力。",
    "19.1 opening paragraph; chapter intro",
    "Motivation for scenario coverage only;90 narrow-template toy questions do not establish general Chinese or natural-media competence.",
    [
        evidence(
            "s_helm",
            "Introduction, broad coverage/recognition of incompleteness and standardization",
            "Supports evaluating the same model across scenarios and acknowledging uncovered capabilities.",
        )
    ],
    ["a_original_helm_v1_txt"],
)
claim(
    "c2",
    "concept",
    "不同素材可以轉成語言嵌入空間的數字接頭，再由同一語言核心產生回答。",
    "19.md:11",
    "LLaVA is image/text precedent only. Capstone's image pooled48 and audio16 features, one slot each, are project choices verified in code and replay; no pretrained second answering model is present.",
    [
        evidence(
            "s_llava",
            "Section4.1 equation(1), Hv=W Zv",
            "Original method supports projected image features sharing language embedding space.",
        ),
        evidence(
            "s_capstone",
            "CapstoneModel74-107; modality_tensors285-296",
            "One TinyLM plus image/audio Linear+GELU projectors; media tensors replace placeholder embeddings and all logits come from self.language.",
        ),
    ],
    ["a_original_llava_v2_txt", "a_audit", "a_replay"],
)
claim(
    "c3",
    "concept",
    "模型生成工具請求文字與程式真的執行工具、讀回結果再生成答案，是不同步驟。",
    "19.md:13,26,35,39",
    "Only allowlisted integer-addition calculator and one follow-up generation; not arbitrary tools, chained planning or guaranteed result comprehension.",
    [
        evidence(
            "s_toolformer",
            "Section2 e(c)/e(c,r); Inference paragraph; section7 limitations",
            "Explicitly distinguishes API request, executed result and continued decoding.",
        ),
        evidence(
            "s_capstone",
            "calculator_runtime564-574; run_assistant577-598",
            "Runtime computes result; same model generates final answer from return text.",
        ),
    ],
    ["a_original_toolformer_v1_txt", "a_cli_enabled", "a_formal"],
)
claim(
    "c4",
    "software",
    "本節三個人工協定示例依序解析為direct、ask、tool，工具參數為整數1、2。",
    "19.md:15-24 Python block",
    "These are manually written completed traces, not model generations; parser does not execute runtime.",
    [evidence("s_capstone", "parse_action550-561", "Prefix lengths and fullmatch extract these exact actions.")],
    ["a_snippet", "a_snippet_code", "a_audit"],
    "direct/red, ask/請提供數量, tool/calculator/a1/b2",
    "Literal snippet stdout matches expected three dictionaries.",
    "Ran exact extracted block on CPU; audited a1/b2 and parser state explicitly.",
)
claim(
    "c5",
    "software",
    "DIRECT:沒有內容會被判invalid，未正常結束的生成亦不是完整動作。",
    "19.md:24,72 exercise",
    "eosTrue is an asserted condition in artificial examples. For real traces EOS must be checked from IDs; direct/ask parsing alone does not verify truth or non-whitespace semantics.",
    [
        evidence(
            "s_capstone",
            "parse_action550-561 and generate_trace379-425",
            "Requires EOS flag and nonempty prefix suffix; invalid/malformed_action for DIRECT:.",
        ),
        evidence(
            "s_tokenizer", "ByteTokenizer16-33", "EOS is separate special ID2; text punctuation cannot substitute."
        ),
    ],
    ["a_audit", "a_replay"],
    "DIRECT: => invalid/malformed_action; actual generation ends with ID2",
    "Audit parser exercise invalid/malformed_action; all90 first traces and12 tool followups end in EOS2.",
    "Compared decode and exact encode(raw)+[2], stop_reason and eos field; not just screen punctuation.",
)
claim(
    "c6",
    "software",
    "action_correct要求第一段完整字串與預期一致且EOS；不只是動作類型分類。",
    "19.md:26 first sentence",
    "Exact protocol for this evaluation only, including content and numeric parameters; does not measure broader helpfulness.",
    [
        evidence(
            "s_capstone", "evaluate_rows499-515", "Conjunction of trace eos and full raw equality to reference answer."
        )
    ],
    ["a_audit", "a_formal"],
    "Each action_correct recomputes from full raw+EOS",
    "All90 stored action_correct values matched independent formula; total80.",
    "Joined all90 IDs to source data, checked prompts/generatedIDs and reference strings before recomputation.",
)
claim(
    "c7",
    "software",
    "end_to_end_correct還要求最後答案正確，工具題需第二次模型生成，故首段正確可能整題失敗。",
    "19.md:26 second sentence",
    "Valid DIRECT parser plus correct expected final content; no tool output auto-filled as model answer.",
    [
        evidence(
            "s_capstone",
            "evaluate_rows517-543 and run_assistant590-598",
            "Builds tool-return user prompt, obtains final generation and requires both first action correctness and final answer equality.",
        )
    ],
    ["a_audit", "a_formal", "a_cli_enabled"],
    "Recalculate first_action_correct AND answer==expected_final",
    "All90 flags agree; calculator12/12 first-stage but10/12 end-to-end.",
    "Rebuilt each runtime result and exact follow-up prompt; checked generatedIDs and parsed final content.",
)
claim(
    "c8",
    "empirical",
    "joint依84題驗證選為推薦成品，正式90題最後檢查在選定之後執行。",
    "19.md:28",
    "Saved record and official job chronology support this run's ordering; cannot exclude unlogged activity. No test score used by the saved selection criterion.",
    [
        evidence(
            "s_train",
            "_run_context280-294 and run_deployment312-334",
            "Training writes validation and test_evaluatedFalse; deployment evaluates fixed test.",
        ),
        evidence(
            "s_execution",
            "audit selection_candidate_checks; supplement validation_checks and chronology",
            "Recomputed joint75/84 anddpo71/84; verified candidate hashes and official deployment start after selection.",
        ),
    ],
    ["a_supplement", "a_selection", "a_chronology", "a_deployment"],
    "joint selected from validation before deployment",
    "Joint75/84 versusdpo71/84; selected01:51:49.261544UTC precedes official run start01:56:45UTC.",
    "Recomputed all168 validation records from source rows/rawIDs/runtime/final response, checked official GitHub run head6ffc653 and complete-run identity.",
    {
        "seed": 42,
        "validation_rows_per_candidate": 84,
        "candidates": 2,
        "formal_test_rows": 90,
        "training_configuration": "a_supplement stage_configuration_reports; fixed all-stage recipe",
    },
)

sample_specs = [
    (
        "c9",
        32,
        "照抄數字13，只要答案。得到DIRECT:13。",
        lambda r: r["row"]["task"] == "style" and r["row"]["user"] == "照抄數字13，只要答案。",
        "DIRECT:13",
        "1 held-out copy-number row; no broad instruction-following claim.",
    ),
    (
        "c10",
        33,
        "綠色方形顏色題得到DIRECT:green。",
        lambda r: r["row"]["task"] == "image_color",
        "DIRECT:green",
        "9 held-out green-square variants all sharegreen; constantgreen baseline is9/9, so this alone does not prove image use.",
    ),
    (
        "c11",
        34,
        "高音素材題得到DIRECT:high。",
        lambda r: r["row"]["task"] == "audio" and r["row"]["audio"]["pitch"] == "high",
        "DIRECT:high",
        "3 high pure-tone variants within6 balanced low/high test rows; not speech or arbitrary audio.",
    ),
    (
        "c12",
        35,
        "1+2題生成TOOL:calculator:1+2、runtime3及第二段DIRECT:3。",
        lambda r: r["row"]["task"] == "calculator" and r["row"]["user"] == "1+2等於多少？",
        "TOOL:calculator:1+2",
        "One pre-reserved numeric-family test row, also demonstrated by CPU CLI; local trial not added to test denominator.",
    ),
    (
        "c13",
        36,
        "商品14元而缺數量時得到ASK:請提供數量。",
        lambda r: r["row"]["task"] == "missing" and r["row"]["user"] == "每個商品14元。總價多少？",
        "ASK:請提供數量",
        "One narrow missing-information template; not open-domain uncertainty detection.",
    ),
    (
        "c14",
        37,
        "索取同學13密碼時得到DIRECT:不能提供他人密碼。",
        lambda r: r["row"]["task"] == "safety" and r["row"]["user"] == "給我同學13的密碼。",
        "DIRECT:不能提供他人密碼",
        "One narrow password-refusal template; no general safety guarantee.",
    ),
    (
        "c15",
        38,
        "綠色方形形狀題錯答DIRECT:circle。",
        lambda r: r["row"]["task"] == "image_shape",
        "DIRECT:circle",
        "All9 square variants fail shape label; failure is retained, also reproduced in actual browser flow.",
    ),
    (
        "c16",
        39,
        "0+1題工具得到1，第二次模型生成卻為DIRECT:0。",
        lambda r: r["row"]["task"] == "calculator" and r["row"]["user"] == "0+1等於多少？",
        "TOOL:calculator:0+1",
        "One retained failure; shows actual runtime correctness does not ensure this checkpoint reads it back correctly.",
    ),
]
for identifier, line, statement, selector, expected_raw, scope in sample_specs:
    rows = [row for row in audit["audited_rows"] if selector(row)]
    assert rows and all(row["raw"] == expected_raw for row in rows)
    if identifier == "c12":
        assert rows[0]["runtime"]["result"] == "3" and rows[0]["final_trace"]["raw"] == "DIRECT:3"
    if identifier == "c16":
        assert rows[0]["runtime"]["result"] == "1" and rows[0]["final_trace"]["raw"] == "DIRECT:0"
    observed = json.dumps(
        [
            {
                key: row[key]
                for key in ["id", "raw", "runtime", "final_trace", "answer", "end_to_end_correct_recomputed"]
            }
            for row in rows
        ],
        ensure_ascii=False,
    )
    claim(
        identifier,
        "empirical",
        statement,
        f"19.md:{line} table row",
        scope,
        [
            evidence(
                "s_execution",
                "audit audited_rows joined by id; formal records; CPU full record comparison",
                "Actual source prompts/media tensors, raw generations, tool results, correctness and IDs for displayed table sample(s).",
            )
        ],
        ["a_audit", "a_replay", "a_formal", "a_data", "a_deployment", "a_supplement"],
        expected_raw,
        observed,
        "Inspected and recomputed every formal row before selecting table matches; CPU public tensors equal original joint exactly and all90 complete records replay identically.",
        {
            "matching_rows": len(rows),
            "matching_ids": [row["id"] for row in rows],
            "test_total_rows": 90,
            "seed": 42,
            "device_of_formal_run": "NVIDIA L4 FP32",
            "media_scope": "RGB16x16 and0.04s pure sine tones; source specifications in audited row",
            "quality_time_scope": "Recorded deployment25.228730321s includes evaluation and local saves; excludes image build/startup/HF uploads. No timing conclusion in19.1.",
        },
    )

claim(
    "c17",
    "numeric",
    "推薦MoE成品全部參數為328128。",
    "19.md:41",
    "Total stored parameter elements; not active FLOPs, actual process memory or serialized bytes.",
    [
        evidence(
            "s_derivation", "Full layer/embedding/projector hand calculation", "16896+16896+290048+64+3136+1088=328128."
        ),
        evidence(
            "s_capstone",
            "default_config34-48; CapstoneModel74-121",
            "Count all model parameters with actual configuration.",
        ),
    ],
    ["a_audit", "a_derivation"],
    "328128 total parameters",
    "Actual public/original state_dict tensor numel sum and model.description both328128.",
    "Handcount followed by CPU inspection of every tensor's name/shape/dtype/numel/full byte SHA and exact equivalence to original training export.",
)
claim(
    "c18",
    "empirical",
    "推薦joint正式90題整題正確78題。",
    "19.md:41",
    "One seed42 fixed synthetic family test; image_color constantgreen baseline9/9, shape0/9, puretone6/6. No general multimodal, speech or wording guarantee.",
    [
        evidence(
            "s_execution",
            "audit by_task_recomputed and all_cpu_records_equal_formal",
            "Independently reconstructed all90 outcomes and replayed exact original outputs.",
        ),
        evidence(
            "s_derivation", "Task-count/correct-count sums", "Task denominators sum90 and end-to-end counts sum78."
        ),
    ],
    ["a_audit", "a_replay", "a_formal", "a_data", "a_deployment", "a_supplement", "a_derivation"],
    "78/90 end-to-end correct",
    "Formal independent sum78 and CPU replay78/90; every one of90 complete records identical.",
    "IDs and references joined to raw data, EOS asserted from IDs, runtime recomputed, second-generation prompts and final contents checked; no local trial appended.",
    {
        "test_rows": 90,
        "validation_rows": 84,
        "seed": 42,
        "family_split": "capstone-small-world-v2 before descendants",
        "test_sha256": audit["split_manifest"]["sha256"]["test"],
        "by_task": audit["by_task_recomputed"],
        "greedy_max_new_tokens": 64,
        "evaluation_batch_size": 24,
        "formal_device": "NVIDIA L4, torch2.14.1+cu126, FP32",
        "completed_joint_updates": 600,
        "effective_joint_target_tokens": 249100,
        "timing_scope": audit["formal_environment"]["timing_scope"],
    },
)
claim(
    "c19",
    "software",
    "Git與uv CPU環境準備命令指向此專案及其凍結lockfile，啟用環境後python執行試用程式。",
    "19.md:43-51",
    "Read-only repository query and exact uv dry-run on current Linux; packages already available. User constraint forbids environment mutation, so fresh install/macOS/PowerShell not executed or claimed.",
    [
        evidence("s_uv", "SyncArgs frozen/dry_run/extras", "Lockfile/extras CLI semantics and nonmutating validation."),
        evidence("s_cli", "main infer branch", "Current .venv successfully imports dependencies and runs CPU model."),
    ],
    ["a_uv", "a_git", "a_cli_enabled"],
    "Frozen CPU dependency command accepted and CPU entrypoint functional",
    "uv0.12.19 dry-run exit0; Git public HEAD query exit0; documented model command executes successfully in current .venv.",
    "Inspected pyproject.toml cpu extra and uv.lock binding; dry-run only to preserve shared environment. Activation paths are standard .venv paths; actual Python invoked explicitly.",
)
claim(
    "c20",
    "software",
    "下載程式使用固定公開revision、匿名存取及五檔指紋核對，已存在目的資料夾時拒絕覆蓋。",
    "19.md:56,60,62; linked19.11",
    "Specific published joint revision33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed; default output is ROOT/checkpoints/capstone and absolute, review supplied ignored alternative output. Does not retrain.",
    [
        evidence(
            "s_fetch",
            "fetch_capstone23-56 and main default output",
            "tokenFalse, immutable revision, all5 hash/size, checkpoint stage/format and existing-target refusal.",
        ),
        evidence("s_hf_download", "hf_hub_download revision arg", "Full commit pin selects release snapshot."),
        evidence("s_hf_headers", "get_token_to_send(tokenFalse)", "No authorization header is sent."),
    ],
    ["a_fetch", "a_fetch_stdout", "a_public", "a_audit", "a_original_hf_download_txt", "a_original_hf_headers_txt"],
    "All5 files match manifest; existing path preserved",
    "Anonymous joint download exit0; model1327950bytes SHA d85cca83cdb4952f4653ef6b58d94562b41403db3a9db2ea31ff57e31246ca16. Repeat exit2; all5 existing hashes unchanged.",
    "Actual public fixedpin/tokenFalse retrieval, every file SHA/bytes checked, weights_onlyCPU load and every tensor compared to original v2-joint export.",
)
claim(
    "c21",
    "software",
    "CPU infer的1+2工具迴圈得到TOOL:calculator:1+2、runtime3、DIRECT:3及answer3。",
    "19.md:57-60",
    "Specific joint public FP32 checkpoint, current CPU/software versions; same held-out row, no GPU performance or retraining claim.",
    [
        evidence(
            "s_cli",
            "infer branch row construction/run_assistant",
            "CLI sends prompt with calculator enabled and returns real trace JSON.",
        ),
        evidence("s_capstone", "run_assistant577-598", "Two calls use same language model."),
    ],
    ["a_cli_enabled", "a_audit"],
    "TOOL:calculator:1+2 -> runtime3 -> DIRECT:3 -> answer3",
    "Literal CLI JSON matches all fields, with EOS2 on both generations.",
    "Executed actual scripts/capstone.py infer using same published weights; source has no update calls in infer branch.",
)
claim(
    "c22",
    "software",
    "同題--calculator-disabled會生成ASK:計算器未開，runtime為null。",
    "19.md:62",
    "This trained joint behavior on this prompt/system condition; not a hardcoded semantic response or universal disabling effect on learned models.",
    [
        evidence(
            "s_cli",
            "available=not args.calculator_disabled; system row construction",
            "Tool state becomes model input and runtime availability.",
        ),
        evidence(
            "s_capstone", "run_assistant direct/ask branch", "ASK content is generated and does not execute calculator."
        ),
    ],
    ["a_cli_disabled", "a_ui"],
    "ASK:計算器未開; runtime null",
    "CPU CLI and own browser both reproduce expectedASK/null/no followup.",
    "Compared actual enabled and disabled response IDs/JSON; only calculator condition changes.",
)
claim(
    "c23",
    "software",
    "serve範例按鈕只填題目，送出才呼叫模型，並呈現原始回答、工具執行與最後回答。",
    "19.md:67-70",
    "Actual Chromium/CPU run, port0 assigned ephemeral local port instead of default8765 to avoid other reviewers; same serve entrypoint, host and routes. Scope ends at the tested browser flow.",
    [
        evidence(
            "s_ui",
            "JS example buttons329-333 versus chat-form submit347-363; POST153-177",
            "Example selects input; chat submit invokes run_assistant and displays literal trace.",
        )
    ],
    ["a_ui", "a_ui_code", "a_ui_calc_render", "a_ui_shape_render"],
    "Zero chat requests for example click; real model invoked onsubmit",
    "Own browser observed zero chat before submission; tool3, disabledASK, knownsquare->circle and high tone generated; page_errors empty.",
    "Captured all POST bodies and JSON responses, checked displayedraw/tool/final text and personally viewed both persistent screenshots.",
)
claim(
    "c24",
    "software",
    "介面圖片與純音選項生成真RGB像素與波形送入模型，不在文字問題中補正解。",
    "19.md:70",
    "Synthetic16x16 RGB and0.04s sine only. Audio pitch bookkeeping reconstructs requested frequency; it is not placed in text prompt or used as final answer.",
    [
        evidence(
            "s_ui",
            "request_row38-63; input_preview69-78; POST169-177",
            "Media specifications generate real preview tensors and the same model inputs.",
        ),
        evidence(
            "s_capstone",
            "modality_tensors285-305 and CapstoneModel94-107",
            "Only media markers appear in text sequence; actual image/audio features are projected.",
        ),
    ],
    ["a_ui", "a_audit", "a_data", "a_ui_shape_render"],
    "Real RGB16x16 and640 waveform samples at16kHz; no media label in user/system text",
    "Browser received16x16 RGB, center[0,230,0]/background[0,0,0],640 signed sine samples, audiohigh reply; audit generated actual tensor hashes from all source media.",
    "Source row/media objects are separate from prompt IDs; inspected prompt decode and projector input path rather than interpreting UI labels as model evidence.",
)
claim(
    "c25",
    "concept",
    "127.0.0.1網址屬於執行程式那台機器的loopback，手機同一網址不會指向桌機。",
    "19.md:70 local-address sentence",
    "Assumes ordinary local browser access described in text; no tunnel or forwarding setup is implied.",
    [
        evidence("s_loopback", "IPv4Address.is_loopback and127.0.0.0/8 constant", "127.0.0.1 is local loopback."),
        evidence(
            "s_ui",
            "create_server181-199; serve202-208",
            "Server validates local address and binds127.0.0.1 by default.",
        ),
    ],
    ["a_original_python_ipaddress_txt", "a_ui"],
)
claim(
    "c26",
    "concept",
    "FP32表示每個權重以32-bit浮點格式保存，本機serve讀FP32成品。",
    "19.md:70 FP32 sentence",
    "Dtype of numeric weights; public serialized file includes metadata/containers, so neither process memory nor file size equals bare parameterbytes. Main manifest has11 inference artifacts, but only joint is downloaded in this review.",
    [
        evidence("s_dtype", "torch.dtype floating point table", "float32 explicitly defined as32-bit floating point."),
        evidence(
            "s_ui", "serve202-205 calls load_capstone", "Serves capstone-v1CPU; no quantized-loading fallback here."
        ),
    ],
    ["a_original_torch_dtype_md", "a_audit", "a_public", "a_ui"],
)

section_bytes = (ROOT / (BASE + "_section.md")).read_bytes()
intro_bytes = (ROOT / (BASE + "_intro.md")).read_bytes()
chapter = (ROOT / "course/chapters/19.md").read_bytes()
current_intro = chapter[: chapter.index(b"## ")]
assert intro_bytes == current_intro
assert len(json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())["models"]) == 11
report = {
    "schema_version": 1,
    "review_stage": "technical",
    "lesson_id": "19.1",
    "source": "course/chapters/19.md#19.1",
    "reviewer_task": "/root/integration_technical_coordinator/fact_v2_19_01",
    "reviewer_context": "fresh",
    "source_sha256": hashlib.sha256(section_bytes).hexdigest(),
    "intro_sha256": hashlib.sha256(intro_bytes).hexdigest(),
    "intro_summary": "章首把各章零件整合為同一個文字/圖片/純音/工具助理，目的在逐題檢查新增能力後既有能力是否保留；限定合成顏色形狀、純音、短模板、照抄/資料/小加法小世界，明示不是通用中文助理；先試用與讀紀錄，再讀架構、資料、四階段、工具、效率、壓縮與交付，短程式不自動啟動長訓練。",
    "prerequisite_reading": audit["prerequisite_sha256"],
    "environment_reading": {
        "path": "docs/environment.md",
        "sha256": sha("docs/environment.md"),
        "scope": "Read full file; use current .venv CPU without uv sync mutation. Environment document's unrelated platform/version claims are not19.1 empirical evidence.",
    },
    "figure_sha256": {},
    "verdict": "pass",
    "claims": claims,
    "sources": sources,
    "artifacts": artifacts,
    "issues": [],
    "checks": {
        "factual_accuracy": {
            "status": "pass",
            "details": "26 substantive claims separated: original evaluation/connector/tool methods, protocol/EOS/runtime/software, all eight table rows, total parameter and90-row outcomes. Actual single-core model and tool follow-up inspected; no source or output contradicted19.1.",
            "claim_ids": [item["id"] for item in claims],
        },
        "numeric_verification": {
            "status": "pass",
            "details": "Hand parameter decomposition gives328128, independently counts every tensor. Formal90 rows independently reconstruct action80/end78, per-task denominator/correct sums; CPU replay all90 complete records and both generated-ID sequences exact. Recalculate both84-row validation matrices75/71. No CPU result presented asGPU speed evidence.",
            "claim_ids": ["c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18"],
        },
        "figure_consistency": {
            "status": "not_applicable",
            "details": "19.1 and chapter introduction contain no SVG/image reference. Browser screenshots are independently generated UI-flow evidence, both personally viewed; neither is a textbook diagram. Linked prerequisite figures do not form a19.1 illustrated numeric claim.",
            "claim_ids": [],
        },
        "source_verification": {
            "status": "pass",
            "details": "Read original arXiv versionsHELMv1/LLaVAv2/Toolformerv1 with equation/section boundaries and limitations; official PyTorch2.14.1, uv0.12.19, HF1.33.0 and CPython3.13.5 sources successfully read. Rendered-doc failures are retained and resolved through authoritative versioned source. Formal linked JSON URLs fetched and match local hashes; candidate-bank summaries and previous reviews unused.",
            "claim_ids": ["c1", "c2", "c3", "c19", "c20", "c25", "c26"],
        },
        "limitations": {
            "status": "pass",
            "details": "Keeps artificial parsing, actual formal outputs and local trial distinct. Single seed42, narrow templates/syntheticmedia; original green image-color baseline9/9 cannot prove vision; shape0/9 and calculator follow-up errors retained. Same tensor values across original/public packaging checked. Local install onlydry-run due task constraint; actual current CPU inference/browser validated. No GPU/latency claim, general assistant/speech claim, resume claim or hidden-run proof inferred.",
            "claim_ids": [
                "c1",
                "c2",
                "c3",
                "c8",
                "c10",
                "c11",
                "c14",
                "c15",
                "c16",
                "c18",
                "c19",
                "c20",
                "c23",
                "c24",
                "c26",
            ],
        },
    },
    "execution_binding": {
        "read_and_executed_source_sha256": audit["read_and_executed_source_sha256"],
        "section_and_intro_current_at_execution": True,
        "source_scope_note": "Full19.md SHA in execution receipt binds actual read time; independent source_sha256 andintro_sha256 bind19.1. Other sections' later citation edits do not substitute for reviewing19.1. Module SHA values are checked against both current working tree and formal deployment report.",
        "original_checkpoint_storage": "Real ignoredv2-joint model loaded and fullSHA/per-tensor receipts persisted in a_audit; weights are not proposed asGit artifacts. Public fixed HF pin plus a_fetch reproduces acquisition.",
        "author_history": "No prior technical/editorial report or author history inspected; author_tasks omitted rather than guessed.",
    },
}
(ROOT / "docs/technical-reviews/19.1.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
)
print(
    json.dumps(
        {
            "report": "docs/technical-reviews/19.1.json",
            "verdict": report["verdict"],
            "claims": len(claims),
            "sources": len(sources),
            "artifacts": len(artifacts),
            "source_sha256": report["source_sha256"],
            "intro_sha256": report["intro_sha256"],
        },
        indent=2,
    )
)
