"""Assemble this reviewer's inspected snapshots and independently authored report."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
REL = BASE.relative_to(ROOT).as_posix()
ENV = json.loads((BASE / "execution_summary.json").read_text())["environment"]
COMMAND = (
    "PYTHONPATH=. .venv/bin/python "
    "docs/technical-reviews/artifacts/fact_finish_t_7/replay.py "
    "> docs/technical-reviews/artifacts/fact_finish_t_7/execution_stdout.txt "
    "2> docs/technical-reviews/artifacts/fact_finish_t_7/execution_stderr.txt"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


# The ranges describe actual personal reads; copying a whole file does not imply
# that every copied line was read. Out-of-bounds sed endpoints are clipped here.
RANGES = {
    "scripts/prepare_data.py": "full file",
    "scripts/train.py": [[37, 116], [200, 263], [303, 358]],
    "scripts/infer.py": "full file",
    "scripts/course_experiments/run.py": [[1, 190]],
    "scripts/course_experiments/common.py": [[1, 180], [190, 318]],
    "scripts/course_experiments/text.py": [[33, 87], [374, 390], [596, 621]],
    "scripts/course_experiments/behavior.py": [[1, 380], [550, 759]],
    "scripts/course_experiments/posttraining.py": "full file",
    "scripts/course_experiments/capstone_student.py": "full file",
    "scripts/course_experiments/capstone.py": [[50, 310]],
    "tiny_perceptron/alignment.py": "full file",
    "tiny_perceptron/posttraining.py": "full file",
    "tiny_perceptron/capstone.py": [[1, 278], [321, 376], [398, 428], [499, 600], [613, 694]],
    "tiny_perceptron/model.py": "full file",
    "tiny_perceptron/data.py": [[54, 116]],
    "tiny_perceptron/modern.py": [[1, 83]],
    "scripts/check_technical_reviews.py": "full file",
}
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
code_receipts = []
for path, ranges in RANGES.items():
    original = ROOT / path
    snapshot = BASE / "code" / path
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(original.read_bytes())
    code_receipts.append(
        {
            "path": path,
            "snapshot": snapshot.relative_to(ROOT).as_posix(),
            "sha256": sha(original),
            "line_count": len(original.read_text().splitlines()),
            "personally_read_ranges_inclusive": ranges,
            "version": revision + "; inspected working-tree bytes identified by sha256",
        }
    )

guide_receipts = []
for path in ["outputs/integration-runs/root-technical-instructions.txt", "docs/technical-review-guide.md"]:
    snapshot = BASE / "instructions" / Path(path).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes((ROOT / path).read_bytes())
    guide_receipts.append(
        {
            "path": path,
            "snapshot": snapshot.relative_to(ROOT).as_posix(),
            "sha256": sha(snapshot),
            "read": "complete, before substantive source review",
        }
    )

weights = json.loads((BASE / "dpo_checkpoint_receipt.json").read_text()) + json.loads(
    (BASE / "natural_checkpoint_receipt.json").read_text()
)
public = json.loads((BASE / "public-models_selected.json").read_text())["models"]
weight_manifest = []
for item in weights:
    model_id = Path(item["path"]).parent.name
    model = next(row for row in public if row["id"] == model_id)
    file = next(row for row in model["files"] if row["output"] == Path(item["path"]).name)
    assert item["sha256"] == file["sha256"]
    assert sha(ROOT / item["path"]) == file["sha256"]
    assert (ROOT / item["path"]).stat().st_size == file["bytes"]
    weight_manifest.append(
        {
            "local": item["path"],
            "repo": model["repo"],
            "revision": model["revision"],
            "public_path": file["path"],
            "sha256": file["sha256"],
            "bytes": file["bytes"],
            "all_loaded_tensors_hashed": True,
        }
    )
save("text_weight_manifest_verification.json", weight_manifest)

save(
    "read_scope.json",
    {
        "reviewer": "/root/fact_finish_t_7",
        "reviewer_context": "fresh",
        "guides": guide_receipts,
        "lesson": {
            "path": "course/training.md#T.7",
            "snapshot": "docs/technical-reviews/artifacts/fact_finish_t_7_section.md",
            "sha256": "08e74b6a67575b17ff5527c7704a44dc669ad1104c990daad82775c9671d8df5",
            "read": "complete",
        },
        "prerequisites": {
            "snapshot": "docs/technical-reviews/artifacts/fact_finish_t_7_prerequisites.json",
            "sha256": sha(ROOT / "docs/technical-reviews/artifacts/fact_finish_t_7_prerequisites.json"),
            "sections_read_complete": [
                "T.5",
                "13.1",
                "13.2",
                "13.4",
                "13.5",
                "13.6",
                "13.7",
                "13.9",
                "13.10",
                "13.11",
                "13.12",
                "13.13",
                "13.14",
                "13.15",
                "13.16",
                "13.17",
                "19.8",
                "19.10",
                "19.12",
                "C.7",
            ],
        },
        "code": code_receipts,
        "original_papers_personally_read": {
            "dpo_original.txt": "lines 1–280, with Eq.7 separately reread at 232–274; Sections 3–4, Eqs.1–7",
            "ppo_original.txt": "lines 1–255; Sections 2–3 and 5, Eqs.6–12 and Algorithm 1",
            "rlhf_original.txt": "lines 421–520; Section 3.5, reward-model Eq.1 and PPO/PPO-ptx Eq.2",
            "kd_original.txt": "lines 1–155; Sections 1–2, temperature Eq.1 and T² gradient scaling",
        },
        "data_and_checkpoint_scope": "Every row of the 64 arithmetic pairs, all 100 cropped natural pairs and source-row provenance, all 165 finite-card contexts with complete raw probability vectors, all 84 joint/DPO validation rows, both 84-row student validation sets and both 90-row student test sets were programmatically inspected. Compact per-row listings were personally read, including supplemental reads of truncated outputs. All tensors of text base/beta0.1/beta1/format/natural-SFT/natural-DPO, four public capstone models and five fresh finite-card CPU checkpoints were loaded and hashed. Full raw JSON snapshots retain fields not displayed in compact listings.",
        "not_read_or_claimed": "No author history, dispatch records, expected verdicts, other review conclusions or private formal reference weight binary were read. No full formal text/natural/capstone GPU training, GPU time, multi-seed generalization or natural free-generation quality was reproduced. Formal GPU per-batch selections are absent; student sampling is a deterministic code reconstruction, not a captured GPU batch log. A prior T.7 report was preserved as unread bytes, not used as evidence. No environment configuration or book changes; no subagents or paid runs.",
    },
)

artifacts = []


def artifact(identifier, name, description, kind="source_snapshot", external=False):
    path = ROOT / name if external else BASE / name
    row = {
        "id": identifier,
        "kind": kind,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha(path),
        "description": description,
    }
    if kind == "execution":
        row.update(
            command=COMMAND,
            result="Exit 0; all independently asserted comparisons passed; see retained stdout, raw records and receipts",
            environment=ENV,
        )
    artifacts.append(row)
    return identifier


artifact(
    "a_section",
    "docs/technical-reviews/artifacts/fact_finish_t_7_section.md",
    "Complete personally read T.7 body, captured before review",
    external=True,
)
artifact(
    "a_prerequisites",
    "docs/technical-reviews/artifacts/fact_finish_t_7_prerequisites.json",
    "Complete bodies and hashes of the explicitly necessary prerequisite sections",
    external=True,
)
artifact(
    "a_scope", "read_scope.json", "Actual personal reading ranges, original-paper locators, data and checkpoint limits"
)
artifact(
    "a_replay",
    "replay.py",
    "Independent reviewer CPU replay, full raw-row checks and denominator reconstruction",
    "code",
)
artifact("a_execution", "execution_summary.json", "Independent CPU results and limits", "execution")
artifact("a_stdout", "execution_stdout.txt", "Exit-0 output of the actual independent replay", "execution")
artifact("a_stderr", "execution_stderr.txt", "Actual replay stderr")
artifact(
    "a_short",
    "short_cpu_checks.json",
    "Executed DPO loss/gradient, frozen capstone reference, masked KD scalar and CLI checks",
    "execution",
)
artifact(
    "a_dpo", "dpo_cpu_replay.json", "Complete CPU scored preference pairs and generated arithmetic answers", "execution"
)
artifact(
    "a_tokens", "dpo_denominators.json", "All deterministic sampled-pair and valid-label denominators", "execution"
)
artifact("a_natural", "natural_complete_records.json", "All twenty held-out natural-pair CPU scores", "execution")
artifact(
    "a_natural_pairs", "natural_all_excerpts.json", "All 100 prepared/cropped natural pairs with inherited provenance"
)
artifact(
    "a_natural_provenance", "natural_provenance.json", "Exact training-asset archive/member SHA and read/crop counts"
)
artifact(
    "a_cards",
    "posttraining_cpu_result.json",
    "Entire fresh short finite-card CPU experiment including all contexts",
    "execution",
)
artifact(
    "a_cards_rows", "cards_all_row_inspection.jsonl", "All 165 finite-card contexts in personally read per-row form"
)
artifact(
    "a_capstone",
    "capstone_summary.json",
    "Public capstone generation counts and student denominator reconstruction",
    "execution",
)
artifact(
    "a_initial",
    "student_initial_state_reconstruction.json",
    "All tensors and CPU reconstructed initial Dense state fingerprint",
    "execution",
)
artifact(
    "a_main_rows",
    "main_validation_inspection.tsv",
    "All 84 joint/DPO validation records in personally read paired form",
)
artifact(
    "a_student_val_rows",
    "student_validation_inspection.tsv",
    "All 84 CE/KD validation records in personally read paired form",
)
artifact(
    "a_student_test_rows", "student_test_inspection.tsv", "All 90 CE/KD test records in personally read paired form"
)
artifact(
    "a_network",
    "network_receipts.json",
    "Independent HTTPS requests, statuses, raw byte checks and source versions",
    "execution",
)
# Network verification was a separate real HTTPS retrieval, not replay.py.
artifacts[-1].update(
    command="Python 3.13 urllib.request.urlopen with default TLS validation for each recorded HTTPS URL; write response bytes and hashlib.sha256; compare GitHub raw bytes with local source; pdftotext -layout on each paper PDF",
    result="All six lesson blob/raw requests and four original-paper PDF requests returned HTTP 200; all three raw JSON files equal local bytes exactly",
)
artifact(
    "a_weights",
    "text_weight_manifest_verification.json",
    "Exact local text weight file SHA/byte comparison with revision-pinned manifest",
    "execution",
)
artifacts[-1].update(
    command=".venv/bin/python docs/technical-reviews/artifacts/fact_finish_t_7/build_report.py",
    result="Six text/natural model files exactly equal the selected pinned public manifest SHA and byte counts; all independently loaded tensors retained in separate receipts",
)
for tag, name in [
    ("dpo", "dpo_checkpoint_receipt.json"),
    ("natural", "natural_checkpoint_receipt.json"),
    ("capstone", "capstone_checkpoint_receipt.json"),
    ("cards", "posttraining_checkpoint_receipt.json"),
]:
    artifact(
        "a_weights_" + tag,
        name,
        "Exact checkpoint file provenance and every actually loaded state tensor for " + tag,
        "execution",
    )

for row in json.loads((BASE / "input_receipt.json").read_text()):
    identifier = "raw_" + row["source"].replace("/", "_").replace(".", "_")
    artifact(
        identifier,
        row["snapshot"],
        "Complete personally inspected original raw evidence: " + row["source"],
        external=True,
    )
for row in code_receipts:
    artifact(
        "snapshot_" + row["path"].replace("/", "_").replace(".", "_"),
        row["snapshot"],
        "Whole-file code bytes; personally read ranges separately documented in a_scope",
        "code",
        external=True,
    )
for tag in ["dpo", "ppo", "rlhf", "kd"]:
    artifact(
        "paper_" + tag,
        tag + "_original.pdf",
        "Original versioned paper personally downloaded and read at documented locators",
    )
    artifact(
        "paper_text_" + tag,
        tag + "_original.txt",
        "pdftotext extraction of the same original paper, with inspected line ranges",
    )
for index in range(3):
    artifact(
        "link_raw_" + str(index),
        "link_" + str(index) + "_raw.json",
        "Actual independently retrieved raw lesson-link bytes",
    )
    artifact(
        "link_page_" + str(index),
        "link_" + str(index) + "_page.html",
        "Actual independently retrieved original GitHub blob page",
    )
for name in [
    "capstone_cpu_joint_validation.json",
    "capstone_cpu_dpo_validation.json",
    "capstone_cpu_student-ce_validation.json",
    "capstone_cpu_student-kd_validation.json",
    "capstone_cpu_student-ce_test.json",
    "capstone_cpu_student-kd_test.json",
]:
    artifact(
        "cpu_" + name.replace(".", "_").replace("-", "_"),
        name,
        "Complete new CPU generated raw actions/final answers and EOS for " + name,
        "execution",
    )

sources = []
for tag, title, url, version, note in [
    (
        "dpo",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "https://arxiv.org/pdf/2305.18290v3",
        "arXiv 2305.18290v3, 2024-07-29",
        "Personally read Sections 3–4, Eqs.1–7 and derivation; Eq.7 is the negative log-sigmoid of beta times chosen/rejected policy-to-reference log ratios. DPO outline permits SFT or existing public preference data. It gives no guarantee that pairwise scoring or a weak SFT start produces correct free generation.",
    ),
    (
        "ppo",
        "Proximal Policy Optimization Algorithms",
        "https://arxiv.org/pdf/1707.06347v2",
        "arXiv 1707.06347v2, 2017-08-28",
        "Personally read Sections 2–3 and 5: Eq.6 importance ratio, Eq.7 min(rA, clip(r)A), Eqs.9–12 value fitting/advantage and Algorithm 1 repeated optimization of a collected rollout. Clipping removes excessive favorable incentive; it is not a hard bound on all probability changes. This project's one-action bandit uses no multistep GAE.",
    ),
    (
        "rlhf",
        "Training language models to follow instructions with human feedback",
        "https://arxiv.org/pdf/2203.02155v1",
        "arXiv 2203.02155v1, 2022-03-04",
        "Personally read Section 3.5, extracted lines 421–520: human answer comparisons train a reward model with Eq.1; PPO optimizes sampled language responses with a frozen SFT reference KL term, and Eq.2 distinguishes optional pretraining mixing. This separates reward model, frozen reference and feedback source; the finite-card rule labels are not the paper's human comparisons.",
    ),
    (
        "kd",
        "Distilling the Knowledge in a Neural Network",
        "https://arxiv.org/pdf/1503.02531v1",
        "arXiv 1503.02531v1, 2015-03-09",
        "Personally read Sections 1–2, extracted lines 1–155, Eq.1 softened softmax and the paragraph on combining hard labels with teacher soft targets and multiplying soft gradients by T². It supports the method, not a promise that this tiny student must outperform its CE control.",
    ),
]:
    sources.append(
        {
            "id": "s_" + tag,
            "kind": "paper",
            "title": title,
            "url": url,
            "version": version,
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "Versioned original paper by the method's authors, independently retrieved from arXiv; original PDF and extracted text preserved",
            "inspection_note": note,
        }
    )

for row in code_receipts:
    sources.append(
        {
            "id": "code_" + row["path"].replace("/", "_").replace(".", "_"),
            "kind": "repository_code",
            "title": row["path"],
            "path": row["path"],
            "sha256": row["sha256"],
            "version": row["version"],
            "verified": True,
            "inspection_note": "Personally read inclusive ranges "
            + str(row["personally_read_ranges_inclusive"])
            + "; full exact source snapshot and scope receipt retained. Only relevant inspected functions are used to substantiate claims.",
        }
    )
sources.extend(
    [
        {
            "id": "s_execution",
            "kind": "execution",
            "title": "Independent CPU replay and complete original-record comparison",
            "artifact_id": "a_execution",
            "verified": True,
        },
        {
            "id": "s_derivation",
            "kind": "derivation",
            "title": "Transparent DPO and masked KD scalar calculations",
            "verified": True,
            "details": "DPO z=0.1[(-4-(-4))-(-3-(-3))]=0, loss=-log(sigmoid 0)=ln2=0.6931471805599453. At z=0 derivatives with respect to chosen/rejected policy log score are -0.1(1-0.5)=-0.05 and +0.05. KD uses student logits [0.8,0.3], teacher [-0.1,0.9], class1, T=2 and alpha0.5. CE= -log softmax(student)[1]; teacher p=softmax([-0.05,0.45]), student q=softmax([0.4,0.15]); total=0.5 CE+0.5*4 sum(p log(p/q)). The first position has label-100 and contributes neither CE nor KL; independently constructed scalar 0.6244522333145142 vs implementation0.6244524717330933 differs2.384185791e-7.",
        },
    ]
)


def ev(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


def code(path):
    return "code_" + path.replace("/", "_").replace(".", "_")


claims = []


def claim(
    identifier,
    kind,
    statement,
    location,
    evidence,
    ids,
    scope,
    expected=None,
    observed=None,
    details=None,
    denominators=None,
    tolerance=None,
):
    row = {
        "id": identifier,
        "kind": kind,
        "statement": statement,
        "location": location,
        "status": "verified",
        "evidence": evidence,
        "artifact_ids": ids,
        "scope": scope,
    }
    if kind != "concept":
        row["verification"] = {"method": "executed", "expected": expected, "observed": observed, "details": details}
        if denominators:
            row["verification"]["denominators"] = denominators
        if tolerance:
            row["verification"]["tolerance"] = tolerance
    claims.append(row)


claim(
    "c1",
    "concept",
    "DPO trains on preferred/rejected answers to the same question using their policy-to-fixed-reference relative log scores.",
    "Paragraphs 1,4–5; preference-data and reference-model explanations",
    [
        ev(
            "s_dpo",
            "Sections3–4, Eqs.1–7, particularly Eq.7",
            "Common prompt x conditions both candidates; fixed pi_ref enters the pairwise logistic objective",
        ),
        ev(
            code("tiny_perceptron/alignment.py"),
            "sequence_log_probabilities; dpo_loss",
            "Answer-mask sequence sums and detached reference scores implement the stated project objective",
        ),
    ],
    ["paper_dpo", "a_short"],
    "DPO is a preference objective over supplied pairs. It neither makes labels true nor guarantees unobserved generation quality.",
)
claim(
    "c2",
    "concept",
    "A frozen SFT reference preserves starting behavior and is distinct from a reward model, value estimator or truth oracle; starting SFT alone does not establish ability.",
    "Reference-model paragraph; CPU value/reference paragraph",
    [
        ev(
            "s_dpo",
            "Section3 Eq.3 and Section4 Eqs.4–7; DPO outline",
            "Reference policy sets the starting behavior for regularized preference learning, not ground truth",
        ),
        ev("s_ppo", "Section5 Eqs.9–12", "Value prediction estimates expected return for an advantage baseline"),
        ev(
            "s_rlhf",
            "Section3.5 reward model and reinforcement-learning paragraphs",
            "A learned reward model supplies scores while a frozen SFT policy supplies a KL reference",
        ),
    ],
    ["a_dpo", "a_cards", "a_short"],
    "The observed weak text start demonstrates the need for independent capability checks in this experiment; the paper does not imply every SFT start is weak.",
)
claim(
    "c3",
    "software",
    "The opening 200-step exploratory recipe loads T.5 style.pt; default preference validation starts with 0+5=?, chosen5/rejected6; inference prints generated assistant text.",
    "First and second shell blocks; default first validation question paragraph",
    [
        ev(
            code("scripts/prepare_data.py"),
            "generate preference; split and main",
            "Synthetic arithmetic pairs and default seeded split",
        ),
        ev(
            code("scripts/train.py"),
            "parser; checkpoint/reference setup; update loop lines303–358",
            "--train enables updates and --steps specifies intended updates from the loaded checkpoint",
        ),
        ev(
            code("scripts/infer.py"),
            "main and chat generation",
            "Same question is supplied through assistant chat generation; non-JSON output is decoded answer",
        ),
    ],
    ["a_short", "a_replay"],
    "Actual test ran one CPU update with intended schedule200 from a width8 untrained temporary checkpoint and verified reference equality/policy change. It tests command/freeze mechanics, not the 200-step quality of T.5 style.pt.",
    "Default first pair0+5/5/6; valid CLI; unchanged saved reference and changed policy",
    "All assertions passed; actual_updates1, intended_schedule200; generated text printed",
    "Executed prepare_data, train with --stop-after1 and infer against isolated temporary outputs; inspect full recorded command/stdout and all saved reference tensors.",
)
claim(
    "c4",
    "empirical",
    "Formal arithmetic DPO starts from weak style/content.pt with held-out generation0/8 and0/7, then branches beta0.1/1 using64 questions, exchanged-operand family split49/8/7, width64/two layers/context128, seed42, lr0.001,batch8,250updates and unchanged frozen reference fingerprint.",
    "Formal experiment recipe paragraphs and model/output description",
    [
        ev(
            code("scripts/course_experiments/behavior.py"),
            "run_dpo; _dpo_train lines550–759",
            "Exact formal starting checkpoint, separate copies, hyperparameters and reference state assertion",
        ),
        ev(
            code("scripts/course_experiments/text.py"),
            "arithmetic_records lines596–621",
            "Exchangeable addends share family",
        ),
        ev(
            code("scripts/course_experiments/common.py"),
            "split_families and new_lm",
            "Seeded family split and architecture defaults",
        ),
    ],
    ["a_dpo", "a_weights_dpo", "a_weights", "a_execution"],
    "All base and published final strategy tensors were read and evaluated on CPU. Frozen original reference weight binaries are not public/read; unchanged formal reference is supported by formal hashes and implementation, plus the actual short freeze test. Formal GPU250-update training was not rerun.",
    "Base matches0/8,0/7;64=49+8+7; formal recipe and before/after reference hashes agree",
    "Exact CPU generated IDs and both zero-match counts agree; base state fingerprint4cf7b1e2182ad1edcc028480246baf2fe55d1fe54576c532c63505071b658d98 matches formal record",
    "Regenerated all64 pairs and family splits, read formal config/raw records and all text checkpoint tensors, CPU replayed every held-out question and candidate",
    {
        "seed": 42,
        "all_pairs": 64,
        "train_pairs": 49,
        "validation_pairs": 8,
        "test_pairs": 7,
        "updates_each_branch": 250,
        "pairs_each_update": 8,
    },
)
claim(
    "c5",
    "numeric",
    "Each formal arithmetic beta branch consumes9448 valid answer target positions, counting both candidates and EOS while excluding prompt and padding.",
    "Paragraph after formal shell block: 9,448 effective targets",
    [
        ev(
            code("tiny_perceptron/data.py"),
            "render_chat; pad_batch lines54–116",
            "Assistant target mask includes terminal EOS, excludes question/padding",
        ),
        ev(
            code("scripts/course_experiments/behavior.py"),
            "_pair_examples and _dpo_train",
            "Both candidate label masks are added for each sampled pair",
        ),
    ],
    ["a_tokens", "a_replay", "a_execution"],
    "This is a deterministic seed42 recipe count, not9448 distinct examples or a reconstructed GPU batch log.",
    "250×8=2000 pair draws per branch; valid masked targets9448",
    "Both model and beta1 produce exactly2000 pair draws and9448 valid labels",
    "Rebuilt rendered full preference pairs and the seeded random.choices sampling schedule; summed labels!=-100 across chosen and rejected independently; no mean-length estimate",
    tolerance="Integer counts exactly equal",
)
claim(
    "c6",
    "empirical",
    "Candidate ranking and relative-margin improvement are separate from free generation: both arithmetic beta branches generate0/8 and0/7; beta0.1 ranks6 above7 for4+2=? but generates8.",
    "Preference/arithmetic denominator definitions and4+2 example",
    [
        ev(
            code("scripts/course_experiments/behavior.py"),
            "_preference_evaluate; run_dpo arithmetic evaluation",
            "Absolute chosen/rejected ranking, policy-minus-reference margin and generation are independently recorded",
        ),
        ev(
            "s_dpo",
            "Section4 Eq.7",
            "Objective compares supplied candidates; it does not evaluate every generated alternative",
        ),
    ],
    ["a_dpo", "a_execution"],
    "One seed and the frozen held-out set support this observed failure. Ranking6 above7 does not imply6 is the highest-scoring possible output, nor does zero tiny arithmetic success establish DPO's general performance.",
    "Full generated records show0/8,0/7 for each beta;4+2 chosen6 higher than rejected7 but output8",
    "Beta0.1 val chosen_higher3/8,relative5/8;test1/7,1/7; beta1 val0/8,5/8;test0/7,0/7.4+2 log6=-24.8808326721,log7=-28.1159667969,generated IDs[64,2] decode8+EOS",
    "Compared every original held-out candidate field with full CPU sequence scores and every generated raw ID/EOS with new greedy generation",
    {
        "validation_pairs": 8,
        "test_pairs": 7,
        "generation_questions_each_branch": 15,
        "candidate_answers_each_pair": 2,
        "seed": 42,
    },
)
claim(
    "c7",
    "empirical",
    "The format-only branch prefers the shorter correct answer over the same answer plus '; answer complete': after200updates test relative margins improve7/7 while free-generation matches remain0/7.",
    "format-model paragraph",
    [
        ev(
            code("scripts/course_experiments/behavior.py"),
            "run_dpo format_pairs and format_train",
            "Only the appended suffix distinguishes the two correct supplied candidate numbers",
        )
    ],
    ["a_dpo", "a_tokens", "a_weights_dpo", "a_execution"],
    "This branch measures a formatting pair objective, with the same weak arithmetic base; it is not a generated-answer quality gain.",
    "Test relative improvement7/7; generated matches0/7;200updates",
    "All7 margins improve and supplied shorter candidates score higher7/7; all7 generated answers remain wrong; deterministic target count34554",
    "Read every full format pair and held-out generation record, loaded all format model tensors and reproduced all scores and raw IDs on CPU",
    {"updates": 200, "batch_pairs": 8, "pair_draws": 1600, "test_pairs": 7, "test_generated_questions": 7, "seed": 42},
)
claim(
    "c8",
    "empirical",
    "Natural-data pilot independently builds width32/one-layer model from100 source pairs split80/10/10, crops prompts/answers to120UTF-8 bytes, runs80SFT then80DPO steps, and reports only cropped held-out candidate comparisons without relabeling or free-generation quality.",
    "Natural-data paragraph",
    [
        ev(
            code("scripts/course_experiments/behavior.py"),
            "_natural_dpo_pilot",
            "Separate fresh architecture, inherited labels, split and80-step stages; only preference eval is returned",
        ),
        ev(
            code("scripts/course_experiments/text.py"),
            "_utf8_prefix lines374–390",
            "Byte truncation drops incomplete UTF-8 trailing bytes",
        ),
    ],
    ["a_natural", "a_natural_pairs", "a_natural_provenance", "a_weights_natural", "a_execution"],
    "Personally inspected all100 prepared excerpts and their source provenance, loaded both published model tensors and CPU scored20 held-out pairs. No natural80-step training or generation quality was rerun; inherited cropped labels can change meaning and were not manually validated as human preferences.",
    "100 source pairs;80/10/10; each encoded field≤120bytes;80SFT/80DPO; no free-generation result",
    "All source/split fingerprints and counts agree; all20 CPU held-out score records agree within5e-4; free_generation_evaluation_present=False",
    "Read exact training-asset archive member, recomputed source-row hashes and prepared/split JSONL bytes, compared published SFT/pilot scores against every validation/test pair",
    {
        "source_pairs": 100,
        "train_pairs": 80,
        "validation_pairs": 10,
        "test_pairs": 10,
        "sft_updates": 80,
        "dpo_updates": 80,
        "batch_pairs": 4,
        "max_utf8_bytes_each_field": 120,
        "seed": 42,
    },
)
claim(
    "c9",
    "concept",
    "PPO updates sampled choices using reward and old/new action-probability ratios with a clipped favorable incentive; a learned value baseline and frozen reference have distinct roles.",
    "Other-routes PPO introduction and value-estimator paragraph",
    [
        ev(
            "s_ppo",
            "Section3 Eqs.6–7; Section5 Eqs.9–12 and Algorithm1",
            "Probability ratio, min of unclipped/clipped surrogates, advantage/value and repeated rollout optimization",
        ),
        ev(
            "s_rlhf",
            "Section3.5 reinforcement-learning paragraph",
            "Frozen policy reference penalizes deviations separately from learned reward",
        ),
    ],
    ["paper_ppo", "paper_rlhf", "a_cards"],
    "Clipping is an objective incentive, not a hard bound on every update. The card code is a one-action contextual bandit, not full autoregressive or multistep language PPO.",
)
claim(
    "c10",
    "software",
    "The CPU posttraining command is independent of style.pt and downloaded weights; it freshly trains a148-parameter four-card policy,241-parameter RM and97-parameter critic, then forks PPO and DPO from one fixed SFT start.",
    "CPU route implementation and command paragraphs",
    [
        ev(
            code("scripts/course_experiments/posttraining.py"),
            "CONFIG; build_dataset; run_posttraining; _train_ppo; _train_dpo",
            "No previous text/capstone checkpoint load; SFT copies share initial reference; rule labels, candidate cards and training roles",
        ),
        ev(
            code("tiny_perceptron/posttraining.py"),
            "FiniteResponsePolicy; RewardModel; ValueModel; clipped_surrogate; bandit_advantage",
            "All model dimensions and distinct ratio, reward and baseline operations",
        ),
    ],
    ["a_cards", "a_cards_rows", "a_weights_cards", "a_execution"],
    "Policy observes four numeric features, not Chinese text; it selects a prewritten card and does no language generation or arithmetic execution. Reference KL is analytically optimized separately from RM-only critic targets. Rule-generated labels contain no recruited human comparisons.",
    "Fresh networks and forks;148/241/97parameters; no text checkpoint dependency",
    "Full independent CPU run completed; all five new checkpoint state dictionaries loaded; fixed SFT/ref fingerprintf6ebe6766d91b324a2af12d628af5e7df9cfd144a1961e5ce551f46f2467aacc agrees across branches",
    "Executed full short official CPU experiment in an isolated temporary output directory and checked every original feature/card/probability record and every saved network tensor",
)
claim(
    "c11",
    "empirical",
    "The independent CPU finite-card experiment improves fixed test top1 selections from6/18 SFT to12/18 for PPO and DPO under the stated rule-label protocol; this is not completed human-feedback language-model RLHF.",
    "Public CPU report link and RLHF limitation paragraphs",
    [
        ev(
            "s_rlhf",
            "Section3.5 reward modeling Eq.1 and language-response PPO",
            "Original RLHF uses human comparisons and generated responses, unlike authored finite-card labels",
        ),
        ev(
            code("scripts/course_experiments/posttraining.py"),
            "build_dataset; _evaluate; _train_ppo and _train_dpo",
            "Family-safe55-family split and exact card argmax correctness protocol",
        ),
    ],
    ["a_cards", "a_cards_rows", "a_execution"],
    "Both branches still choose the short numeric card for explain requests. Scores are only this55-family fixed seed42 card task; equal12/18 does not establish equivalence of algorithms, general language alignment, human preference or GPU speed.",
    "6/18→12/18 for each branch;44/5/6 families and132/15/18 contexts",
    "Full CPU rerun and every original top1 action reproduce6/18,12/18,12/18; all165 contexts checked including explain failures",
    "Reconstructed all candidate text/features and rule-best actions, asserted complete raw probabilities and argmax counts; checked PPO7680 new actions versus23040 three-epoch reuse draws and360optimizer updates",
    {
        "families": 55,
        "train_families": 44,
        "validation_families": 5,
        "test_families": 6,
        "train_contexts": 132,
        "validation_contexts": 15,
        "test_contexts": 18,
        "train_preference_pairs": 660,
        "validation_preference_pairs": 75,
        "test_preference_pairs": 90,
        "sft_updates": 60,
        "sft_draws": 1920,
        "reward_updates": 300,
        "reward_pair_draws": 19200,
        "ppo_rollout_batches": 120,
        "new_ppo_actions": 7680,
        "ppo_reused_action_draws": 23040,
        "ppo_updates": 360,
        "dpo_updates": 360,
        "dpo_pair_draws": 23040,
        "seed": 42,
    },
)
claim(
    "c12",
    "software",
    "Capstone DPO is a separate autoregressive multimodal branch that loads the completed joint predecessor and copies it as a frozen reference, rather than reusing the independent CPU card policy.",
    "Second optional route: capstone joint→DPO paragraph",
    [
        ev(
            code("scripts/course_experiments/capstone.py"),
            "_run_context stage dpo; frozen_reference; preference training lines50–310",
            "Completed joint predecessor is mandatory;100-step DPO plus CE anchor/router term",
        ),
        ev(
            code("tiny_perceptron/capstone.py"),
            "CapstoneModel; prepare_batch; preference_pairs; frozen_reference; save/load",
            "Image/audio representations condition actual autoregressive text and fixed reference copy",
        ),
    ],
    ["a_weights_capstone", "a_short", "a_capstone", "a_execution"],
    "Public deployed binary hashes differ from original source checkpoint metadata because export changes packaging. DPO teacher source metadata is b2428be8adfdbc60dff589ec8ca9270f3377e1688da863169c75d40326396b1f; its recorded parent is joint8f7e85820bd70bf1f16e7ef789b11f2d10873cf26a2688d651e23136bd7c3b47. The formal frozen reference binary was not read; freeze code and CPU identical-copy loss were tested.",
    "Joint predecessor and fixed reference; true text generation; distinct model/state format",
    "All public joint/DPO tensors loaded; metadata links correct parent; first actual pairDIRECT:9 versusDIRECT:9，祝你愉快！ yieldsln2 from identical policy/reference",
    "Independently downloaded revision33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed joint/DPO files with exact manifest SHA/bytes; inspected full metadata/state and executed one masked pair forward loss",
)
claim(
    "c13",
    "empirical",
    "The capstone currently recommends joint and retains DPO as a comparison: the same frozen84-question validation scores75/84 versus71/84 support that selection.",
    "Second optional route: recommendation sentence",
    [
        ev(
            code("tiny_perceptron/capstone.py"),
            "evaluate_rows and assistant/runtime protocol lines499–600",
            "Exact action+EOS, correct tool arguments and final returned answer are checked together",
        )
    ],
    ["a_capstone", "a_main_rows", "a_weights_capstone", "a_execution"],
    "Recommendation is supported by validation selection before test generation, not by importing the text/card task. Both models fail9/9 image-shape rows; DPO additionally loses4 joint rows. This frozen synthetic tiny suite is not broad visual/audio/language quality.",
    "Selection joint; same84 validation questions joint75,DPO71",
    "All84 original generated actions/final traces and EOS per branch exactly reproduce on CPU; joint task18/18→DPO14/18 explains4 regressions",
    "Read capstone-selection.json complete and all84 paired raw rows, rebuilt dataset/expected actions, independently replayed full public-checkpoint generations and tool returns",
    {
        "validation_questions_each_model": 84,
        "joint_task_questions": 18,
        "image_shape_questions": 9,
        "seed": 42,
        "generated_records_checked": 168,
    },
)
claim(
    "c14",
    "concept",
    "Distillation adds imitation of the teacher's next-token probability distribution to hard-label learning, using softened teacher targets; the teacher need not be the currently recommended model.",
    "Third optional route: distillation definition",
    [
        ev(
            "s_kd",
            "Sections1–2, Eq.1 and T² scaling paragraph",
            "Teacher soft probabilities plus hard-label objective are the original distillation recipe",
        ),
        ev(
            code("tiny_perceptron/alignment.py"),
            "distillation_loss",
            "Masked KL(teacher||student),temperature2,T² and alpha0.5 implement the project's blend",
        ),
    ],
    ["paper_kd", "a_short", "a_weights_capstone"],
    "A teacher-distribution objective does not guarantee improvement. The precise choice of DPO teacher is this project's checkpoint provenance fact, not a theoretical requirement of distillation.",
)
claim(
    "c15",
    "empirical",
    "Dense CE/KD students share the same random start,350-step recipe and deterministic data order; DPO is their teacher, and final90-question CE62 versusKD61 gives no KD win in this run.",
    "Third optional route: Dense, teacher identity and62/61 paragraph",
    [
        ev(
            code("scripts/course_experiments/capstone_student.py"),
            "STUDENT_CONFIG; _train_branch; run_capstone_student",
            "Deepcopy same initialization, shared sampler seed5042 and branch recipes, frozen capstone_preference teacher",
        ),
        ev(
            code("tiny_perceptron/model.py"),
            "Block.__init__ experts branch",
            "experts0 uses the same DenseFFN for each token",
        ),
        ev(code("tiny_perceptron/modern.py"), "DenseFFN lines35–51", "No per-token expert selection in the Dense path"),
    ],
    [
        "a_capstone",
        "a_initial",
        "a_student_val_rows",
        "a_student_test_rows",
        "a_weights_capstone",
        "a_short",
        "a_execution",
    ],
    "All teacher/student tensors and all final validation/test rows were read. CPU reconstruction matches initial fingerprint and145163target count but formal GPU per-batch choices were not logged; this supports the intended identical recipe, not an independently observed GPU batch trace. No350-step GPU retraining, broad KD superiority, speed/memory benchmark or compression of recommended joint is claimed.",
    "Same Dense start/order; DPOteacher;CE62/90,KD61/90",
    "Initial state fingerprint3c1018fd7da00fe36dc2cb3ade40709793330ea636f4d667db493be5c3f7d24d reproduced; seed5042 schedule145163valid targets; all90 test IDs/EOS match; validation59/84 each; teacher metadata equals DPO sourceb2428... rather than joint8f7...",
    "Loaded public teacher and both student state dictionaries in full; reconstructed Dense initialization and complete350×24balanced sample schedule; independently generated every validation/test action and final answer; exact CE-only advantage occurs on calculator1+8 questionffe914bb5e9abc53f39f",
    {
        "student_parameters_each": 79920,
        "width": 48,
        "layers": 2,
        "experts": 0,
        "steps_each": 350,
        "batch_rows": 24,
        "sampled_rows_each": 8400,
        "effective_answer_targets_each": 145163,
        "validation_questions_each": 84,
        "test_questions_each": 90,
        "initialization_seed": 42,
        "sampler_seed": 5042,
    },
)
claim(
    "c16",
    "software",
    "All three lesson HTTPS report links retrieve the correct original report bytes at this review date; the two optional-route links are pinned to commit1df335318bda03fd771807f66976953231d5a00b.",
    "Three linked public full reports: arithmetic DPO, CPU posttraining and capstone student",
    [
        ev(
            "s_execution",
            "network_receipts.json and retained blob/raw bodies",
            "Direct default-TLS HTTPS access and exact SHA comparison, not inferred availability",
        )
    ],
    ["a_network", "link_raw_0", "link_raw_1", "link_raw_2", "link_page_0", "link_page_1", "link_page_2"],
    "DPO points to mutable main; only the2026-10-04 captured bytes are verified. Optional posttraining/student commit-pinned bytes were200 and equal the exact local files. This says nothing about permanent future hosting.",
    "Each original blob/raw pair HTTP200 and raw bytes exactly equal local original report",
    "All3raw equality checks true: dpoSHAf298491097b9dcd7f62b2619804e744b8ad660a5831f5bfccbf5f2eedc947735;posttrainingSHA95c8ee6c89b009a9646903c1bccaa6a0ace53786a608307d50ba7d9fb0a09172;studentSHAe724ae3fbe2aa967534a5d52da6a07cc7739121d0846f18a86936b8d0f73f993",
    "Fetched the lesson's exact GitHub blob URL plus corresponding raw URL personally using urllib and default TLS; saved all original bodies and access/status/SHA receipts",
)

all_ids = [row["id"] for row in claims]
report = {
    "schema_version": 1,
    "review_stage": "technical",
    "lesson_id": "T.7",
    "source": "course/training.md#T.7",
    "reviewer_task": "/root/fact_finish_t_7",
    "reviewer_context": "fresh",
    "source_sha256": "08e74b6a67575b17ff5527c7704a44dc669ad1104c990daad82775c9671d8df5",
    "figure_sha256": {},
    "verdict": "pass",
    "claims": claims,
    "sources": sources,
    "artifacts": artifacts,
    "issues": [],
    "checks": {
        "factual_accuracy": {
            "status": "pass",
            "details": "Personally checked same-question preference/fixed-reference mechanisms, separate text/card/capstone starts and objective roles; verified teacher identity, Dense path and all quoted results against full raw records and loaded tensors.",
            "claim_ids": all_ids,
        },
        "numeric_verification": {
            "status": "pass",
            "details": "Independent CPU forward calculations give DPOln2 and gradients±0.05, maskedKD0.62445247; exact valid-target counts9448 each; complete card6/18→12/18 and capstone/student84/90-question generation replays match. No GPU time or complete GPU train was inferred from CPU.",
            "claim_ids": ["c4", "c5", "c6", "c7", "c8", "c11", "c13", "c15"],
        },
        "figure_consistency": {
            "status": "not_applicable",
            "details": "T.7 contains no image/SVG references; no figure claim requires rendering.",
            "claim_ids": [],
        },
        "source_verification": {
            "status": "pass",
            "details": "Independently retrieved and personally read DPOv3Eqs.1–7, PPOv2Eqs.6–12, InstructGPTv1Section3.5 and KDv1Sections1–2. Complete code/raw snapshots and actual read ranges retained. All three lesson report raw URLs fetched200 with exact local-byte equality; two links commit-pinned, arithmetic main version captured.",
            "claim_ids": all_ids,
        },
        "limitations": {
            "status": "pass",
            "details": "Explicitly separate exploratory200steps from formal250steps, supplied-pair scores from generated correctness/EOS, synthetic authored rules from human RLHF, finite cards from language PPO, DPO comparison teacher from recommended joint, deterministic sampling reconstruction from unavailable formal GPU batch logs, and CPU generation agreement from GPU timing or broad quality. Single-seed synthetic results and cropped inherited natural labels support no general quality guarantee.",
            "claim_ids": all_ids,
        },
    },
}
assert sha(ROOT / "docs/technical-reviews/artifacts/fact_finish_t_7_section.md") == report["source_sha256"]
(ROOT / "docs/technical-reviews/T.7.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "report": "docs/technical-reviews/T.7.json",
            "sha256": sha(ROOT / "docs/technical-reviews/T.7.json"),
            "claims": len(claims),
            "artifacts": len(artifacts),
            "verdict": report["verdict"],
        },
        ensure_ascii=False,
    )
)
