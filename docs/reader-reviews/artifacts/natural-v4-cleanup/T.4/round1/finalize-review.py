from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parents[5]
art = Path(__file__).resolve().parent
rel_art = art.relative_to(root)
work = rel_art / "cpu-workspace"


def read(name):
    return json.loads((art / name).read_text())


def path(name):
    return str(rel_art / name)


source = read("source-receipt.json")
figures = read("figure-render-receipts.json")
views = read("figure-view-notes.json")
results = read("cpu-results-summary.json")
current = read("final-current-hashes.json")
prerequisite_ids = ["4.7", "7.1", "7.3", "7.11", "7.17", "7.18", "1.8", "4.5", "5.8", "6.5", "6.6", "16.2"]
reasons = [
    "Next-target shift and the difference from answer-only supervision.",
    "Role/content messages, byte encoding, BOS/EOS and SFT.",
    "Question labels are ignored as answers while questions remain input.",
    "Saved-model continuation versus a fresh independent SFT baseline.",
    "Why general-text and assistant-data stages are separated; branch diagram.",
    "Demonstrations, preference pairs and scored feedback; mandatory signals diagram.",
    "Negative log probability and mean cross entropy.",
    "Width and repeated model blocks; internal block calculations are not needed for T.4.",
    "Teacher-forced cost versus free generation and complete task success.",
    "Total NLL, common raw-byte denominator and tokenizer ranking-reversal diagram.",
    "BPE fragment merges, control IDs and literal text versus role boundaries.",
    "Key/Value definitions and why cached generation still reads earlier context.",
]
prerequisites = []
for ident, reason in zip(prerequisite_ids, reasons, strict=True):
    receipt = read(f"prerequisite-{ident}-receipt.json")
    receipt.update(snapshot=path(f"prerequisite-{ident}.md"), receipt=path(f"prerequisite-{ident}-receipt.json"), why_needed=reason)
    prerequisites.append(receipt)

raw = (root / "course/training.md").read_bytes()
heads = list(re.finditer(rb"^## ", raw, re.M))
start = next(m.start() for m in heads if re.match(rb"## T\.4(?:\D|$)", raw[m.start():]))
end = next((m.start() for m in heads if m.start() > start), len(raw))
section = raw[start:end]
assert section == (art / "section-original.md").read_bytes()
assert hashlib.sha256(section).hexdigest() == source["source_sha256"]
figure_hashes = {}
for fig in figures:
    figbytes = (root / fig["source"]).read_bytes()
    assert figbytes == (root / fig["snapshot"]).read_bytes()
    digest = hashlib.sha256(figbytes).hexdigest()
    assert digest == fig["sha256"]
    figure_hashes[fig["source"]] = digest

order = [{"order": 1, "source": "course/training.md#T.4", "extent": "Full numbered section, initial read", "snapshot": path("section-original.md")}]
for n, receipt in enumerate(prerequisites, 2):
    order.append({"order": n, "source": receipt["source"], "extent": "Complete explicitly linked prerequisite section", "snapshot": receipt["snapshot"]})
for n, view in enumerate(views, 14):
    order.append({"order": n, "source": view["source"], "extent": "Personally viewed the freshly rendered PNG with view_image", "render": view["viewed_render"]})
order.append({"order": 17, "source": "Bounded offline CPU artifacts", "extent": "Saved predictions, executed no-update examples, inspected manifests, generated IDs and assertion failure"})
order.append({"order": 18, "source": "course/training.md#T.4", "extent": "Full current section reread after prerequisites and CPU checks; raw-byte equality confirmed"})

report = {
    "lesson_id": "T.4",
    "source": "course/training.md#T.4",
    "reviewer_task": "/root/v4_review_coordinator/reader_tail_t_4",
    "reviewer_context": "fresh",
    "review_scope": "Readability only for a mathematically capable first-time high-school/college reader; no factual, benchmark or reading-time review.",
    "state": "FINAL",
    "source_sha256": source["source_sha256"],
    "figure_sha256": figure_hashes,
    "verdict": "pass",
    "reader_summary": "I understand two independently selectable learning workflows. Text continuation teaches the next encoded unit at each available target; SFT keeps user questions as context but scores assistant answers and their endings. Hold out entire attribute families so a changed question wording does not masquerade as a new problem. Save one genuinely unupdated model, measure the selected validation file and a fixed prompt, train the selected route, then measure again with the same data and generation settings. Compare both average probability cost and actual generated answer IDs: lower cost need not produce a correct complete answer, and a legal continuation need not reproduce the single held-out text. The larger GPU experiments are explicitly separate models and recipes, so their numerical tables are not promises for the CPU commands. I can follow the manifest reader, baseline saver, comparison assertions, output denominators and exercise using the named prerequisites.",
    "checks": {
        "background_and_links": {"status": "pass", "details": "The opening states alignment and conversation prerequisites, followed by staging context. I read the necessary linked sections listed here, including 7.18 and its required figure. I did not open whole chapters, prior reviews, public experiment reports, optional capstone/download routes or unneeded ablations. The local explanations of optimizer history and total-step continuation suffice for this workflow without deeper optimizer mathematics."},
        "terminology": {"status": "pass", "details": "JSONL, family, held-out validation/test, seed, manifest, fingerprint, SFT, checkpoint, width/layer/head/length, byte/token, EOS, mean NLL, exact/completed matching, cache and resume have usable explanations here or in the explicitly linked prerequisites I read. 6.6 explains BPE as combining common adjacent fragments; 16.2 defines Key and Value. No unresolved term blocks this section. GPU reproduction is clearly a separate hardware route."},
        "examples": {"status": "pass", "details": "Five styles times 9/1/2 families gives 45/5/10 SFT records versus 9/1/2 text records. Circle has six ASCII bytes plus EOS, hence seven scored targets even when generation is capped at two units. The larger tables identify different widths, layers, updates, data and hardware. The left/right suffix example explains why loss and one generated suffix measure different things. My manifest and baseline evaluations confirmed the predicted 35/36 target counts without testing training improvement."},
        "program_explanation": {"status": "pass", "details": "Each of the three Python fences has a line-by-line prose explanation. The manifest reader forms paths, parses JSON dictionaries, iterates split/stat pairs and prints one unchanged record. The start saver seeds a width32/layer1/head1/length128 model and saves step0 without backward or an optimizer update. The comparison loads two reports and checks data, validation labels, equal positive target counts, empty skipped lists and identical row order. It checks finite NLL and valid match rates, saves selected before/after fields, and prints fixed-prompt files. I executed all three original Python fences in an isolated artifact workspace; the comparison used two evaluations of the same zero-update checkpoint as an explicitly labeled control."},
        "exercise": {"status": "pass", "details": "I can say how the same held-out cost and SFT match rate changed, then identify changed samples without choosing only one good answer. I would also check completed_exact_match/eos_rate in the full JSON and verify fingerprints, positive matching denominators and skipped rows. Predictions preceded execution: the no-update control had equal metrics/answers. In a copied report I changed after.effective_tokens from36 to35; the original comparison stopped at that assertion and wrote no comparison. I did not run the200/500-step training exercise under this bounded CPU assignment; its logic and success checks are understandable."},
    },
    "issues": [{"id": "T4-navigation-density", "severity": "low", "status": "nonblocking", "location": "Initial CPU text route through later attributes route and common options", "description": "On my first read I had to retain the short before/train/after workflow while several optional GPU reproduction routes intervened before the shared options and comparison code. Explicit statements about separate models kept the meaning clear, so this did not block the exercise.", "suggested_correction": "Optional: give the CPU routes and common comparison block short subheadings and group the larger cited experiment routes under another subheading."}],
    "remaining_blocking_issues": [],
    "program_walkthrough": {
        "input_records": "toy-text has text and family; attributes-sft has role/content messages and family. red:circle:low gives circle, circle, red, low and circle,low across the five questions.",
        "cli_flow": "--data names JSONL; --mode/--task selects meaning; --checkpoint loads the common start; --train performs updates; --steps is the total destination; --output names the product. --chat adds role boundaries; --tokens caps encoded new units; temperature0 selects the top candidate; cache reuses earlier K/V without removing earlier context.",
        "formula": "For scored positions i=1..N, p_i is the probability of the true next answer unit and mean_token_nll=sum(-ln(p_i))/N. N counts scored units, not visible characters or questions. BPB=total NLL/(raw UTF-8 bytes*ln(2)); the declared convention includes EOS cost above and only original content bytes below.",
        "output_reading": "exact_match compares raw answer-content IDs after removing a normal final EOS only; completed_exact_match also requires EOS. eos_rate measures stopping separately. Text exact-match fields are null. records_read, records_selected, metric_denominators and skipped state which cases were measured. The terminal summary omits samples, so open saved JSON.",
        "resume": "--resume restores update-tool history, steps and random state with weights. Plain --checkpoint starts a new tool/plan. Completed200 cannot resume to total200; changing the total can change the schedule.",
    },
    "numerical_anchors": {
        "cpu_recipe": "Seed42, width32, one layer/head,128 positions. Text200/SFT500 updates are recipes to check, not my executions.",
        "split": "12 families become9/1/2;5 questions each gives45/5/10. Limit20 covers present1/5 validation rows; larger files need --limit all.",
        "scoring": "circle6+EOS=7; generated text validation34 bytes+EOS=35; SFT answers7+7+5+5+12=36.",
        "cited_text_run": "Width64,two layers,141568 parameters,600 L4 updates,batch16,learning rate0.003. Validation5.73454→0.96282 belongs to that separate run.",
        "other_cited_runs": "Independent story/poem models use800 updates and different corpus counts/targets. Tokenizer comparison uses400 updates with matching raw fragments but different events. Direct attributes uses900, another branch250+900 and UltraChat interface80. Ablations use500/300 from the same attributes base but separate branches.",
        "limit": "These larger numerical results and parameter totals are my interpretation of the prose, not reproduced facts.",
    },
    "actual_reading_order": order,
    "prerequisites": prerequisites,
    "figure_render_and_view_evidence": {"renders": path("figure-render-receipts.json"), "personal_view_notes": path("figure-view-notes.json"), "source_snapshots": [f["snapshot"] for f in figures], "assigned_section_figures": [], "required_prerequisite_figures": list(figure_hashes), "all_listed_figures_rendered_and_personally_viewed": True},
    "bounded_execution_evidence": {
        "predictions": path("cpu-predictions.json"),
        "original_code": [path(f"original-python-{n}.py") for n in (1, 2, 3)],
        "initial_success_receipts": path("cpu-initial-success-receipts.json"),
        "corrected_execution_receipts": path("cpu-execution-receipts.json"),
        "results_summary": path("cpu-results-summary.json"),
        "full_reports": [str(work / f"outputs/{name}.json") for name in ("text-before", "attributes-before", "attributes-after")],
        "comparison": str(work / "outputs/attributes-comparison.json"),
        "assertion_failure": path("comparison-denominator-guard-stderr.txt"),
        "my_harness_error": {"description": "My isolation override wrongly included kind in --output, causing kind/kind paths. I inspected output paths, corrected to the shared data root and reran dependent checks. Published defaults were not at fault.", "preserved_artifact": path("first-harness-attempt/diagnosis.json"), "error_logs": path("first-harness-attempt")},
        "runtime": "Existing .venv torch2.14.1+cpu, CUDA unavailable, tensors on CPU. No install/download/training/GPU command executed.",
        "observed_result": results,
        "sample_observation": "All5 baseline answers were wrong with no EOS. Row1 contained BOS ID1 at generated position15, giving invalid_special_tokens status. skipped remained empty: it was evaluated and failed, not skipped. I compared raw IDs rather than only displayed replacement characters.",
        "limits": "after evaluates start.pt a second time with zero updates. Its metrics and raw answers were identical. I did not reproduce the source one-row numerical excerpt, train any model or verify an earlier GPU experiment.",
    },
    "prior_report_preservation": {"path": path("prior-report-unread.json"), "sha256": hashlib.sha256((art / "prior-report-unread.json").read_bytes()).hexdigest(), "read": False, "method": "Copied original bytes before replacement; never opened or parsed the prior report."},
    "source_snapshot": path("section-original.md"),
    "source_receipt": path("source-receipt.json"),
    "final_current_byte_receipt": path("final-current-hashes.json"),
    "intro_required": False,
    "intro_reason": "T.4 is not the first numbered section of its source.",
}
text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
(art / "final-report-snapshot.json").write_text(text)
(root / "docs/reader-reviews/T.4.json").write_text(text)
print(json.dumps({"report": "docs/reader-reviews/T.4.json", "verdict": report["verdict"], "source_sha256": report["source_sha256"], "figure_sha256": figure_hashes, "blocking_issues": 0, "nonblocking_issues": 1, "evidence": str(rel_art)}, indent=2))
