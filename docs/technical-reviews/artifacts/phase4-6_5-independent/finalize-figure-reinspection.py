"""Save actual post-view judgment and update this original reviewer's canonical report."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OUT = BASE / "figure-reinspection"
PREFIX = BASE.relative_to(ROOT).as_posix()
PRIOR_SHA = "4b8dc50adda26f0900317fa14438d6bc04bc42459ddb324f62902711ba8c8446"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
report_path = ROOT / "docs/technical-reviews/6.5.json"
assert sha(report_path) == PRIOR_SHA
report = json.loads(report_path.read_text())
capture = json.loads((OUT / "results.json").read_text())
environment = json.loads((OUT / "environment.json").read_text())
assert capture["current_section_unchanged"]
assert capture["current_section_sha256"] == report["source_sha256"]
assert len(capture["page_receipts"]) == 2
assert all(p["response_status"] == 200 for p in capture["page_receipts"])
assert all(x["exit_code"] == 0 for x in capture["render_receipts"])
viewed_files = ["common-scale.png", "shared-denominator.png", "desktop-empirical-figure-in-page.png", "mobile-empirical-figure-in-page.png",
                "desktop-page-opening.png", "desktop-empirical-page-top.png", "desktop-empirical-page-bottom.png",
                "mobile-page-opening.png", "mobile-empirical-page-top.png", "mobile-empirical-page-bottom.png"]
assert all((OUT / name).is_file() for name in viewed_files)
receipt = {"kind": "original_reviewer_actual_figure_scope_reinspection", "reviewer_task": report["reviewer_task"], "review_date": "2026-10-06",
           "prior_history_path": capture["prior_history_path"], "prior_history_sha256": PRIOR_SHA,
           "source_sha256": capture["current_section_sha256"], "source_read": "Entire current6.5 actually reread; original one-sentence report-scope correction still accurate. No introductory chapter passage needed for a non-first section.",
           "current_figure_sha256": {name: item["sha256"] for name, item in capture["figures"].items()},
           "actual_render_commands": capture["render_receipts"], "actual_page_capture": capture["page_receipts"],
           "actual_view_tool": "tools.view_image called on each saved image listed below after successful rendering/capture, before final verdict",
           "actually_viewed": [{"path": str((OUT / name).relative_to(ROOT)), "sha256": sha(OUT / name)} for name in viewed_files],
           "visual_inspection": "Both current SVGs personally viewed as Inkscape renders. Current common-scale figure has readable separated token NLL and BPB panels; correct20excerpts,4193bytes,4213/2503EOS-inclusive targets and four recorded values. Correct orangebyte/blueBPE mapping and ranking reversal. On actual1280x800 desktop and390x844 mobile page, opened details and viewed opening/figure-top/figure-bottom screenshots; all necessary labels and EOS/scope limitations are visible while scrolling. Mobile figure fits displayed311px wide with all content legible. Desktop figure is taller thanviewport and remains readable through its top/bottom sections. No page horizontal overflow (documentWidth1265<=1280 and375<=390).",
           "capture_limit": "Desktop image-element screenshot includes the page stickyheader as a capture overlay partway down; actual desktop viewport top/bottom screenshots and raw SVG render were separately inspected and establish the underlying figure's full readable content. No claim that one desktop viewport contains the entire tall figure.",
           "bar_scale_verification": capture["bar_checks"],
           "geometry_change": "Old frozen chart baseline x140 and80px/unit, height700. Current chart baseline x32 and120px/unit, height1370; larger fonts, line breaks and panel spacing. These are proportional geometry/layout changes; metric names, event conventions, numbers and ranking remain the same.",
           "metric_unit_scope": "Top panel arithmetic is mean natural-log token NLL (nats per scored token), compared within that panel. Bottom BPB is bits per raw original UTF-8byte. The two different units are separately labelled and are not interchangeable; common visual px/unit does not change either unit. All four new widths equal120times the five-decimal displayed value; raw unrounded record deviation<0.0006px.",
           "denominator_and_event_scope": "Both models:20testexcerpts and4193rawUTF-8bytes. Byte4213targets=4193bytecontent+20EOS; BPE2503targets=2483BPEcontent+20EOS. EOS counted in NLL, absent from original-byte denominator. This supports only this experiment and does not establish general answering ability or tokenizer superiority.",
           "raw_measurement_reinspection_pointers": capture["raw_measurement_pointers_read"], "raw_measurement_sha256": capture["raw_result_sha256"],
           "evidence_reuse": capture["reuse_scope"], "fingerprint_check_count": len(capture["unchanged_formal_artifacts"]),
           "executed_scope": "Render bothSVGs, capture actual selected-section page and verify figure numbers/geometry against fixed original JSON. No originalfence or fullaudit rerun, model downloads/weightloads/inference/training/GPU or newpapers.",
           "verdict": "pass", "unresolved_questions": [], "environment": environment,
           "capture_code_path": str((BASE / "figure-reinspection.py").relative_to(ROOT)), "capture_code_sha256": sha(BASE / "figure-reinspection.py"),
           "finalization_code_path": str(Path(__file__).resolve().relative_to(ROOT)), "finalization_code_sha256": sha(Path(__file__).resolve())}
write(OUT / "actual-reinspection-receipt.json", receipt)
def artifact(identifier, path, kind, description, **extra):
    report["artifacts"].append({"id": identifier, "path": str(path.relative_to(ROOT)), "sha256": sha(path), "kind": kind, "description": description, **extra})
artifact("figure_reinspection_code", BASE / "figure-reinspection.py", "code", "Actually executed rendering/page capture/raw measurement geometry verification code; no model operations.")
artifact("figure_reinspection_finalize_code", Path(__file__).resolve(), "code", "Actually executed post-view receipt/report update script; original review owner preserves prior opaque report.")
artifact("figure_reinspection_capture_results", OUT / "results.json", "source_snapshot", "Real rendering/page-capture/unchanged fingerprint facts before views; later actual views recorded separately.")
artifact("figure_scope_reinspection_20261006", OUT / "actual-reinspection-receipt.json", "execution", "Canonical current figure support review: whole-section read, bothactualrenders, actualdesktop/mobile pageview,120px/unit geometry, original metric scope and explicitunchanged evidence reuse.",
         command=".venv/bin/python docs/technical-reviews/artifacts/phase4-6_5-independent/figure-reinspection.py > docs/technical-reviews/artifacts/phase4-6_5-independent/figure-reinspection.stdout.json 2> docs/technical-reviews/artifacts/phase4-6_5-independent/figure-reinspection.stderr.txt; tools.view_image on ten named rendered/page PNGs; .venv/bin/python docs/technical-reviews/artifacts/phase4-6_5-independent/finalize-figure-reinspection.py",
         result="Capture exit0; bothInkscape renders exit0; bothpageHTTP200 and servedSVG hashes match source;10images personallyviewed. Fourbarwidths match120*displayedmetric with unroundedrawerror<.0006px. Metric/supportscope unchanged; pass.", environment=environment)
for short, filename in [("shared", "rewrite-06-05-shared-denominator.svg"), ("empirical", "tokenizer_common_scale.svg")]:
    artifact("figure_current_" + short + "_svg", OUT / filename, "source_snapshot", "Current SVG snapshot actually read and rendered; current main report figure fingerprint matches this exact version.")
for name in viewed_files:
    artifact("figure_view_" + name.replace(".png", "").replace("-", "_"), OUT / name, "figure_render", "Actual rendered SVG or actual selected-section page screenshot personallyviewed; scope and capture limitations in canonical receipt.")
artifact("figure_reinspection_environment", OUT / "environment.json", "source_snapshot", "Actual Python/CPU/Inkscape1.4/Chromium151 runtime versions for this figure/page reinspection.")
artifact("figure_empirical_diff", OUT / "common-scale.diff", "source_snapshot", "Own frozen original figure versus current SVG diff; geometry and clarification changes personally inspected.")
report["figure_sha256"] = {name: item["sha256"] for name, item in capture["figures"].items()}
report["verdict"] = "pass"
report["figure_reinspection_date"] = "2026-10-06"
report["sources"].append({"id": "current_figure_scope", "title": "Own original-reviewer current figure and page support reinspection", "kind": "execution", "verified": True, "artifact_id": "figure_scope_reinspection_20261006"})
for claim_id in ["seven_byte_example", "test_reversal", "scoring_contract"]:
    claim = next(c for c in report["claims"] if c["id"] == claim_id)
    claim["evidence"].append({"source_id": "current_figure_scope", "locator": "actual-reinspection-receipt.json metric_unit_scope,denominator_and_event_scope,bar_scale_verification and actually_viewed", "supports": "Current actualrendered/viewed figures and desktop/mobile page preserve original metric values,UTF-8denominator,EOS convention and experiment-onlyscope. Current changed chart is120px/unit atx32, not original80px/unit atx140."})
    claim["artifact_ids"].append("figure_scope_reinspection_20261006")
    if claim_id == "test_reversal":
        claim["prior_figure_verification_details"] = claim["verification"]["details"]
        claim["verification"]["details"] = "Originalhash-matched data/CPU/measurement verification reused after allregisteredartifactfingerprints checked. Actualcurrentchart separatelyrendered/viewed: fourwidths333.5364/478.4892/483.4872/412.0812 atx32 equal120*five-decimaldisplayedvalues. Rawunroundedrecorderror<0.0006px; bothactualpageviewports1280x800/390x844 inspected. Units/denominators/ranking unchanged."
        claim["artifact_ids"] += ["figure_current_empirical_svg", "figure_view_common_scale", "figure_view_desktop_empirical_page_top", "figure_view_desktop_empirical_page_bottom", "figure_view_mobile_empirical_page_top"]
report["checks"]["figure_consistency"]["prior_details"] = report["checks"]["figure_consistency"]["details"]
report["checks"]["figure_consistency"]["details"] = "Current bothSVGs actuallyrendered/viewed and actualsectiondesktop/mobile1280x800/390x844 screenshotsviewed. Unchanged7bytehandexample; changedtallchart labels20excerpts4193bytes/4213and2503targets includingEOS. TopmeanNLLnats/token and lowerBPBbits/rawbyte remainseparate. Currentcommonleftx32,120px/unit widths matchdisplayed5decimalvalues exactly (<1e-10px),unroundedrawrecord<.0006px. Noexpandedgeneralizationclaim; fullpage/screenshotcapturelimits explicitlyrecorded."
report["checks"]["numeric_verification"]["details"] += " Figuregeometryreinspection: currentx32/120px/unit confirmed against originalrecord; oldbarwidthchecks retained as historicalevidence only."
report["checks"]["factual_accuracy"]["details"] += " Sameoriginalowner also completed currentfigure/body/page support reinspection on2026-10-06; no new substantivequestion."
report["reading_scope"]["current_figure_reinspection"] = receipt["source_read"] + " " + receipt["visual_inspection"] + " " + receipt["evidence_reuse"]
report["revision_history"].append({"stage": "own_current_figure_support_reinspection", "verdict": "pass", "date": "2026-10-06", "source_sha256": report["source_sha256"],
                                  "prior_report_path": capture["prior_history_path"], "prior_report_sha256": PRIOR_SHA, "execution_artifact_id": "figure_scope_reinspection_20261006",
                                  "current_figure_sha256": report["figure_sha256"], "actualscope": receipt["metric_unit_scope"] + " " + receipt["denominator_and_event_scope"],
                                  "body_unchanged": True, "actual_render_and_view_completed": True, "no_model_execution_or_unrelated_CPU_rerun": True})
write(report_path, report)
print(json.dumps({"verdict":report["verdict"],"report_sha256":sha(report_path),"source_sha256":report["source_sha256"],"prior_history_path":capture["prior_history_path"],
                  "prior_history_sha256":PRIOR_SHA,"canonical_artifact_id":"figure_scope_reinspection_20261006","canonical_receipt_path":str((OUT / "actual-reinspection-receipt.json").relative_to(ROOT)),
                  "canonical_receipt_sha256":sha(OUT / "actual-reinspection-receipt.json"),"unresolved_questions":[]},ensure_ascii=False,indent=2))
