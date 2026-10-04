from pathlib import Path
import hashlib
import json

base = Path("docs/technical-reviews/artifacts/natural-v4-factual/19.10")
out = base / "round2"
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(base / "first-own-report.json") == "3be2fa0b001b27c7fa50de7b6afd8feb82c5e2038362a5c9e3c07a22f5c0b1f9"
report = json.loads((base / "first-own-report.json").read_text())
recheck = json.loads((out / "recheck-results.json").read_text())
render = json.loads((out / "normalization-render-receipt.json").read_text())
assert recheck["all_byte_comparisons_equal"]

def artifact(identifier, kind, tail, description, **extras):
    path = out / tail
    return {"id": identifier, "kind": kind, "path": str(path), "sha256": sha(path), "description": description, **extras}

for item in report["artifacts"]:
    if item["id"] == "a_source":
        item.update(path=str(out / "section.raw.md"), sha256=sha(out / "section.raw.md"),
                    description="Own complete current raw UTF-8 section, personally reread in the original-owner recheck; sole normalization edit independently compared with preserved first source.")
    if item["id"] == "a_execution":
        item["description"] = "Preserved genuine first-round CPU stdout and original GPU record audit, reused only for unchanged claims after71 exact byte comparisons. No new repeat of that experiment is claimed."
    if item["id"] == "a_results":
        item["description"] = "Preserved first audit's detailed results, original c4 counterexample and record/code SHAs. Numerical/empirical results reused for unchanged content; old status text records the genuine first-round revision history."
    if item["id"] == "a_analysis":
        item["description"] = "Preserved first-round original-paper/API analysis and actual normalization contradiction history; current correction is independently verified in a_round2_analysis."

report["artifacts"] += [
    {"id":"a_first_report","kind":"source_snapshot","path":str(base / "first-own-report.json"),"sha256":sha(base / "first-own-report.json"),"description":"True first original-owner report: revise with c4 contradicted, preserved unchanged as issue history, not current factual evidence."},
    {"id":"a_first_source","kind":"source_snapshot","path":str(base / "section.raw.md"),"sha256":sha(base / "section.raw.md"),"description":"True first raw source containing the centering error, preserved unchanged for exact correction comparison."},
    artifact("a_round2_diff","source_snapshot","section.diff.txt","Actual exact diff: one inaccurate normalization sentence corrected, adding necessary14.2 link; no code/formula/table/data edit."),
    artifact("a_round2_reuse","source_snapshot","reuse-byte-comparison.json","Actually executed exact comparison of71 source/evidence entries establishing honest reuse scope, with no mismatches."),
    artifact("a_round2_code","code","recheck.py","Own bounded current-source recheck and literal new prerequisite normalization probe; code asserts exact sole edit and retained original evidence bytes."),
    artifact("a_round2_execution","execution","recheck.stdout.txt","Actual new boundedCPU prerequisite block/shift exercise and source-byte assertions, plus actual new SVG render receipt.",
             command="PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/19.10/round2/recheck.py > docs/technical-reviews/artifacts/natural-v4-factual/19.10/round2/recheck.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-factual/19.10/round2/recheck.stderr.txt",
             result="Exit0. Exact current source and all retained evidence assertions passed; new small14.2 LN/RMS probe and plus10 exercise matched; normalizationSVG rendered. No repeated GPU or original audit execution.", environment=recheck["environment"]),
    artifact("a_round2_results","source_snapshot","recheck-results.json","Own current raw source/chapter/prerequisite SHAs, exact-edit comparison, actual fresh normalization numbers, original-source reinspection and execution-reuse limits."),
    artifact("a_round2_stderr","source_snapshot","recheck.stderr.txt","Actual stderr of the successful bounded owner recheck."),
    artifact("a_round2_analysis","derivation","owner-recheck-analysis.md","Own full current-section reread, originalRMS paper/Torch/project source reinspection, changed-claim resolution and actual renewed/new figure viewing proof."),
    artifact("a_prerequisite14_2","source_snapshot","prerequisite-14.2.md","Full newly necessary14.2 section actually read; linked RMS definition and numerical mechanism checked independently."),
    artifact("a_norm_render_receipt","source_snapshot","normalization-render-receipt.json","Actual new Inkscape render command/exit/stdout/stderr and current source/render SHA receipts."),
    artifact("a_norm_render","figure_render","normalization.png","New14.2 SVG actually rendered and personally viewed in this round; arrows/feature-axis/gamma/beta and LN/RMS distinction checked against source and text."),
]
for source in report["sources"]:
    if source["id"] == "p_rms":
        source["inspection_note"] = "Original owner personally re-read §4 Eq.(4) and following paragraph pp.3–4 in same unchanged actual1910.07467v1 PDF. RMSNorm rescales without mean subtraction, which now agrees with the corrected current c4. The original contradiction remains in the preserved first report."
    if source["id"] == "s_cpu":
        source["title"] = "Preserved genuine original-owner CPU and original GPU-record audit, reused after exact byte comparison"
report["sources"].append({"id":"s_round2_execution","kind":"execution","title":"Fresh bounded normalization prerequisite probe and original-owner source-byte recheck","verified":True,"artifact_id":"a_round2_execution"})

c4 = next(item for item in report["claims"] if item["id"] == "c4")
c4.update(statement="This capstone usesRMSNorm to rescale a feature group by its root mean square without first subtracting the feature mean; the14.2 link correctly distinguishes this rule from LayerNorm centering.",
          location="19.10 line505, corrected RMSNorm sentence and newly necessary14.md#14.2 link",
          status="verified",
          scope="The actual project uses RMSNorm with learned feature gains and epsilon. Rescaling can change the mean's numerical value; it does not explicitly subtract/recenter by the feature mean. This definition gives no universal speed or quality guarantee.",
          evidence=[
              {"source_id":"p_rms","locator":"§4 Eq.(4), pp.3–4 and immediately following paragraph","supports":"Actual original paper divides features byRMS and removes the mean statistic; now directly supports current corrected explanation."},
              {"source_id":"t_RMSNorm","locator":"RMSNorm class343–427, original installed formula and forward423–427; package-file SHA unchanged","supports":"Official2.14.1+cpu implementation documents feature-axis RMS rescaling with epsilon/gain and no mean subtraction; personally reinspected."},
              {"source_id":"r_core","locator":"default_config:34–47, norm='rms'; CapstoneModel.language:85","supports":"Actual capstone uses the named RMSNorm configuration, not a LayerNorm-only explanation."},
              {"source_id":"r_modern","locator":"RMSNorm:8–16, especially forward14–16","supports":"Actual project implementation multipliesx by reciprocal RMS and learned gain, with no x−mean(x)."},
              {"source_id":"s_round2_execution","locator":"recheck-results.new_actual_CPU_probe; literal14.2 block, projectRMSNorm match and plus10 exercise","supports":"Own newly executed small example verifies the linked normalization distinction and mean-preserving versus non-centering mechanism."}],
          artifact_ids=["a_round2_analysis","a_round2_code","a_round2_execution","a_round2_results","a_prerequisite14_2","a_norm_render","a_norm_render_receipt"])
report["source_sha256"] = recheck["current_section_sha256"]
report["figure_sha256"][render["source"]] = render["source_sha256"]
report["prerequisite_sections"]["14.2"] = recheck["new_prerequisite"]
report["whole_chapter_sha256_at_owner_recheck"] = recheck["whole_chapter_sha256"]
report["verdict"] = "pass"
report["issues"] = [{"claim_id":"c4","status":"resolved",
                     "details":"The true first report identified an incorrect feature-center/scale description for this RMSNorm capstone; preserved first report/source and genuine checker1 remain unchanged.",
                     "resolution":"Current full section replaces that sentence with namedRMSNorm rescaling byRMS without mean subtraction and a valid14.2 link. Original owner personally reread the complete current section/new prerequisite, originalRMSNorm §4 Eq.(4)/Torch/project implementation, executed the small linked normalization example, viewed all needed figures again and compared71 retained evidence/source entries exactly. All current substantive claims verified; no unresolved issue remains."}]
report["revision_history"] = [{"prior_report_path":str(base / "first-own-report.json"),"prior_report_sha256":sha(base / "first-own-report.json"),
                              "prior_source_sha256":sha(base / "section.raw.md"),"prior_verdict":"revise","genuine_issue":"c4 normalization centering claim contradicted",
                              "original_checker_exit_code":1,"original_checker_receipt":str(base / "checker-receipt.json"),
                              "current_source_sha256":report["source_sha256"],"owner_recheck_evidence":"a_round2_analysis"}]
report["execution_reuse"] = {"byte_comparison_artifact":"a_round2_reuse","scope":recheck["reuse_scope"],
                            "new_execution":"Only the small newly linked normalization prerequisite probe/source assertions and normalization SVG render; previous CPU/GPU audits were not rerun."}
report["checks"]["factual_accuracy"].update(status="pass",details="Original owner personally reread all current substantive content. Solec4 error is actually corrected and checked against originalRMSNorm/Torch/project source; other claims retain verified unchanged original authorities and owned execution evidence after exact byte comparisons.")
report["checks"]["numeric_verification"]["details"] = "Prior genuine exact lesson/bits8, PTQ storage, KD math and all raw GPU-record recomputations are reused after exact code/input/output byte comparison; fresh bounded14.2 example/projectRMS formula/shift exercise also passed. No new original GPU or first audit execution is claimed."
report["checks"]["numeric_verification"]["claim_ids"].append("c4")
report["checks"]["figure_consistency"].update(status="pass",details="No direct19.10SVG. New normalizationSVG rendered/personally viewed; five byte-identical original prerequisite renders personally viewed again. Six sourceSVG SHA entries are current, and all arrows/numbers/axis/rules/lineage/tool-loop meanings agree with current text and original authorities.")
report["checks"]["figure_consistency"]["claim_ids"].append("c4")
report["checks"]["source_verification"].update(status="pass",details="Original PDFs/package/code/records retain exact bytes and original locators. Original owner personally re-read actualRMS paper §4 Eq.(4), installedTorch RMS formula and project config/implementation for corrected c4, plus fullnew14.2 text. Registered original-code and raw-record provenance verified without donor reviews.")
report["checks"]["limitations"].update(status="pass",details="Retained scoped one-input, fixed-seed and original-record conclusions; explicit reuse versus new bounded probe and no ownGPU replication. File/payload/runtime costs, DPO-versusjoint teacher, prescribed sampler versus unlogged GPU batches and score-versusID equality remain separate. RMS rescaling is stated without mean subtraction or universal speed/quality inference.")
report["checks"]["limitations"]["claim_ids"].append("c4")

Path("docs/technical-reviews/19.10.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"source_sha256":report["source_sha256"],"report_sha256":sha("docs/technical-reviews/19.10.json"),"verdict":report["verdict"],"issues":report["issues"],"figure_count":len(report["figure_sha256"]),"checks":{k:v["status"] for k,v in report["checks"].items()}},ensure_ascii=False,indent=2))
