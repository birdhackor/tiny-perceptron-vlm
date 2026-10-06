"""Record the owner's actual reading/viewing decision and update only 6.7."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
OUT = BASE / "context-reinspection-20261006"
FACTS = json.loads((OUT / "facts.json").read_text())
prior_path = ROOT / FACTS["priorhistory_path"]
prior = json.loads(prior_path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rel(path):
    return path.relative_to(ROOT).as_posix()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

assert sha(prior_path) == FACTS["priorhistory_sha256"]
own_body = next(s for s in FACTS["sections"] if s["lesson_id"] == "6.7")
assert own_body["current_sha256"] == prior["source_sha256"]
receipt_id = "a-context-reinspection-20261006-receipt"
figures = []
for figure in FACTS["figures"]:
    assert sha(ROOT / figure["source_path"]) == figure["source_sha256"]
    note = ("Personally viewed PNG through tools.view_image: readable source 貓 plus drawn yellow face; exact label states U+1F642, 貓3bytes/face4bytes/total7. The same two-denominator boxes do not define tokens as characters. Only this sample/unit correspondence is needed by6.7; broader BPB math is outside this reinspection."
            if "shared-denominator" in figure["source_path"] else
            "Personally viewed PNG through tools.view_image: upper block is per-token cost, lower block is per-original-byte BPB, explicit one token contains different amounts of original text and structural EOS is excluded from original bytes. This does not impose token=word/scalar assumptions on6.7. Historical ranking numbers were not rerun/revalidated here because6.7 does not depend on them.")
    figures.append({**figure, "status":"rendered_and_personally_viewed",
                    "viewer_tool":"tools.view_image", "viewed_path":figure["render_path"],
                    "inspection_note":note,
                    "historical_delta_scope":"Own earlier6.7 report did not use/freeze6.5 figures. Current graphic inspected now; no unmeasured claim that it is identical to a prior figure."})

inspection = {
    "owner": prior["reviewer_task"], "review_type":"same_original_owner_narrow_context_reinspection",
    "completed_at":datetime.now(timezone.utc).isoformat(), "verdict":"pass", "unresolved_questions":[],
    "priorhistory_path":FACTS["priorhistory_path"], "priorhistory_sha256":FACTS["priorhistory_sha256"],
    "canonical_source_sha256":own_body["current_sha256"],
    "actual_read_scope":{
        "own_section":"Personally reread complete current6.7, exact same bytes as the originally verified fence/body; zero figures.",
        "context_discovery":"Personally read current6.5/6.6 text/code to identify actual dependencies and compare only my frozen prerequisite snapshots. No other review report, author repair note or verdict read.",
        "necessary_6_5":"Only the same 貓🙂 seven-byte sample and distinction of token units from raw bytes; current prose byte-identical to my frozen6.5. NLL/EOS bookkeeping, rankings, training exposure and limits are background, not correctness premises of6.7.",
        "necessary_6_6":"Current paragraphs defining IDs0..7 as structural, UTF-8 byte+8 content IDs, decode joining content bytes, explicit role-field insertion, and BPE ordinary-content/structural contract. Exactly four traditional-script substitutions in one paragraph: 记→記,须→須,决→決,编→編; no change to ID values, APIs, record-role rule or lossless-content scope. No figures in6.6.",
        "unchanged_6_3":"Fingerprint only confirms my frozen6.3 prerequisite still byte-identical. Its prior personal ByteLevel BPE contract inspection retained; no new whole-section technical review.",
        "method":"Reread current docs/review-tools/factual-reviewer-instructions.md in full; snapshot saved.",
        "figure_scope":"Personally rendered/viewed both current6.5 graphics as adjacent context, as detailed below; no chart performance remeasurement and no claim that these are6.7 figures.",
        "preview":"Provided http://127.0.0.1:8765/6.7.html was not opened; direct current Markdown and actual context SVG renders were inspected. No browser/mobile layout check claimed."},
    "dependency_decision":"No substantive changed prerequisite affects any of6.7's six original claim groups.6.5's graphic display of smile is explicitly tied to U+1F642, and6.6 script normalization preserves the original content/control semantics. The seven octets and Unicode/token/stream limitations continue to be supported by prior primary-source and CPU evidence, not by treating the current textbook as authority.",
    "authority_and_execution_reuse":{
        "actually_checked":"Recomputed and asserted all52 own prior artifact SHA-256 hashes including official original source snapshots, historical tokenizer/split inputs, original fence/bootstrap/stdout/environment and bounded checks/results/receipts. Also compared live data.py/tokenization.py/text.py/original tokenizer.json byte-for-byte with my actual frozen inputs; all identical.",
        "supports_retained":"CPython encode/decode/replace/incremental final contracts; RFC3629 bit placement/malformed-byte restrictions; HF ByteLevel flatten-before-decode and training pair counts; UAX29 scalar/grapheme distinction; prior exact seven-byte/fence-prefix/stream/control-ID/fixture assertions. These support the same unchanged six claim statements and scope; previous support locators remain exact.",
        "not_redone":"No Python fence/tokenizer test/model evaluation reruns, network source refetches, data preparation, downloads, GPU or model training. The new program only snapshots/hashes/computes textual diff and renders local SVGs."},
    "figures":figures, "facts_path":rel(OUT/'facts.json'), "facts_sha256":sha(OUT/'facts.json'),
    "execution_command":FACTS["command"] + " > /tmp/phase4-6_7-context-reinspection-stdout.txt 2> /tmp/phase4-6_7-context-reinspection-stderr.txt",
    "execution_result":"Exit0;52 prior artifact hash assertions and4 live-input equality assertions succeeded; source snapshots/diffs saved; both Inkscape processes exited0. Owner subsequently personally read/viewed the scope above.",
    "stdout_storage":"Exact original /tmp stdout/stderr byte copies retained permanently as context-reinspection-20261006/stdout.txt and stderr.txt.",
    "environment":{"python":FACTS["python"],"cwd":FACTS["cwd"],"device":"CPU/file checks and SVG rendering","inkscape":"1.4 (e7c3feb100, 2024-10-09)"}}
write(OUT / "receipt.json", inspection)

new_artifacts = []
paths = sorted(p for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('checker-'))
paths += [BASE/'code/context_reinspection_20261006.py', Path(__file__).resolve()]
for p in paths:
    label=p.relative_to(BASE).as_posix()
    aid = receipt_id if p.name=='receipt.json' else "a-context-20261006-" + label.replace('/','-').replace('.','-')
    kind = "figure_render" if p.suffix=='.png' else "code" if p.suffix=='.py' else "execution" if p.name in ('receipt.json','facts.json','stdout.txt') else "source_snapshot"
    artifact={"id":aid,"kind":kind,"path":rel(p),"sha256":sha(p),"description":"Original owner's actual narrow context reinspection evidence: "+label}
    if kind=='execution':
        artifact.update(command=inspection["execution_command"],result=inspection["execution_result"],environment=inspection["environment"])
    if kind=='figure_render':
        figure=next(f for f in figures if f["render_path"]==rel(p))
        artifact.update(render_command_argv=figure["command_argv"], source_sha256=figure["source_sha256"],
                        inspection_note=figure["inspection_note"], viewer_tool=figure["viewer_tool"])
    new_artifacts.append(artifact)

report=prior
report["reviewed_on"]="2026-10-06"
report["verdict"]="pass"
report["artifacts"]+=new_artifacts
report["review_scope"]["context_reinspection"]={"receipt_artifact_id":receipt_id,"receipt_path":rel(OUT/'receipt.json'),"receipt_sha256":sha(OUT/'receipt.json'),"actual_scope":inspection["actual_read_scope"],"dependency_decision":inspection["dependency_decision"],"unresolved_questions":[]}
report["review_scope"]["scope_read"].append("2026-10-06 same original owner context reinspection; current6.7/6.5/6.6 personally read,6.3 fingerprint retained; both current6.5 SVGs rendered/viewed; exact actual dependencies and reuse recorded in receipt.")
report["review_scope"]["figure_inspection"]["context_graphics"]="Two6.5 figures now personally rendered/viewed solely for adjacent byte/token-unit context;6.7 itself still has no figure. See context receipt."
for claim in report["claims"]:
    claim["artifact_ids"].append(receipt_id)
    claim["context_reinspection"]="Same original owner confirmed current prerequisite changes preserve this claim's existing support/scope; primary sources and previous executions reused after all recorded hashes checked, without claiming a new execution."
report["checks"]["factual_accuracy"]["details"]+=" Same-owner2026-10-06 context reinspection found no claim impact; see explicit dependency receipt."
report["checks"]["figure_consistency"]["details"]+=" Current6.5 context figures personally rendered/viewed; their unit correspondence is compatible with6.7, not treated as its figures."
report["checks"]["source_verification"]["details"]+="2026-10-06 reinspection actually recomputed52 own artifact hashes plus4 live-input comparisons; all prior primary sources/code/input/execution evidence remains byte-identical."
report["checks"]["limitations"]["details"]+=" Narrow context reinspection does not revalidate unrelated BPB ranking or all chapter text."
report["revision_history"].append({"at":inspection["completed_at"],"owner":report["reviewer_task"],"reason":"Necessary6.5/6.6 context reinspection after adjacent revisions","priorhistory_path":FACTS["priorhistory_path"],"priorhistory_sha256":FACTS["priorhistory_sha256"],"receipt_artifact_id":receipt_id,"receipt_path":rel(OUT/'receipt.json'),"receipt_sha256":sha(OUT/'receipt.json'),"actual_reinspection":True,"verdict":"pass","issues":[]})
write(ROOT/'docs/technical-reviews/6.7.json',report)
print(json.dumps({"verdict":report["verdict"],"report_sha256":sha(ROOT/'docs/technical-reviews/6.7.json'),"priorhistory_path":FACTS["priorhistory_path"],"priorhistory_sha256":FACTS["priorhistory_sha256"],"receipt_artifact_id":receipt_id,"receipt_path":rel(OUT/'receipt.json'),"receipt_sha256":sha(OUT/'receipt.json'),"unresolved_questions":[]},ensure_ascii=False,indent=2))
