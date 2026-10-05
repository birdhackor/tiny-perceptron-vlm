"""Write this reviewer's complete new report; never read a previous report."""
import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PREFIX = OUT.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_14_9"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_raw = (OUT / "section-original.md").read_bytes()
current = (ROOT / "course/chapters/14.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", current))
i = next(i for i,h in enumerate(headers) if h[0].startswith(b"## 14.9 "))
current_section = current[headers[i].start():headers[i+1].start()]
assert current_section == source_raw, "14.9 changed; cannot issue report for old source"
assert sha(ROOT / "course/figures/rewrite-14-9-nearby-angles.svg") == sha(OUT / "figure-original.svg")
numeric = json.loads((OUT / "geometry-results.json").read_bytes())
paper_prov = json.loads((OUT / "primary-source-provenance.json").read_bytes())
input_prov = json.loads((OUT / "input-provenance.json").read_bytes())

artifacts = []
execution = {
    "geometry-results.json": ("geometry-execution", numeric["command"], "Exit 0; all numeric, dot-product, content-distance and original SVG assertions passed.", numeric["environment"]),
    "geometry.stdout.txt": ("geometry-stdout", numeric["command"], "Exit 0; actual JSON stdout from the CPU check.", numeric["environment"]),
    "geometry.stderr.txt": ("geometry-stderr", numeric["command"], "Exit 0; stderr is empty.", numeric["environment"]),
    "source-fetch.stdout.txt": ("source-fetch", "bash docs/technical-reviews/artifacts/phase4-14_9-independent/fetch_primary.sh", "Exit 0; fetched explicit-version primary PDFs with verified TLS and recorded their SHA-256.", {"device":"CPU; source acquisition only", "tools":"curl; pdftotext (Poppler)"}),
    "source-fetch.stderr.txt": ("source-fetch-stderr", "bash docs/technical-reviews/artifacts/phase4-14_9-independent/fetch_primary.sh", "Exit 0; actual curl progress stderr retained.", {"device":"CPU; source acquisition only", "tools":"curl; pdftotext (Poppler)"}),
    "primary-snapshot.stdout.txt": ("primary-snapshot-execution", ".venv/bin/python docs/technical-reviews/artifacts/phase4-14_9-independent/snapshot_primary.py", "Exit 0; exact inspected source fragments and selected version/section PDF pages preserved.", {"python":"3.13.5", "device":"CPU", "tools":"pdfseparate and pdfunite (Poppler)"}),
    "render.stdout.txt": ("render-stdout", "bash docs/technical-reviews/artifacts/phase4-14_9-independent/render_figure.sh", "Inkscape produced the viewed original diagram; subsequent Chromium desktop capture timed out, script exit 124; no browser screenshot produced.", {"device":"CPU rendering", "inkscape":"1.4 (e7c3feb100, 2024-10-09)", "chromium":"151.0.7922.173 Debian 13"}),
    "render.stderr.txt": ("render-stderr", "bash docs/technical-reviews/artifacts/phase4-14_9-independent/render_figure.sh", "Actual combined logs retained, including overlapping initial-mobile and retry-desktop writes and NUL gaps. Does not establish successful browser rendering.", {"device":"CPU rendering", "inkscape":"1.4", "chromium":"151.0.7922.173"}),
    "render.initial.stdout.txt": ("render-initial-stdout", "bash docs/technical-reviews/artifacts/phase4-14_9-independent/render_figure.initial.sh", "Initial script produced Inkscape image; browser screenshots absent; remaining own mobile process group later terminated, script exit 137.", {"device":"CPU rendering", "inkscape":"1.4", "chromium":"151.0.7922.173"}),
    "render.initial.stderr.txt": ("render-initial-stderr", "bash docs/technical-reviews/artifacts/phase4-14_9-independent/render_figure.initial.sh", "Actual first captured logs preserved before retry/initial process overlap; Inkscape wrapping and Chromium DBus/network messages retained.", {"device":"CPU rendering", "inkscape":"1.4", "chromium":"151.0.7922.173"}),
}
custom_ids = {
    "figure-original.svg":"figure-original", "figure-render.png":"figure-viewed",
    "review-record.md":"inspection-record", "render-events.json":"render-events",
    "input-provenance.json":"input-provenance", "primary-source-provenance.json":"primary-provenance",
    "roformer-primary-selected-pages.pdf":"roformer-primary-pages", "yarn-primary-selected-pages.pdf":"yarn-primary-pages",
    "yarn-page3-inspected.png":"yarn-printed-equation-view", "verify_geometry.py":"geometry-code",
    "section-original.md":"section-original", "chapter14-frozen-input.md":"chapter-frozen-input",
    "dependency-14.1.md":"dependency-14_1", "dependency-14.8.md":"dependency-14_8",
}
for path in sorted(OUT.iterdir()):
    assert not path.is_symlink(), "Permanent proof must not follow runtime symlinks"
    if not path.is_file():
        continue
    name = path.name
    if name in execution:
        identifier, command, result, environment = execution[name]
        artifact = {"id":identifier, "kind":"execution", "path":f"{PREFIX}/{name}", "sha256":sha(path),
                    "description":result, "command":command, "result":result, "environment":environment}
    else:
        identifier = custom_ids.get(name, name.replace(".","-").replace("_","-"))
        if path.suffix in {".py", ".sh", ".html"}:
            kind = "code"
        elif name == "figure-render.png" or name == "yarn-page3-inspected.png":
            kind = "figure_render"
        elif name in {"review-record.md", "render-events.json"}:
            kind = "derivation"
        else:
            kind = "source_snapshot"
        artifact = {"id":identifier, "kind":kind, "path":f"{PREFIX}/{name}", "sha256":sha(path),
                    "description":f"This reviewer's preserved {name}; scope and acquisition/inspection details in review-record.md and provenance JSON."}
    artifacts.append(artifact)

sources = []
for identifier, record in zip(("roformer-v5", "yarn-v3"),paper_prov["records"]):
    sources.append({"id":identifier,"kind":"paper","title":record["title"],"url":record["url"],
                    "version":record["version"],"accessed_on":record["accessed_on"],
                    "authority_reason":record["authority_reason"],"verified":True,"checked_original":True,
                    "inspection_note":record["checked_original_scope"] + " " + (paper_prov["source_limit"] if identifier=="yarn-v3" else "Artificial 60°/15° example is not the actual Θ configuration of RoFormer."),
                    "original_acquired_pdf_sha256":record["original_acquired_pdf_sha256"],
                    "snapshot_artifact_id":"roformer-primary-pages" if identifier=="roformer-v5" else "yarn-primary-pages"})
sources.extend([
    {"id":"geometry-derivation","kind":"derivation","title":"Independent angular and vector derivation","verified":True,
     "details":"For one pair, phase=mθ and Δphase=(m1−m0)θ. Replacing each position by m/d scales every gap by 1/d. 360°/60°=6 positions; 360°/15°=24. At Δm=1, /2 gives 30° and 7.5°; /4 gives 15° and 3.75°. Orthogonal rotation preserves norms and distances. SVG uses screen y down, so atan2(y1−y2,x2−x1) returns the mathematical angle."},
    {"id":"geometry-run","kind":"execution","title":"Bounded independent CPU geometry/figure run","verified":True,"artifact_id":"geometry-execution"},
])

def evidence(identifier, locator, supports):
    return {"source_id":identifier,"locator":locator,"supports":supports}

def concept(identifier, statement, location, scope, refs, ids=()):
    return {"id":identifier,"kind":"concept","statement":statement,"location":location,"scope":scope,
            "status":"verified","evidence":refs,"artifact_ids":list(ids)}

def numeric_claim(identifier,statement,location,scope,refs,expected,observed,details,ids):
    return {"id":identifier,"kind":"numeric","statement":statement,"location":location,"scope":scope,"status":"verified",
            "evidence":refs,"artifact_ids":ids,
            "verification":{"method":"executed","expected":expected,"observed":observed,"details":details,
                            "tolerance":"Exact scalar degree arithmetic; rotation/dot/distance absolute tolerance 1e-12; SVG angle and length absolute tolerance 0.0001 degrees/px."}}

claims=[
    concept("paired-qk-rotation","RoPE rotates pairs of projected Q/K features as two-dimensional vectors; [1,0,1,0] can be represented as two right-pointing arrows.","course/chapters/14.md:311",
            "A geometric explanation of even-dimensional feature pairs. The clocks are a metaphor for features, not additional card data or an actual model-rate setting.",
            [evidence("roformer-v5","§3.2.1–3.2.2, original pp.4–5, Eqs.12–16","Q/K projection followed by position-index-dependent rotations in d/2 paired 2D subspaces." )], ["roformer-primary-pages","section-original"]),
    numeric_claim("rates-and-periods","The artificial 60° and 15° per-position rates complete a full 360° turn after 6 and 24 positions respectively.","course/chapters/14.md:313; 330",
                  "Hand-picked demonstration frequencies, measured in degrees per position. No model configuration or capability measurement.",
                  [evidence("yarn-v3","§2.2, original p.3, Eq.8 and wavelength definition","Period/wavelength is 2π/θ, the number of token positions for one complete rotation."), evidence("geometry-derivation","360/60 and 360/15; review-record.md items 2–3","Exact periods and units in the stated demonstration.")],
                  "360/60=6 and 360/15=24.","Both exact period assertions passed.","verify_geometry.py checks speed*period==360 for both rates; rotations return to the original direction at one full period.",["geometry-code","geometry-execution","yarn-primary-pages"]),
    numeric_claim("uniform-angle-scaling","Dividing positions by 2 halves both adjacent phase gaps to 30° and 7.5°; dividing by 4 gives 15° and 3.75°.","course/chapters/14.md:309; 315; 325",
                  "Only the specified coordinate operation on fixed right-pointing original features; it is not an empirical language-model result.",
                  [evidence("roformer-v5","§3.2.1–3.2.2, Eqs.12–16","The rotation phase is position times rate; relative phase depends on the coordinate gap."), evidence("geometry-derivation","Δphase=Δmθ/d; geometry-results.json /geometry_checks","Applying /2 or /4 to the positions gives the stated angles for both groups." )],
                  "Original angle gaps [60,15]°, /2 [30,7.5]°, /4 [15,3.75]°.","All six angular cases passed; translated 10/11 coordinate variants also preserve the same gap-dependent dot products.","The independent CPU calculation distinguishes scalar angles from the cos(angle) dot products, verifies units, and uses a translated-position variation to check relative-gap behavior.",["geometry-code","geometry-execution","geometry-stdout","roformer-primary-pages"]),
    concept("compression-and-content","Smaller phase gaps do not remove cards or force their content features to coincide; multi-pair rotations still carry position-dependent information.","course/chapters/14.md:321",
            "The text says content features can differ. It does not guarantee that trained attention distinguishes every compressed pair or that the chosen two rates encode unlimited positions uniquely.",
            [evidence("roformer-v5","§3.2.2, Eq.16 and orthogonal-matrix discussion, original p.5","Rotations act on projected content vectors and preserve their norms; they are not token merging."),evidence("geometry-derivation","review-record.md item 4; verify_geometry.py distinct-content variant","A shared rotation preserves the nonzero squared distance of two distinct vectors; position compression alone does not delete data.")], ["geometry-execution","roformer-primary-pages"]),
    concept("nearby-distance-tradeoff","Uniformly compressing RoPE coordinates changes nearby phase relations as well as distant ones; previously learned relations can need adaptation, with stronger compression making the original one-position gap smaller.","course/chapters/14.md:321",
            "Conditional mechanism/tradeoff of changing position features, not a claimed measured degradation, a universal need for fine-tuning, or a guarantee about recovery.",
            [evidence("yarn-v3","§3.1–3.2, original pp.4–5: loss of high-frequency information and relative local distances","Uniform scaling changes high-frequency/local-distance information; the paper motivates preserving dimensions important for nearby order."),evidence("geometry-derivation","Δphase=Δmθ/d","The local gap shrinks monotonically as the coordinate divisor increases.")], ["yarn-primary-pages","inspection-record"]),
    concept("periodicity-and-frequency-dependent-design","A fast group changes phase more per short displacement but wraps sooner; slow groups vary over a longer span. A single periodic hand cannot identify all distances, and different frequencies can receive different amounts of scaling when extending context.","course/chapters/14.md:323; 325",
            "Introductory design direction toward YaRN's frequency-dependent interpolation. It neither asserts that two rates solve all aliasing nor defines the whole YaRN method or its attention-temperature component.",
            [evidence("roformer-v5","§3.2.1–3.2.2, Eqs.12–16","Multiple predetermined rates and periodic cos/sin rotations."),evidence("yarn-v3","§2.2 Eq.8; §3.2 original pp.4–5 and Eqs.10–13","Short wavelengths are treated differently from long wavelengths, with a transition band mixing scaled/unscaled frequencies."),evidence("geometry-derivation","review-record.md item 6; position-6 bounded variant","At 6 positions the artificial fast group returns to its start while the slow group differs; the example has no universal uniqueness guarantee.")], ["geometry-execution","yarn-primary-pages","roformer-primary-pages"]),
    numeric_claim("figure-consistency","The diagram shows original versus /2 angles in the correct fast/slow row order: 60→30° and 15→7.5°; dashed right-pointing baselines are position 0 and the compressed adjacent gap is 0.5.","course/chapters/14.md:317; 319; course/figures/rewrite-14-9-nearby-angles.svg",
                  "Original SVG coordinate geometry and its actual Inkscape render were checked. Chromium browser and mobile page captures failed and are not counted as visual evidence.",
                  [evidence("geometry-run","geometry-results.json /figure_checks","SVG orange-vector angles in screen-corrected coordinates match all four labeled values."),evidence("geometry-derivation","atan2(y1−y2,x2−x1); review-record.md item 7","Correct screen-to-mathematical y-axis conversion and the expected 64 px radii.")],
                  "Four orange angles [60,30,15,7.5]°, all blue baselines horizontal rightward and length 64 px.","Observed [59.9999884324,30.0000115676,14.9999727506,7.5000176602]°, with all angle/length errors below 0.0001; native Inkscape image viewed and all labels visible.","CPU XML endpoint verification plus actual render/view; diagram labels, colors, arrow order, and 0.5 footer checked against manuscript. Actual unsuccessful browser attempts and failed image-view requests are preserved separately.",["geometry-code","geometry-execution","figure-original","figure-viewed","render-events","render-stdout","render-stderr"]),
]

report={
    "schema_version":1,"review_stage":"technical","lesson_id":"14.9","source":"course/chapters/14.md#14.9",
    "source_sha256":hashlib.sha256(source_raw).hexdigest(),"verdict":"pass","reviewer_task":TASK,"reviewer_context":"fresh","author_tasks":[],
    "figure_sha256":{"course/figures/rewrite-14-9-nearby-angles.svg":sha(OUT/"figure-original.svg")},
    "read_scope":input_prov["actual_read_scope"],"frozen_input":input_prov["frozen_input"],"introduction_reviewed":False,
    "independence":{"old_reports_read":False,"author_review_or_revision_summaries_read":False,"source_notes_read":False,
                    "primary_locator_indices_are_evidence":False,"method":"Personally read original section, necessary prior manuscript and primary papers; own short CPU check; actual native image render/view."},
    "fence_coverage":{"python_fences":0,"other_fences":0,"details":"No original fences in 14.9. Independent bounded CPU math/geometry check only; no ordinary API inventory applies."},
    "empirical_scope":{"model_measurements_present":False,"details":"No model scores, empirical denominator claims, actual frequency configurations, training, evaluation, or engineering-stage artifact acceptance in 14.9."},
    "artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","details":"Seven grouped substantive claims checked independently: pair rotation, periods, uniform scaling, content distinction, local-distance tradeoff, multirate/periodic design and diagram geometry.","claim_ids":[c["id"] for c in claims]},
        "numeric_verification":{"status":"pass","details":"All stated scalar angles and periods, /4 exercise, relative-gap variants and SVG axes/lengths passed the saved bounded CPU assertions; units and tolerances are explicit.","claim_ids":["rates-and-periods","uniform-angle-scaling","figure-consistency"]},
        "figure_consistency":{"status":"pass","details":"Original SVG was frozen, rendered by Inkscape and actually viewed; its labeled phase angles and geometry agree with text. Chromium desktop/mobile attempts did not produce images, so browser page/mobile presentation remains unverified and is explicitly excluded from this pass.","claim_ids":["figure-consistency"]},
        "source_verification":{"status":"pass","details":"Personally checked first-page version stamps and relevant original paragraphs/equations of RoFormer v5 and YaRN v3. Permanent selected-page PDFs, raw excerpts, original PDF SHA and source/locator support records are preserved. YaRN v3's inconsistent §2.3 Eq.9 is explicitly excluded as support for compression.","claim_ids":["paired-qk-rotation","rates-and-periods","uniform-angle-scaling","compression-and-content","nearby-distance-tradeoff","periodicity-and-frequency-dependent-design"]},
        "limitations":{"status":"pass","details":"The rates are artificial; CPU work verifies geometry only; no learning, quality, universal positional uniqueness or empirical YaRN capability is claimed. Native diagram inspection was completed; browser/phone presentation was not verified. The paper Eq.9 limitation does not affect the cited §3.1–3.2 tradeoff or the independently derived /2 and /4 arithmetic.","claim_ids":["rates-and-periods","compression-and-content","nearby-distance-tradeoff","periodicity-and-frequency-dependent-design","figure-consistency"]},
    },
    "unresolved_substantive_questions":[],
    "verification_limitations":["Chromium desktop/mobile wrapper capture failed; no course-page/mobile-layout verification.",paper_prov["source_limit"]],
}
path=ROOT/"docs/technical-reviews/14.9.json"
path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
# Check only the newly generated report's identity; never fall back to older reports.
new=json.loads(path.read_bytes())
assert new["reviewer_task"]==TASK and new["source_sha256"]==hashlib.sha256(source_raw).hexdigest()
print(json.dumps({"report":path.relative_to(ROOT).as_posix(),"reviewer_task":new["reviewer_task"],
                  "source_sha256":new["source_sha256"],"report_sha256":sha(path),"claims":len(claims),"verdict":new["verdict"]},ensure_ascii=False,indent=2))
