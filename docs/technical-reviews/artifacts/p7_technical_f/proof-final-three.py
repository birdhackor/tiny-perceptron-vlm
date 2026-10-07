"""Owner's final three guide judgments; never reads another owner's review."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[4]
F = R / "docs/technical-reviews/artifacts/p7_technical_f"
registry = json.loads((F / "source-registry.json").read_text())
guides = json.loads((F / "source-additions-guides.json").read_text())
env = json.loads((F / "source-additions-environment.json").read_text())

def repo(i, p, note):
    return dict(id=i, kind="repository_code", title=p, path=p, verified=True, inspection_note=note)

def original(i, p, url, version, note):
    return dict(id=i, kind="official_source", title=p, url=url, version=version,
        sha256=hashlib.sha256((R / p).read_bytes()).hexdigest(), verified=True,
        authority_reason="Original publisher license/documentation, personally checked at the stated location.",
        checked_original=True, inspection_note=note, accessed_on="2026-10-07")

def artifact(i, name, note):
    return dict(id=i, kind="execution", path=f"docs/technical-reviews/artifacts/p7_technical_f/{name}.json", description=note, result=note)

def ev(s, loc, support):
    return dict(source_id=s, locator=loc, supports=support)

def claim(i, k, statement, location, scope, evidence, ids, observed=None, den=None):
    d = dict(id=i, kind=k, statement=statement, location=location, scope=scope,
        status="verified", evidence=evidence, artifact_ids=ids)
    if k in ("numeric", "software", "empirical"):
        d["verification"] = dict(method="executed", expected=statement, observed=observed, details=scope)
        if k == "numeric": d["verification"]["tolerance"] = "Integer/hash counts exact; MiB division uses 1048576 bytes."
        if k == "empirical": d["verification"]["denominators"] = den
    return d

def checks(rows):
    return {k: dict(status=s, details=d, claim_ids=ids) for k, s, d, ids in rows}

def save(d):
    p = subprocess.run([sys.executable, str(F / "page-record.py")], input=json.dumps(d, ensure_ascii=False),
        text=True, cwd=R, capture_output=True)
    print(p.stdout, end=""); print(p.stderr, end="", file=sys.stderr); assert p.returncode == 0

if sys.argv[1] == "validation":
    sources = [registry[k] for k in ("split", "causal", "eos-criteria", "torch-autograd")]
    sources += [
        repo("moe-raw", "docs/selftrained/results/public-raw/moe/test/metrics.json", "Only raw count/per_task/continuation fields; old review booleans are excluded."),
        repo("dense-raw", "docs/selftrained/results/public-raw/dense/test/metrics.json", "Same raw-only aggregation as MoE; no author's integrity summary."),
        repo("local", "scripts/selftrained/train_local_stage.py", "Current source chooses fresh best/resume latest and manages trainer argv; historical width16 fixture is distinct from production."),
        repo("gpu-result", "docs/gpu-smoke-result.json", "Read original GPU/torch/count/40-to80/max-difference fields only; checks and sha_verified booleans not proof."),
        repo("gpu-runner", "scripts/gpu_smoke_check.py", "140–180 resumes at41 through80 and compares tensors via max absolute difference; no private-state recomputation."),
        repo("kernel", "scripts/check_notebooks.py", "Independent NotebookClient kernel per notebook, CLI mode/workers/output."),
        repo("sync", "scripts/build_course.py", "--check synchronization contract only."),
        repo("reader-gate", "scripts/check_course_reviews.py", "Parsed checker program, not another reader's report or pass."),
        repo("technical-gate", "scripts/check_technical_reviews.py", "Parsed schema/hash checker, not actual historical passing judgments."),
        repo("round-gate", "docs/review-tools/check_review_round.py", "Current-round identity/source gate contract, not a completed-stage result."),
    ]
    arts = [
        artifact("raw", "validation-historical-counts-and-wrapper", "Own raw historical test aggregation and synthetic wrapper receipts."),
        artifact("gpu", "validation-gpu-history-scope", "Own CPU31584 count and historical L4/40-to80/zero-difference fields; no new GPU run."),
        artifact("commands", "validation-command-contracts", "Five Python ASTs and real notebook/ipykernel/pytest help; no whole-course execution."),
        artifact("public-cpu", "environment-raw-cpu-call-audit", "Nine original public CPU argv/result/stdout fingerprints checked, eight chats plus one append-history."),
        artifact("archive", "asset-v2-actual-archive-counts", "Actual tar/JSONL28876/2435/3734 count."),
        artifact("release", "readme-historical-release-and-runtime", "Original16 exports and selected1000 vs completed4000/10000 boundaries."),
    ]
    cs = [
        claim("scope", "concept", "Runnable cells, fixed weights, model learning, held-out capability and publication are distinct forms of evidence.", "six verification questions/table", "Mechanism/hash checks alone do not establish unseen OCR, ASR or open conversation ability.", [ev("split", "train/validation/test separation", "Held-out capability must use appropriate separate data"), ev("causal", "target shift/ignored labels", "Loss-scored outputs are distinct from input context")], []),
        claim("v2-history", "empirical", "Both final v2 models test3734; MoE tools0/276 and voice42/90; Dense tools7/276 and voice37/90; finite clothing357/360 vs356/360 and OCR252/324 vs281/324.", "selftrained boundary", "Personally aggregated existing raw historical fields, not a new test or open-ended quality claim. Voice continuation26/60 vs24/60; failures remain failures.", [ev("moe-raw", "count/per_task/voice continuation", "Original numerators/denominators"), ev("dense-raw", "same", "Original comparison")], ["raw", "archive"], "Exact3734 and stated raw ratios", {"test_each":3734,"tool_each":276,"voice_each":90,"continuation_each":60,"clothing_each":360,"ocr_each":324}),
        claim("wrapper", "empirical", "Stored local wrapper smoke receipts use width16,4880 trainable parameters, pretrain1 step/1 row, SFT1 step/2 rows, resumed2 steps/2 rows; full history selects1000 after4000/10000.", "local wrapper/history boundary", "Historical synthetic orchestration evidence only; source inspection supports best/latest selection and managed args, not full production performance or a fresh remote run.", [ev("local", "run/managed args/local source", "Fresh best and resume latest contract")], ["raw", "release"], "width16/4880;1/1/2 step fixtures;selected1000,completed4000/10000", {"pretrain_train_rows":1,"sft_train_rows":2,"resume_train_rows":2,"pretrain_steps":1,"sft_steps":1,"resume_steps":2}),
        claim("gpu-history", "empirical", "Original GPU smoke has31584 parameters,L4,Torch2.14.1+cu126 and40-to80 resume with reported tensor max difference0.", "historical GPU result", "Own CPU instantiated parameter count and raw-field/source comparison; did not retrieve private checkpoints, recalculate tensor equality, run GPU or publish to HF.", [ev("gpu-result", "parameters/runtime/resume difference", "Historical raw report"), ev("gpu-runner", "140–180", "Definition of compared tensor difference")], ["gpu"], "31584 exact;historical40→80/max difference0;fresh_gpu=false", {"historical_initial_steps":40,"historical_resumed_steps":80,"historical_smoke_runs":1}),
        claim("commands", "software", "The listed synchronization/review-round/kernel/pytest command interfaces exist; public raw CPU evidence contains8 chats and1 history append.", "maintenance commands/outputs", "AST/help and raw receipts only. No reading peer judgments, rerunning all287 notebooks/all tests, completing stage gates or equating local kernels with Colab.", [ev("sync", "CLI--check", "Sync contract"),ev("kernel", "CLI/NotebookClient", "Fresh-kernel interface"),ev("round-gate", "CLI/identity contract", "Current-source round interface")], ["commands", "public-cpu"], "5 ASTs parse;3 real helps;9 raw public CPU calls/status/stdout hashes checked"),
    ]
    save(dict(page_id="validation", verdict="pass", sources=sources, artifacts=arts, claims=cs,
        checks=checks([
            ("factual_accuracy","pass","The six evidence scopes separate runnable mechanisms from capability; historical wrapper/GPU statements retain their actual scope.",["scope","wrapper","gpu-history"]),
            ("numeric_verification","pass","Exact original counts and CPU parameter count checked; all ratios have original denominators.",["v2-history","gpu-history"]),
            ("figure_consistency","not_applicable","No SVG in frozen page. Six-row table and three scope regions viewed at both widths; long code ends not all visible.",[]),
            ("source_verification","pass","Original metrics/receipts/current programs checked; no integrity-review conclusion used.",["v2-history","commands"]),
            ("limitations","pass","No fresh GPU, remote training, upload, all-notebook run or private checkpoint equality claim. Independent20.8 issue persists on its own page.",["wrapper","gpu-history","commands"]),
        ]), issues=[], page_receipt="docs/technical-reviews/artifacts/p7_technical_f/page-validation-view.json",
        page_view_details="Nine necessary table/scope captures personally viewed, desktop/mobile; selected regions only.",
        provenance_notes=["Raw old checks/HF sha_verified booleans incidentally exposed; excluded as judgments.","Wrapper teacher-forced selection/completed metadata was visible; only raw execution counts and source contract are evidence.","Source AST/help establishes CLI contracts, not the success of the listed full workflows."]))

elif sys.argv[1] == "training-assets":
    originals = [
        ("tinystories","tinystories","https://huggingface.co/datasets/roneneldan/TinyStories/raw/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md","f54c09fd23315a6f9c86f9dc80f725de7d8f9c64","frontmatter license cdla-sharing-1.0"),
        ("poetry","chinese-poetry","https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE","b8594f81a89752241442f2ce267d6f66f96704ee","MIT license heading"),
        ("ultrachat","ultrachat","https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/raw/8049631c405ae6576f93f445c6b8166f76f5505a/README.md","8049631c405ae6576f93f445c6b8166f76f5505a","frontmatter license mit"),
        ("ultrafeedback","ultrafeedback","https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/raw/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85/README.md","3949bf5f8c17c394422ccfab0c31ea9c20bdeb85","frontmatter license mit"),
        ("pku","pku","https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/raw/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md","9421ffafec3fa40a1f1a7d567b4d525079477ecb","line39 cc-by-nc-4.0"),
        ("fashion","fashion","https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE","b2617bb6d3ffa2e429640350f613e3291e10b141","MIT heading; raw original README60000 images separately checked"),
        ("gsm","gsm8k","https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE","3101c7d5072418e28b9008a6636bde82a006892c","MIT heading; original upstream README line15 human problem writers"),
        ("fsdd","fsdd","https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/26eb9aaf76e81b692f806f9140c2d2777410d7a1/README.md","26eb9aaf76e81b692f806f9140c2d2777410d7a1","4/53 mono8kHz;88 CC-BY-SA4.0"),
    ]
    sources=[registry[k] for k in ("git-lfs","git-clone","split")]+[guides["whisper-audio"]]
    for i,n,u,v,l in originals:
        sources.append(original(i,"outputs/p7_technical_f/source-cache/basic-assets-originals/"+n+".txt",u,v,l))
    sources += [
        original("powershell","outputs/p7_technical_f/source-cache/basic-assets-originals/powershell-env.md","https://raw.githubusercontent.com/MicrosoftDocs/PowerShell-Docs/main/reference/7.5/Microsoft.PowerShell.Core/About/about_Environment_Variables.md","7.5 fetched original SHA snapshot","101–107 $Env:name=value syntax; not a Windows execution"),
        repo("manifest","assets/training/manifest.json","Eight immutable archive digests and counts; physically verified LFS objects, not status metadata."),
        repo("fetch-assets","scripts/fetch_training_assets.py","Manifest IDs, git-lfs pull include, unpack checks all paths/hashes and existing-file refusal."),
        repo("builder-assets","scripts/build_training_assets.py","Deterministic sorted tar members/mtime0/mode644 and gzip mtime0; own rebuild all8."),
        repo("experiment-list","docs/course-experiments/plan.json","Only seven listed IDs/module/function/dependencies/assets; excludes report/status explanations."),
        repo("conversion-text","scripts/course_experiments/text.py","519–542 paired chat conversion/split and120 UTF8 limit."),
        repo("conversion-behavior","scripts/course_experiments/behavior.py","467–489 safer flags;668–692 last assistant chosen/rejected."),
        repo("conversion-modal","scripts/course_experiments/modalities.py","996–1020 per-class Fashion3/1/1;1030–1057 mono8k to16k windowed-sinc."),
        repo("conversion-distill","scripts/course_experiments/compression.py","667–686 GSM human #### result parsing; teacher output audited, not assumed GSM-trained."),
        repo("conversion-reasoning","scripts/course_experiments/applications.py","952 _gsm8k_pilot and1111 call; source identity only, not a new run."),
        repo("augmentation","scripts/selftrained/augment_voice_v2.py","286–306 by original audio;318–333 parent path;358–372 preserved source IDs; default8 variants."),
        repo("v2manifest","docs/selftrained/v2-manifest.json","Finite Fashion/OCR/MInDS package identity; original payload counts personally checked."),
    ]
    arts=[
        artifact("raw-eight","training-assets-eight-raw-audit","All eight actual LFS objects SHA/members/counts; WAV8000Hz and Fashion60000 parquet."),
        artifact("publishers","training-assets-publisher-originals","Anonymous fixed original publisher text bytes match original archive copies; narrow personally read extracts only."),
        artifact("records","training-assets-record-boundaries","Tiny512 story boundaries and Chinese365 complete paragraph records, original366-row source."),
        artifact("lfs","training-assets-lfs-smudge-contract","Real GitLFS smudge help skip-download contract; no new clone."),
        artifact("unpack","training-assets-real-unpack-contract","Real eight-pack hash-safe unpack174 members/172 writes, repeat0, synthetic differing file refused unchanged."),
        artifact("recipes","training-assets-recipe-mapping","Seven selected module/function ASTs and asset/dependency IDs only."),
        artifact("rebuild","training-assets-deterministic-rebuild","Own CPU rebuild all eight tarballs/manifest byte-for-byte equal originals."),
        artifact("v2-scope","training-assets-v2-recording-boundary-v2","Own finite OCR/intents/no-speaker/license-entry/split-overlap fields;558 is audio paths including generated versions."),
        artifact("original-audio","training-assets-v2-original-recordings-v4","Real nonaugmentation original audio62/15/30;2790 train rows310 original+2480 derived;49 near-duplicate groups."),
        artifact("v2-archive","asset-v2-actual-archive-counts","Actual67862431B v2 archive,8962 members and28876/2435/3734 records."),
    ]
    cs=[
        claim("eight-counts","numeric","Eight pilot training counts512+365+100+100+100+50+200+20=1447; archives30854837 bytes=29.4254656MiB.","eight-pack table","Actual8 LFS objects and raw record boundaries, not eight new model trainings. FSDD20 train plus20 validation/20 test raw8k audio; Fashion upstream60000 vs50 pilot rows.",[ev("manifest","eight entries archive_sha256/counts","Identity/count binding"),ev("fsdd","4/53","mono8k original"),ev("fashion","original README size","Upstream vs pilot size")],["raw-eight","records"],"1447 exact;174 tar members;60 FSDD wav8k;Fashion60000"),
        claim("licenses","concept","Dataset licenses retain their independent permissions; repository MIT does not relicense CC-BY-NC/CC-BY-SA/CDLA/OFL sources.","licenses/source boundary","Original publishers establish only stated license labels/attribution conditions, not blanket legal clearance or redistribution_approved booleans.",[ev("tinystories","frontmatter","CDLA-sharing1"),ev("poetry","LICENSE","MIT"),ev("ultrachat","frontmatter","MIT"),ev("ultrafeedback","frontmatter","MIT"),ev("pku","39","CC-BY-NC4"),ev("fashion","LICENSE","MIT"),ev("gsm","LICENSE","MIT"),ev("fsdd","88","CC-BY-SA4")],["publishers"]),
        claim("fetch-unpack","software","Named --asset choices come from manifest; existing identical members may be skipped, differing files refused; unpack validates all paths and hashes before writing.","download commands","Real local raw-object unpack/fixture and actual smudge help; no new GitHub download/Windows execution. PowerShell assignment checked in official original only.",[ev("fetch-assets","CLI/unpack_asset","IDs/hash/path/refusal"),ev("git-lfs","smudge skip option","Clone skips bulk payload"),ev("powershell","101–107","Environment assignment syntax")],["unpack","lfs"],"174 members verified;172 writes;repeatfirst0;differing synthetic file refused unchanged"),
        claim("recipes","software","Seven experiment IDs point to task-specific conversion functions; Fashion/voice/chat/preference/GSM need their corresponding labels and splits.","six-row recipe table/conversion discussion","AST mapping and personally located conversion functions only; no new experiment training. Upsampling does not create missing high-frequency speech information; safer preference is not automatic safety SFT.",[ev("experiment-list","seven selected entries","module/function/assets/dependencies"),ev("conversion-modal","996–1057","Fashion/image and8k-to16k audio conversion"),ev("conversion-behavior","467–489/668–692","Safety/preference labels"),ev("conversion-distill","667–686","Human GSM answer extraction"),ev("gsm","upstream README line15","Human problem writers")],["recipes","raw-eight"],"7 mapped functions parse;per-class3/1/1 source;resampling code actual16k target"),
        claim("rebuild","software","The fixed eight-pack tar/manifest recipe reproduces identical archive bytes without timestamps.","rebuild command","Own CPU rebuild of local unpacked raw pack data; wrote only outputs, never old checkout or published new assets.",[ev("builder-assets","deterministic tar/gzip helpers","Sorted normalized metadata")],["rebuild"],"8 archive SHA values and manifestSHA7a71a202… byte-equal"),
        claim("v2-recordings","numeric","Fixed v2 uses28876/2435/3734 rows; original voice recordings62/15/30, no known speaker ID,3 intents; OCR12 glyphs length1–4 with Sans/Serif training.","v2有限资料 scope","Original recording paths exclude audio_augmentation. Train558 audio paths include62 originals+496 generated;2790 rows310 original+2480 generated. Near-duplicate source groups49 differ from recording count. Not open OCR/ASR/speaker-independent proof.",[ev("v2manifest","scope/archive","Fixed finite package"),ev("augmentation","286–333/358–372","Original-parent identity and8 variants")],["v2-scope","original-audio","v2-archive"],"62/15/30 original recordings;train558 paths/49 groups/2790 voice rows;12 glyphs;3 intents;8962 archive files"),
    ]
    save(dict(page_id="training-assets",verdict="pass",sources=sources,artifacts=arts,claims=cs,
        checks=checks([
            ("factual_accuracy","pass","Pilot record/type/license boundaries match actual raw sources; v2 original recordings distinguished from groups/generated paths.",["licenses","recipes","v2-recordings"]),
            ("numeric_verification","pass","All8 archives personally verified;1447/29.4254656MiB and original62/15/30 precisely reconciled after preserved failed assertions.",["eight-counts","v2-recordings"]),
            ("figure_consistency","not_applicable","No SVG. Two3-column tables and v2 limit paragraph personally viewed at both widths.",[]),
            ("source_verification","pass","Fixed original publisher licenses and source conversion locations checked, never author review pass/status.",["licenses","recipes"]),
            ("limitations","pass","No new GPU/model training/external download required; local rebuild/unpack do not prove capability. Unknown speakers prevent speaker-isolated claim.",["fetch-unpack","rebuild","v2-recordings"]),
        ]),issues=[],page_receipt="docs/technical-reviews/artifacts/p7_technical_f/page-training-assets-view.json",
        page_view_details="All7 actual necessary desktop/mobile table/scope captures personally viewed; selected regions only.",
        provenance_notes=["Unit7 successful62 count was prewritten before underlying failed command returned; preserved old checkpoint, failed recording assertion artifact and second failed group assertion.","Unit8 explicitly corrected: real v4 nonaugmentation count62/15/30;558 derived-inclusive audio paths and49 near-duplicate groups. No historical data rewritten.","Original source status/check/training_permission field names incidentally appeared; no such approval flag used as proof.","LFS objects from old common Git directory read only; unpack/rebuild wrote only this owner's outputs. Author local README/metadata copied by unpack, never read as review answers."]))

elif sys.argv[1] == "publishing":
    sources=[registry["git-clone"],registry["git-lfs"],*env[:1]]
    sources += [
        original("pages-original","outputs/p7_technical_f/source-cache/github-pages-custom-workflows.html","https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages","2026-10-07 official fetched SHA snapshot","Original text109–129 deploy action needs build/pages write/id-token/environment; only these items and custom workflow introduction personally read."),
        repo("bootstrap","scripts/build_course.py","30–59 defaultbranch clone/pip editable;126–153 cells from common body."),
        repo("export","scripts/export_course.py","94–108 checked_notebook;248–255 pre-check before clearing;288 Colab revision;344–367 strict Zensical/build-info."),
        repo("site-check","scripts/check_site.py","Offline internal references/anchors/Notebook identity/Colab URLs, not external uptime."),
        repo("workflow",".github/workflows/pages.yml","Push main/manual; CPU workers1; export GITHUB_SHA; upload artifact then needs build deploy."),
        repo("dependencies","pyproject.toml","Direct groups notebook ipykernel/matplotlib/jupyterlab/nbclient; site Zensical; dev pytest/ruff/safetensors."),
        repo("reading-time","scripts/reading_time.py","validate --executed parser; current-source metadata contract, not empirical reading time."),
        repo("card-publish","scripts/selftrained/hf_transport.py","682–725 only README Add/single-operation commit, no weights update. Not called."),
        repo("card-descriptor","docs/selftrained/v2-jobs/release-repository-card.json","Only file hash/bytes/parent/write path; redistribution flag excluded."),
    ]
    arts=[
        artifact("cli","publishing-command-contracts","Real4 AST/help commands, bootstrap AST and direct dependency groups."),
        artifact("guard","publishing-notebook-output-guard-v2","Actual current checked_notebook accepts1 matched synthetic fixture and rejects3 changed/unexecuted/error fixtures; real http help."),
        artifact("workflow","publishing-workflow-contract","Own YAML BaseLoader checks and git check-ignore outputs; no GitHub Actions execution."),
        artifact("pages-doc","publishing-official-pages-source","Original official Pages document fetched; necessary short items only inspected."),
        artifact("card","publishing-card-and-export-boundary","Local21510B card SHA binding/header identity and current README-only publish function; no remote write."),
        artifact("exports","readme-historical-release-and-runtime","Original four×four=16 fixed exports/public manifest provenance; no new publication."),
        artifact("public-cpu","environment-raw-cpu-call-audit","Original9 CPU calls argv/result/stdout hash binding, not a fresh execution of all public tasks."),
        artifact("maintenance","validation-command-contracts","Existing personally checked review/kernel interfaces; no peer report inspection."),
    ]
    cs=[
        claim("version-boundary","concept","A fixed Notebook URL, defaultbranch project clone and local frozen dependencies identify different versions; editing local files does not update remote artifacts.","Colab/commit/publication boundaries","Original Git clone and uv frozen semantics plus personally located bootstrap/export. No Colab実機 or remote update claim.",[ev("git-clone","branch/depth clone defaults","No commit pin in default clone"),ev("uv-sync","--frozen","Use lock without updating")],[]),
        claim("build","software","Common Markdown body supplies notebooks/site, exporter invokes true Zensical clean strict build and command interfaces support stated arguments.","build commands","AST/help/source contract only; did not rebuild full frozen site, execute every notebook or deploy externally.",[ev("bootstrap","126–153","body-to-cell conversion"),ev("export","288/344–367","revision URL/strict build-info"),ev("dependencies","dependency-groups","actual tools"),ev("reading-time","288–305","validate executed interface")],["cli","maintenance"],"4 AST parse/4 helps/BOOTSTRAP AST successful;strict warnings abort flag observed"),
        claim("output-guard","software","Exporter checks matching cell kind/source, execution_count and error outputs before changing generated site files.","--executed paragraph","Four own synthetic current-code cases only, not proof that all published287 notebooks are currently executed.",[ev("export","94–108/248–255","guards before cleanup")],["guard"],"matched accepts;changed_source/unexecuted/error each rejects"),
        claim("pages","software","Current workflow builds on main/manual triggers, uses separate CPU kernels/workers1, exports GITHUB_SHA, uploads ignored outputs then deploys with needs build and Pages/OIDC permissions.","GitHub Pages workflow","Source YAML/ignore proof and official deployment contract; no new Action run, credentials or external deployment.",[ev("workflow","build/deploy jobs","actual sequence"),ev("pages-original","text109–129","needs/pages write/idtoken/environment")],["workflow","pages-doc"],"main/manual,workers1,GITHUB_SHA;deploy_needs build;pages/id-token write;outputs ignored"),
        claim("fixed-exports","numeric","Four fixed v2 exports comprise16 files at979cdfacc588ad0536f1c64fff96f264571cf054; root card file21510B/SHA9d3aef… is distinct from weight bytes.","§4公开材料","Actual original descriptors/public inference manifests plus card file SHA and headings; no license approval flag or author's performance summary used. Website edit/local card edit do not constitute remote update.",[ev("card-descriptor","release.file/parent_commit","Local card bytes and export parent identity"),ev("card-publish","682–725","Only README operation")],["exports","card"],"4×4=16;parent979cdf…;card21510B/hashmatches;README-only operation"),
    ]
    save(dict(page_id="publishing",verdict="pass",sources=sources,artifacts=arts,claims=cs,
        checks=checks([
            ("factual_accuracy","pass","Version distinctions and actual build/Pages contracts align; local edit, site build and weight publication remain distinct.",["version-boundary","build","pages"]),
            ("numeric_verification","pass","Four×four16 exports and21510B card hash checked against original descriptor/file.",["fixed-exports"]),
            ("figure_consistency","not_applicable","No SVG/table. Six necessary selected text screenshots actually viewed; complete long commands not fully visible.",[]),
            ("source_verification","pass","Actual exporter/checker/workflow and official Pages deployment requirements located; source read only.",["output-guard","pages"]),
            ("limitations","pass","No all287 run, Colab/GitHub Actions/remote upload; early screenshot exposed futureunit8 and is explicitly retained as a limitation.",["version-boundary","build","pages"]),
        ]),issues=[],page_receipt="docs/technical-reviews/artifacts/p7_technical_f/page-publishing-view.json",
        page_view_details="Six selected actual screenshots; desktop3 viewed before unit8 unlock, mobile3 after real unit8 read; no complete-page viewing claim.",
        provenance_notes=["Unit7 mistaken page-end assumption led to early capture. At about05:22 UTC desktop third image exposed unread§4 v2 revision/16 files/local card. Root notified; unit8 then normally unlocked and independently checked at05:23; no deletion or all-blind claim.","Broad script rg exposed update_course_progress.py historical review-status explanation; excluded as evidence.","Full repository-card descriptor incidentally printed redistribution_approved; not used. Card headings/hash inspected, no historical performance paragraph used.","First exporter import missed scripts path and failed ModuleNotFoundError; preserved failed artifact and real corrected-v2 fixture.","Unit3 overlisted direct notebook dependencies, corrected inunit4 to actual pyproject declarations. Card URL regex count is not used for exact-link or quality judgment."]))
