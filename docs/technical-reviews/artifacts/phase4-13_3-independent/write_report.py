"""Write this reviewer's complete new report without reading the prior canonical report."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
REL = OUT.relative_to(ROOT).as_posix()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def path(n): return REL + "/" + n
checks = json.loads((OUT / "checks.json").read_text())
environment = checks["environment"]
assert environment["device"] == "cpu" and environment["cuda_build"] == "None"
extract = json.loads((OUT / "original-fence/extraction.json").read_text())
assert sha(OUT / "original-fence/section.md") == extract["source_sha256"]
assert extract["svg_references"] == []
assert json.loads((OUT / "original-fence/execution.json").read_text())["exit_code"] == 0
assert (OUT / "checks.stderr.txt").read_bytes() == b""
assert (OUT / "render.stderr.txt").read_bytes() == b""
artifacts = []
names = [
 ("frozen-chapter", "inputs/frozen-chapter-13.md", "source_snapshot", "Whole chapter frozen input at this inspection; not a claim about later current chapter versions."),
 ("section", "original-fence/section.md", "source_snapshot", "Exact original UTF-8 13.3 section bytes, no newline normalization."),
 ("fence-code", "original-fence/fence-1.py", "code", "Exact original Python fence."),
 ("bootstrap", "original-fence/bootstrap.py", "code", "Exact CPU helper bootstrap used for original fence."),
 ("extraction", "original-fence/extraction.json", "source_snapshot", "Extraction/source/fence/figure hashes and source line metadata."),
 ("original-env", "original-fence/environment.json", "source_snapshot", "Actual original-fence CPU/offline environment and imported repository module hashes."),
 ("original-stdout", "original-fence/stdout.txt", "source_snapshot", "Actual original-fence stdout."),
 ("original-stderr", "original-fence/stderr.txt", "source_snapshot", "Actual original-fence stderr; empty."),
 ("checks-code", "check_claims.py", "code", "Independent numeric, masking, EOS and historical sampler verification code."),
 ("checks-stdout", "checks.stdout.txt", "source_snapshot", "Actual independent CPU verification stdout."),
 ("checks-stderr", "checks.stderr.txt", "source_snapshot", "Actual independent CPU verification stderr; empty."),
 ("render-code", "render_section.py", "code", "Actual standalone section rendering code; no external requests."),
 ("render-stderr", "render.stderr.txt", "source_snapshot", "Actual Chromium rendering stderr; empty."),
 ("render-html", "section-render.html", "source_snapshot", "Standalone HTML rendered from frozen section."),
 ("desktop", "section-desktop.png", "figure_render", "Actual 1280×800 Chromium render, supplement opened, personally viewed."),
 ("mobile", "section-mobile.png", "figure_render", "Actual 390×844 Chromium render, supplement opened, personally viewed."),
 ("raw-measurement", "inputs/dpo-measurement.json", "source_snapshot", "Original complete unmodified DPO measurement JSON, with uninspected annotations retained."),
 ("alignment-code", "inputs/alignment.py", "code", "Frozen original sequence-sum and DPO implementations."),
 ("data-code", "inputs/data.py", "code", "Frozen original byte tokenization, target alignment, EOS and padding methods."),
 ("text-code", "inputs/text.py", "code", "Frozen original arithmetic records and split serialization methods."),
 ("common-code", "inputs/common.py", "code", "Frozen original family split and optimizer-step contract."),
 ("historical-behavior", "inputs/behavior-measurement-revision.py", "code", "Exact original measurement revision implementation; matches raw result code SHA."),
 ("slp-pdf", "sources/slp3-chapter3-20260819.pdf", "source_snapshot", "Original author-hosted SLP draft, personally checked header, chain rule, log and length sections."),
 ("slp-text", "sources/slp3-chapter3-20260819.txt", "source_snapshot", "Text extracted from original SLP PDF, personally inspected named sections."),
 ("dpo-pdf", "sources/dpo-v3.pdf", "source_snapshot", "Original arXiv v3 paper; independently downloaded immutable official URL and matched full SHA."),
 ("dpo-text", "sources/dpo-v3.txt", "source_snapshot", "Text extracted from original DPO paper, personally inspected Eq.7 and Appendix B."),
 ("logsoftmax-html", "sources/logsoftmax-pytorch-2.9.html", "source_snapshot", "Original official PyTorch 2.9 log_softmax documentation HTML."),
 ("logsoftmax-text", "sources/logsoftmax-pytorch-2.9.txt", "source_snapshot", "Personally read official log_softmax API extraction."),
 ("gather-html", "sources/gather-pytorch-2.9.html", "source_snapshot", "Original official PyTorch 2.9 gather documentation HTML."),
 ("gather-text", "sources/gather-pytorch-2.9.txt", "source_snapshot", "Personally read official gather API extraction."),
 ("personal-inspection", "personal-inspection.md", "source_snapshot", "Personal source/version/locator/support, exact read scope, provenance and limitations."),
 ("commands", "commands.json", "source_snapshot", "Actual commands, shell/cwd, exit outcomes and permanent output paths."),
 ("review-rules", "inputs/factual-reviewer-instructions.md", "source_snapshot", "Actual instructions read for this independent review."),
 ("checker-code", "inputs/check_technical_reviews.py", "code", "Actual checker schema read; does not determine factual truth."),
 ("helper-code", "inputs/section_facts.py", "code", "Actual source-extraction/CPU execution helper read and used."),
 ("skill", "inputs/SKILL.md", "source_snapshot", "Actual clear-tutorial skill read."),
 ("protocol", "inputs/review-protocol.md", "source_snapshot", "Actual review protocol read."),
 ("report-code", "write_report.py", "code", "This reviewer's complete report-generation code; never reads the old canonical report.")]
for ident, name, kind, description in names:
 artifacts.append({"id":ident, "path":path(name), "sha256":sha(OUT/name), "kind":kind, "description":description})
for ident, name, command, result, env in [
 ("original-execution", "original-fence/execution.json", ".venv/bin/python docs/review-tools/section_facts.py 'course/chapters/13.md#13.3' --output outputs/reviewer-tools/phase4-13_3-personal --execute --timeout 45", "Exit 0; exact fence stdout: 整段log分數 -0.4791; 還原機率 0.6193.", environment),
 ("cpu-execution", "checks.json", "env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-13_3-independent/check_claims.py", "Exit 0; scalar, mask, batch, EOS, exact data hashes and 250×8 seeded sampling-only replay all assertions passed.", environment),
 ("render-execution", "render.stdout.txt", "timeout 45 .venv/bin/python docs/technical-reviews/artifacts/phase4-13_3-independent/render_section.py", "Exit 0; desktop and mobile screenshots rendered and personally viewed.", {"python":environment["python"], "browser":"Chromium 151.0.7922.173", "device":"CPU rendering"})]:
 artifacts.append({"id":ident,"path":path(name),"sha256":sha(OUT/name),"kind":"execution","description":result,"command":command,"result":result,"environment":env})

def official(ident, title, kind, url, version, reason, note, snapshots):
 return {"id":ident,"title":title,"kind":kind,"url":url,"version":version,"authority_reason":reason,"verified":True,"checked_original":True,"accessed_on":"2026-10-05","inspection_note":note,"snapshot_artifact_ids":snapshots}
def repo(ident,title,name,version,note):
 return {"id":ident,"title":title,"kind":"repository_code","path":path(name),"sha256":sha(OUT/name),"version":version,"verified":True,"inspection_note":note}
sources = [
 official("slp", "Jurafsky & Martin, Speech and Language Processing, chapter 3", "official_docs", "https://web.stanford.edu/~jurafsky/slp3/3.pdf", "Author-hosted draft August 19, 2026; date personally checked in PDF header", "Original textbook on its authors' Stanford site, giving probability chain rule and log-space derivation.", "Personally read §3.1 p.3 Eqs.3.3–3.4; p.4 end-symbol footnote; §3.1.3 p.6 Eq.3.13 and natural-log convention; §3.3 p.9 before Eq.3.14 length discussion. Supports chain/product/log/length, not human-quality results.", ["slp-pdf","slp-text"]),
 official("dpo", "Rafailov et al., Direct Preference Optimization", "paper", "https://arxiv.org/pdf/2305.18290v3", "arXiv:2305.18290v3, 2024-07-29", "Original paper by DPO's authors, immutable versioned arXiv PDF.", "Personally read title/version, §4 p.4 Eq.7, §4 p.5 offline DPO outline, Appendix B p.20 original loss. Official immutable download SHA exactly matched the personally read original PDF. Supports same-answer reference ratios and preferred/dispreferred comparison; does not validate toy model quality.", ["dpo-pdf","dpo-text"]),
 official("logsoftmax", "PyTorch functional.log_softmax API", "official_docs", "https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.log_softmax.html", "PyTorch 2.9 documentation; title personally checked", "PyTorch project's official versioned API documentation.", "Personally read API, stable logarithm-of-softmax description and dim parameter. Installed execution is 2.14.1+cpu; tested last-axis contract matches this cited older documentation.", ["logsoftmax-html","logsoftmax-text"]),
 official("gather", "PyTorch gather API", "official_docs", "https://docs.pytorch.org/docs/2.9/generated/torch.gather.html", "PyTorch 2.9 documentation; title personally checked", "PyTorch project's official versioned API documentation.", "Personally read 3-D dim==2 indexing formula, required input/index rank and returned index shape. Installed execution is 2.14.1+cpu; tested contract agrees.", ["gather-html","gather-text"]),
 repo("alignment", "Original sequence_log_probability and dpo_loss", "inputs/alignment.py", "Frozen current file; matches original measurement manifest SHA", "AST definitions first; personally read lines 29–42 for mask, last vocabulary axis, gather and position sum, plus relative DPO margin."),
 repo("data", "Original ByteTokenizer, render_chat and pad_batch", "inputs/data.py", "Frozen current file; matches original measurement manifest SHA", "AST definitions first; personally read lines 10–28,46–86 for byte IDs, assistant content+EOS labels, once-shifted alignment and ignored padding."),
 repo("historical", "Original measurement-revision preference generation, training and comparison methods", "inputs/behavior-measurement-revision.py", "Git revision 8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d; SHA matches original /code_sha256 entry", "Personally AST-located and read original lines 537–605,609–652,697–717: own-answer examples; seeded choices k=8; token counter; 250-step fit; sum helper in train/evaluation. No run_dpo scope/result explanation read."),
 repo("text", "Original arithmetic_records and exact split serializer", "inputs/text.py", "Frozen current file; matches original measurement manifest SHA", "Read lines 34–44,47–62,596–611; arithmetic families, step scale and exact JSONL hashing."),
 repo("common", "Original family split and fit loop", "inputs/common.py", "Frozen current file; matches original measurement manifest SHA", "Read lines 50–74 and 205–239; seeded family split and one real optimizer step per requested step. Reconstructed only the data/sampling; did not invoke fit."),
 {"id":"calculation","title":"Independent scalar probability/log derivation","kind":"derivation","verified":True,"details":"p=exp(2)/(exp(2)+2), ln p=2-ln(exp(2)+2). Two scored targets give 2 ln p and exp(2 ln p)=p²; three give 3 ln p and (3 ln p)/3=ln p. Units are nats for sum, nats/scored token for mean; float32 execution tolerance ≤3e−7, four-decimal rounding ±0.00005."},
 {"id":"cpu","title":"Independent bounded CPU verification","kind":"execution","verified":True,"artifact_id":"cpu-execution"},
 {"id":"original","title":"Exact original fence execution","kind":"execution","verified":True,"artifact_id":"original-execution"}
]
def evidence(source, locator, supports):return {"source_id":source,"locator":locator,"supports":supports}
def claim(ident,kind,statement,location,scope,evidences,artids,verification=None):
 d={"id":ident,"kind":kind,"statement":statement,"location":location,"scope":scope,"status":"verified","evidence":evidences,"artifact_ids":artids}
 if verification:d["verification"]=verification
 return d
def verification(expected, observed, details, tolerance=None, denominators=None):
 d={"method":"executed","expected":expected,"observed":observed,"details":details}
 if tolerance:d["tolerance"]=tolerance
 if denominators:d["denominators"]=denominators
 return d
claims = [
 claim("chain", "concept", "Sequential conditional probabilities multiply to the full answer probability; the second conditional already includes the first target, so no independence assumption is needed.", "13.3 opening paragraph", "Fixed prompt and specified target prefixes; exact probability chain rule, not a Markov approximation.", [evidence("slp","§3.1 p.3 Eqs.3.3–3.4","General conditional chain rule before §3.1.1's later Markov approximation.")], ["slp-pdf","personal-inspection"]),
 claim("product", "numeric", "Two conditional probabilities 0.8 and 0.8 give 0.64.", "13.3 opening paragraph", "Exact decimal hand example; float64 representation is approximate.", [evidence("calculation","0.8×0.8; checks.json numeric.conditional_0_8_product","Recomputed the product independently.")], ["cpu-execution","checks-code"], verification("0.64","0.6400000000000001","Plain scalar multiplication; two conditional terms.","Float64 absolute error <1e−15.")),
 claim("log", "concept", "Natural log converts the answer probability product into a log sum, and exponentiation restores the probability; smaller probabilities compound for longer sequences.", "13.3 second paragraph", "Natural logarithms; sum is in nats; no target-prefix sampling is necessary to evaluate a specified answer.", [evidence("slp","§3.1.3 p.6 Eq.3.13 and following natural-log convention","Log-space addition, underflow motivation, exp inversion and base.")], ["slp-pdf","slp-text"]),
 claim("helper", "software", "The original fence uses logits shaped (1,3,3), selects labels 0 and 2 at the last two positions, masks −100 at the first, applies vocabulary-axis log_softmax and gather, and sums answer positions. It neither generates nor updates a model.", "13.3 Python fence and following logits/labels paragraph", "Static supplied scores; batch×position×candidate axes. The helper returns one sum per batch item and consumes already-aligned labels.", [evidence("alignment","lines 29–35","Actual mask, dim=−1 vocabulary softmax/gather and position sum."),evidence("logsoftmax","API description and dim parameter","Logarithm-of-softmax along specified axis."),evidence("gather","3-D dim==2 formula","Last-axis candidate selection."),evidence("original","exact fence execution","Original code prints advertised score/probability.")], ["original-execution","cpu-execution","fence-code","alignment-code"], verification("Only two valid targets affect the score; ignored positions and batch rows behave independently.","Original fence exit0; ignored-logit perturbation unchanged; constant logit shift invariant; batch values [−0.4790895581,−0.7186344266]; all-ignored row raises ValueError.","Executed exact fence plus targeted variants; no model, generation, gradient or optimizer call.")),
 claim("numbers", "numeric", "Each target has probability ≈0.7870 and natural log ≈−0.2395; two-position sum ≈−0.4791, exp≈0.6193; adding the first label1 yields a third equal contribution, sum≈−0.7186 with mean≈−0.2395.", "13.3 numerical paragraph and first-label exercise", "Three-candidate logits [2,0,0] up to permutation, sums over 2 or 3 scored positions; mean denominator is scored-position count.", [evidence("calculation","p=exp(2)/(exp(2)+2); ln p; checks.json numeric","Independent exact scalar calculation and units."),evidence("cpu","checks.json numeric and batch_scores","Original and exercise float32 outputs compared to scalar values.")], ["cpu-execution","original-execution","checks-code"], verification("p=.7869860422; ln p=−.2395447662; sum2=−.4790895324; exp=.6193470306; sum3=−.7186342987; mean3=−.2395447662.","Float32 original −.4790895581 and .6193470359; exercise −.7186344266; all textbook four-decimal roundings match.","Vocabulary normalization denominator is exp(2)+2=9.3890560989; sums in nats, means in nats/scored token; count denominators 2 and 3.","Float32 absolute deviation <3e−7; textbook rounded values differ <0.00005.")),
 claim("length", "concept", "Sequence log sum and per-token mean are different quantities. More negative raw summed log probability for a longer answer cannot by itself establish a less preferred human answer.", "13.3 paragraph beginning 所以整段加總", "For conditional probabilities ≤1 each additional term is ≤0; different-text preference is not determined by raw length-dependent sequence likelihood.", [evidence("slp","§3.3 p.9 paragraph before Eq.3.14","Raw sequence probability depends on length, motivating per-token normalization."),evidence("dpo","§4 Eq.7 and offline preference-data outline","Preference labels and reference-relative candidate comparison are separate from raw likelihood."),evidence("calculation","sum3 versus mean3","Equal per-token likelihood already yields different sums.")], ["slp-pdf","dpo-pdf","cpu-execution"]),
 claim("reference", "concept", "DPO compares each answer's policy log probability against the fixed reference's probability for that same answer, then compares preferred and rejected answers.", "13.3 paragraph beginning 後面會先對每篇答案", "Equivalent rearrangement of policy chosen−rejected minus reference chosen−rejected; does not imply length bias or preference problems are eliminated.", [evidence("dpo","§4 p.4 Eq.7; Appendix B p.20 original loss","Same-completion log ratios and their preferred/dispreferred difference."),evidence("alignment","lines 38–42","Repository's corresponding relative margin.")], ["dpo-pdf","historical-behavior","alignment-code"]),
 claim("eos-and-sum", "software", "The chapter's implemented full answer scoring includes EOS, excludes prompt/role/padding targets, and uses the same sequence sum for training and formal candidate comparisons. Answer4 has two scored targets, answer10 has three; real question positions must remain ignored.", "13.3 final main paragraph, exercise and supplement", "Byte tokenizer and render_chat contract; complete assistant answers, not the literal synthetic candidate indices as tokenizer IDs.", [evidence("data","lines 10–28,54–86","EOS2 appended to assistant content, prompt/role/padding targets ignored; shift occurs once."),evidence("historical","lines 552–605,609–644","Own-answer examples and identical sum helper for policy/reference in training/evaluation."),evidence("cpu","checks.json render_chat_examples and mask_contract","Executed actual tokenization/padding and selected target counts.")], ["cpu-execution","data-code","alignment-code","historical-behavior"], verification("4 scored targets [60,2], count2; 10 scored targets [57,56,2], count3; padding and question targets excluded; training/evaluation helper sum unchanged.","Actual render_chat targets match; padded counts [2,3]; first ignored-position perturbation has no effect; exact historical methods call same sum helper.","Executed tokenization and masks; inspected original train/evaluate contracts. No model is trained or evaluated.")),
 claim("exposure", "empirical", "The two original addition runs each perform 250 updates and record 9,448 effective target tokens across both answer sides, including EOS. This is repeated sampling exposure rather than 9,448 distinct answers.", "13.3 supplement beginning 兩支各250次更新", "Saved original CUDA measurement plus CPU reconstruction of original data hashes and sampling denominator. No new GPU training or model-performance evaluation.", [evidence("historical","lines 565–605,697–717","Count250; seed42 replacement sample k8, both-side token counter and two beta runs."),evidence("common","lines 205–223","One optimizer.step per loss_fn call in original fit loop."),evidence("text","lines 47–62,596–611","Original arithmetic rows and exact JSONL split serialization."),evidence("cpu","checks.json measurement_provenance, dataset_reconstruction, sampling_only_replay","Read original steps/token fields and independently matched three data hashes plus 9,448-token count.")], ["raw-measurement","cpu-execution","checks-code","historical-behavior","commands"], verification("Each saved run:250 steps and9,448 both-side target tokens; data train49 pairs; seed42 choices8 per step.","Each replay:2,000 pair exposures,4,000 answer exposures,49 distinct pair indices,4,605 chosen+4,843 rejected=9,448 scored targets,including4,000 EOS. All split hashes exactly match original.","Accessed only listed raw/provenance pointers after key/type enumeration; measurement revision behavior SHA verified. Sampling-only replay verifies denominator, not trained model quality.",denominators={"runs":2,"steps_per_run":250,"pairs_per_step":8,"training_pairs_available":49,"pair_exposures_per_run":2000,"answer_exposures_both_sides_per_run":4000,"target_tokens_both_sides_per_run":9448,"eos_tokens_per_run":4000,"chosen_target_tokens_per_run":4605,"rejected_target_tokens_per_run":4843}))
]
ids=[c["id"] for c in claims]
report={
 "schema_version":1,"review_stage":"technical","lesson_id":"13.3","source":"course/chapters/13.md#13.3","source_sha256":extract["source_sha256"],"verdict":"pass",
 "reviewer_task":"/root/phase4_factual_coordinator/factual_13_3","reviewer_context":"fresh","author_tasks":[],"reviewed_on":"2026-10-05","figure_sha256":{},
 "frozen_input":{"path":path("inputs/frozen-chapter-13.md"),"sha256":sha(OUT/"inputs/frozen-chapter-13.md"),"meaning":"Whole chapter frozen input at this inspection; only section SHA identifies the canonical reviewed section."},
 "reading_scope":"All 13.3 plus chapter lines1–69 and109–245; named original calculation methods and original official sources as personally recorded. No prior review/result summary read.",
 "sources":sources,"artifacts":artifacts,"claims":claims,"issues":[],
 "checks":{
  "factual_accuracy":{"status":"pass","details":"All substantive chain-rule/log/mask/length/reference/EOS/exposure statements separately checked with explicit supporting scope.","claim_ids":ids},
  "numeric_verification":{"status":"pass","details":"Independent exp(2)/(exp(2)+2) derivation, actual original fence and three-target variant; precise axes/nats/token denominators and tolerances; historical 250×8 sampler count exactly matches9,448.","claim_ids":["product","numbers","exposure"]},
  "figure_consistency":{"status":"not_applicable","details":"No image or SVG reference in13.3; figure hash dictionary is empty. Nevertheless truly rendered and personally viewed frozen-section desktop/mobile PNGs; this standalone rendering is not production-theme acceptance.","claim_ids":[]},
  "source_verification":{"status":"pass","details":"Personally read original official SLP draft, immutable DPO v3 PDF (own official download SHA matched), PyTorch2.9 APIs; exact original measurement-revision code SHA matches saved manifest. Index used only for location.","claim_ids":ids},
  "limitations":{"status":"pass","details":"CPU arithmetic/tokenization/sampling-only verification supports this section's mechanism/counts, not model quality. Original measurement was CUDA; no weights loaded or GPU work performed. Installed torch2.14.1+cpu differs from cited official docs2.9, with exercised API contract checked. No figure exists; standalone section renders were personally viewed. No unresolved material claim.","claim_ids":ids}
 },
 "limitations":["No GPU training, existing-model reevaluation, data/weight download or capability acceptance performed.","Historical sampling reconstruction verifies token exposure denominator and data provenance; saved raw training steps are inspected, not independently rerun.","Official PyTorch documentation is version2.9; actual CPU execution is2.14.1+cpu.","Rendered frozen section uses a standalone stylesheet; production-site theme and reader usability acceptance are outside this technical review."]
}
target=ROOT/'docs/technical-reviews/13.3.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert json.loads(target.read_text())["reviewer_task"] == report["reviewer_task"]
assert json.loads(target.read_text())["source_sha256"] == extract["source_sha256"]
print(json.dumps({"canonical":str(target),"verdict":report["verdict"],"source_sha256":report["source_sha256"],"report_sha256":sha(target),"claims":len(claims),"artifacts":len(artifacts)},ensure_ascii=False))
