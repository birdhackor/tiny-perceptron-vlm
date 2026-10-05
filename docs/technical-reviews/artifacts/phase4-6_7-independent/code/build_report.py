"""Persist this review's own evidence inventory and grouped claim assessment."""
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
PREFIX = BASE.relative_to(ROOT).as_posix()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

spec = importlib.util.spec_from_file_location("facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
for section in ("6.3", "6.5", "6.6"):
    body, _, _ = facts.original_section(ROOT / "course/chapters/06.md", section)
    (BASE / f"inputs/prerequisite-{section}.md").write_bytes(body)
training = (ROOT / "course/training.md").read_bytes().splitlines(keepends=True)
(BASE / "inputs/training-lines-176-205.md").write_bytes(b"".join(training[175:205]))
run = json.loads((BASE / "execution/results.json").read_text())
original_environment = json.loads((BASE / "original/environment.json").read_text())
extract = json.loads((BASE / "original/extraction.json").read_text())
source_provenance = json.loads((BASE / "sources/provenance.json").read_text())
command = "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-6_7-independent/code/bounded_checks.py > docs/technical-reviews/artifacts/phase4-6_7-independent/execution/stdout.txt 2> docs/technical-reviews/artifacts/phase4-6_7-independent/execution/stderr.txt"
write(BASE / "execution/receipt.json", {"command": command, "cwd": str(ROOT), "exit_code": 0,
    "environment": run["environment"], "result": "All bounded assertions passed; no model weights read/evaluated.",
    "stdout_sha256": sha(BASE / "execution/stdout.txt"), "stderr_sha256": sha(BASE / "execution/stderr.txt"),
    "code_sha256": sha(BASE / "code/bounded_checks.py"),
    "constraints": "Only tiny self-authored tokenizer fitting; no model training, dataset preparation or download, GPU, paid compute."})
notes = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_6_7",
    "independence": "Fresh independent reviewer of this section only; no old reader/technical report body or verdict read; no subagents, content/figure edits or commits.",
    "scope_read": ["course/chapters/06.md#6.7 all original bytes", "6.3 all for byte-level BPE context", "6.5 and 6.6 prose/code for byte units and special ID context", "course/training.md lines 96-112 and 176-205; rg locations only for other lines", "specified method/checker/helper/protocol in full", "current and result-revision data.py, tokenization.py and text.py relevant contracts", "original tokenizer.json metadata, artifacts, code_sha256, result roundtrip and split records", "official source ranges recorded per source/claim"],
    "chapter_intro": {"status": "not_applicable", "reason": "6.7 is not chapter 6's first section; no intro review or invented intro summary."},
    "figure_inspection": {"status": "not_applicable", "references": [], "reason": "Original byte extraction lists zero SVG/image references in 6.7, and direct reading confirms no figure. No diagram, axes or figure numbers exist to render/view. Numeric boundary example is explicitly executable text."},
    "unicode_scope": "貓 (U+8C93) and 🙂 (U+1F642) are each one Unicode scalar; 3+4 octets does not generalize to one visual character per scalar. UAX29 section 3 explains G+grave and emoji sequences; bounded G̀/👩‍💻 have 2/3 code points. This section does not assert equality of tokens, scalars, grapheme clusters or glyphs.",
    "generation_scope": "May be completed is conditional: preserving original content bytes and a matching tokenizer's full ID order restores encoded valid input. Complete generated IDs can form malformed UTF-8, end on an incomplete prefix or contain structural IDs. Tests show strict/replace final flush; FF/C080/EDA080 malformed strings; incomplete EOS; and visible invalid control ID report. valid_answer_tokens in this helper concerns control IDs, not UTF-8 validity. No claim that every generated sequence is valid or losslessly repairable.",
    "bpe_scope": "By frequency describes training's adjacent-pair counts/greedy selection, not new frequency estimation on each inference input. Byte-level BPE is established by prerequisite 6.3 and this course's actual pretokenizer/decoder; no claim that every BPE implementation splits within a code point or crosses word boundaries.",
    "versions": "Executed Python 3.13.5, tokenizers 0.23.2, torch 2.14.1+cpu. CPython consulted tag v3.13.5 matches runtime; HF consulted immutable tag v0.22.1 differs from installed 0.23.2, explicitly recorded. Actual installed API behavior independently executed against historical JSON; official source supports conceptual algorithms, not a byte-for-byte binary identity assertion.",
    "historical_evidence": "Original result revision a253d1262bf5f361f9ac4e19232ae752f0ecc7a3, HF artifact revision 704589cb8c32f5ef818d7a71eb205410962dba27, result_revision 117d5be6791403e75a491fb01cbfe8e8475805fa. Original tokenizer/split hashes and 3 historical code hashes checked; all actual inputs copied permanently under docs. 3 recorded roundtrips/ID lists matched and 153+19+20 input roundtrips executed. No checkpoints read or predictions rerun. Historical generation text with � was not used to infer its cause.",
    "long_recipe": "Read T.4 tokenizer command and actual run_tokenizer/_utf8_prefix/_evaluate_tokenizer contracts only: recipe loads fixed assets, trains tokenizer on train split then two LMs for 400 updates; not executed. Bounded substitute uses existing hash-verified tokenizer JSON plus self-authored 12 copies of 貓狗 solely to exhibit multi-code-point tokenizer token.",
    "numeric_scope": "RFC3629 bit placement independently yields U+8C93 = E8 B2 93 and U+1F642 = F0 9F 99 82, converted to decimal values in original. Exactly 7 bytes and prefixes 1/3/4/7; integer equality, no rounded metric/axis or experimental denominator in 6.7. Auxiliary fixture checks specify counts separately."}
write(BASE / "inspection-notes.json", notes)

artifacts = []
ids_by_path = {}
for p in sorted(BASE.rglob("*")):
    if not p.is_file() or p.name.startswith("checker-"):
        continue
    rel = p.relative_to(BASE).as_posix()
    identifier = "a-" + rel.replace("/", "-").replace(".", "-")
    ids_by_path[rel] = identifier
    kind = "code" if p.suffix in (".py", ".rs") else "source_snapshot"
    description = "Permanent raw input/source snapshot and reproducibility metadata: " + rel
    if rel in ("execution/results.json", "execution/stdout.txt", "execution/receipt.json", "original/stdout.txt", "original/execution.json"):
        kind = "execution"
        description = "Actual CPU execution evidence: " + rel
    artifact = {"id": identifier, "kind": kind, "path": p.relative_to(ROOT).as_posix(),
                "sha256": sha(p), "description": description}
    if kind == "execution":
        original = rel.startswith("original/")
        artifact.update(command=(".venv/bin/python docs/review-tools/section_facts.py course/chapters/06.md#6.7 --output /tmp/phase4-6_7-facts --execute --timeout 45" if original else command),
                        result=("Exit 0; one unmodified Python fence executed with assertion passing, no SVG references." if original else "Exit 0; all bounded assertions passed including exact original/exercise outputs, UTF-8 final/error handling and hashed historical tokenizer fixtures."),
                        environment={k: str(v) for k,v in (original_environment if original else run["environment"]).items()})
    artifacts.append(artifact)

def aid(path):
    return ids_by_path[path]

sources = []
source_details = {
    "python-standardtypes.rst": ("python-types", "official_docs", "CPython 3.13.5 standard type documentation", "v3.13.5", "CPython upstream's versioned language/library documentation", "Read str.encode lines 1690-1711 and bytes.decode lines 2998-3015; UTF-8 and strict defaults."),
    "python-codecs.rst": ("python-codecs", "official_docs", "CPython 3.13.5 codecs documentation", "v3.13.5", "CPython upstream's versioned codec contract", "Read error handlers lines 325-344, incremental decoding lines 570-588 and 641-710, specifically final=True, buffered input getstate and U+FFFD replacement."),
    "python-utf8.py": ("python-utf8", "official_source", "CPython UTF-8 codec implementation", "v3.13.5", "Original CPython UTF-8 codec implementation", "Read entire 42-line file; decode final=True and BufferedIncrementalDecoder with codecs.utf_8_decode."),
    "tokenizers-byte-level.rs": ("hf-byte", "official_source", "Hugging Face tokenizers ByteLevel implementation", "v0.22.1 (runtime is 0.23.2)", "Original upstream implementation of the course's ByteLevel pretokenizer/decoder", "Read bytes_char lines 13-38, UTF-8 byte transformation lines 117-149 and decode_chain and comment lines 150-172; flatten token bytes before from_utf8_lossy, specifically comments warning a single token may not form a String."),
    "tokenizers-bpe-trainer.rs": ("hf-bpe", "official_source", "Hugging Face tokenizers BPE trainer implementation", "v0.22.1 (runtime is 0.23.2)", "Original upstream implementation of BPE training used by the course", "Read Merge ordering lines 14-38, count_pairs lines 378-416 and merge loop lines 450-515; weighted adjacent pair frequencies, max-count heap and concatenated pieces."),
    "unicode-uax29.html": ("unicode", "official_docs", "Unicode Standard Annex #29 Unicode Text Segmentation", "Unicode 17.0.0, revision 47", "Unicode Consortium's normative annex for grapheme/word segmentation", "Read version table, section 3 HTML lines 486-526 defining code point vs user-perceived character/grapheme cluster; lines 1358-1360 about emoji sequence clusters. No rendering/glyph count inferred from Python len."),
    "rfc3629.txt": ("rfc3629", "official_docs", "RFC 3629 UTF-8, a transformation format of ISO 10646", "November 2003, RFC3629", "IETF standards-track UTF-8 definition, RFC Editor original", "Read sections 3 and 4, text lines 177-282; 1-4 octet table, bit placement, invalid surrogate and overlong restrictions.")}
for record in source_provenance:
    name = Path(record["file"]).name
    identifier, kind, title, version, authority, note = source_details[name]
    sources.append({"id": identifier, "kind": kind, "title": title, "url": record["url"],
        "version": version, "authority_reason": authority, "accessed_on": record["accessed_at"][:10],
        "verified": True, "checked_original": True, "inspection_note": note,
        "snapshot_artifact_id": aid(record["file"])})
for identifier, path, version, note in (
    ("repo-byte", "inputs/historical/code/tiny_perceptron/data.py", "a253d1262bf5f361f9ac4e19232ae752f0ecc7a3; SHA matches result code_sha256 and current file", "Read ByteTokenizer lines 13-30: each UTF-8 octet +8; plain decode filters special IDs then decodes joined bytes with replace."),
    ("repo-bpe", "inputs/historical/code/tiny_perceptron/tokenization.py", "a253d1262bf5f361f9ac4e19232ae752f0ecc7a3; SHA matches result code_sha256 and current file", "Read ByteLevelBPE lines 12-55, load_tokenizer and generation_report lines 116-142; matching ByteLevel rules/256 alphabet, remove added tokens, complete content token decode, control-ID reporting."),
    ("repo-experiment", "inputs/historical/code/scripts/course_experiments/text.py", "a253d1262bf5f361f9ac4e19232ae752f0ecc7a3; SHA matches tokenizer.json code_sha256", "Read _BPE lines 348-371, _utf8_prefix 374-375, _evaluate_tokenizer 379-406, run_tokenizer 409-467: prefix bytes incomplete tail dropped during historical preparation, train-only BPE, preserved 3 roundtrip records, whole generated slice decoded once. Training recipe not executed."),
    ("original-result", "inputs/current/docs/course-experiments/results/tokenizer.json", "experiment revision a253d1262bf5f361f9ac4e19232ae752f0ecc7a3; HF revision 704589cb8c32f5ef818d7a71eb205410962dba27", "Read result metadata, results.data, results.roundtrip, artifact and code_sha256 arrays; independently matched fixture hashes and 3 original ID lists without using old review reports or evaluating weights.")):
    sources.append({"id": identifier, "kind": "repository_code", "title": identifier,
        "path": PREFIX + "/" + path, "sha256": sha(BASE / path), "version": version,
        "verified": True, "inspection_note": note})
sources.extend([
    {"id":"original-run","kind":"execution","title":"Unmodified 6.7 fence execution","verified":True,"artifact_id":aid("original/execution.json")},
    {"id":"bounded-run","kind":"execution","title":"Independent UTF-8/tokenizer boundary checks","verified":True,"artifact_id":aid("execution/results.json")},
    {"id":"utf8-derivation","kind":"derivation","title":"Independent RFC3629 bit placement","verified":True,"details":"bounded_checks.py utf8_from_scalar fills RFC3629 3/4-octet payload groups for ord(貓)=0x8C93 and ord(🙂)=0x1F642. Integer results [232,178,147] and [240,159,153,130] match .encode exactly; total 3+4=7 bytes."}])

def evidence(s, locator, support):
    return {"source_id": s, "locator": locator, "supports": support}

def verification(expected, observed, details, **kwargs):
    return {"method":"executed", "expected":expected, "observed":observed, "details":details, **kwargs}

claims = [
    {"id":"c-numeric", "kind":"numeric", "status":"verified", "statement":"貓🙂 的 UTF-8 bytes 正是 232,178,147,240,159,153,130，前三 bytes 是貓、後四 bytes 是🙂；Python str.encode() 預設 UTF-8，完整 decode/assert 還原原文。", "location":"6.7 開場、第二段、fence 與程式後第一/二段", "scope":"Two specified scalar values and one seven-byte UTF-8 sample; numeric units are octets and decimal integer values, not tokens or all visual characters.", "evidence":[evidence("python-types","str.encode lines 1690-1711; bytes.decode lines 2998-3015","UTF-8 default encoder/decoder contracts"),evidence("rfc3629","section 3 table/bit placement, lines 177-259","3-byte U+8C93 and 4-byte U+1F642 encoding formula"),evidence("utf8-derivation","utf8_from_scalar in bounded_checks.py; results.numeric","Independent bit-mask computation matches original decimal octets"),evidence("original-run","stdout.txt and execution.json","Exact original output and passing assert")], "artifact_ids":[aid("original/execution.json"),aid("execution/results.json")], "verification":verification("7 bytes, exactly the seven printed integer values, full decode 貓🙂 and passing assertion", "All expected integer values and decoded strings matched", "Both unmodified original fence and separate RFC3629 payload calculation executed; 3+4=7; no statistical denominator or rounded metric.",tolerance="Exact integer/string equality; no tolerance")},
    {"id":"c-bpe", "kind":"concept", "status":"verified", "statement":"本課 byte-level BPE 在訓練時依相鄰片段頻率合併；token 邊界可能切在完整字的中間，也可能一個 token 含多字。", "location":"6.7 第二段『BPE按頻率合併』；前置 6.3 ByteLevel 契約", "scope":"Training pair-frequency selection for this byte-level BPE, no universal claim about every BPE variant or dynamic frequencies at inference. A Unicode scalar differs from a user-perceived character/grapheme; sample 貓/🙂 each one scalar, no equality of token/code point/glyph asserted.", "evidence":[evidence("hf-bpe","Merge ordering lines 14-38; count_pairs 378-416; queue/merge loop 450-515","Weighted adjacent-pair counts choose highest frequency and concatenate token pieces"),evidence("hf-byte","byte transformation lines 117-149; decoder comments/implementation 150-172","Underlying units are bytes and a single decoded token may not form a Unicode string"),evidence("unicode","section 3, HTML lines 486-526; emoji note lines 1358-1360","Code points need not equal user-perceived characters; no mistaken visual-character unit inferred"),evidence("bounded-run","historical_bpe and tiny_bpe_multiple_scalars result keys","Historical 貓🙂 is seven single-byte tokens; self-authored 貓狗 learns one token ID268 covering two code points")], "artifact_ids":[aid("execution/results.json"),aid("execution/toy-tokenizer.json")]},
    {"id":"c-decode", "kind":"software", "status":"verified", "statement":"raw[:1] 留232，replace 顯示�而 strict 報UnicodeDecodeError；raw[:3] 顯示貓、raw[:4] 顯示貓�；完整原bytes都仍decode為貓🙂。", "location":"6.7 fence、程式後第一段與練習", "scope":"Stateless UTF-8 decoding of these exact byte prefixes under replace/strict. The expected exception is UnicodeDecodeError, a UnicodeError subclass; no exception promised for all arbitrary inputs.", "evidence":[evidence("python-types","bytes.decode lines 2998-3015","Default strict handling raises UnicodeError subclass"),evidence("python-codecs","error handler table lines 325-344","replace on decoding uses U+FFFD"),evidence("python-utf8","decode lines 15-16","Stateless decode treats input as final"),evidence("bounded-run","prefixes result key; stdout FENCE prefix=1,3,4","Original code and exact exercise slice changes executed with full final assert preserved")], "artifact_ids":[aid("original/stdout.txt"),aid("execution/results.json"),aid("execution/stdout.txt")], "verification":verification("prefix1 �/strict error; prefix3 貓; prefix4 貓�/strict error; full7 貓🙂", "Exactly matched, exception reason unexpected end of data at offsets 0 and 3 respectively", "Executed original fence and modifications changing only raw[:1] to raw[:3]/raw[:4] in second print; prefix outputs and final assertions checked.")},
    {"id":"c-loss", "kind":"concept", "status":"verified", "statement":"臨時看到替代字符不足以證明原始 bytes 損壞；若以解出的�覆寫原材料再存回，原來的 byte 資訊已失去，之後只拼接該替代字串不能恢復原文。", "location":"6.7 第三行解完整七byte之後整段", "scope":"Decoding an immutable byte slice leaves original bytes intact; replacement string carries U+FFFD rather than original incomplete byte value. Loss refers to replacing/discarding the original material, not claiming raw bytes already retained elsewhere disappear.", "evidence":[evidence("python-codecs","error handlers lines 325-344","U+FFFD replaces malformed/incomplete source sequence rather than retaining its original octets"),evidence("python-types","str.encode lines 1690-1711; bytes.decode 2998-3015","Encoding the replacement string encodes its character value"),evidence("bounded-run","replacement_loss result key","Original remains exact; persisted � becomes EF BF BD and prefix replacement plus remaining bytes decodes ���🙂, not 貓🙂")], "artifact_ids":[aid("execution/results.json")]},
    {"id":"c-stream", "kind":"software", "status":"verified", "statement":"byte token 生成可能先給不完整碼點而後續 token 補齐；介面可延後解碼或緩存尾段，應保留原bytes／完整 ID 順序，不能普遍把每token独立decode出的替代字串直接拼接。", "location":"6.7 『生成可能先給半字』整段", "scope":"Conditional completion of valid content byte prefixes with a matching decoder. Buffered UTF-8 decoding solves incomplete tails while the stream continues; final truncation, complete malformed UTF-8 and structural IDs need error/termination/role handling. Retaining IDs permits inspection; it does not guarantee arbitrary model outputs can be repaired, and valid_answer_tokens only audits structural IDs.", "evidence":[evidence("hf-byte","decode_chain and comment lines 150-172","Collect all token bytes before UTF-8 decoding expressly to avoid single-token byte fragments"),evidence("python-codecs","IncrementalDecoder.decode/getstate lines 667-704","Preserve state/buffer across calls; final=True flushes and handles incomplete tail errors"),evidence("rfc3629","section 3 invalid sequences and section 4 syntax","Malformed complete strings cannot be made valid by postponing decoding alone"),evidence("repo-byte","ByteTokenizer.decode lines 24-26","Assembles content bytes, uses replace, filters special IDs"),evidence("repo-bpe","ByteLevelBPE.decode lines 52-55; generation_report lines 116-142","Tokenizer-specific complete content decode; unexpected structural IDs are reported, not treated as ordinary byte content"),evidence("bounded-run","incremental, final_truncation, invalid_complete_sequences, byte_tokens, historical_bpe","Strict stream emits 貓 at third and 🙂 at seventh byte; full content decode vs seven U+FFFD single decodes; final and structural counterexamples executed")], "artifact_ids":[aid("execution/results.json")], "verification":verification("Complete stream concatenates to 貓🙂; independent byte/BPE decodes do not; final incomplete strict raises, replace yields�; FF/C080/EDA080 stay invalid; control ID reported", "Every expected output/error/structural report matched; EOS after E8 still yields� rather than guaranteeing repair", "No generation from model or weight evaluation; synthetic ID sequences probe helper contracts and distinguish valid content preservation from malformed or terminated sequences.")},
    {"id":"c-historical", "kind":"empirical", "status":"verified", "statement":"本節所沿用的原 byte/BPE 工具契約可由指定版本 tokenizer、split 與原紀錄核對：3 個記錄 roundtrip 的完整 ID 列／文字均吻合，並可完整還原153/19/20份既有輸入。", "location":"6.7 補充『實作約定與原始紀錄』；原實驗 results.roundtrip/data supporting evidence", "scope":"Audit of original tokenizer fixtures and recorded strings only; this section prints no LM quality score. All source hashes checked against result revision; not retraining or replaying historical predictions. 192 existing strings verify this tokenizer configuration on those strings only, not universal language ability.", "evidence":[evidence("original-result","revision/code_sha256/artifacts/results.data/results.roundtrip","Pins the actual original code/input versions and the three expected ID lists/restored strings"),evidence("repo-experiment","_BPE 348-371; _utf8_prefix374-375; run_tokenizer409-467","Historical no-normalizer ByteLevel setup and explicit three-string original roundtrip checks; recipe behavior inspected but not run"),evidence("bounded-run","historical_bpe/historical_splits; input-provenance checks","Original JSON/tokenizer/split hashes match, three recorded ID lists exactly reproduced, all original split texts roundtrip")], "artifact_ids":[aid("execution/results.json"),aid("input-provenance.json")], "verification":verification("Exact hashes and IDs, 3/3 recorded roundtrips;153/19/20 existing split text roundtrips", "All exact matches; split raw bytes31970/3894/4193; 3+192 tested strings", "Only encoded/decoded existing raw strings using hash-verified tokenizer. Did not read models.pt or re-evaluate training weights. Original roundtrip IDs remain stable in installed tokenizers0.23.2.", denominators={"recorded_roundtrip_cases":3,"train_records":153,"validation_records":19,"test_records":20,"existing_input_records_total":192,"self_authored_tokenizer_examples":12,"model_training_or_evaluation_steps":0})}
]
report = {"schema_version":1,"review_stage":"technical","lesson_id":"6.7",
    "source":"course/chapters/06.md#6.7","source_sha256":extract["source_sha256"],
    "figure_sha256":{},"reviewer_task":"/root/phase4_factual_coordinator/factual_6_7", "reviewer_context":"fresh",
    "author_tasks":[],"verdict":"pass","reviewed_on":"2026-10-05", "artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","claim_ids":[c["id"] for c in claims],"details":"Grouped every substantive concept/API/numeric assertion; official original sources personally inspected, original code + meaningful prefix/token/stream variations executed. No substantive contradiction or unresolved unknown found."},
        "numeric_verification":{"status":"pass","claim_ids":["c-numeric","c-decode","c-historical"],"details":"RFC3629 independent bit derivation and exact seven octets; 1/3/4/7-byte boundaries match original and exercise. No plotted axes/rounded performance metric in section; fixture denominators stated separately."},
        "figure_consistency":{"status":"not_applicable","claim_ids":[],"details":"6.7 has zero image/SVG references in original-byte helper extraction and direct reading; no figure/axes to render or view, figure_sha256={}. Text+executed code provides the complete needed boundary example."},
        "source_verification":{"status":"pass","claim_ids":[c["id"] for c in claims],"details":"Fetched HTTPS original CPython v3.13.5, HF upstream v0.22.1, Unicode UAX29 revision47 and RFC3629; read exact listed locations rather than search snippets. Saved original response bytes/URL/access date/hashes. Explicit runtime/source-version distinction; original experiment code/input hashes match."},
        "limitations":{"status":"pass","claim_ids":["c-bpe","c-loss","c-stream","c-historical"],"details":"Byte-level BPE established by6.3, frequency refers to training; scalar≠grapheme/glyph; only valid retained content sequence roundtrips. Final malformed/truncated UTF-8/control IDs remain separate conditions. No claim that full generated IDs are always valid; no model training/GPU/download/weight evaluation; long recipe inspected and bounded substitute used."}},
    "review_scope":notes,"revision_history":[]}
write(ROOT / "docs/technical-reviews/6.7.json", report)
print(json.dumps({"source_sha256":report["source_sha256"],"report_sha256":sha(ROOT/'docs/technical-reviews/6.7.json'),"claims":len(claims),"artifacts":len(artifacts),"sources":len(sources),"verdict":report["verdict"]},ensure_ascii=False))
