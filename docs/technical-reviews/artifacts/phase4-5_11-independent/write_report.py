"""Serialize this review's own claim-level judgments and permanent evidence."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
BASE=OUT.relative_to(ROOT).as_posix()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
env=json.loads((OUT/"probe-environment.json").read_text())
probe_command="CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 45s .venv/bin/python "+BASE+"/independent_probe.py > "+BASE+"/probe.stdout.json 2> "+BASE+"/probe.stderr.txt"
save(OUT/"command-receipts.json",{
    "shell":"bash","login":False,"cwd":str(ROOT),
    "runs":[
        {"command":".venv/bin/python "+BASE+"/prepare_evidence.py","exit_code":0,
         "receipt":"prepare-receipt.json","inner_execution_receipt":"original/execution.json"},
        {"command":".venv/bin/python "+BASE+"/acquire_sources.py > "+BASE+"/acquire.stdout.json 2> "+BASE+"/acquire.stderr.txt",
         "exit_code":0,"source_code_sha256":digest(OUT/"acquire_sources.py"),"receipt":"sources/acquisition-receipt.json"},
        {"command":probe_command,"actual_executed_path":BASE+"/independent_probe.py",
         "source_code_sha256":digest(OUT/"initial-probe.py"),"exit_code":1,
         "note":"The retained initial source bytes were executed at independent_probe.py before being copied. One-line Python suite assumption failed; not a lesson defect.",
         "stdout":"initial-probe.stdout.txt","stderr":"initial-probe.stderr.txt"},
        {"command":probe_command,"exit_code":0,"source_code_sha256":digest(OUT/"independent_probe.py"),
         "stdout":"probe.stdout.json","stderr":"probe.stderr.txt","environment":"probe-environment.json"}
    ]})
provenance=json.loads((OUT/"input-provenance.json").read_text())
for p in [OUT/"inputs/T.4.md",*sorted((OUT/"historical").rglob("*.py")),*sorted((OUT/"sources").glob("*.pdf")),*sorted((OUT/"sources").glob("*.html"))]:
    if any(i["snapshot"]==p.relative_to(ROOT).as_posix() for i in provenance["inputs"]): continue
    provenance["inputs"].append({"original_path":"course/training.md#T.4" if p.name=="T.4.md" else "See acquisition-receipt.json or data-audit-results.json for official URL/historical Git origin",
        "snapshot":p.relative_to(ROOT).as_posix(),"sha256":digest(p),"bytes":p.stat().st_size})
save(OUT/"input-provenance.json",provenance)

artifacts=[]
paths={}
for i,p in enumerate(sorted(OUT.rglob("*"))):
    if not p.is_file() or p.name in ("artifact-manifest.json","checker.stdout.txt","checker.stderr.txt","checker-receipt.json"): continue
    rel=p.relative_to(OUT).as_posix(); identifier="artifact-"+str(i); paths[rel]=identifier
    kind="code" if p.suffix==".py" else "source_snapshot"
    entry={"id":identifier,"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p),"kind":kind,
        "description":"Permanent original bytes / execution metadata / independent inspection: "+rel}
    if rel=="original/stdout.txt":
        kind="execution"; entry.update(kind=kind,command=json.loads((OUT/"original/execution.json").read_text())["command"],
            result="exit 0; normalized texts [貓 看狗,貓 看狗,狗 看貓], records=3, unique=2, hex chars=64; exact source fence; CPU/offline guard events=[]",environment=env)
    elif rel=="probe.stdout.json":
        entry.update(kind="execution",command=probe_command,result="exit 0; all assertions passed; original/changed-character/extra-whitespace variants; raw 512/365 counts, all six original split SHA-256, preserved raw-text target bytes; no training",environment=env)
    elif rel=="acquire.stdout.json":
        entry.update(kind="execution",command=".venv/bin/python "+BASE+"/acquire_sources.py",result="exit 0; all five primary sources HTTP 200; both pdftotext commands exit 0",environment={"python":env["python"],"device":"CPU","network":"read-only authoritative public document acquisition"})
    artifacts.append(entry)
save(OUT/"artifact-manifest.json",{"files":artifacts,"scope":"All permanent review evidence; no weight inputs or weight copies."})

def original_source(identifier,kind,title,url,version,reason,note):
    return {"id":identifier,"kind":kind,"title":title,"url":url,"version":version,"accessed_on":"2026-10-05",
        "authority_reason":reason,"checked_original":True,"inspection_note":note,"verified":True}
def repo(identifier,title,relative,version,note):
    path=OUT/relative
    return {"id":identifier,"kind":"repository_code","title":title,"path":path.relative_to(ROOT).as_posix(),
        "sha256":digest(path),"version":version,"inspection_note":note,"verified":True}
sources=[
    original_source("python-strings","official_docs","Python 3.13 str.split / str.join / str.encode",
        "https://docs.python.org/3.13/library/stdtypes.html","Python 3.13 docs snapshot updated 2026-10-01; execution 3.13.5",
        "Python Software Foundation defines the executed built-in string APIs.",
        "Personally read str.split, str.join, str.encode; text lines 3500–3568, 3116–3130, 2554–2565. Snapshot and URL receipt retained."),
    original_source("python-hashlib","official_docs","Python 3.13 hashlib.sha256 and hash.hexdigest",
        "https://docs.python.org/3.13/library/hashlib.html","Python 3.13 official documentation, accessed 2026-10-05",
        "Python Software Foundation defines hashlib's constructor and return types.",
        "Personally read Hash algorithms lines 286–303, digest_size 534–540, digest/hexdigest 586–602; byte input and double-length hexadecimal output."),
    original_source("nist-shs","official_docs","NIST FIPS PUB 180-4 Secure Hash Standard",
        "https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf","FIPS PUB 180-4, August 2015",
        "NIST publishes the standard defining SHA-256 and secure digest properties.",
        "Personally read Explanation item 3 lines 74–87, Section 1 Introduction and Figure 1 printed p.2 lines 227–261; one-way, difficult collision/preimage, 256-bit digest."),
    original_source("lee-dedup","paper","Lee et al. Deduplicating Training Data Makes Language Models Better",
        "https://aclanthology.org/2022.acl-long.577.pdf","ACL 2022 long paper 577, pp.8424–8445; publisher PDF",
        "The original peer-reviewed paper directly studies duplicate training content and evaluation overlap.",
        "Personally read abstract and Introduction pp.8424–8425, Section 4 p.8426; used only for overlap risk and exact-matching scope, not transferring model-performance results."),
    original_source("python-whitespace","official_docs","Python 3.13 lexical analysis of indentation and whitespace",
        "https://docs.python.org/3.13/reference/lexical_analysis.html","Python 3.13 language reference; accessed 2026-10-05",
        "The official language reference defines when whitespace affects Python parsing.",
        "Personally read Sections 2.1.8 and 2.1.9, text lines 437–441 and 518–525; indentation groups statements and literal whitespace is special."),
    repo("fingerprint-code","Original tiny_perceptron.data fingerprint", "inputs/tiny_perceptron/data.py",
        "Snapshot at HEAD 8905e5cf043eff2fed430439ec33030228c4cb47; fingerprint/data SHA matches historical report",
        "Read lines 102–104 and ByteTokenizer 14–27; whitespace compare -> UTF-8 bytes -> SHA-256 hexdigest."),
    repo("text-pipeline","Historical/current original real_text cleaning and entrypoint", "historical/scripts/course_experiments/text.py",
        "5581462ef01959636425eb7795ae3153142dbfeb; SHA matches saved report and current source",
        "Read _deduplicate_text 91–99, _save_splits 46–61 and run_real_text 317–345. Executes cleaning before grouping/splitting and preserves row text."),
    repo("split-code","Historical/current original split_records and text_examples", "historical/scripts/course_experiments/common.py",
        "5581462ef01959636425eb7795ae3153142dbfeb; SHA matches saved report and current source",
        "Read split_records 50–70 and text_examples 77–97. Exact family grouping before seed=42 shuffle; tokenization uses raw record text."),
    repo("original-result","Original real_text experiment JSON", "inputs/docs/course-experiments/results/real_text.json",
        "Original run revision 5581462ef01959636425eb7795ae3153142dbfeb; CUDA / torch 2.14.1+cu126; seed42",
        "Read results.runs source_records/deduplicated_records/data/split_warning and code_sha256/assets. Audit compares actual raw data with these counts and six split hashes; does not rerun GPU scores."),
    {"id":"original-execution","kind":"execution","title":"Original 5.11 fence CPU execution","artifact_id":paths["original/stdout.txt"],"verified":True},
    {"id":"independent-execution","kind":"execution","title":"Own bounded fence variants and original-data CPU audit","artifact_id":paths["probe.stdout.json"],"verified":True},
    {"id":"identity-derivation","kind":"derivation","title":"Finite hash output and uniform record sampling derivations","verified":True,
     "details":"SHA-256 is 256 bits = 32 bytes = 64 hex digits (4 bits per digit). Reducing to k hex characters leaves 16^k outcomes. Fixed 256-bit identity cannot encode all longer texts reversibly and is not a semantic-similarity metric. For uniformly sampled three records with two copies, content mass is 2/3; dedup to two distinct records makes it 1/2. No model-quality measurement is inferred."},
]
def evidence(source_id,locator,supports): return {"source_id":source_id,"locator":locator,"supports":supports}
def claim(identifier,kind,statement,location,scope,ev,arts,verification=None):
    v={"id":identifier,"kind":kind,"statement":statement,"location":location,"scope":scope,"status":"verified",
       "evidence":ev,"artifact_ids":[paths[a] for a in arts]}
    if verification: v["verification"]=verification
    return v
claims=[
    claim("duplicate-risk","concept","相同內容可跨訓練／驗證；訓練側副本會增加同內容的抽樣份量。","course/chapters/05.md:395,414",
        "支持重複內容導致重疊的風險及以記錄為抽樣單位的分配變化；不聲稱每種重加權抽樣器必然一樣，也不從 toy 數字推論模型成績。",
        [evidence("lee-dedup","Abstract/Introduction pp.8424–8425, text lines 10–84","原論文確認訓練／驗證的重複內容與評估重疊風險。"),
         evidence("identity-derivation","Uniform record sampling, 2 copies among 3 vs 1 among 2","精確算例展示副本增加同內容所占比例。")],
        ["variation-results.json","inspection.md"]),
    claim("toy-counts","numeric","三筆得到兩種正規化內容；SHA-256 完整十六進位指紋長64，符號為0–9/a–f；換第二筆為鳥 看狗得到三種內容。","course/chapters/05.md:402–416",
        "只核對這三筆輸入及 SHA-256 摘要表示；例子的 set(hash) 沒有碰撞。",
        [evidence("python-hashlib","hash.hexdigest lines 596–602","十六進位字串長度是 digest bytes 的兩倍。"),
         evidence("nist-shs","Section 1 Figure 1, printed p.2, SHA-256 row","摘要256 bits；256/8*2=64。"),
         evidence("independent-execution","variation-results.json variations exact_original/bird_old_assert/bird_corrected_assert/extra_whitespace","實際核對內容數2/3、長64及教材練習。")],
        ["original/stdout.txt","probe.stdout.json","variation-results.json"],
        {"method":"executed","expected":"原輸入3筆/2內容/64 hex；鳥變化3內容；空白變化2內容。",
         "observed":"原stdout 3/2/64，所有四項必要變化吻合；每個摘要字符皆為0–9/a–f。",
         "details":"Exact original fence plus changed-input variants; official 256-bit digest implies 64 hexadecimal characters.","tolerance":"整數、字串與集合精確相等；無浮點容忍差。"}),
    claim("fence-apis","software","原fence及fingerprint按照str.split→單一空格join→UTF-8 encode→sha256→hexdigest；len/set計數與assert如教材所述。","course/chapters/05.md:400–410",
        "是正規化及指紋比較示範；没有建立模型、梯度、更新或評測。split另去除首尾空白並合併Unicode whitespace；本例沒有邊緣空白。",
        [evidence("python-strings","str.split/str.join/str.encode API locators; lines 3500–3568,3116–3130,2554–2565","逐項核對空白分段、接回及預設UTF-8語義。"),
         evidence("python-hashlib","Hash algorithms; hash.hexdigest, lines 286–303/596–602","sha256接受bytes並返回hex摘要。"),
         evidence("fingerprint-code","tiny_perceptron/data.py:102–104","helper實際使用相同規則。"),
         evidence("original-execution","original/fence-1.py and stdout.txt","原碼確實成功執行。"),
         evidence("independent-execution","variations and edge_cases in variation-results.json","舊assert換字後失敗，新不等assert成功；額外空白／Unicode／empty核對。")],
        ["original/stdout.txt","probe.stdout.json","original/fence-1.py","variation-results.json"],
        {"method":"executed","expected":"原碼exit0；換字使舊等式assert失敗；改為不等後成功；增加空白仍相同。",
         "observed":"各項都達到預期；例外AssertionError被記為預期，不冒稱原練習始終通過。",
         "details":"section_facts exact fence exit0 with no guard events, plus separate own bounded CPU probes; Python3.13.5 matches official major/minor API."}),
    claim("hash-scope","concept","完整內容指紋是固定長一向摘要，不是可還原壓縮或語意同義證明；輸入字變化常使摘要改變，碰撞仍可能；極短前綴失去區分資訊。","course/chapters/05.md:412–414",
        "支持內容身份及安全摘要的性質；不提供數學零碰撞保證，不以示例125-bit變化重新證明整個SHA-256，也不作語意相似度評測。",
        [evidence("nist-shs","Explanation item3 lines74–87; Section1 lines227–261","官方一向摘要、難找preimage/collision及變更訊息高機率改變摘要的說明。"),
         evidence("identity-derivation","256-bit finite output and 16^k prefix identities","不可還原所有輸入，前綴減少可區分身份；摘要映射沒有語意規則。"),
         evidence("independent-execution","variation-results.json short_prefix_collision/changed_bits_one_character_example","僅示範1hex前綴碰撞而完整指紋不同，與換字示例。")],
        ["variation-results.json","sources/nist-fips180-4.pdf","inspection.md"]),
    claim("meaningful-whitespace","concept","空白合併是本例比較規則；縮排、詩換行或字串空白可能有意義，不能把全文正規化當普遍保義或要求訓練原文也清空白。","course/chapters/05.md:397,414",
        "本例文字保持非空白字序；正文的『本例』及後續明確保義限制共同限定說法。詩的排版是可有意義的素材資訊，不宣稱去空白不會影響任何任務。",
        [evidence("python-whitespace","Lexical Analysis 2.1.8 and 2.1.9, lines437–441/518–525","縮排決定Python語句分組；字串內空白不能一般視作token分隔。"),
         evidence("python-strings","str.split and str.join API","此轉換確實丟掉某些空白差異，屬選定比較等價關係。"),
         evidence("independent-execution","variation-results.json whitespace_can_be_significant","自寫短例確認扁平化多句Python會改變parse，A double-space B變成single-space；不替普遍語意下結論。")],
        ["variation-results.json","inspection.md"]),
    claim("real-text-pipeline","software","T.4 real_text先按正規化後完整字串去重，保留第一筆原text，加完整SHA-256 family，再按family切分；訓練token仍保留原文空白換行。","course/chapters/05.md:421",
        "核對原始實作契約及短CPU替代，不執行T.4 CUDA/800-step recipe。家族身份以完整正規化內容為限；不包含近重複家族辨識。",
        [evidence("text-pipeline","_deduplicate_text:91–99; run_real_text:323–331; _save_splits:46–61","先去重再split；只新增family而保留原text。"),
         evidence("split-code","split_records:50–70; text_examples:77–97","分组不拆family，後續encode用row原text。"),
         evidence("independent-execution","data-audit-results.json synthetic_helper_variations and raw_whitespace_tokenization_checked","執行原歷史函式：first-row原text保留、同family兩衍生文本同側；700/171原UTF8+EOS targets精確相等。")],
        ["probe.stdout.json","data-audit-results.json","historical/_deduplicate_text.py","historical/split_records.py","historical/text_examples.py"],
        {"method":"executed","expected":"去重只比較normalized完整text；first-row原text和metadata保留；samefamily同側；raw字節目標不改。",
         "observed":"synthetic helper variants成功；原512/365 text逐筆保留；兩筆newline樣本所有target等於原UTF8字節+EOS。",
         "details":"git show historical source SHA matches original report and frozen current source; exact function spans execute on CPU with no model creation/training. Recipe only read for pipeline contract."}),
    claim("original-record-counts","empirical","本次TinyStories512篇與365首詩在完整正規化內容去重後仍分別512/365，這一步未再刪；未做近重複聚類。","course/chapters/05.md:421",
        "核對指定original real_text run及匹配manifest的小資料包；不代表上游從未去重，不代表沒有改寫／相似版本，不是重新訓練或由模型成績推論去重。",
        [evidence("original-result","results.runs.{tinystories,chinese-poetry}.{source_records,deduplicated_records,data,split_warning}; code_sha256","指定版本原測量數量與各split identities。"),
         evidence("independent-execution","data-audit-results.json data.{tinystories,chinese-poetry}","親自讀raw JSONL：512/365 normalized全文全異；去重removed0；6個split hash/count精確重現。"),
         evidence("lee-dedup","Section4 p.8426, text167–179","完整字符串精確相同與片段／近重複不是同一範圍；支持教材限制。")],
        ["probe.stdout.json","data-audit-results.json","inputs/docs/course-experiments/results/real_text.json","inputs/data/training/text-initial/tinystories-train-512.jsonl","inputs/data/training/text-initial/chinese-classical-train-365.jsonl"],
        {"method":"executed","expected":"source=dedup counts512/365；seed42 splits409/51/52及292/36/37；各完整hash吻合原JSON。",
         "observed":"所有整數及6個SHA-256精確相等；兩套removed0，cross-split family overlap0；原資料manifest byte/sha也匹配。",
         "details":"Executed cleaning and deterministic split contract using matching historical code and frozen raw input; no scores recomputed. Original report deviceCUDA/torch2.14.1+cu126 is distinguished from Python3.13.5/torch2.14.1+cpu audit.",
         "denominators":{"tinystories_raw_records":512,"tinystories_normalized_unique_records":512,"poems_raw_records":365,"poems_normalized_unique_records":365,"total_raw_records":877,"split_files_verified":6,"new_training_steps":0,"near_duplicate_clusters_audited":0}}),
]
report={"schema_version":1,"review_stage":"technical","lesson_id":"5.11","source":"course/chapters/05.md#5.11",
    "source_sha256":digest(OUT/"original/section.md"),"figure_sha256":{},"verdict":"pass",
    "reviewer_task":"/root/phase4_factual_coordinator/factual_5_11","reviewer_context":"fresh","author_tasks":[],
    "review_date":"2026-10-05","read_scope":"Whole5.11; T.4 and nearby boundaries; requested review methods/schema/helper; cited source/code scopes in inspection.md. No prior report read; not chapter-first.",
    "artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","details":"每項實質claim核對官方原始來源或原碼；SHA範圍與清理規則明確，沒有轉用別人的判定。","claim_ids":[c["id"] for c in claims]},
        "numeric_verification":{"status":"pass","details":"原fence3→2/64hex；鳥變化→3；raw512→512與365→365；六個原切分SHA和數量精確重現。","claim_ids":["toy-counts","original-record-counts"]},
        "figure_consistency":{"status":"not_applicable","details":"原文與extraction都無圖/SVG引用；沒有原圖可render/view，沒有宣稱瀏覽器或全站視覺驗證。","claim_ids":[]},
        "source_verification":{"status":"pass","details":"親讀Python官方API、NIST FIPS原PDF與Lee ACL原發表PDF；官方URL版本/日期/locator及支持範圍逐claim留存。","claim_ids":["duplicate-risk","fence-apis","hash-scope","meaningful-whitespace","original-record-counts"]},
        "limitations":{"status":"pass","details":"只核對完整內容去重及local既有data；不把零刪除當無近重複、不以toy抽樣/hash變化當模型成績、不重跑訓練。初次probe自身假設失敗已保存並改正，最終驗證成功；無圖視覺檢查NA。","claim_ids":["duplicate-risk","hash-scope","real-text-pipeline","original-record-counts"]}}}
save(ROOT/"docs/technical-reviews/5.11.json",report)
print(json.dumps({"lesson":"5.11","verdict":report["verdict"],"claims":len(claims),"artifacts":len(artifacts),"source_sha256":report["source_sha256"]},ensure_ascii=False))
