"""Write this reviewer's newly authored report; never parse any prior review."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
P = A.relative_to(R).as_posix()
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def artifact(identifier, name, kind, description, **extra):
    return dict(id=identifier, path=f"{P}/{name}", sha256=sha(A/name), kind=kind, description=description, **extra)
cpu = json.loads((A/"cpu-result.json").read_text())
render = json.loads((A/"figure-render-result.json").read_text())
artifacts = [
    artifact("frozen-chapter", "frozen/course/chapters/01.md", "source_snapshot", "首次讀取的完整章原始 bytes；SHA 僅指 frozen input，不冒充目前全章版本。"),
    artifact("section-original", "extracted/section.md", "source_snapshot", "原 UTF-8 1.1 bytes，包含標題與尾端空行，不正規化換行。"),
    artifact("intro-original", "intro.md", "source_snapshot", "原 UTF-8 章01導言 bytes，從首字至第一個 ## 前。"),
    artifact("extraction", "extracted/extraction.json", "source_snapshot", "section_facts 實際擷取 metadata、原始行號、fence 及圖 SHA。"),
    artifact("fence-original", "extracted/fence-1.py", "code", "直接取自原稿行24–34的完整 Python fence。"),
    artifact("reverse-code", "reverse-fence.py", "code", "只將 chars 排序改成 reverse=True 的原練習。"),
    artifact("audit-code", "audit_cpu.py", "code", "本人有界 CPU 程式，執行原 fence、反向、交叉表、未知字及未改 CharTokenizer class。"),
    artifact("cpu-run", "cpu-result.json", "execution", "本人的精確字元／ID／stdout、練習、KeyError、Notebook及實作契約檢查。",
             command="/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-1.1/audit_cpu.py",
             result="Exit 0；所有 exact assertions 通過；不涉及訓練、GPU 或模型下載。", environment=cpu["environment"]),
    artifact("audit-attempt1", "audit-attempt1.txt", "execution", "保存本人初次 Notebook byte 比較失敗及 terminal LF 唯一差異的診斷。",
             command="/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-1.1/audit_cpu.py (first attempt)",
             result="Exit 1；原 fence 267 bytes、Notebook 266 bytes，只缺 terminal LF；之後依實際生成慣例完成核對。", environment=cpu["environment"]),
    artifact("notebook-source", "frozen/notebooks/01/1.1.ipynb", "code", "目前 Notebook 全檔原 bytes，實際讀 code cells 2、3、5；只執行核心 cell 5 相同算例。"),
    artifact("data-source", "frozen/tiny_perceptron/data.py", "code", "原 CharTokenizer 實作快照，實際讀1–57、執行未改 class AST 31–43。"),
    artifact("svg-source", "frozen/course/figures/rewrite-01-character-ids.svg", "source_snapshot", "本人親自核對並渲染的原 SVG 完整 bytes。"),
    artifact("figure-image", "figure-native.png", "figure_render", "本人以 view_image 實際查看的640×710全圖；八組字元／ID和箭頭一致。"),
    artifact("render-code", "render_figure.py", "code", "Chromium 直接渲染 frozen SVG bytes；file://政策限制改由 set_content 載入同 bytes。"),
    artifact("figure-run", "figure-render-result.json", "execution", "本人新渲染命令、SVG與PNG SHA、browser版本、原圖文字。",
             command=render["command"], result="Exit 0；完整SVG載入與截圖成功，之後本人呼叫view_image查看。",
             environment={"browser":"Chromium "+render["browser"],"device":"CPU headless","network":"none; page.set_content from frozen SVG bytes"}),
    artifact("inspection", "inspection.md", "derivation", "本人實際讀取範圍、雙射推導、圖視覺核對、已修復的工具限制及權威定位。"),
    artifact("fetch-code", "fetch_sources.py", "code", "權威原始來源取得程式；保留HTTPS/TLS驗證。"),
    artifact("fetch-receipts", "source-fetch-receipts.json", "source_snapshot", "本次官方HTTPS原始來源版本、實際存取日期、bytes及完整SHA。"),
]
for name in ("functions.rst","stdtypes.rst","exceptions.rst","simple_stmts.rst","unicode.rst","expressions.rst","bengio2003.pdf","bengio2003.txt"):
    artifacts.append(artifact("official-"+name, "official/"+name, "source_snapshot", "本人親讀的官方原始來源完整快照；txt為本次pdftotext -layout產物。"))

sources=[]
source_specs = [
    ("py-functions","functions.rst","library/functions.rst","builtins enumerate、ord、sorted；545–576、1547–1564、1848–1874。"),
    ("py-types","stdtypes.rst","library/stdtypes.rst","list索引、Unicode str、str.join、set去重、dict查值／設值；920–985、1050–1093、1512–1533、1946–1964、4410–4460、4671–4757。"),
    ("py-exceptions","exceptions.rst","library/exceptions.rst","KeyError 定義；268–282。"),
    ("py-assert","simple_stmts.rst","reference/simple_stmts.rst","assert求值、未成功拋AssertionError、-O邊界；380–416。"),
    ("py-unicode","unicode.rst","howto/unicode.rst","字元與Unicode code point定義；35–79。"),
    ("py-comparison","expressions.rst","reference/expressions.rst","str lexicographic比較以Unicode數值碼位；1594–1609。"),
]
for identifier,name,urlpart,note in source_specs:
    sources.append(dict(id=identifier,kind="official_source",title="CPython 3.13.5 official documentation source: "+name,
                        verified=True,url="https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/"+urlpart,
                        version="CPython tag v3.13.5; installed Python 3.13.5",authority_reason="Python官方CPython repository精確版本的語言與標準庫原文件。",
                        checked_original=True,accessed_on="2026-10-06",inspection_note=note,
                        snapshot_artifact_id="official-"+name))
sources.append(dict(id="bengio2003",kind="paper",title="Bengio, Ducharme, Vincent, Jauvin (2003), A Neural Probabilistic Language Model",
                    verified=True,url="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf",version="JMLR 3 (2003), pp.1137–1155",
                    authority_reason="JMLR出版者網站的作者原論文，直接定義下一詞條件機率與詞索引／參數查表。",
                    checked_original=True,accessed_on="2026-10-06",inspection_note="親讀PDF pp.1137–1139、§2 pp.1141–1142；本次txt行1–160、242–280、290–314；只用機制與符號定義，不沿用benchmark。",
                    snapshot_artifact_id="official-bengio2003.pdf"))
for identifier,name,note in [
    ("original-fence","extracted/fence-1.py","原稿1.1行24–34完整原碼，本人完整執行及逐行核對。"),
    ("repository-char","frozen/tiny_perceptron/data.py","讀1–57，AST定位CharTokenizer31–43並執行原class契約；含UNK，與本節獨立字典分開。"),
    ("notebook","frozen/notebooks/01/1.1.ipynb","讀code cells2、3、5；cell5與原fence僅terminal LF不同；未執行clone/pip。"),
    ("frozen-chapter","frozen/course/chapters/01.md","首次原稿快照；本人讀1–190、202–238、433–471、512–556；導言路線背景，非以教材自證技術正確。"),
]:
    sources.append(dict(id=identifier,kind="repository_code",title=identifier,verified=True,path=P+"/"+name,sha256=sha(A/name),version="Frozen current repository input, 2026-10-06",inspection_note=note))
sources.append(dict(id="own-cpu",kind="execution",title="Independent exact CPU check",verified=True,artifact_id="cpu-run"))
sources.append(dict(id="own-render",kind="execution",title="Independent Chromium SVG rendering",verified=True,artifact_id="figure-run"))
sources.append(dict(id="own-proof",kind="derivation",title="Lookup bijection and synchronized permutation",
                    verified=True,details="五個distinct chars以enumerate定義E(c[i])=i、D(i)=c[i]，逐項D(E(c[i]))=c[i]。任意重新排列只要同步重建E/D仍成立；ID不攜帶此定義未指定的字義大小或距離。本人推導見inspection.md；reverse及cross-table結果另實際執行。"))

def ev(source,locator,supports):
    return dict(source_id=source,locator=locator,supports=supports)
def claim(identifier,kind,statement,location,scope,evidence,artifact_ids,verification=None):
    item=dict(id=identifier,kind=kind,statement=statement,location=location,scope=scope,status="verified",evidence=evidence,artifact_ids=artifact_ids)
    if verification is not None: item["verification"]=verification
    return item
def verify(expected,observed,details,tolerance=None,method="executed"):
    d=dict(method=method,expected=expected,observed=observed,details=details)
    if tolerance is not None:d["tolerance"]=tolerance
    return d
claims=[
 claim("intro-next-symbol","concept","章首以已給出的前文對下一字候選打分，依次教字／ID、可調數值與把選出的字接回前文。","course/chapters/01.md:1–4；必要路線背景1.6、1.12、1.14",
       "導言以字元示範離散符號的自回歸接續；輸入法候選是可能的教學例子，未宣稱測過某一輸入法；不是所有文字模型的普遍架構敘述，也不是1.1完成訓練的成績。",
       [ev("bengio2003","§1 p.1138 next-word conditional probability；§2 p.1141 conditional distribution and free-parameter lookup","支持依前文為下一離散符號建立分配和可調查表機制；原論文word-level，本文字元例子為同機制的教學具體化。"),ev("frozen-chapter","lines202–238、433–471、512–556","只核對章內確有可調表、參數更新和接回新字的教學路線；不沿用後節實測數字。")], ["intro-original","inspection","official-bengio2003.pdf"]),
 claim("fixed-ids","numeric","五字字表為。狗看貓，及IDs0..4，八字原句IDs為[3,2,1,4,1,2,3,0]，相同字與標點使用相同ID。","course/chapters/01.md:9–15、24–34",
       "限原句與本節自建五字字表；不是UTF-8 bytes，也不等於含UNK的專案CharTokenizer索引。",
       [ev("own-cpu","cpu-result.json /original/chars、/original/ids、/original/characters、/original/vocabulary_size","本人原碼精確輸出五個字、八個ID及重複一致。"),ev("own-proof","inspection.md 本人算例與範圍","enumerate的唯一符號到索引映射。")],["cpu-run","fence-original","inspection"],verify("chars=['。','狗','看','貓','，']; ids=[3,2,1,4,1,2,3,0]; 8 characters, 5 unique","完全相同。","逐整數、字元、順序與長度assert，無浮點數。","精確相等；不作近似。")),
 claim("set-unicode-sort","software","set去重，sorted依Unicode字元數值碼位排列表內字，不按字義或筆畫。","course/chapters/01.md:21、25、46",
       "每個字在此均是單一Python Unicode code point；不推廣到多碼位字形的語言學拆字，也不將索引本身當碼位。",
       [ev("py-types","stdtypes.rst:1512–1533、4410–4460","str為Unicode code point sequence，set為distinct unordered elements。"),ev("py-functions","functions.rst:1547–1564、1848–1874","ord碼位及sorted直接比較。"),ev("py-comparison","expressions.rst:1600–1603","str比較使用numerical Unicode code points。"),ev("own-cpu","cpu-result.json /original/unicode_code_points","本人計算12290<29399<30475<35987<65292。")],["cpu-run","official-functions.rst","official-stdtypes.rst","official-expressions.rst"],verify("set去重得到5符號；碼位由小到大。","chars順序。狗看貓，；碼位12290、29399、30475、35987、65292。","執行原sorted(set(text))並獨立查ord逐值assert。")),
 claim("enumerate-lookup","software","enumerate提供從0開始的索引／字元；to_id按原句逐字查ID；清單以ID查回字。","course/chapters/01.md:26–30、37、39",
       "普通Python dict與list、同一張表的已知字；不是網路分詞API或模型推論。",
       [ev("py-functions","functions.rst:550–571","enumerate從0產生counter與iterable值。"),ev("py-types","stdtypes.rst:979；4721–4753","list s[i] origin0，dict d[key]查值及d[key]=value。"),ev("own-cpu","cpu-result.json /original/ids、/original/restored","本人執行建立字典、lookup與逆lookup。")],["cpu-run","fence-original","official-functions.rst","official-stdtypes.rst"],verify("0..4位置/字配對，八字lookup保持順序，inverse list還原原文。","全部精確相同。","直接執行完整fence且逐值assert。")),
 claim("roundtrip-limit","concept","文字與ID在同表中可往返；ID數值不表示字義大小／相近；往返成功不表示已會猜字或理解語义。","course/chapters/01.md:15、19、39",
       "本節五個獨立符號的查表雙射；只有原句roundtrip assertion，不宣稱任意未知字通用還原、模型能力或embedding沒有語義。",
       [ev("py-functions","functions.rst:550–571","索引僅由enumerate位置產生。"),ev("py-types","stdtypes.rst:979、1951–1956、4721–4753","list/dict lookup與join純字串操作。"),ev("bengio2003","§2 p.1141、Figure1 p.1142","符號索引與另外學習的feature vector和條件機率參數分開。"),ev("own-proof","inspection.md 雙射及置換推導","E/D與同步置換不依赖數值語義；只證記錄還原。"),ev("original-fence","lines1–11 (original source24–34)","無候選分數、求導、更新或訓練操作。")],["inspection","cpu-run","fence-original"]),
 claim("stdout-assert-notebook","software","三行print為字表、ID、原句；join接成字串；assert相等在普通執行成功時不額外輸出；Notebook核心算例一致。","course/chapters/01.md:30–39；notebooks/01/1.1.ipynb code cell5",
       "本次Python3.13.5正常模式，未用-O；Notebook僅terminal LF排版差異，未執行Colab clone或pip。",
       [ev("py-types","stdtypes.rst:1951–1956","str.join返回按iterable拼接的字串。"),ev("py-assert","simple_stmts.rst:388–413","普通模式true assertion不拋錯；-O會移除。"),ev("own-cpu","cpu-result.json /original/stdout、/notebook_cell_5_match","本人執行三行精確stdout並查Notebook內容。")],["cpu-run","notebook-source","fence-original","audit-attempt1","official-simple_stmts.rst"],verify("三行內容與教材相同；無第四行；Notebook只缺terminal LF。","三行全部精確通過；267 vs266 bytes差異診斷只為末端LF。","直接capture stdout並splitlines逐行assert；教材省略清單repr空格是版面簡寫。")),
 claim("reverse-table","numeric","reverse=True改變字表及ID，但同步用同一張新表編碼／解碼仍還原原句。","course/chapters/01.md:41",
       "依原練習只改sorted reverse參數；必要同步重建to_id；不支持拿不同表解舊IDs。",
       [ev("py-functions","functions.rst:1848–1860","reverse=True逆轉比較排序。"),ev("own-cpu","cpu-result.json /reverse、/cross_table_decode","本人執行新表、新IDs、原句還原及混表反例。"),ev("own-proof","inspection.md 同步置換推導","同表雙射保留roundtrip。")],["cpu-run","reverse-code","inspection"],verify("新chars=['，','貓','看','狗','。']; IDs=[1,2,3,0,3,2,1,4]; 原句不變。","精確相同；舊IDs配新表則狗看貓。貓看狗，。","執行原fence的唯一一行變更與cross-table變體。","逐字／整數精確相等。")),
 claim("unknown-keyerror","software","原五字字典編碼貓看鳥，缺鳥造成KeyError。","course/chapters/01.md:46",
       "只限本例直接d[key]，專案CharTokenizer預留UNK且用.get(c,0)，本人確認兩者契約不同；現稿已限定本例。",
       [ev("py-exceptions","exceptions.rst:273–277","找不到mapping key會KeyError。"),ev("py-types","stdtypes.rst:4721–4732","普通dict d[key]缺key則KeyError。"),ev("own-cpu","cpu-result.json /unknown_character、/repository_contract","本人捕捉KeyError('鳥')並實際執行未改CharTokenizer顯示UNK。"),ev("repository-char","tiny_perceptron/data.py:31–43","repo實作預留UNK並.get，不能替本節聲稱另一契約。")],["cpu-run","data-source","official-exceptions.rst"],verify("KeyError args=('鳥',)；repo實作另有UNK。","KeyError鳥；repo unknown IDs[4,3,0]、decode貓看<unk>。","執行原dict未知字查詢，AST執行原CharTokenizer作契約界線核對。")),
 claim("figure-pairs","numeric","圖的八個字與ID、固定重複對應、雙向箭頭及先上排再下排顺序和原碼一致。","course/chapters/01.md:17–19；course/figures/rewrite-01-character-ids.svg",
       "本人查看本次640×710 SVG原bytes渲染PNG；核對內容與箭頭，不宣稱已測教材網站mobile layout。",
       [ev("own-render","figure-render-result.json /texts、/source_sha256、/screenshot_sha256","本人Chromium新渲染原SVG及文字定位。"),ev("own-cpu","cpu-result.json /original/ids","原碼獨立八ID結果。"),ev("own-proof","inspection.md 本人圖解查看","本人view_image實際看到上下兩排八配對和雙向箭頭。")],["figure-image","figure-run","svg-source","cpu-run","inspection"],verify("上排貓3、看2、狗1、，4；下排狗1、看2、貓3、。0；八雙向配對。","本人實際圖片內容與上述完全相同；兩個貓3、兩個狗1。","目視八組配對順序／方向與獨立CPU結果逐項比較。","字元、ID及方向精確相同。",method="hand_calculation")),
]
report=dict(schema_version=1,review_stage="technical",lesson_id="1.1",source="course/chapters/01.md#1.1",
            source_sha256=sha(A/"extracted/section.md"),verdict="pass",reviewer_task="/root/p6_fact_1_1",reviewer_context="fresh",
            reviewed_on="2026-10-06",intro_source="course/chapters/01.md:1–4",intro_sha256=sha(A/"intro.md"),
            intro_summary="導言借輸入法下一字候選引出前文條件打分，安排先建立字與編號、問題與答案，再調數值表，最後接回選出字來生成；沒有說1.1查表已具語義或预测能力。",
            frozen_input=dict(path=P+"/frozen/course/chapters/01.md",sha256=sha(A/"frozen/course/chapters/01.md"),meaning="完整章首次讀取原始快照；僅對保存版本有效，非報告完成時目前全章hash。"),
            read_scope=["course/chapters/01.md:1–190,202–238,433–471,512–556","1.1 original fence complete","notebooks/01/1.1.ipynb code cells2,3,5","tiny_perceptron/data.py:1–57; executed original class AST31–43","tiny_perceptron/tokenization.py:1–155 background only, not relied on as evidence","current review method/schema/helper complete","official locators listed per source and inspection.md","personally rendered and viewed figure-native.png"],
            figure_sha256={"course/figures/rewrite-01-character-ids.svg":sha(A/"frozen/course/figures/rewrite-01-character-ids.svg")},
            sources=sources,artifacts=artifacts,claims=claims,issues=[],
            checks={
                "factual_accuracy":dict(status="pass",details="逐主張對照原始官方文件、原論文、原碼與本人雙射推導；lookup与學習能力分開。",claim_ids=[c["id"] for c in claims]),
                "numeric_verification":dict(status="pass",details="五字八ID、Unicode碼位、新表新ID與混表反例精確核對；不含任何訓練成績。",claim_ids=["fixed-ids","reverse-table","figure-pairs"]),
                "figure_consistency":dict(status="pass",details="本人Chromium新render後實際view_image，核對八配對、標點、箭頭與上下排順序。",claim_ids=["figure-pairs"]),
                "source_verification":dict(status="pass",details="CPython v3.13.5精確原文件與JMLR作者原論文皆親讀定位；HTTPS/TLS及完整SHA永久保存。",claim_ids=["intro-next-symbol","set-unicode-sort","enumerate-lookup","roundtrip-limit","stdout-assert-notebook","unknown-keyerror"]),
                "limitations":dict(status="pass",details="僅本例已知字的查表還原；自建dict未知字KeyError與repo UNK不同；無預測、語義、模型評測、GPU或訓練宣稱；assert正常模式。",claim_ids=["intro-next-symbol","roundtrip-limit","stdout-assert-notebook","unknown-keyerror"]),
            })
(R/"docs/technical-reviews/1.1.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"verdict":report["verdict"],"source_sha256":report["source_sha256"],"intro_sha256":report["intro_sha256"],"figure_sha256":report["figure_sha256"],"report_sha256":sha(R/"docs/technical-reviews/1.1.json"),"claims":len(claims),"artifacts":len(artifacts)},ensure_ascii=False,indent=2))
