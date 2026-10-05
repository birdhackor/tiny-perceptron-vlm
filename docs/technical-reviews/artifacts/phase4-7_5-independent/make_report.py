"""Build this independently authored review from its permanent evidence files."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREFIX = HERE.relative_to(ROOT).as_posix()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

run = json.loads((HERE / "bounded-controls.stdout.json").read_text())
extraction = json.loads((HERE / "original/extraction.json").read_text())
env = {key: str(value) for key, value in run["environment"].items()}
files = sorted(path for path in HERE.rglob("*") if path.is_file() and not path.name.startswith("checker"))
artifacts, file_ids = [], {}
for index, path in enumerate(files):
    relative = path.relative_to(HERE).as_posix()
    identifier = f"a{index+1:02d}"
    file_ids[relative] = identifier
    kind = "code" if path.suffix == ".py" else "source_snapshot"
    if relative == "inspection.md": kind = "derivation"
    if relative == "context-figure/alignment.png": kind = "figure_render"
    artifact = {"id": identifier, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                "kind": kind, "description": f"本節獨立永久證據：{relative}；用途與親讀範圍見inspection.md及command-receipts.json。"}
    if relative in ("original/stdout.txt", "bounded-controls.stdout.json", "context-figure/render.stdout.txt"):
        artifact["kind"] = "execution"
        artifact["environment"] = env
        if relative == "original/stdout.txt":
            artifact["command"] = ".venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.5 --execute --output /tmp/phase4-7_5-original --timeout 45"
            artifact["result"] = "exit 0；第一位置logits梯度=0.0；Q字向量梯度=0.00571822514757514。原碼與CPU環境均已保存。"
        elif relative == "bounded-controls.stdout.json":
            artifact["command"] = "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-7_5-independent/bounded_controls.py"
            artifact["result"] = "exit 0；Q/Z、讀取阻断、detach、retain_grad、因果軸、有效分母、解析梯度和參數未更新全部assert通過。"
        else:
            artifact["command"] = "timeout 25 inkscape docs/technical-reviews/artifacts/phase4-7_5-independent/context-figure/alignment.svg --export-type=png --export-width=1600 --export-filename=docs/technical-reviews/artifacts/phase4-7_5-independent/context-figure/alignment.png"
            artifact["result"] = "exit 0；Inkscape 1.4 產生前節7.4對齊圖PNG，reviewer已實際view；7.5自身無引用圖。"
            artifact["environment"]["renderer"] = "Inkscape 1.4 (e7c3feb100, 2024-10-09)"
    artifacts.append(artifact)

def ids(*names): return [file_ids[name] for name in names]
orig_ids = ids("original/stdout.txt", "original/fence-1.py", "original/execution.json", "original/environment.json", "original/extraction.json", "original/section.md")
control_ids = ids("bounded_controls.py", "bounded-controls.stdout.json", "bounded-controls.stderr.txt", "command-receipts.json", "inspection.md")
docs = {
    "ce": ("cross-entropy", "CrossEntropyLoss", "generated/torch.nn.CrossEntropyLoss.html", "親讀class-index unreduced/reduction公式、ignore_index與Shape，來源文字123–168、223–226、252–266。"),
    "autograd": ("autograd", "Autograd mechanics", "notes/autograd.html", "親讀How autograd encodes the history與Setting requires_grad，文字132–148、325–359；確認反向鏈式法則與凍結須獨立設定。"),
    "retain": ("retain-grad", "Tensor.retain_grad", "generated/torch.Tensor.retain_grad.html", "親讀API 123–126：保留非leaf梯度；leaf為no-op。"),
    "leaf": ("is-leaf", "Tensor.is_leaf", "generated/torch.Tensor.is_leaf.html", "親讀API 123–135與原例：預設只填leaf的grad，nonleaf需retain_grad。"),
    "detach": ("detach", "Tensor.detach", "generated/torch.Tensor.detach.html", "親讀API 123–140：detach從當前圖分離，返回張量不需梯度但共用storage。"),
    "embedding": ("embedding", "nn.Embedding", "generated/torch.nn.Embedding.html", "親讀123–176：整數查表、learnable weight shape與padding_idx例外；本節沒有設定padding_idx。"),
    "norm": ("torch-norm", "torch.norm", "generated/torch.norm.html", "親讀123–220：default fro、flatten規則、fro與p=2等價条件；記錄API已標deprecated但安裝版可用。"),
    "tensor-norm": ("norm", "Tensor.norm", "generated/torch.Tensor.norm.html", "親讀123–126：default p=fro及See torch.norm。"),
    "vector-norm": ("vector-norm", "linalg.vector_norm", "generated/torch.linalg.vector_norm.html", "親讀123–183：實數向量2-norm和sum(abs(x)^ord)^(1/ord)定義。"),
    "backward": ("backward", "Tensor.backward", "generated/torch.Tensor.backward.html", "親讀123–180：對graph leaves按鏈式法則累積梯度；沒有參數更新操作。"),
    "item": ("item", "Tensor.item", "generated/torch.Tensor.item.html", "親讀123–130：單元素取為Python number，非微分操作。"),
    "seed": ("manual-seed", "torch.manual_seed", "generated/torch.manual_seed.html", "親讀123–135：seed設定亂數生成器，本例固定42。"),
    "sdpa": ("sdpa", "scaled_dot_product_attention", "generated/torch.nn.functional.scaled_dot_product_attention.html", "親讀API等價程式及query/key/mask參數：True為允許，最後兩軸是query/key。本例實際用manual attention，另親讀repo契約。"),
}
sources = []
for key, (filename, title, suffix, note) in docs.items():
    sources.append({"id":key,"kind":"official_docs","title":f"PyTorch 2.9 {title}","verified":True,
        "url":"https://docs.pytorch.org/docs/2.9/"+suffix,"version":"PyTorch official docs 2.9；安裝執行為2.14.1+cpu，另查其精確commit原碼",
        "accessed_on":"2026-10-05","authority_reason":"PyTorch維護者在docs.pytorch.org發佈的版本化原始API文件。",
        "checked_original":True,"inspection_note":note+f" 原始HTML與可讀衍生文字保存在{PREFIX}/sources/{filename}.*。"})
sources.append({"id":"paper","kind":"paper","title":"Attention Is All You Need — original arXiv v7 PDF","verified":True,
    "url":"https://arxiv.org/pdf/1706.03762v7","version":"arXiv:1706.03762v7, 2023-08-02; NIPS 2017 original author paper",
    "accessed_on":"2026-10-05","authority_reason":"原作者原論文的arXiv版本化PDF；已親讀封面作者、版本及§3正文，未沿用來源庫摘要。",
    "checked_original":True,"inspection_note":"親讀PDF封面、pp.3–4 §3.1 Decoder與§3.2.1 Eq.(1)，因果遮罩禁止未來、attention是依key/query權重合成values；不把原論文當repo標籤格式或非零梯度保證。"})
for key, filename, note in [
    ("installed-functional","installed-functional.py","親讀cross_entropy 3478–3571及SDPA等價實作6366–6409、mask形狀6480–6508，與安裝commit版本一致。"),
    ("installed-tensor","installed-tensor.py","親讀Tensor.backward 566–630及Tensor.norm 888–902，backward只累積梯度、norm dispatch到torch.norm。")]:
    suffix = "torch/nn/functional.py" if key == "installed-functional" else "torch/_tensor.py"
    sources.append({"id":key,"kind":"official_source","title":"PyTorch original source "+suffix,"verified":True,
        "url":"https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/"+suffix,
        "version":"commit 5c4886908584029761b579af026dcfb627c84070; installed torch.version.git_version",
        "accessed_on":"2026-10-05","authority_reason":"PyTorch官方pytorch/pytorch上游原碼，固定到實際安裝CPU wheel所回報的commit。",
        "checked_original":True,"inspection_note":note})
for key, path, note in [
    ("data","tiny_perceptron/data.py","親讀10–21和54–68：byte+8、IGNORE=-100、messages角色及一次shift契約。"),
    ("model","tiny_perceptron/model.py","親讀14–28、31–86、92–108：預設single-layer/manual/untied、embedding和logits、loss_sum及masked_loss的有效分母>0要求。"),
    ("attention","tiny_perceptron/attention.py","親讀10–28、48–74：key<=query，valid限制key，手寫scores/softmax/values運算。"),
    ("modern","tiny_perceptron/modern.py","親讀42–61 DenseFFN：本例選擇的FFN是各位置獨立運算，跨位置依賴由attention提供。")]:
    sources.append({"id":key,"kind":"repository_code","title":path,"verified":True,"path":path,"sha256":sha(ROOT/path),
        "version":"git base 022dc9b2ffde92c14ca133406849c1c378bb8e8f + exact current file SHA-256","inspection_note":note+" 已保存親讀版本bytes在inputs/。"})
sources += [
    {"id":"original-run","kind":"execution","title":"Executed unchanged 7.5 Python fence","verified":True,"artifact_id":file_ids["original/stdout.txt"]},
    {"id":"controls-run","kind":"execution","title":"Independent bounded CPU controls","verified":True,"artifact_id":file_ids["bounded-controls.stdout.json"]},
    {"id":"derivation","kind":"derivation","title":"CE mask, chain rule and norm derivation","verified":True,"details":"inspection.md：M_t/N*(softmax-onehot)，N=2；ignored logits derivative=0；E89梯度取決於後續attention Jacobian；264維與8維梯度範數不同座標，不作同單位比較。float64手工代入與assert結果保存在bounded-controls.stdout.json。"},
]
def ev(source_id, locator, supports): return {"source_id":source_id,"locator":locator,"supports":supports}
def verification(expected, observed, details, numeric=False):
    out={"method":"executed","expected":expected,"observed":observed,"details":details}
    if numeric: out["tolerance"]="離散ID／index／0梯度精確相等；norm手工檢查絕對差<1e-8；float64 CE/解析梯度絕對差<1e-14。"
    return out
claims = [
    {"id":"c1","kind":"concept","statement":"問題Q不算直接答案代價，模型仍需要讀它回答A；問題格候選分數可梯度0，Q輸入特徵從後面答案收到梯度，兩者針對不同變量。",
     "location":"course/chapters/07.md:148,169","scope":"本例有限logits、class-index ignore mask與未凍結的預設untied單層因果模型；不宣稱所有參數狀態均有非零embedding梯度。","status":"verified",
     "evidence":[ev("ce","class-index loss公式／ignore_index參數","忽略的是各位置logits直接代價及梯度。"),ev("paper","PDF pp.3–4 §3.1、§3.2.1 Eq.(1)","允許的早先key/value可影響後續位置；因果mask只禁未來。"),ev("autograd","How autograd encodes the history","損失沿實際依賴圖按鏈式法則回傳到輸入特徵。"),ev("attention","attention.py:10–28,48–74","repo實際保留Q為可讀key/value，loss labels不傳入attention。")],"artifact_ids":orig_ids+control_ids},
    {"id":"c2","kind":"numeric","statement":"最短Q/A對話的Q在X索引2、ID89；embedding.weight第89列是Q字向量。",
     "location":"course/chapters/07.md:150","scope":"本課ByteTokenizer，Q單ASCII byte81加8；不是通用tokenizer ID。","status":"verified",
     "evidence":[ev("data","data.py:10–21,54–68","直接讀byte編碼、角色邊界、shift。"),ev("embedding","nn.Embedding Variables/Shape","input索引對應weight列。"),ev("controls-run","checks.baseline_Q x/y/question_id/embedding_shape","實際Q位置2和row89。")],"artifact_ids":control_ids,
     "verification":verification("Q ID=81+8=89；X=[1,3,89,2,4,73]；表shape=[264,8]。","CPU輸出逐项吻合。","沒有按肉眼字元數外推其他tokenizer。",True)},
    {"id":"c3","kind":"software","statement":"原程式manual_seed(42)、TinyLM(width=8)、retain_grad、masked_loss.backward後，第一位置logits梯度應0，Q字表列梯度通常大於0。",
     "location":"course/chapters/07.md:153–167","scope":"原fence完整未修改實跑；CPU torch2.14.1+cpu；候選logits shape[1,6,264]、Q梯度8維。","status":"verified",
     "evidence":[ev("original-run","original/stdout.txt and execution.json","原fence實際輸出符合敘述。"),ev("model","model.py:14–28,53–86,92–108","設定、forward、masked_loss實際契約。"),ev("seed","manual_seed API","固定seed設定。"),ev("backward","Tensor.backward API","scalar loss回傳圖葉梯度。"),ev("retain","Tensor.retain_grad API","明確保留中間logits.grad。"),ev("item","Tensor.item API","norm單值讀成Python number。"),ev("tensor-norm","Tensor.norm signature","預設norm參數。")],"artifact_ids":orig_ids+control_ids,
     "verification":verification("首位置logits norm=0；Q row norm>0；輸出可取得。","原stdout 0.0與0.00571822514757514。","同時查所有前4個ignored位置logits，均精確0；不是僅看整表grad non-None。")},
    {"id":"c4","kind":"concept","statement":"logits為中間結果，PyTorch一般不保留中間節點.grad；retain_grad明確要求保留它，與葉參數不同。",
     "location":"course/chapters/07.md:167","scope":"通常grad-mode下requires_grad的nonleaf logits；retain_grad不是打開一條本来不存在的梯度路徑。","status":"verified",
     "evidence":[ev("leaf","Tensor.is_leaf正文123–135","只有葉子的grad預設填值，nonleaf可retain。"),ev("retain","API123–126","retain_grad保留中間grad。"),ev("controls-run","checks.without_retain_grad／baseline_Q","logits.is_leaf=False，embedding.is_leaf=True；不retain時logits.grad=None但Q仍有梯度。")],"artifact_ids":control_ids},
    {"id":"c5","kind":"numeric","statement":"norm將各格平方加總開根；本節只看是否有影響，不把logits與embedding兩種梯度大小當同一單位比較。",
     "location":"course/chapters/07.md:167,169","scope":"本節兩個一維實數向量：264候選分數座標與8輸入特徵座標；不外推任意矩陣norm或影響大小排名。","status":"verified",
     "evidence":[ev("norm","torch.norm Parameters p/dim","一維default fro等於p=2的sum squares sqrt。"),ev("installed-tensor","_tensor.py:888–902","實際安裝Tensor.norm dispatch到torch.norm。"),ev("derivation","inspection.md Repository contract and independent calculations","不同變量座標的導數不應直接以範數大小比因果能力。"),ev("controls-run","checks.baseline_Q.norm_manual_sqrt_sum_squares_float64","親算sqrt sum squares並比較。")],"artifact_ids":control_ids,
     "verification":verification("Q的8維grad norm等於sqrt(sum平方)，容差1e-8。","手算0.005718225021817441，Tensor.norm 0.00571822514757514；差1.26e-10。","另外logits.grad[0,0]是264維全0，norm精確0。",True)},
    {"id":"c6","kind":"concept","statement":"只檢查整個embedding層是否None，無法知道Q那列是否參與；只看到第一格0不能推斷所有user內容被凍結。",
     "location":"course/chapters/07.md:169","scope":"可學lookup table各列累積自己的依賴梯度；table non-None不等於每列均非零。","status":"verified",
     "evidence":[ev("embedding","nn.Embedding Variables/Shape","每個input ID對應特定可學weight列。"),ev("autograd","Setting requires_grad","凍結是requires_grad的獨立設定，ignore target不改參數旗標。"),ev("controls-run","checks.replace_Q_with_Z／block_Q_key_in_the_single_attention_layer","整表有梯度，但未使用Q row與blocked Q row為0；baseline Q row為正。")],"artifact_ids":control_ids},
    {"id":"c7","kind":"concept","statement":"共享輸入表可因user內容被後續讀取而調整；忽略user loss沒有凍結它；detach輸入或禁止回答讀Q會切掉另一條影響路線。",
     "location":"course/chapters/07.md:171","scope":"更新需之後optimizer step，本段沒有；detach只切輸入表路徑，不刪forward特徵。Q key阻斷在所有本例跨位置通路上施加，單層untied模型。","status":"verified",
     "evidence":[ev("autograd","history／Setting requires_grad","沿依賴梯度與凍結參數不同操作。"),ev("detach","Tensor.detach API123–140","從當前圖分離使embedding路徑不回傳。"),ev("attention","attention.py:10–28,48–74","valid禁Q key；手寫權重值加權和。"),ev("modern","DenseFFN42–61","位置獨立FFN沒有額外跨位置路線。"),ev("controls-run","checks.detach_embedding_output／block_Q_key_in_the_single_attention_layer","detach數值完全相同但embedding.grad=None；禁Q key使Q row精確0。")],"artifact_ids":control_ids},
    {"id":"c8","kind":"software","statement":"這段只求梯度，没有step；核對資料依賴，尚未調整出回答能力。",
     "location":"course/chapters/07.md:171","scope":"整段原fence沒有optimizer、step或評測；不把隨機模型反向微分當作SFT訓練成果。","status":"verified",
     "evidence":[ev("backward","Tensor.backward API123–138","backward計算/累積梯度。"),ev("installed-tensor","_tensor.py:566–630","同安裝commit的原始backward只dispatch至autograd。"),ev("controls-run","parameters_unchanged與checks.limits","所有參數前後精確相等，optimizer steps=0，training_runs=0。")],"artifact_ids":orig_ids+control_ids,
     "verification":verification("只有grad改變，所有parameter values保持精確相等。","baseline、Z、blocked、detached、no-retain所有parameters_unchanged=true。","未執行任何長訓練配方、GPU、既有權重評測或模型/資料下載。")},
    {"id":"c9","kind":"software","statement":"練習只把user Q改成Z，保持A與seed；第一格logits仍0，Z字表列通常有梯度，X索引2自動查新列。",
     "location":"course/chapters/07.md:173","scope":"同格式同seed，只更換一個ASCII問題byte，ID98；不是Z語意回答能力。","status":"verified",
     "evidence":[ev("data","ByteTokenizer.encode/render_chat","新byte仍在同一位置。"),ev("controls-run","checks.replace_Q_with_Z","Z input ID98在index2，first logits 0，新row norm>0。")],"artifact_ids":control_ids,
     "verification":verification("首格logits=0；X[2]=98；Z row norm>0。","首格=0；Z norm=0.004994203336536884，沒有再讀Q row，舊Q row=0。","與原Q分別固定seed42及答案A，保持同模型配置。")},
    {"id":"c10","kind":"software","statement":"masked_loss(logits,y[None])忽略-100目標位置，但與模型讀取許可不同；logits候選軸及有效答案分母必須正確。",
     "location":"course/chapters/07.md:158–162,167；本fence所調helper的實際契約","scope":"有限無權重無smoothing class-index CE；B=1,T=6,C=264，N=2>0；整批ignored必須拋錯。因果mask[batch,head,query,key]不讀labels。","status":"verified",
     "evidence":[ev("ce","class-index公式、ignore_index、Shape","按位置忽略，不把-100當候選類；分母只算有效答案。"),ev("installed-functional","functional.py:3478–3571,6366–6409,6480–6508","loss形狀／ignore語義與attention讀取mask最後兩軸。"),ev("model","loss_sum92–108","reshape(-1,last C)，count有效labels，reject0。"),ev("paper","§3.1 decoder／Eq.(1)","後面的first-answer預測只可依賴已知早先輸入。"),ev("controls-run","checks.CE_axes_denominator_and_derivative／causal_axes","親跑有效分母2、解析梯度、任意ignored logits不改loss、全ignored錯誤與因果變化。")],"artifact_ids":control_ids,
     "verification":verification("loss=sum CE/2；ignored derivative0；query4可看key2但禁key5；全ignored ValueError。","float64手算5.618060945615395 vs實際5.618060945615394；梯度最大差2.60e-18；future A->B對query4差0。","完整命令、原碼、CPU環境、stdout及hash均永久保存；未用玩具測試代替通用來源。")},
]
report={"schema_version":1,"review_stage":"technical","lesson_id":"7.5","source":"course/chapters/07.md#7.5",
    "source_sha256":extraction["source_sha256"],"figure_sha256":{},"reviewer_task":"/root/phase4_factual_coordinator/factual_7_5",
    "reviewer_context":"fresh","author_tasks":[],"verdict":"pass","reviewed_on":"2026-10-05",
    "read_scope":"7.5全文；第7章導言及7.1–7.4前置格式/對齊；7.6–7.7分母/attention上下文；列出的repo和原始authority內容親讀。未讀舊技術/reader報告或他人判定。",
    "chapter_intro":{"status":"not_applicable","reason":"7.5非本章首節；導言因章檔前置讀取看到，未宣稱完成首節intro特有檢查。"},
    "applicability":{"substantive_claims":True,"empirical_measurements":False,"reason":"本節是隨機模型梯度展示，沒有模型成績/歷史raw measurement，數值直接原fence和有界控制核算；無長recipe；自身無引用圖。"},
    "artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","claim_ids":[c["id"] for c in claims],"details":"10項實質主張逐項核原論文/版本化官方文件/精確安裝commit原碼/本repo契約及必要有界控制；ignore與visibility、非leaf梯度和更新分開，未見實質未知或錯誤。"},
        "numeric_verification":{"status":"pass","claim_ids":["c2","c3","c5","c9","c10"],"details":"Q=89、Z=98、position2；原fence首logits梯度0、Q norm0.005718225；Z norm0.004994203；norm手算差1.26e-10<1e-8；float64 CE差8.88e-16、解析梯度差2.60e-18<1e-14；有效分母2。"},
        "figure_consistency":{"status":"not_applicable","claim_ids":["c2","c10"],"details":"7.5無任何圖引用，extraction svg_references=[]、figure_sha256={}。額外親讀前節7.4圖，Inkscape實render+view確認Q index2/ID89、assistant index4→A73、A index5→EOS2，context-figure和receipt留存；沒有冒稱7.5自帶圖或完整HTML/mobile驗證。"},
        "source_verification":{"status":"pass","claim_ids":[c["id"] for c in claims],"details":"親取/親讀arXiv v7 PDF封面與§3、PyTorch2.9版本化原始文件以及2.14.1+cpu對應git commit的functional/_tensor原碼；每source記HTTPS URL、版本、日期、authority reason、inspection與每claim locator/supports；原始bytes和hash留docs永久證據。"},
        "limitations":{"status":"pass","claim_ids":["c1","c3","c7","c8","c9","c10"],"details":"例子只有求梯度，沒有optimizer step或accuracy；未綁定輸出且僅單layer的控制不能證明任意配置都非零；原文『通常』而非保証。detach保持forward輸入，只切圖；loss要求有效目標>0。沒有重訓、既有權重評測、資料模型下載、GPU或付費。"},
    },
    "required_artifacts":[artifact["path"] for artifact in artifacts],
    "judgment_note":"本輪全新獨立判讀，未沿用舊結果。正式checker只驗schema/版本，不能代替本人的內容核實；未修改教材、圖或commit。"
}
(ROOT/"docs/technical-reviews/7.5.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"source_sha256":report["source_sha256"],"report_sha256":sha(ROOT/"docs/technical-reviews/7.5.json"),"claims":len(claims),"artifacts":len(artifacts),"verdict":report["verdict"]},ensure_ascii=False))
