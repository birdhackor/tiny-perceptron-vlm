"""Write this review from independently inspected evidence, never old reports."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
relative = HERE.relative_to(REPO).as_posix()
extraction = json.loads((HERE / "extraction.json").read_text())
bounded = json.loads((HERE / "bounded-results.json").read_text())
original_run = json.loads((HERE / "execution.json").read_text())
bounded_run = json.loads((HERE / "bounded-execution.json").read_text())
fetches = {item["name"]:item for item in json.loads((HERE / "fetch-receipt.json").read_text())}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

artifacts = []
def artifact(identifier, name, kind, description, command=None, result=None, environment=None):
    value = {"id":identifier,"path":f"{relative}/{name}","sha256":digest(HERE/name),
             "kind":kind,"description":description}
    if kind == "execution":
        value.update(command=command,result=result,environment=environment)
    artifacts.append(value)

original_env = {"python":"3.13.5", "torch":"2.14.1+cpu", "device":"CPU",
                "cuda":"None / unavailable", "numeric_type":"Python binary64 float"}
bounded_env = {"python":"3.13.5", "device":"CPU", "numeric_type":"binary64 float; Decimal precision 40"}
artifact("section","section.md","source_snapshot","原始 UTF-8 1.9 全節 bytes，未正規化換行")
artifact("extraction","extraction.json","source_snapshot","原始小節、fence、helper/BOOTSTRAP hash；無圖引用")
artifact("original-code","fence-1.py","code","教材唯一完整 Python fence 原碼")
artifact("bootstrap","bootstrap.py","code","實际使用的本機準備 cell；Colab 下載分支未執行")
artifact("original-execution","execution.json","execution","正式 helper 原碼 CPU 執行記錄及 stdout/env hash",
         original_run["command"],"exit_code=0；已執行 fence 1，無 guard_events；輸出梯度 -3.9999999999995595",original_env)
artifact("original-stdout","stdout.txt","source_snapshot","原碼真實 stdout：左右代價、有限差分與公式")
artifact("original-stderr","stderr.txt","source_snapshot","原碼真實 stderr，空檔")
artifact("original-env","environment.json","source_snapshot","實際 Python、torch CPU、裝置、離線限制與 attempted_fences")
artifact("bounded-code","bounded_checks.py","code","獨立 Decimal 算表、近似誤差與原碼三種 w 變化；附數值限制小例")
artifact("bounded-runner","run_bounded.py","code","保留子程序真實 exit status 的有界 CPU 執行器")
artifact("bounded-execution","bounded-execution.json","execution","實際有界命令、裝置、exit status、stdout/stderr 和結果 hash",
         bounded_run["command"],"exit_code=0；全部有意義斷言完成；passed=true",bounded_env)
artifact("bounded-results","bounded-results.json","execution","獨立原碼/練習和算例的完整實測結果",
         bounded_run["command"],"w=1/5/3 梯度約 -4/+4/0 且 w 不變；表格精確；ΔL=-.003999；小 h 出現消去",bounded_env)
artifact("bounded-stdout","bounded-stdout.txt","source_snapshot","有界檢查實際 stdout")
artifact("bounded-stderr","bounded-stderr.txt","source_snapshot","有界檢查實際 stderr，空檔")
for value in ("1.0","5.0","3.0"):
    artifact(f"exercise-{value}",f"exercise-w-{value}.py","code",f"实际执行的 w={value} 代码；除 w 初始化外原碼未變")
artifact("derivation","inspection-and-derivation.md","derivation","親讀範圍、逐主張支持范围、公式推導、軸/單位/分母/tolerance 與限制")
artifact("fetch-code","fetch_sources.py","code","官方原始來源 fetch 與可讀文字轉換實作")
artifact("fetch-receipt","fetch-receipt.json","source_snapshot","五個官方 HTTPS 來源 HTTP200、URL、access date、SHA-256")
artifact("fetch-stdout","fetch-stdout.txt","source_snapshot","官方原始來源取得 stdout")
artifact("fetch-stderr","fetch-stderr.txt","source_snapshot","官方原始來源取得 stderr，空檔")
artifact("report-code","build_report.py","code","本次新報告生成碼；不讀任何舊報告")
for name in fetches:
    artifact(f"{name}-raw",f"{name}.html","source_snapshot",f"{name} 官方原始 HTML；支持范围見相應 source.inspection_note")
    artifact(f"{name}-text",f"{name}.txt","source_snapshot",f"{name} 原始 HTML 的可讀文字；用於親讀與定位")

def official(identifier,name,title,version,authority,note):
    return {"id":identifier,"kind":"official_docs","title":title,"url":fetches[name]["url"],
            "version":version,"accessed_on":"2026-10-05","authority_reason":authority,
            "verified":True,"checked_original":True,"inspection_note":note,
            "artifact_ids":[f"{name}-raw",f"{name}-text"]}

sources = [
    official("derivative","openstax-derivative","OpenStax Calculus Volume 1 §3.1 Defining the Derivative",
             "Gilbert Strang / Edwin Herman; publication 2016-03-30; official web content ©2026-07-15",
             "教材作者與 Rice University/OpenStax 官方出版網站的完整原始數學教材",
             "親讀 §3.1 導數定義 equations (3.5),(3.6)、Example 3.4（text lines704–739）及瞬時變化率定義1314–1316；僅支持導數極限及敏感度，具體數字另自算。"),
    official("linear","openstax-linear-approximation","OpenStax Calculus Volume 1 §4.2 Linear Approximations and Differentials",
             "Gilbert Strang / Edwin Herman; publication 2016-03-30; official web content ©2026-07-15",
             "OpenStax 官方出版的原始教科書數學定義",
             "親讀 text24–46 近/遠切線範例、equation4.1及779–785 differential equation4.2；支持 ΔL≈L′Δw 的局部近似及遠處不保證，未把近似當精確。"),
    official("nist","nist-finite-difference","NIST DLMF §3.4 Differentiation",
             "DLMF 1.2.8, release 2026-09-15",
             "美國 NIST 官方數學參考原文，提供数值微分公式与光滑性/remainder 條件",
             "親讀 §3.4(i) equations3.4.1–6（text25–161）與footer版本；3.4.6设t=0得中心差分分母2h，3.4.3–4为步長/remainder；本平方的remainder=0由自己代入，非概括所有函式。"),
    official("autograd","pytorch-autograd","PyTorch A Gentle Introduction to torch.autograd",
             "official tutorial last updated 2025-10-01, last verified 2024-11-05; accessed 2026-10-05; installed torch 2.14.1+cpu",
             "PyTorch 維護團隊官方 autograd 教程與原始公式/API說明",
             "親讀 Background及.grad/optimizer step text323–373、partial derivative375–406、scalar-loss gradient vector/chain rule426–464。支持梯度為各參數偏導及標準NN沿運算關係反傳；未執行官方pretrained resnet下载範例，1.9本身也沒autograd。"),
    official("floating","python-floatingpoint","Python Tutorial §15 Floating-Point Arithmetic: Issues and Limitations",
             "Python docs 3.13.16, updated 2026-10-01; execution interpreter 3.13.5",
             "Python Software Foundation 官方原始語言/執行數值限制說明",
             "親讀 text34–149 binary representation、顯示/實值與每次運算rounding，15.1 text202–212 binary64/53-bit；支持浮點誤差存在。除以2h的誤差放大是独立推導；不假裝该頁提供有限差分截斷定理。"),
    {"id":"algebra","kind":"derivation","title":"1.9 獨立平方代價與差商推導","verified":True,
     "details":"见 inspection-and-derivation.md：L(1+d)-4=-4d+d²；中心差分=2(w-3)；局部ΔL=-.003999 vs-.004；誤差(ε+−ε−)/(2h)；軸w、L任意loss單位、分母2h。"},
    {"id":"original-code","kind":"repository_code","title":"1.9 原始 Python fence 精確快照","verified":True,
     "path":f"{relative}/fence-1.py","sha256":digest(HERE/"fence-1.py"),
     "version":f"current section SHA-256 {extraction['source_sha256']}; fence SHA-256 {digest(HERE/'fence-1.py')}",
     "inspection_note":"逐行親讀 course/chapters/01.md:325–335 与原始bytes fence；只做 Python scalar loss/左右evaluate/quotient/print；无torch tensor、backward或參數更新。"},
    {"id":"original-run","kind":"execution","title":"原碼正式 CPU helper 執行","verified":True,"artifact_id":"original-execution"},
    {"id":"bounded-run","kind":"execution","title":"有界原碼練習與獨立數值驗證","verified":True,"artifact_id":"bounded-results"},
]

def evidence(source,locator,supports):
    return {"source_id":source,"locator":locator,"supports":supports}
def verification(expected,observed,details,tolerance=None):
    item={"method":"executed","expected":expected,"observed":observed,"details":details}
    if tolerance is not None:
        item["tolerance"]=tolerance
    return item

claims = [
    {"id":"quadratic-table","kind":"numeric","status":"verified",
     "statement":"L=(w−3)² 在 w=3 最小；w=1 的 L=4，表格1.1/1.01的L、改變及差商3.61/3.9601、−.39/−.0399、−3.9/−3.99正確；代價大小單獨不提供參數方向。",
     "location":"course/chapters/01.md:312–320","scope":"此一任意單位、單參數平方算例。w=1與5均L=4但導數符號相反，不把平方loss當上節交叉熵。",
     "evidence":[evidence("algebra","inspection-and-derivation.md Independent algebra and axes","平方非負與唯一零點；前向差商−4+d的逐項代入"),evidence("bounded-run","bounded-results.json checks.exact_table/exact_centered_difference","Decimal核對表格與w=1/5反向敏感度")],
     "artifact_ids":["derivation","bounded-results"],
     "verification":verification("精確表格與差商；L(1)=L(5)=4但導數−4/+4","Decimal表格全部精確相等；原碼w=1/5梯度约−4/+4","Decimal precision40；一個scalar，无資料平均。","表格Decimal精確相等，gradient binary64 absolute <=2e−12")},
    {"id":"derivative-gradient","kind":"concept","status":"verified",
     "statement":"目前位置的代價變化/參數變化極限是導數；它表示局部敏感度。標量代價對各參數的偏導按參數位置組成梯度。",
     "location":"course/chapters/01.md:320","scope":"可微loss；多參數時每個分量是對該參數的偏導，其他參數保持不變；本節一維梯度就是導數。",
     "evidence":[evidence("derivative","§3.1 Definition, equations3.5–3.6; Instantaneous rate of change definition","導數是變化率差商的極限且為瞬時敏感度"),evidence("autograd","Differentiation in Autograd partial Q/partial a,b; scalar l gradient vector in More on vector-Jacobian product","模型參數逐項偏導與scalar loss梯度向量")],
     "artifact_ids":["openstax-derivative-raw","pytorch-autograd-raw","derivation"]},
    {"id":"centered-method","kind":"concept","status":"verified",
     "statement":"以w+h與w−h的loss差除以2h是中心有限差分，用來估計導數。",
     "location":"course/chapters/01.md:322–333","scope":"h非零，中心有限差分一般含截斷誤差；此二次函式在精確算術中恰好等於解析導數，实际float仍會有roundoff。",
     "evidence":[evidence("nist","§3.4(i) Three-Point Formula equation3.4.6, t=0","消掉中間f0項後得(f1−f−1)/(2h)，並含remainder"),evidence("algebra","inspection-and-derivation.md centered-difference identity","實際平方的h²項相消，得到2(w−3)")],
     "artifact_ids":["nist-finite-difference-raw","derivation"]},
    {"id":"centered-values","kind":"numeric","status":"verified",
     "statement":"w=1,h=.001時L(.999)=4.004001、L(1.001)=3.996001；左右差−.008跨参数距离.002，导数−4；解析公式是2(w−3)。",
     "location":"course/chapters/01.md:325–338","scope":"单参数轴w；导数单位為loss單位/w單位。正确分母2h而非h。",
     "evidence":[evidence("algebra","inspection-and-derivation.md arbitrary-w centered identity","公式及數值獨立代入"),evidence("original-run","stdout.txt two output lines; execution.json exit_code0","完整原始fence實際左右值/商与解析值"),evidence("bounded-run","bounded-results.json exact_centered_difference[0], actual_code_runs[0]","Decimal与实码两种独立核對")],
     "artifact_ids":["original-execution","original-stdout","bounded-results","derivation"],
     "verification":verification("4.004001/3.996001；ΔL=−.008，Δw=.002；梯度−4","stdout左4.004001，右3.9960010000000006，商−3.9999999999995595；Decimal精確−4","打印未round，教材数字是正确显示精度；centred不做训练。","Decimal精確相等；binary64 gradient abs error<=2e−12，loss值abs error<=2e−15")},
    {"id":"local-direction","kind":"concept","status":"verified",
     "statement":"負導數表示當前稍增w使loss下降；導數不是下一個參數值，局部方向不保證遠跳改善。",
     "location":"course/chapters/01.md:340","scope":"此可微平方loss的w=1鄰域；由局部线性化理解方向，沒有聲稱任意步長下降。",
     "evidence":[evidence("linear","§4.2 Linear Approximation equation4.1 and close/far example","f(a+d)≈f(a)+f′(a)d只支持near-a的解讀"),evidence("algebra","inspection-and-derivation.md local expansion and distant-step example","負號含義与1→6時loss4→9反例")],
     "artifact_ids":["openstax-linear-approximation-raw","derivation","bounded-results"]},
    {"id":"linear-change","kind":"numeric","status":"verified",
     "statement":"w=1附近w增加.001，導數−4預測loss約減少.004，属于一階近似。",
     "location":"course/chapters/01.md:340","scope":"原文明确說約；實際單側变化−.003999，不將左右總差−.008混入此增量。",
     "evidence":[evidence("linear","§4.2 Differentials equation4.2:dy=f′(x)dx","敏感度乘输入微量是局部loss变化近似"),evidence("bounded-run","bounded-results.json checks.local_linear_approximation","精确Decimal delta和近似误差独立算出")],
     "artifact_ids":["bounded-results","derivation","openstax-linear-approximation-raw"],
     "verification":verification("一阶ΔL=−.004；实际ΔL=−.003999","Decimal actual −.003999、linear−.004、abs_error=.000001、relative_error≈.0002500625","分母为单侧Δw=.001；误差是二次项d²。","Decimal恒等式精确相等；文章‘約’的一階近似誤差1e−6 loss units（约.02500625%），不是零誤差")},
    {"id":"program-exercises","kind":"software","status":"verified",
     "statement":"實際程式定義平方loss，左右兩次評估後用2h計差商並print；改w=5得到+4且左邊较近3，改w=3两侧loss约1e−6且gradient0；所有执行都只测敏感度，w未更新。",
     "location":"course/chapters/01.md:325–335,342","scope":"一個原始fence与仅替换w初始化的两種练习；无autograd/backward/optimizer、沒有文字模型训练。",
     "evidence":[evidence("original-code","fence-1.py lines1–12, matching course lines325–335","def loss、**2、函数调用、减法、/ (2*h)、print的完整实际契约；赋值gradient不修改w"),evidence("bounded-run","bounded-results.json checks.actual_code_runs entries1/2; exercise-w-5.0.py,exercise-w-3.0.py","逐项验证左右loss、梯度、解析formula、stdout和w_unchanged")],
     "artifact_ids":["original-code","original-execution","bounded-code","bounded-results","exercise-5.0","exercise-3.0"],
     "verification":verification("w5左/右3.996001/4.004001、gradient4；w3左右1e−6、gradient0；w不动","w5 gradient4.000000000001336；w3左右9.999999999997797e−7、gradient0；三次w_unchanged=true","运行原码与兩个仅初始化改动，不用新实现代替原码；每个候选算例是一个scalar，30秒上限内完成。")},
    {"id":"finite-precision-scope","kind":"concept","status":"verified",
     "statement":"一般有限差分h過大可能不足以表示局部敏感度；h太小時近似loss相減与除以小間距能放大數值誤差。",
     "location":"course/chapters/01.md:347","scope":"一般数值微分警语；平方例的中心截断误差在精确算术中为0，並非自身展示大h坏结果。float roundoff和一般函式截断分開核对。",
     "evidence":[evidence("nist","§3.4(i) equations3.4.3–4 remainder; equation3.4.6 centered rule","光滑函数有限差分存在依h变化的remainder；不等於任何步长均精確"),evidence("floating","§15 Floating-Point Arithmetic per-operation rounding; §15.1 binary64/53-bit precision","浮點近似与每次運算可能rounding為誤差來源"),evidence("algebra","inspection-and-derivation.md error (ε+−ε−)/(2h)","除以小2h能放大输出数值误差，不能机械令h无限小")],
     "artifact_ids":["nist-finite-difference-raw","python-floatingpoint-raw","derivation","bounded-results"]},
    {"id":"neural-network-method","kind":"concept","status":"verified",
     "statement":"標準大型神經網路求梯度沿運算關係和鏈式法則计算，而不是逐參數試左右兩次的中心有限差分。",
     "location":"course/chapters/01.md:347","scope":"此课程后续可微NN训练方法的比較；不否认有限差分可用于gradient-checking或其他黑盒方法；本scalar实码不自动求导。",
     "evidence":[evidence("autograd","Background; Computational Graph text448–464; scalar-loss vector-Jacobian chain-rule formula","官方NN訓練引擎遍历操作圖，计算/累加参数梯度；由原始方法文件支持，未拿toy检查代替成熟方法证据")],
     "artifact_ids":["pytorch-autograd-raw","derivation"]},
]

report={
    "schema_version":1,"review_stage":"technical","lesson_id":"1.9","source":"course/chapters/01.md#1.9",
    "source_sha256":extraction["source_sha256"],"figure_sha256":{},"verdict":"pass",
    "reviewer_task":"/root/phase4_factual_coordinator/factual_1_9","reviewer_context":"fresh",
    "reviewed_on":"2026-10-05","author_tasks":[],
    "read_scope":"当前1.9全部310–350，1.8上下文274–309，incidental当前270–273/351–355/520–586不纳入判定；完整review方法/checker/helper与build_course.py1–125；亲读五个官方原始文档相关段落，詳見永久inspection。未讀旧1.9或其他历史报告正文/结论。",
    "artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
       "factual_accuracy":{"status":"pass","details":"九個合組實質claims覆蓋平方算例、導數/梯度、中心差分、local sign、近似、实际code/练习、数值限制与标准NN方法。每项记各source支持範圍，无未知实质claim。","claim_ids":[c["id"] for c in claims]},
       "numeric_verification":{"status":"pass","details":"Decimal精确表格/分母2h/导數式；真实binary64三次fence，绝对梯度容忍2e−12；Δw=.001一阶ΔL−.004 vs精确−.003999，误差1e−6；无资料/token平均分母。","claim_ids":["quadratic-table","centered-values","linear-change","program-exercises"]},
       "figure_consistency":{"status":"not_applicable","details":"当前全節无圖、SVG或HTML image引用；原始extraction svg_references=[]，不需要render。","claim_ids":[]},
       "source_verification":{"status":"pass","details":"親讀官方OpenStax/NIST/PyTorch/Python原文并保留五个HTTP200原始HTML、readable、版本、URL/date/hash与定位；不沿用旧report判断；數字/代码由独立推導与真实CPU执行支持。","claim_ids":[c["id"] for c in claims]},
       "limitations":{"status":"pass","details":"单参数任意loss单位，无交叉熵/训练/模型成绩；derivative是局部率不是新w；平方中心差分exact而一般h仍有remainder/roundoff；浮点显示与exact区分；标准autograd由官方文档支持，本节没调用；无完整训练/GPU/下载数据模型/上传。","claim_ids":["derivative-gradient","centered-method","local-direction","linear-change","program-exercises","finite-precision-scope","neural-network-method"]}
    }
}
(REPO/"docs/technical-reviews/1.9.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"report":"docs/technical-reviews/1.9.json","verdict":report["verdict"],"claims":len(claims),"sources":len(sources),"artifacts":len(artifacts),"source_sha256":report["source_sha256"]},ensure_ascii=False))
