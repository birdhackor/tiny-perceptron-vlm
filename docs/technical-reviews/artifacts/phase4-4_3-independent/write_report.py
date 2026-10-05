"""Assemble this reviewer's independently checked section 4.3 evidence."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
PREFIX = BASE.relative_to(ROOT).as_posix()
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):
    return json.loads((BASE / name).read_text())
env = read("probe-environment.json")
probe = read("probe-results.json")
original = read("execution.json")
extraction = read("extraction.json")
receipts = read("checks-execution.json")
commit = env["torch_git_version"]
assert (BASE / "section.md").read_bytes() == __import__("re").search(rb"(?ms)^## 4\.3 .*?(?=^## 4\.4 )", (ROOT / "course/chapters/04.md").read_bytes())[0]
artifacts = []
names = {
    "execution.json": ("original-execution", "execution", "原稿兩個 Python fence 的受限 CPU 執行命令、退出碼與 stdout/environment SHA-256；各原碼另永久保留。"),
    "checks-execution.json": ("probe-execution", "execution", "獨立推導與有意義小變化的實際 CPU 命令、退出碼、時間及 stdout/stderr hash；並記 PDF 擷取/渲染命令。"),
    "probe-results.json": ("probe-results", "derivation", "逐排純 Python 計算與 float32 LN 比較、分母、epsilon、平移、單格改動、affine、常數、batch與錯軸反例。"),
    "section.md": ("section-snapshot", "source_snapshot", "本次獨立讀取的 4.3 UTF-8 原始 bytes，沒有換行正規化。"),
    "ba-2016-layer-normalization-v1.pdf": ("paper-pdf", "source_snapshot", "作者在 arXiv 提交的 Layer Normalization v1 原始 PDF。"),
    "ba-2016-layer-normalization-v1.txt": ("paper-text", "source_snapshot", "pdftotext 原論文擷取文字；親讀摘要、§3、§3.1與§5.1。"),
    "paper-page-02.png": ("paper-page2", "source_snapshot", "pdftoppm 實際 render 並以 view_image 親看原論文第2頁 Eq.(3)。"),
    "paper-page-03.png": ("paper-page3", "source_snapshot", "pdftoppm 實際 render 並以 view_image 親看第3頁 Eq.(4)(5)、共享 gain/bias 說明。"),
    "torch-normalization-installed-commit.py": ("official-normalization", "source_snapshot", "與本地 torch git commit 相同的官方 normalization.py；親讀 LayerNorm 105–230 行。"),
    "torch-docs-installed-commit.py": ("official-torch-docs", "source_snapshot", "同 git commit 的官方 API docstring 原碼；親讀 allclose、mean、var 定位。"),
    "torch-functional-installed-commit.py": ("official-functional", "source_snapshot", "同 git commit 的官方 functional.py；親讀 layer_norm 2972–2995 行。"),
    "torch-normalization-installed-local.py": ("installed-normalization", "code", "實際執行環境的 normalization.py 完整原碼；與官方 commit 原碼 SHA-256 相等。"),
    "source-retrieval.json": ("source-retrieval", "source_snapshot", "實際官方下載的 HTTPS URL、access date、status、bytes、SHA-256 收據。"),
    "fence-1.py": ("fence1", "code", "原稿第一段 Python bytes，未修改。"),
    "fence-2.py": ("fence2", "code", "原稿第二段 Python bytes，未修改，按原順序與第一段共用 namespace。"),
    "verify_layernorm.py": ("independent-probe-code", "code", "本審閱者自行寫的有界 CPU 推導核對和變化；先用純 Python 求期望，再與 PyTorch 比。"),
}
for name, (identifier, kind, description) in names.items():
    item = {"id": identifier, "kind": kind, "path": f"{PREFIX}/{name}", "sha256": sha(BASE/name), "description": description}
    if kind == "execution":
        item.update(environment=env)
        if identifier == "original-execution":
            item.update(command=original["command"], result="exit_code=0；兩 fence 均實際執行；平均[0,0]，印出的兩布林為 True/True；guard_events=[]；完整 stdout/stderr/environment 永久保存。")
        else:
            item.update(command=receipts[0]["command"], result="exit_code=0；獨立 scalar 推導與 LN 最大誤差1.1920928955078125e-7；全部數值、單格/全排/affine/batch/constant/wrong-axis checks 通過；PDF extract/render 均退出0。")
    artifacts.append(item)
for name, identifier, kind in [
    ("stdout.txt","original-stdout","source_snapshot"), ("stderr.txt","original-stderr","source_snapshot"),
    ("environment.json","original-environment","source_snapshot"), ("extraction.json","original-extraction","source_snapshot"),
    ("bootstrap.py","original-bootstrap","code"), ("section_facts.py","helper-snapshot","code"),
    ("probe-stdout.txt","probe-stdout","source_snapshot"), ("probe-stderr.txt","probe-stderr","source_snapshot"),
    ("probe-environment.json","probe-environment","source_snapshot"),
    ("pdf-extract-stderr.txt","pdf-extract-stderr","source_snapshot"),
    ("pdf-render-stderr.txt","pdf-render-stderr","source_snapshot"),
    ("fetch_sources.py","fetch-code","code"), ("run_checks.py","execution-runner","code"), ("write_report.py","report-builder","code")]:
    artifacts.append({"id":identifier,"kind":kind,"path":f"{PREFIX}/{name}","sha256":sha(BASE/name),"description":f"本輪4.3實際證據/可重跑原碼：{name}；stdout、環境與限制可追查。"})
paper = {"id":"ba-layernorm","kind":"paper","title":"Layer Normalization — Ba, Kiros, Hinton", "url":"https://arxiv.org/pdf/1607.06450v1", "version":"arXiv:1607.06450v1, 2016-07-21", "accessed_on":"2026-10-05","authority_reason":"提出 Layer Normalization 的作者原論文，作者提交至 arXiv 的 v1 PDF。", "verified":True,"checked_original":True,"inspection_note":"我親讀 PDF 擷取中的摘要、§3 Eq.(3)、§3.1 Eq.(4)、§5.1 Eq.(5)及 Eq.(6)(7)；並親看由原PDF渲染的第2–3頁，確認H分母與逐時步統計/共享gain,bias。原論文不寫本節PyTorch epsilon；epsilon 與預設係數另由官方原碼查證。"}
def official(identifier, title, suffix, note):
    return {"id":identifier,"kind":"official_source","title":title,"url":f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{suffix}", "version":f"git {commit}; installed torch {env['torch']}","accessed_on":"2026-10-05","authority_reason":"PyTorch 官方 pytorch/pytorch 原始碼，固定至實際安裝 wheel 的 torch.version.git_version。", "verified":True,"checked_original":True,"inspection_note":note}
sources = [paper,
    official("pytorch-ln", "PyTorch LayerNorm official implementation and docstring", "torch/nn/modules/normalization.py", "親讀105–230行：公式、last-D axes、biased variance、eps=1e-5、elementwise_affine/bias defaults、ones/zeros初始化、(3,)參數形狀及forward。實際安裝 normalization.py SHA-256 96284b63a2088c7d378716d5114eb5b6a1c072d479116170fcfd284c57e84b4f 與此原碼完全相同。"),
    official("pytorch-api", "PyTorch mean, var, allclose official API docstrings", "torch/_torch_docs.py", "親讀833–865行 allclose 逐元素atol+rtol*abs(other)判準；7194–7247行mean(dim)；12505–12550行var N-correction及unbiased=False對應correction=0。只支持這些实际使用API。"),
    official("pytorch-functional", "PyTorch functional.layer_norm official implementation", "torch/nn/functional.py", "親讀2972–2995行：eps=1e-5，將normalized_shape/weight/bias/eps傳入torch.layer_norm；並非跨位置或跨batch另求統計。"),
    {"id":"repository-model","kind":"repository_code","title":"Actual repository LN configuration", "path":"tiny_perceptron/model.py", "sha256":sha(ROOT/"tiny_perceptron/model.py"), "version":"working tree read on 2026-10-05 (sha256 pinned)","verified":True,"inspection_note":"親讀1–115行，尤其Block 34–35、46–48行及TinyLM 63行：預設norm=layer，以width建立nn.LayerNorm，先LN再給各分支。本節無repo helper fence。"},
    {"id":"independent-arithmetic","kind":"derivation","title":"Independent scalar arithmetic for 3-feature LN", "verified":True,"details":"verify_layernorm.py scalar_row：m=sum(x)/3；v=sum((x-m)^2)/3；z=(x-m)/sqrt(v+1e-5)；var(z)=v/(v+1e-5)。全排+c的m也+c而v不變；y=2z+1的mean=1、var=4var(z)。所有數字由自己代入計算，非教材輸出抄寫。"},
    {"id":"original-run","kind":"execution","title":"Actual section 4.3 original two-fence CPU execution","verified":True,"artifact_id":"original-execution"},
    {"id":"independent-run","kind":"execution","title":"Bounded independent CPU checks and intentional perturbations","verified":True,"artifact_id":"probe-execution"}]
def e(source, locator, supports):
    return {"source_id":source,"locator":locator,"supports":supports}
def claim(identifier, kind, statement, location, scope, evidence, artifact_ids, verification=None):
    item = {"id":identifier,"kind":kind,"statement":statement,"location":location,"scope":scope,"status":"verified","evidence":evidence,"artifact_ids":artifact_ids}
    if verification is not None: item["verification"] = verification
    return item
def v(expected, observed, details, tolerance=None):
    value = {"method":"executed","expected":expected,"observed":observed,"details":details}
    if tolerance: value["tolerance"] = tolerance
    return value
claims = [
    claim("ln-center-scale","concept","LN 先依本排特徵平均中心化，再以sqrt(v+epsilon)整理尺度；十倍正比例輸入可得到近似同結果；epsilon避免零變異數分母為0。", "course/chapters/04.md:90–92,109,128", "本例正比例十倍與eps=1e-5；不宣稱任意數值變化、負倍率或可調係數後仍精確相同。", [e("ba-layernorm","§3 Eq.(3); §3.1 Eq.(4); §5.1 Eq.(7)","支持H格內中心化/標準化與整排正比例縮放的理想化機制；不以論文代替epsilon實作細節。"), e("pytorch-ln","normalization.py:111–121,142,188–229","支持實際sqrt(Var+eps)公式、eps預設1e-5與biased variance。")], ["paper-pdf","paper-page2","official-normalization","probe-results"]),
    claim("ln-axes-independent","concept","輸入[1,2,3]的軸為batch、position、feature；LayerNorm(3)只沿各位置最後3格求統計，不依賴另一位置或同批另一筆。", "course/chapters/04.md:107,111,128", "指定nn.LayerNorm(3)與本repo按width建立的LN；一般LayerNorm若normalized_shape有多維可跨更多軸。", [e("ba-layernorm","§3 Eq.(3)末段; §3.1首段","不同training case各自統計，逐時步統計只取当前時步。"),e("pytorch-ln","normalization.py:114–121,140–145,163–168","整數3就是singleton normalized_shape，只normalize最後一維；官方NLP示例是batch,sequence,embedding。"),e("repository-model","tiny_perceptron/model.py:34–35,46–48,63","確認repo實際使用LayerNorm(config.width)，沿最後特徵寬度。")], ["official-normalization","probe-results"]),
    claim("ln-affine-shared","concept","LN後每格乘可調γ再加β；預設γ=1、β=0，同格係數在各位置共用，最終不要求永遠零平均。", "course/chapters/04.md:109,128", "elementwise_affine=True、bias=True的預設；學習後可非均一，零平均/單位變異數只屬正規化中間量的近似。", [e("ba-layernorm","Abstract; §3.1 Eq.(4)前後; §5.1 Eq.(5)","每hidden feature有gain/bias，統計逐時步而gain/bias跨時步共用。"),e("pytorch-ln","normalization.py:118–127,143–155,205–229","learnable per-element參數shape=(3,)，ones/zeros初始化、forward沿prefix軸共用。")], ["paper-page3","official-normalization","probe-results"]),
    claim("ln-example-numbers","numeric","[1,2,3] mean=2、v=2/3、sd≈0.816；[10,20,30] mean=20、sd≈8.165；兩排輸出≈[-1.225,0,1.225]，mean≈0、var≈1。", "course/chapters/04.md:92,107,128", "本例每排3特徵，分母N=3；sd示意未加epsilon，實際defaultLN denominator加1e-5；輸出是無物理單位的標準化數值。", [e("independent-arithmetic","verify_layernorm.py:scalar_row; probe-results.independent_arithmetic","自行代入得到sd=0.816496580927726、8.16496580927726及eps後期望var0.9999850002249967、0.9999998500000224。"),e("independent-run","probe-results.float32_output,mean,variance_denominator_N,max_manual_error","实际CPU核對輸出和N=3變異數；mean=[0,0]，var=[0.999984860420227,0.9999998807907104]。")], ["probe-execution","probe-results","independent-probe-code"], v("sd四捨五入至0.816與8.165；輸出至3小數為±1.225；mean為0，var依v/(v+eps)接近1。", "sd與四捨五入吻合；LN float32輸出±1.2247356176、±1.2247447968；mean兩排0；var如上；manual max error=1.1920928955078125e-7。", "純Python算術的分母是3特徵；PyTorch var(unbiased=False)與correction=0精確相同。epsilon使十倍兩排差9.179115295410156e-6，所以教材寫近似正確。", "原稿3位小數四捨五入誤差≤0.0005；float32輸出對独立算術atol=1e-6,rtol=0；var期望atol=2e-7,rtol=0。")),
    claim("ln-affine-numbers","numeric","若全設γ=2、β=1，本例LN輸出平均1，變異數接近4。", "course/chapters/04.md:109", "本例同排全部feature使用相同γ/β；一般非均一可調參數不套用這個均值與變異數結論。", [e("independent-arithmetic","y=2z+1; Var(y)=4Var(z)","自行推導shift只改mean，uniform倍數使variance乘平方；epsilon使variance略小於4。"),e("independent-run","probe-results.gamma_2_beta_1_mean/variance; verify_layernorm.py affine block","實際將參數fill_(2)/fill_(1)後與2*y+1比較。")], ["probe-execution","probe-results","independent-probe-code"], v("mean=[1,1]，var=4v/(v+eps)≈[3.9999400009,3.9999994000]。", "mean=[1.0,1.0]，var=[3.999939441680908,3.999999523162842]。", "在torch.no_grad中設定affine參數；只做forward核對，沒有模型訓練。", "affine輸出/mean與公式atol=1e-6,rtol=0；Var=4*原Var比較atol=1e-6,rtol=0；教材近4的最大差6.06e-5。")),
    claim("ln-original-api","software","第一fence依原碼建立nn.LayerNorm(3)，mean(dim=-1)、var(dim=-1,unbiased=False)，allclose對零平均用atol=1e-6，完成forward與斷言。", "course/chapters/04.md:94–107,128", "核對這組實際使用API，無loss/backward/optimizer.step；allclose的一般判準也包含rtol，本例other=0時相對項為0。", [e("pytorch-api","_torch_docs.py:833–865,7194–7247,12505–12538","逐項支持allclose絕對/相對容忍差、mean指定維度、var N-correction與unbiased=False=correction0。"),e("pytorch-ln","normalization.py:140–155,188–229","建立的LN保留形狀[1,2,3]，初始perfeature可調參數。"),e("pytorch-functional","functional.py:2972–2995","真實nn.LayerNorm.forward經F.layer_norm傳入相同shape/affine/eps。"),e("original-run","execution.json attempted_fences; stdout.txt","原fence實際執行，均值斷言通過、var輸出近1。")], ["original-execution","original-stdout","original-environment","fence1","official-torch-docs"], v("兩排LN值、mean[0,0]、var近[1,1]，零均值斷言成功。", "原碼退出0，stdout印值±1.2247、mean[0,0]、var列印[1.0000,1.0000]；独立probe用較高精度存实际var。", "CPU torch2.14.1+cpu／Python3.13.5；官方來源固定至實際wheel git commit，安裝normalization source與official SHA完全相同。")),
    claim("ln-shift-and-indexing","software","第二fence將x[0,1]整排加100，兩個allclose印True；重設原輸入只改x[0,1,0]，印True/False。", "course/chapters/04.md:111–121,128", "操作位置1的整排或第0格；不累積前一次shift；整排浮點平移在本例保持相同，此非對任何極大shift的浮點精度保證。", [e("pytorch-ln","normalization.py:111–121,140–145","最後feature軸內中心化，別位置獨立；全排常數被平均一起扣除。"),e("pytorch-api","_torch_docs.py:833–865","支持第二fence用預設rtol=1e-5/atol=1e-8比較。"),e("original-run","fence-2.py; stdout.txt末兩行","整排+100原碼印True/True。"),e("independent-run","verify_layernorm.py shift/single/batch/constant blocks; probe-results whole_row_shift_equal/single_feature_shift_equal","reset原始x再改單格印True/False；其他batch改動也保留原第一筆。")], ["original-execution","probe-execution","fence2","probe-results"], v("整排+100：[True,True]；單格+100：[True,False]；另一位置/另一batch不改本位置。", "整排差0.0；[True,True]；單格[True,False]，改動後第二排[1.4069299698,-0.8276057839,-0.5793240666]；other_batch_preserved_exactly=true。", "clone原始輸入，清楚分開batch索引0、position索引1、feature索引0；不是把两练習連續加在同一修改後輸入。")),
    claim("ln-causal-wrong-axis","concept","誤將後面位置納入平均/變異數時，前面輸出可能間接依賴後面資料，即使attention mask正確仍可能偷看。", "course/chapters/04.md:123", "條件性風險：全序列統計會引入另一條跨位置依賴。正常LayerNorm(3)本身保留位置獨立；不宣稱它替attention/其他跨位置計算保證完整因果性。", [e("pytorch-ln","normalization.py:114–117,140–141","normalized_shape=(2,3)可一起normalize最後兩維，反面就是把position納入統計。"),e("independent-arithmetic","跨位置mean/variance公式的input集合","若前位置以包含後位置的mean與variance減除/縮放，則改後位置可改前位置；不經attention也有這條依賴。"),e("independent-run","verify_layernorm.py wrong-axis block; probe-results.wrong_axis_early_change","有界反例LayerNorm((2,3))只變第二排100，第一排最大差0.2312439084；正確LayerNorm(3)無此差。")], ["probe-execution","probe-results","independent-probe-code"])
]
report = {"schema_version":1,"review_stage":"technical","lesson_id":"4.3", "source":"course/chapters/04.md#4.3", "source_sha256":extraction["source_sha256"],"figure_sha256":{},
          "reviewer_task":"/root/phase4_factual_coordinator/factual_4_3","reviewer_context":"fresh","author_tasks":[],"verdict":"pass","reviewed_on":"2026-10-05",
          "reading_record":{"section":"完整亲讀course/chapters/04.md:88–131，raw UTF-8 bytes snapshot；兩fence逐項核對。", "other_actual_reads":["course/chapters/04.md:1–330（導言與相鄰上下文；只評4.3）", "course/chapters/03.md:141–183,190–243（3.5詞義前置及因果遮罩前置）", "tiny_perceptron/model.py:1–115", "docs/review-tools/factual-reviewer-instructions.md完整", "scripts/check_technical_reviews.py完整", "docs/review-tools/section_facts.py完整及本次bootstrap完整", "原論文摘要、§3/3.1/5.1與第2–3頁render，官方來源定位如sources"] , "prior_reports_read":False,"summary":"LN每位置最后feature軸取自己的N分母統計，eps避免零分母，再以跨位置共用perfeatureγ/β可調；純整排shift可被中心化消除而单格改动改變格間關係。"},
          "artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
          "tool_limitations":[{"tool":"pdftotext -layout (Poppler)","observed":"退出0但stderr有 Internal Error: xref num 765 not found but needed, try to reconstruct。", "impact":"擷取文字有方程式排版符號損失，未把此輸出當唯一原文核查。", "resolution":"由原PDF另用pdftoppm render第2–3頁，退出0/stderr空，已以view_image親看Eq.(3)(4)(5)及文本；原PDF與所有輸出永久保留。"}],
          "checks":{
              "factual_accuracy":{"status":"pass","details":"逐個concept/numeric/software核對，公式、H=3分母、eps、可調係數與位置獨立符合作者論文與wheel對應官方原碼。","claim_ids":[c["id"] for c in claims]},
              "numeric_verification":{"status":"pass","details":"純Python獨立求mean/v/sd/eps輸出，再與CPU執行比；float32誤差與四捨五入容許差已列於各numeric claim。γ2β1、translation/singlefeature/constant/batch/wrong-axis作有意義變化。","claim_ids":["ln-example-numbers","ln-affine-numbers"]},
              "figure_consistency":{"status":"not_applicable","details":"4.3原始section與extract均無引用SVG或其他圖；figure_sha256={}。原論文2–3頁渲染僅用於來源方程式核查，並非本節教材图。","claim_ids":[]},
              "source_verification":{"status":"pass","details":"直接取得作者arXiv v1原PDF及PyTorch官方git commit原碼並自己讀指定原文。下載URL/日期/版本/status/hash有永久收據；安裝normalization原碼與official hash完全相同；未讀舊4.3 report/history正文。","claim_ids":[c["id"] for c in claims]},
              "limitations":{"status":"pass","details":"本節只支持短forward示範，不支持語言能力或訓練成效；variance≈1/4而非精確；LayerNorm一般可跨多維，本節固定最後3格；全排shift/正倍率近似不可推廣至單格改動；allclose對非零other一般含rtol。沒有GPU、訓練、data/model下載或上传。PDF工具限制已由实际原頁render核實補足。","claim_ids":["ln-center-scale","ln-axes-independent","ln-affine-shared","ln-example-numbers","ln-affine-numbers","ln-original-api","ln-shift-and-indexing","ln-causal-wrong-axis"]}
          }}
(ROOT / "docs/technical-reviews/4.3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
manifest = [{"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(BASE.iterdir()) if path.is_file() and path.name != "manifest.json"]
(BASE / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"report":"docs/technical-reviews/4.3.json", "source_sha256":report["source_sha256"], "claims":len(claims),"artifacts":len(artifacts),"verdict":report["verdict"]}))
