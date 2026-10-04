"""Persist the reviewer's own complete reading observations and exact source identities."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
initial = json.loads((OUT / "source-manifest.json").read_text())
if not (OUT / "source-manifest.initial.json").exists():
    (OUT / "source-manifest.initial.json").write_text(json.dumps(initial, ensure_ascii=False, indent=2)+"\n")
    shutil.copytree(OUT / "sources", OUT / "sources.initial", dirs_exist_ok=True)
else:
    initial = json.loads((OUT / "source-manifest.initial.json").read_text())
for item in initial["chapters"] + initial["figures"]:
    item["snapshot"] = item["snapshot"].replace("/sources/", "/sources.initial/")
(OUT / "source-manifest.initial.json").write_text(json.dumps(initial, ensure_ascii=False, indent=2)+"\n")
current = json.loads(json.dumps(initial))
for item in current["chapters"] + current["figures"]:
    item["snapshot"] = item["snapshot"].replace("/sources.initial/", "/sources/")
for chapter in current["chapters"]:
    data = (ROOT / chapter["source"]).read_bytes()
    (ROOT / chapter["snapshot"]).write_bytes(data)
    chapter["sha256"] = sha(data)
    chapter["byte_count"] = len(data)
    headings = list(re.finditer(rb"^## ([0-9]+\.[0-9]+) ([^\n]+)\n", data, re.M))
    for i, section in enumerate(chapter["sections"]):
        end = headings[i+1].start() if i+1<len(headings) else len(data)
        section["sha256"] = sha(data[headings[i].start():end])
        section["start_line"] = data[:headings[i].start()].count(b"\n")+1
        section["end_line"] = data[:end].count(b"\n")

understanding = {
"1.1":"字與ID是可逆查找約定；Unicode排序固定編號，不產生字義；反序字表會變ID但不變還原文字。",
"1.2":"先去重並分完整來源，再切短片段；train調參、validation選設定、test作最後檢查，交集檢查只排除完全重複。",
"1.3":"s[:-1]和s[1:]建立每題下一字答案；長度相等仍可能學錯任務，shift與限制可見前文分工不同。",
"1.4":"unigram將次數除總數，所有前文得到同一分配；最大值選取與依機率抽樣是另一層規則。",
"1.5":"bigram每列是目前字、每欄是下一字；貓列有兩次看，加一後看為3/9、其他各1/9，每列用自己的分母。",
"1.6":"Embedding在此是V×V可調候選分數表；輸入選列，no_grad只管手動填表，查表本身未更新參數。",
"1.7":"exp後除總和得到softmax；先减最大值保留比例而避免溢位，候選軸各自正規化，高機率不等於答案正確。",
"1.8":"答案ID指定正確機率那格，-log(p)衡量代價；F.cross_entropy收原始logits，三候選示例loss約0.2395。",
"1.9":"有限差分估附近敏感度；w=1的平方代價導數為-4，符號說微調方向，h過小會放大捨入影響。",
"1.10":"前向w=3→u=6→L=36，反向局部敏感度12乘2得24；計算圖記運算關係，多路影響要相加。",
"1.11":"每參數偏導排成梯度[-4,2]；總代價6不是更新量，參數與梯度形狀相同且數值用途不同。",
"1.12":"backward只求導；no_grad內按w−ηgrad更新，再用新w重算loss，η=0.1得1.4和2.56，大步可能惡化。",
"1.13":"每次重算平方的新圖都把梯度加進同一.grad；清除桶得4/8/4，這與對舊圖第二次backward的釋放問題不同。",
"1.14":"逐步取當前字那列、抽一字、接回輸入；固定十二次決定長度，實訓錯填形狀由等號前文相同解釋。",
"1.15":"正溫度縮放logits後再softmax，改抽樣分配不改參數；greedy排序仍相同，T=0不代入除法公式。",
"2.1":"同一ID在[V,D]表取同一特徵向量，梯度可回到用過的列；它與經上下文混合後的表示不同。",
"2.2":"按位置拼接[B,C,D]→[B,C×D]保留順序，reshape的-1推軸長；固定窗口使下一層輸入格數固定。",
"2.3":"Linear每個輸出一套加權配方和bias；權重[out,in]需轉置，最後特徵軸作用而前置筆/位置軸保留。",
"2.4":"兩層純線性仍可合成一層；手設x和−x再ReLU後相加得到x的絕對值，三點形成單一直線無法表示的V形。",
"2.5":"窗口1/3/4缺顏色線索、5字才可分兩題；新版先命名查表拼接後的MLP及非線性作用，表格顯示更長窗口不保證驗證較好。",
"3.1":"非負且和為1的讀取比例加權混合values，位置軸被消去而特徵保留；比例大小不等於實際數值貢獻。",
"3.2":"query與各key點積得到匹配分數，再softmax；改query改偏好，點積還受向量長度影響。",
"3.3":"Q的列是查詢、K的列是候選，QKᵀ輸出[Tq,Tk]；同源自注意力仍需分開需求與匹配角色。",
"3.4":"Q/K決定候選比例，再用相同比例混合V；輸出特徵數由V決定，候選與V列需對齊，例值約7.042/14.083/21.125。",
"3.5":"在獨立零均值單位變異數假設下，D項加總變異數D、標準差√D；除√D控制特徵數引起的尺度，並非訓練後一律保證std=1。",
"3.6":"j≤i許可目前和過去、禁止未來；-inf在softmax後成0，最後value加100不影響前3位置，shift與mask缺一不可。",
"3.7":"先投影Q/K/V再拆head，每頭仍讀全部位置；[1,4,8]拆為[1,2,4,4]再拼回，K快取不是4×4權重表。",
"4.1":"字表特徵加同寬絕對位置查表向量，使同ID不同位置起始表示不同；位置表上限與學會語序是兩種限制。",
"4.2":"y=x+f(x)的原路與修正路相加；f=2x時數值[3,6]、對每格梯度1+2=3，殘差不保證無損保留一切。",
"4.3":"LN每位置最後特徵軸單獨算均值/biased variance，epsilon避免除零，再作每特徵gamma/beta；整排平移與改一格不同。",
"4.4":"逐位置FFN先4→16、GELU、16→4，共用配方但不跨位置讀取；Dense讓每位置使用完整配方參數。",
"4.5":"pre-norm先整理分支再Attention/FFN，加回未被替換的主路；shape保持，cache供後續生成，auxiliary=0不是語言loss=0。",
"4.6":"TinyLM每位置D特徵轉V候選logits；三輸入位置各有不同可見前文，續寫用最後列，loss先展成三題二軸，約3.336。",
"4.7":"原序列[1,2,3,4]只shift一次成x=[1,2,3],y=[2,3,4]；因果許可使三題可平行，masked_loss不再shift，backward仍未更新。",
"4.8":"深度只複製block而非固定入口出口；width8每block840、固定1360，總2200/3040/3880，成本與實際準確率分開驗證。"
}
figure_notes = {
"character_ids.svg":"原句貓看狗逗號狗看貓句號逐位映射3/2/1/4/1/2/3/0；重複貓同色同ID，無裁切。",
"foundations_shift.svg":"三列貓→看、看→狗、狗→句號對齊；箭頭指一題而非複製目前字。",
"foundations_bigram_row.svg":"七候選列、看計數3及3/9、其餘1及1/9與正文相符，橘框清楚。",
"foundations_chain.svg":"前向藍箭頭3→6→36與反向橘敏感度12、2清楚分開；總梯度24可讀。",
"foundations_embedding.svg":"五行三特徵完整；右三位置ID1/2/1取相同/不同/相同列，顏色無誤導語義。",
"foundations_nonlinearity.svg":"同權重左右兩組三點，純線性0/0/0對ReLU2/0/2；x軸與y軸可分且V形清楚。",
"window_training.svg":"橫軸1/3/5窗口及參數、縱軸loss，訓練0.334/0.213/0.199與驗證0.431/0.306/0.513皆符表；非時間曲線。",
"foundations_attention.svg":"Q=[1,0]對兩K得1/0，softmax比例0.731/0.269配對各自V，再到三特徵結果；流程與正文一致。",
"foundations_causal.svg":"對角與左下綠格顯示1、1/2、1/3、1/4，右上灰×；查詢列/可讀欄標示清楚。",
"foundations_heads.svg":"三投影後拆兩頭，各有全部四位置和每位四特徵，再拼回[1,4,8]；未暗示不同頭只读不同半句。",
"foundations_residual.svg":"[1,2]直接路和乘2修正路到加號後[3,6]，敏感度1+2=3與正文一致。",
"foundations_prediction_positions.svg":"三列前文[1]/[1,2]/[1,2,3]各自對候選0–19打分；綠色最后列用於新增1ID，沒有誤作三字同時生成。"
}
section_records = []
old_sections = {s["section_id"]:s for c in initial["chapters"] for s in c["sections"]}
changed_sections = []
for chapter in current["chapters"]:
    for section in chapter["sections"]:
        sid=section["section_id"]
        changed = section["sha256"] != old_sections[sid]["sha256"]
        if changed: changed_sections.append(sid)
        section_records.append({**section, "source":chapter["source"], "fullfile_sha256":chapter["sha256"],
                                "snapshot":chapter["snapshot"], "reading_ordinal":len(section_records)+1,
                                "understanding":understanding[sid], "open_issues":[],
                                "initial_sha256":old_sections[sid]["sha256"],
                                "version_read_basis":"Final 2.5 was fully reread after the author revision; all other section bytes match the initial complete read." if changed else "Completely read in initial sequential pass; unchanged at final identity check."})
assert changed_sections == ["2.5"], changed_sections
for f in current["figures"]:
    assert sha((ROOT/f["source"]).read_bytes()) == f["sha256"]
    f["visual_inspection"] = {"method":"Browser-rendered PNG inspected through view_image after reading SVG source", "understanding":figure_notes[Path(f["source"]).name],"issues":[]}

support = []
for rel in ["course/first-steps.md", "tiny_perceptron/attention.py", "tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/simple.py", "tiny_perceptron/data.py"]:
    p=ROOT/rel; data=p.read_bytes(); dest=OUT/"sources"/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
    support.append({"source":rel, "sha256":sha(data), "snapshot":str(dest.relative_to(ROOT)),
                    "read_scope":"W.1–W.6 complete text (lines 1–190); W.7 opening table was also returned but not reviewed as a whole section; prerequisite figures were not visually inspected." if rel.startswith("course/") else "Direct source read, not prior review artifacts; relevant implementation checked."})
findings = [
{"id":"01-04-P2-MLP-first-use", "priority":"P2", "section_id":"2.5", "category":"term introduction", "initial_quote":"讓三個MLP分別看最近1、3、5字", "problem":"首次出現MLP時未展開名稱，也未明說它是前述查表/拼接/線性/非線性組成的小網路；初學者無法把實驗名稱接回2.1–2.4。", "action":"在表格之前加一句淺白定義：『每個模型先查表、拼接，再用線性層與非線性轉換串成的小網路猜字；這種網路叫多層感知器（multilayer perceptron，MLP）。』", "status":"closed_after_actual_final_section_reread", "initial_section_sha256":old_sections["2.5"]["sha256"], "final_section_sha256":next(s["sha256"] for s in section_records if s["section_id"]=="2.5"), "closure_understanding":"現稿在表格前將MLP接回查表→拼接→多層網路，並連2.4彎曲規則；讀者可理解模型名稱而無新計算跳步。"},
{"id":"01-04-P2-intermediate-activation", "priority":"P2", "section_id":"2.5", "category":"implementation mismatch in an intermediate revision", "observed_quote":"再把線性層與[2.4 的 ReLU](#2.4)串成的小網路用來猜字", "problem":"中途作者補述把實驗網路說成ReLU；直接讀ContextMLP.forward發現真實為torch.tanh(self.hidden(features))。", "evidence":"tiny_perceptron/simple.py ContextMLP.forward; source snapshot and SHA in supporting_sources.", "action":"用『線性層與非線性轉換』描述此網路並連2.4的原理，避免指定錯誤激活。", "status":"closed_after_actual_final_section_reread", "intermediate_source_identity":"Intermediate text was genuinely read in exec output, but was revised again before a hash/snapshot was recorded; no fullfile/section hash claim is made for that transient version.", "closure_understanding":"現稿不再宣稱本實驗使用ReLU；泛稱非線性轉換與tanh實作相容，且MLP原理仍連回2.4。"}
]
actual_read_events = [
{"ordinal":1,"source":"course/chapters/01.md","lines":"1–240","type":"complete sequential text read, first chunk"},
{"ordinal":2,"source":"course/chapters/01.md","lines":"241–487","type":"complete sequential text read, concluding chunk"},
{"ordinal":3,"source":"course/chapters/02.md","lines":"1–193","type":"complete sequential text read, initial version"},
{"ordinal":4,"source":"course/chapters/03.md","lines":"1–222","type":"complete sequential text read"},
{"ordinal":5,"source":"course/chapters/04.md","lines":"1–180","type":"complete sequential text read, first chunk"},
{"ordinal":6,"source":"course/chapters/04.md","lines":"181–302","type":"complete sequential text read, concluding chunk"},
{"ordinal":7,"sources":["tiny_perceptron/attention.py","tiny_perceptron/model.py","tiny_perceptron/modern.py"],"type":"direct implementation source reading"},
{"ordinal":8,"sources":[f["source"] for f in initial["figures"]],"type":"SVG source read in chapter order; window_training.svg final lines read in a followup"},
{"ordinal":9,"source":"course/first-steps.md","lines":"1–201 returned; W.1–W.6 complete","type":"explicitly linked prerequisite text read"},
{"ordinal":10,"sources":["character_ids.png","foundations_shift.png","foundations_bigram_row.png","foundations_chain.png"],"type":"actual rendered visual inspection"},
{"ordinal":11,"sources":["foundations_embedding.png","foundations_nonlinearity.png","window_training.png","foundations_attention.png"],"type":"actual rendered visual inspection"},
{"ordinal":12,"sources":["foundations_causal.png","foundations_heads.png","foundations_residual.png","foundations_prediction_positions.png"],"type":"actual rendered visual inspection"},
{"ordinal":13,"source":"course/chapters/02.md","section_id":"2.5","type":"complete reread of intermediate revision; found introduced ReLU mismatch"},
{"ordinal":14,"sources":["tiny_perceptron/simple.py","tiny_perceptron/data.py"],"type":"direct primary implementation read; ContextMLP tanh confirmed"},
{"ordinal":15,"source":"course/chapters/02.md","section_id":"2.5","type":"final revision fully read from body to EOF; no heading in this first final-version chunk"},
{"ordinal":16,"source":"primary-sources/transformer.txt","lines":"150–205,227–264","type":"actual original paper excerpt read; scaled attention, independence assumption, causal mask, position-wise FFN"},
{"ordinal":17,"source":"course/chapters/02.md","section_id":"2.5","type":"final revision fully reread with its heading through EOF; term and activation mismatch closed"},
{"ordinal":18,"source":"primary-sources/installed-pytorch-layernorm.py.txt","type":"official installed class docstring read; normalization axes, correction=0, gamma/beta/epsilon confirmed"}
]
inspection = {
"review_type":"fresh independent supplemental chapter precheck, not canonical baseline review or final whole-book approval",
"identity":{"task_path":"/root/v4_wholebook_scan_01_04","parent_task_path":"/root","role":"independent existing-chapter reader","model_identifier":"not exposed in this task; no invented model or human identity","generated_at":datetime.now(timezone.utc).isoformat()},
"scope":{"chapters":["01","02","03","04"],"complete_sections":35,"rendered_figures_actually_inspected":12,"reader_lens":"數學不錯高中生／基本數學大學生，第一次接觸LLM；不替首次出現術語補作者背景。","no_old_review_verdicts_read":True,"canonical_files_changed_by_this_reviewer":False,"chapter_20_read":False,"fullbook_final_pass_claim":False},
"actual_read_order":actual_read_events,
"file_identity":{"initial":initial["chapters"],"current":current["chapters"],"changed_sections":["2.5"],"verification":"All other section bytes and all twelve canonical figures still match the versions completely read. The final 2.5 is an actual reread, not a hash-only update."},
"section_observations":section_records,
"figures":current["figures"],
"supporting_sources":support,
"primary_source_evidence":[
 {"source":"Vaswani et al., Attention Is All You Need, arXiv:1706.03762","url":"https://arxiv.org/pdf/1706.03762","receipt":"primary-sources/receipt.json","actually_read":"PDF-to-text lines150–205,227–264","claims_checked":"§3.2.1 QKᵀ/√dk then softmax V; footnote4 independent zero-mean unit-variance components give dot-product variance dk; §3.2.3 illegal connections masked by −∞; §3.3 FFN acts separately/identically on positions. The paper’s ReLU choice was not assumed to be this repository’s tanh/GELU choice."},
 {"source":"Official PyTorch LayerNorm class docstring from installed torch2.14.1+cpu","receipt":"primary-sources/installed-pytorch-layernorm-receipt.json","actually_read":"complete docstring printed by inspect.getdoc","claims_checked":"last normalized_shape dimensions; variance correction=0; epsilon denominator; learnable per-feature gamma/beta with initial ones/zeros","web_limitation":"stable API URL returned only HTML redirect; its exact 2.14 target returned HTTP403. Those responses were not treated as API source content; local first-party class source supplied actual evidence."}
],
"findings":findings,
"open_findings":[],
"failure_history_assessment":{"deletions_recommended":[],"retained_for_learning":["1.12 η过大使代價变差：解释局部梯度与步长分离。","1.13 重复backward累计/旧图释放：解释PyTorch两个不同机制。","1.14 顏色=三角：最直接说明只看等号不能区分欄位。","2.5 5字窗口验证更差：提醒信息可见、训练拟合、泛化不是同一件事。","3.6 移除因果mask使反例断言失败：可观察验证防偷看。"],"judgment":"01–04没有要求删除的无学习价值工程事故或修补历史；所有负例直接支持该节概念。"},
"numeric_execution":{"records_file":"numeric-execution.json","stdout":"numeric.stdout.txt","stderr":"numeric.stderr.txt","environment":"numeric-environment.txt","selected_sections":["1.8","2.4","3.4","3.6","4.6","4.8"],"executed_python_fences":7,"all_exit_zero":True,"important_observed_values":{"4.6_mean_loss":3.335638999938965,"4.8_parameters":[2200,3040,3880]},"not_a_full_notebook_run":True},
"limitations":["未跑完整notebook/kernel流程或所有练习，root会另执行。","未重新训练T.3/T.4或独立重建训练报告中的loss曲线；本报告核读叙述的合理范围、所选数字例子、接口与参数计数。","未评判实际自然语言能力，也未读新第20章。","未全面读W.7以后暖身及训练章节；W.1–W.6是限定的明确前置文本核读，前置图未作视觉复审。","网页发布、Colab、公共报告链接的可访问性未作端到端验收。","中途ReLU版读过但未及时保存SHA；只保留真实观察引句，无虚构版本身份。"],
"uncertainties_remaining":["T.3/T.4训练数字的独立重跑不属于本次小数字验证；须由完整训练/实验流程审查处理。"]
}
(OUT/"source-manifest.json").write_text(json.dumps(current,ensure_ascii=False,indent=2)+"\n")
(OUT/"inspection.json").write_text(json.dumps(inspection,ensure_ascii=False,indent=2)+"\n")
rows = "\n".join(f'| {s["section_id"]} | {s["understanding"]} | 無未解決問題 |' for s in section_records)
file_rows="\n".join(f'| `{c["source"]}` | `{c["sha256"]}` |' for c in current["chapters"])
report=f'''# 第01–04章獨立完整閱讀預檢

任務身分：`/root/v4_wholebook_scan_01_04`；parent：`/root`。這是 supplemental 預檢，不替代既有 formal reader／technical review，也不宣稱全書 final pass。沒有改教材、基線 review、checker 或時間 metadata。

實讀順序為01→02→03→04，35節全部完整閱讀；01與04依行段分兩次讀完，不靠搜尋或雜湊代替閱讀。之後直接讀必要實作、12張SVG來源、W.1–W.6明確前置文字，再查看全部12張浏览器渲染圖。沒有讀舊review verdict或作者歷史來補背景。逐次實讀事件、每節範圍和SHA、原始來源快照及每張圖的SHA都在[inspection.json](inspection.json)与[source-manifest.json](source-manifest.json)。

本範圍目前沒有未解決的重大技術或閱讀問題。發現的2項P2問題都已向root回報，由root修文後實讀新版2.5確認解除。教材中的失敗例是概念反例，沒有找到應刪的無學習價值事故歷史。

## 發現與實際修正後重讀

1. **P2：2.5首次用MLP但未介紹。** 初稿說「讓三個MLP分別看最近1、3、5字」，2.1–2.4此前沒有為網路命名。建議在表格前加「查表→拼接→線性與非線性小網路」的淺白定義。現稿實際讀到「這種由線性層與非線性轉換串成的網路，叫多層感知器（multilayer perceptron，MLP）」並連回2.4的彎曲規則，已能理解實驗名稱。
2. **P2：中途補述把實驗激活說成ReLU。** 實讀中途修文的「線性層與2.4的ReLU串成」後，直接讀`tiny_perceptron/simple.py`，發現`ContextMLP.forward`實際使用`torch.tanh`。建議泛稱非線性轉換並連原理。現稿已照此處理，完整重讀2.5到末尾，與實作相容。中途版在又一次改文前沒有保存SHA，因此沒有偽造该版source身份；只留真實讀到的引句。

初讀與最後完整重讀的2.5 section SHA：`{old_sections['2.5']['sha256']}` → `{next(s['sha256'] for s in section_records if s['section_id']=='2.5')}`。初稿快照保留於`sources.initial/course/chapters/02.md`；現稿快照在`sources/course/chapters/02.md`。其他34節與12張圖内容仍與初讀相同。

## 每節自己的理解／問題

| sectionID | 實際理解 | 問題 |
| --- | --- | --- |
{rows}

## 來源與可核對證據

- 原論文[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)實際下載HTTP200，讀§3.2.1、footnote4、§3.2.3與§3.3的原文。`primary-sources/transformer.txt`第150–205、227–264行支持attention的縮放、独立假设、因果遮罩和逐位置FFN。沒有把原論文的ReLU或權重共享當成此專案必然配置。
- PyTorch官方`LayerNorm`網頁stable入口只給HTML redirect，2.14目的頁403；沒有把redirect當正文。改讀已安裝官方PyTorch2.14.1+cpu的`LayerNorm`完整docstring，確認最後特徵軸、correction=0變異數、epsilon及γ/β初始化。原始class來源與SHA保存在`primary-sources/installed-pytorch-layernorm.py.txt`及receipt。
- 直接讀本專案`attention.py`、`model.py`、`modern.py`、`simple.py`、`data.py`核對手寫attention、K/V介面、DenseFFN、ContextMLP(tanh)、shift及split。來源快照與SHA見inspection的supporting_sources；未參考任何既有review結論。

## 真實执行與圖檢查

用既有`.venv/bin/python`執行1.8、2.4、3.4、3.6、4.6、4.8的原樣Python fences，共7段、6程序，全部exit0。环境實錄是Python3.13.5、Torch2.14.1+cpu、CUDA不可用。4.6平均loss=`3.335638999938965`；4.8參數格數=`2200,3040,3880`。數值、ReLU三點、attention混合及因果遮罩检查皆符合正文；原码、argv、exit、stdout/stderr与SHA在[numeric-execution.json](numeric-execution.json)，不是事后虚构日志。

12張图已從真实SVG渲染成PNG並实际查看：字ID、shift、bigram、chain rule、embedding、ReLU、窗口曲線、Q/K/V、causal mask、多頭、residual、各位置下一字预测。标签、箭头、数据与正文一致，沒有发現裁切或把shape／模型能力画錯的問題。每圖的獨立理解、來源SHA、render SHA及檔案都在inspection.figures。

## 範圍、未跑與限制

没有跑全部notebook或所有练习，也没有重训T.3/T.4；训练报告中的loss数字不在本次独立重跑的保证范围。root会另外跑完整kernel。沒有讀第20章，未检验自然語言能力、公共網站／Colab的端到端可用性。W.1–W.6仅作为明确前置文字读完，未对前置图进行视觉复审。没有对全书宣布最终通过。

## 最後正文檔案身份

| 完整來源檔案 | SHA-256 |
| --- | --- |
{file_rows}

每个section和figure的完整SHA另见inspection.json；这些身份记录只是绑定真正已读文字与图，未用来替代阅读。
'''
(OUT/"report.md").write_text(report)
print("Wrote report.md and inspection.json; sections",len(section_records),"figures",len(current["figures"]),"changed actual-reread sections",changed_sections)
print("Final chapter identity",[(c["source"],c["sha256"]) for c in current["chapters"]])
