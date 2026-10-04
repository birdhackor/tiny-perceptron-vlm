from pathlib import Path
import json,hashlib,datetime,re
root=Path('/workspace/tiny-perceptron-vlm'); out=root/'outputs/natural-v4/wholebook-review/13-16'
initial=json.loads((out/'initial-manifest.json').read_text())
notes={
'13.1':'偏好是同一問題、同一要求下的比較，兩篇都答對時格式也可決定勝者；家族切分能避免交換加數洩漏，但一長段測試流程例外解說會打斷最基本比較單位。',
'13.2':'prompt共享，chosen/rejected各自render成已shift輸入與答案目標；助手角色預測第一個答案、答案預測EOS，不能再shift或混搭兩篇前文；截短標籤的限制說清楚。',
'13.3':'回答機率是條件鏈連乘，log把它換成有效答案格的加總，含約定EOS且忽略prompt/PAD；兩項0.7870得0.6193，程式實跑一致。',
'13.4':'reference是獨立複製並凍結的固定SFT基準，policy相對候選差距改善不代表正確候選已排前；clone、deepcopy、eval與凍結責任分得清楚。',
'13.5':'DPO對相對margin取負log-sigmoid，負chosen梯度與正rejected梯度使下降更新改進差距；共同模型權重與二選一排序／自由生成的差異有交代。',
'13.6':'beta既是KL理論目標係數，也在DPO局部導數中影響sigmoid飽和，固定正margin時梯度幅度可不單調；三組數值實跑一致，沒有把局部結果外推。',
'13.7':'正文bytes與含EOS目標長度分开，資料偏好長篇和sum本身的長度效應不能靠偷換mean解決；控制支線只支持格式排序，生成仍未算對。',
'13.8':'溫和修正可查證錯誤和粗魯反駁、無條件附和不是同一目標；字典只是偏好資料展示，還未訓練，正確前提與主觀喜好對照有學習價值。',
'13.9':'配方對數不等於有效目標曝光量，未知None不等於失敗0，偏好改善要另測其他能力；自然資料截短與未測生成的界線清楚。',
'13.10':'獎勵模型從同題分數差估计比較胜率，sigmoid差0／2的結果可手算；有限回答卡讀結構特徵且標籤由規則產生，沒有冒稱讀中文或真人RLHF。',
'13.11':'advantage是本次reward減收集時的value，當作固定評語教策略，critic另外學；一次選卡的bandit與逐token多步任務有明確區分。',
'13.12':'ratio是目前機率除收集時舊機率，old紀錄一輪內固定、下一輪更新，reference全程固定；兩格是兩次樣本，沒有假裝它們组成完整分布。',
'13.13':'min(rA,clip(r)A)在正A的高ratio與負A的低ratio側變平，錯方向仍被懲罰；六格目標與導數實跑一致，圖沒有把PPO平台說成機率硬上限。',
'13.14':'critic看情境估計策略預期reward，獎勵模型看已選回答；平方誤差與枚舉KL(p||q)分别更新估計／限制策略，四角色圖可對回各自凍結時間。',
'13.15':'舊策略抽卡、凍結評分、固定advantage、新策略求導及critic更新串成一次；原程式實跑選0、ratio1、policy改變且RM/reference無梯度，但「新增的兩行」屬無用修訂語句。',
'13.16':'長度獎勵可與真值／格式衝突；正式結果是RM排序正確但選卡仍沒完成句子要求，沒有把人工漏洞反例冒充實驗觀察；其13.5前置與導讀PPO跳讀路線衝突。',
'13.17':'DPO省獨立RM但仍用偏好標籤與固定reference，兩卡0.7／0.3可直接手算；同360更新不能稱同算力，其13.5前置再與導讀路線衝突，還留「本章新增」修訂語句。',
'14.1':'成對向量按位置旋轉，共同平移保留角度差與點積，sin/cos與弧度有從單位箭頭導入；兩圖和旋轉程式一致，沒有保證超長外推。',
'14.2':'LN消去共同偏移，RMS只調均方根尺度，兩列結果能手算；正文分清gamma乘法／beta加法，但圖將gamma/beta合稱可學縮放，應修正圖標籤。',
'14.3':'方向相同時點積仍受向量長度影響，L2正規化拿掉長度、另有scale調softmax尖銳度；特徵軸／候選軸-1的差別講清楚，未冒稱整模型加速。',
'14.4':'矩陣線性層連乘可合成，activation提供非線性；GELU用常態累積比例、ReLU平方处理負／正值的規則和五個固定輸出可理解。',
'14.5':'SwiGLU的SiLU gate可負或大於1，逐項控制content，三張矩陣比普通MLP兩張多；等hidden不等參數及未等容量的正式比較限制明確。',
'14.6':'輸入表／輸出表共享須是同Parameter，單純copy初值會各自變動；三候選打分與改表後4／1／3可追蹤，起點分布不同的實驗限制有說明。',
'15.1':'同FFN權重供每個token使用，增加列數增加工作而不增加權重；up、relu、down首列4／5可手算，玩具12參數與真模型規模分得清楚。',
'15.2':'expert是獨立儲存的規則，剛初始化並無學科專長，ModuleList重複引用不增加容量；三組變換與12／4計數清楚。',
'15.3':'router線性分數經softmax成混合係數，所有expert皆算的soft routing尚未省計算；log2／exp2轉換與2.25倍輸出可以逐步理解。',
'15.4':'top-k逐token挑整數expert索引並保留連續比例，topk本身不跑expert也不重算比例；離散選擇和連續學習路徑有解釋。',
'15.5':'多個expert沿相同特徵加權相加，top2重新正規化與top1保留比例約定明确；圖和例子沒有把expert相加與外層殘差混成同一件事。',
'15.6':'dispatch需保存來源token列号，combine按原列累加，覆蓋會漏掉同token第二份貢獻；図中列1零是刻意缺工作而非正常dropless遺失。',
'15.7':'top1把p除自身後gate恆1，任務梯度會斷；保留softmax比例時兩個scores都有相反梯度，原碼1.1852／-1.1852與0／0已實跑核對。',
'15.8':'幾乎均勻softmax仍可全送同expert，實際被選次数和平均機率必須分開；熵從-f log f及零份額規則導入，明確只總括全驗證集。',
'15.9':'N∑f_i p_i將離散負載視固定、由連續平均機率教router，toy均衡1／集中2.4可核對；也清楚說明auxiliary不是完美不均量且可小於1。',
'15.10':'總參數決定儲存、每token啟動只是容量代理，router隨expert數增加，完整字表列入proxy並非真正讀取格數或FLOPs；計數口徑充分交代。',
'15.11':'分派、排名、小矩陣与合併開銷可讓MoE較慢，CPU只量前向、GPU量更新是不同範圍；峰值起點差異有交代，但长方法記錄適合移往實驗頁。',
'15.12':'固定capacity按平均工作×factor向上取整，總容量足夠仍會單expert溢出；drop／fallback／reroute與殘差需另定，教材dropless並未實測drop收益。',
'15.13':'同總容量和同使用代理會得到不同Dense尺寸，需要相同資料與曝光量並另量速度；正式表承認約3.15%近似差距、單seed与proxy不等FLOPs。',
'16.1':'profiler的self／total／calls与權重bytes不一樣，CPU例子可以定位瓶頸；但暖機之後接TF32、六支訓練、allocator／reserved／獨立子程序等多段方法細節，讀者尚未進入效率主線。',
'16.2':'prefill算完整提示、decode算新Query但讀過去KV，四軸形状可追蹤；TTFT与單call時間、含prefill總時間與純decode吞吐量分清楚。',
'16.3':'因果前文各層不變可缓存K/V，新位置仍投影与读前文；完整／cache最後logit實跑差2.38e-7，位置offset与改前文要重建都說清楚。',
'16.4':'多Query共用少KV能縮cache，4／1KV的384／96bytes與形状實跑一致；MQA是端點、重新初始化不等原論文平均轉換，品質和儲存收益分開。',
'16.5':'EOS不自動隔離文件，packing須同segment及因果交集、重置位置並避跨文件下一字目標；圖的四乘四矩陣正確，正式兩段等长例只檢語意未省PAD。',
'16.6':'microbatch的誤差和除共同有效目標總数，最後一次step与裁剪才等价同大batch；原碼整批／正确累積9.3333、错平均7.5，實跑核對一致。',
'16.7':'FP16精度細但範围窄，BF16範围大但捨入较粗，autocast不改FP32原參數儲存；scaler先放大後还原、遇非有限梯度可跳步，有限不等答對。',
'16.8':'SDPA自带1/√D縮放、布林True允許且dropout需傳0，實跑手寫差5.96e-8；API不等指定Flash，正式FP32memory-efficient與低精度Flash補驗各自分開。',
'16.9':'online softmax更新最大值時同縮舊分母与分子，两塊得到24.2857；原程式與图數值一致，Flash仍完整注意力並減HBM搬移，灰基線圖不把新增峰值當整GPU用量。',
'16.10':'activation checkpoint保存區域邊界再在反向重算中間值，以時間換峰值；例子只比獨立輸入梯度而不錯比已累加權重梯度，與存模型checkpoint區別清楚。',
'16.11':'編譯首次準備與穩態每call要分量，500次回本是算式，eager backend只是接口示範；真Inductor例成功但未加速，null回本與未测整模型界線明確。'
}
findings=[
 {'id':'R13-route','priority':'P1','kind':'prerequisite_route','sections':['13.16','13.17'],'source_location':'course/chapters/13.md:5, 517, 552','problem':'導讀給PPO路線13.1–13.2 → 13.10–13.16 → 13.17；但13.16與13.17均要求讀過13.5。13.17又說接著回13.3–13.6。沿指定路線的初學者會先遇到被跳過的必要DPO代價。','reader_impact':'讀者會不知道要先補DPO還是本節已足夠，分流路線與前置表互相矛盾。','recommended_change':'選一種一致安排：最小修法是导讀將13.1–13.6列為共同基礎，再選讀13.10–13.17；若要保留PPO獨立路，將13.16的DPO結果標為可選对照，13.17用兩卡例补齊log与margin並取消13.5必需前置。','acceptance':'按導讀逐節走一遍，不需在未說明處回補另一支線；每節前置均為该路線已读或明列先補。'},
 {'id':'R13-revision-language','priority':'P2','kind':'unnecessary_revision_history','sections':['13.15','13.17'],'source_location':'course/chapters/13.md:5, 501, 567','problem':'「新增的PPO實驗」「新增的兩行顯示」「與本章新增的有限選卡實驗分開」是在說修訂時間，對理解當前方法沒有增益。','recommended_change':'直接改為「PPO實驗」「後兩行顯示」「與有限選卡實驗分開」。13.17所說「新增偏好含示範階段沒教的條件」则保留方法含义，寫成「偏好資料包含示範階段未教的條件」，不可删掉SFT比較限制。','acceptance':'學生不必知道教材曾新增哪些章節或輸出行，也仍能分清語言模型DPO與有限選卡PPO。'},
 {'id':'R14-normalization-label','priority':'P2','kind':'figure_math_label','sections':['14.2'],'source_location':'course/figures/normalization.svg','problem':'圖把「gamma / beta」都放在「可學縮放」框；beta是加法偏移，正文已正确解释，图標籤卻使兩者看似同一種縮放。','recommended_change':'框名改成「可學縮放／偏移」，下行標「gamma乘、beta加（RMS常只有gamma）」或拆成兩個短標籤。','acceptance':'單看圖也不會把beta當作乘法係數，且與正文LN／RMS約定一致。'},
 {'id':'R1316-experiment-density','priority':'P2','kind':'learning_load','sections':['13.1','13.4','13.6','13.7','13.9','13.15','13.17','15.11','16.1','16.4','16.7','16.8','16.9','16.10','16.11'],'source_location':'尤其course/chapters/13.md:33, 503–513；course/chapters/16.md:36–44','problem':'核心小例子後，经常接精密run流程、五位小數、每批／每回合曝光次数和記憶體管理邊界；16.1在prefill等主題之前就連續解說TF32、六支訓練、allocator、reserved、子程序與全報告范围。這些細節可核對，但多次插入会讓第一次學原理的讀者把方法与本機記錄混在一起。','recommended_change':'正文保留一個能支持概念的真實結果、分母定义與关键限制；完整排程／指紋／精密逐題數值／backend清單／每支記憶體方法移至T.8或各節「實驗細節」可選區。13.1的家族切分保留，測前後是否算挑參數用一句說明即可；16.1留self/total、暖機及權重不等峰值，L4方法總卡放在效率實驗頁。不要刪除toy和真模型、候選排序和生成、記憶體起點和增量的概念区别。','acceptance':'讀者先完成概念→小例子→現象→練習主線，只有想重跑／核對時才進入精密run資料；原實報與限制仍可回查。'}
]
figs=json.loads((out/'figure-source-labels.json').read_text()); figure_to_section={}
sections=[]
for f in initial['files']:
 lines=(out/'original-sources'/f['path']).read_text().splitlines(keepends=True)
 for s in f['sections']:
  text=''.join(lines[s['start_line']-1:s['end_line']]); refs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
  sectionfigs=[]
  for ref in refs:
   fp=(root/Path(f['path']).parent/ref).resolve(); rel=str(fp.relative_to(root)); sectionfigs.append(rel);figure_to_section.setdefault(rel,[]).append(s['id'])
  sections.append({**s,'source':f['path'],'read_ordinal':len(sections)+1,'understanding_and_issue':notes[s['id']],'status':'improvement_needed' if any(s['id'] in z['sections'] for z in findings) else 'no_material_issue_found','finding_ids':[z['id'] for z in findings if s['id'] in z['sections']],'figure_sources':sectionfigs,'sha256_matches_initial':hashlib.sha256(text.encode()).hexdigest()==s['sha256']})
source_paths=['tiny_perceptron/alignment.py','tiny_perceptron/posttraining.py','tiny_perceptron/modern.py','tiny_perceptron/attention.py','tiny_perceptron/model.py','scripts/course_experiments/architecture.py']
code_sources=[{'source':p,'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest(),'bytes':(root/p).stat().st_size,'read_scope':'依工具日誌读取定义；architecture.py仅100–145和176–218行核對有效輸入auxiliary與累積分母；不是全源码审计'} for p in source_paths]
figure_notes={
'multimodal_preference_pair':'共同问题分两篇，事实均对但B格式失敗，与13.1一致。',
'multimodal_dpo_direction':'两差距-1到margin0，梯度±0.05與下降0.1后的差距-0.99正确。',
'ppo_clip':'正優勢右側1.2平台、負優勢左側0.8平台，圖形与公式／六格导数一致。',
'ppo_roles':'RM、critic、old紀錄、reference的視野與凍結時間分別明确。',
'ppo_dpo_routes':'从同SFT选卡起点分叉后共同验收，没有画成PPO之后再DPO。',
'rope':'单对Q/K箭头夹角45度、共同旋转保夹角，配合正文多转速限制可理解。',
'rotation_components':'45与135度同单位圆；正／负水平cos与同正sin数值正确。',
'normalization':'同token沿特征轴處理、形状保持；gamma/beta均称可学縮放會混淆乘法／偏移，见R14-normalization-label。',
'dispatch':'按expert分组后依原row加回，提醒同token贡献相加。',
'architecture_dispatch_example':'来源[2,0,2]的贡献[1,1]、[2,2]、[3,3]回归列2为[4,4]，列1无工作与正文一致。',
'cache':'前三格保留、新位置3接在末尾，图标號為位置而非输入ID；與说明一致。',
'architecture_gqa_cache':'四Query各讀四KV共384bytes／四Query讀单KV96bytes，连线和单位一致。',
'packing':'四行允许矩阵为下三角两块，B禁止读A，与代码明确同segment约束一致。',
'causal':'四乘四下三角含對角，未来右上角禁止，True允许語意清楚。',
'architecture_online_softmax':'第一块分母1.5分子25，旧统计乘0.5後加新格，最后1.75／42.5與24.2857都正确。',
'flash_allocated_memory':'同65MiB灰基线的四条柱按同尺度呈現，新增0.27／2.03與总65.27／67.03分開，无把增量当整GPU比例。'
}
for f in figs:
 f['section_ids']=figure_to_section.get(f['source'],[]); f['visually_inspected']=True;f['visual_read_method']='Inkscape 1.4静态渲染；view_image实际查看sheet-1至sheet-4；source文字／幾何数值另核对'; f['interpretation']=figure_notes[Path(f['source']).stem]; f['unchanged_since_capture']=hashlib.sha256((root/f['source']).read_bytes()).hexdigest()==f['sha256']; f.pop('text_labels',None)
authorities=[
 {'source':'dpo','sections':['13.4','13.5','13.6','13.10','13.14','13.17'],'actually_read':'原论文第3–4节，公式1–7（extracted text 75–255）','conclusion':'reference KL目标的beta、reward比较负log-sigmoid和DPO相对log概率公式一致；正文未把不同实现保证为同模型。'},
 {'source':'ppo','sections':['13.12','13.13','13.15'],'actually_read':'第3节公式6–7和Figure1说明，Algorithm1（101–146、238–249）','conclusion':'ratio与固定采样记录、多epoch复用和正负优势不对称裁切一致。'},
 {'source':'switch','sections':['15.5','15.9','15.12'],'actually_read':'2.1–2.2相关gate说明、容量式3、平衡项式4–6（243–330、347–436）','conclusion':'top1原softmax gate、capacity与N∑fP形式一致；本书top2改归一化與扩展分母已明示。'},
 {'source':'roformer','sections':['14.1'],'actually_read':'3.2.2公式14–16（257–293）','conclusion':'二维对的正交旋转与相对点積明确，书中没有把其不变性当成超长泛化保证。'},
 {'source':'gqa','sections':['16.4'],'actually_read':'2.1–2.2转換和端点（38–114）','conclusion':'GQA-1相当MQA，论文转换用KV平均；本书明确随机新KV不是复现同配方。'},
 {'source':'flash','sections':['16.8','16.9'],'actually_read':'3.1 tiling/recomputation、Algorithm1和Theorem1（236–322）','conclusion':'online统计共同重缩放、完整注意力及HBM/SRAM限制一致。'},
 {'source':'pytorch-sdpa','sections':['16.4','16.8'],'actually_read':'2.14官方页面GQA段、布林mask与dropout警告，摘录存documentation-excerpts.json','conclusion':'目前2.14确列Flash、cuDNN、math与NVIDIA CUDA memory-efficient支持GQA；True允许、dropout不随eval自动清零正确。'},
 {'source':'pytorch-amp','sections':['16.7'],'actually_read':'Gradient Scaling及FP16 underflow/max65504段落','conclusion':'损失扩大带同比例梯度、更新前缩回与学习率区别正确。'},
 {'source':'pytorch-grad-scaler-official','sections':['16.7'],'actually_read':'官方2.14.1 commit的GradScaler docstring；infs/NaNs skip段','conclusion':'非有限梯度时跳过optimizer.step、成功尝试与真正更新次数需分计的说法正确；生成API页尝试404后改查真实源文件。'}
]
receipts=json.loads((out/'sources/receipts.json').read_text())
for a in authorities:
 match=next(x for x in receipts if x['name']==a['source']); a.update({k:v for k,v in match.items() if k!='name'})
read_order=[{'ordinal':i+1,'path':p,'line_range':span,'method':'完整工具輸出逐段读，非只搜标题／关键词'} for i,(p,span) in enumerate([('course/chapters/13.md','1–170'),('course/chapters/13.md','169–351'),('course/chapters/13.md','352–579'),('course/chapters/14.md','1–208'),('course/chapters/15.md','1–193'),('course/chapters/15.md','194–343'),('course/chapters/15.md','344–469'),('course/chapters/16.md','1–167'),('course/chapters/16.md','168–338'),('course/chapters/16.md','339–474')])]
file_integrity=[]
for f in initial['files']:
 current=hashlib.sha256((root/f['path']).read_bytes()).hexdigest(); file_integrity.append({'source':f['path'],'initial_sha256':f['sha256'],'final_sha256':current,'unchanged_bytes':current==f['sha256'],'bytes':f['bytes']})
inspection={'schema':'wholebook-independent-precheck-slice-v1','task_identity':'/root/v4_wholebook_scan_13_16','parent_task':'/root','scope':'既有第13–16章預檢，非整份教材最终验收；未读／未判第20章','reader_simulation':'数学不错高中生／有基本数学大学生初次接触LLM；不是声称招募真人学生','started_utc':initial['started_utc'],'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'skill_applied':'cloud-environment-onboarding:setup；仅现有环境检查与验证原则，未变更依赖或环境配置','independence':'未读取旧review verdict或作者修订历史补背景；git status仅见檔名；先完整读本章文本再核对源码／權威资料','read_order':read_order,'sections':sections,'files':file_integrity,'figures':figs,'code_sources':code_sources,'authorities':authorities,'findings':findings,'checks':{'command':'.venv/bin/python outputs/natural-v4/wholebook-review/13-16/numeric-checks.py','execution_exit_code':0,'python':'3.13.5','torch':'2.14.1+cpu','device':'cpu','original_python_fences_executed':['13.3','13.6','13.13','13.15','15.7','16.3','16.4','16.6','16.8','16.9'],'stdout':'outputs/natural-v4/wholebook-review/13-16/numeric-checks.stdout.txt','stderr':'outputs/natural-v4/wholebook-review/13-16/numeric-checks.stderr.txt','manifest':'outputs/natural-v4/wholebook-review/13-16/numeric-checks-manifest.json','result':'十段均完成，输出和已存在assert与正文相符；未把此抽核称为全notebookrun','figure_rendering':'convert首试因rsvg-convert缺失失败；使用已有Inkscape1.4后16图均exit0并实际view_image','authority_fetch':'9个有效来源均HTTP200；GradScaler猜测generated API页HTTP404后改官方commit源文件取證'},'unrun_and_limits':['未执行整套notebook（由root统一执行）。','未重跑L4训练、CUDA Flash/AMP/compile性能，正文实报数值不是此次重新实测。','只执行指定十段CPU原码；其余公式／代码逐行理解与手算，不能声称均已run。','未完整重审实验JSON与每题生成；完整报告链接被当作证据入口而非此次全验收。','SVG静态渲染核對了内容与布局，未验证网页动画、手机宽度、站点markdown渲染。','未读第20章及其写作／训练，不能作为wholebookfinal。','前置章節由明列链接确认其角色，本次未重复全文阅读13–16之外章节。'],'mutation_scope':'只新增outputs/natural-v4/wholebook-review/13-16中的独立证据；未改canonicalbook、baselinecheckers、reviews或timingmetadata','unchanged_book_and_figures':all(x['unchanged_bytes'] for x in file_integrity) and all(f['unchanged_since_capture'] for f in figs)}
(out/'inspection.json').write_text(json.dumps(inspection,ensure_ascii=False,indent=2)+'\n')
rows=['| 節ID | 我理解到的概念與問題 |','| --- | --- |']+[f'| {s["id"]} | {s["understanding_and_issue"]} |' for s in sections]
parts=['# 第13–16章独立逐节预检\n',
'实际任务身份：`/root/v4_wholebook_scan_13_16`。本报告是整本教材检查中的既有章节切片；不涉及仍在写作／训练的第20章，也不是整本最终验收。\n',
'依序完整读取第13–16章全部47节，读法模拟数学不错的高中生／有基本数学的大学生初次接触LLM；不是假称真人学生调查。先读教材，再检查本章直接依赖源码与重要原始资料，未读取旧review verdict或作者历史来补文本。\n',
'主线机制与小例子多数能接上，未发现确定的核心公式错误。必须先处理第13章分流路线上13.5的前置冲突；另有一处正規化圖標籤不準、少數修訂語句，以及正文run方法與統計過密的學習負擔。\n',
'## 可执行建议\n']
for f in findings:
 parts += [f'### {f["priority"]}：{f["id"]}\n',f'位置：`{f["source_location"]}`。\n',f'{f["problem"]}\n',f'建议：{f["recommended_change"]}\n',f'验收：{f["acceptance"]}\n']
parts += ['## 每节实际阅读记录\n','\n'.join(rows)+'\n','## 图、来源与实际核对\n',
'16张实际SVG全部用现有Inkscape 1.4渲染，实际查看四张contact sheet，并对照图内数字、矩阵、箭头与正文。DPO梯度、PPO两侧平台、旋轉分量、dispatch回归、GQA连线、packing／causal布尔矩阵、online softmax、Flash灰基线图均相符；normalization图的问题已列出。源SVG的完整SHA、关联节ID和解读在inspection.json。第一次ImageMagick依赖rsvg-convert的渲染失败，改用已有Inkscape后全部成功，沒有把失败当完成。\n',
'实际运行10段原文CPU Python fence：13.3、13.6、13.13、13.15、15.7、16.3、16.4、16.6、16.8、16.9。Python 3.13.5／PyTorch 2.14.1+cpu，退出0。PPO目标及导数为`[0.7,1,1.2,-0.8,-1,-1.3]`／`[-1,-1,0,0,1,1]`；top-1 gate梯度为约`[1.1852,-1.1852]`或重算后的零；KV储存384／96bytes；大批與正確累積9.3333、错平均7.5；online结果24.2857。实际stdout、stderr、执行原文fence的SHA分别保存，不能把这十段抽核说成全部notebookrun。\n',
'实际核对的權威来源（下载原件、抽取文本、URL与SHA已保存sources/receipts.json）：\n']
for a in authorities:parts += [f'- [{a["source"]}]({a["requested_url"]})：{a["actually_read"]}。{a["conclusion"]}']
parts += ['\n\n本地源码核对了DPO、RM／PPO、MoEFFN gate与dispatch、attention cache／mask和模型默认形状；architecture.py的有效输入forward hook确实在正式实验中重算不含PAD的辅助项，因此没有把stock MoEFFN内部未收PAD mask误报为实报错误。相关文件SHA和读取范围在inspection.json。\n',
'## 限制与不变证据\n',
'未重新执行L4训练、CUDA后端、AMP／compile性能，也未完整重审每个实验JSON；正文旧实测数字没有冒称是本次实验。完整notebook由root统一执行。图是静态render，尚未验证动画、手机或整站渲染。没有重复全文阅读前置章节，也没有读取第20章。\n',
'初读的四章与16图原始完整字节已保存在original-sources，全部核对初始SHA吻合。root收到发现后授权修改13章导读／13.15／13.17、14.2与normalization图；因此最终canonical与初读SHA并非全部相同，详见下面closure。此review agent只写指定outputs目录，没有修改教材、基线检查器、既有review或阅读时间元数据。fullfile／section／figure／code／authority原始SHA均在inspection.json及initial-manifest.json中，hash仅用于版本锚定，不代替上述完整阅读。\n']
(out/'report.md').write_text('\n'.join(parts)+'\n')
print('sections',len(sections),'figures',len(figs),'authorities',len(authorities),'findings',len(findings),'unchanged',inspection['unchanged_book_and_figures'])
print('report',out/'report.md');print('inspection',out/'inspection.json')
