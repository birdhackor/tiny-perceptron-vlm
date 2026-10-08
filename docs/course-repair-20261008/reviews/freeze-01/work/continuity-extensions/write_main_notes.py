import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
B=Path('docs/course-repair-20261008/reviews/freeze-01')
M=json.loads((B/'manifest.json').read_text()); P={p['page_id']:p for p in M['inventory']['pages']}
now=datetime.now(timezone.utc).isoformat()
N={
'reviewer':'/root/repair_continuity_extensions','role':'獨立銜接／累積負擔 AI 來源審閱，非人類實測','at':now,
'gate':{'path':'checks/technical-source-all-sealed.json','at':json.loads((B/'checks/technical-source-all-sealed.json').read_text())['at'],'meaning':'只核封存 gate，未讀技術判斷'},
'criteria':M['criteria_sha256'],
'scoped_route':{'assigned':M['groups']['extensions']['pages'],'main_only':True,'read_dependencies':['A.1','A.3','B.1','B.2','B.5','B.7','C.5','C.6','7.9','1.12','6.1','7.1','7.2','7.3','13.11','course','first-steps','5.10','13.4','13.5','14.2','15.1','15.4','17.7','18.8','18.9'],'route_rationale':'A/B/C以鄰節銜接必要規則；training依其自行公布的分支操作路線與回讀概念；readme/environment/curriculum/readme-en是導覽與入口，不要求入口推導整門課。僅讀上述 main；未點外部來源、舊審閱或實作。','actual_delivery':'continuity_source.py extensions main；training與長入口輸出先存本人 delivery，再分段逐行實讀，截斷不算已讀'},
'pages':[]}
def page(pid,promise,dependencies,q1,q2m,q2n,q3,q4,unknowns=None,findings=None,burden='未見必須猜規則的斷點；相關前文回讀屬有明確目的的正常複習。'):
 p=P[pid]
 N['pages'].append({'page_id':pid,'source_sha256':p['source_sha256'],'figures_sha256':p['figures_sha256'],'promise':promise,'dependencies':dependencies,'four_questions':{'identity_role':q1,'why_mechanism_property':q2m,'why_need_use':q2n,'operation':q3,'example_and_boundary':q4},'unknowns':unknowns or [],'findings':findings or [],'cumulative_burden':burden})
page('A.2',{'quote':'即使模型訓練時不知道搬遷，只要當前拿到公告，就有資料可用。','level':'解釋外部資料的工作分配；操作程式只展開提示'},
[{'from':'A.1 主文','quote':'例子和問題一起成為本次輸入，權重沒有因此更新。','needed_here':'判公告是本次前文還是永久權重；由提供線索未更新→本頁公告同屬輸入。'},{'from':'A.3 主文（範圍內下一步）','quote':'這個分數能排序，還不是答案正確的機率。','needed_here':'A.2把檢索交給下一節；A.3真的拆字詞取回，且區分選中和可回答。'}],
{'answer':'RAG=檢索增強生成；檢索找相關原文，生成讀資料回答，來源編號回查。非永久學入權重。','evidence':'「檢索負責找資料，生成負責讀資料回答；來源編號用來回查。」','judgment':'已足夠'},
{'answer':'把外部公告放在問題前文，使這次回答能取用地址。','evidence':'prompt帶來源公告1、date和address；SVG「檢索公告1」→「問題＋公告1原文」→期待答案→回查。','judgment':'已足夠'},
{'answer':'原需求是答訓練時不知道的新地址，文件多時先找到相關段。公告已給地址→這次可讀→有回答依據；不是宣稱一定讀對。','evidence':'首段「答案應是青街8號，引用公告1」與「文件多時，先依問題找相關段落」。','judgment':'已足夠'},
{'answer':'從source欄位展開prompt，再讓同模型有/無公告、換地址/編號；紙上答案不是模型生成。','evidence':'「程式插入公告欄位…沒有生成」「直接把source[address]當模型回答，會跳過…閱讀能力」。','judgment':'已足夠'},
{'answer':'date與address不同欄；來源公告1先被檢索再隨問題送入，圖中青街8號〔公告1〕明標紙上期待，不能推實測能力。','evidence':'「日期只是支持版本的另一欄，不應拿來代替地址」「模型可能讀錯欄」。','judgment':'已足夠'})
page('B.3',{'quote':'下一站要把這個真返回值交給模型，再生成最後回答。','level':'回填流程、表示轉換與真實生成範圍'},
[{'from':'B.1','quote':'模型要生成能被程式讀出的名稱與參數，外層再做運算。','needed_here':'請求不是計算；人工multiply例與舊0–9 CALC/COPY模型的範圍不同。'},{'from':'B.2','quote':'解析是讀結構，驗證是查允許內容；…通過才執行。','needed_here':'図第二站驗證與第三站運算各做什麼。'},{'from':'7.1/7.2','quote':'一筆對話是消息清單…role…content；…推論和訓練使用同樣格式。','needed_here':'外層紀錄須轉成模型受訓模板，不能因字典role存在就認為可直接輸入。'}],
{'answer':'trace是順序留下的請求/真結果紀錄；短例assistant/tool兩筆是外層Python表示，正式回填採user文字TOOL_RESULT，模型再回答。','evidence':'「這裏要區分兩種表示」及render_chat只收system/user/assistant。','judgment':'已足夠'},
{'answer':'實際call_tool產生5535，再把數字變成模型可接受的user訊息；模型再生成done+answer，與工具算式分開。','evidence':'SVG五站依序請求、驗證、真執行、user: TOOL_RESULT:5535、done＋answer。','judgment':'已足夠'},
{'answer':'原需求是讓真工具資訊參與最後回答並定位錯站；真值回填可核模型有沒有讀回，保留來源可分模型請求/外程式結果。','evidence':'「請求與結果各有來源」；9×9返回81但生成8而被拒。','judgment':'已足夠'},
{'answer':'人工request→call_tool→result→外層trace；正式流程以user TOOL_RESULT回填，同一模型再生成。改b=46也改assert與trace期待，不引用5535。','evidence':'「這段只建立外層紀錄，沒有最後模型回答」與9+9舊完整軌跡。','judgment':'已足夠'},
{'answer':'短程式支持真呼叫及紀錄；舊9+9例支持一個完整往返，9×9反例限制整題完成。圖multiply最後done框是工作站角色，未標成短例已生成。','evidence':'圖第一框「模型／示例提出請求」；正文明确人工指定與「工具成功還不是任務完成」。','judgment':'已足夠'})
page('B.4',{'quote':'若只容許一個動作，可能已經算完卻來不及再答；若沒有完成訊號，不能由外層默認成功。','level':'停止規則、動作計量与不同結局'},
[{'from':'B.3','quote':'第一…assistant提出要做什麼…工具成功還不是任務完成。','needed_here':'工具執行後還要模型最後回答。'},{'from':'7.9','quote':'EOS是模型選的結束，上限是外部預算。','needed_here':'一次生成結束與整題done是不同動作界線。'}],
{'answer':'done宣告整題完成；EOS只結束一次生成；max_steps是處理動作數，done佔一步。此trace每次運算一筆，與B.3兩筆訊息不同。','evidence':'「trace在此每次運算記一筆，不是上節…兩筆對話」「done也算一個」。','judgment':'已足夠'},
{'answer':'有剩餘動作預算才能在工具後再交done；無done、未知工具與上限耗盡有不同status。','evidence':'四run status done/step_limit/needs_more_model_output/invalid_request，工具次數1/1/1/0。','judgment':'已足夠'},
{'answer':'需求是知道是否真的完成，而非外層默認；保留不同終點可區分算完沒答、資料不足和拒絕。','evidence':'原承諾與「四種結局應保留，不把都叫已回答」。步数从1改2只留回答步，不增工具数。','judgment':'已足夠'},
{'answer':'人工add/done清單單測流程；真模型一動作一生成，回填後再產生下一動作；done answer=6仍需內容評分。','evidence':'「測流程的清單，不是模型一開始就生成了整套計劃」「完整評分核對工具是否符合原題…」。','judgment':'已足夠'},
{'answer':'四份run揭示各停止原因。舊20題18工具+2COPY而19/20完成；一動作18題全step_limit只支持預留回答步。零圖頁靠四行清單、四status與計數已足够追蹤，不須自行想像未展示的循環。','evidence':'本頁四run、max_steps说明与「不是計算器不會算」。','judgment':'已足夠'})
page('B.6',{'quote':'目前卡是人填的；希望模型看到…能生成TOOL。','level':'示範監督位置與選卡模型范围'},
[{'from':'B.5','quote':'精確計算…TOOL；概念解釋或照抄…DIRECT；缺資訊/工具停用…ASK。','needed_here':'標籤依任務與工具狀態，而非見數字就工具；是教材策略。'},{'from':'6.1 / 7.1–7.3 / 7.9','quote':'普通內容每byte一token…Y…-100…user問題仍在X…EOS放進回答尾端…有效下一項目標。','needed_here':'TOOL四ASCII byte+EOS五目標，-100不刪輸入、decode不顯EOS。'},{'from':'B.7（下一步）','quote':'不能把選卡器96/96與另一份JSON模型19/20拼成端到端成功率。','needed_here':'後續驗收承接本頁不同權重与局部能力限制。'}],
{'answer':'DIRECT/TOOL/ASK是逐token生成的動作卡；ByteTokenizer編bytes；labels=-100只排除直接答案代價。選卡訓練是示範標記的监督，非工具執行。','evidence':'正文「TOOL…加EOS，共五個有效目標」「問題仍可讀」。','judgment':'已足夠'},
{'answer':'把assistant TOOL當有效答案，真正訓練更新可調參數提高合適標記概率；自然問題移除CALC/COPY前綴。','evidence':'「沒有更新模型」「避免只學讀標頭」。','judgment':'已足夠'},
{'answer':'需要按問題/狀態選不同路；人填卡變成模型輸出前，須有多類與狀態對照，不能只學標頭或數字線索。','evidence':'首段資料含計算、解釋、照抄、缺資訊和可用/停用；B.5相同問題只換狀態動作变。','judgment':'已足夠'},
{'answer':'system策略狀態、user題、assistant TOOL→render_chat→取有效標籤→decode并查EOS；正式訓練另更新。選TOOL後名稱/參數/執行/回讀另驗。','evidence':'本頁兩段操作說明；7.1建立X/Y與7.3有效位置。','judgment':'已足夠'},
{'answer':'短程式只驗資料五目標，不教成端到端；人工完整規則system與舊訓練仅狀態system明分。隨機另建選卡器不同JSON權重，不夸自我認知。零圖頁 messages與labels配合已讀7章足夠。','evidence':'末段兩種system及「不是同一權重」。','judgment':'已足夠'})
page('C.7',{'quote':'希望提高這個動作的機率，但不是一次變成必選。','level':'策略梯度最小更新、baseline角色與能力界限'},
[{'from':'C.6','quote':'reward只是表示符合要求的數字，目前還沒有改權重。','needed_here':'2+2答3/4的0/1回饋来源与評分漏洞。'},{'from':'1.12 / W.4 / W.6','quote':'新參數=舊參數−學習率×梯度；p越接近1…負log代價越接近0。','needed_here':'loss正/負advantage方向，以及backward與真正更新分開。'},{'from':'13.11','quote':'實際分數減更新前的基準…高於或低於預期，不直接等於答案正誤。','needed_here':'baseline為參考，不是真值；單次整卡與逐token不同。'},{'from':'C.5','quote':'保持模型權重固定，只改作答流程。','needed_here':'RLVR改權重与作答時計算有不同角色。'}],
{'answer':'policy是概率規則，advantage=reward−baseline；最小策略梯度用−advantage×logp(action)。RLVR用可驗真值回饋並更新權重，不等同固定權重多次回答。','evidence':'主文定义与最后RLVR完整中英文；C.5与C.6已讀。','judgment':'已足夠'},
{'answer':'固定action=1答4，advantage+0.5，降低负log代價会增答4概率；梯度±0.25经0.1步后分数±0.025、概率0.487503/0.512497。baseline改变相對參考方向，适当者可减抽样波动，本例未量。','evidence':'正文公式、更新数字与图按答3/答4固定顺序；reward改0给负advantage相反。','judgment':'已足夠'},
{'answer':'原需求是把好于平常參考的選擇变得更常、差于參考变少；C.6提供正确回馈，13.11明确这一需求。減baseline并非提供正確答案，而是决定相對更新。','evidence':'C.7「希望提高…」；13.11「更常作出比預期好的選擇，較少…差的選擇」。','judgment':'已足夠（限方向目的；降波动仅概念限定，非本例测量）'},
{'answer':'分數索引0答3/1答4；人工action固定；log_softmax[action]→backward填.grad→no_grad算updated→重算概率。reward與baseline常量，不對驗證器求導；真策略需先抽樣再更新生成分数的参数。','evidence':'程式及逐动作说明；1.12求导/调參数前文。','judgment':'已足夠'},
{'answer':'圖前後兩数均按答3/答4，分數與概率兩行分清；正負reward小變化可由同规则预测。只支持一個人工动作的方向更新，未訓練語言模型、未抽樣、未量baseline效果或新題能力。','evidence':'正文末段舊策略新題全错/列举，以及「真正更新…不能取代新題能力」。','judgment':'已足夠'})
page('training',{'quote':'這頁帶你準備材料、確認數值通路、更新參數，再用沒拿來教模型的題目檢查。各節是不同主題的操作入口…不必一次跑完所有實驗。','level':'選擇局部實作路線；完整配方另開，非一套順訓共同模型'},
[{'from':'W.1/W.7','quote':'啟用後python指向…工具箱；…起點移到有pyproject.toml的專案根目錄。','needed_here':'python或.venv/bin/python與相對路徑的實際前提、Windows換路徑规则。'},{'from':'5.10','quote':'訓練資料更新權重，驗證資料選設定，測試資料等方法定案才計分。','needed_here':'T.1/T.4家族隔離、T.8/T.9固定validation比較并保留最後test。'},{'from':'7.1/7.3/7.9','quote':'user問題仍在X；有效助手目标及EOS。','needed_here':'T.4对話只计assistant与完成率，非删问题。'},{'from':'13.4/13.5','quote':'參考…原行為的基準，並非真值裁判；候選排序與實際回答還要分開測。','needed_here':'T.7 DPO固定副本身份及排序不保证自由答对。'},{'from':'14.2/15.1/15.4','quote':'RMS…保留共同偏移；Dense…同一組完整FFN；先選擇，再只執行被選的expert。','needed_here':'T.8究竟换了哪個零件及同宽度不等成本。'},{'from':'17.7','quote':'存成整數與用整數乘法是不同的事。','needed_here':'T.9低位元保存仍反量化浮點，模型bytes非运行记忆体。'},{'from':'18.8/18.9','quote':'p是教師比例，q是學生比例…同一詞表；保留真值的要求，又利用教師的候選關係。','needed_here':'T.10 ce/ce_kl分支比較與同位置字表对齐。'}],
{'answer':'頁內分11個操作問題；資料準備、通路、更新與留出驗收各有身份；文字、對話、模態、偏好、架構、量化、蒸餾是獨立路線。','evidence':'開頭「可以選…一條」；T.2「選一條即可」；T.4从同一起點分支；T.10换固定教師。','judgment':'已足夠（各單元见unit_checks）'},
{'answer':'真正更新由--train開啟；前後比較固定题/起点/生成预算；量化改变支援Linear保存格式；DPO、教师KL和MoE改变不同工作。','evidence':'各節明确模式、更新部位、模型来源與量測欄位，非只有命令。','judgment':'已足夠'},
{'answer':'独立需求是能判任务答案并把改动/结果连起来。每条练习用可核對問題、同条件与边界回應；保存小／平均loss低不代答能力。','evidence':'T.1先定输入答案；T.2「不是能力考試」；T.11「改了什麼、哪個條件固定、結果支持哪件事」。','judgment':'已足夠'},
{'answer':'先准备资料/模式，再训练保存并同题评估。T.6需T.4属性权重及两编码器；T.7需T.5 style；T.9需T.4属性；T.10特别需固定sft/style/moe教师model.pt与配对dataset.json，不能替代小练习。','evidence':'各路径与父权重主文明示。T.10目前尚未打开折叠配方，故仅理解依赖，未驗完整操作。','judgment':'主要本機小練習已足夠；T.10固定路線操作待補讀（非已驗可執行）'},
{'answer':'手写A/B同家族与C新家族、AB两位置人工评估表、4/5→3/5量化例均显式示意；实际结果要求读者自行记录而不借旧数字。零图頁用表格、消息/字段、路径与两版本命令足够识别当前材料与输出；模态空间机制的深教不由本页承担。','evidence':'T.8「不是三條命令的實測結果」；T.11「數字不是模型實測」。','judgment':'已足夠'},
unknowns=[{'position':'T.10','quote':'此路線必須先打開並完成T.4…T.5…T.8…三組','state':'纯主文已知道必须另外选完整固定路线，但尚未读该三份折叠配方；不声称固定蒸餾路线可直接照主文完成。','necessity':'對選擇T.10操作者為條件式必要步驟；不影響T.1–T.9本機分支的概念理解。'},{'position':'完整CLI与GPU','state':'未执行训练/数据下载/模型评估；未读实现；参数契约只依当前主文，工程可跑性未驗。'}],
findings=[{'id':'CE-OPT-1','type':'閱讀路徑／正文與選讀','severity':'選讀改善','position':'T.10 → T.4/T.5/T.8固定配方','quote':'此路線必須先打開並完成…三組，不能用前面小練習的…替代。','already_taught':'主文已经把父權重、三種固定教師与六份必须产物說清楚；讀者不用猜模型身份。','missing_or_optional':'被折叠块统一标为补充/選讀，但这三组对T.10操作者已是必要前置；可在块摘要注明对T.10路線必要，减少来回确认。','impact':'仅改善选择完整固定蒸餾路線者辨认必做入口；没有据此要求把完整长训练配方全搬回正文。','judgment':'条件式操作路線已显式授权打開，故不列主要语义缺口； extras后另记是否能定位。'}],
burden='T.4/T.5/T.8小練習与固定完整实验、19章v2与20章成熟环境都有明确版本分界，不能串成單一成果。T.10会回跳三次展开，操作负担存在但来源显式交代必要性；未把这种有意分支算成隐含先备。')
N['pages'][-1]['unit_checks']=[
 {'unit':'T.1','promise_quote':'先說清楚…輸入、要求與答案','mechanism':'A/B相同素材换问法归一家族；C换属性留出','need':'避免把已教素材的另问法算全新素材','operation_boundary':'describe明确约定形状；下载不是训练','judgment':'已足夠'},
 {'unit':'T.2','promise_quote':'确认…有限代價…參數…更新方向；不是能力考試','mechanism':'无--train只forward/backward；grad_norm有限>0','need':'先排除数值通路故障才训练','operation_boundary':'null有效位置数=未提供；>0不是每参数有梯度','judgment':'已足夠'},
 {'unit':'T.3','promise_quote':'先用相同短文與切分…前後代價','mechanism':'bigram一字；MLP context3 width16；--train200真正更新','need':'隔离前后与模型差异；避免只看训练下降','operation_boundary':'9/1/2篇；最后一步更新前loss另名；不能交Transformer infer','judgment':'已足夠'},
 {'unit':'T.4','promise_quote':'文字階段…接續原文；對話階段…助手回答示範','mechanism':'随机start两支；载text接SFT则新stage非resume','need':'同一未训起点及同题比较，不把不同父权重混同','operation_boundary':'NLL/内容ID match/完成率/EOS/逐題/skipped分母；text匹配null；byte上限','judgment':'已足夠'},
 {'unit':'T.5','promise_quote':'把風格、指令遵循與安全拆成不同目標','mechanism':'同题简短/生动保5与一句格式；拒绝/澄清分别示范','need':'风格不能盖错数或误拒；不同标准分项评','operation_boundary':'两独立模型500步，仅有限盒子規則；字符串匹配不足判比喻','judgment':'已足夠'},
 {'unit':'T.6','promise_quote':'入口…學到線索，接頭…送進文字模型','mechanism':'先两编码器再属性模型+接头；freeze projector/partial/none定义不同范围','need':'基礎特徵够用后才检查能否转成语言答案','operation_boundary':'2题编码器holdout=1才续；素材参数制作输入不塞答案；circle/high/square,low是期待非成绩','judgment':'已足夠'},
 {'unit':'T.7','promise_quote':'先檢查回答能力，再學較偏好的回答','mechanism':'同题chosen/rejected与固定起点副本DPO；微调style另产preferred','need':'在相同起点上更偏合适候选，并另核生成及保原能力','operation_boundary':'候选排高非自由生成必对；本机与下载外部偏好完整路线分开','judgment':'已足夠'},
 {'unit':'T.8','promise_quote':'先保存基準，再只換一個零件','mechanism':'相同toy-text seed200；LayerNorm→RMS或FFN→4expert top2','need':'知道差异是否来自那个零件，避免同宽误当同参数/时间','operation_boundary':'NLL低不证明生成/速度好；AB人工表分明；参数/批/有效目标/时间另记','judgment':'已足夠'},
 {'unit':'T.9','promise_quote':'真的縮小保存格式，再量品質','mechanism':'原属性→int4/int8独立转换；支援Linear，码/scale/浮点保留部分计入bytes','need':'保存大小与同题回答两独立需求，不把四舍五入或訓練状态差算收益','operation_boundary':'state中模型张量bytes vs完整file vs运行记忆体/速度；相同row记录增对/增错/不变','judgment':'已足夠'},
 {'unit':'T.10','promise_quote':'两支学生…共用起點、資料與更新安排','mechanism':'较小架构+ce/ce_kl比较；固定sft/style/moe模型與dataset各配对','need':'辨别变小来自架构，教师讯号是否有用来自同条件對照','operation_boundary':'三份折叠配方是显式前置；GSM8K域外诊断非已能力；当前未读配方','judgment':'语义已足够；完整固定操作未验，见unknown/CE-OPT-1'},
 {'unit':'T.11','promise_quote':'記錄要把材料、做法和結果接起來','mechanism':'版本/数据指纹/命令/生成条件/逐题停止/量测范围','need':'隔週仍能判断改什么和支持哪件事；保证分母可查','operation_boundary':'4/5与3/5人工判準；SHA身份非正确性；同4/5可错不同题','judgment':'已足夠'}]
page('readme',{'quote':'從幾個字的編號與接字表開始…逐步解釋文字、圖片與聲音模型的運作。','level':'入口導覽与安装；非架构/训练推导'},
[{'from':'course R.1–R.4','quote':'局部範例各自成立，最終整合專題才追蹤同一個模型如何接續學習。','needed_here':'前章可换模型、第19章追同底座，第20章成熟上游另验。'},{'from':'W.1/W.7 / environment','quote':'啟用後python指向…工具箱；…uv run也帶同一extra。','needed_here':'clone→根目录→套件→kernel→开第一节，两个环境不能混。'}],
{'answer':'文字ID/接字表是课程起点；v2是自训有限成品，20章是上游图文/ASR延伸；Notebook与脚本各有用途。','evidence':'「也不必一路訓練同一個模型」与19/20两段。','judgment':'已足夠（導覽）'},
{'answer':'先小矩阵/短例看運作再复杂模块；安装选cpu或相容GPU；有extra时执行保持同配套。','evidence':'前段/后段角色与安装表、两个PyTorch环境说明。','judgment':'已足夠（路線選擇）'},
{'answer':'入口原需求是理解怎么接下一字并可修改例子；先在线阅读不需装机，想动手再Notebook，完整训练另入口。','evidence':'首段问题、首次1.1、暖身、训练操作链接与「只想閱讀」。','judgment':'已足夠（導覽深度）'},
{'answer':'clone/cd/uv sync/activate/check_env/kernel/jupyter；最小流程无--train不更新；推论下载不能精确resume，自己完整checkpoint各stage接续。','evidence':'安装命令、最小流程解释与模型下载两段。','judgment':'已足夠'},
{'answer':'图原句上排貓看狗，/下排狗看貓。按双向箭头ID3/2/1/4再1/2/3/0查回，不把数大当义大；v2/旧30组/旧整合/20章成品成绩各分开。加载通路不是任务准确率。','evidence':'已看SVG及640/360；图「依原句順序」「編號可查回字，不代表字義大小」；正文「不是本版成品的成績」。','judgment':'已足夠'},unknowns=[{'state':'完整网页中readme图位置及点击未看；Colab未实机、MPS整课未验是来源公开限制，未代验。'}])
page('environment',{'quote':'先選自己要做的事情，再準備對應工具。','level':'選設備/套件及两条成熟/自训路线'},
[{'from':'readme / W.1/W.7','quote':'先clone/cd…啟用.venv；相對路徑从目前資料夾。','needed_here':'本页从已取项目根目录运行，相同extra与绝对解释器规则。'}],
{'answer':'uv管工具，lock/frozen固定配套；PyTorch处理张量，CUDA需驱动/套件/设备相容；.venv与.venv-natural两套角色不同。','evidence':'「兩套程式環境」及2/3節明确工具角色。','judgment':'已足夠'},
{'answer':'按阅读/短例/试v2/正式训练/成熟成品选择不同依赖；cpu/cu126/cu130互斥，uv run重新同步需保持extra。','evidence':'硬體任務表與「uv run會先同步環境」。','judgment':'已足夠'},
{'answer':'需求是完成眼前工作且避免没用的下载/版本衝突；独立环境保各自配套，性能与兼容各验。','evidence':'开头与「兩個工具箱」「cu126/cu130不是慢快按鈕」。','judgment':'已足夠'},
{'answer':'根目录uv sync/run；v2加selftrained；20章专用Python，GPU先nvidia-smi再官方配套；音图片文件处理各模型对应接口。','evidence':'命令和4/5节。','judgment':'已足夠'},
{'answer':'例子是安装操作/用途表；一次LinuxCPU觀察非最低需求，8chat+1history只是选定validation demo的通路，不当未知题能力。声音v2直接特征与WhisperASR明确不同；无图页文字已足够选路线。','evidence':'「不是未知題成功率」「沒有一般逐字聽寫或語音輸出」「LoRA…仍需要完整底座」。','judgment':'已足夠'},unknowns=[{'state':'外部安装链接与硬体最低资源未验；当前角色未安装任何环境，不将來源指令當实机证明。'}])
page('curriculum',{'quote':'這份大綱保留教學設計與研究取捨。…共同基礎：猜下一個字…文字對話。','level':'設計导览、可复查的规则与图示；非原模型/研究报告技术核对'},
[{'from':'course R.2–R.4','quote':'各主題…分開學；最終整合…同一個模型…接續學習。','needed_here':'公开概念依赖和局部独立实验与最终整合边界。'},{'from':'7.1/7.3 与 1.12/W.6','quote':'X…可读问题；Y…有效答案；backward只算…没有调旋钮。','needed_here':'图中X/mask与Y/loss位置，clear/backward/step对象各异。'}],
{'answer':'大纲是设计取舍，路線只维护course一份；模态/行为对齐/架构比较、量化/蒸餾各完成不同任务，别把方法名当统一能力。','evidence':'「本稿區分三類目標」与「縮小部署成本則分兩條路」。','judgment':'已足夠（導覽）'},
{'answer':'从Dense可训/验基準、问答监督，再在已有接口上加模态；现代架构与效率分单项，19章才核同底座。图将数据、预测/计分、一次更新三步分开。','evidence':'排序原则及图三块、相邻「X是…Y是…两种遮罩」「这次不累积旧梯度」。','judgment':'已足夠'},
{'answer':'目标是读者看懂具体变化且能查能力；小任务可控制，已有接口承接新的输入；比较需要匹配预算而不单靠只改一个变量。','evidence':'「容易理解…最高優先度」「只改一個變數…理解機制」「研究方法品質…另宣告匹配」。','judgment':'已足夠'},
{'answer':'读者按course/正文小节，实作选training父权重/步骤；作者维护单源notebook与三轮证据，模型质量验收另阶段。','evidence':'「前置連結…不取代任務、材料與答案」「工程、訓練與能力驗收…另一階段」。','judgment':'已足夠'},
{'answer':'图始样本→角色ID→X/Y；X加可读mask产生预测，Y加计分位置对比;zero_grad清旧梯度→backward新梯度→step改权重。其scope明示一次不累积，不包刻意累积。研究表为导览引用、原实验未全训，旧成绩非v2；图源码与孤立两尺寸可读。','evidence':'已看五张指定图中curriculum原SVG/640/360以及1280/390图裁图；字、框、箭头分工明确。','judgment':'源码语义与孤立图已足够；图裁图浮动界面见unknown'},unknowns=[{'position':'page-captures/curriculum-1280-figure-0.png / curriculum-390-figure-0.png','observation':'桌面工具条压在输入X/下一步答案Y块附近；手机固定栏压在原始样本块。已有源码/孤立图可辨对象，但这两张截图里局部被遮。','state':'尚未实时捲动或点击，无法判断出现频率/可恢复读法；页面图文路徑未驗通过，不自动升为已确定持续必要缺陷。'},{'state':'研究引用、旧模型数字只读本页导览，未点原始模型报告/旧review，事实核对不由本角色接管。'}])
page('readme-en',{'quote':'This from-scratch course starts with character IDs and a next-character table…main course is in Traditional Chinese.','level':'英文導覽和安裝'},
[{'from':'readme / environment / course / W.1','quote':'主文语义对应相同起点、路線和环境边界。','needed_here':'英文不是另一本教材或另一套版本；source已标主课繁中。'}],
{'answer':'同一课程英文入口，起点char IDs、bigram；19章自训v2与20章上游extension分别验收。','evidence':'「neither passed every original capability criterion」「Upstream model capabilities are evaluated separately」。','judgment':'已足夠'},
{'answer':'installation CPU等extra对应驱动；activate或uv run同extra；20章独立2.8.0环境。','evidence':'Setup全段与environment一致；无图也不需要想像空间流程。','judgment':'已足夠'},
{'answer':'入口希望从短语下一char的具体问题带入、可看同节notebook动手；英文读者已知主要正文是繁中，路线明确。','evidence':'首段、main course语言、Start with1.1 / linked warmups。','judgment':'已足夠（導覽）'},
{'answer':'clone到kernel打开notebook；minimum workflow dryrun；localchapter weights加载/执行，完整checkpoint才精确resume；v2自训与成熟LoRA另指南。','evidence':'「no optimizer update or checkpoint write」「cannot reproduce an exact resumed trajectory」。','judgment':'已足夠'},
{'answer':'版本选step1000与完成4000/10000明确，finite任务与3,734每架構分母明确；没有借加载运行宣称准确率。','evidence':'The second command…checks loading and execution; task accuracy…separate held-out evaluation。','judgment':'已足夠'},unknowns=[{'state':'无图；英文入口全页布局和链接操作未验，未对词表计数或模型数据重新审技术。'}])
page('tool-trace-errata',{'quote':'早先步驟的文字欄位可能已包含後來生成的工具請求與工具回傳，不能拿它當作當時模型已看見的內容。','level':'历史证据快照解释与重建边界'},
[{'from':'B.3/B.4','quote':'每一箭頭各做不同工作；真模型…回填後再生成下一個。','needed_here':'后面回填不能算成前一次输入；工具trace与generation输入来源分开。'},{'from':'W.2/7.1/7.2','quote':'清單按順序存；消息role/content转成输入ID；當前尾端只放回答起點。','needed_here':'同一列表继续增加导致早期文字快照污染，但真正input_ids另存可核。'}],
{'answer':'generation.input_ids是实际模型输入；generation.messages是应对应同次输入的文本；旧同一可变列表被后来append改变，重建只用之前事件。','evidence':'首两段字段定义与具体原因。','judgment':'已足夠'},
{'answer':'20正常+18一步共56生成；36 messages有差异；从初始system/user只加之前合法结果重建，56逐ID/总数相同。深层副本保存独立快照，避免后续更改早期记录。','evidence':'「從每題保存的起始…只加入先前…」「日後…深層副本」。','judgment':'已足夠（源文解释，不是独立实跑验证）'},
{'answer':'需要准确判断当时模型看到什么，不把未来请求/工具结果错当早期可用信息；保原档/指纹/补件pointer能定位证据。','evidence':'「不能拿它當作當時模型已看見的內容」「并非重新执行」。','judgment':'已足夠'},
{'answer':'用original_json_pointer找原步、messages_at_generation读重建文；项目根离线脚本仅核指纹每步，补件不同停止。','evidence':'补件字段段与命令说明。','judgment':'已足夠'},
{'answer':'重建不会改生成/工具数值/input_ids/19/20评分，只补文字对照，不是新能力或重跑。零图页借明确字段、前后时序即可理解，不需虚拟完整树状图。','evidence':'「也沒有藉此增添新的模型能力證據」。','judgment':'已足夠'},unknowns=[{'state':'依角色禁令未打开tools.json/重建实现或执行脚本；56/36和逐ID全相同是此页来源主张，独立技术复核未驗。'}])
# Preserve exact source identifiers and a list of what was visually inspected.
for p in N['pages']:
 assert hashlib.sha256(Path(P[p['page_id']]['snapshot']).read_bytes()).hexdigest()==p['source_sha256']
 for name,h in p['figures_sha256'].items():
  assert hashlib.sha256((B/'freeze/original'/name).read_bytes()).hexdigest()==h
N['visual_read']={'assigned_original_svgs_and_isolated_pngs':['rewrite-A-rag-flow','rewrite-B-tool-flow','rewrite-C-policy-update','rewrite-01-character-ids','curriculum_learning_flow'],'sizes_seen':[640,360],'page_capture_images_seen':['C.7-1280-figure-0.png','C.7-390-figure-0.png','curriculum-1280-figure-0.png','curriculum-390-figure-0.png','training-1280.png','training-390.png'],'capture_limits':'training整頁長圖被縮成186×2048及39×2048，只確認閉合details的大致位置，沒有據此判文字可讀。圖裁圖不是親自瀏覽器操作；沒有驗点击、展开、链接滚动或重复遮挡。','dependency_figures':'依賴前文只使用已實讀文字，未声称其余前文图已视验。'}
N['independence']={'peer_reports_read':False,'author_answers_read':False,'old_audits_read':False,'implementation_or_original_model_reports_read':False,'public_source_references':'已读指定公共正文中的历史数字/旧審阅引用，但未点开对应旧档或报告。','extra_sources_read':False,'text_truncation_recovery':'初批criteria/manifest及training出现工具截断，criteria与training随后分段重读；manifest只提取必要组/指纹/inventory，没有把未读整份实现指纹当判断。'}
N['overall_main_judgment']='核心来源語义与已教依赖可接起；无已确定的主要概念阻礙或必要语义缺口。保留T.10条件式必读折叠的选读改善；实际图文路径仍有未验，尤其curriculum浮动界面遮挡不能算已通过。'
path=B/'reports/continuity-extensions-main-notes.json'
path.write_text(json.dumps(N,ensure_ascii=False,indent=2)+'\n')
print(path,hashlib.sha256(path.read_bytes()).hexdigest(), 'pages',len(N['pages']))
