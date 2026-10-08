"""Serialize this reviewer's actual independent technical checks and limits."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent.parent
E = R / "technical-extensions-evidence"
manifest = json.loads((R / "manifest.json").read_text())
inventory = {x["page_id"]: x for x in manifest["inventory"]["pages"]}
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

# Each row is individually authored, not a shared chapter-pass template.
# page | exact quote | source judgment | mechanism | need/use | actual checks | not checked
authored = """
chapter-0A|各段短程式把輸入和中途資料展開，模型的實際回答另查。|導讀将提示示例、檢索、來源核對、長度與長對話更新分工，未把短程式當模型能力證據。|例子與公告進入當前提示，不等於改權重。|提供本次規則或新資料，後續各站分開核是否真正使用。|全文與A.1–A.10對照：預告的每一項都有對應小節，章前言沒有額外訓練承諾。|導讀不是額外模型實測。
A.1|例子和問題一起成為本次輸入，權重沒有因此更新。|示例、shot計數、固定權重比較和SFT區別正確；人預期與模型實答分開。|join只改本次提示例子；固定問題、權重及生成設定再改例或順序。|讓同一問題按本次規則解讀，查前文介入而非持久更新。|stdlib三提示實際執行；rag同權重ICL四筆核ID與EOS：zero UNKNOWN、few-shot預期5實9、改例預期8實9、反序預期5實8。|沒有重訓或重新生成；未測一般模型的上下文學習。
A.2|檢索負責找資料，生成負責讀資料回答；來源編號用來回查。|RAG分工、地址欄位、引用與紙上答案邊界正確；配對實測數字吻合。|先找相關公告再放提示；只换地址或來源ID分別對照。|提供新資料並回查根據，查是否跟新欄位或引用變化。|stdlib公告例執行；全部84 RAG行核輸入/生成ID/EOS及指标：correct5/12、換地址7/12但雙對5、換來源5/12但雙對2。flow圖640/360分工正確。|沒有重訓生成；未測自然公告、衝突版本或注入。
A.3|每項貢獻取查詢和文件次數較小者，再加總；查詢的書只一次，文件重複十次也只得1。|詞片段計數分數、零分排除、k上限、id同分排序吻合實作；同義失敗有機制解釋。|中文單字/ASCII連續詞計次，共同片段min相加排序非零前k。|用可逐項追蹤的找法取來源，不把分數當答案正確機率。|frozen retrieval.py全文與stdlib執行：書店地址取d2，營業處在哪回[]；RAG72文件12事實+60雜訊，原實測9/12命中、3別名未中。|未實作或比較向量/語義檢索。
A.4|編號真的存在，內容仍沒得到支持。|引用存在與主張支持分開，substring只限肯定地址窄例，否定/同義限制明示。|先查ID，再查同來源是否支持主張。|避免有效編號與正確格式冒充事實根據。|stdlib青街True/True、紅街True/False；84 RAG逐行重算，正確+干擾來源引用有效12/12但支持2/12，未混兩層。|未建一般自然語言蕴涵或否定驗證器。
A.5|不能當絕對因果證明。|兩站分母、未找到/讀錯定位及補公告介入的因果限定正確。|每題保存必需來源、實取ID和最終回答，補正確來源時記所有聯動條件。|決定先查檢索或提示/閱讀，避免單次介入過度歸因。|stdlib三人工行來源2/3答案1/3；RAG retrieved12行核：來源9/12、地址3/12、含UNKNOWN協定6/12。|未重跑補公告介入或估普遍因果效應。
A.6|這是程式的固定規則，沒有教模型誠實。|空檢索、相關來源缺欄位與衝突分開；非空不等於可答，拒答率不等於恰當拒答。|查取回來源與資訊是否提供，冲突再查版本。|缺資料要指出需補何種來源，不編地址；詞分數不能判最新事實。|stdlib[]與d2兩狀態執行；RAG無context UNKNOWN12/12，但correct context仍UNKNOWN7/12，原逐題分母相符。|未測模型誠實、校準或衝突公告；處理原則未冒稱實測。
A.7|先把系統、歷史、問題與公告序列化成最終輸入，再數ID。|P+G≤C、输入105自身超限與輸出預留關係正確，非中文字數。|全部role/分隔/內容序列化計P再預留G，輸入過長先縮輸入。|留回答空間且不漏仍有效條件/引用，少答不能救已超限輸入。|stdlib75+25=100、增加歷史30后130超30；84 RAG输入58–94加預留16均在max160。|未執行一般裁切/摘要策略；算例未宣稱已裁歷史。
A.8|U1 到 U7 的相對順序保持不變，沒有新增公告、改寫線索或另換問題。|位置介入可追踪，資料段T在1/4/8而問題不動，其餘順序與token總量固定；圖非實測。|只搬同一T，固定其餘材料/問題/權重/解碼/上限，先核實際token一致。|查放得下的線索是否穩定被使用，隔離長度和內容改變。|clue-position圖640/360逐列核T/M17與U次序；Liu原文§2.1–2.3及§3.1–3.2指定段支持QA文件重排/位置測試用途。|未跑三排序模型測試或benchmark；不聲稱全論文已讀。
A.9|材料相同，三種問題需要用到資訊的方式卻不同。|單筆、多筆、M17連接與摘要的成功條件具體分開；樓層由U3而非代碼猜得。|單筆查T、多筆按T/U1/U2列全，T的M17接U3同鍵取2F。|單個needle命中不能證列全或分散關係連接；摘要另需整合取捨。|linked-records圖640/360與紙上M17,B04,C26、T→M17→U3→2F吻合；RULER§3.1–3.4、LongBench§3.2.1原段支持任務區分。|未生成單/多筆、join或摘要；未跑RULER/LongBench。
A.10|只改時間，就不連帶改掉活動日期與地點。|更新按欄位而非最後整訊息覆蓋；訊息日期與活動日分開，16:00/M17/兩條可共存。|明確替換欄位取最新，保留未改欄位，對不存在/缺失資訊另拒答。|長對話更新時間不能抹地點，單提醒成功不等于全面記憶。|conversation-updates640/360核10/1原15:00,M17、10/3只16、10/4只格式，活動仍10/10；LongMemEval§3.1–3.4五類能力原段吻合。|未生成提醒，未建或驗長期記憶系統。
chapter-0B|選卡與整個往返分開驗收。|導讀把請求、驗證、執行、回填、完成與選卡分開，两模型未拼端到端分數。|先查工具JSON往返，再另教自然問題动作卡。|會依請求算不等于会判断何時算，各需自己的證據。|全文對照B.1–B.8：前半真正執行，後半選卡；B.7明说不能拼96/96與19/20。|導讀非額外實測。
B.1|JSON是資料格式，寫出它不會讓Python自行執行。|名稱/參數、人工訓練例、序列化與真執行边界正確，合法參數亦可能抄錯題。|生成可解析名称與參數，外層才依規約運算。|精確運算並回查原題/請求，定位格式或參數錯誤。|stdlibJSON例只印請求，123×45=5535、×54=6642核算；原工具20題按原問題比參數，末一失败在最終非法doneJSON。|沒有重訓自然語言到工具參數模型；短例人填。
B.2|解析是讀結構，驗證是查允許內容；格式完整的delete_all仍不在本例工具清單裏。|allowlist/schema/exact數字型別與非有限拒绝吻合，運算overflow保存已執行而失败。|JSON解析後查名稱、欄位、exact int/float、有限輸入，再只call允許函數。|完整格式不是允許操作，bool/字串不能混數字，也不eval模型字串。|retrieval全读與applications解析/執行段；stdlibadd5、delete_all/bool拒绝；原7人工fault checks独立于20生成題，NaN/Inf/重複keys/1e309拒绝。Toolformer摘要/引言读API用途。|未跑Torch整套protocol tests，未測外部副作用工具；未重做Toolformer訓練。
B.3|這份按順序留下的紀錄叫trace。|正文真实结果回填与tool/user表示正确；旧原始generation.messages有TEX-N1時序缺陷，保留input_ids仍支持真流程。|call_tool後以user TOOL_RESULT回填再生成；外層tool紀錄不等于render_chat支持role。|讓最終回答讀真返回，避免人工gold替工具，避免不支持的角色模板。|stdlib真算5535、tool-flow640/360吻合；56生成/36执行原始IDs/EOS/返回值核對，36首步messages含未来回填；56步逐ID歷史重建可不重訓。|沒有重新取樣；不能因分數正确宣称所有trace metadata忠實。
B.4|四份run的status依次是done、step_limit、needs_more_model_output、invalid_request；真工具次數1、1、1、0。|動作/實際工具次數/完成/答案/等待分開；正常19/20、capped18/18与人工故障不混分母；metadata仍TEX-N1。|max_steps限模型动作，工具請求與done都占动作；真結果尚需合法最後回答。|避免算完或没完成宣告被外層默認成功，定位停限/非法请求。|stdlib四run吻合；20normal19correct、18calls；18capped每次1call但step_limit；multiply9,9返回81而done生成answer8加多餘}，正確算失败。56步快照核ID。|未量完整服務延遲，原生成秒數未重测；未重訓。
B.5|這是教材選擇的習慣，不是所有小加法永遠只能用工具。|人工策略是規範而非算術必需性，狀態改後gold需改；ASK原因不冒作模型解釋。|依任務/可用狀態标DIRECT/TOOL/ASK。|同樣有數字不能決定工具，先可核對規則才监督標註。|stdlib五例TOOL,DIRECT,DIRECT,ASK,ASK；tool_choice.py全讀：正式策略进labels、system只狀態，与B.6教學完整system有別。|未測零單價等報價特例或ASK完整解释生成。
B.6|TOOL四個字母加EOS，共五個有效目標。|assistant目标、UTF8byte、EOS與-100 loss mask關係正确；教学system策略不混正式只状态。|render_chat作X/labels，assistant與EOS監督，其他位置-100仍在輸入。|学選卡又能读system/user，避免把问题當答案或将lossmask當遮蔽。|data.py全文與手核TOOL IDs[92,87,87,84,2]；7.2/7.9前置全文；tool_choice原900步、80val/96test/48diagnostic生成与固定規約對照。|Torch未安裝，未执行tensor cell或900步訓練；ID為手核和静态关系。
B.7|選對卡也沒有實際運算，真呼叫與最後答案另計。|分项分母和问法診斷正確：96/96与42/48未掩蓋新工具題0/6，也未拼另一模型端到端。|unordered数字对先分家族，再混3卡與狀態分項評；新問法另診斷。|防同素材改问法漏測，總分不能掩盖漏/多用與停用错误。|80val+96test+48diagnostic全部重算：test96/96,need12/12,extra0/84,unavailable0/48；diag42/48,need0/6全ASK,extra0/42,unavailable0/24；alwaysDIRECT48/96,24/48，TOOL12/96,6/48；55家族44/5/6，原16/診8筆。stdlib3/6漏1/2多1/4。|仅卡片模型，未驗其JSON參數、执行或最終答案，也未推任意問法泛化。
B.8|點數是人為尺度，不是秒或美元。|期望损失公式和假設數字正确；全往返成功率不是函数算对或next-token概率。|平均損失=(1-success)×错答代價+额外成本同尺度比較。|完整成功率與人定成本一起决定是否值得等工具。|stdlibdirect1.0/tool.12选TOOL，direct.999時.01可DIRECT；原工具81但回答8支持全链要求。|没有估置信度、等待、美元或訓練選卡；數值人設。
chapter-0C|它們可以按用途選用，不是每位助理都得把全部方法串完。|導讀區分步骤、候選、選擇和更新，不承諾全套串完必須或更優。|按任务选回答形式、多候选、验证或参数更新。|建立不同干预边界，后续分别看成效与成本。|全文對C.1–C.7：C.1–C.5主要回答/选择/预算，C.6评分，C.7更新；分界一致。|導讀沒有額外模型實測。
C.1|這個差別不能說成等計算量比較。|可見步骤用途/风险、UTF8长度和相同步数不等有效token預算正确。|寫中間过程讓人/後續生成讀，比较另記共同起點與有效目标量。|可核方法但不保可靠或推理忠實，900步不等计算预算。|1与47byte核算不含role/EOS；reasoning720候選查，同base hash、900步各，direct48803/steps466322有效token比9.555≈9.6，k1 correct1/24与11/24。|未重訓生成或做等token/FLOP額外消融；未執行Torch tokenizer cell。
C.2|還要分開查每條等式、前後是否接續，以及是不是在解原題。|三條件分開，短程式未查原题明示；正式窄驗证器含原参数。|核每式、接續claimed，再核原任務a/b/c與最終truth。|結尾對或內部接上不能救错步骤/換題，限定可程序核的算术。|stdlib原False/True、接續True、结尾True；claimed改4则式全True接續False。applications _verify_reasoning静态和720原候選格式/連接/任务旗標回算。|没有通用文字證明器或内部推理忠實驗證。
C.3|候選集合命中稱coverage；pass@k也描述k份至少一份通過，但要依具體採樣與估計方法說明。|獨立同分布1-(1-p)^n正确；平均p、真估计、各预算新采樣与非单调有限結果分開。|全失敗補集算同題n獨立候選包含正确機會，真实每预算分別取新樣本。|多試找到不等最後選對；不同題p不能用平均值代。|stdlib.3,.51,.7599,.942352；两个模型8預算720候選：direct覆盖1,1,2,1；steps11,13,13,14，每分母24。Chen§2.1式1 n=k退成有無通過0/1，与direct coverage相符。|没有证明实际生成普遍獨立，未重取樣；只讀原論文指定段。
C.4|正解已經在集合，多數決仍會選3。|coverage與交付正解分開，精確算术验证强条件不推未知事实。|同集合Counter多數/首個算術通过，無通过None；tie依出现順序。|单独查選擇器能否交出已找到正解，不把多數當理解真值。|stdlib3,3,4選3错驗4對；candidate-choice640/360同集合；reasoning steps k8覆盖14/24、多數12/24、驗14/24，条件多數12/14另分母，逐集合回算。|未測开放事實/地址驗證器；窄truth不是一般選擇能力。
C.5|只量生成的一段不能冒充完整服務延遲。|token预算、程序CPU、训练/冷啟動/完整等待分開；85/100是纸上，批次生成秒不冒全服務。|多候选拆每份预算，另计验证选择；固定权重改使用流程。|比较最终正确/集合命中/选错与全成本，候选非免费，與C.7更新不同。|stdlib80+5=85、4×20+4×5=100；原direct k1/k8 token51/400 seconds.238729/.119536；steps512/4132 seconds1.160041/1.189938，仅批次生成。|未量完整服务延迟、CPU验证成本、冷启动或FLOP；未重新计时。
C.6|reward只是表示符合要求的數字，目前還沒有改權重。|whole-string整数解析、內容/格式、弱奖励枚举漏洞與24×16分母正确。|fullmatch -?[0-9]+后比truth给0/1；弱any-number代理分另查strict。|固定可验格式避免偷抽正確數，合用途规约与新题防奖励投机。|stdlib3/4/5/答案是4 reward0/1/0/0；两有限策略after各384动作逐解析：weak384枚举/代理384但strict0，strict greedy0/24、sample0/384；24不同題非384新題。|有限規約不是自然解释评分，未做语言模型RL训练。
C.7|reward與baseline是固定數字，沒有對驗證器求導。|梯度/advantage/baseline/action/更新边界正确，正式17动作策略与LM分开且新題失敗如實；RLVR全名可選TEX-O1。|loss=-(r-b)logp(action)调两logits；正式18维one-hot→17动作采样REINFORCE+AdamW更新。|正/负advantage提高/降低本次概率，参考不是truth；真更新需新题验证，不等LM token RL。|解析梯度[.25,-.25]，步幅.1更新[-.025,.025]概率[.4875026035,.5124973965]图640/360吻合；r0方向反轉。原1200×64=76800训练动作，两同起点策略新24题全不strict correct，weak枚举。|Torch未安裝，tensor cell未执行，数值独立解析核算；无策略/LM重訓。
training|未量過的欄位就填未量測。|T.1–T.11入口的任务/数据/更新/留出和比較边界成立，静态CLI/格式匹配；不是所有完整配方本轮实机通过，T.11旧tool记录归TEX-N1。|家族split、dry-run无step、共同checkpoint、SFT目标、冻结encoder/参考、单条件架构、真packed量化、同起点CE/KL与完整记录分工。|防泄漏或把梯度當学习；把排序/生成、储存/速度、尺寸/蒸馏收益分别验收，排版不必重訓。|全547行分段已讀；train/evaluate/train_simple/prepare_data/training/pretrain_encoders/infer_modal静态：--train门控step；12家族9/1/2和45/5/10，eval原IDs/EOS/skipped。quantize/quantization全讀：int4两码/byte、int8、scale保留，forward反量化，不能量化resume。T.11人工80000/60000与4/5→3/5非实测。|未运行长训练或全11节所有链接的旧JSON、所有CLI/LoRA/DPO/KL/QAT；未执行Torch代码/CUDA/音频/真实图片能力；v2全stage运行未驗。
readme|Colab 雲端執行尚未實機驗證。|首页入口、CPU短例/完整模型路线和限制清楚，未把Colab链接或历史weights当新能力。|同源网页/Notebook，按需要安裝CPU小例或取指定完整模型。|先阅读或短练习再按用途下载，安装/运作不代模型能力。|全文与pyproject群组/build_course bootstrap静态；字元ID图640/360顺序吻合；原清单30组120.pt、旧整合11.pt即8+3。|未uv安装/Jupyter/Colab或跑README全部流程，未取历史权重。
environment|GPU可用、數值格式可用，以及整個任務能完成，也是三個不同的檢查。|uv小模型与Python3.12自然专用环境分开，extras互斥和uv run一致原则正确；一次Linux不推全部硬件，v2有限feature非Whisper。|CPU/cu126/cu130与驱动搭配；小模型feature与Qwen/Whisper原processor分开。|防切换套件，不把版本当快慢；区分文件可读、ASR两站和有限从零主题。|pyproject全讀requires-python>=3.11、3显式index互斥、selftrained .8.0；natural public-release只查模型/版本：torch2.8、固定Qwen/Whisper、selected base。v2 scope三语音主题、12字/单ROI1–4/SansSerif训练见过。|未check_env/uv/natural安装，未复验OS/MPS/CUDA/最低内存；8chat+1history未重跑。
asset-storage|下載核對成功只證明取得相同檔案。|Git/LFS pointer实体/HF推论与续训分工正确；三数据范围byte一致，文件/题/素材未混。|Git文字清单，LFS固定二进制，HF指定推论配套；optimizer/RNG/sampler为精确续训另存。|读者只取所需并核版本，LFS非压缩；LoRA需base，下载正确不等能力。|原manifest基础8包30,854,837 bytes；v2压缩67,862,431、12JSONL+8950=8962；natural3包69,308,918+59,746,570+17,487,821=146,543,309，解包160,683,823。checkpoint全读含训练状态，公开4套×4件承诺与固定rev吻合。|未下载LFS/HF逐byte核SHA或加载；未精确续训，未核当前存储政策额度。
curriculum|這些是局部機制和歷史輔助證據，不是v2最後測試。|教学取舍与三个实证范围分开，20+3章、287lessons与313编号正确；大参数/未达标、旧分数不混比赛。|图由source→render→X/Y→visibility/label→backward/step，梯度累积后才step；从零与成熟模型分路。|表示/注意力可见/监督目标别混，局部机制不能替最终成品；小例尺度不逼全算法。|learning-flow640/360/数据实现吻合；inventory287+R4+W7+T11+G4=313。v2参数5447107/2288067、28876/2435/3734、acceptance false；旧joint/dpo78/90、student62/90,61/90、120+11.pt一致。flash原[2,4,512,32] fp16/bf16强制FLASH/3warmup9测，仅此配置成功。|外部参考项目/公开学习者反馈未逐一查；v2完成4000/10000步运行/选版权威review被隔离未驗；未跑Flash或真人学习效果。
validation|這些是AI審閱，不是實測真人學生的學習成效。|检查层级、执行/质量/hash、validation选版/final冻结分界正确；v2原始汇总支持任务数字，旧integrity review未作依据。|证据只回答对应检查；先validation选再冻结final；ASR真文字/来源文字两站对照，GPU中途回取对照。|文件一致与能跑不代学会，示范成功不改未知率，听错/回错与传输/能力各分。|原v2每版3734，MoE工具0/276、voice42/90、continuation26/60、clothing357/360、OCR252/324及双acceptance false；gpu-smoke原31584参数80步从40接80、maxdiff0支持那次小模型续训。|未逐3734生成审或加载权重重算maxdiff；未读旧review-final/integrity结论，未跑全checks/pytest/kernel/GPU/CPU示范/真人学习或阅读时间。
training-assets|此處發布8個分來源資料包|8包是pilot资料非训练成果，1447/29.4MiB和用途许可对应manifest；all不混v2/natural。|固定archives+逐文件SHA、按来源分包，缓存解包本机并保授权/选样。|取得同材料不等学会；非商用不套repo MIT，题数不等自然任务规模。|8条manifest训练512+365+100+100+100+50+200+20=1447，总30,854,837≈29.4MiB，许可对应表；fetch/assets全读all枚举基础8，SHA/安全解包。FSDD8k需明示resample，Fashion50pilot，GSM8K非教师生成。|未下载解包逐档SHA或完整源集license法律审查；仅查清单和下载机制。
publishing|網站的Colab連結指向發布commit中的Notebook。|正文同源、固定Notebook/默认bootstrap、commit版本标记和资料/权重/网站三路分开；静态所读转换段吻合，不声称本轮发布成功。|build_course产Notebook，export_course配执行输出后建站；先commit再revision，HTTP测试，v2HF固定另发。|同源阅读/改代码，标记不能保存未commit内容；文字版次不改模型bytes或要求全书重训。|build_course1–63/bootstrap默认branch与export_course96–183图/details/输出核对，pyproject Zensical .0.67群组存在；v2固定979cdf…与四套16配套同资产页。|未run build/export/check_site/Pages/浏览实际布局，未跑旧review检查器；hashgate状态不作教材缺陷。
""".strip()
pages = []
for row in authored.splitlines():
    pid, quote, judgment, mechanism, use, check, unknown = row.split("|")
    source = R / "freeze/sources" / (pid + ".md")
    text = source.read_text()
    assert quote in text, (pid, quote)
    assert sha(source) == inventory[pid]["source_sha256"], pid
    pages.append({
        "page_id": pid, "title": inventory[pid]["title"], "source_sha256": sha(source),
        "source": str(source), "source_judgment": judgment,
        "quoted_basis": [{"quote": quote, "line": text[:text.index(quote)].count("\n") + 1}],
        "actual_read": {"lines": [1, len(text.splitlines())], "scope": "full source including details/code; technical round, not blind first-read"},
        "methods": [{"mechanism": mechanism, "need_or_use": use}],
        "checks": [{"actual_scope_and_result": check, "evidence": "own evidence registry below; frozen implementation and assigned source"}],
        "figures": list(inventory[pid]["figures_sha256"]),
        "unverified": [unknown], "issue_ids": [],
    })
assert [p["page_id"] for p in pages] == manifest["groups"]["extensions"]["pages"]
assert len(pages) == len({p["page_id"] for p in pages}) == 36
for p in pages:
    if p["page_id"] in ("B.3", "B.4", "training"):
        p["issue_ids"].append("TEX-N1")
    if p["page_id"] == "C.7":
        p["issue_ids"].append("TEX-O1")

issue = {
    "id": "TEX-N1", "severity": "burden", "required": True,
    "affected_pages": ["B.3", "B.4", "training"], "kind": "historical raw evidence step-input metadata defect",
    "original_quotes": [
        {"page": "B.3", "quote": "這份按順序留下的紀錄叫trace。"},
        {"page": "B.4", "quote": "完整紀錄見[實測報告]"},
        {"page": "training", "quote": "RAG、工具與推理報告保存各自的原始輸入、輸出及分母。"}
    ],
    "already_taught": "B.3已教真工具返回、外层trace及user TOOL_RESULT回填；B.4已教动作/执行/完成；T.11要求原始输入输出。正文机制、展示轨迹与19/20并没有错误。",
    "missing_minimum_relation": "每次generation.messages必须是该次实际input_ids对应的当时输入快照，不得包含该步生成之后才得到的assistant请求或工具返回。",
    "current_effect": "读messages复查第一步会以为模型已经见到自己的请求与工具结果，错误重放输入/判断时序。保留input_ids显示真正先请求再执行，不是实际答案泄漏，不推翻19/20。",
    "discovery_timing": "全部36页及图已读后，第一次逐generation从messages重建input_ids断言失败发现；随后只回查_sample与_tool_episode，未看其他review或旧结论。",
    "independent_source": {
        "raw": "docs/course-experiments/results/tools.json at baseline 5224564f52b0bbe04c6958ea2693ad5afd188420",
        "raw_sha256": sha(E / "tools.json"), "historical_revision": "910aebc6419c9fc6217279a27fde9851c5cfad30",
        "implementation": "freeze/implementation/scripts/course_experiments/applications.py:94,570",
        "evidence": ["tool-input-metadata-mismatches.json", "tool-input-snapshot-reconstruction.json", "reconstruct_tool_input_snapshots.py"]
    },
    "concrete_step": {
        "json_pointer": "/results/samples/1/trace/0/generation", "question": "CALC:add(9,9)", "step": 1,
        "stored_messages": [
            {"role": "system", "content": "Reply JSON: tool name+arguments, or done+answer. TOOL_RESULT is external data."},
            {"role": "user", "content": "CALC:add(9,9)"},
            {"role": "assistant", "content": '{"name":"add","arguments":{"a":9,"b":9}}'},
            {"role": "user", "content": "TOOL_RESULT:18"}
        ],
        "actual_input_ids_count": 97, "stored_messages_serialized_count": 155,
        "id_check": "原始97个IDs逐ID等于首两笔system/user序列化加assistant生成起始ID；四笔metadata重建155个ID。完整实际IDs与对应当时两笔消息保存在独立重建补件。",
        "serializer": "BOS1 + role(system7,user3,assistant4) + UTF8bytes+8 + EOS2 each message + trailing assistant4",
        "cause": "_sample返回messages共享可变列表；_tool_episode执行后messages += [assistant request,user TOOL_RESULT]原地扩展，先前generation metadata随之带入后来的内容。"
    },
    "check_scope": "全部20normal与18 one-step-cap episodes；56次generation有36首步错位(normal18+capped18)，其余20次一致。36真实执行返回、全部生成IDs/文字/EOS及分母另核。",
    "minimal_repair": "未来_sample保存copy.deepcopy(messages)。旧JSON和SHA不覆写；另发布历史逐step输入重建补件，从system/user开始只加入此前请求和真实结果，并逐ID要求等于旧input_ids。B.3/B.4报告入口可附一段字段勘误及补件链接。",
    "repair_feasibility_verified": "独立补件已对56次generation全部重建并逐ID一致；原tools.json SHA前后相同。无模型调用、取新样本或重训即可修记录可回查性。",
    "rewrite_scale": "paragraph", "rewrite_scale_note": "局部记录代码/历史证据补件与一句入口；不需整页、章或顺序重写。",
    "unverified": "未修改教材、源码或原历史JSON；重建不是新模型运行证据。"
}
optional = {
    "id": "TEX-O1", "severity": "optional", "required": False, "affected_pages": ["C.7"],
    "original_quotes": [{"page": "C.7", "quote": "可驗真值提供回饋的訓練常稱RLVR，會改權重"}],
    "already_taught": "可验证回馈、会改权重、baseline/advantage及最小更新已教，不懂全名仍可理解本节。",
    "missing_minimum_relation": "无必要机制/用途缺口；首次缩写全名可帮助日后外部搜索。",
    "current_effect": "仅术语查找便利，不阻塞公式、代码或更新边界。",
    "minimal_repair": "可补RLVR英文全名Reinforcement Learning with Verifiable Rewards。",
    "rewrite_scale": "paragraph", "discovery_timing": "C.7源文本阅读提出，未参考其他review。",
    "independent_source": "freeze/sources/C.7.md", "unverified": "不要求新增完整LLM RL训练或算法变体。"
}
for i in (issue, optional):
    for q in i["original_quotes"]:
        assert q["quote"] in (R / "freeze/sources" / (q["page"] + ".md")).read_text(), q

criteria = {p: sha(R / "freeze/criteria" / p) for p in ("SKILL.md", "references/review-protocol.md", "references/calibration.md", "references/project-context.md")}
prereqs = []
for pid, why in [("1.12", "求梯度与更新"), ("2.3", "线性层混合"), ("5.7", "阶段/精确续训及保存"), ("7.2", "X/Y和assistant监督"), ("7.9", "byte与EOS"), ("7.17", "文字与SFT阶段用途"), ("7.19", "生成预算与停止"), ("9.7", "模态入口和冻结")]:
    source = R / "freeze/sources" / (pid + ".md")
    prereqs.append({"page_id": pid, "source_sha256": sha(source), "actual_read": "full text/details", "purpose": why, "visual": "text-only, prerequisite figures not inspected"})
prereqs.append({"page_id": "first-steps", "actual_read": "lines1–100: W.1/W.2 and W.3 beginning; W headers only additionally counted", "purpose": "安装、列表/字段和入口；未冒称W.3–W.7全文", "visual": "not inspected"})
figures = sorted({f for p in pages for f in p["figures"]})
assert len(figures) == 9

evidence_scope = {
    "rag.json": {"checked": "all84 rows, all prompt/generated IDs/EOS/control tokens, all metric flags, paired both-correct5/12 and2/12, same-weightICL4", "unverified": "no new training/sampling; natural facts/injection/conflict/long conversations not tested"},
    "tools.json": {"checked": "20normal+18capped episodes,56generations,36actual executions; calls/params/results/status/EOS; mutablemetadata defect and56-step exact reconstruction", "unverified": "no new model run or full-service latency"},
    "tool_choice.json": {"checked": "all80validation+96test+48diagnostic row classes and denominators; family split and900step fixed recipe from exact script", "unverified": "no complete execution or generated ASK explanation"},
    "reasoning.json": {"checked": "all720LM candidates; canonical code-assisted equation/task verifier flags plus independent coverage/Counter/first-verified/token totals; after finite strategies each384actions; shared base and effective tokens", "unverified": "no model training/weights/new sampling, no general fact verifier/faithfulness"},
    "v2-final-public-results": {"checked": "aggregate data_scope/capacity/per-task metrics,3734 count perarch, acceptance boolean false", "unverified": "not all3734 row generations; linked old integrity-review conclusions deliberately unopened"},
    "packages": {"checked": "base8 counts/bytes/licenses, v2 manifest filecounts/bytes, natural3 bytes/files;120historical.pt+11capstone.pt", "unverified": "actual LFS/HF bytes not downloaded or loaded; full licensing legal review not done"},
    "gpu/flash": {"checked": "original narrowgpu-smoke JSON andflash_probe config/results; static scope accurately attributed", "unverified": "no GPU reexecution or checkpoint-weight comparison"},
    "natural-public-release": {"checked": "selected model/revision/version/file fields only", "unverified": "review/approval conclusions not used; no actual model download"},
    "papers": {"checked": "original PDFs+pdftotext saved/hashes; Liu145–300,348–405; Hsieh181–366; Bai133–255; Wu78–95,245–452; Chen125–167; Toolformer1–66", "unverified": "these line ranges only, not whole papers; no benchmark reruns"}
}
notes = {"reviewer": "/root/technical_extensions", "kind": "own technical per-page work notes formalized from actual checks, not blind-reader chronological traces", "pages": pages, "discovery_notes": [issue["discovery_timing"], "第一次工具输入断言失败继续调查，区分messages错误和input_ids正确，原JSON未覆写。", "全部9图在640和360实际取得画面后下一回合才记判断，未浏览实际网站。"]}
(E / "page-work-notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
files = {}
for p in E.rglob("*"):
    if p.is_file():
        files[str(p.relative_to(E))] = {"sha256": sha(p), "bytes": p.stat().st_size}
report = {
    "schema_version": 1, "kind": "independent frozen-source initial technical diagnosis",
    "reviewer": "/root/technical_extensions", "role": "technical", "group": "extensions",
    "created_at": datetime.now(timezone.utc).isoformat(), "baseline_commit": manifest["baseline_commit"],
    "criteria_sha256": criteria, "actual_prerequisites": prereqs, "pages": pages, "issues": [issue, optional],
    "independence": {"did_not_read": ["reports/", "synthesis/", "other reviewers/reader notes or traces", "old reader/technical/continuity reports or human judgments", "linked review-final/integrity-review conclusions"], "authorized_read": "own frozen pages, needed prerequisites, frozen implementation, directly cited originalJSON/manifests, originalpaper selectedsections", "no_curriculum_mutation": True, "ai_not_human_student_test": True, "not_blind_first_read": True},
    "additional_source_reads": [
        {"source": "docs/natural-assistant/v4/DATA.md at baseline", "scope": "full direct data explanation; copied in own evidence; not its linked review conclusions"},
        {"source": "course.md/glossary.md", "scope": "numberedheadings only to countR4/G4; not full reader judgment"},
    ],
    "coverage": {"assigned": 36, "full_source_reads": 36, "per_page_individual_judgments": 36, "per_page_exact_quotes": 36, "source_hashes_verified": 36, "page_ids": [p["page_id"] for p in pages], "missing": [], "complete": True, "limit": "every page read does not mean every linked artifact/runtime/platform fully tested"},
    "visual_scope": {"unique_figures": 9, "actually_viewed": 9, "sizes": [640, 360], "figures": figures, "method": "view_image actual pixels, next turn interpretation; notSVG-source-only", "judgment": "inputs/outputs, ordering, joins/updates/arrows and paper example boundaries traceable at640/360", "unverified": ["actual desktop/mobile website layout, positions, collapse/navigation/link reachability", "prerequisite figures and all other book figures"]},
    "execution_scope": {
        "actually_executed": ["19pages stdlib authored examples: A1–7,B1–5,B7–8,C2–6", "raw-record IDs/EOS/metrics and listed groupwise reconstruction", "56tool step-input append-only reconstruction without modifying originalJSON", "C1 byte lengths/B6 answer IDs/C7 analytic gradient andprobability", "data/public-weight manifest arithmetic"],
        "not_executed": ["no weights loaded, training or newmodel sampling", "PyTorch notinstalled: tensorcells andtraining commands notrun", "CUDA/MPS/Windows/macOS/Colab/uv/fullNotebook/site/Pages/HF-LFS download", "all3734v2 samples/GPU weight diff/fullT linked original experiments"],
        "historical_versions": "rag/tools/reasoning revision910aebc6419c9fc6217279a27fde9851c5cfad30,seed42,L4,torch2.14.1+cu126; tool_choice revision0c3d474efbde8cb2fb5378cbe08e9cf562be84db,CPU torch2.14.1+cpu,900steps; exact tool_choice script hash preserved and matchesfreeze.",
        "source_hash_scope": "23/24 shared original code hashes matchfreeze; compression.py difference not used forA/B/C claims; no oldhashgate drift treated as curriculum defect.",
        "static_read_scope": ["retrieval.py,data.py full", "applications.py1–951 and1071–1125; GSM8K pilot952–1070 notread", "tool_choice.py full", "train.py1–148,188–395;149–187notfull", "evaluate,prepare_data,train_simple,pretrain_encoders,infer_modal,quantize,quantization,training,fetch_training_assets,assets,pyproject full", "build_course.py1–63;export_course.py96–183only"]
    },
    "raw_evidence_scope": evidence_scope,
    "diagnosis": {"necessary": 1, "optional": 1, "blocker": 0, "required_rewrite_scale": "paragraph", "page_rewrites": 0, "chapter_rewrites": 0, "order_rewrites": 0, "conclusion": "主要机制、示意/生成边界、比较条件与分母成立；必要修补仅旧tool trace输入metadata的局部可回查记录。未来deepcopy与历史不覆写补件足够，不需重训或大改。未执行范围保留未验，不据未验自动判错。"},
    "work_notes": str(E / "page-work-notes.json"), "evidence_root": str(E), "evidence_files": files,
}
target = R / "reports/technical-extensions-initial.json"
target.parent.mkdir(exist_ok=True)
assert not target.exists(), "Do not overwrite sealed initial"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
seal = {"reviewer": report["reviewer"], "role": "technical", "group": "extensions", "sealed_at": datetime.now(timezone.utc).isoformat(), "initial_report": str(target), "sha256": sha(target), "bytes": target.stat().st_size, "coverage": 36, "criteria_sha256": criteria, "notes_sha256": sha(E / "page-work-notes.json"), "reconstruction_sha256": sha(E / "tool-input-snapshot-reconstruction.json"), "independence": "Sealed before cross review; other/old review conclusions not read; retain initial bytes unchanged."}
sealpath = R / "reports/technical-extensions-initial.seal.json"
assert not sealpath.exists()
sealpath.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(seal, ensure_ascii=False, indent=2))
