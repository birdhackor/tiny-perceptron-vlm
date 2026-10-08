import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/work/tutorial-audit-20261008')
REPO = Path('/workspace/tiny-perceptron-vlm')
WORK = ROOT / 'work/technical-integration'
manifest = json.loads((ROOT / 'manifest.json').read_text())
inventory = {p['page_id']: p for p in manifest['inventory']['pages']}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def page(pid, quote, judgment, checks, unverified=()):
    p = inventory[pid]
    text = Path(p['snapshot']).read_text()
    assert quote in text, (pid, quote)
    assert digest(p['snapshot']) == p['source_sha256'], pid
    line = text[:text.index(quote)].count('\n') + 1
    return {
        'page_id': pid, 'title': p['title'], 'source': p['source'],
        'source_sha256': p['source_sha256'], 'source_read_scope': '全文，包括details；技術輪，不是盲首讀',
        'source_judgment': judgment,
        'quoted_basis': [{'quote': quote, 'snapshot_line': line}],
        'checks': checks, 'unverified': list(unverified),
        'figures_sha256': p['figures_sha256'],
    }

P = []
P.append(page('chapter-19', '文字核心、三個感知入口及它們的接頭，都由本專案從零訓練，沒有沿用既有預訓練神經權重',
 '導覽把起點、四種輸入與共同文字核心限定清楚；本章的成功例與19.12全卷結論分開，沒有把單一展示當成通用能力。',
 ['E1：隨機模型建構與感知入口實作未呼叫預訓練模型；E2：公開輸出的模型來源為同一MoE safetensors。', '章首只預告，不要求在本頁推導全部訓練演算法。'],
 ['未重做完整訓練，隨機起點的整條來源鏈只核凍結程式與正文連結的紀錄。']))
P.append(page('19.1', '這裡的「先前確認」是公開示範檔提供的對話歷史；最後兩點才是本次模型生成的內容。',
 '公開示範與歷史邊界交代準確，沒有把預先給的確認句偽裝成這輪生成。共同入口的目的與有限成功例相稱。',
 ['E2：raw invocations中system／user／過去assistant含格式歷史，目前兩點答覆在output，EOS為真。', 'V1：共同核心圖四路輸入匯到同一核心，工具的實算位於模型之外。', 'E1：fetch_public_export固定revision、核完整SHA後載入完整張量。'],
 ['未在本輪重新下載公開權重或執行展示命令；安裝命令未在全新環境重跑。']))
P.append(page('19.2', '下面建立與公開成品**相同結構的隨機模型**，只數參數，不訓練也不生成答案。',
 'MoE機制、引入目的及Dense對照有分別說明。總參數與結構活躍參數未冒充實測速度；對照容量差異沒有被隱藏。',
 ['E1：每層4expert、top2加權路由；兩結構共用層寬／注意力／感知結構。', 'X1：MoE總5447107、文字5140224、活躍文字3036928；Dense總2288067、文字／活躍1981184，與正文一致。', '實測兩者感知91907+126829+88147=306883；查字與輸出表確實tied。', '實讀15.13後核比較限定：活躍參數不是FLOPs或latency，並非同總容量勝負試驗。'],
 ['未量GPU稀疏核心速度或等計算預算的訓練對照；正文沒有承諾這兩者。']))
P.append(page('19.3', '先決定家族去哪份資料，再製作不同問法與版本。',
 '來源家族先分、衍生後製與配對同份的原因具體；對話題數與原始素材數分開，錄音不能宣稱跨speaker的限制合理。',
 ['V2：source A及single／pair／swap皆train，B val、C test，配對素材同份。', 'E3：train28876／validation2435／test3734；錄音62／15／30，來源無speaker ID，與正文一致。', 'E1：read_records拒绝group跨split、重複id及不合法message欄位；train_tokenizer只用train文字與預先固定格式字元。', 'E1：prepare_text_tools的tree共組與numeric canonical operand家族先分，(a,b)/(b,a)不跨份。'],
 ['未重建v2全部影像／OCR／語音資料，未獨立讀所有來源資產以排除語意重複。']))
P.append(page('19.4', '如果下一站重新建立隨機文字權重，檔名即使也叫 `model.pt`，仍不是接續同一成品。',
 '跨stage帶權重與同stage精確續跑分清；感知骨幹、head及橋接可更新範圍與實作一致，未把凍結整個入口和凍結head混為一談。',
 ['E1：--init-from載state_dict但建立新optimizer/sampler；resume讀更新器／進度／random/sampler並核設定。', 'E1：set_trainable與PERCEPTION_BRIDGE_PREFIXES涵蓋image symbol/coordinates、OCR projection/position、audio projection/position；joint保留骨幹/head，更新bridge與文字核心。', 'E4：前段stage-index與native selection記錄自身best來源；native完成4000選1000，沒有以最後步等同最佳。', 'V3：lineage圖random到native箭頭及best／fresh optimizer標示對應正文。'],
 ['未執行中斷再resume與未重算完整weight lineage；紀錄的hash不是本轮重新訓練產物。']))
P.append(page('19.5', '這是挑選出的成功驗證例，可以核對介面與生成流程，不能由一題推出所有新問法都成功。',
 'SFT的任務、gold與prompt邊界正確：歷史可讀而只監督assistant/EOS。特定成功例有清楚限定。',
 ['E2：公開純輸入無當前target，兩點App回答在output。', 'E1：dataset.encode_record用ids[:-1]／targets[1:]，user等label=-100，assistant及EOS有label；不截答案。', 'X1：本頁clarification示範decode計分目標為「請說明是地址、App還是卡片的問題。<eos>」。'],
 ['未重新推論完整text heldout或独立核全部語意規則的合理性。']))
P.append(page('19.6', '最終文字仍由文字核心逐字生成；感知分類頭的最大分數預測與最後回答，要各自評分。',
 'image/OCR/direct-audio三入口的表示與任務邊界清楚；ROI供位置不供類別，soft scores不是argmax查固定句，保存真實先前回答與學會使用它分開。',
 ['E1：image兩slots由公開coordinates裁像素；OCR ROI letterbox保32columns；audio全波形log-mel40，encoder整理16序列向量；全logit softmax經可調projection。', 'E1：OCR CTC訓練loss與greedy collapse解碼分開，正文沒有宣稱greedy等於總路徑最大字串。', 'E2：fashion、relation、OCR實際outputs和圖標注相合；voice續聊history含真的第一輪output與原音訊message，沒有用gold替換。', 'V4–V6：實際三張圖的鞋／褲包／下區大小符合標注；聲音波形明標示意。', '實讀10.6、11.14、11.18、12.16：寬度轉换、保位置、外給ROI及history保存/學用關係各有來源。'],
 ['特定公開語音例「回答正確但分類頭錯」未從本輪讀到的raw invocation找到head score/argmax，列U1；不是判定其為假。', '未重聽錄音、跑感知頭／CTC與重新推論；CTC論文正文未另下載閱讀。']))
P.append(page('19.7', '合法也不一定正確：若模型把26抄成25，請求仍可能合法，工具也會正確計算25×16，整個任務卻已偏離原題。',
 'JSON請求、執行和同模型讀回結果的三界線正確；接口合法與原問題參數正確分開，有相稱需要／失敗例。',
 ['E1：tools.py嚴格檢查工具名、keys、operation、int且不接受bool／超範圍；直接加減乘、不eval；run_tool_loop先模型生成再parse，未由user程式抽gold。', 'X1：本頁固定multiply26,16經真calculator得416。', 'E2：公開往返兩個神經呼叫同權重hash，初次JSON→tool返回416→同模型「結果是416。」；V7一致。'],
 ['沒有重新跑整卷tool call生成；X1只證固定合法請求之執行，沒有證明模型新題選對參數。']))
P.append(page('19.8', '方法要對應現在缺少的學習訊號。',
 '預訓練、示範SFT、加權joint与偏好方法用途分開；完成主線使用示範objective，没有把加權loss叫PPO/DPO。',
 ['實讀7.17：預訓練與對話示範用途、權重接續已交代；13.17：PPO/DPO是可選替代路線，不是本v2必經主線。', 'E1：語言stage loss與task-family multiplicities，weighted/native仍用target監督，無policy/reference/reward訓練更新。', 'E4：MoE完成全部更新但selected1000，與「最後不一定最好」相合。'],
 ['未重現任務權重消融，不從這頁推導加權一定因果提升；正文亦未承諾。']))
P.append(page('19.9', '本成品的設定記錄支持「使用 SDPA」，沒有支持「已選到 FlashAttention 核心」。',
 'cache保存的內容、位置／mask等價前提與各架構優化作用分開；沒有把小型分數一致檢查升格成速度或品質證据。',
 ['E1：讀全model.py核RoPE cache position、RMSNorm、4Q/2KV GQA、tied表、SDPA與4/top2 MoE；generation初輪模態編碼，之後cache追加。', 'X1：寬16／1層／5人工ID、split3最大差4.76837158203125e-07、split2差0，allclose皆True，容差1e-4。', 'E1：正式batch一record一列PAD，沒有packing／MTP／draft verification；正文明說未採用。'],
 ['未在本輪完整公開模型比cache speed/memory或查GPU backend；沒有驗FlashAttention已被dispatch。']))
P.append(page('19.10', '少用一部分計算，並沒有把其他權重從 RAM 移走。',
 '儲存參數、RAM工作空間和訓練狀態分開；MiB與M定義正確，未拿參數bytes當最低機器要求。',
 ['X1參數數量乘4：MoE21788428bytes=20.778MiB，Dense9152268bytes=8.728MiB，正文20.78／8.73一致。', 'E1：所有expert實例都屬model parameters，選2不移除其他expert。', '模型/optimizer/code states支持訓練與推論負擔不同；未將FP16/量化無條件等同品質不變。'],
 ['未量本輪全程RAM峰值或訓練optimizer實際佔用；估算未當測量。']))
P.append(page('19.11', '只拿到回答器的權重，不能還原這些狀態。',
 '推論四檔、跨stage best與same-stage latest的功能區別具體；固定revision與SHA用於配套身份，不宣稱內容hash保能力。',
 ['E1：inference.fetch四檔核SHA/size固定revision979cdfacc588ad0536f1c64fff96f264571cf054，loader核config和strict state_dict；training checkpoint另帶optimizer/random/sampler/progress。', 'V8：四檔推論包與下方完整訓練states图的best/newstage及latest/samestage对应。', 'E2/E4：公開MoE safe hash26102d2edbd4574bef6733cba9e0dfd039118b206babdc66dbd16d1e57032d8c，selected1000、completed4000。'],
 ['沒有本輪下載全部四個輸出或核遠端檔當時可用性；resume軌跡未執行。']))
P.append(page('19.12', '所以高分能支持讀回結果，不能替真正往返補分。',
 '主要分數、單位及阈值与原始metrics一致，尤其預供返回值與真正往返、語音真實第一答續聊没有合併。能力不足由全卷而非展示控制。',
 ['E3：兩者metrics schema selftrained-generation-v2、count3734、complete true、safe_export、非teacher forcing；task分母加總3734。', 'MoE/Dense：text464/560 vs429/560；OCR252/324 vs281/324；服飾357/360 vs356/360；關係1427/1440 vs1426/1440；語音42/90 vs37/90；續聊26/60 vs24/60；tool_roundtrip0/276 vs7/276；预供tool_reply551/552 vs468/552。', 'E3：獨立OCR head324/324与core OCR不是同指標；audio perception的50x3是控制數而非新錄音數。', 'E3比較限制：總容量/active容量與最終objective不同，只支持這兩條成品，不可因果歸因MoE。'],
 ['未逐條重新評分3734生成或重新跑模型；固定關鍵詞語意規約不等同真人普遍可用性。']))
P.append(page('chapter-20', '各節短Python隔離一個機制；成熟模型下載、推論與候選重訓另有操作入口，不能把CPU手算當成完整模型能力證據。',
 '成熟底座與第19章隨機訓練起點分清，候選訓練與部署採用分清；小例範圍不過度。',
 ['E5/E6：v4配置Qwen3-VL-2B-Instruct＋Whisper-large-v3-turbo、adapter null；曾評1,039與2,077步候選，數據沒有支持採用。', '同一聊天core承接图文與ASR文字，凍結實作沒有將Qwen當成v2 MoE續訓。'],
 ['未重新下載/載入成熟權重或本輪重訓LoRA。']))
P.append(page('20.1', 'ASR 是語音辨識，輸出「聽到哪些字」，不是問題的答案。',
 '兩路會合到同一history user位置，有明說人工兩份文字相同的示範不是ASR能力測量；原圖／歷史保留用意清楚。',
 ['E1：natural_assistant.render_message/image流程与natural_ui transcribe/chat分開；ASR原稿或修改後prompt送同core。', 'V9：typed与Whisper文字匯同Qwen，photo processor是圖入口，LoRA標目前未開。', '示例相等只因人工固定同字串，判斷True沒有擅稱錄音結果。'],
 ['未實際播放本頁想定錄音或生成本頁招牌對話。']))
P.append(page('20.2', '「開始新對話」清除本段歷史與上傳素材。',
 '本機服務、等待載入、圖文／ASR先後及新對話狀態合理；操作要求与UI程式一致，能力限度另由全卷說明。',
 ['E1：natural_ui.reset删除history/assets/transcriptions與上傳file；load_core一次，ASR lazy另載，聊天保存旧图与实际回复。', 'E9：原始UI report已有四次chat、一次ASR、reset history/assets/transcription皆0；model_load_counts core1/asr1；只当旧执行事实核對，不宣称本輪瀏覽器重跑。', '本地host127.0.0.1:8766與serve程式一致。'],
 ['本輪未啟動UI／視覺操作或重驗全新安裝；既有自動化UI測試不是真人學生。']))
P.append(page('20.3', '可更新的數字少，不等於整套權重少。',
 'LoRA可更新容量与底座常駐、dtype估算与實際峰值有分開；底座为新Dense来源而非从v2接续。',
 ['手算2127532032×4=8510128128bytes=7.925GiB；×2約3.963GiB，与正文7.93/3.96一致。', 'E7：加入LoRA总2129137664減1605632=2127532032；固定base在前向、optimizer只LoRA。', 'E9：既有CPU FP32 server RSS约12.87GiB包含base/ASR/历史/图/库，与純權重估算没有矛盾。'],
 ['未本輪實例化2B數參數，來源總数／tensor統計只由保存執行紀錄回查；没有最低RAM保證。']))
P.append(page('20.4', '這兩份回答是本課AI助理實際看圖、讀原始人工英文描述後寫的繁中標註，並非DOCCI官方人工中文答案。',
 '問題/答案配對、短描述与事实题用途、訓練示例不能当heldout分清；翻譯標註与原官方描述來源角色透明。',
 ['V10：實際圖两猫桌面螢幕，左猫前脚抬起，与题/答案所述相合。', 'E5：訓練manifest中该DOCCI source有scene/fact问题，内容与题型对应；照片family留train。', '來源license/作者/原英文与繁中改写有所区分；没有说AI標註等於真人中文gold。'],
 ['未從Google重新下載指定generation原圖／核原英全描述；未独立重審全部繁中標註。']))
P.append(page('20.5', '修正是讓家族一起走，不能只替它改一個名字。',
 'split檢查与自動切分區別明確，示意4列不偽裝真資料統計；真manifest來源／衍生family不跨份。',
 ['人工列表cat A train、sign B test交集空；改变第二cat為test會產生交集；重命名ID不能修來源重疊。', 'E5：全manifest2077／124／170及audio0／16／22；train-val、train-test、val-test family交集皆0。', 'E5：照片unique439／28／42，near-duplicate clusters126／11／12；V11示意同源crop/paraphrase/adaptation一起train。'],
 ['未用影像embedding重新判near-duplicate cluster；family字段空交集不能自行证明无未知跨来源視覺重複。']))
P.append(page('20.6', '這個局部例子以輸出平方平均為代價，讓輸出接近0，不是讀照片。',
 '低秩支路／alpha/r／固定base前向與更新范围機制正確，有明確目的；單步示例不當成圖文品質驗收。',
 ['實讀8.8後核A4→2、B2→3及W3×4，A随机、B零初始化；optimizer只requires_grad，第一步B会更新。', 'X1：trainable14，base_before与after完全相等，B changed為True。', 'V12：Wx和alpha/r BAx相加，圖没有把W省略。', 'E1：正式LoRA target語言attention q/v，rank8/alpha16，視覺／ASR不在optimizer。'],
 ['未重跑正式LoRA训练；本例只核一次小張量更新。']))
P.append(page('20.7', '不是把題目從視野遮掉。',
 'causal attention、label -100與shift各自作用正確，答案首字的prediction位置沒有偷看到答案；局部编码与成熟模板边界有說清。',
 ['實讀7.4：本課render_chat回傳已對齊y，不再重shift。X1：6input、2targets、first4、ids73/2，与正文。', 'V13：assistant位置預測A，A位置預測end，其余labels ignored，图可讀位置与计分位置不混。', 'E1：成熟training_batch核prompt exact prefix，前缀mask-100，最後assistant及模板结尾计分，交HF模型內部shift；超長報錯不截断。'],
 ['未本輪用Qwen實際tokenizer造圖文batch；模板相等只核程式前綴檢查。']))
P.append(page('20.8', '這列省掉ASR，已足以說明退步不全來自聽錯；真ASR路線也下降。',
 '候選比较的固定条件、用途門檻、全部EOS与綜合分數嚴格提升分開；以正確文字控制定位聊天退步合理，不把照片改善當整體改善。',
 ['E6：分母28summary56fact18presence10single3order9chat4typed4ASR=132。base16/31/18/6/2/3/4/3；1039候選23/53/18/5/2/2/2/2，2077候选24/51/18/5/2/5/1/2。', 'E6：固定五组平均分base53277/75600，1039=52285/75600，2077=52705/75600；候選各3回答不完整，且typed/ASR均低於base。', 'source不是聲稱base已通所有要求；base本身有錯、作為未接受修正時fallback，正文能力卡保留其不足。'],
 ['语义判分依既有固定逐題记录，未獨立全題重評；未重新产三组answers。']))
P.append(page('20.9', '兩個正確名詞還不能回答「人在做什麼」。',
 '名词集合不能代替动作/关系證據，人工卡片只隔離評分差異，有明確不讀圖限定；訓練猫图与heldout測量分開。',
 ['示例集合相等不會使「站在車旁」等於「騎車」，可直接核字串/集合逻辑。', 'V14：riding／standing示意在物件名相同時動作不同；V10猫圖前脚姿勢支持可見事实而不支持「等主人」。', 'E8：原逐题描述子集动作3/11、关系21/29均包含在84fact中，非额外测试；引用用途與分母正确。'],
 ['没有重新看全42test圖並重評所有动作/关系或换图控制。']))
P.append(page('20.10', '本例按Unicode字元數，不按UTF-8 bytes',
 'CER编辑距离/参考字数分母正確，逐字字形與語意相同區分；例子保繁簡差異，没有以近義当OCR正确。',
 ['X1：今天去臺北→今天去台北，exactFalse、edits1、reference_characters5、CER.2。', 'E1：text_error_report是Levenshtein按Python Unicode字元；E8：真实singleOCR規約先NFKC去全部空白但保繁簡／正規化標點，8/10不是raw完全相同率。'],
 ['不从本例推模型抄字能力；未本輪重新跑真OCR。']))
P.append(page('20.11', '若去掉所有空白再比較，會把換行要求一起丟掉。',
 'Counter丟順序、字串與分行保順序的关系正确；singleOCR正規化與orderedOCR保持行次/换行不是同规约。',
 ['人工reference牛奶換行麵包、pred麵包換行牛奶：Counter相等但str不等、splitlines逆序可直接核。', 'V15：購物單图牛奶上麵包下，读序箭头与reference一致。', 'E8：ordered_ocr1/3，各题文字/顺序/换行一起评分；未拿single8/10替代。'],
 ['未評任意整頁多欄／無指定读序的OCR；正文已限定。']))
P.append(page('20.12', '一個否定詞就能反轉要求，所以平均字元差距不能代替條件是否保留。',
 '轉寫错误与模型對收到文字作答分開，两路固定其他输入的診斷关系正確，手工例不假裝ASR實測。',
 ['X1：含句點reference9char、缺不為1delete、CER1/9=11.1%，问题不同。', 'E1：ASR真waveform16k重取樣、最多30秒、Whisper中文greedy；evaluate同audio產ASR後再给相同Qwen，另typed reference路线。', 'V16：正確文字／實錄→ASR文字进入相同LM，照片與history固定。', 'E8：22音訊raw112/674／normalized96/661；4AISHELL兩CER0/42而两路chat各2/4，聽對不代表答對。'],
 ['未重聽22段或重新Whisper／Qwen推論；小否定例沒有音訊實測需求。']))
P.append(page('20.13', '因此178份回答不是178個獨立情境',
 '交付身份核对与质量驗收分開；最终分母、截断保留与共享照片/問句單位正确，未把沿用底座能力歸本課從零訓練。',
 ['E8：170visual/text＋4typed＋4ASRchat=178，171EOS、7chat達384截断；13chat分母保留7。', 'E8：25/42summary、58/84facts、18/18presence、8/10OCR、1/3order、2/13chat、2/4typed/2/4ASR；actions3/11和relations21/29是fact子集。', 'E8：22audio、18FLEURS/4AISHELL，raw112/674=16.62%、96/661=14.52%；NFKC去空白保标点繁简。', 'E5/E9：public配置9262bytes非完整weights，base固定89644892e4d85e24eaac8bacfd4f463576704203，ASR固定41f01f3fe87f28c78e2fbf8b568835947dd65ed9、无LoRA。', '人工hash例完整SHA比較False；前12字只顯示，不用短hash验文件。'],
 ['所有語意正確數由固定score記錄回查，沒有獨立全題重評；本輪未下載大权重、未驗多欄長對話。']))
P.append(page('natural-v4-student', '原稿與更正是兩份不同記錄，介面會保留這個差別。',
 '實際操作入口、載入顺序、限制与ASR可修改状态都与代码相符；資源表有測量条件和非最低需求界限，既有快取没有被寫成新下载。',
 ['E1：image8MiB/16Mpixel/static，audio8MiB与WAV/FLAC/MP3/OGG、30秒；user修改prompt与ASR原稿分别存，历史保留原圖。', 'E9：旧UI source FP32CPU，5threads/1interop；ready24.24秒、wall170.33、首次ASR18.03、serverRSS12.8709GiB，四次chat3.42/42.87/36.58/37.11秒。', 'E9：disk raw unique bytes5889111977约5.4847GiB、23snapshot symlinks；fresh public2files9262，cache metadata/RSS范围没有混称。', 'E1：student_options只接受公开两revision与配置，依source_sha/runtime版本检查；安装GPU/CPU分支没有替換官方base。'],
 ['本輪没有全新CPU/CUDA安装、浏览器实际截图/交互及全新官方权重下载；23files的cache树只回查既有disk记录，没有独立filesystem清点原机器。']))
P.append(page('natural-v4-data', '真人錄音用於驗收兩站流程，不拿來更新這次圖文LoRA或重新訓練Whisper。',
 '資料用途、來源授權、AI繁中標註與原人工source区分，候選训练并不保证部署；family先分和重取驗证的規約有相稱限制。',
 ['E5：全文manifest重新计数train2077/val124/test170，audio0/16/22；scene878/84/126、chat122/9/13、presence103/18/18、OCR851/10/10、order123/3/3。', 'E5：全部family跨split交集0；photo439/28/42及clusters126/11/12；source与descendant同组。', 'E5：vision压69308918/511files/解72155176；OCR59746570/961/62382076；voice17487821/41/26146571，说明区分了下载、解压、模型权重。', 'E1：fetch_natural_data固定manifest pin与archive size/SHA、路径受限，目标非空拒overwrite；正文没以取清单成功当成全部素材成功。'],
 ['没有重新下载/解包三archive、重建标注或核所有授权原站；未全面独立检测近重照片。', '初判读manifest有一次过宽source输出包含5条已有标注peer_review元数据，见independence例外；不用其中passed作技术通过依据。']))
P.append(page('natural-v4-training', '建立候選與最後採用候選，是兩個分開的決定。',
 '长操作页把底座、训练范围、精确续训、验证门槛、选择冻结及新test角色分清；公开配置不能替学生自选新候选背书，数值与既有run对应。',
 ['E1：LoRA只语言q/v、rank8/alpha16；training_batch最後assistant及模板結束计分、exact prefix和2048超長報錯。checkpoint保存optimizer/RNG/schedule/row cursor，resume核配方。', 'E7：已完成2077更新、每步2rows共4154、supervised78872、LoRA1605632/112tensors；图文总2129137664；仅少量冻结tensor index核对的限制在来源有写，不當全部2B bitwise证明。', 'E7：L4/bfloat16学习率3e-5，training timer1708.8144sec、runner1728.6763、CUDA allocated5719892480bytes=5.327GiB；非全部reserved显存/最低需求。', 'E6：三variant共享Whisper transcripts，132responses；五组均权平均和语音各自保留门槛，候选没有满足，retainbase与结果相称。', 'E1/E8：final测试只selected base、不adapter；公开data-root／fixedrevisions／manifest检查保存来源，不让最后题替验证选版。'],
 ['没有重新跑长训练或精确resume，同源初始hash/row序列为既有provenance回查，非本輪重新作因果对照。', '没有本輪加载adapter文件重新比112tensors/2Bbase或下载完整私有续训state；旧run数值仅能支持保存的执行范围。']))

expected = manifest['groups']['integration']['pages']
assert [p['page_id'] for p in P] == expected

prereq_specs = [
 ('7.17', '19.8核預訓練與對話示範分工／權重接續，不當成清空前段。'),
 ('15.13', '19.2、19.12核活躍／總容量和比較成本限定。'),
 ('10.6', '19.6核轉換向量寬度不自動增加影像資訊。'),
 ('11.14', '19.6、20.11核保留順序與平均摘要的用途區別。'),
 ('11.18', '19.6核使用公開ROI和模型自動偵測不同。'),
 ('12.16', '19.6核介面保history與神經模型學會使用history不同。'),
 ('7.4', '20.7核assistant首字的shift與causal位置。'),
 ('8.8', '20.6核LoRA低秩支路、初始化及第一步梯度。'),
 ('13.17', '19.8核PPO/DPO為偏好路線而非加權joint。'),
]
prereqs = [{'page_id': pid, 'reason': reason, 'read_scope': '全文', 'source_sha256': digest(ROOT/'freeze/sources'/f'{pid}.md')} for pid,reason in prereq_specs]

visuals = [
 ('19.1','p6-19-start-core','四路材料匯入同MoE，tool executor在神經core外，圖限制明確。'),
 ('19.3','p6-19-start-splits','A來源的single/pair/swap都train；B驗證、Ctest，格位與箭頭相符。'),
 ('19.4','p6-19-start-lineage','random一路到native，own best接續與newstage fresh optimizer/sampler写明。'),
 ('19.6','p6-19-modal-fashion','真灰階鞋圖及左褲右包；绿色左框與兩個generated回答對應。'),
 ('19.6','p6-19-modal-ocr','上口大口、下大小，绿色ROI框下區，[18,50,90,86]與answer大小一致。'),
 ('19.6','p6-19-modal-voice','波形/區塊寫示意；16ordered區間與softscores/historical真答流程相符。'),
 ('19.7','p6-19-modal-tool','26×16→same model JSON→真executor416→same model答416。'),
 ('19.11','p6-19-delivery-package-roles','上4safe檔infer、下full states，best新stage/latest同stage功能清楚。'),
 ('20.1','rewrite-20-input-routes','typed/ASR逐字稿與image processor会入同Qwen，旧history与noLoRA标示一致。'),
 ('20.4','natural-v4-training-cat','两猫桌上对屏幕、左猫抬前脚实可見。'),
 ('20.5','natural-v4-family-crops','A原圖/crop/paraphrase/繁中改写train；B两问test，为示意非真计数。'),
 ('20.6','natural_base_adapter','x经fixed W和A→B→alpha/r支路相加，更新箭头只A/B。'),
 ('20.7','natural-v4-answer-mask','前4labels忽略，assistant位置预测A、A预测end，6input和2targets对应。'),
 ('20.9','natural_photo_evidence','人工骑车与旁站示意名词一致，动作不同，不是模型测试截图。'),
 ('20.11','natural_reading_order','上牛奶下麵包，顺序箭头、ref与swapped output符合全文。'),
 ('20.12','natural-v4-asr-two-routes','source正确字/实际audio→ASR字进入same LM，photo/history同条件。'),
]
visual_records=[]
for n,(pid,stem,obs) in enumerate(visuals,1):
    original=f'course/figures/{stem}.svg'
    visual_records.append({'id':f'V{n}', 'page_id':pid, 'figure':original,
       'figure_sha256':inventory[pid]['figures_sha256'][original],
       'viewed_render':str(ROOT/'renders'/f'{stem}-640.png'),
       'view_method':'view_image实际取得图像后才在后续回合登记观察', 'observation':obs})

impl_scopes = {
 'tiny_perceptron/selftrained/model.py':'全文1–460；all modules/config/counts/forward/cache/generate',
 'tiny_perceptron/selftrained/dataset.py':'全文1–283；read_records/tokenizer/preprocess/encode/CTC目标',
 'tiny_perceptron/selftrained/tools.py':'全文；parse/executor/one-hop loop',
 'tiny_perceptron/selftrained/inference.py':'全文1–file end；fixed fetch/strict load/history/ChatSession',
 'scripts/selftrained/train.py':'25–80、284–299、731–842、850–933；stage trainable/checkpoint init-vs-resume/save及更新；其余仅定位搜索，未全读',
 'scripts/selftrained/prepare_text_tools.py':'404–489、494–630；其余仅定位搜索，未全读全部原創模板',
 'tiny_perceptron/natural_assistant.py':'33 target、140–175、263–307、444–666、687–775、814–960；mask/load/LoRA/train state/ASR/eval；非全文',
 'tiny_perceptron/natural_ui.py':'38–95、131–215、338–365；upload/chat/reset/serve；非全文',
 'scripts/fetch_natural_release.py':'88–175、323–430；validate/fetch/verify/runtime/student；非全文',
 'scripts/fetch_natural_data.py':'263–318；archive/fetch pin及验证/非覆盖，其他定位命中未算全文阅读',
}
impl_manifest=json.loads((ROOT/'checks/implementation-source-manifest.json').read_text())['files_sha256']
implementation=[]
for path,scope in impl_scopes.items():
    actual=digest(ROOT/'freeze/implementation'/path)
    assert actual==impl_manifest[path],path
    implementation.append({'path':path,'sha256':actual,'read_scope':scope})

evidence_specs = [
 ('E2','docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-index.json','原始执行文件列表／hash；只核source绑定，不用审阅pass支持结论'),
 ('E2','docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json','invocations各记录inputs/outputs/model hash/EOS/真实history；其余历史结论未作为判断证据'),
 ('E3','docs/selftrained/results/v2-final-public-results.json','architectures/data_scope/comparison_limit数字与配置'),
 ('E3','docs/selftrained/results/public-raw/moe/test/metrics.json','schema/count/completion/checkpoint与全per_task/perception/threshold数值'),
 ('E3','docs/selftrained/results/public-raw/dense/test/metrics.json','同上，全task分母与比较条件'),
 ('E4','docs/selftrained/v2-training-stage-index.json','各stage completed/selected/source hash；不用旧判定值'),
 ('E4','docs/selftrained/results/v2-final-training-source-selection.json','native MoE completed/selected/checkpoint SHA与Dense对应'),
 ('E5','docs/natural-assistant/v4/manifest.json','全JSON解析重新count family/split/task/assets/archive；不独立重审全部语义标注'),
 ('E5','docs/natural-assistant/v4/public-release.json','全JSON；固定配置/两file size/SHA及revision'),
 ('E6','docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json','全variant计数/denominators/completion/aggregate；语义case是既有判分，不称重评'),
 ('E6','docs/natural-assistant/v4/selection.json','全JSON；冻结候选/验证来源配置；决定按数值回核而非继承passed'),
 ('E6','docs/natural-assistant/v4/asr-selection.json','各模型param/revision/CER raw/normalized/validation denominator'),
 ('E7','docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json','直接正文链接执行紀錄的step/rows/tokens/LoRA tensors/source hashes/resources；含旧audit词句，不采用其status证明正确'),
 ('E8','docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/selected-final-test-audit-summary.json','counts/generation_summary/asr edits/reference/model revision；没有重新加载该run所有raw tensors'),
 ('E8','docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/scored/scores.json','final用途counts/denominators/completion与全部22ASR reference/hypothesis及CER；未独立重评开放语义'),
 ('E8','docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/descriptive-subsets.json','photo_fact_subsets分类correct/denominator，属于同84题不是新评'),
 ('E9','docs/natural-assistant/evidence/v4-research/student-selected-public-ui/README.md','正文直接链接，全读条件/测量/范围；旧自动化操作不是本轮/真人试用'),
 ('E9','docs/natural-assistant/evidence/v4-research/student-selected-public-ui/actual-ui/report.json','request响应/ready/wall/server资源/model_load_counts/reset_states/token stops；字段选择阅读，不全读UI provenance narrative'),
 ('E9','docs/natural-assistant/evidence/v4-research/student-selected-public-ui/actual-ui/disk-observation.json','全文，cache inode/symlink/bytes与fresh public范围'),
]
evidence=[{'id':i,'path':p,'sha256':digest(REPO/p),'read_scope':scope} for i,p,scope in evidence_specs]

report={
 'schema_version':1, 'reviewer':'/root/technical_integration', 'role':'technical', 'group':'integration',
 'review_kind':'來源獨立初判；全教材診斷，非發布驗收；AI輔助技術審閱',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'baseline_commit':manifest['baseline_commit'], 'criteria_sha256':manifest['criteria_sha256'],
 'criteria_actually_read':['SKILL.md','references/review-protocol.md','references/calibration.md','references/project-context.md'],
 'independence': {
   'initial_judgment_before_cross_review':True,
   'other_course_reader_technical_continuity_reports_read':False,
   'forbidden_reports_notes_traces_human_judgments_read':False,
   'direct_linked_evidence_exception': '正文直接链接的manifest一次过宽source输出意外看到5条照片标注peer_review author/passed等既有标注元数据，已通知root并停止展开。直接链接执行JSON亦自带旧audit status/verification等字段；它们不是本批技术/reader报告，不用passed/status支持结论，只抽实际数值、source hash、生成输入输出。不能宣稱完全未見任何歷史審閱字樣。',
   'judgment_basis':'先全读30页作者来源，再核必要原始数值/冻结实作；不借先前教材问题清单找缺口。',
   'human_student_testing':False,
 },
 'actual_prerequisites':prereqs, 'pages':P,
 'issues':[{
    'id':'U1','page_id':'19.6', 'severity':'unverified',
    'quote':'這個公開語音例的回答正確，獨立分類頭卻沒有選對類別，兩件事不能混稱通過。',
    'already_taught':'soft bridge保留整組候選比例，分類head argmax與core生成分開評分；機制已教且程式支持。',
    'missing_minimum_relation':'沒有教學關係缺口；目前讀到的公開invocation只可對上音訊路徑、生成答覆與權重hash，未見該同一次音訊head argmax/gold的原始對照。',
    'current_impact':'無法在本轮独立证特定反例的分類頭錯誤，不能以它新增模型能力主張；不表示句子错误。',
    'minimal_repair':'若需补可核证据，链接一个与该音訊SHA、同权重绑定的raw head scores/predicted class/reference记录；无须改写soft bridge教学。',
    'rewrite_scale':'paragraph', 'repair_required_for_curriculum':False,
    'discovered_at':'30页来源全文读毕后的技术证据回查，尚未交叉或读旧课程审阅',
    'independent_source':'19.6冻结正文及直接链接公开CPU invocations，未看其他课程reviewer结论。',
 }],
 'coverage':{
    'assigned_pages':len(expected),'actually_read_full_pages':len(P),'unread_pages':[],
    'page_ids':expected,'all_pages_have_concrete_quotes':all(p['quoted_basis'] for p in P),
    'coverage_type':'全指定来源阅读；技术执行/语义重评均按实际范围注明，不以coverage冒称全实测。',
    'required_issue_count':0, 'optional_issue_count':0, 'unverified_issue_count':1,
    'repair_scale_judgment':'未找到必要段落/页/章/顺序重写；U1仅为可选补具体原始记录的段落规模。此结论只覆盖integration技术轮，不替reader或continuity全教材结论。',
 },
 'visual_scope':{
    'actually_viewed_count':len(visual_records),'actually_viewed':visual_records,
    'unverified':['360px版本未看','成品教材网页/mobile渲染未看','来源之外全部测试照片/音频未逐个看/听'],
    'not_claimed':'640px图内容核对，不是盲首读视觉路径或学生读图测试。',
 },
 'execution_scope':{
    'X1':{'artifact':str(WORK/'miniature-checks.json'),'sha256':digest(WORK/'miniature-checks.json'),
      'runtime':'repo .venv Python；torch2.14.1+cpu；PYTHONPATH指向冻结implementation',
      'scope':'random models参数计数；19.5 labels；19.7 calculator；19.9 cache split3/2；20.6 LoRA单步；20.7 mask；20.10/20.12CER',
      'not_verified':'不是正文正式torch2.8配套新环境，不载预训模型、不训练成品、不作速度或全能力验收。'},
    'manifest_recount':{'artifact':str(WORK/'natural-manifest-counts.json'),'sha256':digest(WORK/'natural-manifest-counts.json'),'scope':'全JSON parse累计family/task/split/photo/source clusters/archives，未重建或重评标注。'},
    'public_cpu_extract':{'artifact':str(WORK/'public-cpu-raw-extract.json'),'sha256':digest(WORK/'public-cpu-raw-extract.json'),'scope':'保存本轮读到的旧raw invocation字段；未运行神经推论。'},
    'inspection_probe_note':'少数自写JSON检查探针曾以speaker/source同形假设触发KeyError或TypeError，已按真实schema修正；这是review脚本假设，不是教材执行失败。',
    'implementation':implementation,'original_evidence':evidence,
    'not_executed':['长训练','真实2B/Whisper权重本轮加载','成品神经推论与全卷生成','全部语意重新判分','精确resume轨迹','全新CPU/CUDA安装','在线发布/上传','本轮网页UI或真人试用'],
 },
}

target=ROOT/'reports/technical-integration-initial.json'
assert not target.exists(), 'Do not overwrite sealed initial report'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
seal={'schema_version':1,'reviewer':report['reviewer'],'role':'technical','group':'integration',
  'report_path':str(target),'report_sha256':digest(target),'sealed_at_utc':datetime.now(timezone.utc).isoformat(),
  'coverage_pages':len(P),'source_sha256':{p['page_id']:p['source_sha256'] for p in P},
  'seal_scope':'独立初判封存，之后交叉任务另存；不覆盖本报告。'}
seal_path=ROOT/'reports/technical-integration-initial.seal.json'
assert not seal_path.exists()
seal_path.write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(target),'seal':str(seal_path),'sha256':seal['report_sha256'],'coverage':len(P),'required':0,'optional':0,'unverified':1},ensure_ascii=False))
