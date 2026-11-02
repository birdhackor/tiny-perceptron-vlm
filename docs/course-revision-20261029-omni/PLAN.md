> 執行授權：使用者於2026-10-29回覆「可以，GO」，批准此v3範圍、額外Modal gross上限US$60、必要新訓練／外部儲存與檢查後發布。下文保留批准前計畫原貌；實際進度與偏差另見STATUS.md。
> 追加要求：易讀易理解／正確性／連貫性三輪獨立審閱；理解必要的程式碼、關鍵輸出和說明直接內建教材，不能依賴打開Colab。

# 小小感知機 Kev／Omni 修訂計畫 v3：融入主線與第 19 章新版產物

日期：2026-10-29。狀態：修訂規劃，尚未取得教材修改、付費研究、訓練或新版發布的批准。

## 1. 修正前提與目標

撤回 v2「既有 A/B/C 不發展新版、主要新增第 21／22 章」的範圍限制。那是作者保守選擇，不是使用者要求，也不是技術必要；Astra 上一輪是在該範圍下審查，不代表它否定主線整合。

本輪目標是讓讀者先在前面學到必要方法，最後在第 19 章訓練出實際使用這些方法的新版本；成熟模型作為有來源的設計案例及對照。歷史權重、測試集、成績、tag 與證據保留為基線，不代表禁止新模型與新資料。第 20 章以後不再承擔首次教授核心概念的責任。

保持從零 PyTorch、筆電 CPU 小例子、正式訓練可用 Modal；不把 Qwen／Kev 預訓練權重悄悄變成第 19 章的自訓練成品，不承諾重現其規模與通用能力。

## 2. 已確認現況與來源

教材／本地／公開網站基線為 2b3332ca4392a946a065782815f9f60dae9a87c7。A 的 AudioEncoder 已有時間卷積、雙向 GRU、16 個有順序的音訊區間和位置 embedding；不能把「保留音訊順序」假裝成此次首創。但它仍不是具實際共同時間戳的影音事件對齊模型，也不是因果串流編碼器。

C 已有 state/question/candidate 意義的獨立共享評分、Choice／資訊足夠／rubric 和校準。現在每候選重新編碼完整輸入，且既有 block 是雙向 attention；不能直接聲稱切出舊 C 的 prefix 就與全算等價。固定候選語義與有限字元表也不是自由候選語意泛化。舊 C 核心 Choice 11797/11797，換說法 11493/11797，並有極高機率錯答；這些是已見診斷，不可再拿來選新模型的 final 門檻。

Jev 式規劃的直接依據仍是 TypeSafe 官方型別／confidence 文件及 DeskMind 公開訓練說明；C 為本課自行設計。Kev 是獨立開源實作，其二手 Jev 架構文章不當成 TypeSafe 官方披露。

Kev 程式 pin 2074cc44256200bb37540103fa4572901faa10f3；Kev-4B v1.0 Hub metadata 6cfce5c2fa4b4bd64026336ab649c5ca78857d52；底座 Qwen3.5-4B-Base 1001bb4d826a52d1f399e183466143f4da7b741b。Omni 主來源為 Qwen2.5-Omni 2503.20215v1、Qwen3-Omni 2509.17765v1、官方程式及實際 config；3B Hub pin f75b40e3da2003cdd6e1829b1f420ca70797c34e。開工核對版本、adapter/base/head/tokenizer 與 tensor digest。

## 3. 課程配置：先學原理，再累積到產物

不預先以新增章節或小節數作工作量目標。優先在原有問題旁深入改寫；舊 numbered URL 保留，新增小節優先接在章末，以章內導讀和前置連結安排實際路線。若安全保留 anchors 不足，先產生明確 redirect／舊連結驗證表，不能默默重編破壞引用。

第 1–6 章保留理解梯度、分類、attention 等基礎順序；必要時調整例子，提前累積「同一材料有多種問題、答案與證據要對得上」的直覺，不提前堆模型名詞。

第 7–9 章：在既有 SFT、候選語義、proper scoring、校準與決策處，實作小型 pointer／候選評分頭，對照各候選獨立表示及因果序列中一起比較的表示。用新說法、干擾／移除正解、未知任務及資訊不足最小配對建立訓練資料。補清楚固定語義換句話與全新候選語義是不同泛化；分開 raw p_max、溫度、API 集中度和固定門檻。

第 10–11 章：把影像 token 的空間座標、模態／片段界線、同一畫面不同問題，以及缺失一條線索的對照，連到後續短片輸入。先用離散影格與明示時間，不提前依賴 RoPE。

第 12 章：由目前的聲音特徵／順序／有限語音意圖，延伸到「第幾個區間」與「實際幾秒」的差別、共同影音時間軸及事件對齊。用自製短聲音與固定單碼本首次教波形→聲音碼→重建，不在此要求讀者先懂離散選碼的梯度。先用加法時間embedding；TMRoPE到14章、學習碼本到17章才引入。

第 11–12 章配合既有凍結／解凍與遺忘段落補分階段訓練、模態混合與舊能力回歸。第13章保留偏好／後訓練主線，只修必要銜接，不要求所有產品跑完每種後訓練。

第 14–15 章：在已學 RoPE 後實作小型 time/height/width 位置分配，與相同容量的簡單時間 embedding 作對照；依輸入是否需要而接入 A。用 Qwen3-Omni 解釋哪些子網路使用 MoE、總參數與啟用量，保留 Dense 公平對照，不把 A3B 解讀為 3GB 或可忽略所有權重。

第 16 章：共享材料的多問題計算、分支隔離及快取先從本課小模型教。新 C 採可共享的結構後才談 cache 等價。Kev-4B 的 recurrent state 與 KV 同為必教狀態，不只在選讀一筆帶過。輸出串流分清模型因果性、lookahead、實際介面與可播放 chunk；A 的雙向輸入編碼器不能因此改稱即時串流輸入。

第 17 章：區分權重量化與音訊離散碼本，當場教argmin不可微、採用的straight-through estimator、碼本更新／commitment loss及塌縮診斷，再訓練小型codec。以CPU例子教residual codec與多碼本預測；同一音框的殘差碼本不能等同未來文字MTP。第18章保留蒸餾主線，不另加本輪蒸餾訓練。

第 19 章：重寫產品規格、資料、訓練、路由、驗收和版本下載，實際交付下列新版。每項機制都回指首次教學處，不能留到此章才第一次解釋。

第 20 章及後續：保留目前 Qwen3-VL＋Whisper 成品與原結果，集中比較 Kev／Omni 的完整成熟架構、規模、訓練配方、授權、部署與有限實測。能在前面教的概念前移；不再硬性新增兩章各七節。

核心對應表（確切小節ID於P1按依賴／anchor表凍結）：

| 首次教學 | CPU例子 | 第19章實際模組 | 必備證據 |
|---|---|---|---|
| 7–9候選與可靠決策 | pointer、重排／移除正解、校準 | C語義評分及固定轉交策略 | 重排、sealed final、接受數與錯誤 |
| 10–12共同時間 | 灰階影格＋聲音移時／線索刪除 | A影片payload／局部聲音token | 配對兩題均對、不同時長／間距、回歸 |
| 12固定碼本；17學習式量化 | 波形→碼→重建及梯度 | A有限聲音生成模組 | 重建、gold條件及真A預測條件全鏈 |
| 14時間RoPE／TMRoPE | time/height/width小矩陣 | A受控時間表示變體 | 同容量／輸入／下游seed對照 |
| 16共享記憶體與因果輸出 | 重算／快取；可播放分塊 | C共享state；A聲音decoder | 中間狀態等價／失效；chunk因果性 |

## 4. 第 19 章 A 新版：時間對齊與有限聲音輸出

A新版沿用小型自訓練助理定位。最低影音方案使用灰階圓／方或衣物事件，不新增RGB訓練；紅藍影格只留成熟Omni研究。新增video-audio-v1 payload明列像素、frame timestamps、16kHz聲音、共同起點／毫秒單位、有效長度、片段界線與padding mask。上限4秒、16張32×32灰階影格；開發2–4秒、抽樣間距含250／500ms；事件避開狀態切換至少125ms。token因果序號與物理時間是兩個獨立座標。

同樣影格與聲音內容只改聲音時間，必須判斷發聲時的畫面。新事件分支的聲音特徵使用明確最多250ms的局部支持範圍，公開邊界／padding，不給舊雙向GRU的16個全局特徵貼時間標籤就宣稱局部解析度。既有真人語音意圖分支保留；舊16格路線是相對時間參考，不宣稱任意長度／細度。

正式最低比較為實際時間加法embedding與時間RoPE變體，同資料、相近參數／計算及固定seeds，各三次下游訓練。兩者都有明示實際時間；相對槽位例先留CPU。完整TMRoPE以CPU的time/height/width例說明；灰階任務無空間變化時不稱測到完整TMRoPE空間優勢。成品預設由dev選，不由final選。跨3B／30B比較不作位置法因果證據。

增加「A可獨立下載的有限聲音生成模組」：從零訓練單碼本codec encoder／量化碼本／因果decoder，凍結codec後才產生Talker穩定targets；由有限文字／狀態條件預測代碼。素材為8kHz、最多2秒及最多4事件的自製低／高音、節奏、停頓；檢查塌縮、靜音捷徑和重建。殘差多碼本／MTP先CPU教學，不列最低產品訓練。解析振盪器可作重建參考，不必宣稱codec勝過解析合成。這是有限聲音成品，不能叫通用TTS、自然真人聲音或Qwen Talker重現。

整合必須實際跑A預測答案→公開有限聲音表示約定→學習式代碼生成→decoder，例如圓／方對應各自有限音型；分別報gold條件模組表現與真A預測條件下全鏈結果。不能餵gold答案給Talker，再把兩個獨立成功相加冒稱整合成功。若codec或整鏈不達門檻，保留候選，不用固定WAV掩飾。

codec 重建與條件生成分開評；前者報時域／頻譜誤差，後者用獨立訊號分析器判頻率、順序、節奏及完整度。新模態能力與原文字、影像、OCR、有限真人語音意圖都回歸測；歷史沒通過的標準保留，不把舊失敗改寫成成功。有限聲音輸出可用實際因果分塊播放，但不因此承諾整個系統 full duplex／barge-in。

## 5. 第 19 章 C 新版：共享材料、候選語義與可靠決策

C 新版明確增加可共用的 state encoder，再以問題與各候選的文字意義讀取材料並計分；採共享參數的獨立候選分支，使候選清單重排仍有結構依據。具體可用 query/key pointer 計分，但 dot product 本身不構成等變保證；位置、遮罩與 readout 才決定。另做因果序列 joint-pointer 的 CPU 對照，說明它與 Kev 題內互動的關係，不預設它必然更佳。

新結構與舊C雙向全輸入scorer是不同模型；state-only encoder可雙向，但不讀question/candidate。只限定凍結權重的推論共享記憶體，各分支唯讀；cache key包含state/tokenizer/model revision、位置、長度、padding、dtype/device與前處理版本，任一變動即失效。比較同一新架構重算／快取的中間狀態與機率，不是直接快取舊C，也不必硬套decoder prefix KV。現有C無recurrent混合層，Kev狀態限制在對照中另教。

輸入為明確question及自足candidate descriptions，不只固定ID，不支援「前一項」等跨選項指涉。固定UTF-8 byte tokenizer，不從final fit詞彙；state上限512 bytes，question128、每候選128，最多8候選，超限明確拒絕。byte覆蓋只解除OOV；未見措辭、未見組合和全新概念分列。訓練含候選增減、移除正解、缺資訊及干擾，不宣稱自動學會新語義。

固定世界unsupported與目前列表none分開；none用對實質候選分數的對稱彙整產生，因此實質候選各自獨立、集合保持重排等變，但none正確性仍需學習驗收。新增候選會改softmax分母與none，不把其正確性說成結構保證。

保留 Choice／資訊足夠／rubric 的型別差異。新資料分 train、model-dev、calibration、policy、sealed final；T 只看 calibration，threshold 只看 policy，final 用固定決策報錯誤／接受／轉交與群組分母。溫度可以惡化，unknown 偵測沒有保證，結果照實呈現。架構、資料增廣與校準至少有可區分的對照，不能把所有改進歸給 pointer head。

## 6. B 與 Auto 整合

B的Looped架構無須因Kev／Omni而改，保留既有權重、tokenizer、輸入文法和能力邊界。Auto改用新版C固定策略處理支援任務、缺資訊、未知／轉交、進入A或B及B深度；有明確B文法適配器和拒絕條件。C只讀文字／附件存在資訊時，就只依這些路由，不宣稱已理解影音內容。分別測gold路由及真C路由的全鏈，人工選檔優先。

不用 Choice 信心假裝 B 的更多迴圈一定有幫助；既有加法失敗不能宣稱由 C 即可解決。若新增 learned depth controller，必須另列有各深度品質／成本標籤的資料、訓練與獨立驗收，不在本版暗中承諾。整合失敗要分清 C 判斷、規則、B 解題與 A 輸出責任。

## 7. 訓練、資料與成效判準

各新增神經元件均有random-init、optimizer更新證據和checkpoint lineage。A沿用自家從零基線的參數稱warm start／續訓，不稱重新隨機初始化；不相容才另建從零lineage。共用舊A的三個下游seed不是三次獨立預訓練。新舊A的資料、可訓練參數、步數／token／音訊秒數與總成本明列；C新架構另建lineage。

A影音最多12000個train情境家族、1000個model-dev、200個sealed-final配對家族；final含不同時長／抽樣間距、移時及刪除線索，同家族兩題不能跨split。codec/Talker最多5000個train聲音程式家族、500個dev、200個final，以音調／時值／停頓組合為family，振幅／相位／微噪聲變體不算新family；codec及Talker都不可看final組合。新版保留能力另凍結每舊任務至少100家族的回歸面板；若有限母體不足100，明列實際母體及數量，不以變體湊數。已見舊測試只報歷史回歸，不當新sealed final。C的train最多30000家族、model-dev2000，外加下述可靠性面板。以上是數量上限，不宣稱生成家族等於真實獨立來源。

P1 在正式訓練前凍結任務、資料生成器、來源／模板／情境／變體 IDs、切分、評分器、隨機種子、數值門檻與最多嘗試數；本文不預填無依據的準確率門檻。定門檻只能用 development 和既有基線，不能先看 sealed final 再降門檻。通過與否必須至少有非平凡 chance baseline、事先列明的能力目標、配對影音兩題均正確、音訊線索移除控制、聲音 code 非查表及 C 固定門檻實際表現。A主要位置法比較採3個固定下游seeds；C與codec/Talker先單seed，不宣稱跨seed穩定優勢。

最低jobs矩陣：A兩表示×三下游seed共6次；C核心一次＋同架構同seed資料對照一次，共2次；codec一次，凍結後Talker一次，共2次；Kev和3B各一限額研究job，共2次，共12個正式jobs。另有CPU/infra小probe及final評估，全部入同一預約表。新C重算／快取和重排用同權重，不另訓練；刪除無明確問題的架構網格、所有組件一律三seed、MoE Talker與完整多碼本訓練。失敗無自動重試，新attempt另占已批准上界，不刪核心final換成功。

C 新可靠性面板至少 200 個獨立家族：calibration80、policy60、final60；不含較大的 train/model-dev。final 至少分開開發來源新家族30與完整來源留出30。先分 source，再分 family，再產生措辭／翻譯／重排 variants；同生成器換 seed 不是新 source。自製模板留出只能稱模板留出；若取得不到真正不同資料來源，不冒稱 source generalization，也不能把留出來源內容用於 prompt／T／threshold。

每家族預指定一個主要 Choice；變體、資訊足夠、rubric 是次要結果，不把同家族幾十題當獨立樣本。報各來源／語言／任務的 n、家族權重和區間；zero acceptance 的 risk 未定義。獨立 Bernoulli 零錯 n60 單側95%上界約4.87%、n30約9.50%；不能以退化 bootstrap [0,0] 保證零風險。新準則必須承認有效接受數可能更少；偏差和來源不足不因區間消失。未知預訓練來源不叫真正 pretrain-unseen。

所有 final 僅在 code/model/data/prompt/T/threshold 凍結後執行；已看 final 的材料只能成下輪 development。失敗保存，不用無限重訓挑最好。品質門檻未達時，新版可作研究候選，但不得替換舊正式產品／標成已完成能力。

## 8. 成熟模型研究的角色與界限

先讀原始碼與報告即可支撐主線設計，跑不動成熟模型不妨礙寫原理或訓練微型產品。Modal 主要價值改為訓練／評估本課新版 A、C 和聲音零件，而不只租卡看成熟模型輸出。

仍保留小量 Kev-4B 和 Qwen2.5-Omni-3B 研究作實作核對。Kev 以同一可靠性面板可適用任務為主，英文為模型卡主要語言，中文另列語言轉移；它不是 C 公平架構比較。Omni 固定上限為24個短音訊／文字配對、24個自製影音移時配對及8個短聲音輸出提示；前處理、timestamps、processor flags、model config 與案例正解全部存證。

Kev raw／上游 T／新 T 分開，head 若已除 T 不再重複縮放；無可靠 raw 就只測原 T。post-hoc coverage_at_error 曲線與 fixed policy 的 final 成績不同。cache 比較看中間狀態／機率，不只答案；bf16／fp32 容忍差和低 margin 規則先定。

2.5 報告的7B家族與3B實際 config／結果分開；2.5 Talker 用 Thinker 高階文字表示及 selected text tokens，3不沿用相同高階文字表示但仍有文字及多模態條件。Qwen3 MTP 補同一音框殘差碼本，不能套成未來文字 MTP。若 API 只回整段音訊，只報整段完成時間，不能推定 streaming TTFA；語音內容正確需獨立核對，ASR 是有錯誤的代理，自然度不冒稱真人評測。TMRoPE 因果收益只由同規模本課對照討論，非跨代成熟模型比較。

不執行30B推論、不微調Kev／Omni、不承諾通用真人語音輸出。Qwen3 公開資料足夠作架構補充，A3B不代表小顯存。

## 9. 授權、依賴與儲存

Qwen2.5-Omni-3B 為研究授權，預設僅本輪非商業研究／評估；不包3B權重／衍生模型到主線。不把 MIT 覆蓋研究授權，也不把全部輸出誤說自動同授權。原創 PyTorch 實作、自製資料與論文原理解說各自記來源；不把 Qwen／Kev 輸出或權重蒸餾進新版 A/C。若用途屬商業，3B執行需換經核對方案，不暗換名字。

逐資產授權表含模型、底座、adapter、程式、資料、聲音、結果與摘錄；開源架構啟發和直接複製程式分清。研究環境與主線 torch／uv.lock 隔離，CPU課文不依賴成熟模型下載。

新模型權重／可續訓狀態以 HF 新版本／新 repo 的不可變 commit 保存；既有 token／repo 需核對 read/write 權限。不覆寫舊權重、tag或 final 文件。固定教學資料包只依現行 AGENTS 放授權 LFS 路徑或既有外部資產通道；本次批准明確包含此存放方式，新大型檔不進普通 Git。研究證據以版本化 GitHub Release 小型公開包保存，Git 只留 URL、SHA、manifest 和重建指令。

第三方全文／權重cache只在忽略區或新研究Volume；模型cache完成後24小時內清理，其他私有暫存最長30天，公開支持證據先匯出。續訓 checkpoint 的保留與存放要在 manifest 中另列，不可一起誤刪。只清本輪資源，不能把 Modal Volume 當永久公開證據。

## 10. 階段與檢查

P0：完整 v3→Astra max(priority/fast) 重新諮詢→逐項修訂→完整回交使用者；未獲准不改教材、訓練或付費租卡。

P1：核對 main／乾淨分支、AGENTS／clear-tutorial、既有審閱基線與 baseline replay；建立概念依賴圖、逐節變更表、A/C/codec產品規格、資料／統計／版本／授權與新預算契約。優先處理權限及儲存阻塞，凍結開發與final隔離。本階段優先唯讀檢查。若HF／secret必須啟動Modal CPU驗證，明列為有限基礎設施例外：先完成獨立審過的最小預約／finance機制，從準備款保留最壞bound後才呼叫；不能用三輪後才有finance的規劃先派未受控容器。正式GPU smoke與訓練仍在P3之後。

P2：按學生實際依賴順序先寫前面新增原理與可跑CPU例子／SVG，再改19的訓練整合與20以後成熟對照，包含README/README_en、curriculum、glossary、training、導航與下載指令。未實測處明示待驗，不預填成果。

P3：依repo要求完成獨立首次閱讀、技術、前後銜接三輪；全受影響舊章、實作、入口與圖都在範圍，不能預設只審新章。方法和實驗規格通過後，完成新finance runner與必要工程。

P4：有預算保留後依序做 runtime/Volume/HF小probe、codec與A/C正式訓練、配對對照、凍結final、成熟模型有限研究；保存 optimizer、metrics、原始輸出及失敗。允許先做小型工程sanity，不冒稱正式產品驗證。

P5：用真結果重寫解釋、圖表、練習與19結論；凍結後對完整受影響閱讀單位按 reader→technical→continuity 複核，保留原首讀，不把重讀冒稱初讀。顧問／作者已知答案不提供給首次讀者。

P6：重建Notebook、reading time、全站和受影響模型配套；跑build_course --check、lesson_order、三道review檢查、相關pytest/ruff及提交前規定全套pytest；沿用現行全站kernel/export/check_site程序，實際桌面／手機檢視圖、公式、導航、下載。PR記錄結果、失敗與未執行；在批准涵蓋合併／發布下 merge main、Pages、HF新版本及證據Release，遠端驗build-info SHA與下載digest，舊URL和歷史下載仍有效。

新輪次不改舊hash假裝通關；baseline supplements只依現行規約保存真實還原資料。外部資產用新版本，回退以revert commit和產品指標回指舊固定revision，不改Git歷史。

## 11. 預算與停止規則：重新提案，不沿用推論預算假設

v2 的額外 US$30 是成熟模型推論提案，不能當成本次訓練已獲授權，也不能宣稱一定足夠。v3 建議提交使用者審閱的新 round gross 上限 US$60：準備／權限／下載／暫存5、A時間對齊訓練與對照16、C訓練／資料對照8、codec與Talker8、Kev／Omni有限研究8、失敗／修正／收尾保留15。這是規劃上限而非已知帳單或完成保證；P1再依官方rates、舊訓練耗時與每job實際上限分解，不能用保留款偷偷加工作範圍。

使用者尚未同意任何新額度。本次批准若涵蓋60美元，仍不得調用舊round的額度或重設舊policy。舊finance.py／v2 quota／ledger／gross sidecar與所有receipt不變；新增round policy、獨立wrapper、sidecar與資源前綴，明記和舊全域ceiling的關係。先read-only分類legacy已終止待入帳、active、storage、unknown；不釐清active/unknown不派GPU，不擅停其他工作，也不把legacy carry當新花費或自行釋放。

共享付費互斥＋單寫入者＋持久原子交易；每attempt唯一ID，先按CPU/RAM/GPU/build/startup/idle/失敗/storage/cleanup全部上界保留，再派工；所有新attempt不可遞減reserved合計≤60。不因成功、失敗或billing延遲自動返還。官方gross與high-water另交叉守門；不明增量先停，不能把保守封鎖稱為實花60。

單GPU、序列、max_containers=1、retries=0、明確timeout、token/media/request/step限制；微型訓練先選實際供應且合適的低成本卡，3B原生推論才評估L40S48GB。不得自動升級H100、加卡、長駐服務或無限重訓。每次嘗試成本／清理有責任人和receipt，訓練checkpoint及證據先匯出；仍有storage或帳務未決不能稱財務結案。

預算不足時先保留本課核心A/C/聲音機制與核心final，停止尚未派的成熟模型對照或非必要比較；A主要比較若未完成三seed就明列不完整、不得宣稱穩定差異，逐項標未執行；必要驗收缺項時整輪不標完成。不以改低門檻／偷減必要功能換成發布成功。任何加額、30B、成熟模型微調另議。

## 12. 工具與權限：已通與新增缺口分開

已實測：本地repo歷史、GitHub讀取/push與新增workflow寫入（一次性probe分支已刪，main不變）、Actions dispatch/原生日誌/artifact取回；Pages既有發布正常、main與網站SHA一致。Modal現有Actions憑證可讀 birdhackor workspace 官方billing，main environment沿用使用者既有選擇。沒有本地token並非阻塞，不要求使用者再次貼或另綁秘密。

本地CPU torch前向反向、pytest/ruff/Notebook kernel、uv/gh/git-lfs、ffmpeg/ffprobe、隔離Zensical與桌面/手機Chromium已可用。Kev／底座／3B均可匿名讀public ungated metadata；tensor下載和GPU kernel仍需獲准後的小probe。

新版產品增加HF權重與checkpoint寫入需求；上一輪「不需HF寫token」不再成立。既有lesson-products-publish workflow透過Modal HF secret與HF_RELEASE_REPO提供通道，但本次尚未實測token權限／可用目標repo／新revision發佈。先在P1、長篇改寫前用既有憑證做最小read/write或repo metadata權限檢查，若不能非付費直查，列入已批准CPU小probe；測試資產使用隔離revision、不覆寫舊模型。GPU供應、Volume寫入、HF checkpoint備份／讀回亦未實測，不能把billing通當租卡／發布全通。

目前無證據顯示必須使用者補新的API金鑰。若HF既有token不足才請使用者更新到環境secrets，絕不貼到對話；若指定repo不允許新權重，可規劃GitHub Release固定模型包作備援，但須先驗上傳／取回，完整包含推論、optimizer/RNG續訓狀態和資料重建規格，不靜默降格或宣稱資產寫入已測。

## 13. 完成與審核方式

承諾交付已批准的實作、訓練及獨立驗收；只有通過門檻者升為正式新版，失敗者保留研究候選及舊正式產品，不保證研究必然成功。具體交付包括：主線實質修訂與CPU例子／SVG；A時間對齊和有限聲音輸出、C共享材料與可靠決策的新訓練版本；B/Auto整合與分責驗收；成熟模型原碼／報告對照與已完成實測；原始資料生成與授權、checkpoint／新權重、固定final、版本卡與真失敗；三輪獨立審閱、Notebook／網站及遠端下載驗證。

完成實驗但結果差、未達成產品門檻、因預算／授權／runtime未執行三者分開。只在必要規格／審閱／發布條件全過時稱整輪完成；不承諾每種成熟架構都成預設，也不因新來源存在就為B安排無理由重訓。

提交使用者的是完整v3、Astra本輪意見與採納，以及改動／訓練／預算／儲存／新權重與Pages發布的具體授權範圍；未同意之前停留在規劃，不把這次提問當成開工批准。

## 14. 本輪Astra max複核與採納

Astra已完整讀完v3草稿13節、153行，並有限核對A/C實作及教材順序，支持主線及19產物轉向；此為計畫諮詢，不替代日後獨立逐節審閱。八項必修全部採納：1依賴→CPU→產品→驗收對應、12/17離散梯度分工；2新版灰階影音契約／局部時間支持／雙座標；3單碼本、freeze後Talker和真A輸出全鏈；4共享記憶體／byte限制／none集合語義；5B適配／C可觀測資訊／A與聲音家族切分／新回歸面板；6最低12正式jobs、主要比較限定三seed、60僅cap；7P1有限CPU預檢必先受審finance／HF及續訓儲存；8研究完成與產品升版分開。

保留的取捨：MoE Talker／完整多碼本／通用人聲不列最低產品；完整TMRoPE機制在CPU教，但無空間變化的正式A任務只論時間部位；B不因題材新而強行重訓。原有歷史權重與結果保留，新模型另版。真正待使用者審閱的是此產品邊界、主線改寫與新訓練／外部儲存／發布範圍，以及最多US$60新round提案；目前未發現需要補新API key。
