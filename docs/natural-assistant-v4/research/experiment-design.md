# 補資料重訓的實驗建議

這是一份獨立研究建議，不是已執行結果、選模決定或新增 GPU 授權。核對日期為 2026-10-04；作者實際讀了既有 manifest、plan、rubric、selection rule、core、CLI、Modal runner 與 Actions workflow，並取得官方原始碼。研究只做了 CPU 圖像處理與 API 檢查，沒有模型 forward、GPU 或新 holdout 輸出。

這份建議寫於 runtime 修改之前；可執行的受限選項、checkpoint 支持與實際更新量算法以 [runtime handoff](runtime-handoff.md) 為準。它保留提出方案時的觀測與因果限制，不是目前 runner 功能清單。

## 先解決什麼

優先做 **相同 2B 底座，加上符合用途的新訓練資料**。目標是確認這套資料／訓練方案是否改善產品任務；不能把這項實验稱為「已證明唯一根因是資料量」。新資料同時改善語言、題型與覆蓋，這幾個因素仍一起改變。

既有訓練的確有可核實的缺口：272 筆 record 中，120 筆照片 target 是英文原 caption 開頭片段；96 筆是合成 OCR、24 筆是文字有無；30 筆單輪聊天只來自 10 組各重複三次，另有 2 組多輪。讀過的產品問題則要求繁中照片關係問答、自然場景文字與聊天。前版只看 360 次 record，監督 token 約 5,791。這支持先補匹配資料，沒有足夠證據說容量、frozen vision、greedy 或學習率是單一原因。

前版曾比較 1e-4 與 3e-5，後者沒有勝過前者；不應再把相同降 LR 對照當作進展。固定相同底座與架構能省下載、保留可比基準；4B 是有條件的後續選項，不能拿不同資料、像素、解碼與參數量一起更換，再將全部改善歸功於模型大小。

## 資料設計與最少必要檢查

先選授權與來源可確認的公開資料；dataset card 的授權標籤不足以替混合 corpus 裡每張圖片取得新授權。訓練使用、標註、衍生翻譯與原圖片再發布分別核對。可以只發布 ID、來源／固定 revision、checksums、派生器與合法下載步驟，不必為了教材把所有原圖再發布。

建議第一份可實跑資料以 2,000–4,000 個 training records 為量級起點，而不是追求最大下載量。這是成本設計起點，不是學會的最低門檻。按 task、獨立圖片／對話數及實際 supervised token 數一起公布，不能將同圖多問或重複對話算成相同數量的獨立經驗。

| 訓練內容 | 必須符合的用途 | 要避免的替代品 |
| --- | --- | --- |
| 照片 | 繁中物件、動作、關係、計數、短描述；每個答案由原始標註或實際看圖確認 | 只翻譯同一句「這是一張……的照片」；由 caption 猜畫面中未提及的細節 |
| 自然中文 OCR | 可讀的招牌、物件文字、不同字體／背景／視角，包含整張與明確指定區域；無文字樣本要有 | 只有合成乾淨字卡；只練裁切字塊卻聲稱學會整張圖的文字定位與閱讀顺序 |
| 聊天 | 多樣的簡短回應、指令、改寫／整理、情境與先前約束，含真正不同的多輪對話 | 用 10 組模板重複成幾千筆；把「一般聊天」評成答案重述比對 |
| 語音路徑 | 實際說話的問題／請求與簡短回應；另保留一般敘述作不同題型 | 將朗讀新聞的成功率命名為自由語音聊天成功率；以正確稿取代 ASR 假設 |

公開英文照片標註可以做繁中衍生，但原標註、翻譯／QA 產生方法、衍生標籤作者與檢查方式要保留。翻譯不會自動把沒有的動作／位置標籤變出來。自然 OCR 採 region annotation 時，整圖與 crop 的父圖片屬同 family；多頁文件用原文件 family；聊天用原 conversation family。變體、裁切、翻譯、同圖多問及展開的多輪必須一起分 split。再查實際檔案 SHA 與可用的近重複圖／重複對話，不靠 row ID 不同假定獨立。

現有 core **只監督最後一個 assistant 回覆**，之前的 assistant 當 context。可將一段公開多輪對話展開成每個 assistant turn 各一筆 record，保留之前有效 context；不能只將全篇對話塞進 history 然後以為所有回答都被訓練。先 CPU 編碼所有 rows，查 prompt/full-prefix 一致、使用者與 history 的 label 均為 -100、回答含正常結束符號、shift 後 supervised tokens 大於零，以及總長度沒有超限。拒絕／過長／缺圖數也保留。改成監督所有 assistant turns 是另一種合理方案，但需要新的 mask 測試與明示實作。

## 影像與解碼的可比條件

現有底座固定 `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`，vision 與原 LM 凍結，28 層語言 q/v rank-8 LoRA、alpha 16，共 1,605,632 可訓練參數。官方訓練框架明確有 `tune_mm_vision=False`／`tune_mm_mlp=False` 的選項，frozen vision 本身不是程式錯誤。它也不能恢復 resize 已丟掉的字形；是否需要訓練 projector／vision，須在匹配資料與有效解析度後另比較。

現有 processor 上限 524,288 pixels、下限 65,536；sequence 上限 2,048，生成上限 384，greedy。實際 Transformers 4.57.6 的 Qwen3VLProcessor 使用 AutoImageProcessor／Qwen2VLImageProcessor 路徑，merged image tokens 為 `image_grid_thw.prod()/merge_size²`。底座 patch=16、merge=2，面積除以 1,024 是近似量級；**必須記錄每張原圖實際 grid**，不能將 512 或 1,024 當所有圖片的固定 token 數。

這次 CPU uniform-image probe 沒有模型推論：1024×2048 原圖在 524,288 cap 下變 512×1024／512 image tokens；在 1,048,576 cap 下變 704×1440／990 image tokens。2000×2000 圖的兩種 cap 則得到 484 與 1,024 tokens。影像縮小後，細字可能不可辨；增加 cap 不會證明 OCR 變好，還會增加圖像與多輪 history 的 token／記憶體成本。

最省錢做法：先在 **開發集**做固定底座／固定 decoding 的小幅像素診斷，事前指定兩 cap 和同一組自然 OCR／關係題。若需要換 cap，將產品比較基準與所有 candidate 一起用相同新 cap，訓練前凍結。此時產品基準是新設定下的 base，不是前版數字；舊設定可另外當回歸。若要獨立識別解析度效果，只比較同一模型／prompt／解碼的兩 cap，不能與 adapter 效果混稱。

Qwen 官方 model card／vLLM 評測示例建議 sampling 與 `presence_penalty`；現有 HF 4.57.6 `generate` 的 `_get_logits_processor` 實作有 `repetition_penalty`，沒有該 presence penalty。CPU probe 更核對 `GenerationConfig(presence_penalty=1.5)` 能保存額外 attribute，但保存參數**不代表 generate 使用它**。不要直接抄 vLLM 參數。先維持 greedy 做訓練比較；若對照解碼，事前固定 supported knobs、種子與多次樣本數，base／adapter 配對比較，別試到某個幸運答案就公布。

保留完整 raw token IDs、原始 suffix、EOS 與 stop reason。切掉重複尾巴、縮小 max_new_tokens，或者遇迴圈提前停止，都不是完整答對。部署可以加防迴圈，評估仍要記錄原始回應與停止原因，並獨立判斷資訊是否完成；OCR 裡字元重複可能本來就是正確內容，不能用一般去重器美化成績。

## 一次足夠資訊量的重訓

這輪 v4 用 fresh adapter。現有 resume contract 拒絕 manifest／assets／LR／seed 變更，因此不能拿 v3 checkpoint 當新資料的無縫 resume。第一次實跑前要將 LR 變成明示配置：目前 Modal train 是硬編 `3e-5`，而已發布的原 candidate 是 `1e-4`。主比較建議固定既有勝出 candidate 的 `1e-4`、q/v rank 8、alpha 16、grad accumulation 2、AdamW、clip 1、seed 42，不同 LR 只能再列不同因子。

採兩個事前宣告的訓練量候選，例如完整資料 1 與 2 輪；2,000–4,000 rows × 2 輪／grad accumulation 2 對應約 2,000–4,000 updates。row 長度不同，不能承諾沿用舊版約 150 秒速度。先測最大 image／history 案例與小批實際吞吐，確認在硬 timeout 內完成。訓練量用 **實際 unique families、row visits、shifted assistant tokens** 記錄，不能只寫 updates。

最省工作是同一次 train 保存兩個不可變 checkpoint（約 1 輪與 2 輪），再同一 validation stage 比 base、兩個候選。現有 save_checkpoint 只保留最新 adapter；要明確加 checkpoint archival，保存所有推論與 resume 所需檔案／hash。若不改 runner，先一輪 train→validation，再決定是否 resume 到第二輪；不可宣稱現有 runner 已支持一次評兩候選。保持每個 checkpoint 的 provenance、optimizer／RNG、續跑語意與 train／validation 隔離。更新時間 soft cap 不等於整個 container 硬 wall time。

若訓練時間／資料編码預算不容兩輪，先凍結可完成的訓練量，不在看到 final 後追加。train loss 降低不能作為部署選模條件。

## 選模、回歸與新測試

前版 final／CC-OCR 10 張已經被看到，全部只能叫 **已知回歸集**。新 test 在輸出之前按 source image／document／conversation family 固定，另留至少一種未用於微調的資料來源或拍攝條件作轉移測試。公開 test 可能已被底座 pretrain 看过，不能承諾 pretrain 去汙染；這裡保證的是本專案微調／選模隔離。新 test gold 在實際看圖或原始人工標註基礎上先寫，不能開輸出後改 rubric 讓 candidate 過關。

可負擔的 validation 起點是照片 24 題、自然 OCR 24、文字有無 12、一般單輪／多輪聊天 20，另 8 段語音各生成 typed-reference 與真 ASR 的兩個回應，共 **96 回應／variant**。這些是暫定數字，來源到齊後必須在首次輸出之前確認，不是現在的資料現況。final 可以相近大小，增加 source／條件的不同，不用反覆擴測到指標好看。

語音三項分開：ASR 原始／明示正規化 CER；給正確稿時 LM 回應品質；給原始 ASR 假設的端到端回應。ASR 先在 variants 共用同一次實際轉寫。正確逐字稿表現差時先修聊天 supervision；正確稿已好但假設差時才花下一筆錢比較 ASR。仍用 frozen Whisper-small 測這輪匹配聊天資料的效果；不必同時啟動 ASR 訓練、4B、PPO 或 DPO。

OCR 保留整串 exact、micro CER、閱讀順序及正常完成率。整串 exact 不是「有認出任何字」；CER 大於 100% 可以是亂加字，不代表計算錯誤。原資料 GT 缺漏、大小寫、標點、繁簡與空白政策在輸出前註明；若有人視覺核對多種合法轉寫，另凍結 accepted alternatives，原來源 GT 指標仍另外保留。不要只看 keyword proxy；照片／聊天用來源支持、沒有新增虛構事實、符合指令、約束延續等先定語意規則，由看過原圖／context 的獨立 reviewer 看完整輸出。

建議事前採用的 **相對**選模門檻：各 domain success rate 的等權平均必須嚴格高於 paired base；照片、自然 OCR、文字有無、聊天四類各不退超過一題；typed／ASR 路徑各不退超過一題；正常完成率不降低；語意 hallucination 案例數不能增加。並要求至少一個弱項增加兩個完成且正確的回應，避免只靠飽和字卡多一題選模。兩個 eligible checkpoint 同分選較早／较小訓練量，均沒贏選 base。這是有限教學用途的偏好，沒有統計顯著或廣泛產品可靠度保證。

「可部署 candidate」與「可以說用途已達標」分開。小考即使比 base 好，也不能稱可靠讀取任意招牌／任意人語音。若要給學生一個有用的狹義成品，先定可重現題型，例如可讀清楚短中文字、單一問題與短照片 QA；對每類顯示 numerator／denominator 和上限，沒有通過的能力不包裝成已完成。

freeze 檔至少綁 model／processor pins、資料／file SHA、family split、label 來源、rubric、生成設定、checkpoint schedule、選模規則、評分完成條件；先 commit 到遠端可辨版本，再執行。validation 後另 commit selected exact weight／raw result SHA，再執行一次 new test。看到 new test 後不重新選或微調；要再訓練就另版並承認此 test 已變開發資訊。

## 預算與執行順序

最後已確認的保守 cumulative reservation 是 US$24.81／40.00，差額 US$15.19，不是實際帳單。執行前讀既有 durable ledger 與即時價，不重設 prior spend。現有每 L4 stage 最壞預留約 US$1.79，CPU prepare 約 US$0.62、release 約 US$0.58；這是之前的規格／價格，不是保證未來價。

先預留完成所需的 test／release，再選可選比較。以至多 6 個 L4 stages、2 個 CPU stages、1 個 release 為研究規劃上限，過去 rates 約新增 US$12.56、cumulative US$37.37，留下 US$2.63 保守餘量。重試也消耗一次完整 reservation。現有 ledger guard 必須維持序列、每 attempt 新 run ID、無 retry、GPU 不含 HF token；image build 前 reserve、每分鐘／checkpoint persist、finally backup／finish、private evidence 與公開 inference allowlist 分開。

建議順序為 CPU legal/data/encode audit→CPU prepare→一個有 paired base、像素或最大案例診斷的開發 GPU job→匹配資料 train→paired validation→如驗證指向再用一個候選 job→事前選模→new-test／已知回歸各明示→release／學生試用。4B 或不同 ASR **二選一**，只有已見的開發證據支持且 final／release 餘額足夠時啟動。任何超時、OOM、partial evaluation 都原樣保留，不把 partial 當 complete。

教材重新寫成一條「能力需要哪些輸入、資料、訓練與考卷」的主線。原始紀錄與可重現失敗放 evidence；主文只留能說明關鍵知識的最少一兩個對照，不列多輪救火歷史。依最後真結果描述成品、限制與操作方式。不能為了刪工程紀錄把仍存在的能力限制從產品介紹刪掉。

## 官方依據與此次實際檢查

- [Qwen 官方訓練說明](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/README.md)：多輪格式、vision／MLP／LLM 開关、解析度與訓練設置。文件示例包含其他 Qwen 型號與 full tuning 設置，不能照抄成此 q/v LoRA 最優 LR。
- [Qwen 官方評測設定](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction)：明示 vLLM、sampling 與 presence penalty，不是本 HF runner 的直接等效 API。
- [HF 4.57.6 Qwen3VLProcessor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)、[實際 image processor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)：grid／merge 與 smart resize。
- [HF generation config](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/configuration_utils.py)、[generation implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py)：supported generation 路徑。
- [來源與 SHA 紀錄](../../natural-assistant/evidence/v4-research/experiment/receipt.json)、[CPU API／uniform-image 實際輸出](../../natural-assistant/evidence/v4-research/experiment/cpu-api-probe.json)。下載原件與 installed source SHA 相符；未測真正照片／OCR 品質，沒有模型載入或 GPU。

