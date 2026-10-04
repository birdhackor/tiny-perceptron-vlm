# 中文對話與一般語音資料：v4 來源研究

本文件是資料與授權研究紀錄，不是學生必讀教材。研究者為
`/root/natural_v4_research_chat_speech`。本次直接下載、閱讀官方 dataset/model cards、
專案授權與貢獻者使用條款，並實際解析 OASST2 的公開訊息；沒有進行 GPU 訓練或語音辨識。
來源快照、下載指紋及完整機器可讀結論在 [chat-speech.json](chat-speech.json)。

## 建議採用：原生中文 OASST2 對話，加精確繁中指令示範

優先使用 [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)，
固定 revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`。
官方 card 對**資料集**明列 Apache-2.0，並描述 open-assistant.io 收集的樹狀對話；
它不是只有程式碼授權。兩個使用者回覆分支並非两段獨立對話，必須以整棵樹的
`message_tree_id` 分組，再決定 train/validation/test。

本次實際匿名下載 `2023-11-05_oasst2_ready.messages.jsonl.gz`：

- 54,301,900 bytes，SHA-256 `a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751`。
- 實際解析 135,174 條訊息；其中 8,615 條標示 `lang=zh`。
- 中文部分包含 3,199 條使用者訊息與 5,416 條助理訊息。
- 8,615 條中文訊息的來源 `synthetic` 欄位全部為 `false`。
  這是來源註記，不能當成研究者獨立確認每條文字皆為真人原創的證明。

篩選接受且未刪除、無來源標示個資、無垃圾訊息或不適當內容、首選回答、適度篇幅，
並要求完整祖先鏈皆通過，可得 **259 條候選對話路徑、218 棵不同對話樹**：
218 條兩訊息路徑與 41 條四訊息路徑。此處是自訂研究筛選，**不是官方 train split**。
兩輪路徑會包含首輪，因此不能把 259 當成完全獨立的 259 種題目。

候選檔暫存在 `/tmp/natural-v4-oasst2-zh-candidates-relaxed.jsonl`，
389,345 bytes，SHA-256 `d231b75eda44a2583b91f8c921573d182278df89fcc450274b715c8648b6a924`。
原始 54 MB 壓縮檔與完整中文訊息也只在 `/tmp`，沒有加入普通 Git。

### 品質標籤不能取代實際閱讀

研究者完整抽讀了多個候選回答，发现社区给高分的文字仍可能有事實錯誤、
漏掉指定結尾、冒稱能收集即時資訊，或不必要的助理身分自述。
因此，259 條只能當**候選池**，不能直接宣稱是乾淨高品質的训练資料。

已完整閱讀、判斷適合作為低風險例子的 14 條放在
[chat-speech.samples.jsonl](chat-speech.samples.jsonl)。每條保留 source tree/message IDs，
不保留 user ID、事件、表情投票等無用資訊；訊息文字仍為來源原文，包含簡繁混合與字形錯誤。
例子包含缺文章時要求文章、缺組會主題時先澄清、五項學習建議、一般生活規劃等。
這只是研究样本，尚未宣告训练/驗證切分。

正式資料建議採以下規則：

1. 實際檢查回答是否完成要求，排除時效資訊、法律／醫療細節、錯誤能力或身分自述。
2. 繁中派生版可使用固定版本 OpenCC，再人工检查用語、專名與指定字串。
   翻譯或改錯後必須保留原文與變更說明；不要說转换过的文字是來源原文。
3. 優先保留每棵樹最長的合格分支，避免首輪另存又在第二輪重复加权。
   若保留多个分支，明确记录共享根與取樣权重。
4. 先做全樹切分、內容去重，再依正式 tokenizer 長度篩選。不要把首輪放 train、
   同一棵樹的追問放 validation，也不要默默切掉過長的答案尾巴。
5. 對所有最终评估題目做 exact/near-duplicate 排查，不能把上一版公開測試答案變成新训练示範。

資料散布時保留 Apache-2.0 授權文字、OpenAssistant/LAION 與貢獻者归属，
並明列筛选、移除欄位、翻譯、修正等變更。随附
[授權文字](chat-speech.Apache-2.0-LICENSE) 與 [來源說明](chat-speech.NOTICE.txt)。
OpenAssistant 官方 ToS 已按 Git revision
`f1e6ed9526f5817531f3ab85441a40b3671ddccb` 閱讀；它描述收集训练互動式助理的科學目的，
並沒有另写资料集非商业限制。本研究依官方资料集 Apache-2.0 授权使用，
不是依据该 ToS 对著作权成立与否的概括断言。

### 補足聊天弱點，不能只學「你好」

OASST 可提供不同說法、生活問題、澄清與上下文風格；另外應準備清楚標示为
**本專案編寫**的繁中精確指令示範。這不應冒稱是從網路收集的真人對話。
建議涵蓋：

- 给一小段来源，要求只用一句话中性重述，不添加来源没有的事实。
- 缺來源、指代不明、無法判定時，問一個必要的澄清問題。
- 恰好兩點、指定字串、只給指定內容等可客觀檢查的指令。
- 首輪给背景、後一輪使用背景，再修改格式或長度的多輪任務。
- 对陳述作出合適回應，而不是沒有证据就回「這句話是錯誤的」。

若從 FLEURS 衍生內容重述示範，只使用其官方 **train** 逐字稿。
來源逐字稿與本專案新寫的聊天回答分開保存；读稿并不会自動有标准聊天答案。
若要教模型承受 ASR 小錯，可先用選定 ASR 處理 train 錄音，把錯誤逐字稿作為額外
輸入、另加來源忠實或澄清的回答；不要依最终测题补答案。
严重识别不确定时，数据应示范澄清，而不是凭常识猜名字或数字。

## 建議先比較的 ASR：Whisper large-v3-turbo

首選低整合成本候選是
[openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)，
固定 revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`。
官方 model card 明列 **MIT**；OpenAI Whisper 原專案授權亦為 MIT。
學生可匿名從固定上游下載，不必再散布整個基礎辨識器；若複製權重或程式則保留其授權文字。

本次實讀 pinned config 並用現有 Transformers 4.57.6 在 meta device 實例化、绑定共享權重：

| 項目 | 核實內容 |
|---|---|
| 不重複參數計數 | 808,878,080 |
| 結構 | 32 層 encoder、4 層 decoder、128 mel bins |
| `model.safetensors` 大小 | 上游 LFS API 1,617,824,864 bytes |
| 權重檔 LFS SHA-256 | `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1` |
| 與既有實作關係 | 同 `WhisperProcessor` / `WhisperForConditionalGeneration`，CLI 已有模型與 revision 參數 |

計數並未下載權重，也不是 RAM、延遲或辨識準確率實測。
現有辨識器 loader 使用 float32，單參數儲存约 3.24 GB；整體推論還有 activation、
快取與聊天模型，不能把这个數字說成顯存最低需求。
官方說 decoder 從 large-v3 的 32 層減為 4 層，推論较快且品質略有代價；
本教材是否改善仍須用本教材的 validation 音訊實測。

比較流程應先 freeze 逐字稿规范与评估集，讓原 small 與 turbo 处理同一组 validation 录音。
决定之后再跑新的 held-out 音讯，不根据上一版最终12题反覆挑模型。
必须分开记录 ASR 逐字稿错误、聊天回應是否适当，以及輸出是否正常停下。

原始 CER 与 NFKC／空白处理后的 CER 分开报告。
若另做簡繁统一，作为具名的额外诊断，不能悄悄把 ASR 错误逐字稿換成标准答案。

## 次選：Qwen3-ASR-0.6B

[Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)，
固定 revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`，官方 model card Apache-2.0。
上游 `model.safetensors` 是 1,876,091,704 bytes，
LFS SHA-256 `79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`。
卡宣稱支持 Chinese、30 種語言與 22 種方言，但这些是作者測試，不是本課結果。

官方 `qwen-asr` pyproject 当前版本 0.0.6，要求 Transformers 4.57.6、Accelerate 1.12.0，
與現有版本相容；但另外有自訂架构注册、`qwen-omni-utils`、librosa 及其他依赖。
它不能只把 Whisper ID 換成 Qwen ID 就載入，應用隔離环境整合。
本次沒有下載或實際執行其權重，所以仍只是比較候選。

## 語音資料的適用範圍

已直接核查官方 [FLEURS](https://huggingface.co/datasets/google/fleurs)
固定 revision `d7c758a6dceecd54a98cac43404d3d576e721f07` 的 CC-BY-4.0 card。
既有 bundle 裡的三筆 train WAV 也實際用 SoundFile 查看檔案、SHA、16 kHz、单声道与長度，
並閱讀來源逐字稿；**沒有聆聽波形，也沒有重跑 ASR**。
結果在 `../../natural-assistant/evidence/v4-research/chat-speech/fleurs-train-small-actual-inspection.json`。

這是一般人的朗讀聲音，適合先驗證普通華語辨識與輸入通路；
它不是人跟助理自由聊天的資料。官方 card 說 train speakers 與 dev/test 不同，
但這個小 bundle 沒有個別 speaker ID，不能自行證明其他人的精細说话者切分。

如果日後要涵蓋自然談話與多人重疊，可考慮官方
[AliMeeting / OpenSLR 119](https://www.openslr.org/119)，資料授權 **CC BY-SA 4.0**：
104.75 小時 train、4 小時 eval、10 小時 test；真人普通話會議，2–4 人、每場 15–30 分鐘。
它更貼自然談話，但仍是會議逐字稿，不是助理回答標註。
近場 train 22.85G、eval 3.42G、test 8.90G；本次只讀官方來源，沒有下載或聆聽。
應以整場會議切分，再切短音段；切片与转写改动的再散布保留归属及 share-alike。
在剩餘費用內，先改善成熟 ASR 比增添大型會議 corpus 的训练工作更直接。

## 沒採用的來源及原因

| 來源 | 實讀後的原因 |
|---|---|
| COIG-CQIA | 官方 card 的 License 是「More Information Needed」，資料從中文互联网問答與文章取得；不能拿開放下載當作数据授权。 |
| SenseVoiceSmall | 官方 card 是 `other/model-license`，链接 FunASR 自訂 MODEL_LICENSE；不是 Apache 權重，含行為、終止與自動更新條款。 |
| Taiwan-Tongues-ASR-CE zhtw | 实际 LICENSE 为 TRAIL 0.1，区分模型/程式/資料 copyleft 与使用限制，不是標準政府開放資料授权。 |
| TaiwanChat | CC BY-NC-4.0，此輪選擇较容易开放散布的替代來源。 |
| SpokenWOZ | 官方资料 CC BY-NC-4.0；主要英文 task-oriented 对话，语言与开放散布需求不如其他候选直接。 |
| Common Voice 非官方 mirror | Mozilla 已将官方资料移到 Data Collective；当前官方 FAQ 解释新条款与撤回同意的治理问题。不能只凭旧 mirror 卡的 CC0 标签作新分发的依据。 |
| ryL/Taiwan-mandarin | 已读 card 没有资料授权。 |
| Dolly 15k | 合法的 CC BY-SA-3.0 人工指令资料，但英语且需翻译、归属及 share-alike；此轮优先原生中文。 |

以上是针对本輪选取工作的判断，不是对这些项目全部应用情境的法律断言。
所有原始 card、官方條款、固定模型版本、下載大小與篩選紀錄均可由本研究 JSON 的 source files 核對。


## 本輪已完成的資料補充（不是模型訓練結果）

已完整閱讀並精選 **60 個原生 OASST2 對話 tree**，另撰寫 **50 個原創繁體中文教學指令／對話**。其中 30 個原創對話包含真正依賴前文的改寫、訂正或狀態更新；每個 assistant turn 各形成一列，只監督目前這次回答。全套共 **144 列：train 122、validation 9、test 13**，110 個 tree 按來源分隔。兩個字串反轉例及兩個同型四句短詩也放在同一 split，避免明顯近似例跨組。原生資料每個 heldout split 各保留一個依賴前文的 tree；原創資料全部只用於 train。社群的 synthetic=false 是來源註記，未獨立證明全部文字皆為人類原創。

[chat-sources.json](../../natural-assistant/v4/data/chat-sources.json) 記錄來源與切分；[逐筆校正 ledger](../../natural-assistant/v4/data/chat-curation-ledger.jsonl) 保留各處改動，[原始訊息快照](../../natural-assistant/evidence/v4-research/chat-speech/oasst2-selected-original-messages.jsonl) 保存 source message ID 與原文。轉成繁體時不把『請轉換下面簡體字』的輸入一起轉掉；字元反轉題則重新由轉換後輸入計算答案。誤把重量等同質量、把 JSON 值全稱為字典、漏掉奇偶性定義域條件等處已明確改正。這是首次資料編輯，仍應由後續獨立審查確認品質。

已實跑固定 Qwen3-VL-2B-Instruct revision 的 processor 與目前 encode_training_row，所有 144 列的 exact prefix／目前回答遮罩檢查通過，最長 **618 tokens**，沒有截斷、没有下載語言模型權重，也沒有在這個子任務使用 GPU。

真人語音除 FLEURS 的 **12 validation／18 test 朗讀陳述**，另從官方 **AISHELL-1（Apache-2.0）** 選了 **4 validation／4 test 真人朗讀問句**。AISHELL 每題來自不同 speaker，兩組不重疊，題目、回應條件及語意 rubric 在 ASR／模型輸出前固定。它們能用於比較同一問句的 gold 文字路線與實際 ASR 路線，但不能代表自然、自發的語音聊天成功率。所有音檔只查了 SoundFile metadata 與官方逐字稿，沒有聆聽；[voice-question-sources.json](../../natural-assistant/v4/data/voice-question-sources.json) 如實記錄此限制。

需要打包的 **38 個 holdout WAV 共 23,086,966 bytes**。FLEURS 另外 36 個本地 train 音檔未用來訓練 ASR，預設不打包。scripts/replay_natural_v4_voice.py 可由固定來源 URL 與 tar member 重取相同檔案，不重新挑題；每一 WAV 均核對精確 SHA256、bytes 與聲音規格，並保留授權歸屬。原始有界串流下載已成功，本輪重新獨立下載遇官方 Hugging Face HTTP503；現有 38 檔的重新 checksum 檢查已全通過，不能把 cache 重查宣稱成新下載驗證。
