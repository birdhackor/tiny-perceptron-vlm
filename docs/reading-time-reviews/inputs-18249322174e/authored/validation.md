# 教材實作驗證

教材本機驗證環境為 Linux、Python 3.13.5、PyTorch 2.14.1 CPU；正式訓練另使用 Modal NVIDIA L4、Python 3.13.3、PyTorch 2.14.1+cu126。本頁保留不同階段的紀錄。最新逐組訓練、自然資料短訓、私有備份與學生權重證據見[正式實驗](course-experiments/README.md)及[當前進度](course-experiments/progress.json)；早期流程測試與改寫前的審閱不能代替本輪驗收。

## 分階段訓練與整合成品

2026-10-04：新增[7.17–7.18](../course/chapters/07.md#7.17)，先說明大量文字學習與助手行為訓練分別解決什麼問題，再介紹兩階段之間可能重疊的工作。[13.10–13.17](../course/chapters/13.md#13.10)從偏好分數走到獎勵模型、PPO、KL 與 DPO，最後比較適合使用哪些方法。PPO 的可重跑 CPU 實驗使用事先寫好的回答卡，沒有把它說成新訓練的語言模型或真人回饋；原始數據見[實報](course-experiments/results/posttraining.json)。

[第19章](../course/chapters/19.md)把文字、簡單圖片、合成聲音、受限工具與助手行為接到同一個小型 MoE。正式訓練依序完成 300 步 pretrain、1,400 步 SFT、600 步圖音共同訓練與 100 步混合式 DPO。模型有 328,128 個總參數，排除每個位置未選中專家權重的啟用參數代理量為 195,776；所有專家權重仍須儲存，這個代理量不能直接當成記憶體或速度。DPO 分支留作比較，依預先固定的驗證準則選出的示範成品是共同訓練版，見[選擇紀錄](course-experiments/capstone-selection.json)。

固定測試集共有90題，共同訓練版與DPO版皆為78/90。這是自行建立的有限任務世界，沒有一般中文、自然照片、語音辨識或安全能力的品質保證；逐題結果見[共同訓練版](course-experiments/capstone-evidence/deployment/test-joint.json)與[DPO版](course-experiments/capstone-evidence/deployment/test-dpo.json)。工具請求合法、工具真的算對與模型最後答對分開評分。另以同一初始權重訓練79,920參數的Dense學生：一般交叉熵版62/90、蒸餾版61/90，沒有把較小或經蒸餾說成能力必然較好。打包int4／int8權重在推論時還原成FP32，沒有宣稱整數運算加速。

[11份整合成品權重](course-experiments/capstone-public.json)已有固定Hugging Face版本，包含各階段、壓縮版與Dense學生。實際匿名下載核對檔案大小／SHA-256、重新建模及逐tensor比對，見[權重核對](course-experiments/student-checks/capstone-public-tensors.json)；[圖片、聲音與工具操作](course-experiments/student-checks/capstone-public-ui.json)和[學生模型操作](course-experiments/student-checks/capstone-public-student.json)另保留真實成功與失敗。這些操作沒有增加正式測試集的分母。全部實驗及重跑的Modal運算預留額為US$9.84，上限US$10，見[保留紀錄](course-experiments/budget-reservations.json)；這是預留額，並非實際帳單。

## 整合內容的三輪審閱與閱讀範圍

2026-10-04：本輪新增與受影響的49節完成理解審閱後，交由另一批49個全新獨立AI任務逐節查核原論文、官方文件、原碼與實際資料。暖身修句另使一份舊報告的完整檔案依賴失效，該節再交給新的獨立審查者完整重審，沒有只替換指紋。全書274節的理解與技術報告，以及26份首節導讀，都與目前正文相符；見[本輪核對](validation-artifacts/integration-review-round-all.json)與[實際技術派工及回查](validation-artifacts/integration-technical-dispatches.json)。這是AI審閱，並非真人學生的學習成效測試。

第三批四個全新任務沿閱讀指南、分階段訓練、偏好更新與整合成品的路線閱讀，涵蓋上述49節，連同必要前置實讀193個不同小節。讀者發現13.10沒有說清分數差如何換成勝出機率；補上規則及0、2的逐步算例後，由提出問題的讀者回讀前後路線，原理解讀者與原技術審查者也真正回查。四組最後均通過，實際讀序、原文快照、親看圖解及修正歷史見[銜接核對](validation-artifacts/integration-continuity-round.json)。

248份Notebook曾逐份啟動獨立CPU kernel，完整一輪耗時490.67秒；後續改說明、顯示格式與引用時，再對受影響的15份Notebook各自重新執行，必要時保留多次回查。最終[248份輸出核對](validation-artifacts/integration-current-notebooks.json)使用233份未變的完整輪次副本及15份最新重跑副本，逐cell確認來源相同、程式已執行且無錯誤。這不代表又重跑了整套248份，也不代表Colab雲端或所有學生裝置已測試。

閱讀時間標示採用逐頁AI估計的方法，以數學基礎良好的高中生，或有基本數學能力但不夠熟的大學生為對象。範圍包含正文、圖解、範例程式、已展示輸出及短暫思考，不包含安裝、下載、親手操作、等待訓練或額外補讀前置。章導讀只算自己的前言，同一頁在總量中只算一次；時間區間不是實測平均或學習成效保證。本頁列出的程式執行耗時與閱讀估計是兩種不同的量。

## 工具選擇補充與三輪審閱

2026-10-04：在工具請求、執行、回填與停止之後，加入[B.5–B.8](../course/chapters/0B.md#B.5)。四節分開教任務策略、以短回答訓練選擇、檢查漏選與多選，以及能力／成本比較；附一張自製SVG、四份Notebook與可重跑的CPU實驗。B.1補齊COPY分組及seed／CUDA／L4解釋，B.4補上生成次數與新內容的過渡。

[tool_choice實報](course-experiments/results/tool_choice.json)記錄143,616參數、900次更新、121,297個有效回答／EOS位置，訓練約35.55秒、完整實驗約39.41秒。原模板留出數字對96/96選對；新問法總分42/48，卻有6/6應用工具題全選ASK。這是固定策略的選卡元件，沒有與原工具模型合併評估；自評能力和成本取捨只用人工算例說明。此輪沒有新增Modal工作、費用預留或HF學生權重。

依序完成逐節理解、另一批逐節技術核對、兩位不同讀者從前節順讀的銜接審閱。每輪問題修正後由原審閱者親讀回查，另獨立重查15節受目錄、實驗計畫及匯出程式變更影響的舊證據。三輪實際任務與報告指紋見[收尾紀錄](validation-artifacts/tool-choice-review-closure.json)；252節正文、圖、證據及25份導言的版本／身分核對見[全站核對](validation-artifacts/tool-choice-review-round.json)。這些是AI審閱，沒有把它當成真人學生測試。

本機先以獨立CPU kernel執行B.4–B.8，最後文字修訂後再執行B.1與B.8；其他來源相同的執行副本重新比對後重用。核對範圍見[五節執行](validation-artifacts/tool-choice-notebooks.json)及[最後兩節補跑](validation-artifacts/tool-choice-final-kernels.json)，沒有把副本比對說成全數重跑。GitHub Pages發布流程會重新逐份執行全部226份Notebook。完整本機Linux測試342 passed、1 skipped，見[當次測試](validation-artifacts/tool-choice-tests.json)。

最終本機網站260頁、226份下載與Colab入口全部通過連結檢查；[Chromium核對](validation-artifacts/tool-choice-browser/report.json)實際查看桌面1440px與手機390px視窗的B.1–B.8、C.1及SVG文字範圍。這是瀏覽器視窗測試，尚未做實體手機或Colab雲端kernel驗證。

## 本輪正式訓練與公開權重驗收

2026-10-02：30組正式實驗全部完成，對應段落已依實測修改；另完成L4上的Flash Attention輸出／梯度、實際CUDA後端、時間與記憶體探針。[實驗清單](course-experiments/plan.json)逐項連到報告；支持範圍與失敗結果都保留，沒有以下載成功或低代價推論一般能力。

30組共120份公開推論權重已固定在[下載清單](course-experiments/public-models.json)的Hugging Face版本。最新一輪以CPU完成全部匿名下載、檔案大小／SHA-256核對、模型重建與實際執行，共148個子命令；逐組證據及程式版本見[完整操作驗收](course-experiments/student-checks/all-models-validation.json)。另實際檢查數字圖片、原始服飾圖、原8kHz錄音重採樣、兩個原始LoRA接頭，以及檢索、受限工具回填和算式步驟外部檢查；這些單次操作不增加正式測試集的答對數。

第一次完整驗收發現兩份多模態教師副本缺少task宣告，檢查器如實拒絕。修復核對原教師metadata，在匯出清單明示vision／joint並增加缺漏與衝突檢查；重新發布後，六份同組模型的全部tensor與推論設定逐項相同，見[修復比對](course-experiments/student-checks/multimodal-task-repair.json)。[原失敗紀錄](course-experiments/student-checks/multimodal_distillation.json)與[原發布版本](course-experiments/public-releases/history/multimodal_distillation.pre-task-repair.json)仍可回查。修正後完整程式測試為328 passed、1 skipped，13.94秒；這不是模型品質分數。

本階段累計Modal運算預留額US$8.22，上限US$10；[費用保留紀錄](course-experiments/budget-reservations.json)包含原執行、發布與修復，不是實際帳單。本輪新讀者審閱與222份Notebook執行已完成，結果見下文；另一批技術審閱與最終網站發布仍在準備。

後續跨平台CI在Windows找到發布控制工具的兩處問題：以字串斜線比對私有目錄，沒有正確處理Windows路徑；隔離Python的中文help重導至非UTF-8 pipe也會失敗。修正採路徑元件比對並明示CLI輸出UTF-8，未更動上面操作驗收記錄的24份訓練／推論程式或公開權重。固定版本`1ac7fcbfafca30a7ec1d46d82947a16f82fb7bbe`的[實際CI證據](validation-artifacts/release-compatibility-ci.json)確認Linux與Windows各329 passed、1 skipped，macOS為330 passed；格式檢查亦通過。macOS環境檢查實際選用Apple MPS，完成256×256矩陣前向／反向，以及測試中的32×32矩陣CPU比對；這只驗證基本運算，不代表整套模型訓練或教材已在MPS跑過。Colab雲端仍未實機驗證。

## 本輪逐節理解審閱

2026-10-03：正式實驗改寫後，248節全部交由248個不同的新AI讀者任務審閱，與改寫前的身分分開。每節只提供指定正文及明確連結的必要前置；首節另外審閱導言，共25份。讀者檢查背景、術語、例子、程式解釋與練習，短CPU例子實跑、已讀SVG實際渲染查看；問題退回作者修正，再由讀者回查。這是AI審閱，尚未做真人學生測試。

理解審閱階段結束時，正文、圖、五項檢查與248份報告全部通過；新身分、25份導言及原始UTF-8指紋另由[本輪核對](validation-artifacts/fresh-reader-round.json)保存。實際派工與結果消化見[派工核對](validation-artifacts/actual-reader-dispatches.json)；最早一批協調紀錄只保存報告路徑，後續紀錄保存小節或完整報告雜湊，這份核對明列其差別，不把名稱或hash當成已看懂的自動證明。另一批全新技術審閱正在進行；若發現事實問題而修改正文，還會安排讀者回查並更新最終核對。正式發布前也需完成網站檢查。

## 本輪 Notebook 執行

2026-10-03：依完成理解審閱的正文重新產生222份Notebook，再以單一worker逐份啟動獨立CPU Jupyter kernel。全部222份通過、0失敗，耗時467.08秒；497個程式cell都有執行序號，共保留497個輸出區塊。逐份原始檔與執行副本的SHA、套件版本和結果見[當次核對](validation-artifacts/reader-round-notebook-kernels.json)。

核對另逐cell確認執行副本與當前教材內容相同，沒有錯誤輸出。這份紀錄對應理解審閱後的版本；技術查核若再修改某份Notebook，該份仍須重新執行。尚未在Colab雲端執行，也尚未完成本輪網站發布。

## 早期 GPU 連線與 HF checkpoint 流程測試

2026-10-02：透過 GitHub Actions 啟動 Modal NVIDIA L4，使用 PyTorch 2.14.1+cu126，
以既有 `scripts/train.py --train` 訓練 31,584 個參數的文字模型。
資料為課程自建的顏色、形狀與左右位置規則；這次驗證訓練流程，不推論一般語言能力。

| 檢查 | 實際結果 |
| --- | --- |
| GPU 與更新 | CUDA、NVIDIA L4；80 步訓練，權重確實更新，梯度有限且非零 |
| 訓練 loss | 5.624956 → 0.986167 |
| 驗證 loss | 5.683379 → 1.317075 |
| 訓練迴圈 | 1.256 秒；包含逐步記錄與第 40／80 步存檔，不包含映像建置、GPU 啟動、HF 傳輸、迴圈結束後存檔與續訓時間 |
| 中途 checkpoint | 第 40 步，包含 optimizer、Python／PyTorch／CUDA RNG |
| HF 上傳下載 | 私有 repo 寫入成功；依指定 commit 重新下載，SHA-256 完全一致 |
| HF 檔案續訓 | 從第 41 步跑到第 80 步，與不中斷訓練的最大權重差為 0 |
| Modal Volume | 另一個 CPU container 可讀到相同 checkpoint，SHA-256 一致 |

相同 80 步配方在本機 CPU 的單次計時為 0.214 秒；這次 GPU 迴圈花費約 5.87 倍時間。
模型只有 31,584 個參數、batch size 4，且迴圈包含記錄與存檔；CPU 寫入本機目錄，
GPU 寫入 Modal Volume。這是兩個環境的一次流程觀測，不能當成純運算效能或正式訓練的速度倍率。

實測原始碼 commit 為 `b75ae446797e138a7ab733ae3c9b02d5ab7fbda4`。
[成功工作紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37028829401)、
[原始結果 artifact](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37028829401/artifacts/11236417007)
與 [JSON 證據副本](gpu-smoke-result.json) 可核對完整數值與 HF revision。
操作方式見 [GPU 指引](gpu-training.md)。

checkpoint 位於 HF 私有 Model repo `birdhackor/tiny-perceptron-checkpoints` 的
`smoke-tests/gha-37028829401-1/training/`。第 40 步上傳 commit 為
`307c7cef6234fe9e853448175a82fa727c0d5c70`，續訓完成上傳 commit 為
`3bcd0e9ab3e170a06d728e8a0a4e63e43cab0f00`。
這次早期測試對公開教學模型 repo 僅驗證可讀，沒有執行學生權重發布。
它未涵蓋圖音訓練、大模型記憶體需求、長時間中斷恢復或自然資料能力；後續實驗須各自核對，不能由連線測試推論。

## 正式訓練前一輪的改寫與圖解驗證

2026-10-02：重寫222節正文，以及7節暖身、4節閱讀指南、4節名詞說明與11節訓練操作，共248個讀者小節。前置材料改成可點擊的具體小節；正文用例子說明問題、數字與練習，圖解保留在對應說明的位置。

| 檢查 | 實際結果 |
| --- | --- |
| 獨立讀者審閱 | 248份、248位不同任務的獨立AI讀者報告全部通過；目前正文、直接圖解與報告記錄的必要前置SVG皆通過SHA-256核對 |
| Notebook執行 | 最終222份副本由獨立CPU kernel執行，222通過、0失敗；3個worker，共161.64秒 |
| 圖解 | 72張自行繪製SVG資產，本次新增32張且全部有正文引用；正文按需引用41張；有圖的小節另核對圖中位置、數字與箭頭 |
| 手機版 | Chromium以390px寬度檢查最終254頁，無頁面水平溢出、破圖或重複段落ID；實看第一節、暖身、聲音、int4、重疊頻帶與蒸餾溫度圖 |
| 圖解放大 | 手機可放大並在圖內左右滑動，縮回後恢復全圖；頁面本身維持390px |
| 桌面與輸出 | 1440px桌面與390px手機均能閱讀實際CPU輸出；搜尋「量化」找到5節 |
| 網站與下載 | 254頁、222份下載與222個Colab入口、61,230個內部連結、637個Notebook閱讀連結全部核對通過 |
| 前置跳轉 | 手機實際從1.1點W.2連結，抵達暖身正確標題與段落，再返回原小節 |
| 格式 | 最終Ruff lint通過，300個Python檔案的format檢查通過，Git空白檢查通過 |

上表記錄的是正式訓練前一輪，不能算成本輪修改後的新審閱。當時獨立AI讀者只取得分配的小節與正文明確連結的前置，審閱背景、術語、例子、程式解釋及練習；遇到問題會退回作者修正，再由原讀者核對。這是AI審閱紀錄，不是實際學生的使用測試。發布流程會拒絕缺少審閱、要求修訂、正文或圖解已改動的舊報告，也會拒絕未執行、失敗或來源不符的Notebook輸出。

## 初版程式與資料驗證

以下為2026-10-01的初版驗證，只描述當時的核心模型與CLI，不代表後續新增實驗程式也已由這32項測試涵蓋。

| 檢查 | 實際結果 |
| --- | --- |
| 大綱編號／正文／Notebook 對應 | 222 節完整，建置同步檢查通過 |
| Notebook 快速執行 | 222 通過，0 失敗；每節使用新 namespace |
| 獨立 Jupyter kernel | 222 通過，0 失敗；每節由乾淨 kernel 開始 |
| 核心離線測試 | 32 passed、1 skipped；含4項資料包檢查，略過項目為 CUDA／MPS 裝置測試 |
| CLI 操作 | 32 條路徑通過，0 失敗；訓練入口均未加 `--train` |
| JupyterLab 啟動 | 實際 `/lab` 回應200，API包含 tiny-perceptron kernel；測完停止自行啟動的伺服器 |
| 初版圖解 | 40 張自製 SVG；抽樣光柵化檢查繁體中文、圖形與位置 |
| 離線閱讀版 | 222 節網頁與操作文件匯出成功，可核對內部連結與圖檔 |
| 規則資料 | 5 種 JSONL 的原始 SHA-256 與家族切分核對；另有100種字串家族的 OCR 產生器 |
| 依賴與格式 | frozen CPU／Notebook 安裝、離線 lock 檢查、Ruff lint／format 檢查通過 |

重跑會在 `outputs/notebooks/validation-kernel.json` 產生當次kernel報告，逐節執行副本也保存在該資料夾。CLI 報告為 `outputs/cli-validation.json`。這些工作區產物被 Git 忽略；新 checkout 可用下面指令重新產生，當前檔案可能已由新一輪執行更新。

## 改寫前的網站發布驗證

2026-10-02：加入 Colab 自動設定後，重新以獨立 CPU kernel 執行222份 Notebook，222通過、0失敗；核心測試仍為32 passed、1 skipped。匯出254頁並核對222個 Colab入口、222份下載與60,338個內部連結。發布程式也確認會拒絕過期原始碼、未執行與失敗的結果，並轉義輸出文字中的HTML。

當時Chromium檢查1440px桌面與390px手機版；搜尋「量化」顯示5節，手機頁面無水平溢出，並可見兩個練習／下載按鈕。離線公式降級保留TeX文字。檢查也發現SVG動畫將邊框筆畫繼承到文字，已改成只動畫矩形邊框並重新產生圖解。這次網站檢查未執行Colab雲端或GPU訓練；後續Modal實測另有自己的證據，也不能替代Colab實機驗證。

## 測試核對哪些容易出錯的事情

- 遮住未來、padding 不直接算 loss、左側 padding 的位置，以及第一個答案的 shift。
- 手寫 attention 與 SDPA 的輸出／梯度、cache 與完整重算、RoPE 和不同 KV head 數。
- checkpoint 接續的下一次數值更新、top-1 router 的有效梯度、LoRA 零初始化與合併。
- 多模態展開後的答案位置、缺失模態、上下文上限、圖音讀取與真實取樣率。
- int4 的負數／奇數長度 packing，packed checkpoint 重載後的 logits。
- DPO 與 KL 的方向、凍結教師、有效答案遮罩，以及分塊 softmax 的穩定性。

checkpoint 測試只使用最小數值更新核對接續一致性；它不是模型能力訓練。CLI 推論與評估使用標示為未訓練的暫存權重，只證明檔案、介面與數值路徑可用。

## 固定訓練資料快照

首批8個分來源資料包壓縮後共30,854,837 bytes。已逐包解開到新的暫存目錄，核對每個檔案的SHA-256，再由解包內容重建；8個重建壓縮包均與原始包的SHA-256完全一致。新增測試確認重複解包不改寫檔案、既有修改受到保留、包雜湊先驗證，以及拒絕越界路徑。這些檢查不執行模型訓練。

LFS遠端驗證日期：2026-10-02。GitHub runner從固定公開來源重建8個包，原始逐檔與壓縮包SHA-256全部符合，再上傳8/8物件。本環境以新的LFS storage、不提供額外認證重新下載全部物件，核對大小／SHA-256並逐包解開，包內檔案雜湊也全數通過。遠端驗證報告在`outputs/lfs-remote-validation.json`；[runner執行紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/36946663981)。

原生雲端上傳受到代理限制：有效認證已讓LFS batch API回應200，但S3 PUT因`Transfer-Encoding`回報501；Release附件也回報400 `Bad Content-Length`。因此物件透過GitHub runner上傳。這是本實例的上傳傳輸限制，沒有略過checksum、TLS或LFS pre-push驗證。

## 重跑

啟用已安裝的 `.venv` 後，在 repo 根目錄執行：

```bash
python scripts/build_course.py --check
python scripts/check_course_reviews.py
python scripts/check_technical_reviews.py
python docs/review-tools/check_review_round.py
python scripts/check_notebooks.py --mode python
python scripts/check_notebooks.py --mode kernel --workers 3
pytest -ra
ruff check .
ruff format --check .
uv lock --check --offline
python scripts/export_course.py --executed outputs/notebooks --revision main
python scripts/check_site.py
```

核心／Notebook 測試離線執行。kernel 模式先依[暖身指南](../course/first-steps.md)註冊 tiny-perceptron kernel。Modal實驗的自然資料品質、速度與記憶體結果，須按各份報告的資料、分母與配置閱讀；小型短訓不能保證一般語言、視覺、推理或安全能力。具體訓練與評估方式見[訓練操作](../course/training.md)。Colab雲端未實機驗證；Apple MPS僅完成上面的CI基本矩陣運算，尚未核驗整套教材或正式模型訓練。
