# 教材實作驗證

教材本機驗證環境為 Linux、Python 3.13.5、PyTorch 2.14.1 CPU；正式訓練另使用 Modal NVIDIA L4、Python 3.13.3、PyTorch 2.14.1+cu126。本頁保留不同階段的紀錄。最新逐組訓練、自然資料短訓、私有備份與學生權重證據見[正式實驗](course-experiments/README.md)及[當前進度](course-experiments/progress.json)；早期流程測試與改寫前的審閱不能代替本輪驗收。

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
