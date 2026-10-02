# 教材實作驗證

驗證日期：2026-10-01。環境為 Linux、Python 3.13.5、PyTorch 2.14.1 CPU。程式、Notebook 與介面已檢查；沒有執行 `--train` 的正式模型訓練，沒有將隨機權重的分數當成能力結果。

| 檢查 | 實際結果 |
| --- | --- |
| 大綱編號／正文／Notebook 對應 | 222 節完整，建置同步檢查通過 |
| Notebook 快速執行 | 222 通過，0 失敗；每節使用新 namespace |
| 獨立 Jupyter kernel | 222 通過，0 失敗；每節由乾淨 kernel 開始 |
| 核心離線測試 | 32 passed、1 skipped；含4項資料包檢查，略過項目為 CUDA／MPS 裝置測試 |
| CLI 操作 | 32 條路徑通過，0 失敗；訓練入口均未加 `--train` |
| JupyterLab 啟動 | 實際 `/lab` 回應200，API包含 tiny-perceptron kernel；測完停止自行啟動的伺服器 |
| 圖解 | 40 張自製 SVG；抽樣光柵化檢查繁體中文、圖形與位置 |
| 離線閱讀版 | 222 節網頁與操作文件匯出成功，可核對內部連結與圖檔 |
| 規則資料 | 5 種 JSONL 的原始 SHA-256 與家族切分核對；另有100種字串家族的 OCR 產生器 |
| 依賴與格式 | frozen CPU／Notebook 安裝、離線 lock 檢查、Ruff lint／format 檢查通過 |

kernel 報告在 `outputs/notebooks/validation-kernel.json`，逐節執行副本也保存在該資料夾。CLI 報告為 `outputs/cli-validation.json`。這些是本次工作區的產物，被 Git 忽略；新 checkout 可用下面指令重新產生。

## 網站發布驗證

2026-10-02：加入 Colab 自動設定後，重新以獨立 CPU kernel 執行222份 Notebook，222通過、0失敗；核心測試仍為32 passed、1 skipped。匯出254頁並核對222個 Colab入口、222份下載與60,338個內部連結。發布程式也確認會拒絕過期原始碼、未執行與失敗的結果，並轉義輸出文字中的HTML。

Chromium檢查1440px桌面與390px手機版；搜尋「量化」顯示5節，手機頁面無水平溢出，並可見兩個練習／下載按鈕。離線公式降級保留TeX文字。檢查也發現SVG動畫將邊框筆畫繼承到文字，已改成只動畫矩形邊框並重新產生圖解。Colab實際雲端執行、GPU訓練與收斂仍未驗證。

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
python scripts/check_notebooks.py --mode python
python scripts/check_notebooks.py --mode kernel
pytest -ra
ruff check .
ruff format --check .
uv lock --check --offline
python scripts/export_course.py
```

核心／Notebook 測試離線執行。kernel 模式先依[暖身指南](../course/first-steps.md)註冊 tiny-perceptron kernel。硬體加速、訓練收斂、自然資料能力、效能提升與一般安全性尚未驗證；具體訓練與評估方式見[訓練操作](../course/training.md)。
