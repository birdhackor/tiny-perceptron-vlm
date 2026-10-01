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
