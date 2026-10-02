# Modal GPU 與 Hugging Face 的小規模驗證

這個入口驗證「GitHub Actions 提交 → Modal GPU 訓練 → HF 存檔 → 下載續訓」。
只使用課程自建的顏色、形狀規則資料與 31,584 個參數的小模型，不代表一般語言能力。
一般 push、PR 和教材發布不會觸發 GPU 工作；必須明確執行此工作流程。

## 帳號設定

Modal 使用 Starter 方案即可。在 `Usage & Billing` 設定 Workspace 用量上限與需要的自付上限。
Environment 先使用 `main`；建立 API token 後，將下列兩個值存到 GitHub 的
`Settings → Secrets and variables → Actions → Secrets`，不要放進對話、程式或 commit：

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

HF 建立私有 Model repo 保存 checkpoint，以及公開 Model repo 保存日後驗證完成的教學權重。
建立只允許這兩個 repo 讀寫的 fine-grained token，存到 Modal 的 `main` Environment：

- Secret 名稱：`tiny-perceptron-hf`
- Key：`HF_TOKEN`
- Value：HF token

GitHub 的 `Actions → Variables` 設定：

| 名稱 | 值 |
| --- | --- |
| `MODAL_ENVIRONMENT` | `main` |
| `HF_CHECKPOINT_REPO` | `你的HF帳號/tiny-perceptron-checkpoints` |
| `HF_RELEASE_REPO` | `你的HF帳號/tiny-perceptron-course-models` |

## 執行與判讀

在 GitHub `Actions → GPU training smoke test → Run workflow` 執行。
也可以請 Codex 提交這個工作並讀取結果。使用單張 L4、2 CPU、4 GiB RAM；
GPU 函式最多 600 秒、不自動重試，同一工作流程一次只跑一件。
首次需要建置 CUDA 映像，建置時間不等於 GPU 訓練時間。
Modal SDK 1.6.0 僅安裝在執行工具環境；模型依賴仍由原本 `uv.lock` 與 `cu126` extra 決定。

通過必須同時確認：

1. HF 私有 repo 可讀、可寫；公開 repo 可讀。
2. CUDA 確實可用，模型權重更新，梯度有限且非零，訓練 loss 下降。
3. 第 40 步保存 optimizer 和 RNG；第 80 步正常完成。
4. 第 40 步 checkpoint 上傳 HF，依指定 HF commit 重新下載，SHA-256 一致。
5. 下載檔恢復後從第 41 步跑到第 80 步，權重與不中斷訓練一致，容許浮點誤差。
6. 另一個 Modal container 可讀到 Volume 的相同 checkpoint。

結果在 Actions 的 Summary 和 `modal-gpu-smoke-result` artifact。
HF 檔案放在私有 repo 的 `smoke-tests/gha-工作ID-嘗試次數/training/`；
Modal Volume 名稱為 `tiny-perceptron-checkpoints`，每次工作有獨立子目錄。
測試不向公開 Model repo 發布權重；公開 repo 的寫入與正式模型能力另行驗證。

若設定不完整，工作會指出缺少的名稱；若 Modal 或 HF 拒絕存取，依原始錯誤修正後再試。
測試失敗不應宣稱 GPU、上傳或續訓成功。權重檔與輸出不提交到 Git。

本機 CPU 可先驗證相同訓練入口，不需要 Modal 憑證：

```bash
.venv/bin/python scripts/gpu_smoke_check.py --device cpu
```

官方說明：[Modal 費用上限](https://modal.com/docs/guide/budgets)、
[Modal Secrets](https://modal.com/docs/guide/secrets)、
[HF Token](https://huggingface.co/docs/hub/security-tokens)。
