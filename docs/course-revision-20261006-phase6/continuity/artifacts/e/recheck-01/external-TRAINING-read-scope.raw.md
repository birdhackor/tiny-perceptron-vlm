# 從隨機初始化執行 V2 本機訓練

這份指引帶你重做第19章的有限助理：先練習接續文字，再學問答格式；分別教圖片、讀字與聲音入口後，讓入口和文字核心共同練習。後段接的是你自己前段的權重，不是另一個成熟模型。各段的圖解與承接理由見[19.4](../../course/chapters/19.md#19.4)。

本機訓練入口 `train_local_stage.py` 會先核對材料，再呼叫 `train.py` 執行一段訓練。每次保存完整檢查點（權重與訓練進度）及執行紀錄，供下一段或中斷後接續使用；不需要作者的私人 Modal 儲存空間或 GitHub 工作編號。命令中的 `stage` 就是這次要練的階段。

所有神經網路模組從 PyTorch 隨機初始化開始。字元 tokenizer 只由 train 資料建立，後續 stage 沿用自己的 tokenizer 與權重。公開 pretrain/SFT/joint 的 safetensors 是推論輸出，沒有 optimizer、RNG 或 sampler 狀態；目前 trainer 沒有將公開 safe export 當作 training init 的載入器。

## 安裝與固定資料

安裝 Git LFS 與 uv 後取得 repository。以下以 CPU 環境示範；正式全量訓練可依 README 的硬體說明改用 `cu126` 或 `cu130`，後續 `uv run` 要沿用同一個 extra。完整配方不是一個 batch 的教學 smoke test，CPU 執行所需時間會較長。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch selftrained-v2 https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
git lfs install --local
uv sync --frozen --extra cpu --extra selftrained
```

重現時保留所用 repository commit；正式發布的 commit 與結果來源見本輪 reproduction capsule。Wrapper 會核對 tracked trainer/model/scripts 的內容與當下 Git HEAD，再記錄實際 source SHA。

資料固定於 [v2-manifest.json](v2-manifest.json)，SHA256 為 `3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b`。壓縮包固定於 Git commit `08761dac87a6ef360db95883d9bcd338c44fe76d` 的 `assets/training/selftrained-v2.tar.gz`，大小 67,862,431 bytes，SHA256 `0976073a3bc7c331a65cedb4c31d54f5c6e9a5014ff2443698a8e2b8cad7e78a`。使用既有 transport helper 驗證 LFS pointer、實際物件與解包上限：

```bash
uv run --frozen --extra cpu --extra selftrained python - <<'PY'
import hashlib
import json
from pathlib import Path
from scripts.selftrained.hf_transport import materialize_git_lfs, unpack_verified_archive, verify_file

repo = Path.cwd()
manifest_path = repo / "docs/selftrained/v2-manifest.json"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == "3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b"
manifest = json.loads(manifest_path.read_text())
destination = repo / "data/selftrained-v2"
assert not destination.exists(), "使用新的資料解包目錄，保留既有檔案"
archive = materialize_git_lfs(manifest["package"], repo, repo / "data/selftrained-v2.tar.gz")
unpack_verified_archive(archive, destination, manifest["package"])
files = manifest["records"] + manifest["assets"]
for item in files:
    verify_file(destination, item)
assert len(files) == 8962
print("verified", len(files), "frozen files")
PY
```

完整資料含 12 個 JSONL 與 8,950 個 assets，共 8,962 個固定檔案。每次 stage 啟動都核對全部 declarations，並確認每個實際 image/audio 引用的 SHA 都在 manifest 的 assets 中。訓練與 validation 的選擇仍由原 trainer 負責；test 不用來選 checkpoint。

