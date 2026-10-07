# 從隨機初始化執行 V2 本機訓練

這份指引帶你重做第19章的有限助理：先練習接續文字，再學問答格式；分別教圖片、讀字與聲音入口後，讓入口和文字核心共同練習。後段接的是你自己前段的權重，不是另一個成熟模型。各段的圖解與承接理由見[19.4](../../course/chapters/19.md#19.4)。

本機訓練入口 `train_local_stage.py` 會先核對材料，再呼叫 `train.py` 執行一段訓練。每次保存完整檢查點（權重與訓練進度）及執行紀錄，供下一段或中斷後接續使用；不需要作者的私人 Modal 儲存空間或 GitHub 工作編號。命令中的 `stage` 就是這次要練的階段。

所有神經網路模組從 PyTorch 隨機初始化開始。字元 tokenizer 只由 train 資料建立，後續 stage 沿用自己的 tokenizer 與權重。公開 pretrain/SFT/joint 的 safetensors 是推論輸出，沒有 optimizer、RNG 或 sampler 狀態；目前 trainer 沒有將公開 safe export 當作 training init 的載入器。

## 安裝與固定資料

安裝 Git LFS 與 uv 後取得 repository。以下以 CPU 環境示範；正式全量訓練可依 README 的硬體說明改用 `cu126` 或 `cu130`，後續 `uv run` 要沿用同一個 extra。完整配方不是一個 batch 的教學 smoke test，CPU 執行所需時間會較長。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch main https://github.com/birdhackor/tiny-perceptron-vlm.git
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

## 手動執行 stage

以下從 repository 根目錄執行。`--` 前是訓練入口的參數，之後是 `train.py` 的完整參數名稱；不接受縮寫。入口提供 `--records`、`--asset-dir`、`--config` 與 `--export-inference`，不用自行加這四項。

先做 `pretrain`：使用固定資料包中的訓練短文，練習預測下一個字。字表只從訓練份建立，後段沿用它；這一段尚未教圖片或聲音，也不是完整聊天訓練。

```bash
uv run --frozen --extra cpu --extra selftrained python scripts/selftrained/train_local_stage.py \
  --manifest docs/selftrained/v2-manifest.json \
  --manifest-sha256 3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b \
  --data-root data/selftrained-v2 -- \
  --output-dir checkpoints/selftrained-local/moe-pretrain \
  --stage pretrain --architecture moe --steps 1000 \
  --batch-size 16 --context 512 --seed 20261006 \
  --learning-rate 0.001 --eval-every 250 --save-every 250 --device cpu
```

接著做 `sft`：使用固定包中的問答示範，讓文字核心練習依問題、對話歷史與格式要求作答；只在助理答案位置計算代價，資料表示見[19.5](../../course/chapters/19.md#19.5)。後續階段保留相同 manifest/data、architecture、batch16、context512 與 seed20261006，使用新 output directory，並以 `--init-checkpoint` 接自己的前段 `best.pt`。以下接的是你剛才訓練出的文字權重：

```bash
uv run --frozen --extra cpu --extra selftrained python scripts/selftrained/train_local_stage.py \
  --manifest docs/selftrained/v2-manifest.json \
  --manifest-sha256 3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b \
  --data-root data/selftrained-v2 -- \
  --output-dir checkpoints/selftrained-local/moe-sft \
  --stage sft --architecture moe --steps 8000 \
  --batch-size 16 --context 512 --seed 20261006 \
  --learning-rate 0.001 --eval-every 1000 --save-every 1000 \
  --init-checkpoint checkpoints/selftrained-local/moe-pretrain/best.pt --device cpu
```

再按下表依序練 `vision`、`ocr`、`audio` 三個入口：材料分別是三類服飾圖、12個已知字元的字卡，以及三種銀行客服主題錄音；任務是分類商品、辨識字序和判別有限語音主題。這時先練入口本身，不把分類正確率當成完整助理的作答率。[19.6](../../course/chapters/19.md#19.6)說明它們怎樣把特徵交給同一文字核心。

`joint` 才把文字、圖片、讀字、聲音和計算工具題混合起來，練習由共享文字核心生成回答。後面的 `weighted` 與 `native` 都沿用共同訓練：前者加重工具與連續數字的指定答案位置，後者調整數字權重並加重直接接收聲音的回答。它們改的是訓練代價中不同位置的分量，不是新增 ASR 聽寫器；感知骨幹與分類頭保持固定，文字核心和接頭繼續更新。

依序調整 stage、output directory、下表參數與自己的來源路徑。這些步數和更新設定來自 [v2-jobs](v2-jobs/) 的實際訓練工作；本機不使用其中的私人儲存空間路徑。

| Output directory 尾段 | `--stage` | `--steps` | `--learning-rate` | `--eval-every` / `--save-every` | 自己的 `--init-checkpoint` | 額外參數 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| moe-pretrain | pretrain | 1000 | .001 | 250 / 250 | 無，隨機初始化 | 無 |
| moe-sft | sft | 8000 | .001 | 1000 / 1000 | moe-pretrain/best.pt | 無 |
| moe-vision | vision | 1200 | .001 | 300 / 300 | moe-sft/best.pt | 無 |
| moe-ocr | ocr | 3000 | .001 | 500 / 500 | moe-vision/best.pt | 無 |
| moe-audio | audio | 1000 | .0002 | 200 / 200 | moe-ocr/best.pt | 無 |
| moe-joint | joint | 6000 | .0005 | 1000 / 1000 | moe-audio/best.pt | `--freeze-perception-backbones --sampling-mode task-family` |
| moe-weighted | joint | 10000 | .0002 | 1000 / 1000 | moe-joint/best.pt | freeze + task-family；`--tool-loss-weight 4 --numeric-run-loss-weight 4` |
| moe-native | joint | 4000 | .0002 | 1000 / 1000 | moe-weighted/best.pt | freeze + task-family；`--tool-loss-weight 4 --numeric-run-loss-weight 1 --native-voice-loss-weight 4` |

Dense 使用 `--architecture dense` 與對應 `dense-*` 目錄，保留自己的完整前段 chain；本輪 frozen final Dense 停在 weighted joint，不承接 MoE 的 checkpoint。

完成自己的 weighted 段後，再用它選定的 `best.pt` 啟動 MoE native 段。這段讓直接接收音訊的回答位置得到較大分量，仍由文字核心輸出文字；它沒有改成一般語音辨識。完整指令為：

```bash
uv run --frozen --extra cpu --extra selftrained python scripts/selftrained/train_local_stage.py \
  --manifest docs/selftrained/v2-manifest.json \
  --manifest-sha256 3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b \
  --data-root data/selftrained-v2 -- \
  --output-dir checkpoints/selftrained-local/moe-native \
  --stage joint --architecture moe --steps 4000 \
  --batch-size 16 --context 512 --seed 20261006 \
  --learning-rate 0.0002 --eval-every 1000 --save-every 1000 \
  --freeze-perception-backbones --sampling-mode task-family \
  --tool-loss-weight 4 --numeric-run-loss-weight 1 --native-voice-loss-weight 4 \
  --init-checkpoint checkpoints/selftrained-local/moe-weighted/best.pt --device cpu
```

Native fresh source 必須是相同 architecture、真正完成的 tool4/numeric4/native1 joint，使用其 selected-validation `best.pt`，並保有同目錄 execution、training receipt、outer receipt 與 safe export。既有 trainer 會核對真正載入的 checkpoint、來源選定 SHA、完整 objective policy、source/options/config/data、origin 與 stage history。新段可改 objective、steps 和 eval/save interval；LR、batch、seed、context、freeze、family 等 inherited options 必須相同。

## 新 stage 與 exact resume

`--init-checkpoint` 是新 stage：載入自己的 selected `best.pt`，重建 optimizer，以新 stage seed 啟動 RNG/sampler，stage steps/tokens 從零開始。`--resume` 則只接受自己的 `latest.pt`，恢復同 stage 的 optimizer/RNG/sampler/steps/tokens；保留原 objective 與 solver 設定，`--steps` 是該 stage 的總目標步數。兩項不能同時使用。

中斷 weighted 後的 exact resume 範例：

```bash
uv run --frozen --extra cpu --extra selftrained python scripts/selftrained/train_local_stage.py \
  --manifest docs/selftrained/v2-manifest.json \
  --manifest-sha256 3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b \
  --data-root data/selftrained-v2 -- \
  --output-dir checkpoints/selftrained-local/moe-weighted-attempt2 \
  --stage joint --architecture moe --steps 10000 \
  --batch-size 16 --context 512 --seed 20261006 \
  --learning-rate 0.0002 --eval-every 1000 --save-every 1000 \
  --freeze-perception-backbones --sampling-mode task-family \
  --tool-loss-weight 4 --numeric-run-loss-weight 4 \
  --resume checkpoints/selftrained-local/moe-weighted/latest.pt --device cpu
```

每次 attempt 都使用尚不存在的 output directory，不覆寫前次輸出。SIGTERM/SIGINT 會轉交 trainer，在既有 step boundary 儲存；wrapper 沒有內建 wall timer。未完成目標步數的 graceful child exit0 仍記為 failed，不能拿來 fresh init。若 wrapper 被直接強制終止，可能留下 running metadata，也不能冒稱 completed source。

每個真實 attempt 保留 `runner.log`、實際 child return code、command、effective parser options、source Git/檔案 SHA、manifest SHA、wrapper SHA、parent checkpoint/receipt SHA 與開始/結束時間。`execution.json`、`train-receipt.json`、`receipt.json` 和 checkpoints/export 在同一目錄，outer receipt 記錄實際檔案 bytes/SHA。普通舊 `train.py` 輸出若沒有這組 genuine sidecars，不能事後補寫成歷史 completed native source。`.pt` 是自己產生的可信 pickle checkpoint；公開 safe export 仍供推論使用。

## 本輪實際完成步數與選定步數

下表只描述已實核的本輪 production lineage，供比對 checkpoint selection；新執行不硬編碼這些選定步數。`best.pt` 由 unchanged unweighted teacher-forced validation loss 選定，並不代表 generation success。

| Stage | 實際完成 steps | MoE selected step | Dense selected step |
| --- | ---: | ---: | ---: |
| pretrain | 1000 | 250 | 250 |
| sft | 8000 | 8000 | 6000 |
| vision | 1200 | 1200 | 1200 |
| ocr | 3000 | 3000 | 3000 |
| audio | 1000 | 200 | 200 |
| joint | 6000 | 2000 | 2000 |
| weighted tool4/numeric4 | 10000 | 3000 | 1000 |
| native tool4/numeric1/native4 | 4000 | 1000 | — |

來源見 [12 段 baseline index](v2-training-stage-index.json)、[MoE weighted index](results/v2-moe-weighted-10000-training-stage-index.json)、[Dense weighted index](results/v2-dense-weighted-10000-training-stage-index.json) 和 [MoE native index](results/v2-moe-native-balanced-4000-training-stage-index.json)。本輪 MoE native 確實從 completed10000、selected3000 的 own weighted source 啟動新段，沒有將 latest10000 當作改 objective 的 exact resume。

本機 wrapper 的工程驗證使用 width16 合成 CPU 資料、真 batch16/context512：三段 joint 各一步、genuine failure/interrupt、禁止 incomplete promotion 與 ownlatest exact resume。這只驗證執行契約，不代表重跑了全量 production training 或證明模型能力。[Frozen final results](results/v2-final-public-results.json) 記錄兩個 architecture 各 3,734 筆固定 final test；原判準仍未全數達標，不能將本操作指引或 CPU smoke 當作 pass 宣稱。
