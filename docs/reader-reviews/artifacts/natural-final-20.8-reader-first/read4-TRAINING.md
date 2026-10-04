# 用自己的GPU重做第20章的LoRA訓練

這份額外實作指引，帶你從固定的圖文底座建立一份新的LoRA，先用訓練題更新，再用驗證題檢查自己的結果。沿用的是第20章第一次成功實驗的180步、學習率`1e-4`配方；這次只訓練語言注意力的LoRA，沒有重新訓練視覺底座或Whisper語音辨識器。請另外預留安裝、下載與GPU運算時間。

理解原理可先讀[20.3資料切分](../../course/chapters/20.md#20.3)與[20.4微調範圍](../../course/chapters/20.md#20.4)。下面的指令都在專案根目錄執行，使用Linux、Python 3.12與你自己的NVIDIA GPU。

## 1. 先準備GPU環境

先依[學生操作指引](STUDENT.md)的第1、2步取得專案，建立`.venv-natural`，安裝CUDA 12.8版PyTorch／torchvision及`requirements-natural.txt`。前面小實驗的`.venv`繼續保留。這次訓練從底座新建LoRA，準備環境後即可接下面的資料步驟。

本配方使用bfloat16；在自己的GPU電腦上檢查：

```bash
.venv-natural/bin/python -c "import torch; print('GPU可用：', torch.cuda.is_available()); print('支援bfloat16：', torch.cuda.is_bf16_supported())"
```

兩項都應為True。若找不到GPU或不支援bfloat16，先處理環境與設備問題；改用其他精度屬於另一份實驗設定，不能直接稱為重做這份配方。

## 2. 取得三個固定的教學資料包

先看下載清單，再下載、核對：

```bash
.venv-natural/bin/python scripts/fetch_natural_data.py --list
.venv-natural/bin/python scripts/fetch_natural_data.py --manifest docs/natural-assistant/manifest.json --output data/natural
.venv-natural/bin/python scripts/fetch_natural_data.py --manifest docs/natural-assistant/manifest.json --output data/natural --verify
```

這個入口固定到公開Git版本`21a24124353487e08881302bbd71614766a489ac`，取得自然照片、合成中文字卡與真人華語錄音三包Git LFS資料。程式直接匿名下載，不必另外安裝Git LFS；三包合計70,648,031 bytes，約70.65 MB。包裡已經是這輪所需的小份材料，不需要再下載上游完整照片或錄音大包。

成功時，最後一個指令會顯示`status: verified`與`files_verified: 345`。資料位於`data/natural/vision/`、`data/natural/ocr/`、`data/natural/speech/`；訓練程式用`docs/natural-assistant/manifest.json`配對它們。清單中有272筆訓練題、52筆驗證題與66筆最後測試題；錄音另外分成24／6／12段，沒有用來更新Whisper。

下載程式會核對遠端清單、壓縮包與解開檔案的完整指紋。已有相同資料時會核對後重用；內容不同時會停止並保留原資料，請換一個新的`--output`，也同步改後續的`--data-root`。不要把自己的檔案放進這個資料夾。來源、署名與授權見[資料說明](DATA.md)。

## 3. 先下載固定模型

資料包的70.65 MB不包含模型。下面的準備指令下載固定版本的Qwen3-VL-2B-Instruct與Whisper-small，並保存模型檔案核對紀錄：

```bash
.venv-natural/bin/python scripts/natural_assistant.py prepare \
  --output outputs/natural-my-models \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16
```

兩個模型使用程式中固定的版本，與[公開成品清單](public-release.json)所列版本相同，公開檔案合計約5.24 GB。這一步只準備模型，還沒有更新LoRA；Whisper留給後面的語音驗證使用。完成後保留`.cache/natural-models`，後面的`--local-files-only`會使用這份快取。

## 4. 保存配方，開始180步訓練

目前CLI的預設學習率是`3e-5`、預設步數是100，與第一次180步實驗不同。因此下面明寫`--learning-rate 1e-4`與`--steps 180`。

我們用`outputs/natural-my-experiment`存放自己的新實驗。若這個名字已經用過，請把下面各處換成新的資料夾名稱；不要把公開成品或別人的訓練資料夾當成輸出。把完整指令保存成腳本，再執行：

```bash
mkdir -p outputs/natural-my-experiment
cat > outputs/natural-my-experiment/train.sh <<'SH'
.venv-natural/bin/python scripts/natural_assistant.py train \
  --manifest docs/natural-assistant/manifest.json \
  --data-root data/natural \
  --output outputs/natural-my-experiment/train \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 \
  --steps 180 --learning-rate 1e-4 \
  --lora-rank 8 --gradient-accumulation 2 --seed 42 \
  --checkpoint-every 25 --max-seconds 3300 \
  --local-files-only > outputs/natural-my-experiment/train.stdout.json
SH
bash outputs/natural-my-experiment/train.sh
```

180步指180次更新，每步累積兩道題，所以共使用360次題目；這不是360道不同題。rank 8的修正裝在語言模型28層注意力的q與v投影，alpha為16，合計112份LoRA張量、1,605,632個可更新參數。原底座仍參與計算，只有LoRA進入更新器。

`--max-tokens 2048`是每道訓練輸入與答案的長度上限，`--max-pixels 524288`限制圖片處理預算。程式遇到超長訓練輸入會報錯，沒有偷偷截掉答案。`--max-seconds 3300`是這次執行的時間上限；到時仍未完成180步，會保存目前進度，不能算成已完成180步。

原本[NVIDIA L4實驗](evidence/train/result.json)完成180步，112份修正指紋都改變，底座的少量數值抽查保持相同。149.47秒的計時包含模型載入、更新及迴圈中的檢查點，止於最後指紋抽查與最終存檔之前；不含資料核驗、模型傳輸、容器或環境準備。PyTorch已配置GPU記憶體峰值約5.23 GiB，這不是全部GPU占用，也不是設備最低容量保證。你的GPU與軟體環境可能產生不同時間與權重指紋，請保存自己的結果。

## 5. 檢查結果；中斷時用自己的檢查點續訓

下表路徑省略開頭的`outputs/natural-my-experiment/`：

| 產生的內容 | 用途 |
| --- | --- |
| `train/adapter/adapter_model.safetensors`、`train/adapter/adapter_config.json` | LoRA權重與配對設定，供推論載入 |
| `train/adapter/training_state.pt`、`train/adapter/training.json` | 更新器、隨機狀態與進度，連同整個adapter資料夾用於續訓 |
| `train/result.json`、`train/training.json` | 完成狀態、步數、學習率與訓練紀錄 |
| `train/provenance.json` | 底座、資料指紋、套件版本及推論設定 |

完成時，`train/result.json`應記錄`status: completed`、`completed_steps: 180`、`learning_rate: 0.0001`、`trainable_parameters: 1605632`與`optimizer_only_lora: true`。這些是訓練執行檢查，回答品質仍要在下一步驗收。

程式每25步或約60秒保存一次檢查點。若中斷，先查看`train/adapter/training.json`的`completed_steps`；未保存的更新不能恢復。只有已有完整檢查點且尚未到180步時，才使用下面的續訓指令：

```bash
.venv-natural/bin/python scripts/natural_assistant.py train \
  --manifest docs/natural-assistant/manifest.json \
  --data-root data/natural \
  --adapter outputs/natural-my-experiment/train/adapter \
  --output outputs/natural-my-experiment/resumed \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 --max-tokens 2048 \
  --steps 180 --learning-rate 1e-4 \
  --lora-rank 8 --gradient-accumulation 2 --seed 42 \
  --checkpoint-every 25 --max-seconds 3300 --local-files-only
```

這裡的180仍是目標總步數，不是再增加180步。新結果存到`resumed/`，原檢查點保留；若走了續訓，後面的驗證、固定指紋與最後測試都把`train/adapter`改成`resumed/adapter`。保持資料、底座、精度、圖片預算、rank、累積方式、種子與學習率一致；想改配方，另開新實驗。公開試用小包沒有這份更新器與進度，不能用來恢復作者的訓練。

## 6. 先驗證自己的模型，再固定版本做最後測試

先用`validation`，並把自己的adapter路徑寫清楚：

```bash
.venv-natural/bin/python scripts/natural_assistant.py validation \
  --manifest docs/natural-assistant/manifest.json \
  --data-root data/natural \
  --adapter outputs/natural-my-experiment/train/adapter \
  --output outputs/natural-my-experiment/validation \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 --max-tokens 2048 \
  --max-new-tokens 384 --seed 42 --max-seconds 3300 \
  --split validation --local-files-only
```

這個指令在同一底座上分別關閉與開啟你的LoRA，保存`generations-base.json`、`generations-adapter.json`及`result.json`。它也保存`transcripts.json`，把真正辨識結果與官方逐字稿兩路分開比較；沒有訓練Whisper。完整跑完時，`result.json`應是`completed`，base與adapter各有64個生成紀錄。

先讀兩版完整回答，依[20.5至20.7](../../course/chapters/20.md#20.5)核對圖片、中文字與語音兩站，保留退步、截斷及未完成。自動關鍵字與逐字分數是輔助，不能代替照片整段描述的核對。只用驗證題決定要保留哪份自己的模型；更換配方時另存實驗與驗證輸出。

選定後，先保存新權重、設定、資料清單與驗證結果的指紋：

```bash
sha256sum \
  outputs/natural-my-experiment/train/adapter/adapter_model.safetensors \
  outputs/natural-my-experiment/train/adapter/adapter_config.json \
  docs/natural-assistant/manifest.json \
  outputs/natural-my-experiment/validation/result.json \
  > outputs/natural-my-experiment/chosen-before-test.sha256
```

這份紀錄屬於你的新實驗。教材的`selection.json`與公開能力表對應作者先前選定的權重，不能套到你的新輸出；保留教材原檔，另存自己的選版紀錄。固定權重與設定、不再用它們繼續更新後，才執行：

```bash
.venv-natural/bin/python scripts/natural_assistant.py evaluate \
  --manifest docs/natural-assistant/manifest.json \
  --data-root data/natural \
  --adapter outputs/natural-my-experiment/train/adapter \
  --output outputs/natural-my-experiment/final \
  --cache-dir .cache/natural-models \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 --max-tokens 2048 \
  --max-new-tokens 384 --seed 42 --max-seconds 3300 \
  --split test --local-files-only
```

保存這次真正的最後回答與失敗。若看過test後又修改配方，這些題已經參與你的決策，不能反覆挑版本再把最高分稱為一次未見考卷的測試。它們仍可作為已知錯題練習；新的公平最後評估需要另準備尚未用來選版的題目。

## 7. 常見停住的地方

- **資料核對失敗或找不到圖片。**重跑第2步的`--verify`，確認`--data-root`指向解包資料，而不是壓縮檔或清單所在目錄。資料不同時換新下載目錄，保留舊檔。
- **離線快取缺檔。**回到第3步完成模型準備；兩個模型共用同一個`--cache-dir`，再執行帶`--local-files-only`的訓練或驗證。
- **GPU記憶體不足或不支援bfloat16。**先檢查設備與其他占用，依[學生操作指引](STUDENT.md)處理。若改小圖片預算或換精度，保存為新配方與新實驗，重新驗證。
- **時間上限或程式中斷。**看實際`completed_steps`與最新完整檢查點，依第5步續訓。驗證若是`time_limit_partial`，保留這次部分結果，增加時間後另存一份完整驗證，選版前確認所有題確實跑完。
- **出現非有限代價、梯度或凍結樣本改變。**保留錯誤與輸出，先核對套件、資料及設定；這次不能當成成功訓練。

使用Modal的讀者，可參考[既有GPU指南](../gpu-training.md)及[自然助理工作流程](../../.github/workflows/natural-assistant.yml)。目前該工作流程沒有可讓使用者指定學習率的輸入，訓練命令固定用`3e-5`；這份`1e-4`重做配方使用上面的自有GPU CLI路線。
