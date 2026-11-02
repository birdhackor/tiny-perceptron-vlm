## T.6 讓圖片與聲音先有可用的基礎特徵

本節先練局部玩具任務：圖片分方形、圓形，聲音分低音、高音。圖片入口與聲音入口（encoder）要從像素或聲音數字學到這些線索，接頭才能把它們送進文字核心。真人語音需求和短中文字卡是不同材料，另沿第 19 章整合設計。本節命令在你電腦上的時間未量測。

```bash
.venv/bin/python scripts/pretrain_encoders.py --modality vision --train --steps 300 --output checkpoints/vision-encoder.pt
.venv/bin/python scripts/pretrain_encoders.py --modality audio --train --steps 300 --output checkpoints/audio-encoder.pt
```

每條命令最後印出一份 JSON，依序確認三件事：

1. **留出了什麼。** 圖片留出兩張藍色圖，方形、圓形各一張，圖形中心比置中的位置往右移 1 像素；同樣往右移的紅、綠圖仍在訓練裡，所以留出的是沒見過的「藍色加右移」組合。聲音留出 180 Hz 的低音與 1000 Hz 的高音。類別編號是方形 0、圓形 1，低音 0、高音 1。
2. **這次真的訓練了。** 命令有 `--train`，`mode` 應為 `train`；若是 `dry-run-no-weight-update`，表示只檢查通路，尚未完成這次訓練。
3. **留出的驗證題全對才接續。** `holdout_examples` 應為 2，`holdout_accuracy` 要等於 1；0.5 只是答對一題。存下的檔案存在，不能代替這項檢查。

取得 [T.4](#T.4) 對話路線保存的 `checkpoints/attributes.pt` 與兩個入口後，先選要開放哪些參數更新。凍結（freeze）是指這次訓練不更新某些參數。`--freeze` 有三個選項：

- `projector`：只更新圖片與聲音的接頭，兩個入口與文字核心都凍結。
- `partial`：接頭之外，再開放文字核心的第一個與最後一個區塊。
- `none`：所有參數都更新，包括兩個入口。

這是本工具自己的凍結方案，不是照名稱只凍結某個零件。開放更新也要有本次輸入帶來的梯度，沒用到的入口不會因此自動學習。本練習的聯合路線選 `partial`，目的是讓接頭與首末文字區塊一起適應圖片、聲音兩種輸入。下面三條500步路線先示範接合與抽查，沒有固定的整合驗證題，不能據此驗收這組設定的效果。要用未參與更新的固定題檢查圖片問答與文字保留，再接著完成本節正文的固定圖文路線；兩套配方的模型與成績分開。

接著選圖片、聲音或聯合路線：

```bash
.venv/bin/python scripts/train.py --task vision --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --freeze projector --train --steps 500 --output checkpoints/vision.pt
.venv/bin/python scripts/train.py --task audio --checkpoint checkpoints/attributes.pt --audio-encoder checkpoints/audio-encoder.pt --freeze projector --train --steps 500 --output checkpoints/audio.pt
.venv/bin/python scripts/train.py --task joint --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --audio-encoder checkpoints/audio-encoder.pt --freeze partial --train --steps 500 --output checkpoints/joint.pt
```

接著製作素材，再請模型回答。沒有指定圖片或音訊檔時，`--color` 與 `--shape` 用來畫出合成圖片，大小依載入的圖片入口決定；`--frequency` 則製作該音高、取樣率 16 kHz 的單音波形。模型只收到圖片像素或聲音數字，加上 `--prompt` 的問題文字；`--shape` 這類設定只用來畫圖，不會寫進模型讀到的文字。輸出 JSON 的 `input_sources` 會另外記下素材怎麼產生，供你核對。

下面三題分別問形狀、音高，以及兩者：

```bash
.venv/bin/python scripts/infer_modal.py checkpoints/vision.pt --color blue --shape circle --prompt "shape?" --tokens 16
.venv/bin/python scripts/infer_modal.py checkpoints/audio.pt --frequency 880 --prompt "pitch?" --tokens 16
.venv/bin/python scripts/infer_modal.py checkpoints/joint.pt --color red --shape square --frequency 220 --prompt "joint?" --tokens 16
```

希望答案依序為 `circle`、`high`、`square,low`。開啟 JSON 的 `answer` 和它們比較，再看結束是否完整。兩個入口的驗證題全對，也沒有保證接上文字核心後每題都對。模型小、訓練短，同一個固定配方在不同電腦上也可能訓練出不同結果，所以你看到的答案可能和希望的不同；不同時先確認命令照抄、檢查點存在。這三題只是抽查：`train.py` 這三條路線的訓練例子是每一步隨機產生的，不要只憑三題判斷整體能力。

`vision.pt`、`audio.pt`、`joint.pt` 是多模態檢查點：裡面同時有文字核心（`model.language`）與圖片、聲音的入口和接頭。`evaluate.py` 只接受純文字的檢查點，直接把多模態檢查點交給它會報錯，所以前面的手動命令沒有檢查文字能力有沒有退步。下方固定實驗在圖文訓練前後，用同一批文字測試題各核對一次，[11.7](chapters/11.md#11.7)解釋怎麼讀它的結果。

### 用固定題完成圖文比較

這條路線依序建立固定的SFT文字起點、圖片／聲音入口、圖片接頭，最後做圖片問答（visual question answering，VQA）比較。命令共用`outputs/course-experiments/course-v1/`根目錄，後續實驗從這裡讀取所需的已完成配套；工具不自動補齊前置。若[T.4的固定SFT配方](#fixed-sft-recipe)已完成並保留`sft/model.pt`與`dataset.json`，可從`encoders`開始；不能用上方手動保存的`attributes.pt`替代：

```bash
.venv/bin/python scripts/fetch_training_assets.py --asset ultrachat-sft
.venv/bin/python -m scripts.course_experiments.run --experiment sft --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment projector --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment vqa --device cuda
```

作者在 Modal 的 NVIDIA L4 上的紀錄：`encoders` 10.2 秒、`projector` 15.9 秒、`vqa` 83.8 秒。沒有 NVIDIA GPU 時可以改成 `--device cpu`，但這三組屬於模態實驗，工具會把 CPU 的結果標成只檢查通路（`evidence_status` 為 `interface_smoke_only`），不算正式結果；CPU 時間未量測。

開啟自己的`outputs/course-experiments/course-v1/vqa/result.json`，先看`results.data.splits`：同一顏色、形狀與位移的問答留在同一組，訓練、驗證、測試各有36、12、12題。再看`results.variants.partial`：`test.correct`與`test.examples`是固定圖片測試題的答對數與題數；`text_before.matches`、`text_after.matches`各配上自己的`examples`，比較同一批文字留出題在更新前後的表現。現有[歷史紀錄](../docs/course-experiments/results/vqa.json)中，這支圖片12／12題對，文字卻由5／10降到2／10，不能只看圖片答對就說原能力保住了。重做時保存自己的前後結果，不能抄歷史數字代替驗收。

這套 `vqa` 的 `partial` 只開圖片接頭與最後文字區塊，和前面手動CLI不同；固定圖文比較也不代替聲音／圖音聯合的驗收。完整配方的依賴、數據和歷史結果見[實驗說明](../docs/course-experiments/README.md)，不能套成上面500步的預期成績。

<details>
<summary>接續手動訓練與使用自己的檔案</summary>

自己用命令保存的完整 `multimodal-v1` 檢查點含最佳化器狀態、進度、隨機狀態與可訓練名單。原排程未完成時，用相同任務、資料、總步數和凍結範圍，加 `--resume` 接續；不要再加 `--vision-encoder` 或 `--audio-encoder`，入口已經存在檢查點裡。僅供推論的學生包不能精確恢復原訓練，格式與例子見[5.7](chapters/05.md#5.7)。

自己的圖像紀錄可寫成 `{"image":"images/12.png","question":"read digits","answer":"12"}`，路徑相對 JSONL。這個局部工具會轉 RGB 並縮至 16×16，可能丟失細字；聲音入口要求非空、取樣率16kHz的單聲道檔案，即每秒保存16,000個聲音數字，不會自動重新取樣。取樣率不同於前面`--frequency`所設定的單音音高頻率，區別見[12.1](chapters/12.md#12.1)。輸入必須和模型學過的任務匹配。

做數字圖的正常／空白／錯配對照，可用 `prepare_ocr.py` 產生資料，再以 `train.py --task vision` 訓練，並用 `evaluate_modal.py --ablation none`、`blank`、`shuffle` 檢查同一批題目。錯配報告的 `donor_row` 指出換入圖來源；它仍拿原圖的正確答案（`target`）計分，要另外核對生成是否符合換入圖。完整欄位見[11.13](chapters/11.md#11.13)及 [OCR 實驗紀錄（JSON）](../docs/course-experiments/results/ocr.json)。

</details>

