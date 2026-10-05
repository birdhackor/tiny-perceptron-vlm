## T.6 讓圖片與聲音先有可用的基礎特徵

本節先練局部玩具任務：圖片分方形、圓形，聲音分低音、高音。入口要從像素或聲音數字學到這些線索，接頭才能把它們送進文字模型。真人語音需求和短中文字卡是不同材料，另沿第 19 章整合設計。

```bash
.venv/bin/python scripts/pretrain_encoders.py --modality vision --train --steps 300 --output checkpoints/vision-encoder.pt
.venv/bin/python scripts/pretrain_encoders.py --modality audio --train --steps 300 --output checkpoints/audio-encoder.pt
```

視覺留出兩張向右偏移的藍色圖，方形、圓形各一張；同位置的紅、綠圖仍用於訓練，留出的是未見過的顏色與位置組合。方形類別 0、圓形 1；音訊留出 180Hz 低音類別 0、1000Hz 高音 1。查看最後JSON：本次命令有`--train`，`mode`應為`train`；若是`dry-run-no-weight-update`，表示只檢查通路，尚未完成這次訓練。`holdout_examples`應為2，`holdout_accuracy`要等於1才接續；0.5只是答對一題，保存檔存在不能代替檢查。

取得 T.4 的 `attributes.pt` 與兩個編碼器後，選圖片、聲音或聯合路線：

```bash
.venv/bin/python scripts/train.py --task vision --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --freeze projector --train --steps 500 --output checkpoints/vision.pt
.venv/bin/python scripts/train.py --task audio --checkpoint checkpoints/attributes.pt --audio-encoder checkpoints/audio-encoder.pt --freeze projector --train --steps 500 --output checkpoints/audio.pt
.venv/bin/python scripts/train.py --task joint --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --audio-encoder checkpoints/audio-encoder.pt --freeze partial --train --steps 500 --output checkpoints/joint.pt
```

`projector` 只更新接頭；這個 CLI 的 `partial` 另允許首末文字區塊更新，`none` 則開放所有零件。開放更新也要有本次輸入帶來的梯度，沒用到的入口不會因此自動學習。

用明確題目檢查實際回答：

```bash
.venv/bin/python scripts/infer_modal.py checkpoints/vision.pt --color blue --shape circle --prompt "shape?" --tokens 16
.venv/bin/python scripts/infer_modal.py checkpoints/audio.pt --frequency 880 --prompt "pitch?" --tokens 16
.venv/bin/python scripts/infer_modal.py checkpoints/joint.pt --color red --shape square --frequency 220 --prompt "joint?" --tokens 16
```

希望答案依序為 `circle`、`high`、`square,low`。開啟 JSON 的 `answer` 和它們比較，再看結束是否完整。兩道編碼器題全對，也沒有保證轉接後每題都對。

原生模態檔裡有文字模型與入口。檢查文字能力時，取 `model.language` 用同一份文字留出題評估；不要把模態包裝直接交給只接受文字 checkpoint 的入口。[11.7](chapters/11.md#11.7)解釋這個對照。

<details>
<summary>續訓、固定模態比較與自己的檔案</summary>

自己命令保存的完整 `multimodal-v1` 檔含更新器、進度、隨機狀態與可訓練名單。原排程未完成時，用相同任務、資料、總步數和凍結範圍，加 `--resume` 接續；不要再加外部編碼器。僅供推論的學生包不能精確恢復原訓練，格式與例子見[5.7](chapters/05.md#5.7)。

### 重跑第11章的固定實驗

先完成 T.4 的固定 `sft`，再依序執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment projector --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment vqa --device cuda
```

這套 `vqa` 的 `partial` 只開圖片接頭與最後文字區塊，和本節 CLI 不同。完整入口的依賴、數據和歷史結果見[實驗說明](../docs/course-experiments/README.md)，不能套成上面 500 步的預期成績。

自己的圖像紀錄可寫成 `{"image":"images/12.png","question":"read digits","answer":"12"}`，路徑相對 JSONL。這個局部入口會轉 RGB 並縮至 16×16，可能丟失細字；聲音入口要求非空、採樣率16kHz的單聲道檔案，即每秒保存16,000個聲音數字，不會自動重採樣。採樣率不同於前面`--frequency`所設定的單音音高頻率，區別見[12.1](chapters/12.md#12.1)。輸入必須和模型學過的任務匹配。

做數字圖的正常／空白／錯配對照，可用 `prepare_ocr.py` 產生資料，再以 `train.py --task vision` 訓練，並用 `evaluate_modal.py --ablation none`、`blank`、`shuffle` 檢查同一批題目。錯配報告的 `donor_row` 指出換入圖來源；它仍對原答案計分，要另外核對生成是否符合換入圖。完整欄位見[11.13](chapters/11.md#11.13)及[原始 OCR 報告](../docs/course-experiments/results/ocr.json)。

</details>

