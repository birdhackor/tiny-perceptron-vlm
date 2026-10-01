# 訓練操作：選一個小問題，留下真實結果

下面是提供給讀者之後執行的配方。本次教材開發只驗證程式、梯度與介面，沒有執行正式模型訓練。步數是起點，並不保證模型學會任務。先固定資料、提示與評估，再觀察實際結果。

所有指令從 repo 根目錄、啟用 `.venv` 後執行。預設 `--device auto` 依序選 CUDA、MPS、CPU；想指定 GPU 可加 `--device cuda`。音訊 STFT 的裝置支援要先跑 dry-run，CPU 是已驗證的基準。

## 已收集的真實資料

[首批固定資料](../assets/training/README.md)以分來源 Git LFS 包發布。可先用 `python scripts/fetch_training_assets.py --list` 查看，再以 `--asset tinystories` 等 ID 按需取得。快照有來源、固定 revision、授權與逐檔 SHA-256；解包後存入 `data/training/`。它們是小型 pilot，仍須建立獨立 holdout；教材下方的規則資料則可離線生成，不必先下載外部資料。FSDD 保留原始8 kHz音訊，16 kHz入口前要明確轉換。

## 先檢查一個 batch

```bash
python scripts/train.py --task text --device cpu
python scripts/train.py --task sft --device cpu
python scripts/train.py --task vision --device cpu
```

沒有 `--train` 時只算一次 forward/backward，不呼叫 optimizer 更新或保存 checkpoint。這可發現尺寸、遮罩、資料與梯度問題；它不能衡量模型學習效果。Notebook 的短數值更新是公式驗算，不是持續訓練模型。

## 1–2章：接字表與固定窗口

```bash
python scripts/train_simple.py --model bigram
python scripts/train_simple.py --model mlp --context 3
```

準備好要訓練時：

```bash
python scripts/train_simple.py --model bigram --train --steps 200 --output checkpoints/bigram.pt
python scripts/train_simple.py --model mlp --context 3 --train --steps 200 --output checkpoints/mlp.pt
```

這條支線用字元詞表、完整文件切分與留出 loss。它的存檔格式獨立，不能送進 Transformer 的 `infer.py`。比較兩者時用相同文件切分，並記錄參數數量；MLP 能看更多卡片，也使用不同的計算量。

## 4–7章：文字、對話與資料切分

先產生少量可核對的規則資料：

```bash
python scripts/prepare_data.py --kind toy-text
python scripts/prepare_data.py --kind attributes-sft
```

產生器先按題目／屬性家族分組，再分 train、validation、test。同題的風格變體留在同一側；`manifest.json` 記錄筆數、來源、詞表與原始 JSONL 的 SHA-256。小規則資料只適合教學與除錯，尚不足以學會一般語言。

```bash
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --width 32 --output checkpoints/text.pt
python scripts/evaluate.py checkpoints/text.pt --data data/generated/toy-text/validation.jsonl --mode text --output outputs/text-validation.json
python scripts/infer.py checkpoints/text.pt --prompt "color=" --tokens 32 --cache
```

`text` 模式讀 `{"text":"..."}`。Transformer CLI 使用固定 UTF-8 byte 詞表；第6章的字元與 BPE 是切詞比較實驗，不會默默換掉存檔的詞表。評估按有效 token 合計 loss，再算包含邊界目標的 BPB；長文件按窗口重設上下文，因此要和相同評估版本比較。

獨立練習「文字屬性→答案」，可以從新的小模型開始：

```bash
python scripts/train.py --task sft --data data/generated/attributes-sft/train.jsonl --train --steps 500 --output checkpoints/attributes.pt
python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --output outputs/attributes-validation.json
python scripts/infer.py checkpoints/attributes.pt --chat --prompt "color=red;shape=square;pitch=low;shape?"
```

SFT 讀 `messages` 或 `question/answer`；只監督 assistant 的內容與 EOS。評估同時保存 teacher-forced loss 和自由生成的逐題答案。完全匹配是一個窄指標；多種合理措辭的任務還要人工規準。超過上下文的評估資料會列出跳過理由，不會當成答對。

### 接續與換階段

`--checkpoint` 載入模型開始新階段；`--resume --checkpoint` 同時恢復 optimizer、步數與 RNG，目前限 text／sft。`--steps` 表示整份排程的總步數，不是額外步數：

```bash
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --resume --checkpoint checkpoints/text.pt --train --steps 200 --output checkpoints/text.pt
```

這用來接續**中途**存檔；已完成200步會明確拒絕。text／sft 預設每100步與正常結束時存檔，可用 `--save-every` 調整間隔。改變總步數、學習率、資料或裝置後，不再聲稱和原配方逐步一致。

## 8–9章：個性、格式、誠實與安全

```bash
python scripts/prepare_data.py --kind style
python scripts/prepare_data.py --kind safety
python scripts/train.py --task sft --data data/generated/style/train.jsonl --train --steps 500 --max-length 256 --output checkpoints/style.pt
python scripts/evaluate.py checkpoints/style.pt --data data/generated/style/validation.jsonl --mode sft --tokens 128 --output outputs/style-validation.json
python scripts/train.py --task sft --data data/generated/safety/train.jsonl --train --steps 500 --max-length 256 --output checkpoints/safety.pt
```

風格資料對同一道算術題提供短答、帶譬喻的回答、JSON。請分別檢查答案數字、格式／長度遵循、風格評分；較會修辭不代表較不遵循指令。安全資料只教「盒子與測試碼的權限」這個小規則；讀者需要另備無害完成、正當邊界與資訊不足的樣本，分開計算完成率、過度拒絕與澄清是否相關。這組 toy 分數無法支持開放世界的安全結論。

LoRA 的小修正層在 `alignment.LoRALinear`，8.8–8.9直接驗證凍結、梯度與合併。要做完整微調實驗時，先明確選擇包哪些 Linear、保存 adapter 配置，再沿第5章的迴圈更新 `requires_grad=True` 的參數。上述 CLI 採完整 SFT，沒有暗中啟用 LoRA。

## 10–12章：圖片、聲音與聯合輸入

先用有真值的分類任務學編碼器，保存後移除分類頭：

```bash
python scripts/pretrain_encoders.py --modality vision --train --steps 300
python scripts/pretrain_encoders.py --modality audio --train --steps 300
```

視覺任務是合成形狀分類，音訊任務是合成高／低音分類，各自有留出組合。它們不代表自然圖片理解或語音辨識。載入已檢查的文字與模態基礎能力，再練轉接頭：

```bash
python scripts/train.py --task vision --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --freeze projector --train --steps 500 --output checkpoints/vision.pt
python scripts/train.py --task audio --checkpoint checkpoints/attributes.pt --audio-encoder checkpoints/audio-encoder.pt --freeze projector --train --steps 500 --output checkpoints/audio.pt
python scripts/train.py --task joint --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --audio-encoder checkpoints/audio-encoder.pt --freeze partial --train --steps 500 --output checkpoints/joint.pt
python scripts/infer_modal.py checkpoints/vision.modal.pt --color blue --shape circle
python scripts/infer_modal.py checkpoints/audio.modal.pt --frequency 880
python scripts/infer_modal.py checkpoints/joint.modal.pt --color red --shape square --frequency 220
```

`--freeze none` 更新所有零件，`projector` 只更新轉接頭，`partial` 再開啟文字模型首末層。轉接頭尺寸相容不保證能對齊；比較時先確認文字回答與編碼器分類各自已學會。模態資料逐筆展開，`--batch-size` 筆 loss 等例平均；尚不是高效批次模態訓練。

每次模態訓練保存兩份：`.pt` 是抽出的文字模型，`.modal.pt` 是含編碼器與轉接頭的完整模型。用前者做固定文字測試，觀察語言能力是否變差；用後者做模態推論。此入口尚未保存模態 optimizer，不能直接 resume。

### 自己的圖片／音訊資料

多模態 JSONL 使用 `question`、`answer`，以及 `image`／`audio` 檔案路徑，路徑相對於 JSONL 的資料夾：

```json
{"image":"images/12.png","question":"read digits","answer":"12"}
```

圖片轉 RGB 並縮成16×16；音訊必須事先轉成非空、16 kHz 單聲道 WAV 等可解碼格式，沒有默默重採樣。長聲音展開後可能超過 `max-length`，應先分片。這個小 encoder 的尺寸是為玩具實驗選的；自然照片與文件不能只靠提高訓練步數解決。

### 數字圖片與模態消融

```bash
python scripts/prepare_ocr.py
python scripts/train.py --task vision --data data/generated/ocr/train.jsonl --train --steps 1000 --max-length 128 --output checkpoints/ocr.pt
python scripts/infer_modal.py checkpoints/ocr.modal.pt --image data/generated/ocr/images/12-0.png --prompt "read digits"
python scripts/evaluate_modal.py checkpoints/ocr.modal.pt --data data/generated/ocr/validation.jsonl --output outputs/ocr-normal.json
python scripts/evaluate_modal.py checkpoints/ocr.modal.pt --data data/generated/ocr/validation.jsonl --ablation blank --output outputs/ocr-blank.json
python scripts/evaluate_modal.py checkpoints/ocr.modal.pt --data data/generated/ocr/validation.jsonl --ablation shuffle --output outputs/ocr-shuffled.json
```

產生器手畫固定5×7字形，拼成一到兩位數；同一答案的位移變體放在同側。validation／test 保留未見的數字字串，而非宣稱新字體能力。比較正常、空白、錯配圖片時保持同一題、同一模型，逐題查看原本就答錯的例子。若正常能力還很低，遮圖後也低無法證明忽略圖片。

12章的影片部分先驗證逐幀切片、順序與時間位置；現成 CLI 的模態選項是 vision／audio／joint，沒有提供影片生成或完整影片訓練入口。

## 13章：偏好

```bash
python scripts/prepare_data.py --kind preference
python scripts/train.py --task dpo --checkpoint checkpoints/style.pt --data data/generated/preference/train.jsonl --train --steps 200 --output checkpoints/preferred.pt
```

`chosen/rejected` 可是共用 `prompt` 的兩個回答，或兩份完整 messages。參考模型是開始這階段時的凍結副本；先用 SFT 建立合理能力，再檢查內容與遵循能力是否隨偏好訓練退化。偏好分數和一般答案品質不是同一件事。第13章的 reward／PPO，以及C支線的 REINFORCE，是小型公式實驗，不是另一套通用 RLHF 訓練系統。

## 14–16章：只改一個架構或效能條件

```bash
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --output checkpoints/baseline.pt
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --norm rms --output checkpoints/rms.pt
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --experts 4 --top-k 2 --output checkpoints/moe.pt
```

其他開關有 `--rotary`、`--activation swiglu`／`relu2`、`--tied`、`--heads`／`--kv-heads`、`--backend sdpa`。先逐一和基準比，最後才組合。Dense／MoE要另外匹配總參數或活躍計算量；相同 width 並不公平。SDPA是後端介面，是否走 Flash kernel 由裝置與條件決定；CPU等價測試不能證明GPU加速。

記錄固定資料與版本、seed、有效訓練 token、參數量、留出 loss、任務分數、生成樣本、tokens/s、峯值記憶體。GPU計時先暖身，區間兩側同步；不同硬體的秒數不要直接當架構結論。玩具 MoE 有 Python 派工成本，可能比 Dense 慢。

## 17章：真正低位元儲存

```bash
python scripts/quantize.py checkpoints/attributes.pt --bits 4 --output checkpoints/attributes-int4.pt
python scripts/infer.py checkpoints/attributes-int4.pt --chat --prompt "color=red;shape=square;pitch=low;shape?"
python scripts/evaluate.py checkpoints/attributes-int4.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --output outputs/int4-validation.json
```

這會把 Linear 權重量化並真正 pack 成4／8 bit的 buffer；embedding、正規化與 bias 留浮點，推論會反量化回浮點運算。共享輸入／輸出權重在這份格式中解除，報告會明示。原始 checkpoint 可能包含 optimizer；量化檔案較小的比例不能直接叫純權重壓縮率，更不能推成執行記憶體或速度。

QAT 單元使用 `fake_quantize` 看前向誤差與近似梯度；要得到硬體加速，仍需相容的準備／匯出格式與低位元 kernel。教材沒有悄悄加入未驗證的硬體後端。

## 18章：較小學生與蒸餾

黑盒路線先把教師回答存成有來源、revision、prompt 的 messages，再沿 SFT 入口訓練學生。白盒路線使用同一 byte 詞表與位置：

```bash
python scripts/train.py --task distill --teacher checkpoints/attributes.pt --data data/generated/attributes-sft/train.jsonl --width 16 --layers 1 --train --steps 500 --alpha 0.5 --temperature 2 --output checkpoints/student.pt
python scripts/evaluate.py checkpoints/student.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --output outputs/student-validation.json
```

教師必須先經過真值與行為檢查，模型容量與 context 也要相容。CLI 正式蒸餾沒有 `--teacher` 會停下來；dry-run的隨機教師只驗證介面。比較同樣小架構的「CE-only」與「CE+KL」，固定步數和資料，再檢查省下的容量、內容正確與格式遵循。多模態蒸餾先用18.13核對展開後答案位置，不能直接把不同長度的 logits 相減。

## 一份實驗紀錄至少留下什麼

1. 資料 manifest、切分家族、tokenizer、程式 revision與各模型配置。
2. 開始權重、更新步數、有效 token、seed、可訓練參數與凍結策略。
3. 固定留出資料的 loss、生成樣本、分項能力，以及失敗／跳過數量。
4. 硬體、精度、實際檔案 bytes、峯值記憶體與計時方式。

選配方使用 validation，test 留到最後。一次看到較好結果，先換幾個 seed 重做，再討論它在這組資料與硬體上的意義。
