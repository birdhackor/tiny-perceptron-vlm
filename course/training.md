# 親手訓練：從一個有限任務開始

選一個你能判斷答案的小任務，例如從「顏色=紅；形狀=圓」回答形狀。這頁帶你準備材料、確認數值通路、更新參數，再用沒拿來教模型的題目檢查。各節是不同主題的操作入口，可以選眼前需要的一條，不必一次跑完所有實驗。

指令從有 `pyproject.toml` 的專案根目錄執行。第一次安裝與開啟請看[W.1](first-steps.md#W.1)，檔案位置請看[W.7](first-steps.md#W.7)。短練習可用 CPU；標明 CUDA 的完整配方需要相容 GPU 環境。自己的報告和模型放在 `outputs/` 與 `checkpoints/`，不要用課程的歷史數字代替自己的結果。

## T.1 先決定要拿哪些例子教模型

一筆屬性問答可以是「顏色=紅；形狀=圓；形狀？」→「圓」。同樣的材料若改問顏色，答案就要改成「紅」。先說清楚你要教的輸入、要求與答案，才知道資料是否適合。文字接續則不同：整篇文章本身提供下一個文字單位，沒有另外一份問答標籤。

先用三筆材料看分組：

| 題號 | 輸入 | 答案 | 素材家族 |
| --- | --- | --- | --- |
| A | `color=red;shape=square;pitch=low;shape?` | square | red:square:low |
| B | `color=red;shape=square;pitch=low;describe` | square | red:square:low |
| C | `color=blue;shape=circle;pitch=high;shape?` | circle | blue:circle:high |

這套產生器的 `describe` 約定回答形狀，所以 A、B 只是同一組屬性換問法。可以把 A、B 一起用來教，把 C 的整個家族留作新題；把 A 教過再拿 B 當新素材，就會高估能力。實際切分方法見[5.10](chapters/05.md#5.10)。

初次練習可用程式產生的小資料。需要公開故事、對話或模態素材時，再查看資料包：

```bash
python scripts/fetch_training_assets.py --list
python scripts/fetch_training_assets.py --asset tinystories
```

`--list` 列名稱與大小；第二行只下載 TinyStories 起步包，核對後放進 `data/training/`。下載不更新模型。打開幾筆，檢查 `text` 或 `messages` 等欄位，再安排訓練、驗證與最後測試。每包的固定來源、授權和內容指紋見[資料說明](../assets/training/README.md)。

圖片的裁切、錄音的加噪版本，也要跟原始素材放在同一側。先核對圖像尺寸、聲音取樣率與任務答案，不能以為檔案能打開就適合模型。第 19 章使用的有限商品、字卡與語音需求，和這些起步實驗分開記錄；成熟模型延伸的資料另見[第 20 章資料指引](../docs/natural-assistant/v4/DATA.md)。

練習替另一組屬性寫兩個問法，標出答案與共同家族，再寫一組不同屬性的新題。先把「教過什麼」和「準備檢查什麼」分清楚。

## T.2 先讓一組資料走過模型

模型還沒更新前，先讓一批材料進去，確認它能算出有限代價，參數也收到更新方向。這像開工前先試機器的通路；它不是能力考試。

```bash
python scripts/train.py --task text --device cpu
python scripts/train.py --task sft --device cpu
python scripts/train.py --task vision --device cpu
```

三行分別檢查文字接續、助手回答與圖片問答。沒有 `--train` 時，工具只做前向與反向，不更新、不保存訓練權重。前向用目前參數預測，反向計算代價對參數的敏感度；真正調整要由更新工具執行。

先看報告的幾個欄位：

| 欄位 | 這一步要確認什麼 |
| --- | --- |
| `mode` | 應為 `dry-run-no-weight-update` |
| `task`、`device` | 與命令選擇相符 |
| `loss` | 是有限數字，沒有 NaN 或無窮大 |
| `grad_norm` | 有限且大於零，表示至少有參數收到非零梯度 |
| `effective_tokens` | 若入口有統計，確認有效計分位置大於零 |

這個圖片入口未統計最後一欄，可能寫 `null`；它表示未提供數量，不能讀成零個目標。`grad_norm` 大於零也沒有證明每個參數都有梯度。要查特定零件，回到[5.1](chapters/05.md#5.1)或[10.7 的回答位置](chapters/10.md#10.7)看逐項檢查。

選一條命令，把模式、代價和梯度大小記下來。能解釋「通路已工作，但尚未更新」後，才進入訓練。

## T.3 先訓練接字表，再比較固定窗口

接字表只看前一字；固定窗口 MLP 可以同時讀幾個字。先用相同短文與切分，分別保存更新前、更新後的代價，看看猜法是否改變。

預設資料是十二篇顏色與形狀短文。工具先去掉完全重複的文件、按整篇切分，再取窗口；種子 42 的訓練／驗證／測試是 9／1／2 篇。字表只看訓練側。先建立輸出資料夾，再取得起點：

```bash
mkdir -p outputs checkpoints
.venv/bin/python scripts/train_simple.py --model bigram --seed 42 --device cpu > outputs/bigram-before.json
.venv/bin/python scripts/train_simple.py --model mlp --context 3 --width 16 --seed 42 --device cpu > outputs/mlp-before.json
```

這兩次沒有更新。報告的 `train_loss` 與 `validation_loss` 是兩側平均下一字代價；`parameters` 是可調數字總數，`split_unit` 說明切分單位。

現在先更新接字表：

```bash
.venv/bin/python scripts/train_simple.py --model bigram --seed 42 --device cpu --train --steps 200 --output checkpoints/bigram.pt > outputs/bigram-after.json
```

`--train` 開啟更新，`--steps` 是次數，`--output` 保存模型。報告中的 `train_loss`、`validation_loss` 使用最後更新後的參數；`last_batch_loss_before_update` 是最後一步更新之前，不能把兩個時點混在一起。

打開前後 JSON，分別計算兩個 loss 的「更新後減更新前」。負值表示代價下降。訓練側下降而驗證側上升，說明這批練習題改善沒有延伸到留出題；不要只挑訓練側看。

接著用相同資料訓練 MLP：

```bash
.venv/bin/python scripts/train_simple.py --model mlp --context 3 --width 16 --seed 42 --device cpu --train --steps 200 --output checkpoints/mlp.pt > outputs/mlp-after.json
```

核對 MLP 自己的前後報告，再比較不同模型。改窗口也會改參數量，不能把所有差異都歸因於前文長度。這些存檔屬於接字表／MLP格式，不能直接交給 Transformer 的 `infer.py`。

<details>
<summary>取得已訓練模型與固定比較</summary>

[公開模型入口](https://huggingface.co/birdhackor/tiny-perceptron-course-models)列出各獨立實驗的模型卡。可以只取得需要的一組：

```bash
.venv/bin/python scripts/fetch_course_models.py --list
.venv/bin/python scripts/fetch_course_models.py --model text_foundation
.venv/bin/python scripts/infer.py checkpoints/course/text_foundation/model.pt --prompt "color=blue;shape=circle;" --tokens 32 --device cpu --json
```

`text_foundation` 是英文規則文字 Transformer，和本節中文接字表不同。`--tokens 32` 限制新增編碼單位，遇 EOS 可以提前結束。原始 ID 與停止資訊也保存在 JSON，不能只看可讀文字。

要重做本節四種固定模型比較，用：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment simple_models --device cpu
```

結果在 `outputs/course-experiments/course-v1/simple_models/`，歷史結果與完整設定見[原始報告](../docs/course-experiments/results/simple_models.json)。公開學生包只含推論所需狀態；精確接續原訓練所需的完整檔案，見[5.7](chapters/05.md#5.7)。

</details>

## T.4 把文字訓練和對話練習分開看

同一組屬性可以寫成短文，也可以寫成問題與答案。文字階段練習接續原文；對話階段則把助手回答作為示範。先把兩種資料放在眼前，再選要練的路線。

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/prepare_data.py --kind attributes-sft --seed 42
```

產物在 `data/generated/`，各有 train、validation、test 和 manifest。兩種主要格式是：

```jsonl
{"text":"color=red;shape=circle;side=left.","family":"color=red;shape=circle;side=left."}
{"messages":[{"role":"user","content":"color=red;shape=circle;pitch=low;shape?"},{"role":"assistant","content":"circle"}],"family":"red:circle:low"}
```

第一行練下一文字單位；第二行的問題提供線索，只有助手的 `circle` 與結束目標計回答代價。同一屬性家族的各種問題放在同一側。預設文字分成 9／1／2 篇；問答分成 45／5／10 題，每個家族有五個要求。

先保存同一個未訓練起點。把下段存為 `outputs/save_start.py`，用 `.venv/bin/python outputs/save_start.py` 執行：

```python
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import save_checkpoint, seed_everything

seed_everything(42)
model = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
save_checkpoint("checkpoints/start.pt", model, step=0, metadata={"seed": 42, "note": "untrained baseline"})
print("保存隨機起點；沒有更新參數")
```

這個模型每個位置有 32 個特徵、一層、一個注意力頭，最多 128 個位置。保存起點是為了前後比較；沒有更新器狀態，不是中斷後的續訓檔。

文字路線先評估、更新、再評估：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/start.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --device cpu --limit all --output outputs/text-before.json
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --checkpoint checkpoints/start.pt --seed 42 --device cpu --train --steps 200 --output checkpoints/text.pt
.venv/bin/python scripts/evaluate.py checkpoints/text.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --device cpu --limit all --output outputs/text-after.json
```

想直接練習對話，則從同一個隨機起點走另一支：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/start.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --device cpu --limit all --output outputs/attributes-before.json
.venv/bin/python scripts/train.py --task sft --data data/generated/attributes-sft/train.jsonl --checkpoint checkpoints/start.pt --seed 42 --device cpu --train --steps 500 --output checkpoints/attributes.pt
.venv/bin/python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --device cpu --limit all --output outputs/attributes-after.json
```

若要沿用文字階段再教對話，把 SFT 的 `--checkpoint` 改成 `checkpoints/text.pt`，另取輸出檔名；同時先評估那份文字權重作為對話的起點。這是新的階段，載入父權重並開新更新工具，不加 `--resume`。兩階段的理由見[7.17](chapters/07.md#7.17)，共同成品的接續見[19.4](chapters/19.md#19.4)。

評估後打開 `--output` 指定的完整 JSON。主要欄位如下：

| 欄位 | 要核對什麼 |
| --- | --- |
| `mean_token_nll`、`effective_tokens` | 有效位置的平均代價，以及實際計分位置數 |
| `exact_match` | SFT 回答內容 ID 是否和目標一致；不刪空白或藏非法角色標記 |
| `completed_exact_match`、`eos_rate` | 內容匹配且正常結束；另記主動結束比例 |
| `samples` | 逐題的目標、實際生成、原始 ID 與停止原因 |
| `skipped`、`metric_denominators` | 哪些題未完成評估，以及各指標的實際分母 |

文字模式沒有標準問答，因此匹配率為 `null`。SFT 的 `row=0` 對應同一份 JSONL 的第一筆；從該筆 `messages` 找問題，再和 `samples` 的目標及生成並排。`--limit all` 選全檔，仍要確認是否有跳過題。`--tokens` 算新增 byte／結構單位，不等於中文字數；預算與結束見[7.19](chapters/07.md#7.19)。

用同樣的資料、題號、生成上限比較前後。練習先指出平均代價是否改變，再從逐題回答找一個有意義的變化；不只挑一個變好的答案，也不以提高上限遮住結束問題。

<details>
<summary>故事、切詞與固定對話實驗</summary>

固定課程配方使用與上述小型 CPU 練習不同的設定，不要混用成績。按需要選一組：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset tinystories --asset chinese-poetry
.venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment tokenizer --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset ultrachat-sft
.venv/bin/python -m scripts.course_experiments.run --experiment sft --device cuda
```

各組在 `outputs/course-experiments/course-v1/<組名>/` 保存結果、權重和原始切分。`sft/model.pt` 是直接對話版；`pretrain.pt` 與 `pretrain-sft.pt` 是另一條兩階段支線。後面的固定比較若依賴 `sft/`，先完成這個完整入口，不能拿本節 `checkpoints/attributes.pt` 當成相同模型。

BPE 權重必須配對它的 tokenizer JSON，推論要加 `--tokenizer`。不同切詞的 ID 不能互換，來源與完整步驟見[第 6 章](chapters/06.md)及[實驗入口](../docs/course-experiments/README.md)。要比較錯標與回放，先完成固定 `sft`，再執行 `--experiment sft_ablation`。

</details>

## T.5 把風格、指令遵循與安全拆成不同目標

希望回答生動，和希望只回一個數字，是不同要求。先替同一題寫正確短答與有用比喻，再檢查兩者是否都遵守格式。安全材料也要分開可完成、應拒絕與缺資訊的情境，不能只獎勵拒絕句。

下面建立兩個獨立練習模型：

```bash
python scripts/prepare_data.py --kind style
python scripts/prepare_data.py --kind safety
python scripts/train.py --task sft --data data/generated/style/train.jsonl --train --steps 500 --max-length 256 --output checkpoints/style.pt
python scripts/evaluate.py checkpoints/style.pt --data data/generated/style/validation.jsonl --mode sft --tokens 128 --limit all --output outputs/style-validation.json
python scripts/train.py --task sft --data data/generated/safety/train.jsonl --train --steps 500 --max-length 256 --output checkpoints/safety.pt
python scripts/evaluate.py checkpoints/safety.pt --data data/generated/safety/validation.jsonl --mode sft --tokens 128 --limit all --output outputs/safety-validation.json
```

`style` 材料有短答、比喻及 JSON 要求；`safety` 用作者設定的盒子權限規則。每次示範要真的完成對應要求。這是有限題目，不是對一般安全性的保證。500 步是練習設定，結果由你的報告決定。

把 validation 的提問、理想回答與生成並排，分別記內容、指定範圍、格式與結束。風格可以接受不同措辭，需要另看比喻是否有用；固定字串匹配不能代替這項判斷。詳細方法見[8.16](chapters/08.md#8.16)與[9.6](chapters/09.md#9.6)。

練習只替換「簡短」為「生動」，保留同一問題與格式。若多了比喻卻把答案數字改錯，就沒有完成目標。

<details>
<summary>固定風格、安全與 LoRA 比較</summary>

完整配方按依賴先後執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment style --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset pku-safe-rlhf
.venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment lora --device cuda
```

固定 `style/` 另保存 `content.pt` 等起點，和上面的 `checkpoints/style.pt` 不同。`lora/` 的 adapter 要載入原始浮點基模並核對指紋；已合併修正的 `merged-*.pt` 不再加同一 adapter。使用示例與原始欄位見[實驗說明](../docs/course-experiments/README.md)。

PKU-SafeRLHF 起步包標示 CC-BY-NC-4.0；它與本課原創盒子題的授權不同。選取、截短支線和衍生權重沒有放進公開學生包，公開安全報告保留聚合數字。使用或分享前，先核對[資料授權](../assets/training/README.md)。

</details>

## T.6 讓圖片與聲音先有可用的基礎特徵

本節先練局部玩具任務：圖片分方形、圓形，聲音分低音、高音。入口要從像素或聲音數字學到這些線索，接頭才能把它們送進文字模型。真人語音需求和短中文字卡是不同材料，另沿第 19 章整合設計。

```bash
.venv/bin/python scripts/pretrain_encoders.py --modality vision --train --steps 300 --output checkpoints/vision-encoder.pt
.venv/bin/python scripts/pretrain_encoders.py --modality audio --train --steps 300 --output checkpoints/audio-encoder.pt
```

視覺留出兩張新位置圖，方形類別 0、圓形 1；音訊留出 180Hz 低音類別 0、1000Hz 高音 1。查看最後 JSON 的 `mode`、`holdout_examples` 與 `holdout_accuracy`。這條局部流程先要求兩題全對才接續；0.5 只是答對一題，保存檔存在不能代替檢查。

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

希望答案依序為 `circle`、`high`、`square,low`。開啟 JSON 的 `answer` 和它們比較，再看結束是否完整。两道編碼器題全對，也沒有保證轉接後每題都對。

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

自己的圖像紀錄可寫成 `{"image":"images/12.png","question":"read digits","answer":"12"}`，路徑相對 JSONL。這個局部入口會轉 RGB 並縮至 16×16，可能丟失細字；聲音入口要求非空 16kHz 單聲道，不會自動重採樣。輸入必須和模型學過的任務匹配。

做數字圖的正常／空白／錯配對照，可用 `prepare_ocr.py` 產生資料，再以 `train.py --task vision` 訓練，並用 `evaluate_modal.py --ablation none`、`blank`、`shuffle` 檢查同一批題目。錯配報告的 `donor_row` 指出換入圖來源；它仍對原答案計分，要另外核對生成是否符合換入圖。完整欄位見[11.13](chapters/11.md#11.13)及[原始 OCR 報告](../docs/course-experiments/results/ocr.json)。

</details>

## T.7 先檢查回答能力，再學較偏好的回答

偏好資料讓同一題有兩篇候選，再指出較合適的一篇。例如 `0+5=?` 的候選 5 與 6，可以依真值選 5。先確認比較同一問題，再用 DPO 改變回答傾向。

```bash
.venv/bin/python scripts/prepare_data.py --kind preference
.venv/bin/python scripts/train.py --task dpo --checkpoint checkpoints/style.pt --data data/generated/preference/train.jsonl --train --steps 200 --output checkpoints/preferred.pt
```

這裡用 T.5 的 `style.pt` 作起點，另保留不更新的參考副本。先看 `chosen`、`rejected` 是否符合自己的判準；參考是原回答傾向，不是真值老師。

訓練前後用同一題生成：

```bash
.venv/bin/python scripts/infer.py checkpoints/style.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0
.venv/bin/python scripts/infer.py checkpoints/preferred.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0
```

保存原問題、候選與兩份實際回答，再對其餘驗證題做相同檢查。候選 5 比 6 的機率高，仍不表示自由生成一定是 5；也要保留原來的內容、格式與安全題，確認新階段沒有破壞它們。[13.5](chapters/13.md#13.5)說明排序與生成的差別。

<details>
<summary>DPO 固定配方與 PPO 選卡路線</summary>

固定 DPO 先完成 T.5 的正式 `style`，再取偏好包：

```bash
.venv/bin/python scripts/fetch_training_assets.py --asset ultrafeedback-dpo
.venv/bin/python -m scripts.course_experiments.run --experiment dpo --device cuda
```

`dpo/` 保存策略、固定參考與偏好切分；`preference` 欄核對候選比較，`arithmetic` 欄核對實際生成，各自有分母。完整格式見[報告](../docs/course-experiments/results/dpo.json)。

想逐步理解獎勵、價值估計與 PPO，可另跑獨立 CPU 選卡例子：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment posttraining --device cpu
```

它選一張預寫卡，不逐token生成；評分標籤來自作者規則。角色和更新流程見[13.10–13.17](chapters/13.md#13.10)。DPO與PPO是兩種路線，不必依序套用。

</details>

## T.8 架構比較一次只換一個條件

先保存基準，再只換一個零件。若同時換資料、寬度與步數，便很難知道差異來自哪裡。下面固定同一份短文，對照原模型、RMS正規化與四專家MoE：

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --output checkpoints/baseline.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --norm rms --output checkpoints/rms.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --experts 4 --top-k 2 --output checkpoints/moe.pt
```

MoE每次選兩位專家，但全部專家權重仍需保存。同寬度不等於同參數或同計算預算，比較條件要一起記錄。接著用同一個 validation 評估三份模型：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/baseline.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/baseline-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/rms.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/rms-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/moe.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/moe-validation.json
```

把平均代價、有效位置數與逐筆生成並排；另記參數量、裝置與時間。品質、儲存和實際速度各有自己的單位，不能只用「比較快」替整個設計下結論。先在驗證側選設定，再使用最後測試。

<details>
<summary>現代零件與效率的固定配方</summary>

取得 TinyStories 包後，可選 `--experiment modern` 或 `--experiment moe` 重做第14／15章的固定比較。效率組另依賴 T.4 的完整 `sft/model.pt` 和 `dataset.json`：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment efficiency --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment precision --device cuda
```

### 選讀：效率實驗的量測條件

推論先暖機，再在計時兩側等待 GPU 完成；保存多次採樣與中位數。記錄提示長度、生成長度、精度、後端與工作範圍。訓練計時是否含反向和更新，也要寫清楚。

PyTorch已配置記憶體是交給張量的空間，不等於整張GPU用量。保存量測前基線、過程峰值與新增量；不要只比較新增量便宣稱整張卡節省同樣倍數。詳細操作見[16.1](chapters/16.md#16.1)與[效率報告](../docs/course-experiments/results/efficiency.json)。

真正Flash核心的獨立核驗是：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment flash_probe --device cuda
```

它只測固定 Q/K/V 的注意力前向與反傳，不訓練整個模型。要核對實際後端、數值容差及記憶體範圍，不能由 SDPA 這個 API 名稱推定已用 Flash；完整條件在[原報告](../docs/course-experiments/results/flash_probe.json)。

</details>

## T.9 真的縮小保存格式，再量品質

把數字四捨五入後仍存成浮點數，未必能縮小檔案。先從同一份已訓練浮點模型，各自轉換不同位數，再檢查保存大小與相同題目的回答。

```bash
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 4 --output checkpoints/attributes-int4.pt
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 8 --output checkpoints/attributes-int8.pt
```

兩份都從原始 `attributes.pt` 出發，不把8-bit再轉成4-bit。轉換只改支援的Linear權重；嵌入、正規化、偏移與刻度仍有自己的儲存。本課推論會先反量化成浮點數，所以檔案變小沒有直接證明執行記憶體同樣變小或更快。

```bash
.venv/bin/python scripts/infer.py checkpoints/attributes.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py checkpoints/attributes-int4.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
```

希望答案是 `blue`。這一題先讓你查看前後有沒有改變；完整能力檢查仍用同一份留出題。原訓練檔可能含更新器與隨機狀態，比較部署大小時要用相同用途的匯出檔，或另看純張量bytes，不能把訓練狀態的差異也算作量化收益。

練習保存原版與兩個量化版的大小，再記「新增答對、新增答錯與不變」的題數。相同總答對數可能是不同題目答對。

<details>
<summary>固定量化與量化感知訓練</summary>

固定比較先完成T.4的完整`sft`，再跑：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment quantization --device cpu
```

這個入口另做120次更新才轉換，和上面的單獨 `quantize.py` 不同；不能互換成績。它保存浮點、4-bit、8-bit與共同題目，格式見[量化報告](../docs/course-experiments/results/quantization.json)。[17.14](chapters/17.md#17.14)介紹QAT：讓訓練先適應模擬誤差；真正低位元核心仍需格式與硬體支援。

</details>

## T.10 用較小學生學教師，再與普通訓練比較

先選較小的學生架構，再比較它只學真值、與另加教師機率的差別。兩支學生要共用起點、資料與更新安排；否則看到差異，也難知道是不是教師訊號帶來的。

這套固定入口使用已完成的屬性、條件風格與MoE三種教師。先取得T.4完整`sft`、T.5完整`style`、T.8完整`moe`，保留各自的模型與資料切分，再執行：

```bash
.venv/bin/python scripts/fetch_training_assets.py --asset gsm8k
.venv/bin/python -m scripts.course_experiments.run --experiment distillation --device cpu
```

這是多支線的完整訓練，CPU需要時間；相容環境可改CUDA。產物在 `distillation/`，不同任務、學生尺寸與教師訊號各有檔名。GSM8K包只用作有限域外診斷，並不是這份屬性模型已具備解應用題的證據。

先選同寬度的兩支，用相同題目檢查：

```bash
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce_kl.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
```

希望答案是 `square,high`。有的題改善，也可能另有題退步；核對整份相同題目，並一起記學生參數、教師成本與有效訓練目標。只變小是架構的效果，教師訊號是否有幫助由這個對照回答。

教師和學生的文字單位也必須對齊：同一位置、同一候選字表，才可以直接比較機率。多模態學生還可能減少圖像位置，對齊方式另見[18.13](chapters/18.md#18.13)。蒸餾後再量化是另一個改動，要分開驗收。

<details>
<summary>原始資料、教師與多模態路線</summary>

完整教師、學生、對照與生成見[蒸餾報告](../docs/course-experiments/results/distillation.json)，不同任務有各自分母，不用單一數字代表全部能力。

多模態蒸餾另依賴固定 `encoders`、`projector`、`vqa` 與 `joint` 來源，再執行 `--experiment multimodal_distillation`。先保留來源權重和原始切分；僅下載推論包不足以恢復訓練資料。操作與格式見[實驗說明](../docs/course-experiments/README.md)。

</details>

## T.11 留下別人能核對的實驗紀錄

隔一週回頭看「這次比較好了」，你可能忘了改哪個地方、用哪些題目。記錄要把材料、做法和結果接起來，才方便自己檢查或讓別人重做。

用量化比較作例子，開始前先記下：

| 要留下的內容 | 用途 |
| --- | --- |
| 原始模型與程式版本 | 確認兩版的共同起點 |
| 資料來源、授權、切分與指紋 | 確認同一批可使用的題目 |
| 兩條轉換／評估命令 | 確認主要改動是位數 |
| 生成上限、裝置與精度 | 確認比較條件 |
| 逐題目標、回答與停止原因 | 確認內容、格式、結束，以及題目分母 |
| 檔案大小與實際量測範圍 | 確認儲存、記憶體、時間各指什麼 |

SHA-256是檔案指紋，Git commit是程式版本；兩者讓你取得同一份內容，不會自動證明資料正確或模型有能力。未量過的欄位就填未量測。

例如先訂「4-bit檔案較小，且同批題目答對數不減」。假設原版80,000bytes、4/5題對，新版60,000bytes、3/5題對，它符合保存變小，沒有符合保住答對數。這是手寫判準例子，數字不是模型實測。

比較時也留原始逐題結果。兩版都4/5，可能錯在不同題；一版先丟掉難題，也不能和另一版的全檔比例直接比較。驗證題可以用來選設定，最後題留到選完後才開啟。

紀錄做到能回答「改了什麼、哪個條件固定、結果支持哪件事」，就能完成一次有限比較。文字、圖解或排版修訂本身不需要重跑訓練；模型、資料或能力宣稱改變時，再做相應驗收。

<details>
<summary>共同成品、成熟模型與完整歷史紀錄</summary>

[第19章](chapters/19.md)追蹤我們自行訓練的共同模型及各階段父權重；其能力表和交付指引與局部實驗分開。[第20章](chapters/20.md)是沿用成熟Qwen／Whisper的延伸，專用操作見[重訓指引](../docs/natural-assistant/v4/TRAINING.md)，上游能力不算成我們從零學會。

完整歷史結果在[實驗紀錄](../docs/course-experiments/README.md)。RAG、工具與推理報告保存各自的原始輸入、輸出及分母。工具要留真正執行與回填後的回答；只生成請求文字，還沒有完成操作。

</details>
