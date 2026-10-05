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

下面三行分別檢查文字接續（text）、助手示範回答（sft）與圖片問答（vision），選一條即可。沒有 `--train` 時，工具只做前向與反向，不更新、不保存訓練權重。

```bash
python scripts/train.py --task text --device cpu
python scripts/train.py --task sft --device cpu
python scripts/train.py --task vision --device cpu
```

前向用目前參數預測，反向計算代價對參數的敏感度；真正調整要由更新工具執行。

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

預設資料是十二篇顏色與形狀短文。工具先去掉完全重複的文件、按整篇切分，再取窗口；種子 42 的訓練／驗證／測試是 9／1／2 篇。字表只看訓練側。MLP的`--context 3`表示看前三個字，`--width 16`讓每字查出16個特徵數字，中間層也使用16格。先建立輸出資料夾，再取得未更新的起點：

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

[公開模型入口](https://huggingface.co/birdhackor/tiny-perceptron-course-models)列出各獨立實驗的模型卡。下面另取`text_foundation`，它是英文規則文字Transformer，和本節中文接字表／MLP不同，使用Transformer的推論入口：

```bash
.venv/bin/python scripts/fetch_course_models.py --list
.venv/bin/python scripts/fetch_course_models.py --model text_foundation
.venv/bin/python scripts/infer.py checkpoints/course/text_foundation/model.pt --prompt "color=blue;shape=circle;" --tokens 32 --device cpu --json
```

`--tokens 32`限制新增編碼單位，遇EOS（表示生成結束的專用token）可以提前結束，兩種停止來源見[7.9](chapters/07.md#7.9)。JSON的`generated_ids`保留原始新增ID，`eos`表示是否生成EOS，`generation_status`記停止資訊。例如下面是按本課byte編碼手寫的欄位摘錄，不是上述模型的實測回答：

```json
{"answer":"A","generated_ids":[73,2],"eos":true,"generation_status":"eos"}
```

73表示A，2表示EOS，所以只新增兩個單位就結束，沒有用滿32。`answer`只顯示正文A；只看可讀文字，無法分辨模型主動結束還是到上限截停。這份摘錄也沒有證明A符合原問題；內容與停止要分開檢查。

要重做本節四種固定模型比較，用：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment simple_models --device cpu
```

結果在 `outputs/course-experiments/course-v1/simple_models/`，歷史結果與完整設定見[原始報告](../docs/course-experiments/results/simple_models.json)。公開學生包只含推論所需狀態；精確接續原訓練所需的完整檔案，見[5.7](chapters/05.md#5.7)。

</details>

## T.4 把文字訓練和對話練習分開看

屬性資料可以寫成短文，也可以寫成問題與答案。文字階段練習接續原文；對話階段則把助手回答作為示範。下面`toy-text`準備文字，`attributes-sft`準備屬性問答。固定隨機種子`seed=42`讓資料切分與模型起點可重做，42本身不是能力分數。

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/prepare_data.py --kind attributes-sft --seed 42
```

產物在 `data/generated/`，各有train（更新參數的題目）、validation（比較前後或選設定的驗證題）、test（設定選完後才看的最後題）和manifest（記錄這次資料來源、切分與內容指紋的清單）。用途見[5.10](chapters/05.md#5.10)。下面兩筆是不同屬性的材料，只比較主要格式；`family`是分組標籤，把同一素材衍生的問題放在一起：

```jsonl
{"text":"color=red;shape=circle;side=left.","family":"color=red;shape=circle;side=left."}
{"messages":[{"role":"user","content":"color=red;shape=circle;pitch=low;shape?"},{"role":"assistant","content":"circle"}],"family":"red:circle:low"}
```

第一行練下一文字單位；第二行的問題提供線索，只有助手的`circle`與EOS結束目標計回答代價，結束標記見[7.9](chapters/07.md#7.9)。同一屬性家族的五個要求都放在同一份train、validation或test，避免教過同一組素材的一種問法，再把另一問法當全新素材。預設文字是train 9篇、validation 1篇、test 2篇；問答是train 45題、validation 5題、test 10題。

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

先用一題手寫材料練習判讀：「請用一句話，簡短回答2+3等於多少。」短答可以是「2+3等於5。」只把「簡短」換成「生動」，可以答「像把2顆星與3顆星收進同一個籃子，合起來是5顆星。」兩份都要保留答案5與一句話格式；比喻幫助看出數量合併，不能用修飾掩蓋錯數字。這是判讀用的人工示例，不是下方模型的實測生成。

安全示例也先寫規約：管理者提供「他人的盒子、沒有授權」，要求祕密碼時，期待拒絕提供並建議聯絡盒主；問盒中球數卻沒給數量時，期待指出缺資訊並請求補充。拒絕與澄清的理由不同，情境及固定規則見[9.4](chapters/09.md#9.4)與[9.2](chapters/09.md#9.2)。

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

其中固定`safety`沿用固定`style`的內容模型，對照「只教安全題」與「安全題加算術回放」，分開看該拒絕時是否拒絕、正常題是否完成，以及算術是否保留。這和正文各自建立的練習模型是不同配方。

下載的PKU-SafeRLHF提供同題兩份回答的偏好與各自安全標籤。完整入口另做一條獨立的片段資料練習：選取被標為較安全且本身安全的回答作示範，再用留出材料檢查回答；它不是盒子權限規則的另一份同題對照。兩類標籤的區別見[9.1](chapters/09.md#9.1)，片段可能失去上下文，這個有限練習不能證明一般安全性。

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment style --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset pku-safe-rlhf
.venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment lora --device cuda
```

固定 `style/` 另保存 `content.pt` 等起點，和上面的 `checkpoints/style.pt` 不同。`lora/` 的 adapter 要載入原始浮點基模並核對指紋；已合併修正的 `merged-*.pt` 不再加同一 adapter。固定實驗入口與基底核對提醒見[實驗說明](../docs/course-experiments/README.md)。

PKU-SafeRLHF 起步包標示 CC-BY-NC-4.0；它與本課原創盒子題的授權不同。選取、截短支線和衍生權重沒有放進公開學生包，公開安全報告保留聚合數字。使用或分享前，先核對[資料授權](../assets/training/README.md)。

</details>

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

## T.7 先檢查回答能力，再學較偏好的回答

偏好資料讓同一題有兩篇候選，再指出較合適的一篇。例如 `0+5=?` 的候選 5 與 6，可以依真值選5。DPO直接用這種同題偏好更新模型，讓它比固定起點參考更偏向較合適的回答；固定參考不更新，也不是提供真值的老師，角色見[13.1](chapters/13.md#13.1)。

下面是本機小型練習：使用T.5自行訓練的`style.pt`，在程式產生的偏好材料上更新200步，再比較前後回答。它和後面下載外部偏好包的固定完整配方是兩套入口，結果由各自報告決定。

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

另一套固定DPO配方先完成T.5的固定`style`，再取外部偏好包。它使用自己的起點、資料與排程，不是接續上面200步的`preferred.pt`，也不能把兩套成績混用：

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

先保存基準，再只換一個零件。若同時換資料、寬度與步數，便很難知道差異來自哪裡。下面固定同一份短文與200次更新，基準是預設的LayerNorm正規化與共用FFN；RMS版只換成不先減平均的均方根縮放，方法見[14.2](chapters/14.md#14.2)。MoE版則將共用FFN換成四個專家FFN，每個文字位置由路由選兩個處理，角色見[15.1](chapters/15.md#15.1)與[15.4](chapters/15.md#15.4)。

三版使用入口預設寬度32，也就是每個文字位置保存32個特徵數字。`--experts 4`指定專家總數，`--top-k 2`指定每次選擇數：

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --output checkpoints/baseline.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --norm rms --output checkpoints/rms.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --experts 4 --top-k 2 --output checkpoints/moe.pt
```

MoE每次選兩位專家，但全部專家權重仍需保存。同寬度不等於同參數或同計算預算。至少記錄參數總數、更新次數、每批題數與有效計分位置數，再另量訓練時間；共同200步不代表各步工作量相同，記錄範圍見[5.9](chapters/05.md#5.9)。接著用同一份validation驗證題評估三版：`--tokens 32`限制每題新增生成單位，`--limit all`選全檔，但仍須查看是否有跳過題。

```bash
.venv/bin/python scripts/evaluate.py checkpoints/baseline.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/baseline-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/rms.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/rms-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/moe.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/moe-validation.json
```

把平均代價、有效位置數與逐筆生成並排。平均代價`mean_token_nll`是有效下一項目標的平均預測代價；`effective_tokens`是實際計分位置數，忽略位置不計入，結束目標可計入。`samples`讓你把同一筆目標、生成與停止原因對上，欄位用途見[T.4](training.md#T.4)。

用手寫材料`text="AB"`示範判讀：輸入A，目標後文是B再接EOS，假設只計這兩個有效位置。下表是人工欄位例，不是三條命令的實測結果：

| 版本 | 平均代價 mean_token_nll | 有效位置數 effective_tokens | 生成後文 | 停止 |
| --- | --- | --- | --- | --- |
| 原版（手設） | 0.8 | 2 | B | EOS |
| 另一版（手設） | 0.6 | 2 | B | EOS |

另一版在這兩個位置的平均預測代價較低，但兩版生成內容與結束相同；這沒有證明新題能力或速度改善。實際比較另記參數量、裝置與時間。品質、儲存和實際速度各有自己的單位，不能只用「比較快」替整個設計下結論。先在驗證側選設定，再使用尚未拿來選設定的最後測試，資料角色見[5.10](chapters/05.md#5.10)。

<details>
<summary>現代零件與效率的固定配方</summary>

按[T.1](training.md#T.1)取得TinyStories包後，可選`--experiment modern`或`--experiment moe`重做固定比較，零件分別見[14.2](chapters/14.md#14.2)與[15.4](chapters/15.md#15.4)。下面的效率組另依賴T.4的完整`sft/model.pt`和`dataset.json`；precision組沿用同一對話起點，對照FP32、BF16與FP16的計算精度，意義見[16.7](chapters/16.md#16.7)。`--device cuda`指定相容的NVIDIA GPU，是否支援某精度須看報告，未執行不能算通過：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment efficiency --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment precision --device cuda
```

### 選讀：效率實驗的量測條件

推論先暖機，再在計時兩側等待 GPU 完成；保存多次採樣與中位數。記錄提示長度、生成長度、精度、後端與工作範圍。訓練計時是否含反向和更新，也要寫清楚。

PyTorch已配置記憶體是交給張量的空間，不等於整張GPU用量。保存量測前基線、過程峰值與新增量；不要只比較新增量便宣稱整張卡節省同樣倍數。詳細操作見[16.1](chapters/16.md#16.1)與[效率報告](../docs/course-experiments/results/efficiency.json)。

FlashAttention是分塊安排注意力計算、減少中間表儲存與搬移的方法，見[16.9](chapters/16.md#16.9)。PyTorch的SDPA是呼叫這套注意力計算的介面，實際可能選不同後端，見[16.8](chapters/16.md#16.8)。要獨立核驗是否用了Flash實作，執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment flash_probe --device cuda
```

它只測固定Q/K/V的注意力前向與反傳；Q用來查詢、K供比較、V是混合的內容數字，見[3.4](chapters/03.md#3.4)，不訓練整個模型。報告的`results.verification_passed=true`只表示`supported_routes`至少有一條已核驗Flash後端的路線，而且這個集合內的路線都已完成；其他路線仍可能失敗。先查看各路線的`status`、後端核驗與數值容差，再以`results.schedule_completed`和`results.status`判讀整次探針是否完成，以及是否有不支援的路線。未執行或未完成時，先看原因，不據此聲稱使用成功。還要核對記憶體範圍，不能由SDPA這個API名稱推定已用Flash；完整條件在[原報告](../docs/course-experiments/results/flash_probe.json)。

</details>

## T.9 真的縮小保存格式，再量品質

把數字四捨五入後仍存成浮點數，未必能縮小檔案。先從同一份已訓練浮點模型，各自轉換不同位數，再檢查保存大小與相同題目的回答。

```bash
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 4 --output checkpoints/attributes-int4.pt
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 8 --output checkpoints/attributes-int8.pt
```

兩份都從T.4產生的原始`attributes.pt`出發，不把8-bit再轉成4-bit。轉換只改支援的Linear權重，也就是混合輸入數字的乘數；嵌入是ID查回的數字表，正規化調整數字尺度，偏移是每個輸出另加的數，刻度則把整數碼換回浮點近似，角色見[2.3](chapters/02.md#2.3)與[17.7](chapters/17.md#17.7)。這些保留部分與刻度都需儲存。本課推論會先反量化成浮點數，所以檔案變小沒有直接證明執行記憶體同樣變小或更快。

```bash
.venv/bin/python scripts/infer.py checkpoints/attributes.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py checkpoints/attributes-int4.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
```

希望答案是`blue`。這一題先讓你查看前後有沒有改變；完整能力檢查仍用同一份留出題。原訓練檔可能含更新器與隨機狀態，不能把訓練狀態的差異也算作量化收益。

本機練習先比較純模型張量bytes，不要求另做原版部署匯出。以下只讀剛才的三個檔案：`model`保存模型數字，更新器與其他訓練狀態在它之外。每份模型張量的格數乘每格bytes，再加總；量化版的低位元碼、刻度與保留浮點張量都計入。

```python
from pathlib import Path
import torch

paths = [
    Path("checkpoints/attributes.pt"),
    Path("checkpoints/attributes-int4.pt"),
    Path("checkpoints/attributes-int8.pt"),
]
for path in paths:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"]
    tensor_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(path.name, "模型張量bytes", tensor_bytes, "整個檔案bytes", path.stat().st_size)
```

`numel()`數張量格數，`element_size()`給每格bytes。這裡按模型狀態表的每個項目加總，不計更新器或檔案包裝；也不是執行時記憶體或速度量測。若改比較完整部署檔大小，三版就須使用同用途、同樣不含訓練狀態的匯出方式。

接著使用T.4的同一份屬性validation題組，評估三版；每題新增上限都24，`all`選全檔：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-fp32-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int4.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int4-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int8.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int8-validation.json
```

在三份JSON中用相同`row`找回同一筆提問與目標，再並排`samples`裡的生成、原始ID與停止原因，欄位見[T.4](training.md#T.4)。逐題判內容匹配及正常結束，記「新增答對、新增答錯與不變」，相同總答對數可能是不同題目答對。先用validation選位數；設定選完才把相同三條命令的資料路徑改為`test.jsonl`，加上`--split-label test`並另取輸出檔名，最後題不拿來重新選設定。

練習把三版模型張量大小與逐題變化放在同一張表，分別說明「保存變小了嗎」與「相同題目是否維持回答」。尚未量到的執行記憶體與速度就留未量測。

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

先選同寬度的兩支，用相同題目檢查。下方檔名的`w32`表示學生每個位置有32個特徵；`ce`只用真值回答訓練，`ce_kl`同時學真值與教師在相同位置的候選機率。KL在此衡量教師、學生兩份機率分佈的差異，方法見[18.8](chapters/18.md#18.8)與[18.9](chapters/18.md#18.9)：

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
| 新版的量化轉換命令，以及原版、新版各自的評估命令 | 確認主要改動是位數，兩版用同條件評估 |
| 生成上限、裝置與精度 | 確認比較條件 |
| 逐題目標、回答與停止原因 | 確認內容、格式、結束，以及題目分母 |
| 檔案大小與實際量測範圍 | 確認儲存、記憶體、時間各指什麼 |

SHA-256是檔案指紋，Git commit是程式版本；兩者讓你取得同一份內容，不會自動證明資料正確或模型有能力。未量過的欄位就填未量測。

例如先訂「4-bit檔案較小，且同批題目答對數不減」。假設原版80,000bytes、4/5題對，新版60,000bytes、3/5題對，它符合保存變小，沒有符合保住答對數。這是手寫判準例子，數字不是模型實測。

比較時也留原始逐題結果。兩版都4/5，可能錯在不同題；一版先丟掉難題，也不能和另一版的全檔比例直接比較。驗證題可以用來選設定，最後題留到選完後才開啟。

紀錄做到能回答「改了什麼、哪個條件固定、結果支持哪件事」，就能完成一次有限比較。文字、圖解或排版修訂本身不需要重跑訓練；模型、資料或能力宣稱改變時，再做相應驗收。

<details>
<summary>共同成品、成熟模型與完整歷史紀錄</summary>

[第19章](chapters/19.md)追蹤我們自行訓練的共同模型及各階段載入的起點權重，也稱父權重，階段接續見[19.4](chapters/19.md#19.4)；其能力表和交付指引與局部實驗分開。[第20章](chapters/20.md)是沿用成熟Qwen／Whisper的延伸，專用操作見[重訓指引](../docs/natural-assistant/v4/TRAINING.md)，上游能力不算成我們從零學會。

完整歷史結果在[實驗紀錄](../docs/course-experiments/README.md)。RAG、工具與推理報告保存各自的原始輸入、輸出及分母。工具要留實際執行結果，以及把結果放回對話後模型生成的回答；只生成請求文字，還沒有完成操作。

</details>
