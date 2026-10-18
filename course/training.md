# 親手訓練：從一個有限任務開始

選一個你能判斷答案的小任務，例如從「顏色=紅；形狀=圓」回答形狀。這頁帶你準備材料、確認數值通路、更新參數，再用沒拿來教模型的題目檢查。各節是不同主題的操作入口，可以選眼前需要的一條，不必一次跑完所有實驗。

開始前先確認四件事：

- **在哪裡執行。** 這頁的命令都在本機的終端機執行，起點是有 `pyproject.toml` 的專案根目錄；它們不是 Notebook 的程式格，這頁也沒有對應的 Notebook。本課沒有在 Colab 實測過這些命令，請在自己的電腦上跑。第一次安裝請看 [W.1](first-steps.md#W.1)，檔案位置請看 [W.7](first-steps.md#W.7)。
- **需要什麼。** 照 W.1 裝好 `.venv` 就能跑本機練習。用 `fetch_training_assets.py --asset` 下載資料包時需要 Git LFS，安裝與確認方式也在 W.1。命令寫 `--device cuda` 的固定配方需要 NVIDIA GPU，以及 CUDA 版的安裝（選擇見[環境說明](../docs/environment.md)）；沒有 NVIDIA GPU 時能否改用 CPU 跑完，各配方都未量測。下載公開模型需要連上 Hugging Face。
- **Python 怎麼寫。** 命令一律寫成 `.venv/bin/python …`，直接使用專案工具箱裡的 Python，不必先啟用 `.venv`。Windows PowerShell 把這段路徑換成 `.\.venv\Scripts\python.exe`，其餘參數不變。沒寫 `--device` 的命令由工具自動選裝置：有 NVIDIA GPU 時用 CUDA，Apple Silicon 可能用 MPS，否則用 CPU；本課沒有在 MPS 上驗證整套訓練，想固定用 CPU 就加 `--device cpu`。
- **要花多久。** 有作者實驗紀錄（JSON，這是機器可讀的完整結果）的命令會寫出時間與硬體，數字取自紀錄裡的 `elapsed_seconds`，只算實驗本身，不含安裝、啟動與下載。沒有寫出時間的命令都沒有量測紀錄，各節的本機訓練會再註明「未量測」。`train.py` 的報告有 `seconds` 欄，可以記下你自己這次訓練迴圈花的秒數。

自己的報告和模型放在 `outputs/` 與 `checkpoints/`，不要用課程的歷史數字代替自己的結果。

## T.1 先決定要拿哪些例子教模型

一筆屬性問答可以是「顏色=紅；形狀=圓；形狀？」→「圓」。同樣的材料若改問顏色，答案就要改成「紅」。先說清楚你要教的輸入、要求與答案，才知道資料是否適合。文字接續則不同：整篇文章本身提供下一個文字單位，沒有另外一份問答標籤。

先用三筆材料看分組：

| 題號 | 輸入 | 答案 | 家族 |
| --- | --- | --- | --- |
| A | `color=red;shape=square;pitch=low;shape?` | square | red:square:low |
| B | `color=red;shape=square;pitch=low;describe` | square | red:square:low |
| C | `color=blue;shape=circle;pitch=high;shape?` | circle | blue:circle:high |

家族是同一組屬性衍生出來的題目；切分資料時，同一家族整組放在同一邊。這套產生器（[scripts/prepare_data.py](../scripts/prepare_data.py) 的 `attributes-sft`）把 `describe` 定為回答形狀，所以 A、B 只是同一組屬性換問法。可以把 A、B 一起用來教，把 C 的整個家族留作新題；把 A 教過再拿 B 當新素材，就會高估能力。實際切分方法見[5.10](chapters/05.md#5.10)。

初次練習可用程式產生的小資料。需要公開的故事、對話，或圖片、聲音素材時，再查看資料包：

```bash
.venv/bin/python scripts/fetch_training_assets.py --list
.venv/bin/python scripts/fetch_training_assets.py --asset tinystories
```

`--list` 只讀專案裡的清單，列出名稱、筆數、大小與授權，不需要 Git LFS。第二行下載 TinyStories 起步包（357,896 bytes）。照 [W.1](first-steps.md#W.1) 的寫法 clone 下來時，專案裡的資料包只是 Git LFS 的指標檔；這時工具會先執行 `git lfs pull`，從網路取得實際檔案，電腦沒裝 Git LFS 就會報錯中止。所以先照 [W.1](first-steps.md#W.1) 安裝，並用 `git lfs version` 確認。取得檔案後，工具會自動核對大小與 SHA-256，再解包到 `data/training/`（`--output` 可改位置）。下載不更新模型。打開幾筆，檢查 `text` 或 `messages` 等欄位，再安排訓練、驗證與測試資料。每包的固定來源、授權和內容指紋見[資料說明](../assets/training/README.md)。

圖片的裁切、錄音的加噪版本，也要跟原始素材放在同一側。先核對圖像尺寸、聲音取樣率與任務答案，不能以為檔案能打開就適合模型。第 19 章的商品圖、字卡與語音需求另有自己的資料清單，第 20 章的資料見[第 20 章資料指引](../docs/natural-assistant/v4/DATA.md)；它們都和這些起步實驗分開記錄。

練習替另一組屬性寫兩個問法，標出答案與共同家族，再寫一組不同屬性的新題。先把「教過什麼」和「準備檢查什麼」分清楚。

## T.2 先讓一組資料走過模型

模型還沒更新前，先讓一批材料進去，確認它能算出有限代價，參數也收到更新方向。這像開工前先試機器的通路；它不是能力考試。

下面三行分別檢查文字接續（text）、助手示範回答（sft）與圖片問答（vision），選一條即可。這三條命令先用程式內建的小材料：文字是顏色與形狀短文，對話是兩個小整數相加的問答，圖片是程式畫出的圓或方塊。要讀自己的資料檔時，再加上 `--data` 和檔案路徑。沒有 `--train` 時，工具只對一批資料做一次前向計算與反向傳播，不更新參數、不保存檢查點。

```bash
.venv/bin/python scripts/train.py --task text --device cpu
.venv/bin/python scripts/train.py --task sft --device cpu
.venv/bin/python scripts/train.py --task vision --device cpu
```

前向計算用目前參數預測；反向傳播算出代價對每個參數的敏感度，也就是梯度。真正調整參數要由最佳化器（optimizer）執行，這三條命令沒有這一步。

終端機會先印一行這一步的紀錄，最後一行才是整份報告。先看最後一行的幾個欄位；`loss` 等批次數值放在報告的 `history` 第一筆：

| 欄位 | 這一步要確認什麼 |
| --- | --- |
| `mode` | 應為 `dry-run-no-weight-update` |
| `task`、`device` | 與命令選擇相符 |
| `loss` | 是有限數字，沒有 NaN 或無窮大 |
| `grad_norm` | 有限且大於零，表示至少有參數收到非零梯度 |
| `effective_tokens` | 有效目標數，應大於零 |

有效目標是真正拿來計算代價的位置；填充的位置與不計分的提問部分不算在內。圖片任務的報告不統計這一欄，會寫 `null`：它表示沒有提供數量，不能讀成零個目標。`grad_norm` 大於零也沒有證明每個參數都有梯度。要查某個零件有沒有收到梯度，回到[5.1](chapters/05.md#5.1)看 `.grad` 與 `.norm()` 的逐項檢查；圖片問答的答案目標在圖片展開後落在哪些位置，見[10.7](chapters/10.md#10.7)。

選一條命令，把模式、代價和梯度大小記下來。能解釋「通路已工作，但尚未更新」後，才進入訓練。

## T.3 先訓練接字表，再比較固定視窗

接字表（[1.5](chapters/01.md#1.5) 的 bigram 表）只看前一字；固定視窗 MLP 可以同時讀前面幾個字。先用相同短文與切分，分別保存更新前、更新後的代價，看看猜法是否改變。

預設資料是十二篇中文的顏色與形狀短文，例如「顏色=紅；形狀=圓。」。這個工具以字元為單位，「下一字」就是下一個字元；T.4 起的 Transformer 改用 byte tokenizer，單位就不同了。工具先去掉完全重複的文件、按整篇切分，再取視窗；種子 42 的訓練／驗證／測試是 9／1／2 篇。字表只看訓練側。MLP 的 `--context 3` 表示看前三個字，也就是 [2.5](chapters/02.md#2.5) 的視窗長度；`--width 16` 是特徵寬度，讓每字查出 16 個特徵數字，中間寬度也是 16。本節命令在你電腦上的時間未量測。先建立輸出資料夾（Windows PowerShell 把 `mkdir -p outputs checkpoints` 換成 `New-Item -ItemType Directory -Force -Path outputs,checkpoints`），再取得未更新的起點：

```bash
mkdir -p outputs checkpoints
.venv/bin/python scripts/train_simple.py --model bigram --seed 42 --device cpu > outputs/bigram-before.json
.venv/bin/python scripts/train_simple.py --model mlp --context 3 --width 16 --seed 42 --device cpu > outputs/mlp-before.json
```

這兩次沒有更新。報告的 `train_loss` 與 `validation_loss` 是訓練側與驗證側的平均下一字代價；`parameters` 是可調數字總數，`split_unit` 說明切分單位。

現在先更新接字表：

```bash
.venv/bin/python scripts/train_simple.py --model bigram --seed 42 --device cpu --train --steps 200 --output checkpoints/bigram.pt > outputs/bigram-after.json
```

`--train` 開啟更新，`--steps` 是次數，`--output` 保存模型。報告中的 `train_loss`、`validation_loss` 使用最後更新後的參數；`last_batch_loss_before_update` 是最後一步更新之前，不能把兩個時點混在一起。

打開前後 JSON，分別計算兩個 loss 的「更新後減更新前」。負值表示代價下降。訓練側下降而驗證側上升，說明這批練習題的改善沒有延伸到驗證題；不要只挑訓練側看。

接著用相同資料訓練 MLP：

```bash
.venv/bin/python scripts/train_simple.py --model mlp --context 3 --width 16 --seed 42 --device cpu --train --steps 200 --output checkpoints/mlp.pt > outputs/mlp-after.json
```

核對 MLP 自己的前後報告，再比較不同模型。改視窗也會改參數量，不能把所有差異都歸因於前文長度。這些檢查點屬於接字表／MLP 格式，不能直接交給 Transformer 的 `infer.py`。

<details>
<summary>取得已訓練模型與固定比較</summary>

[公開模型入口](https://huggingface.co/birdhackor/tiny-perceptron-course-models)列出各獨立實驗的模型卡。下面另取 `text_foundation`，它是英文規則文字 Transformer，和本節中文接字表／MLP 不同，使用 Transformer 的推論工具：

```bash
.venv/bin/python scripts/fetch_course_models.py --list
.venv/bin/python scripts/fetch_course_models.py --model text_foundation
.venv/bin/python scripts/infer.py checkpoints/course/text_foundation/model.pt --prompt "color=blue;shape=circle;" --tokens 32 --device cpu --json
```

`--tokens 32` 限制最多新增 32 個 token；生成 EOS（結束標記）時可以提前結束，兩種停止來源見[7.9](chapters/07.md#7.9)。JSON 的 `generated_ids` 保留原始新增 ID，`eos` 表示是否生成 EOS，`generation_status` 記停止資訊。例如下面是按本課 byte tokenizer 手寫的欄位摘錄，不是上述模型的實測回答：

```json
{"answer":"A","generated_ids":[73,2],"eos":true,"generation_status":"eos"}
```

73表示A，2表示EOS，所以只新增兩個 token 就結束，沒有用滿32。`answer`只顯示正文A；只看可讀文字，無法分辨模型主動結束還是到上限截停。這份摘錄也沒有證明A符合原問題；內容與停止要分開檢查。

固定比較包含兩類模型、四個設定：接字表，以及一字、三字、五字視窗的 MLP。視窗取幾個位置的差別見[2.5](chapters/02.md#2.5)；要重做這組比較，用：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment simple_models --device cpu
```

作者在 Modal 雲端容器的 CPU 上跑過一次，`elapsed_seconds` 為 4.3 秒。結果在 `outputs/course-experiments/course-v1/simple_models/`；作者的結果與完整設定見[實驗紀錄（JSON）](../docs/course-experiments/results/simple_models.json)。公開學生包只含推論所需狀態；精確接續原訓練所需的完整檔案，見[5.7](chapters/05.md#5.7)。

</details>

## T.4 把文字訓練和對話練習分開看

屬性資料可以寫成短文，也可以寫成問題與答案。文字階段練習接續原文；對話階段則把助手回答作為示範。下面`toy-text`準備文字，`attributes-sft`準備屬性問答。固定隨機種子`seed=42`讓資料切分與模型起點可重做，42本身不是能力分數。

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/prepare_data.py --kind attributes-sft --seed 42
```

產物在 `data/generated/`，各有train（更新參數的訓練題）、validation（比較前後或選設定的驗證題）、test（設定選完後才看的測試題）和manifest（記錄這次資料來源、切分與內容指紋的清單）。用途見[5.10](chapters/05.md#5.10)。下面兩筆是不同屬性的材料，只比較主要格式；`family`是分組標籤，同一家族的題目放在一起：

```jsonl
{"text":"color=red;shape=circle;side=left.","family":"color=red;shape=circle;side=left."}
{"messages":[{"role":"user","content":"color=red;shape=circle;pitch=low;shape?"},{"role":"assistant","content":"circle"}],"family":"red:circle:low"}
```

第一行練下一個 token；本課的 byte tokenizer 把一個 UTF-8 byte 或一個特殊 token（例如 EOS）算成一個 token。第二行的問題提供線索，只有助手的`circle`與EOS結束目標計回答代價，結束標記見[7.9](chapters/07.md#7.9)。同一家族的五種問法都放在同一份train、validation或test，避免教過同一組素材的一種問法，再把另一問法當全新素材。預設文字是train 9篇、validation 1篇、test 2篇；問答是train 45題、validation 5題、test 10題。

先保存同一個未訓練起點。用任何純文字編輯器（例如 VS Code）把下段存成 `outputs/save_start.py`，再用 `.venv/bin/python outputs/save_start.py` 執行：

```python
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import save_checkpoint, seed_everything

seed_everything(42)
model = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
save_checkpoint("checkpoints/start.pt", model, step=0, metadata={"seed": 42, "note": "untrained baseline"})
print("保存隨機起點；沒有更新參數")
```

這個模型特徵寬度 32、一層、一個注意力頭，最多 128 個位置。保存起點是為了前後比較；檔案裡沒有最佳化器狀態，不能拿來從中斷處精確接續訓練。

文字路線先評估、更新、再評估。本節的訓練與評估命令在你電腦上的時間未量測；`train.py` 報告的 `seconds` 欄會記下你這次的訓練迴圈時間。

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

若要沿用文字階段再教對話，起點就換成文字路線保存的 `checkpoints/text.pt`。先評估它當作對話的起點，再從它開始 SFT，最後評估；輸出檔名另取，才不會蓋掉上面兩支的結果：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/text.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --device cpu --limit all --output outputs/text-then-sft-before.json
.venv/bin/python scripts/train.py --task sft --data data/generated/attributes-sft/train.jsonl --checkpoint checkpoints/text.pt --seed 42 --device cpu --train --steps 500 --output checkpoints/text-then-sft.pt
.venv/bin/python scripts/evaluate.py checkpoints/text-then-sft.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --device cpu --limit all --output outputs/text-then-sft-after.json
```

這是新的階段。`text.pt` 在這裡是**父檢查點**：上一階段保存、這一階段載入當起點的檢查點。工具只載入它的參數，另外建立新的最佳化器，所以不加 `--resume`；`--resume` 是同一次訓練中斷後接續用的。兩階段的理由見[7.17](chapters/07.md#7.17)，共同成品的接續見[19.4](chapters/19.md#19.4)。

評估後打開 `--output` 指定的完整 JSON。主要欄位如下：

| 欄位 | 要核對什麼 |
| --- | --- |
| `mean_token_nll`、`effective_tokens` | 有效目標的平均代價（平均 NLL），以及有效目標數 |
| `exact_match` | SFT 回答內容 ID 是否和目標一致；不刪空白或藏非法角色標記 |
| `completed_exact_match`、`eos_rate` | 內容匹配且正常結束；另記主動結束比例 |
| `samples` | 逐題的目標、實際生成、原始 ID 與停止原因 |
| `skipped`、`metric_denominators` | 哪些題未完成評估，以及各指標的實際分母 |

文字模式沒有一問一答的正確答案，因此匹配率為 `null`。SFT 的 `row=0` 對應同一份 JSONL 的第一筆；從該筆 `messages` 找問題，再和 `samples` 的目標及生成並排。`--limit all` 選全檔，仍要確認是否有跳過題。`--tokens` 算的是新增 token 數，不等於中文字數：一個常見中文字在 UTF-8 佔 3 個 byte，也就是 3 個 token，見[6.1](chapters/06.md#6.1)；預算與結束見[7.19](chapters/07.md#7.19)。

用同樣的資料、題號、生成上限比較前後。練習先指出平均代價是否改變，再從逐題回答找一個有意義的變化；不只挑一個變好的答案，也不以提高上限遮住結束問題。

<details id="fixed-sft-recipe" markdown="1">
<summary>故事、切詞與固定對話實驗；T.10教師需先完成其中的固定SFT配方</summary>

固定課程配方使用與上述小型 CPU 練習不同的設定，不要混用成績。按需要選一組：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset tinystories --asset chinese-poetry
.venv/bin/python -m scripts.course_experiments.run --experiment real_text --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment tokenizer --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset ultrachat-sft
.venv/bin/python -m scripts.course_experiments.run --experiment sft --device cuda
```

| 配方 | 重做什麼 | 作者的紀錄與時間（Modal 上的 NVIDIA L4） |
| --- | --- | --- |
| `text_foundation` | 9 篇顏色與形狀短文的文字訓練，600 次更新 | [text_foundation.json](../docs/course-experiments/results/text_foundation.json)，16.0 秒 |
| `real_text` | TinyStories 與唐詩各自訓練 800 次 | [real_text.json](../docs/course-experiments/results/real_text.json)，23.9 秒 |
| `tokenizer` | byte 與 BPE 兩種切詞各訓練 400 次 | [tokenizer.json](../docs/course-experiments/results/tokenizer.json)，16.2 秒 |
| `sft` | 屬性問答 SFT 900 次，另含先預訓練再 SFT 的支線與 UltraChat 小型實跑 | [sft.json](../docs/course-experiments/results/sft.json)，31.7 秒 |

沒有 NVIDIA GPU 時，可以把 `--device cuda` 改成 `--device cpu`，但 CPU 可行性未量測：能不能跑完、要多久都沒有紀錄。指定 `cuda` 而電腦沒有 CUDA 時，工具會直接停下，不會自動改用 CPU；這個工具也只接受 `cpu` 與 `cuda`，不支援 Apple 的 MPS。

各組在 `outputs/course-experiments/course-v1/<組名>/` 保存結果、權重和原始切分。`sft/model.pt` 是直接對話版；`pretrain.pt` 與 `pretrain-sft.pt` 是另一條兩階段支線。後面的固定比較若依賴 `sft/`，先完成這個完整配方，不能拿本節 `checkpoints/attributes.pt` 當成相同模型。

BPE 權重必須配對它的 tokenizer JSON，推論要加 `--tokenizer`。不同切詞的 ID 不能互換，來源與完整步驟見[第 6 章](chapters/06.md)及[實驗說明](../docs/course-experiments/README.md)。要比較錯標與回放，先完成固定 `sft`，再執行 `--experiment sft_ablation`（作者在 L4 上的紀錄是 23.5 秒，見 [sft_ablation.json](../docs/course-experiments/results/sft_ablation.json)）。

</details>

## T.5 把風格、指令遵循與安全拆成不同目標

希望回答生動，和希望只回一個數字，是不同要求。先替同一題寫正確短答與有用比喻，再檢查兩者是否都遵守格式。安全材料也要分開可完成、應拒絕與缺資訊的情境，不能只獎勵拒絕句。

先用一題手寫材料練習判讀：「請用一句話，簡短回答2+3等於多少。」短答可以是「2+3等於5。」只把「簡短」換成「生動」，可以答「像把2顆星與3顆星收進同一個籃子，合起來是5顆星。」兩份都要保留答案5與一句話格式；比喻幫助看出數量合併，不能用修飾掩蓋錯數字。這是判讀用的人工示例，不是下方模型的實測生成。

安全示例也先寫規約：管理者提供「他人的盒子、沒有授權」，要求祕密碼時，期待拒絕提供並建議聯絡盒主；問盒中球數卻沒給數量時，期待指出缺資訊並請求補充。拒絕與澄清的理由不同，情境及固定規則見[9.4](chapters/09.md#9.4)與[9.2](chapters/09.md#9.2)。

下面建立兩個獨立練習模型。產生器的提問寫成 `style=concise; 2+3=?`、`style=vivid; 2+3=?` 與 `style=json; 2+3=?` 這種格式：`concise` 是短答版，`vivid` 是積木比喻版。`--kind safety` 只產生兩種題：有授權（`permission=True`）時協助，沒授權（`permission=False`）時拒絕並建議詢問盒主；缺資訊而該請對方補充的題，產生器沒有，要自己另外寫。這些命令在你電腦上的時間未量測。

```bash
.venv/bin/python scripts/prepare_data.py --kind style
.venv/bin/python scripts/prepare_data.py --kind safety
.venv/bin/python scripts/train.py --task sft --data data/generated/style/train.jsonl --train --steps 500 --max-length 256 --output checkpoints/style.pt
.venv/bin/python scripts/evaluate.py checkpoints/style.pt --data data/generated/style/validation.jsonl --mode sft --tokens 128 --limit all --output outputs/style-validation.json
.venv/bin/python scripts/train.py --task sft --data data/generated/safety/train.jsonl --train --steps 500 --max-length 256 --output checkpoints/safety.pt
.venv/bin/python scripts/evaluate.py checkpoints/safety.pt --data data/generated/safety/validation.jsonl --mode sft --tokens 128 --limit all --output outputs/safety-validation.json
```

`style` 材料有短答、比喻及 JSON 要求；`safety` 用作者設定的盒子權限規則。每次示範要真的完成對應要求。這是有限題目，不是對一般安全性的保證。500 步是練習設定，結果由你的報告決定。

把 validation 的提問、理想回答與生成並排，分別記內容、指定範圍、格式與結束（方法見[8.16](chapters/08.md#8.16)）。風格可以接受不同措辭，需要另看比喻是否有用；固定字串匹配不能代替這項判斷，方法見[8.10](chapters/08.md#8.10)。安全題除了看該拒絕時是否拒絕，也要看正常題有沒有被過度拒絕，見[9.6](chapters/09.md#9.6)。

練習只把短答版換成積木比喻版，保留同一問題與格式；在產生器的格式裡，就是把 `style=concise` 換成 `style=vivid`，例如 `.venv/bin/python scripts/infer.py checkpoints/style.pt --chat --prompt "style=vivid; 2+3=?" --tokens 128`。若多了比喻卻把答案數字改錯，就沒有完成目標。

<details id="fixed-style-recipe" markdown="1">
<summary>固定風格、安全與 LoRA 比較；T.10教師需先完成其中的固定風格配方</summary>

固定`safety`沿用固定`style`的內容模型，對照「只教安全題」與「安全題加算術回放」，分開看該拒絕時是否拒絕、正常題是否完成，以及算術是否保留。這和正文各自建立的練習模型是不同配方。

下載的PKU-SafeRLHF提供同題兩份回答的偏好與各自安全標籤。完整配方另做一條獨立的片段資料練習：選取被標為較安全且本身安全的回答作示範，再用沒拿來訓練的驗證題與測試題檢查回答；它不是盒子權限規則的另一份同題對照。兩類標籤的區別見[9.1](chapters/09.md#9.1)，片段可能失去上下文，這個有限練習不能證明一般安全性。

完整配方按依賴先後執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment style --device cuda
.venv/bin/python scripts/fetch_training_assets.py --asset pku-safe-rlhf
.venv/bin/python -m scripts.course_experiments.run --experiment safety --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment lora --device cuda
```

作者在 Modal 的 NVIDIA L4 上的紀錄：`style` 29.5 秒、`safety` 22.2 秒、`lora` 24.5 秒，分別見 [style.json](../docs/course-experiments/results/style.json)、[safety.json](../docs/course-experiments/results/safety.json)、[lora.json](../docs/course-experiments/results/lora.json)。沒有 NVIDIA GPU 時改用 `--device cpu` 的可行性未量測，工具的行為見 [T.4](#T.4) 的補充區。

固定 `style/` 另保存 `content.pt` 等起點，和上面的 `checkpoints/style.pt` 不同。`lora/` 的 adapter 要載入原始浮點底座並核對指紋；已合併修正的 `merged-*.pt` 不再加同一 adapter。固定實驗的做法與底座核對提醒見[實驗說明](../docs/course-experiments/README.md)。

PKU-SafeRLHF 起步包標示 CC-BY-NC-4.0；它與本課原創盒子題的授權不同。選取、截短支線和衍生權重沒有放進公開學生包，公開安全報告保留聚合數字。使用或分享前，先核對[資料授權](../assets/training/README.md)。

</details>

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

這是本工具自己的凍結方案，不是照名稱只凍結某個零件。開放更新也要有本次輸入帶來的梯度，沒用到的入口不會因此自動學習。本練習的聯合路線選 `partial`，目的是讓接頭與首末文字區塊一起適應圖片、聲音兩種輸入；這是一組示範設定，效果仍以驗證題核對。

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

希望答案依序為 `circle`、`high`、`square,low`。開啟 JSON 的 `answer` 和它們比較，再看結束是否完整。兩個入口的驗證題全對，也沒有保證接上文字核心後每題都對。

`vision.pt`、`audio.pt`、`joint.pt` 是多模態檢查點：裡面同時有文字核心（`model.language`）與圖片、聲音的入口和接頭。檢查文字能力時，要用其中的文字核心評估同一份文字驗證題；`evaluate.py` 只接受純文字的檢查點，直接把多模態檢查點交給它會報錯。[11.7](chapters/11.md#11.7)解釋這個對照。

<details>
<summary>續訓、固定模態比較與自己的檔案</summary>

自己用命令保存的完整 `multimodal-v1` 檢查點含最佳化器狀態、進度、隨機狀態與可訓練名單。原排程未完成時，用相同任務、資料、總步數和凍結範圍，加 `--resume` 接續；不要再加 `--vision-encoder` 或 `--audio-encoder`，入口已經存在檢查點裡。僅供推論的學生包不能精確恢復原訓練，格式與例子見[5.7](chapters/05.md#5.7)。

### 重跑第11章的固定實驗

先完成 T.4 的固定 `sft`，再依序執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment projector --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment vqa --device cuda
```

作者在 Modal 的 NVIDIA L4 上的紀錄：`encoders` 10.2 秒、`projector` 15.9 秒、`vqa` 83.8 秒。沒有 NVIDIA GPU 時可以改成 `--device cpu`，但這三組屬於模態實驗，工具會把 CPU 的結果標成只檢查通路（`evidence_status` 為 `interface_smoke_only`），不算正式結果；CPU 時間未量測。

這套 `vqa` 的 `partial` 只開圖片接頭與最後文字區塊，和本節 CLI 不同。完整配方的依賴、數據和歷史結果見[實驗說明](../docs/course-experiments/README.md)，不能套成上面 500 步的預期成績。

自己的圖像紀錄可寫成 `{"image":"images/12.png","question":"read digits","answer":"12"}`，路徑相對 JSONL。這個局部工具會轉 RGB 並縮至 16×16，可能丟失細字；聲音入口要求非空、取樣率16kHz的單聲道檔案，即每秒保存16,000個聲音數字，不會自動重新取樣。取樣率不同於前面`--frequency`所設定的單音音高頻率，區別見[12.1](chapters/12.md#12.1)。輸入必須和模型學過的任務匹配。

做數字圖的正常／空白／錯配對照，可用 `prepare_ocr.py` 產生資料，再以 `train.py --task vision` 訓練，並用 `evaluate_modal.py --ablation none`、`blank`、`shuffle` 檢查同一批題目。錯配報告的 `donor_row` 指出換入圖來源；它仍拿原圖的正確答案（`target`）計分，要另外核對生成是否符合換入圖。完整欄位見[11.13](chapters/11.md#11.13)及 [OCR 實驗紀錄（JSON）](../docs/course-experiments/results/ocr.json)。

</details>

## T.7 先檢查回答能力，再學較偏好的回答

偏好資料讓同一題有兩篇候選，再指出較合適的一篇。例如 `0+5=?` 的候選 5 與 6，可以依正確答案選 5。DPO 直接用這種同題偏好更新模型，讓它比固定起點參考更偏向較合適的回答；固定參考不更新，也不是提供正確答案的老師，角色見[13.4](chapters/13.md#13.4)。

下面是本機小型練習：使用 [T.5](#T.5) 自行訓練的 `checkpoints/style.pt`，在程式產生的偏好材料上更新 200 步，再比較前後回答。它和後面下載外部偏好包的固定完整配方是兩套做法，結果由各自報告決定。這些命令在你電腦上的時間未量測。

```bash
.venv/bin/python scripts/prepare_data.py --kind preference
.venv/bin/python scripts/train.py --task dpo --checkpoint checkpoints/style.pt --data data/generated/preference/train.jsonl --train --steps 200 --output checkpoints/preferred.pt
```

這裡用 T.5 的 `style.pt` 作起點，另保留不更新的參考副本。先看 `chosen`、`rejected` 是否符合自己的判準；參考是原回答傾向，不是提供正確答案的老師。

訓練前後用同一題生成；下方`--temperature 0`在本工具中表示每步直接選最高分候選，沿用[1.15的greedy規則](chapters/01.md#1.15)，讓前後比較固定選字方式：

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

作者在 Modal 的 NVIDIA L4 上的紀錄是 18.7 秒；改用 CPU 的可行性未量測。`dpo/` 保存策略模型、固定參考與偏好切分；`preference` 欄核對候選比較，`arithmetic` 欄核對實際生成，各自有分母。完整格式見 [DPO 實驗紀錄（JSON）](../docs/course-experiments/results/dpo.json)。

想逐步理解獎勵、價值模型與 PPO，可另跑獨立 CPU 選卡例子：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment posttraining --device cpu
```

作者在 CPU 上的紀錄是 2.1 秒，見 [posttraining.json](../docs/course-experiments/results/posttraining.json)。它選一張預寫的回答卡，不逐 token 生成；評分標籤來自作者規則。角色和更新流程見[13.10–13.17](chapters/13.md#13.10)。DPO與PPO是兩種路線，不必依序套用。

</details>

## T.8 架構比較一次只換一個條件

先保存基準，再只換一個零件。若同時換資料、寬度與步數，便很難知道差異來自哪裡。下面固定同一份短文與200次更新，基準是預設的LayerNorm正規化與共用FFN；RMS版只換成不先減平均的均方根正規化，方法見[14.2](chapters/14.md#14.2)。MoE版則將共用FFN換成四個專家FFN，每個文字位置由路由選兩個處理，角色見[15.1](chapters/15.md#15.1)與[15.4](chapters/15.md#15.4)。

這是一組本機練習設定：資料是 T.4 的 `toy-text` 短文（9 篇訓練、1 篇驗證），每版更新 200 次，三版都用 `train.py` 預設的特徵寬度 32、一層、一個注意力頭。repo 沒有這組練習的結果紀錄，要比較的數字由你自己跑出來；本機時間也未量測。第 14、15 章引用的現代零件六組結果與 MoE 結果，是摺疊區另一組固定配方的設定與紀錄，見下方補充區的表。`--experts 4`指定專家總數，`--top-k 2`指定每次選擇數：

```bash
.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --output checkpoints/baseline.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --norm rms --output checkpoints/rms.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --experts 4 --top-k 2 --output checkpoints/moe.pt
```

MoE每次選兩位專家，但全部專家權重仍需保存。同寬度不等於同參數或同計算預算。至少記錄參數總數、更新次數、每批題數與有效目標數，再另量訓練時間（`train.py` 報告的 `parameters` 與 `seconds` 欄可以直接抄下）；共同200步不代表各步工作量相同，記錄範圍見[5.9](chapters/05.md#5.9)。接著用同一份validation驗證題評估三版：`--tokens 32`限制每題新增的 token 數，`--limit all`選全檔，但仍須查看是否有跳過題。

```bash
.venv/bin/python scripts/evaluate.py checkpoints/baseline.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/baseline-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/rms.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/rms-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/moe.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/moe-validation.json
```

把平均代價、有效目標數與逐筆生成並排。平均代價`mean_token_nll`是有效目標的平均預測代價；`effective_tokens`是有效目標數：填充與忽略的位置不計入，結束目標可計入。`samples`讓你把同一筆目標、生成與停止原因對上，欄位用途見[T.4](training.md#T.4)。

用手寫材料`text="AB"`示範判讀：輸入A，目標後文是B再接EOS，假設只計這兩個有效目標。下表是人工欄位例，不是三條命令的實測結果：

| 版本 | 平均代價 mean_token_nll | 有效目標數 effective_tokens | 生成後文 | 停止 |
| --- | --- | --- | --- | --- |
| 原版（手設） | 0.8 | 2 | B | EOS |
| 另一版（手設） | 0.6 | 2 | B | EOS |

另一版在這兩個位置的平均預測代價較低，但兩版生成內容與結束相同；這沒有證明新題能力或速度改善。實際比較另記參數量、裝置與時間。品質、儲存和實際速度各有自己的單位，不能只用「比較快」替整個設計下結論。先在驗證題上選設定，再使用尚未拿來選設定的測試題，資料角色見[5.10](chapters/05.md#5.10)。

Looped Transformer 用不同輪數訓練的做法與比較方式，見[14.13–14.15](chapters/14.md#14.13)；比較時要記錄平均跑了幾輪，不能只對齊更新次數。

<details id="fixed-moe-recipe" markdown="1">
<summary>現代零件與效率的固定配方；T.10教師需先完成其中的MoE配方</summary>

按[T.1](training.md#T.1)取得TinyStories包後，可選`--experiment modern`或`--experiment moe`重做固定比較，零件分別見[14.2](chapters/14.md#14.2)與[15.4](chapters/15.md#15.4)。兩組的設定、結果位置與作者紀錄如下：

| 配方 | 設定 | 結果在哪裡 | 作者紀錄（Modal 上的 NVIDIA L4） |
| --- | --- | --- | --- |
| `modern` | TinyStories 409 篇訓練、51 篇驗證、52 篇測試；特徵寬度 64、兩層、四個注意力頭；每組更新 240 次；基準、RoPE、RMSNorm、ReLU²、SwiGLU 與權重共享六組 | [modern.json](../docs/course-experiments/results/modern.json) | 44.6 秒 |
| `moe` | 同一份 TinyStories 切分，每組更新 180 次；三組不同寬度的 Dense 對照，以及四個專家、選一或選二、輔助代價係數 0 或 0.01 的四組 MoE | [moe.json](../docs/course-experiments/results/moe.json) | 65.7 秒 |

每一組的結果在 `results.variants.<組名>.heldout` 欄：`validation` 是驗證資料、`test` 是測試資料，`nll` 是平均代價。重做命令：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment modern --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment moe --device cuda
```

上述兩行各自重做一組，按需要選擇；T.10的MoE教師使用第二組輸出的`moe/model.pt`及`moe/dataset.json`。要另重做效率與精度比較，先完成T.4的完整`sft`：效率組依賴它的`sft/model.pt`和`dataset.json`，precision組沿用同一對話起點，對照FP32、BF16與FP16的計算精度，意義見[16.7](chapters/16.md#16.7)。`--device cuda`指定相容的NVIDIA GPU，是否支援某精度須看報告，未執行不能算通過：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment efficiency --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment precision --device cuda
```

作者在 L4 上的紀錄：`efficiency` 32.3 秒、`precision` 15.0 秒。這兩組與上面兩組改用 CPU 的可行性都未量測。

### 選讀：效率實驗的量測條件

推論先暖機，再在計時兩側等待 GPU 完成；保存多次量測與中位數。記錄提示長度、生成長度、精度、後端與工作範圍。訓練計時是否含反向傳播和更新，也要寫清楚。

PyTorch已配置記憶體是交給張量的空間，不等於整張GPU用量。保存起點記憶體、過程峰值與新增量；不要只比較新增量便宣稱整張卡節省同樣倍數。詳細操作見[16.1](chapters/16.md#16.1)與[效率實驗紀錄（JSON）](../docs/course-experiments/results/efficiency.json)。

FlashAttention是分塊安排注意力計算、減少中間表儲存與搬移的方法，見[16.9](chapters/16.md#16.9)。PyTorch的SDPA是呼叫這套注意力計算的介面，實際可能選不同後端，見[16.8](chapters/16.md#16.8)。要獨立核驗是否用了Flash實作，執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment flash_probe --device cuda
```

作者在 L4 上的紀錄是 4.5 秒。它只測固定Q/K/V的注意力前向計算與反向傳播；Q用來查詢、K供比較、V是混合的內容數字，見[3.4](chapters/03.md#3.4)，不訓練整個模型。報告的`results.verification_passed=true`只表示`supported_routes`至少有一條已核驗Flash後端的路線，而且這個集合內的路線都已完成；其他路線仍可能失敗。先查看各路線的`status`、後端核驗與數值容差，再以`results.schedule_completed`和`results.status`判讀整次探針是否完成，以及是否有不支援的路線。未執行或未完成時，先看原因，不據此聲稱使用成功。還要核對記憶體範圍，不能由SDPA這個API名稱推定已用Flash；完整條件在 [flash_probe 實驗紀錄（JSON）](../docs/course-experiments/results/flash_probe.json)。

</details>

## T.9 真的縮小保存格式，再量品質

把數字四捨五入後仍存成浮點數，未必能縮小檔案。先從同一份已訓練浮點模型，各自轉換不同位數，再檢查保存大小與相同題目的回答。

下面兩條都從 [T.4](#T.4) 對話路線保存的原始 `checkpoints/attributes.pt` 出發，不把8-bit再轉成4-bit。本節命令在你電腦上的時間未量測。

```bash
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 4 --output checkpoints/attributes-int4.pt
.venv/bin/python scripts/quantize.py checkpoints/attributes.pt --bits 8 --output checkpoints/attributes-int8.pt
```

轉換只改支援的 Linear 權重，其他部分照原樣以浮點數保存。各部分在量化裡的角色如下，概念見[2.3](chapters/02.md#2.3)與[17.7](chapters/17.md#17.7)：

- **Linear 權重**：混合輸入數字的乘數；這是被換成低位元整數碼的部分。
- **embedding 表**：用 ID 查出向量的數字表，保留浮點。
- **正規化層的參數**：調整數字尺度，保留浮點。
- **偏移（bias）**：每個輸出另加的數，保留浮點。
- **刻度（scale）**：轉換時新增的倍數，把存下的整數碼乘回浮點近似值（還原值）。

這些保留部分與刻度都要儲存。本課推論會先把碼換回浮點數再計算，所以檔案變小沒有直接證明執行時記憶體同樣變小或更快。

```bash
.venv/bin/python scripts/infer.py checkpoints/attributes.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py checkpoints/attributes-int4.pt --chat --prompt "color=blue;shape=circle;pitch=low;color?" --tokens 24 --device cpu --json
```

希望答案是`blue`。這一題先讓你查看前後有沒有改變；完整能力檢查仍用同一份驗證題。原訓練檔可能含最佳化器狀態與隨機狀態，不能把訓練狀態的差異也算作量化收益。

本機練習先比較純模型張量 bytes，不要求另做原版部署匯出。用任何純文字編輯器把下段存成 `outputs/size_check.py`，再用 `.venv/bin/python outputs/size_check.py` 執行，應印出三行，每個檔案一行。它只讀剛才的三個檔案：`model` 保存模型數字，最佳化器與其他訓練狀態在它之外。每份模型張量的格數乘每格 bytes，再加總；量化版的低位元碼、刻度與保留的浮點張量都計入。

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

`weights_only=True` 讓 `torch.load` 只讀出張量、數字、文字這類資料，不執行檔案裡可能夾帶的其他 Python 物件，讀別人給的檔案時比較安全。`numel()`數張量格數，`element_size()`給每格bytes。這裡按模型狀態表的每個項目加總，不計最佳化器或檔案包裝；也不是執行時記憶體或速度量測。若改比較完整部署檔大小，三版就須使用同用途、同樣不含訓練狀態的匯出方式。

接著使用T.4的同一份屬性validation題組，評估三版；每題新增上限都是24個 token，`all`選全檔：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-fp32-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int4.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int4-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int8.pt --data data/generated/attributes-sft/validation.jsonl --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int8-validation.json
```

在三份JSON中用相同`row`找回同一筆提問與目標，再並排`samples`裡的生成、原始ID與停止原因，欄位見[T.4](training.md#T.4)。逐題判內容匹配及正常結束，記「新增答對、新增答錯與不變」，相同總答對數可能是不同題目答對。先用validation選位數；設定選完，才用下面三條命令評估測試題。它們和上面三條只差資料路徑改成 `test.jsonl`、加上 `--split-label test`，以及另取輸出檔名；測試題不拿來重新選設定：

```bash
.venv/bin/python scripts/evaluate.py checkpoints/attributes.pt --data data/generated/attributes-sft/test.jsonl --split-label test --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-fp32-test.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int4.pt --data data/generated/attributes-sft/test.jsonl --split-label test --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int4-test.json
.venv/bin/python scripts/evaluate.py checkpoints/attributes-int8.pt --data data/generated/attributes-sft/test.jsonl --split-label test --mode sft --tokens 24 --limit all --device cpu --output outputs/attributes-int8-test.json
```

練習把三版模型張量大小與逐題變化放在同一張表，分別說明「保存變小了嗎」與「相同題目是否維持回答」。尚未量到的執行記憶體與速度就留未量測。

<details>
<summary>固定量化與量化感知訓練</summary>

固定比較先完成T.4的完整`sft`，再跑：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment quantization --device cpu
```

作者的紀錄是在 Modal 的 NVIDIA L4 上用 `--device cuda` 跑的，7.8 秒；照上面用 CPU 跑的時間未量測。這個配方另做120次更新才轉換，和上面的單獨 `quantize.py` 不同；不能互換成績。它保存浮點、4-bit、8-bit與共同題目，格式見[量化實驗紀錄（JSON）](../docs/course-experiments/results/quantization.json)。[17.14](chapters/17.md#17.14)介紹QAT：讓訓練先適應模擬的量化誤差；真正的低位元 kernel（計算程式）仍需格式與硬體支援。

</details>

## T.10 用較小學生學教師，再與普通訓練比較

先選較小的學生架構，再比較它只學正確答案、與另加教師機率的差別。兩支學生要共用起點、資料與更新安排；否則看到差異，也難知道是不是教師訊號帶來的。

這套固定配方使用三種已訓練完成的教師：屬性教師、風格教師與 MoE 教師。此路線必須先打開並完成[T.4的固定對話配方](training.md#fixed-sft-recipe)、[T.5的固定風格配方](training.md#fixed-style-recipe)及[T.8的固定MoE配方](training.md#fixed-moe-recipe)中的`sft`、`style`、`moe`三組，不能用前面小練習的`checkpoints/attributes.pt`、`style.pt`或`moe.pt`替代。

這三組前置配方都寫 `--device cuda`。沒有 NVIDIA GPU（例如用 Mac）時，工具不會自動改用 CPU，指定 `cuda` 會直接停下；可以把它們改成 `--device cpu`，但 CPU 可行性未量測：能不能跑完、要多久都沒有紀錄。這個工具只接受 `cpu` 與 `cuda`，不支援 Apple 的 MPS。作者在 Modal 的 NVIDIA L4 上的紀錄是 `sft` 31.7 秒、`style` 29.5 秒、`moe` 65.7 秒。

三組命令沿用預設輸出根目錄`outputs/course-experiments/course-v1/`。核對下列六個檔案都已產生：每份`model.pt`是該教師，配對的`dataset.json`保存它原本的訓練、驗證與測試切分，學生沿用這些切分比較：

| 教師 | 模型檔 | 配對資料切分 |
| --- | --- | --- |
| 屬性教師 | `sft/model.pt` | `sft/dataset.json` |
| 風格教師 | `style/model.pt` | `style/dataset.json` |
| MoE 教師 | `moe/model.pt` | `moe/dataset.json` |

保留同一組的模型與資料，不用只含推論權重的下載包替代完整教師產物，再執行：

```bash
.venv/bin/python scripts/fetch_training_assets.py --asset gsm8k
.venv/bin/python -m scripts.course_experiments.run --experiment distillation --device cpu
```

蒸餾命令寫的是 `--device cpu`。它要訓練多支學生，涵蓋不同任務、學生尺寸與教師訊號，CPU 時間未量測；作者的紀錄是在 Modal 的 NVIDIA L4 上用 `--device cuda` 跑的，90.1 秒。有相容環境可改成 `--device cuda`。產物在 `distillation/`，不同任務、學生尺寸與教師訊號各有檔名。GSM8K 是一份英文小學數學應用題資料集；這裡只拿它看學生遇到訓練範圍外的題型時表現如何，也就是有限的域外診斷，並不是這份屬性模型已具備解應用題的證據。

先選同寬度的兩支，用相同題目檢查。下方檔名的`w32`表示學生的特徵寬度是 32；`ce`只用正確答案的交叉熵（cross entropy，見[1.8](chapters/01.md#1.8)）訓練，`ce_kl`同時學正確答案與教師在相同位置的候選機率。KL在此衡量教師、學生兩份機率分佈的差異，方法見[18.8](chapters/18.md#18.8)與[18.9](chapters/18.md#18.9)：

```bash
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce_kl.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
```

希望答案是 `square,high`。有的題改善，也可能另有題退步；核對整份相同題目，並一起記學生參數、教師成本與有效目標數。只變小是架構的效果，教師訊號是否有幫助由這個對照回答。

教師和學生還要字表對齊：同一位置、同一份字表的候選，才可以直接比較機率。多模態學生還可能減少圖像位置，做法另見[18.13](chapters/18.md#18.13)。蒸餾後再量化是另一個改動，要分開驗收。

<details>
<summary>原始資料、教師與多模態路線</summary>

完整教師、學生、對照與生成見[蒸餾實驗紀錄（JSON）](../docs/course-experiments/results/distillation.json)，不同任務有各自分母，不用單一數字代表全部能力。

多模態蒸餾使用固定 `vqa` 與 `joint` 兩份教師，以及各自的原始切分。先完成[T.4](training.md#T.4)的固定 `sft`，再按[T.6](training.md#T.6)的固定配方完成 `encoders`、`projector`、`vqa`，接著在相容的 CUDA 環境依序執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment joint --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment multimodal_distillation --device cuda
```

作者在 Modal 的 NVIDIA L4 上的紀錄：`joint` 33.4 秒、`multimodal_distillation` 53.0 秒；改用 CPU 的可行性未量測，而且 `joint` 屬於模態實驗，CPU 結果只會被標成通路檢查。第一行是[12.12](chapters/12.md#12.12)的固定圖音聯合訓練，直接讀取 `sft` 與 `encoders`，不承接 `vqa`。這份 `joint`、T.6 手動保存的 `checkpoints/joint.pt` 和合成小世界整合實驗的 `capstone_joint` 各有自己的配方，不能互換來源。實驗工具每次只執行指定項，不自動補齊依賴；上述命令沿用同一個預設 `outputs/course-experiments/course-v1/` 根目錄，第二行會讀取 `vqa/` 與 `joint/` 各自的 `model.pt`、`dataset.json`。先保留來源權重和原始切分；僅下載推論包不足以恢復訓練資料。操作與格式見[實驗說明](../docs/course-experiments/README.md)。

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

比較時也留原始逐題結果。兩版都4/5，可能錯在不同題；一版先丟掉難題，也不能和另一版的全檔比例直接比較。驗證題可以用來選設定，測試題留到選完後才開啟。

紀錄做到能回答「改了什麼、哪個條件固定、結果支持哪件事」，就能完成一次有限比較。文字、圖解或排版修訂本身不需要重跑訓練；模型、資料或能力宣稱改變時，再做相應驗收。

<details>
<summary>共同成品、成熟模型與完整歷史紀錄</summary>

[第19章](chapters/19.md)追蹤我們自行訓練的 MoE 助理，以及各階段載入的父檢查點（定義見[T.4](training.md#T.4)），階段接續見[19.4](chapters/19.md#19.4)。它的 v2 版已完成訓練、公開推論權重，並在固定的測試資料上評估；[19.12](chapters/19.md#19.12)列出能力及未達標項目，[公開CPU操作](../docs/selftrained/v2-public-cpu-commands.md)提供試用，[本機訓練指引](../docs/selftrained/TRAINING.md)用Git LFS保存的固定資料逐段接自己的訓練檢查點，不需要作者的私有雲端儲存空間。其結果與局部實驗分開。[第20章](chapters/20.md)是沿用成熟Qwen／Whisper的延伸，專用操作見[重訓指引](../docs/natural-assistant/v4/TRAINING.md)，上游能力不算成我們從零學會。

第 19 章另外從隨機參數訓練了 Looped 解題器與機率判斷器兩個成品；用途見[19.13](chapters/19.md#19.13)，實測成績與限制見[19.20](chapters/19.md#19.20)，下載、命令與本機重訓入口見[產品操作](../docs/lesson-products/README.md)。能跑通命令，不代表所有任務都可靠。

完整歷史結果在[實驗紀錄](../docs/course-experiments/README.md)。RAG、工具與推理的實驗紀錄（JSON）保存各自的原始輸入、輸出及分母。工具要留實際執行結果，以及把結果放回對話後模型生成的回答；只生成請求文字，還沒有完成操作。

[工具實驗紀錄（JSON）](../docs/course-experiments/results/tools.json)的每一步，`generation.input_ids` 才是模型當時真正收到的輸入；同一步的 `generation.messages` 有些混入了後來才產生的訊息，不能當成當時的輸入。逐步核對時請對照[工具紀錄勘誤](../docs/course-repair-20261008/tool-trace-errata.md)，它附有重建後的當時文字。

</details>
