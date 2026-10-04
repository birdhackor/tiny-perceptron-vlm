## T.3 先訓練接字表，再比較固定窗口

先備是[1.5的相鄰字統計](chapters/01.md#1.5)、[2.2的固定窗口](chapters/02.md#2.2)，再讀[1.8的猜錯代價](chapters/01.md#1.8)。bigram 模型只看前一個字；MLP 是 multilayer perceptron（多層感知器），把幾個前文位置的特徵一起混合，混合方式見[2.3](chapters/02.md#2.3)。此處訓練調整模型的可調數字，不是直接把觀察次數加進 1.5 的計數表。

每個位置取正確答案的機率 `p`，以 `-log(p)` 當代價，平均後叫 loss；給正確答案的機率越高，代價越低。它不是猜錯題數或答對率。訓練代價來自教過的文件；留出（held-out）代價來自沒有用來更新數字的文件。本節用 validation 這份留出集合調整設定，test 留到最後檢查。

先看工具究竟拿哪些文件。沒有 `--data` 時，它使用內建的 12 篇規則短文：四種顏色配三種形狀，例如「顏色=紅；形狀=圓。」。下面程式可從根目錄的 Python 執行，列出各側的完整內容：

```python
from tiny_perceptron.data import split_documents, toy_documents

split = split_documents(toy_documents(), seed=42)
for name, documents in split.items():
    print(name, len(documents), documents)
```

seed 42 會分出 train／validation／test 共 9／1／2 篇，validation 是「顏色=紅；形狀=圓。」。種子（`--seed`）決定隨機起點與文件切分；比較時保持資料內容、文件順序、程式版本和種子相同。工具先去掉完全重複的文件、按整篇切分，再取窗口，避免同一句相鄰窗口落入兩側。字表只看 train；留出文件中未見過的字會用未知字 ID。

自己的資料可用 `--data 路徑` 指向 JSONL，每行的 `text` 是一篇完整文件，例如：

```jsonl
{"text":"顏色=紅；形狀=圓。"}
{"text":"顏色=藍；形狀=方。"}
{"text":"顏色=綠；形狀=三角。"}
```

至少要有三篇不同文件才能分三側；這三行只示範格式。正式比較時前後兩條命令都加相同的 `--data`，不要把一份已切好的 train 檔交進來後誤以為工具不會再切分。

先在 CPU 檢查通路，並把單行 JSON 報告保存成開始基準：

```bash
mkdir -p outputs checkpoints
.venv/bin/python scripts/train_simple.py --model bigram --seed 42 --device cpu > outputs/bigram-before.json
.venv/bin/python scripts/train_simple.py --model mlp --context 3 --width 16 --seed 42 --device cpu > outputs/mlp-before.json
```

`--model` 選模型；`--context 3` 讓 MLP 看前三個字的位置，`--width 16` 讓每個字的特徵有 16 個數字。bigram 固定看一個字。沒有 `--train` 時只算一次前向與反向，不更新、不保存模型；重導向 `>` 保存的是報告。預設資料、seed 42 的 bigram CPU 報告如下，小數尾數可能依環境不同：

```json
{"mode":"dry-run-no-weight-update","model":"bigram","train_loss":3.773265838623047,"validation_loss":3.6166253089904785,"split_unit":"whole deduplicated document","parameters":289}
```

| 欄位 | 如何解讀與核對 |
| --- | --- |
| `mode` | 開始基準應是 `dry-run-no-weight-update`，表示還沒有學習 |
| `train_loss` | 訓練文件中下一字目標的平均代價；須為有限數字 |
| `validation_loss` | 同一份留出文件中下一字目標的平均代價；須為有限數字，較低表示這組新題的正確答案獲得較高機率 |
| `split_unit` | `whole deduplicated document` 表示先按去重後的整篇文件切分 |
| `parameters` | 模型可調數字的總數；預設 bigram 是 289，這個 MLP 是 1345 |

MLP 的開始報告約為 `train_loss=3.0041`、`validation_loss=3.0252`；這些都是未訓練模型的數字，不能當作訓練成效。成功條件是模式與模型正確、兩個代價有限、切分單位正確，並知道資料來源。

練習先只選 bigram。先寫下預測：更新通常會降低訓練代價，但留出代價可能降低，也可能升高；後者表示教過的文件改善未必延伸到新題。準備正式更新時才執行：

```bash
.venv/bin/python scripts/train_simple.py --model bigram --seed 42 --device cpu --train --steps 200 --output checkpoints/bigram.pt > outputs/bigram-after.json
```

`--train` 明確開啟更新，`--steps 200` 是更新次數，`--output` 指定 `.pt` 模型檔。結束報告的 `train_loss` 與 `validation_loss` 都用最後一次更新完成後的模型重新計算，才能在同一個時點比較。另一欄 `last_batch_loss_before_update` 保留最後一步更新之前算出的代價，別把它和更新後的 `train_loss` 混為一個數字。200步是這次實驗的設定，仍不保證未教過的題目會改善。

更新完成後，將下面程式存為 `outputs/compare_bigram.py`，執行 `.venv/bin/python outputs/compare_bigram.py`，它會讀真正的前後報告並保存比較：

```python
import json
import math
from pathlib import Path

before = json.loads(Path("outputs/bigram-before.json").read_text())
after = json.loads(Path("outputs/bigram-after.json").read_text())
assert before["mode"] == "dry-run-no-weight-update" and after["mode"] == "train"
for key in ("model", "split_unit", "parameters"):
    assert before[key] == after[key]
comparison = {}
for key in ("train_loss", "validation_loss"):
    assert math.isfinite(before[key]) and math.isfinite(after[key])
    comparison[key] = {
        "before": before[key],
        "after": after[key],
        "change": after[key] - before[key],
    }
text = json.dumps(comparison, ensure_ascii=False, indent=2)
Path("outputs/bigram-comparison.json").write_text(text + "\n")
print(text)
```

`change` 小於零代表代價降低。通過練習的條件是能指出同一份留出集合的平均代價如何變化，並把它與訓練代價分開說；平均改善不表示每篇文件都改善。這個 `train_simple.py` 命令印的是平均代價，不會印生成文字。它保存的是接字表或固定窗口MLP的數字與字表，和後面Transformer的結構不同，不能直接交給 `infer.py`；下面的完整實驗入口會替這兩種小模型產生句子。

完成 bigram 後，才用同一資料與種子執行 MLP 的 `--train` 命令：

```bash
.venv/bin/python scripts/train_simple.py --model mlp --context 3 --width 16 --seed 42 --device cpu --train --steps 200 --output checkpoints/mlp.pt > outputs/mlp-after.json
```

複製上面的比較程式，另存為 `outputs/compare_mlp.py`。程式內三個路徑也要改：讀取起點的 `outputs/bigram-before.json` 改成 `outputs/mlp-before.json`，讀取終點的 `outputs/bigram-after.json` 改成 `outputs/mlp-after.json`，寫出結果的 `outputs/bigram-comparison.json` 改成 `outputs/mlp-comparison.json`。只改程式檔名不會改它讀取或寫入的資料。保存後執行 `.venv/bin/python outputs/compare_mlp.py`，核對model是mlp、前後參數格數一致，再讀取它印出的代價與change。

現在把這些步驟接起來看真實結果。我們在CPU上，用相同的文件切分與種子42，實際各更新200次；MLP的特徵寬度都為16。四組設定是在執行前固定的，完成後才並排查看另外兩篇test，沒有再用test挑步數。下表每一格都是「訓練前 → 最後一次更新後」的平均下一字代價。

| 模型 | 參數格數 | 訓練：9篇 | 驗證：1篇 | 最後檢查：2篇 |
| --- | ---: | ---: | ---: | ---: |
| 接字表，只看1字 | 289 | 3.77327 → 1.05676 | 3.61663 → 1.07435 | 3.59724 → 1.08973 |
| MLP，看1字 | 833 | 2.97138 → 0.33352 | 2.94521 → 0.43103 | 3.04028 → 0.43627 |
| MLP，看3字 | 1,345 | 3.00414 → 0.21302 | 3.02520 → 0.30577 | 3.02634 → 0.31105 |
| MLP，看5字 | 1,857 | 2.83350 → 0.19883 | 2.78032 → 0.51252 | 2.82708 → 0.61287 |

這裡平均的是每一個下一字目標的代價，不是每篇先算平均再把篇數除一遍。9篇訓練短文共103個目標、1篇驗證有11個、2篇test有22個，都包括每篇最後的結束標記。三側的分母不同，但四個模型在同一側用的題目與分母相同，所以可以對照；1篇與2篇仍是很小的測驗。

四種模型的練習代價都下降了，說明它們確實調整了猜法。然而5字MLP的練習代價最低，驗證與test卻比3字高。這正是我們保留新題的理由：不能只挑最會做練習題的模型。窗口變長也讓MLP的參數變多；接字表換成MLP則連結構都改了，所以不能把整張表的差別都歸因於窗口長短。

再看它們自己寫出的文字。相同開頭`顏色=`，接字表每次選最高機率的下一字，生成`三角。`；3字MLP則生成`藍；形狀=三角。`。前者連顏色與形狀欄位都混在一起，後者至少保留了這個短文格式。這不是問答正確率：這個開頭沒有指定必須是哪種顏色或形狀，能寫出一篇合格式的短文也不等於理解所有未見組合。平均逐字代價與自由生成要各自看，不能拿其中一項替另一項作保證。

想一次重做表裡四組及生成例子，在根目錄執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment simple_models --device cpu
```

終端會印出完成摘要；詳細結果保存於 `outputs/course-experiments/course-v1/simple_models/result.json`，`results.runs` 下的 `before_nll` 與 `after_nll_same_post_update_time` 對應表裡前後代價，`samples` 保存完整開頭與新生成文字。同一目錄的 `bigram.pt`、`mlp1.pt`、`mlp3.pt`、`mlp5.pt` 是這四種簡單模型的存檔，不能當成TinyLM權重混用。[課程實測原始報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/simple_models.json)還保留版本、資料指紋與各次紀錄，方便核對這些數字來自哪一次執行。

練習先遮住表的驗證與test欄，只按訓練代價選一組，再揭開新題欄：你會選5字MLP，但這一次它在兩組新題的平均代價都比3字高。用自己的話說出為什麼不能只追練習成績，再回看[2.5的圖](chapters/02.md#2.5)。若換資料、種子或步數，需重新保存前後報告，不能沿用這張表當作你的成績。

也可以先拿課程已訓練好的模型來試，不必先等一次訓練跑完。權重檔保存的是模型學到的數字；要用相同的模型結構與字表載入，才能把它變成可做預測的程式。接字表的存檔交給 `infer_simple.py`，Transformer 的存檔交給 `infer.py`；兩種檔案不能因為都叫 `.pt` 就互換。

[公開模型首頁](https://huggingface.co/birdhackor/tiny-perceptron-course-models)提供30組實驗的120份存檔。同一組可能包含不同尺寸、教師與學生或多個比較版本；每組模型卡會說明任務、成績、授權及使用方式。先挑眼前章節需要的一組，依卡片準備匹配的字表、輸入與推論程式即可。下面的`--list`能列出目前已公開的組別；完整固定版本與檔案指紋也可查[下載清單](../docs/course-experiments/public-models.json)。

下面先用已公開的文字 Transformer `text_foundation`。先列出可下載的實驗，再只取這一組：

```bash
.venv/bin/python scripts/fetch_course_models.py --list
.venv/bin/python scripts/fetch_course_models.py --model text_foundation
.venv/bin/python scripts/infer.py checkpoints/course/text_foundation/model.pt --prompt "color=blue;shape=circle;" --tokens 32 --device cpu --json
```

`--list` 只列清單；第二行從公開 Hugging Face 下載固定版本，核對每個檔案的內容指紋，放到 `checkpoints/course/text_foundation/`。第三行才載入模型並接寫文字，`--tokens 32` 是最多新生成32個小單位，遇到結束標記可提早停下。這個模型學的是英文規則短文，和上面中文接字表的資料不同；它不是一般聊天助手。

我們用公開檔案在 CPU 實際取得 `answer="side=right."`、`eos=true`。`answer` 是新接出的文字，`eos` 表示模型自己產生結束標記；完整輸出也保留 `generated_ids` 與非法特殊標記檢查。題目只給顏色和形狀，沒有指定左右，因此這一例只能確認能載入、生成並結束，不能算成「答對一道左右問答」。

若要連下載內容、每份權重的重建與 CPU 執行一併核對，可執行：

```bash
.venv/bin/python scripts/check_course_models.py --model text_foundation
```

工具會保存一份含指令、版本、檔案指紋與實際輸出的紀錄到 `docs/course-experiments/student-checks/`。這是操作檢查，不代替留出測驗。公開學生包只保留推論需要的數字，沒有更新器和隨機狀態；可以拿來開始另一次微調，但不能用它逐步重現被中斷的原訓練。要延續同一次訓練，請看[5.7的完整存檔與續訓](chapters/05.md#5.7)。

