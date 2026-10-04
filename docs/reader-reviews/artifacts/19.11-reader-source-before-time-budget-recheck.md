## 19.11 下載一份權重，就能從中斷處繼續訓練嗎？

把助理交給另一位學生時，有兩種需求：一位想直接試用，另一位想接著完成訓練。第一位需要模型權重與正確的結構、tokenizer；第二位還需要更新器狀態、隨機狀態、抽樣位置與訓練階段。只帶走期末作業，和帶走能繼續作業的完整工作桌，資料量與用途不同。

前置是[checkpoint與resume](05.md#5.7)、[tokenizer與模型的配套](06.md#6.5)、[本章父階段](19.md#19.4)與[量化、Dense學生兩條壓縮路線](19.md#19.10)。FP32成品使用`capstone-v1`格式，量化儲存版使用`capstone-ptq-v1`；檔案另外標明資料版本與階段。公開推論檔可以去掉更新器等狀態，方便下載；去掉之後就不能稱能完全恢復原訓練軌跡。

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_perceptron.capstone import CapstoneModel, load_capstone, save_capstone

model = CapstoneModel()
with TemporaryDirectory() as directory:
    path = Path(directory) / "example.pt"
    save_capstone(path, model, stage="sft", step=0, inference_only=True)
    loaded, metadata = load_capstone(path)
    print("格式", metadata["format_version"], "訓練步數", metadata["step"])
    print("只供推論", metadata["inference_only"], "參數相同", model.description() == loaded.description())
```

這段把隨機模型存到暫存資料夾，再真的讀回，結束後暫存夾自動清除。step=0明確表示沒有完成SFT；`stage="sft"`是格式標籤，不能拿來冒充已訓成果。輸出應為capstone-v1、0、True、True；最後一個True只確認架構與參數數量描述一致，完整數值與逐題輸出相等還要另檢。

正式交付把小型合成資料與來源規格保留在本專案，權重放[Hugging Face公開模型庫](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2)。本輪已逐檔匿名下載、核對指紋並讀回全部11份模型，正式[公開清單](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-public.json)固定在revision `33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed`。revision像這一版包裹的封條編號；SHA-256則是每個檔案的內容指紋，不只檢查檔名相同。下載不需要登入或提供token。

先照[19.1](19.md#19.1)準備Python環境，在專案資料夾執行：

```bash
python scripts/fetch_capstone.py --list
python scripts/fetch_capstone.py --stage joint
```

`--list`只印出11份清單，不下載權重，也不訓練。每一行的`stage`是下一行`--stage`可用的名字；`joint`是驗證後選出的推薦成品。其他名稱可這樣讀：

| 可以下載的名字 | 用途 |
| --- | --- |
| `pretrain`、`sft`、`joint`、`dpo` | 四站FP32推論檔，用來比較加入一項訓練後的變化 |
| `joint-int4`、`joint-int8` | 推薦joint的4／8-bit儲存版 |
| `dpo-int4`、`dpo-int8` | 偏好分支的4／8-bit儲存版 |
| `student-ce`、`student-kd`、`student-kd-int4` | [19.10](19.md#19.10)較小Dense學生的答案訓練、蒸餾與蒸餾後量化比較 |

FP32指權重以32-bit浮點數儲存；`int4`與`int8`指這裡的壓縮儲存格式，載入後仍還原成FP32計算，並不保證較少執行記憶體或更快。Dense學生的能力也不能直接當成joint能力。每個下載資料夾都包含`model.pt`、模型卡`README.md`、`LICENSE`、`THIRD_PARTY_NOTICES.md`與這次合成資料`data.json`；設定與tokenizer另存在模型檔內。各份用途、父檔來源與失敗案例要連同模型卡閱讀。

第二行會安裝到`checkpoints/capstone/joint/`。其中公開的`model.pt`是1,327,950 bytes，SHA-256為`d85cca83cdb4952f4653ef6b58d94562b41403db3a9db2ea31ff57e31246ca16`。這是移除訓練內部狀態與整理交付資訊後的正式檔案，不能直接拿[19.10](19.md#19.10)實驗當時的檔案bytes當它的大小；權重數值保留相同，封裝資訊有變。下載程式會先核對全部五個檔案的大小、SHA與模型身分，成功後才把完整資料夾放到目的地。若joint已在，就直接使用它；想另留一份，可以改用`python scripts/fetch_capstone.py --stage joint --output checkpoints/capstone-second`，後續模型路徑也改成`checkpoints/capstone-second/joint/model.pt`。

量化版可以另外下載，並用相同的CPU命令讀回：

```bash
python scripts/fetch_capstone.py --stage joint-int4
python scripts/capstone.py infer --checkpoint checkpoints/capstone/joint-int4/model.pt --prompt "1+2等於多少？" --device cpu
```

這份公開joint-int4檔是270,855 bytes；下載清單保留它自己的SHA。命令會選正確載入方式，再跑同一工具迴圈。若換成`student-kd-int4`，要把下載名稱與路徑一起換；成功下載、成功載入，仍不代表答案正確。例如本輪這位學生把1+2請求錯寫成2+9，工具回11後自己答12。我們保留這類失敗，不把「程式能跑」算成「模型會答」。本機網頁的`serve`目前只接受FP32格式，所以仍使用19.1的`joint/model.pt`；不要把`joint-int4/model.pt`交給它。

中斷續訓使用另外的`model-training.pt`與原有資料版本。DPO還需儲存reference狀態；只重新建立一份會隨當前policy改變的reference，並不等於接著原實驗跑。是否能逐步重現，也受裝置與運算實現影響，因此報告應區分同環境恢復測試與跨環境重新訓練。

如果要自己重做四站，下面才是會更新權重的命令；它們不是上方試用的必要步驟，也不會在本節短程式中自動執行。`--batch-size 24`表示每次更新取24筆訓練資料一起計算誤差；`--seed 42`用同一個種子數字設定新實驗的隨機初值與取樣起點，方便在相同條件下比較實驗。先用自己的資料版本與程式產生pretrain，再把每站真正完成的原始`model.pt`交給下一站。請一行一行執行，先讀該站的`train-report.json`，確認`schedule_completed`為`true`，才往下走：

```bash
python scripts/capstone.py train --stage pretrain --output checkpoints/my-capstone/pretrain --steps 300 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage sft --input-checkpoint checkpoints/my-capstone/pretrain/model.pt --output checkpoints/my-capstone/sft --steps 1400 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage joint --input-checkpoint checkpoints/my-capstone/sft/model.pt --output checkpoints/my-capstone/joint --steps 600 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage dpo --input-checkpoint checkpoints/my-capstone/joint/model.pt --output checkpoints/my-capstone/dpo --steps 100 --seed 42 --batch-size 24 --device cpu
```

這四行示範本課原配方的階段關係，並沒有另跑一次CPU訓練來宣稱重現正式GPU分數；各站實報在[19.4](19.md#19.4)。DPO是要觀察的比較分支，不是成品必須接受的最後更新：本輪推薦的是joint。命令每次預設最多訓練540秒，電腦較慢時可能只完成部分步數。到了時間上限仍會留下工作檔；例如自己那站joint原本排定600步、中途停下，可以用：

```bash
python scripts/capstone.py train --stage joint --resume checkpoints/my-capstone/joint/model-training.pt --output checkpoints/my-capstone/joint --steps 600 --seed 42 --batch-size 24 --device cpu
```

這裡的600是原本總步數，不是追加600步。`--resume`要求同一站、原定步數、batch大小、資料指紋與程式指紋一致，並讀回更新器、隨機狀態和進度；取樣接著工作檔的位置走，不會因為再次寫seed 42就從第一筆重來。DPO還要原reference。若尚未完成，先繼續同一站，不要以不完整父檔開始下一站。跨站的`--input-checkpoint`則以已完成父模型開始一份新的更新器與抽樣流程，兩者用途不同。實際檢查條件可見[訓練器原始碼](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/scripts/course_experiments/capstone.py)。

公開推論檔沒有上述完整工作狀態，也沒有本課串階段命令所需的原始`data_manifest`與完成排程證明，因此不能把公開`checkpoints/capstone/joint/model.pt`代入這裡的`--resume`或正式`--input-checkpoint`。推論權重本身仍可作為另寫訓練程式的初始數值；那是建立一個新實驗，並不是恢復本課已驗證的訓練。公開交付的是可試用、可比較的11份推論版本，不包含作者的完整續訓工作檔。

練習把step改成1，觀察metadata確實顯示1。但模型仍沒有更新過，這說明格式欄位可以寫入，卻不能自行證明訓練發生；正式報告還需有效目標、權重變更與執行記錄。

