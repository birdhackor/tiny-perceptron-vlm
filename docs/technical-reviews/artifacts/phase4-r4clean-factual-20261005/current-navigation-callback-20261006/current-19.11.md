## 19.11 下載一份權重，就能從中斷處繼續訓練嗎？

另一位同學拿到權重，想做兩件不同的事：直接回答，或接著中斷的訓練。推論需要權重、結構、tokenizer、模態處理與工具契約；續訓還需要更新器、隨機狀態、取樣進度和原階段。下載一個檔名叫 model.pt 的包裹，不足以知道是哪一種。

先用未訓練的隨機模型，示範推論包裹的存檔與讀回：

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

這段把隨機模型真的存下、載回，step=0 表示沒有訓練。`stage="sft"` 只是一個格式欄位，不能用它證明學過SFT。`description` 相同只確認結構與參數數量；數值和任務輸出還需另查。

共同成品的交付應讓所有神經元件的來源可追溯：文字、router／experts、影像、讀字、語音和投影都列入，不能只交核心再暗中依賴一個成熟入口。資料版本、適用字表、意圖範圍與已知失敗也隨包裹帶走。若只有推論權重，就明確標為推論，不宣稱能恢復原取樣軌跡。

下方提供的是既有合成任務的發布與續訓說明，新成品的操作與權重待實作完成後另回填。讀者可以重用它學檔案角色；不能把舊下載成功當成新任務完成。

<details>
<summary>補充：舊合成任務的11份推論檔與續訓配方</summary>

正式交付把小型合成資料與來源規格保留在本專案，權重放[Hugging Face公開模型庫](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2)。本輪已逐檔匿名下載、核對指紋並讀回全部11份模型，正式[公開清單](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-public.json)固定在revision `33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed`。revision像這一版包裹的封條編號；SHA-256則是每個檔案的內容指紋，不只檢查檔名相同。下載不需要登入或提供token。

先照[19.1](19.md#19.1)準備Python環境，在專案資料夾執行：

```bash
python scripts/fetch_capstone.py --list
python scripts/fetch_capstone.py --stage joint
```

`--list`只印出11份清單，不下載權重，也不訓練。每一行的`stage`是下一行`--stage`可用的名字；`joint`是驗證後選出的推薦成品。其他名稱可這樣讀：

| 可以下載的名字 | 用途 |
| --- | --- |
| `pretrain`、`sft`、`joint`、`dpo` | 四份階段／分支的FP32推論檔，用來比較加入一項訓練後的變化 |
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

中斷續訓使用另外的`model-training.pt`與原有資料版本。DPO 的 policy 是正在更新的模型，reference 是固定的比較基準（見[19.8](19.md#19.8)）。續訓還需儲存原 reference 狀態；只重新建立一份會隨當前policy改變的reference，並不等於接著原實驗跑。是否能逐步重現，也受裝置與運算實現影響，因此報告應區分同環境恢復測試與跨環境重新訓練。

如果要自己重做四份階段／分支模型，下面才是會更新權重的命令；它們不是上方試用的必要步驟，也不會在本節短程式中自動執行。`--batch-size 24`表示每次更新取24筆訓練資料一起計算誤差；`--seed 42`用同一個種子數字設定新實驗的隨機初值與取樣起點，方便在相同條件下比較實驗。先用自己的資料版本與程式產生pretrain，再把每站真正完成的原始`model.pt`交給下一站。請一行一行執行，先讀該站的`train-report.json`，確認`schedule_completed`為`true`，才往下走：

```bash
python scripts/capstone.py train --stage pretrain --output checkpoints/my-capstone/pretrain --steps 300 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage sft --input-checkpoint checkpoints/my-capstone/pretrain/model.pt --output checkpoints/my-capstone/sft --steps 1400 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage joint --input-checkpoint checkpoints/my-capstone/sft/model.pt --output checkpoints/my-capstone/joint --steps 600 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage dpo --input-checkpoint checkpoints/my-capstone/joint/model.pt --output checkpoints/my-capstone/dpo --steps 100 --seed 42 --batch-size 24 --device cpu
```

這四行示範本課原配方的階段關係，並沒有另跑一次CPU訓練來宣稱重現正式GPU分數；各站實報在[19.4](19.md#19.4)。DPO是要觀察的比較分支，不是成品必須接受的最後更新：本輪推薦的是joint。命令每次預設以540秒作為訓練循環的時間預算，程式在每批更新前檢查時間；最後一批更新與存檔可能超過預算，之後的匯出、驗證與整理報告也另外花時間。電腦較慢時可能只完成部分步數，訓練循環停止時仍會留下工作檔；例如自己那站joint原本排定600步、中途停下，可以用：

```bash
python scripts/capstone.py train --stage joint --resume checkpoints/my-capstone/joint/model-training.pt --output checkpoints/my-capstone/joint --steps 600 --seed 42 --batch-size 24 --device cpu
```

這裡的600是原本總步數，不是追加600步。`--resume`要求同一站、原定步數、batch大小、資料指紋與程式指紋一致，並讀回更新器、隨機狀態和進度；取樣接著工作檔的位置走，不會因為再次寫seed 42就從第一筆重來。DPO還要原reference。若尚未完成，先繼續同一站，不要以不完整父檔開始下一站。跨站的`--input-checkpoint`則以已完成父模型開始一份新的更新器與抽樣流程，兩者用途不同。實際檢查條件可見[訓練器原始碼](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/scripts/course_experiments/capstone.py)。

公開推論檔沒有上述完整工作狀態，也沒有本課串階段命令所需的原始`data_manifest`與完成排程證明，因此不能把公開`checkpoints/capstone/joint/model.pt`代入這裡的`--resume`或正式`--input-checkpoint`。推論權重本身仍可作為另寫訓練程式的初始數值；那是建立一個新實驗，並不是恢復本課已驗證的訓練。公開交付的是可試用、可比較的11份推論版本，不包含作者的完整續訓工作檔。

</details>

