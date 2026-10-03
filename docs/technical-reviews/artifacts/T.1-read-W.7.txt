## W.7 檔案放哪裡

你已經能跑一格程式，接下來最常遇到的麻煩是：同一個檔名找不到、昨天能跑今天不能跑、或紅色訊息看不懂。先把專案想成一個工作資料夾，正文是書，Notebook 是工作桌，共用 Python 檔是工具，資料與輸出是材料和成果。分清楚它們放在哪裡，就不用把所有錯誤都猜成套件壞了。開啟與安裝步驟見 [W.1](#W.1)。

| 路徑 | 你會在裡面找到什麼 |
| --- | --- |
| `course/chapters/` | 正文與短程式，依章節編號命名 |
| `notebooks/` | 可操作的工作頁，例如 `01/1.1.ipynb` |
| `tiny_perceptron/` | 共用 Python 零件，章節會按需要介紹 |
| `scripts/` | 環境檢查、資料製作、訓練與評估的入口 |
| `data/` | 本機資料，通常不存入 Git |
| `checkpoints/` | 儲存模型狀態的檔案，通常不存入 Git |
| `outputs/` | 報告、執行結果與網頁等產物 |

相對路徑是從「目前所在資料夾」出發找東西，而不是從電腦固定的位置找。例如 `scripts/check_env.py` 的意思是先進 `scripts` 再找那個檔案；若目前在 `notebooks/`，就會走錯地方。在終端機先 `cd tiny-perceptron-vlm`，是把起點移到有 `pyproject.toml` 的專案根目錄。Notebook 的工作位置也可能不同，故短實驗盡量直接寫小資料，不先要求讀檔。

下面用 Python 內建的 `pathlib` 查起點。`Path` 是表示檔案路徑的工具，`cwd()` 回傳目前資料夾，`exists()` 回答某個路徑是否存在。這段不會建立、修改或刪除任何檔案。

```python
from pathlib import Path

print(Path.cwd())
print(Path("pyproject.toml").exists())
print(Path("tiny_perceptron").exists())
```

第一行輸出依你電腦而不同。在專案根目錄執行時，後兩行應都是 `True`；若都是 `False`，先看第一行，確認是否站在別的資料夾。這比把所有路徑改成電腦上的絕對位置容易重現。若用終端機執行這段，先輸入 `python` 並按 Enter，看到 `>>>` 就進入 Python 互動模式；逐行貼上包含 `from pathlib import Path` 在內的程式，輸出會緊接在後。輸入 `exit()` 並按 Enter 可回到終端機指令模式，才能再用 `cd` 切換資料夾。練習先退出 Python，在專案根目錄輸入 `cd notebooks`，再輸入 `python` 進互動模式重跑這段。先預測 `pyproject.toml` 是否還會被找到；核對後用 `exit()` 退出，再 `cd ..` 回到上一層專案根目錄。只換起點，檔案本身仍存在。

紅色錯誤通常會列出哪個檔案、哪行程式，最後一行說明原因。`ModuleNotFoundError: No module named 'torch'` 表示現在的 Python 找不到工具；先確認 Notebook 選 **Tiny Perceptron**，或終端機已啟用 `.venv`。`FileNotFoundError` 是找不到指定檔案，先查目前起點和檔名。`KeyError` 是字典中沒有那個鍵，可回看 [W.2](#W.2)。`IndexError` 是索引超出範圍。例如 `animals = ["貓", "狗"]` 只有索引 0、1；取 `animals[2]` 是取第三項，就會超出範圍。`SyntaxError` 則先檢查括號、冒號與縮排。

模型訓練的指令碼有些預設只檢查資料與流程。報告中的 `dry-run-no-weight-update` 表示這次沒有更新模型；依[操作配方](training.md)加上 `--train` 才正式訓練。不要只因看到報告檔就以為模型學過了。閱讀錯誤時也採同樣原則：指出程式實際做到了哪一步，再處理阻擋下一步的那個原因。
