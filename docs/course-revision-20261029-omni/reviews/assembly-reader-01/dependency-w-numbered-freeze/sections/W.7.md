## W.7 檔案放哪裡

只在網站上閱讀時用不到這一節。開始在自己的電腦跑程式之後，若遇到同一個檔名找不到、昨天能跑今天不能跑、或紅色訊息看不懂，再回來讀。先把專案想成一個工作資料夾，正文是書，Notebook 是工作頁，共用 Python 檔是工具，資料與輸出是材料和成果。分清楚它們放在哪裡，就不用把所有錯誤都猜成套件壞了。開啟與安裝步驟見 [W.1](#W.1)。

| 路徑 | 你會在裡面找到什麼 |
| --- | --- |
| `course/chapters/` | 正文與短程式，依章節編號命名 |
| `notebooks/` | 可操作的工作頁，例如 `01/1.1.ipynb` |
| `tiny_perceptron/` | 共用 Python 零件；各檔的實作在哪一節講，見表下可展開的對照 |
| `scripts/` | 環境檢查、資料製作、訓練與評估的入口 |
| `data/` | 本機資料，通常不存入 Git |
| `checkpoints/` | 訓練時存下的檢查點（checkpoint）檔，見 [5.7](chapters/05.md#5.7)；通常不存入 Git |
| `outputs/` | 報告、執行結果與網頁等產物 |
| `.venv/` | `uv sync` 建立的工具箱（Python 與套件），不存入 Git |

<details>
<summary>tiny_perceptron 各檔在哪一節講實作</summary>

下表依各檔第一次出現的閱讀順序排列，列出節錄該檔程式、或點名該檔並說明其中零件的小節；只匯入來用的小節不列。

| 檔案 | 在哪一節看到它的實作 |
| --- | --- |
| `tiny_perceptron/data.py` | [1.2](chapters/01.md#1.2)、[4.7](chapters/04.md#4.7)、[5.11](chapters/05.md#5.11)、[6.6](chapters/06.md#6.6)、[7.1](chapters/07.md#7.1)、[7.10](chapters/07.md#7.10)、[7.11](chapters/07.md#7.11) |
| `tiny_perceptron/simple.py` | [2.5](chapters/02.md#2.5) |
| `tiny_perceptron/attention.py` | [3.6](chapters/03.md#3.6)–3.7、[16.5](chapters/16.md#16.5) |
| `tiny_perceptron/modern.py` | [4.4](chapters/04.md#4.4)、[14.1](chapters/14.md#14.1)–14.2、[15.11](chapters/15.md#15.11) |
| `tiny_perceptron/model.py` | [4.5](chapters/04.md#4.5)–4.6、[5.2](chapters/05.md#5.2) |
| `tiny_perceptron/training.py` | [5.6](chapters/05.md#5.6)–5.7 |
| `tiny_perceptron/capstone.py` | [6.6](chapters/06.md#6.6) |
| `tiny_perceptron/tokenization.py` | [6.6](chapters/06.md#6.6) |
| `tiny_perceptron/alignment.py` | [8.8](chapters/08.md#8.8)、[9.10](chapters/09.md#9.10)、[13.3](chapters/13.md#13.3)、[13.5](chapters/13.md#13.5)、[18.8](chapters/18.md#18.8)–18.9 |
| `tiny_perceptron/multimodal.py` | [10.1](chapters/10.md#10.1)–10.2、[10.5](chapters/10.md#10.5)、[10.7](chapters/10.md#10.7)–10.8、[12.1](chapters/12.md#12.1)、[12.6](chapters/12.md#12.6)、[12.8](chapters/12.md#12.8) |
| `tiny_perceptron/natural_concepts.py` | [11.14](chapters/11.md#11.14)–11.16、[12.13](chapters/12.md#12.13)–12.14 |
| `tiny_perceptron/selftrained/` | [12.15](chapters/12.md#12.15)、[19.2](chapters/19.md#19.2)–19.3、[19.5](chapters/19.md#19.5)–19.7、[19.9](chapters/19.md#19.9) |
| `tiny_perceptron/posttraining.py` | [13.10](chapters/13.md#13.10)–13.11、[13.13](chapters/13.md#13.13)–13.15 |
| `tiny_perceptron/lesson_products/` | [14.15](chapters/14.md#14.15)、[19.14](chapters/19.md#19.14)–19.17 |
| `tiny_perceptron/quantization.py` | [17.5](chapters/17.md#17.5)、[17.7](chapters/17.md#17.7)–17.9、[17.14](chapters/17.md#17.14) |
| `tiny_perceptron/retrieval.py` | [A.3](chapters/0A.md#A.3)、[B.2](chapters/0B.md#B.2)、[B.4](chapters/0B.md#B.4) |

</details>

相對路徑是從「目前所在資料夾」出發找東西，而不是從電腦固定的位置找。例如 `scripts/check_env.py` 的意思是先進 `scripts` 再找那個檔案；若目前在 `notebooks/`，就會走錯地方。在終端機先 `cd tiny-perceptron-vlm`，是把起點移到有 `pyproject.toml` 的專案根目錄。Notebook 的工作位置也可能不同，故短實驗盡量直接寫小資料，不先要求讀檔。

下面用 Python 內建的 `pathlib` 查起點。`Path` 是表示檔案路徑的工具，`cwd()` 回傳目前資料夾，`exists()` 回答某個路徑是否存在。這段不會建立、修改或刪除任何檔案。

```python
from pathlib import Path

print(Path.cwd())
print(Path("pyproject.toml").exists())
print(Path("tiny_perceptron").exists())
```

第一行輸出依你電腦而不同。在專案根目錄執行時，後兩行應都是 `True`；若都是 `False`，先看第一行，確認是否站在別的資料夾。這比把所有路徑改成電腦上的絕對位置容易重現。若用終端機執行這段，先在專案根目錄輸入工具箱裡 Python 的路徑並按 Enter：macOS 與 Linux 是 `.venv/bin/python`，Windows PowerShell 是 `.\.venv\Scripts\python.exe`；已照 [W.1](#W.1) 啟用工具箱時，只輸入 `python` 也可以。看到 `>>>` 就進入 Python 互動模式；逐行貼上包含 `from pathlib import Path` 在內的程式，輸出會緊接在後。輸入 `exit()` 並按 Enter 可回到終端機指令模式，才能再用 `cd` 切換資料夾。練習先退出 Python，在專案根目錄輸入 `cd notebooks`，再進互動模式重跑這段：工具箱這時在上一層，路徑要改成 `../.venv/bin/python`（Windows 是 `..\.venv\Scripts\python.exe`），啟用時仍只要輸入 `python`。先預測 `pyproject.toml` 是否還會被找到；核對後用 `exit()` 退出，再 `cd ..` 回到上一層專案根目錄。只換起點，檔案本身仍存在。

紅色錯誤通常會列出哪個檔案、哪行程式，最後一行說明原因。`ModuleNotFoundError: No module named 'torch'` 表示現在的 Python 找不到工具；先確認 Notebook 選 **Tiny Perceptron**，或終端機已啟用 `.venv`；也可以在專案根目錄直接用 `.venv/bin/python` 執行，Windows 是 `.\.venv\Scripts\python.exe`（見 [W.1](#W.1)）。`FileNotFoundError` 是找不到指定檔案，先查目前起點和檔名。`KeyError` 是字典中沒有那個鍵，可回看 [W.2](#W.2)。`IndexError` 是索引超出範圍。例如 `animals = ["貓", "狗"]` 從前面用 0、1 編號；取 `animals[2]` 是取第三項，就會超出範圍。`SyntaxError` 則先檢查括號、冒號與縮排。

模型訓練的指令碼有些預設只檢查資料與流程。這類命令最後印出的那行 JSON 是報告；其中的 `mode` 若是 `dry-run-no-weight-update`，表示這次沒有更新模型；依[操作配方](training.md)加上 `--train` 才正式訓練。不要只因看到報告就以為模型學過了。閱讀錯誤時也採同樣原則：指出程式實際做到了哪一步，再處理阻擋下一步的那個原因。
