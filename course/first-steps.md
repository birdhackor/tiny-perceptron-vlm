# 操作與數學暖身

把模型想成一組可調旋鈕，程式負責告訴它旋鈕目前在哪裡、這次猜錯多少、往哪裡移一小步。先讓一段小程式跑起來，再慢慢補上名字。

## 第一次開啟教材

「終端機」是一個讓你輸入指令的視窗。打開它後，先確認目前在下載好的專案資料夾。下列指令都從有 `pyproject.toml` 的那一層執行。尚未有專案時，先安裝 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再執行：

```bash
git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync --frozen --extra cpu --group notebook
```

這是 CPU 閱讀與練習用的安裝。GPU 安裝選擇見 [README](../README.md)與[環境說明](../docs/environment.md)。`--frozen` 使用 repo 的套件版本清單；`.venv` 是這份專案自己的工具箱。

在 macOS／Linux 啟用工具箱：

```bash
source .venv/bin/activate
```

Windows PowerShell 改用：

```powershell
.venv\Scripts\Activate.ps1
```

若系統限制啟用，可以直接用 `.venv\Scripts\python.exe` 執行下面的 Python 指令。啟用後，`python` 和 `jupyter` 都來自同一個工具箱，不需要每次重裝。

```bash
python scripts/check_env.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
jupyter lab notebooks
```

JupyterLab 會提供本機瀏覽器入口。開啟 `01/1.1.ipynb`，右上角選 **Tiny Perceptron**。一格格的框稱為 cell；點進程式框，按 **Shift+Enter** 執行並移到下一格。先選「Restart Kernel and Run All」，確認整份由上到下可以跑。

kernel 就像這份 Notebook 的工作桌：執行過的變數會留在桌上。改了前面一格，就要重跑依賴它的後面各格。結果奇怪時，重啟工作桌再從頭執行。紅色錯誤訊息的最後一行通常最有用；`ModuleNotFoundError` 常表示選到了另一個工具箱。

雲端環境不一定提供手機可直接開啟的 Jupyter 預覽。手機可先看章節 Markdown 和 SVG；需要可執行工作桌時再使用電腦、支援 Notebook 的服務或自己的 GPU 主機。網頁版可用 `python scripts/export_course.py` 匯出後離線閱讀。

## 先認得五種 Python 寫法

```python
colors = ["red", "blue"]  # 清單：有順序的東西
record = {"question": "1+1=?"}  # 字典：用名字找欄位
for color in colors:  # 逐一拿出來
    print(color)  # 縮排表示同一個區塊
assert len(colors) == 2  # 不成立就停下來，提醒我們檢查
```

`=` 是把右邊結果取一個名字，`==` 是問兩邊是否相等。`def` 定義一段可重用的做法，呼叫時用括號放入材料。`import` 是把工具帶進來；閱讀前幾章時先只認得 `torch`。

## tensor 是有軸的數字盒子

一個數叫純量，一排數像向量，一張數字表像矩陣。tensor 只是把這些統稱起來，也可以有更多軸。名字不難，難的是每個軸代表什麼。

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(x.shape)  # torch.Size([2, 3])：2列、每列3個數
print(x[0])  # 第0列；程式從0開始編號
print(x.mean(-1))  # 每列自己平均，結果有2個數
```

模型常用 `[B,T,D]`：B 是同時處理幾筆、T 是每筆有幾張字卡、D 是每張字卡有幾個數。這些字母是方便標示軸的名稱，並不是 Python 的特殊語法。把所有軸壓成一排會丟掉原來的分組方式；`reshape` 必須知道要分回哪些組。

「token」可先理解成字卡；早期一張卡是一個字，後來也可能是一個位元組或一段常見文字。「embedding」是查表取得的向量。ID=3 是第三個索引，並不表示這張字卡比 ID=2 大一點。

![查表](figures/lookup.svg)

## 忘了線性代數，可以先想配方

矩陣乘法是在做多種加權配方。`[2,3] @ [3,4]` 會得到 `[2,4]`：每筆有3個材料，配成4種結果。Linear 是共用這組配方，再加一個偏移量。它沿最後一軸工作，其他軸保留。

```python
from torch import nn

layer = nn.Linear(3, 4)
print(layer(x).shape)  # 2筆仍是2筆；每筆從3個數變4個數
```

注意 PyTorch 的權重放成 `[out,in]`，實際計算是 `x @ weight.T + bias`。遇到 shape 錯誤時先寫下每個軸的意思，再看相乘的內側數字是否一致。

## 忘了微積分，可以先想旋鈕敏感度

導數問：「旋鈕微調一點，代價會怎麼變？」梯度把每個旋鈕的敏感度排起來。梯度是正的，表示旋鈕往上調會增加代價；所以更新通常往反方向走。

```python
w = torch.tensor(1.0, requires_grad=True)
loss = (w - 3).square()
loss.backward()
print(loss.item(), w.grad.item())  # 4、-4；手算 d(w-3)^2/dw=2(w-3)
```

`.backward()` 算敏感度；它還沒有移動旋鈕。`optimizer.step()` 才更新參數。`.grad` 會累加，因此每次新更新通常先 `zero_grad()`。第1章會把這三件事分開驗算。

![旋鈕與梯度](figures/gradient.svg)

## 機率、log 與猜錯的代價

模型先產生分數。softmax 將分數換成加總為1的比例；分數差越大，分配越偏向高分。機率0.9的正確答案比機率0.1的正確答案更好，因此可用 `-log(p)` 表示代價。

PyTorch 的 cross entropy 接收原始分數，內部已做穩定的 log-softmax。不要先做 softmax 再送入它。程式裡的 `log` 是自然對數；要轉成「幾個 bit」，再除以 `log(2)`。學會這些關係比背 API 名稱更重要。

## 檔案放哪裡

| 路徑 | 用途 |
| --- | --- |
| `course/chapters/` | 教材正文與短程式，是 Notebook 的來源 |
| `notebooks/` | 每小節一份工作桌，原檔不保存執行輸出 |
| `tiny_perceptron/` | 可逐檔閱讀的共享模型零件 |
| `scripts/` | 資料產生、訓練、推論、評估與教材建置入口 |
| `data/` | 訓練資料，Git 忽略 |
| `checkpoints/` | 模型存檔，Git 忽略 |
| `outputs/` | 報告、已執行 Notebook、離線網頁，Git 忽略 |

`--train` 是正式更新權重的開關。初次檢查流程先不加，確認報告是 `dry-run-no-weight-update`；想開始訓練時再依[操作配方](training.md)選擇一個小任務。
