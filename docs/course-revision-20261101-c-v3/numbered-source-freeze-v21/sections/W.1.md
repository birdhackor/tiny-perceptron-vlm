## W.1 第一次開啟教材

最簡單的讀法是直接打開[教材網站](https://birdhackor.github.io/tiny-perceptron-vlm/)，不必安裝任何東西。第 1 章起，各節程式的下方都附有用 CPU 實際執行的輸出，可以像讀書一樣從頭讀到尾。本課每節會先交代一個問題，再用小例子說明做法；程式不是需要全部背起來的答案，而是讓你核對「剛才的解釋是否真的會產生這個結果」。暖身各節也可按需閱讀：遇到清單就看 [W.2](#W.2)，遇到數字表的形狀就看 [W.3](#W.3)，不用先修完一套數學課才開始第 1 章。

等某個數字引起好奇、想自己改了再跑，再打開那一節的 Notebook，也就是把正文和可執行程式放在同一份文件的工作頁。有兩種開法。第一種是在自己的電腦執行，步驟在本節下半段；下載的專案裡已經有每一節的 Notebook，放在 `notebooks/` 資料夾，直接開那裡的檔案即可。頁尾的「下載本節 .ipynb」存下的也是同一份 Notebook；它的第一個程式格會從檔案所在的資料夾往上層找專案的共用程式，下面的步驟也只從 `notebooks/` 資料夾開檔，所以下載的檔案要放進專案的 `notebooks/` 資料夾才能在本機執行。第二種是按頁尾的「在 Colab 開啟」，用 Google 的雲端服務打開同一份內容；這需要 Google 帳號，而且本課還沒有實測過 Colab 的雲端執行。在 Colab 要先跑第一個程式格（「準備本節的工具」），它會下載專案並安裝工具，等它完成才能使用後面的程式；若下載失敗，可以先讀網站正文，或改用本機方式。

本機方式先準備 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和 [Git](https://git-scm.com/install/)：uv 負責安裝與管理 Python 套件，需要時也會自己下載合適版本的 Python，不必另外安裝；Git 負責下載與管理專案檔案。安裝 Git 後在終端機執行 `git --version`，應顯示版本號。終端機是輸入指令的視窗。

下列三行依序下載專案、進入專案資料夾、安裝 CPU 與 Notebook 所需工具；`git clone` 會在終端機目前所在的資料夾裡建立 `tiny-perceptron-vlm` 資料夾。本課的訓練資料包用 Git LFS（Git 處理大型檔案的擴充工具）保存，第一行開頭的 `GIT_LFS_SKIP_SMUDGE=1` 讓 clone 不下載這些資料包；沒有安裝 Git LFS 也可以照打。Windows PowerShell 不寫這段開頭，改成先執行 `git config --global core.longpaths true`，再依序執行 `$env:GIT_LFS_SKIP_SMUDGE="1"` 與 `git clone https://github.com/birdhackor/tiny-perceptron-vlm.git`。專案裡有些檔的路徑很深，Windows 上的 Git 預設建不出超過 260 字元的路徑；設定 `core.longpaths` 的那一行讓這台電腦上的 Git 都允許長路徑，之後的 clone 與切換版本也適用。每行完成、回到可輸入狀態後再輸入下一行。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync --frozen --extra cpu --group notebook
```

`uv sync` 根據專案的套件清單建立 `.venv`，這是只供本專案使用的工具箱。`--frozen` 採用已記錄的套件版本；`--extra cpu` 選 CPU 用的 PyTorch（本課做模型計算的套件）；`--group notebook` 加入開工作頁的工具。GPU 選擇另見 [環境說明](../docs/environment.md)，現在用 CPU 足夠做短實驗。下載和安裝需要網路。完成後可在本機 CPU 環境執行本書的小型示範程式；若需要下載新的專案版本、套件或實驗檔案，請保持網路連線。

在 macOS 或 Linux 執行 `source .venv/bin/activate` 啟用工具箱；Windows PowerShell 執行 `.venv\Scripts\Activate.ps1`。啟用後，`python` 指向剛安裝的 Python 工具箱。仍在專案資料夾內，接著執行：

```bash
python scripts/check_env.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
jupyter lab notebooks
```

也可以不啟用，直接寫出工具箱裡 Python 的路徑；這時第三行改由這個 Python 啟動 JupyterLab（在瀏覽器裡開工作頁的程式），寫成 `-m jupyterlab`。[操作頁](training.md)的命令就用這種寫法。Windows PowerShell 啟用時若出現紅色錯誤，說啟用腳本無法載入，並提到 `about_Execution_Policies`（Windows PowerShell 5.1 還會列出 `PSSecurityException`），是系統的執行原則不允許執行 `Activate.ps1` 這類腳本，不表示工具箱沒有裝好，改用不啟用的寫法即可。macOS 與 Linux：

```bash
.venv/bin/python scripts/check_env.py
.venv/bin/python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
.venv/bin/python -m jupyterlab notebooks
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts/check_env.py
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

第一行檢查 Python 與 PyTorch 能否使用；看到 `compute` 那列為 `OK`（括號中的裝置名稱依電腦而異），且套件沒有「未安裝」或紅色錯誤訊息，就表示計算環境可以使用。第二行把工具箱登記成工作頁可以選用的執行環境，名稱叫 **Tiny Perceptron**。第三行啟動 JupyterLab，終端機會提供瀏覽器網址。JupyterLab 從 `notebooks` 資料夾開始列出檔案；在其中開啟 `01/1.1.ipynb`，選 **Tiny Perceptron**，點進一個程式格並按 **Shift+Enter**。一個程式格叫 cell；負責執行程式格的程序叫 Notebook 執行程序（kernel），可想成工作桌：前面程式格取過的名字還留在桌上，後面的程式格才能使用。

因為工作桌會保留前面的結果，更改前面的輸入後，需要重跑依賴它的後續程式格；只跑最後一個程式格，可能仍使用舊資料。第一次開啟、或想檢查整份能否從零重現時，在 JupyterLab 選「Kernel → Restart Kernel and Run All Cells」，清空工作桌再由上到下跑；Colab 則在「執行階段」選單找「重新啟動工作階段並全部執行」（介面也可能稱重新啟動執行階段）。想動手的話，練習打開 [1.1 正文](chapters/01.md#1.1) 對應的工作頁，先執行一次，再把輸入那行 `text = "貓看狗，狗看貓。"` 改成 `text = "鳥看狗，狗看鳥。"`，重啟並全部執行。程式最後的文字輸出應為 `鳥看狗，狗看鳥。`，而非舊句；這就是要核對的還原結果。若看到紅色錯誤，先讀最後一行再查 [W.7](#W.7)，不要急著重裝所有工具。

要試跑暖身中的短程式，先開一節的 Notebook（例如 `01/1.1.ipynb`），跑完它的第一個程式格「準備本節的工具」，再用工具列的「＋」新增程式格；Colab 則用上方「＋程式碼」。把暖身的整段程式貼入新的程式格，按 Shift+Enter。貼入時保留每行開頭的空格，因為 Python 靠行首的縮排判斷哪幾行屬於同一段，例如迴圈要重複的那幾行（見 [W.2](#W.2)）。

上面這些步驟不需要安裝 Git LFS；只讀教材與跑各節短程式也不需要它，要照[操作頁](training.md)下載訓練資料包時才需要：依 [Git LFS 官方指引](https://git-lfs.com/)安裝後，在終端機執行 `git lfs version`，應顯示 `git-lfs/` 開頭的版本號；再到專案資料夾裡執行一次 `git lfs install --local`，讓這個專案使用它。

