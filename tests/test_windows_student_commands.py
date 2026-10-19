"""頁面教 Windows 讀者在 PowerShell 輸入的每一條命令，都要原樣寫進 Windows CI，或明列為刻意不跑。

.github/workflows/windows-student-commands.yml 在 windows-latest 照 README.md、README_en.md、course/first-steps.md、
course/training.md 的寫法跑學生命令。workflow 與頁面是兩份副本，這裡逐條把它們釘在一起：頁面上的 Windows 命令被
改寫、新增或刪除，而 workflow 沒跟著改，就在這裡失敗，訊息指出頁面與行號。

驗證：依頁面寫明的 Windows 換法，推出讀者實際輸入的命令序列，再和 workflow 裡 Invoke-Page（執行）與 Skip-Page
（不跑，-Reason 是印進記錄的簡短理由，完整理由寫在呼叫處的註解）標了該頁的命令依序逐條比對。命令取自 shell 類 fenced
區塊的每一行（MACOS_ONLY_BLOCKS 除外），以及 fence 外的 inline code——開頭是命令（COMMAND_START），或在提到
Windows／PowerShell 的句子裡帶 PowerShell 才有的寫法（POWERSHELL_ONLY）；說明裡提到的命令名稱與其他平台的寫法列在
NOT_WINDOWS。標成 text 這類純文字（PLAIN_LANGUAGES）的區塊不比對，但其中有一行符合 COMMAND_START 就報錯；fence 外出現
HTML 的 <code> 或 <pre> 也報錯，因為這裡只認得反引號與 fenced 區塊，寫在 HTML 裡的命令看不到。頁面的換法由
REPLACEMENTS（bash 行換成同頁句子裡給 Windows 的寫法）與 PREFIXES（整頁的 `.venv/bin/python` 換成 Windows 路徑）表示。
另外核對 workflow 的 Get-PageCode 找得到它要取的 Python 程式、每個 Skip-Page 都有 -Reason、頁面上 Windows 讀者的
git clone 之前先有 LONG_PATHS 且其他命令不另外設 core.longpaths、workflow 在 clone 之前用 Invoke-Page 執行
LONG_PATHS，以及 Windows PowerShell 5.1 與 PowerShell 7 都跑。kernel 核對（workflow 某一步 env 裡的 KERNEL_CHECKS
程式；這裡載入它的函式來用）從頁面取的 W.1 改字練習原句、新句、期望輸出，與 W.6 第一行輸出的期望值，各對到頁面上
剛好一句，原輸入行在 notebooks/01/1.1.ipynb 的程式格裡剛好一行。觸發條件是預設觸發：push 與 pull_request 的 paths
相同、第一條是 '**'、其餘每條都是 '!<樹>/**' 而且那棵樹在 repo 裡。排除的樹的名字，不以一整串（各段在檔案文字裡以單一個
/ 或反斜線相連）出現在四頁的程式區塊、workflow 的 env 與每一步的 run:、env、KERNEL_CHECKS 執行的 Notebook 程式格，以及
workflow 會執行的 repo 內 Python 檔（python_run_by_workflow：Invoke-Page 執行的腳本與模組，連同它們在 repo 內匯入的模組）
裡；on: 區塊本身寫著排除樣式，不掃。workflow 檔本身、四頁、KERNEL_CHECKS 執行的 Notebook 與 python_run_by_workflow 列的檔，
repo 內的 POSIX 路徑都不以任何排除的樹加 / 開頭。

觸發條件的讀檔集合：排除的樹裡不能有頁面命令或 workflow 會讀的檔。這份集合由三種來源合起來推出。(a) 在本機用
sys.addaudithook 收 open 事件，執行 workflow 會跑、又能在本機安全執行的 Python 頁面命令（check_env.py、
fetch_training_assets.py 與 fetch_course_models.py 的 --list、scripts.course_experiments.run 對 workflow 所跑實驗的
--list-assets、各腳本的 --help、W.7 的頁面程式）與 KERNEL_CHECKS（要用裝了 Notebook 套件的環境），記下開啟的追蹤檔；
kernel 是另一個程序，audit hook 看不到它執行程式格時開的檔，那些程式格由 (b) 掃。(b) 靜態掃描 Invoke-Page 執行的 Python 進入點、
docs/course-experiments/plan.json 替 workflow 所跑實驗指定的模組、Get-PageCode 取的頁面程式、notebooks/01/1.1.ipynb
的程式格與 KERNEL_CHECKS，連同它們在 repo 內匯入的模組，找出寫死的 repo 路徑（ROOT /、Path(__file__)、字面的 .json、
.jsonl、.tar.gz、.ipynb、.svg）。(c) workflow 自己的 PowerShell 讀的檔：Get-PageCode 讀的頁面、Assert-CheckEnv 讀的
scripts/check_env.py、Assert-LfsPointers 讀的 assets/training/*.tar.gz、JupyterLab 探測讀的 notebooks/01/1.1.ipynb。
讀到的檔落不落在排除的樹，這個檔每次核對的只有兩件事：test_excluded_trees_are_not_named_where_the_workflow_reads 核對
排除的樹的名字不以一整串出現在上面列的地方；test_known_workflow_reads_are_outside_excluded_trees 核對那幾類已知會讀的檔
不在排除的樹底下。(a) 實際開啟的檔、(b) 與 (c) 裡不是一整串寫出的路徑，以及下面〈不涵蓋〉列的看不到的地方，要在新增或改動
Invoke-Page、Get-PageCode 或 KERNEL_CHECKS，或 runner 那一側新讀了檔時，照三種來源重推一次，確認讀到的檔都不在排除的樹裡。

不涵蓋：
- 開頭不是 COMMAND_START、又不在 Windows 句子裡帶 PowerShell 寫法的 inline code（例如只寫腳本名的 `train.py --task vision`）。
- 純文字區塊裡開頭不符合 COMMAND_START 的命令（例如前面帶 `PS>` 提示字元的行）。
- 句意只認錨點文字，換法句子改了措辭只會報「找不到」；換法句子以外的 inline code 命令依所在行號和程式區塊的命令排序，句子寫的先後（例如「clone 之後再執行」）不比對；續行的命令逐行比對。
- Get-PageCode 取用的 Python 程式內容不比對（CI 執行時直接從頁面讀）。
- KERNEL_CHECKS 從頁面讀的值以外，頁面寫給讀者核對的預期值（例如 W.7 印出的 `True`、training.md 的 `holdout_examples` 2、kernel 名稱 Tiny Perceptron）沒有和 workflow 的核對值互相釘住，頁面改了預期值而 workflow 照舊核對舊值，這裡不會失敗。
- Skip-Page 的 -Reason 與註解寫的理由對不對；workflow 的命令在 Windows 上是否真的成功，要看 CI 本身。
- 排除的樹的名字只在一整串寫出時比對得到，不是一整串寫出的路徑都看不到：用變數組出的、分段拼接的（Python 的 `ROOT / "docs" / "x"`、`Path("docs", "x")`、`os.path.join`，PowerShell 的 `Join-Path`），以及 Python 字串裡把反斜線跳脫成兩個字元的寫法。排除安不安全，靠的是 workflow 排除旁的分類理由與新增排除時照三種來源的推導，不靠這個檔的測試。
- 讀檔集合的推導看不到用變數組出的路徑（例如 fetch_training_assets.py 從清單取的資料包路徑）、子程序讀的檔（git、git lfs、uv 與它呼叫的建置工具、jupyter、kernel）、只在 Windows 才走的分支；它們讀到的檔若落在排除的樹裡，改了它不會觸發。
- git clone、checkout、status --porcelain 與 ls-files 經手排除樹裡每個檔的路徑：排除樹裡新增只差大小寫的兩個路徑時，學生 clone 出的工作樹不完整，這裡不跑；Windows 不允許的檔名（例如含 `:`）會讓每次 push 都跑的 ci.yml 在 windows-latest 取出檔案時失敗。
- 一次改到超過 3,000 個檔、而 GitHub 先列出的 3,000 個都在排除的樹裡時，GitHub 不跑這個 workflow。
- course/first-steps.md W.1 在瀏覽器裡的操作：選 Tiny Perceptron、點進程式格按 Shift+Enter、選單的 Restart Kernel and Run All Cells、用工具列的「＋」新增程式格貼上暖身程式。kernel 執行程式格這件事，由 KERNEL_CHECKS 用 nbclient 代替瀏覽器核對。
- course/first-steps.md W.1 在 Colab 開啟與執行，以及把網站頁尾下載的 .ipynb 放進 notebooks/ 再開：CI 不經過網站與 Colab。

Windows CI 只核 workflow 裡旁邊以 `# 頁面說：` 引了頁面原話的結果，頁面要讀者核對的其他結果，CI 都不核。下面記的是
其中容易被誤以為有核的，以及不核的理由，不宣稱列盡；另有一些記在 workflow 核對旁的註解，例如 Assert-CheckEnv 的紅色
錯誤只靠結束碼。入列準則：
頁面明確要讀者核對、確認、檢查、比較或查看的結果，寫成「應為／應看到／應顯示／要等於」這類期望的結果，讀者用來判斷的
條件（「看到…就表示」），以及寫「希望」的答案；描述命令做什麼、工具自己核對的事、給維護者的流程與一般建議不算。下面的
引文沒有釘住頁面，頁面改寫後可能對不上。練習的改法或期望值只寫在句子裡時，要核對就得把句意寫死成 workflow 裡的程式，
頁面改了練習，那份副本也不會跟著變，所以不核：
- course/first-steps.md W.2「核對時應看到 `鳥 3` 和三行「看到」」：要換掉清單與最後檢查的哪兩行、要數的「三行」只寫在句子裡。
- course/first-steps.md W.2「動物清單的輸出應不受影響」：頁面沒寫 `caption` 要改成什麼。
- course/first-steps.md W.3「核對應得到 `[80, 90]` 與 `[70, 80, 105]`」：改法「從 80 改成 110」只寫在句子裡，程式印出的也是 tensor 的寫法。
- course/first-steps.md W.4「再執行核對，應約為 `1.386`」：「在清單加入 `0.25`」沒寫出改後的那一行。
- course/first-steps.md W.5「先預測兩杯成本都增加 3，再執行核對」：改法「把偏移的 5 改成 8」與期望「增加 3」只寫在句子裡。
- course/first-steps.md W.6「再試 1.1，應到 5.4、代價 5.76」與「核對這兩個結果」：改法「把學習率 0.1 改成 0.6」只寫在句子裡。
- course/first-steps.md W.7「看到 `>>>` 就進入 Python 互動模式」：CI 把程式從管線送進 Python，是腳本模式，不印 `>>>`。
- course/first-steps.md W.7 錯誤訊息那段的「先確認 Notebook 選 **Tiny Perceptron**」「先查目前起點和檔名」「先檢查括號、冒號與縮排」：只在出現那個錯誤時才做，沒有期望值。
- course/training.md T.1「打開幾筆，檢查 `text` 或 `messages` 等欄位」：沒有期望值。
- course/training.md T.1「先核對圖像尺寸、聲音取樣率與任務答案」：對象是讀者自己的素材，頁面沒有命令也沒有期望值。
- course/training.md T.3「打開前後 JSON，分別計算兩個 loss 的「更新後減更新前」」：沒有期望值。
- course/training.md T.3「核對 MLP 自己的前後報告，再比較不同模型」：沒有期望值。
- course/training.md T.4 評估 JSON 的欄位表（「要核對什麼」那一欄）：寫的是各欄的意思，沒有期望值；文字模式匹配率為 `null` 另有核對。
- course/training.md T.4「從該筆 `messages` 找問題，再和 `samples` 的目標及生成並排」：沒有期望值。
- course/training.md T.4「`--limit all` 選全檔，仍要確認是否有跳過題」：沒有期望值。
- course/training.md T.4「用同樣的資料、題號、生成上限比較前後」：沒有期望值。
- course/training.md T.5「再檢查兩者是否都遵守格式」：檢查的是讀者自己寫的短答與比喻。
- course/training.md T.5「把 validation 的提問、理想回答與生成並排」：沒有期望值。
- course/training.md T.5「需要另看比喻是否有用」：沒有期望值；style 的訓練與評估有跑，CI 只核對評估 JSON 解得開，沒有看生成。
- course/training.md T.5「也要看正常題有沒有被過度拒絕」：沒有期望值；safety 的訓練與評估有跑，CI 只核對評估 JSON 解得開，沒有看生成。
- course/training.md T.5「若多了比喻卻把答案數字改錯，就沒有完成目標」：這是讀者評估自己訓練的模型的判準，頁面沒說這次訓練的模型一定答對。
- course/training.md T.5 固定配方「`lora/` 的 adapter 要載入原始浮點底座並核對指紋」：固定配方用 `--device cuda`，這裡不跑。
- course/training.md T.5 固定配方「使用或分享前，先核對資料授權」：要讀的是授權說明，沒有命令輸出。
- course/training.md T.6「依序確認三件事」的「留出了什麼」：pretrain_encoders.py 印出的 JSON 只有留出題數，沒有記下留出哪幾題。
- course/training.md T.6「效果仍以驗證題核對」：沒有期望值。
- course/training.md T.6「希望答案依序為 `circle`、`high`、`square,low`」與「再看結束是否完整」：頁面寫「希望」，沒說模型一定答對；結束是否完整沒有期望值。
- course/training.md T.6「檢查文字能力時，要用其中的文字核心評估同一份文字驗證題」：頁面沒給命令，也沒有期望值。
- course/training.md T.6 補充區「要另外核對生成是否符合換入圖」：頁面沒給這條的命令。
- course/training.md T.7「先看 `chosen`、`rejected` 是否符合自己的判準」：判準由讀者自訂。
- course/training.md T.7「再對其餘驗證題做相同檢查」與「確認新階段沒有破壞它們」：沒有期望值。
- course/training.md T.8「但仍須查看是否有跳過題」：沒有期望值。
- course/training.md T.8「把平均代價、有效目標數與逐筆生成並排」：沒有期望值。
- course/training.md T.8 固定配方「是否支援某精度須看報告」：固定配方用 `--device cuda`，這裡不跑。
- course/training.md T.8 固定配方 flash_probe「先查看各路線的`status`」與「還要核對記憶體範圍」：固定配方用 `--device cuda`，這裡不跑。
- course/training.md T.9「希望答案是`blue`」與「這一題先讓你查看前後有沒有改變」：頁面寫「希望」，沒說模型一定答對；前後有沒有改變沒有期望值。
- course/training.md T.9「在三份JSON中用相同`row`找回同一筆提問與目標」：沒有期望值。
- course/training.md T.9「分別說明「保存變小了嗎」與「相同題目是否維持回答」」：沒有期望值。
- course/training.md T.10「核對下列六個檔案都已產生」：產生它們的 sft、style、moe 固定配方用 `--device cuda`，這裡不跑。
- course/training.md T.10「先選同寬度的兩支，用相同題目檢查」與「希望答案是 `square,high`」：學生要用那三組固定配方的教師，這裡不跑。
- course/training.md T.10「核對整份相同題目，並一起記學生參數、教師成本與有效目標數」：沒有期望值；這一節的命令全部不跑，蒸餾要先有那三組 `--device cuda` 固定配方的教師。
- course/training.md T.11 補充區「逐步核對時請對照工具紀錄勘誤」：要讀的是實驗紀錄，沒有命令輸出。
"""

import ast
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/windows-student-commands.yml"
PAGES = ("README.md", "README_en.md", "course/first-steps.md", "course/training.md")
# 觸發條件裡排除一棵樹的寫法：'!'、樹的路徑、'/**'。路徑裡不能有 GitHub 路徑樣式的特殊字元（* ? + [ ] !），
# 否則排除的就不只那一棵樹。
TREE_EXCLUSION = re.compile(r"!(?P<tree>[^/*?+\[\]!][^*?+\[\]!]*)/\*\*")
# workflow 某一步 env 裡的 Python 程式：用登記好的 kernel 執行 W.1 的改字練習與 W.6 的程式。
KERNEL_CHECKS = "PAGE_KERNEL_CHECKS"
# 命令裡啟動 Python 的那個字（反斜線先換成 /）：python，或結尾是 python.exe 的路徑。
PYTHON_COMMAND = re.compile(r"(?:.*/)?python(?:\.exe)?")

FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
SHELL_LANGUAGES = {"bash", "sh", "shell", "zsh", "fish", "console", "powershell", "pwsh", "ps1", "ps", "cmd", "bat"}
# 純文字區塊不比對命令，但其中有一行符合 COMMAND_START 就報錯：讀者照樣會把它打進 PowerShell。
PLAIN_LANGUAGES = {"text", "txt", "plaintext", "output"}
OTHER_LANGUAGES = {"python", "json", "jsonl", "yaml", "toml", *PLAIN_LANGUAGES}
SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
# fence 外的 HTML code 元素。這裡不解析 HTML，寫在裡面的命令看不到，所以出現就報錯。
HTML_CODE = re.compile(r"<(?:code|pre)\b", re.IGNORECASE)
# 句子在這些標點之後結束；英文的句點、分號後面要接空白或行尾。
TERMINATOR = re.compile(r"[。；！？]|[.;!?](?=\s|$)")
WINDOWS_WORDS = ("Windows", "PowerShell")
COMMAND_START = re.compile(
    r"^(?:(?:git|uv|uvx|cd|python3?|py|pip|jupyter|pytest|ruff|source|mkdir|powershell|pwsh|winget|choco|scoop)(?=\s|$)"
    r"|\.venv[/\\]|\.\.?[/\\]|\$env:|[A-Z][a-z]+-[A-Z][A-Za-z]+(?=\s|$))"
)
POWERSHELL_ONLY = re.compile(r"\\|\$|\.(?:exe|ps1)\b|^[A-Z][a-z]+-[A-Z][A-Za-z]+(?=\s|$)")

# 頁面句子裡給 Windows 的寫法，取代 bash 區塊裡的一行：(bash 行, 錨點)。取代的寫法是錨點之後、同一句裡的 inline code。
REPLACEMENTS = {
    "README.md": [
        (
            "GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git",
            "Windows PowerShell 不寫這段開頭，改成先執行",
        ),
        ("source .venv/bin/activate", "Windows PowerShell 啟用指令是"),
    ],
    "README_en.md": [
        (
            "GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git",
            "In Windows PowerShell, leave that prefix out: run",
        ),
        ("source .venv/bin/activate", "On Windows PowerShell, activate with"),
    ],
    "course/first-steps.md": [
        (
            "GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git",
            "Windows PowerShell 不寫這段開頭，改成先執行",
        ),
    ],
    "course/training.md": [
        ("mkdir -p outputs checkpoints", "Windows PowerShell 把 `mkdir -p outputs checkpoints` 換成"),
    ],
}
# Windows 讀者在 git clone 之前先執行的長路徑設定。在 windows-latest 上觀察到：clone 帶 -c core.longpaths=true 時，
# 取出檔案仍報 Filename too long；先執行這一行、clone 不帶 -c，檔案就取得出來。
LONG_PATHS = "git config --global core.longpaths true"
# 整頁命令開頭的路徑換法：(原寫法所在的錨點, Windows 寫法所在的錨點)；兩者都取錨點之後的第一段 inline code。
PREFIXES = {
    "course/training.md": ("命令一律寫成", "Windows PowerShell 把這段路徑換成"),
}
# 錨點所在那一行之後的第一個 fenced 區塊只給 macOS／Linux；Windows 讀者改用同頁另給的 PowerShell 區塊。
MACOS_ONLY_BLOCKS = {
    "course/first-steps.md": ["macOS 與 Linux："],
}
# inline code 看起來像命令，但不是 Windows 讀者要在 PowerShell 輸入的：(含那段 inline code 的原文片段, 理由)。
NOT_WINDOWS = {
    "README.md": [
        ("把 `uv sync` 那一行換成下表的寫法", "指出要換掉哪一行"),
        ("| Apple Silicon | `uv sync --frozen --group notebook` |", "Apple Silicon 的安裝選擇"),
        ("直接用 `python`", "說明啟用後直接用 python"),
        ("或每次 `uv run` 都帶相同 extra", "提到 uv run 的用法，不是完整命令"),
    ],
    "README_en.md": [
        ("replace the `uv sync` line", "指出要換掉哪一行"),
        ("use `uv sync --frozen --group notebook` on Apple Silicon", "Apple Silicon 的安裝選擇"),
        ("Installation and the `git lfs version` check are in", "提到「第一次開啟教材」那節的檢查命令"),
        ("Activate `.venv` and use `python` directly", "說明啟用後直接用 python"),
        ("pass the same extra to each `uv run`", "提到 uv run 的用法，不是完整命令"),
    ],
    "course/first-steps.md": [
        ("`git clone` 會在終端機目前所在的資料夾裡建立", "說明 git clone 建立的資料夾"),
        ("`uv sync` 根據專案的套件清單建立", "說明 uv sync 做什麼"),
        ("在 macOS 或 Linux 執行 `source .venv/bin/activate` 啟用工具箱", "macOS／Linux 的啟用寫法"),
        ("啟用後，`python` 指向剛安裝的 Python 工具箱", "說明啟用的效果；CI 在啟用後另外核對"),
        ("不允許執行 `Activate.ps1` 這類腳本", "錯誤說明裡的檔名"),
        ("| `.venv/` | `uv sync` 建立的工具箱", "路徑表的資料夾與說明"),
        ("在終端機先 `cd tiny-perceptron-vlm`，是把起點移到", "回頭說明前面已經輸入過的 cd"),
        ("macOS 與 Linux 是 `.venv/bin/python`，Windows PowerShell 是", "macOS／Linux 的寫法"),
        ("才能再用 `cd` 切換資料夾", "提到 cd 的用途，不是完整命令"),
        ("路徑要改成 `../.venv/bin/python`（Windows 是", "macOS／Linux 的寫法"),
        ("也可以在專案根目錄直接用 `.venv/bin/python` 執行", "macOS／Linux 的寫法"),
    ],
    "course/training.md": [
        ("這時工具會先執行 `git lfs pull`", "說明下載工具自己會執行的命令"),
    ],
}

CALL = re.compile(
    r"^[ \t]*(?:\$\w+[ \t]*=[ \t]*)?(?P<verb>Invoke-Page|Skip-Page)[ \t]+-Pages[ \t]+(?P<pages>\S+)"
    r"(?:[ \t]+-Reason[ \t]+'(?P<reason>(?:[^'\n]|'')*)')?"
    r"(?:[ \t]+-Stdin[ \t]+\$\w+)?[ \t]+-Commands[ \t]+(?:@'\n(?P<block>.*?)\n'@|'(?P<line>(?:[^'\n]|'')*)')",
    re.M | re.S,
)
CALL_START = re.compile(r"^[ \t]*(?:\$\w+[ \t]*=[ \t]*)?(?:Invoke|Skip)-Page\b", re.M)
PAGE_CODE = re.compile(r"Get-PageCode[ \t]+-Page[ \t]+(?P<page>\S+)[ \t]+-After[ \t]+'(?P<anchor>(?:[^'\n]|'')*)'")


@dataclass(frozen=True)
class Command:
    line: int
    text: str


@dataclass(frozen=True)
class Copy:
    step: str
    verb: str
    pages: tuple
    text: str


@dataclass
class Span:
    line: int
    start: int
    end: int
    text: str
    sentence: int
    windows: bool


def read(relative):
    # .gitattributes 讓 checkout 用 LF；沒套用時（例如 Windows 上手動複製的檔案）也照 LF 比對。
    return (ROOT / relative).read_text(encoding="utf-8").replace("\r\n", "\n")


def load_workflow():
    return yaml.load(read(WORKFLOW), Loader=yaml.BaseLoader)


def excluded_trees():
    """觸發條件排除的樹：push 的 paths 在第一條之後、寫成 '!<樹>/**' 的每一條。寫法不符的條目不在這裡，
    由 test_workflow_triggers_and_shells 報錯。"""
    paths = load_workflow()["on"]["push"]["paths"]
    return [match["tree"] for match in map(TREE_EXCLUSION.fullmatch, paths[1:]) if match]


def kernel_checks_source():
    """(步驟名稱, KERNEL_CHECKS 的 Python 程式)。"""
    found = [
        (step.get("name", ""), step["env"][KERNEL_CHECKS])
        for job in load_workflow()["jobs"].values()
        for step in job.get("steps", [])
        if KERNEL_CHECKS in (step.get("env") or {})
    ]
    assert len(found) == 1, f"{WORKFLOW} 要剛好一個步驟在 env 寫 {KERNEL_CHECKS}，現在有 {len(found)} 個"
    return found[0]


def kernel_checks():
    """KERNEL_CHECKS 那段 Python 程式當成模組載入，回傳它的名稱空間。這裡只用到不需要 Notebook 套件的部分：
    nbformat 與 nbclient 在它的 main() 裡才 import，載入時不執行 main()。"""
    name, source = kernel_checks_source()
    namespace = {"__name__": KERNEL_CHECKS}
    exec(compile(source, f"{WORKFLOW} 步驟「{name}」的 {KERNEL_CHECKS}", "exec"), namespace)
    return namespace


def page_code_anchors():
    """workflow 每個 Get-PageCode 的 (頁, 錨點)。"""
    return [
        (match["page"], match["anchor"].replace("''", "'"))
        for job in load_workflow()["jobs"].values()
        for step in job.get("steps", [])
        for match in PAGE_CODE.finditer(step.get("run", ""))
    ]


def page_code(page, anchor):
    """Get-PageCode 取的程式：含錨點的第一行之後的第一個 python 程式區塊。找不到時回傳空字串，由
    test_page_code_anchors_point_at_python_blocks 報錯。"""
    text = read(page)
    first = next((number for number, line in enumerate(text.split("\n"), 1) if anchor in line), None)
    blocks = [body for start, language, body in split_page(text)[0] if first and start > first and language == "python"]
    return "\n".join(line for _, line in blocks[0]) if blocks else ""


def repo_module(name, directory, level=0):
    """import 的模組名稱對到的 repo 內檔案；第三方套件與標準函式庫回傳 None。level 是相對匯入的點數；沒有點時先找
    repo 根，再找程式所在的資料夾（python scripts/x.py 執行時，scripts/ 在 sys.path 上）。"""
    if level:
        base = directory
        for _ in range(level - 1):
            base = base.parent
        bases = [base]
    else:
        bases = [ROOT, directory]
    for base in bases:
        candidate = base.joinpath(*name.split(".")) if name else base
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if path.is_file():
                return path.resolve()
    return None


def imported_repo_files(source, directory):
    """一段 Python 程式匯入的 repo 內檔案；寫在函式裡的 import 也算。"""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            level, names = 0, [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            level = node.level
            names = [prefix, *(".".join(filter(None, (prefix, alias.name))) for alias in node.names)]
        else:
            continue
        for name in names:
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                if path := repo_module(".".join(parts[:depth]), directory, level):
                    found.add(path)
    return found


def python_run_by_workflow():
    """workflow 會執行的 repo 內 Python 檔：Invoke-Page 執行的腳本與 -m 模組、scripts.course_experiments.run 依
    plan.json 用名字載入的同資料夾模組（整個資料夾都算），以及這些檔、Get-PageCode 取的頁面程式、KERNEL_CHECKS 與它
    執行的 Notebook 程式格在 repo 內匯入的模組，任何深度。"""
    queue = []
    for copy in workflow_copies()[0]:
        words = copy.text.replace("\\", "/").split()
        position = next((index for index, word in enumerate(words) if PYTHON_COMMAND.fullmatch(word)), None)
        if copy.verb != "Invoke-Page" or position is None:
            continue
        target = words[position + 1 : position + 3]
        if target[:1] == ["-m"] and len(target) == 2:
            queue.append(repo_module(target[1], ROOT))
            if target[1] == "scripts.course_experiments.run":
                queue += (ROOT / "scripts/course_experiments").glob("*.py")
        elif target and (ROOT / target[0]).is_file():
            queue.append(ROOT / target[0])
    snippets = [(page_code(page, anchor), ROOT) for page, anchor in page_code_anchors()]
    snippets.append((kernel_checks_source()[1], ROOT))
    notebook = kernel_checks()["NOTEBOOK"]
    snippets += [
        ("".join(cell["source"]), (ROOT / notebook).parent)
        for cell in json.loads(read(notebook))["cells"]
        if cell["cell_type"] == "code"
    ]
    for source, directory in snippets:
        queue += imported_repo_files(source, directory)
    found = set()
    while queue:
        path = queue.pop()
        if path is None or (path := path.resolve()) in found:
            continue
        found.add(path)
        queue += imported_repo_files(path.read_text(encoding="utf-8"), path.parent)
    return sorted(found)


def split_page(text):
    """(區塊, fence 外的行)：區塊是 [(開頭行號, 語言, [(行號, 內容)])]；fence 外的行是 [(行號, 內容)]。"""
    blocks, prose, opened = [], [], None
    for number, line in enumerate(text.split("\n"), 1):
        fence = FENCE.match(line)
        if opened is None:
            if fence:
                opened = (fence[1][0], len(fence[1]))
                blocks.append((number, (fence[2].split() or [""])[0].lower(), []))
            else:
                prose.append((number, line))
        elif fence and fence[1][0] == opened[0] and len(fence[1]) >= opened[1] and not fence[2].strip():
            opened = None
        else:
            blocks[-1][2].append((number, line))
    return blocks, prose


def sentences(line):
    """(遮掉 inline code 的行, 各句的起點與終點)：inline code 裡的標點不算句子結束。"""
    masked = list(line)
    for match in SPAN.finditer(line):
        masked[match.start() : match.end()] = "\0" * (match.end() - match.start())
    masked = "".join(masked)
    bounds = [0, *(match.end() for match in TERMINATOR.finditer(masked)), len(line) + 1]
    return masked, list(zip(bounds, bounds[1:]))


def sentence_at(line, position):
    """position 落在這一行的第幾句。"""
    _, ranges = sentences(line)
    return next(index for index, (start, end) in enumerate(ranges) if start <= position < end)


def spans_in(number, line):
    """一行 fence 外文字裡的每段 inline code，記下它在第幾句、那句有沒有提到 Windows 或 PowerShell。"""
    masked, ranges = sentences(line)
    result = []
    for match in SPAN.finditer(line):
        content = match[2]
        if content.startswith(" ") and content.endswith(" ") and content.strip():
            content = content[1:-1]
        sentence = sentence_at(line, match.start())
        start, end = ranges[sentence]
        windows = any(word in masked[start:end] for word in WINDOWS_WORDS)
        result.append(Span(number, match.start(), match.end(), content, sentence, windows))
    return result


def occurrences(prose, text):
    """text 在 fence 外每次出現的 (行號, 起點)。"""
    found = []
    for number, line in prose:
        start = line.find(text)
        while start != -1:
            found.append((number, start))
            start = line.find(text, start + 1)
    return found


def unique_anchor(page, prose, anchor, problems):
    """錨點在 fence 外剛好出現一次時回傳 (行號, 起點, 終點)；否則記下問題並回傳 None。"""
    hits = occurrences(prose, anchor)
    if len(hits) != 1:
        where = f"出現在第 {'、'.join(str(number) for number, _ in hits)} 行" if hits else "找不到"
        problems.append(f"{page}：「{anchor}」要在頁面上剛好出現一次，現在{where}；頁面的 Windows 寫法改了嗎？")
        return None
    number, start = hits[0]
    return number, start, start + len(anchor)


def windows_commands(page):
    """(命令, 問題)：照頁面寫明的 Windows 換法，依頁面順序列出 Windows 讀者在 PowerShell 輸入的命令。"""
    problems = []
    blocks, prose = split_page(read(page))
    for number, line in prose:
        masked, _ = sentences(line)
        if HTML_CODE.search(masked):
            problems.append(
                f"{page}:{number}：fence 外有 HTML 的 <code> 或 <pre>，這個測試不支援；寫在裡面的命令不會被比對，"
                "請改用反引號的 inline code 或 fenced 區塊"
            )
    spans = [span for number, line in prose for span in spans_in(number, line)]
    picked = {
        id(span)
        for span in spans
        if COMMAND_START.match(span.text) or (span.windows and POWERSHELL_ONLY.search(span.text))
    }

    excluded = set()
    for anchor in MACOS_ONLY_BLOCKS.get(page, []):
        if (hit := unique_anchor(page, prose, anchor, problems)) is None:
            continue
        following = [block for block in blocks if block[0] > hit[0]]
        if not following or not dict(prose)[hit[0]].rstrip().endswith(anchor):
            problems.append(f"{page}:{hit[0]}：「{anchor}」要在行尾，後面接著 macOS／Linux 的程式區塊")
            continue
        excluded.add(following[0][0])

    lines = []
    for start, language, body in blocks:
        if language in PLAIN_LANGUAGES:
            problems += [
                f"{page}:{number}：標成「{language}」的區塊裡，「{content.strip()}」看起來是命令；"
                "shell 命令請標 bash 或 powershell，這個測試才會比對它"
                for number, content in body
                if COMMAND_START.match(content.strip())
            ]
        if language in OTHER_LANGUAGES:
            continue
        if language not in SHELL_LANGUAGES:
            problems.append(f"{page}:{start}：程式區塊的語言「{language}」沒有分類；shell 命令請標 bash 或 powershell")
            continue
        if start not in excluded:
            lines.extend(Command(number, content.strip()) for number, content in body if content.strip())

    def after(hit, count=None):
        """錨點之後、和錨點同一句的 inline code；count 只取前幾段。"""
        number, _, end = hit
        sentence = sentence_at(dict(prose)[number], end - 1)
        chosen = [span for span in spans if span.line == number and span.start >= end and span.sentence == sentence]
        return chosen[:count] if count else chosen

    replaced = {}
    for bash, anchor in REPLACEMENTS.get(page, []):
        targets = [command for command in lines if command.text == bash]
        hit = unique_anchor(page, prose, anchor, problems)
        if len(targets) != 1:
            where = "、".join(str(command.line) for command in targets) or "無"
            problems.append(f"{page}：要換成 Windows 寫法的 bash 行「{bash}」不只一處或找不到（第 {where} 行）")
        if hit is None or len(targets) != 1:
            continue
        chosen = after(hit)
        if not chosen:
            problems.append(f"{page}:{hit[0]}：「{anchor}」之後同一句裡沒有 Windows 寫法的 inline code")
            continue
        for span in spans:
            if span.line == hit[0] and hit[1] <= span.start < chosen[-1].end:
                picked.discard(id(span))
        replaced[targets[0]] = [Command(span.line, span.text) for span in chosen]

    prefix = None
    if page in PREFIXES:
        hits = [unique_anchor(page, prose, anchor, problems) for anchor in PREFIXES[page]]
        if None not in hits:
            pair = [after(hit, 1) for hit in hits]
            if all(pair):
                for (number, start, _), [span] in zip(hits, pair):
                    for other in spans:
                        if other.line == number and start <= other.start < span.end:
                            picked.discard(id(other))
                prefix = (pair[0][0].text.rstrip(" …"), pair[1][0].text)
            else:
                problems.append(f"{page}:{hits[0][0]}：路徑換法的兩個錨點之後要各有一段 inline code")

    for snippet, _ in NOT_WINDOWS.get(page, []):
        hits = occurrences(prose, snippet)
        covered = [
            span
            for number, start in hits
            for span in spans
            if span.line == number and start <= span.start < start + len(snippet) and id(span) in picked
        ]
        if len(hits) != 1 or not covered:
            problems.append(f"{page}：NOT_WINDOWS 的「{snippet}」對不到頁面上的命令（找到 {len(hits)} 處）；請更新清單")
        for span in covered:
            picked.discard(id(span))

    items = [(command.line, 0, [command]) for command in lines if command not in replaced]
    items += [(command.line, 0, replaced[command]) for command in lines if command in replaced]
    items += [(span.line, span.start, [Command(span.line, span.text)]) for span in spans if id(span) in picked]
    commands = [command for _, _, group in sorted(items, key=lambda item: item[:2]) for command in group]
    if prefix:
        source, target = prefix
        commands = [
            Command(command.line, target + command.text[len(source) :])
            if command.text == source or command.text.startswith(source + " ")
            else command
            for command in commands
        ]
    return commands, problems


def workflow_copies():
    """(副本, 問題)：workflow 每個 run 步驟裡 Invoke-Page／Skip-Page 依序列出的命令。"""
    copies, problems = [], []
    for job in load_workflow()["jobs"].values():
        for step in job.get("steps", []):
            script = step.get("run", "")
            matches = list(CALL.finditer(script))
            if len(matches) != len(CALL_START.findall(script)):
                problems.append(f"{WORKFLOW}：步驟「{step.get('name', '')}」有 Invoke-Page／Skip-Page 的寫法認不出來")
            for match in matches:
                pages = tuple(match["pages"].split(","))
                unknown = [page for page in pages if page not in PAGES]
                if unknown:
                    problems.append(f"{WORKFLOW}：步驟「{step.get('name', '')}」的 -Pages 有不認得的頁面 {unknown}")
                if match["verb"] == "Skip-Page" and not (match["reason"] or "").strip():
                    problems.append(
                        f"{WORKFLOW}：步驟「{step.get('name', '')}」的 Skip-Page 沒有 -Reason；"
                        "記錄裡要印出不跑的理由，讀記錄的人才分得出「這裡沒跑」與「前面已經跑過」"
                    )
                if match["verb"] == "Invoke-Page" and match["reason"] is not None:
                    problems.append(f"{WORKFLOW}：步驟「{step.get('name', '')}」的 Invoke-Page 不收 -Reason")
                body = match["block"].split("\n") if match["block"] is not None else [match["line"].replace("''", "'")]
                copies.extend(
                    Copy(step.get("name", ""), match["verb"], pages, line.strip()) for line in body if line.strip()
                )
    return copies, problems


def compare(page, expected, actual):
    """逐條比對頁面推出的命令與 workflow 標了這頁的命令；回傳指出頁面行號的問題。"""
    matcher = difflib.SequenceMatcher(a=[c.text for c in expected], b=[c.text for c in actual], autojunk=False)
    problems = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        paired = min(i2 - i1, j2 - j1) if tag == "replace" else 0
        for offset in range(i2 - i1):
            command = expected[i1 + offset]
            if offset < paired:
                copy = actual[j1 + offset]
                problems.append(
                    f"{page}:{command.line}：頁面的 Windows 命令是 `{command.text}`，"
                    f"workflow 的對應位置（步驟「{copy.step}」的 {copy.verb}）是 `{copy.text}`"
                )
            else:
                problems.append(
                    f"{page}:{command.line}：頁面的 Windows 命令 `{command.text}` 在 workflow 沒有對應的 Invoke-Page 或 Skip-Page"
                )
        last = i1 + paired - 1
        near = expected[last].line if last >= 0 else (expected[0].line if expected else 1)
        for copy in actual[j1 + paired : j2]:
            problems.append(
                f"{page}:{near}：workflow 步驟「{copy.step}」的 {copy.verb} 列了 `{copy.text}`，頁面上找不到；"
                f"它在頁面上應接在第 {near} 行的命令之後"
            )
    return problems


@pytest.mark.parametrize("page", PAGES)
def test_page_windows_commands_match_workflow(page):
    commands, problems = windows_commands(page)
    copies, _ = workflow_copies()
    problems += compare(page, commands, [copy for copy in copies if page in copy.pages])
    assert not problems, (
        "頁面與 workflow 的 Windows 命令要一起改；不跑的命令寫進 Skip-Page，-Reason 寫印進記錄的簡短理由，"
        "註解寫完整理由。\n" + "\n".join(problems)
    )


def test_windows_clone_comes_after_global_long_paths():
    """頁面教 Windows 讀者 git clone 之前先執行 LONG_PATHS，其他命令不另外設 core.longpaths（例如 clone 的 -c）；
    workflow 也在 clone 之前用 Invoke-Page 執行它。"""
    problems, clones = [], []
    for page in PAGES:
        commands, _ = windows_commands(page)
        enabled = False
        for command in commands:
            if command.text == LONG_PATHS:
                enabled = True
            elif "core.longpaths" in command.text:
                problems.append(f"{page}:{command.line}：`{command.text}` 另外設了 core.longpaths")
            elif re.match(r"git\s+clone\b", command.text):
                clones.append(command)
                if not enabled:
                    problems.append(f"{page}:{command.line}：`{command.text}` 之前沒有 `{LONG_PATHS}`")
    # workflow 那一側：比對只看命令文字，寫進 Skip-Page 也對得上，所以另外核對它真的在 clone 之前執行。
    runs = [copy.text for copy in workflow_copies()[0] if copy.verb == "Invoke-Page"]
    first_clone = next((index for index, text in enumerate(runs) if re.match(r"git\s+clone\b", text)), len(runs))
    if LONG_PATHS not in runs[:first_clone]:
        problems.append(f"{WORKFLOW}：要先用 Invoke-Page 執行 `{LONG_PATHS}`，再用 Invoke-Page 執行 git clone")
    headline = (
        f"Windows 讀者要在 git clone 之前先執行 `{LONG_PATHS}`，這個全域設定讓之後的 clone 與切換版本都允許長路徑；"
        "在 windows-latest 上，clone 帶 -c core.longpaths=true 時，取出檔案仍報 Filename too long。"
    )
    assert not problems, headline + "\n" + "\n".join(problems)
    assert clones, "頁面上找不到 Windows 讀者的 git clone；取得專案的寫法改了，這個測試要跟著改"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "```text\nRemove-Item -Recurse -Force .venv\n```\n", "x.md:2：標成「text」", id="text 區塊裡的 cmdlet"
        ),
        pytest.param(
            "```output\ngit clone https://github.com/birdhackor/tiny-perceptron-vlm.git\n```\n",
            "x.md:2：標成「output」",
            id="output 區塊裡的 git",
        ),
        pytest.param(
            "```plaintext\n  .\\.venv\\Scripts\\python.exe scripts/check_env.py\n```\n",
            "x.md:2：標成「plaintext」",
            id="plaintext 區塊裡縮排的 Windows 路徑",
        ),
        pytest.param('```txt\n$env:GIT_LFS_SKIP_SMUDGE="1"\n```\n', "x.md:2：標成「txt」", id="txt 區塊裡的環境變數"),
        pytest.param(
            "```text\ncourse/chapters/     正文\nuv sync --frozen --extra cpu --group notebook\n```\n",
            "x.md:3：標成「text」",
            id="text 區塊第二行才是命令",
        ),
        pytest.param(
            "或先執行 <code>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned</code>。\n",
            "x.md:1：fence 外有 HTML",
            id="句子裡的 code 元素",
        ),
        pytest.param("<pre>\ngit lfs install --local\n</pre>\n", "x.md:1：fence 外有 HTML", id="跨行的 pre 元素"),
        pytest.param("說明 <CODE>uv sync</CODE> 的用途。\n", "x.md:1：fence 外有 HTML", id="大寫的 CODE 元素"),
        pytest.param("```text\ncourse/chapters/     教材正文與短程式\n```\n", None, id="專案結構不是命令"),
        pytest.param("HTML 的 `<code>` 標籤只是舉例。\n", None, id="反引號裡的 code 字樣"),
    ],
)
def test_commands_in_plain_blocks_or_html_code_are_reported(monkeypatch, text, expected):
    """寫在純文字區塊或 HTML code 元素裡的命令，比對時看不到，所以要報錯並指出行號；不是命令的內容不報。"""
    monkeypatch.setitem(globals(), "read", lambda relative: text)
    _, problems = windows_commands("x.md")
    if expected is None:
        assert not problems, problems
    else:
        assert any(problem.startswith(expected) for problem in problems), problems


def test_workflow_page_calls_are_recognized():
    copies, problems = workflow_copies()
    assert copies, f"{WORKFLOW} 沒有任何 Invoke-Page／Skip-Page"
    assert not problems, "\n".join(problems)


def test_page_code_anchors_point_at_python_blocks():
    found, problems = page_code_anchors(), []
    assert found, f"{WORKFLOW} 沒有 Get-PageCode；頁面上要讀者複製的 Python 程式改由誰取？"
    for page, anchor in found:
        text = read(page)
        blocks, _ = split_page(text)
        # Get-PageCode 在全頁（含程式區塊）找第一個含錨點的行，所以這裡也看全頁。
        hits = [number for number, line in enumerate(text.split("\n"), 1) if anchor in line]
        following = [block for block in blocks if hits and block[0] > hits[0]]
        if len(hits) != 1:
            problems.append(f"{page}：Get-PageCode 的錨點「{anchor}」出現在 {len(hits)} 行，要剛好一行")
        elif not following or following[0][1] != "python":
            problems.append(f"{page}:{hits[0]}：「{anchor}」之後的第一個程式區塊不是 python")
    assert not problems, "\n".join(problems)


def test_kernel_checks_find_their_page_sentences():
    """KERNEL_CHECKS 從頁面取 W.1 改字練習的原句、新句、期望輸出，與 W.6 第一行輸出的期望值：每一句在頁面上剛好一處，
    原輸入行在 Notebook 的程式格裡剛好一行。頁面的句子或 Notebook 那一行改到對不上時，三種作業系統的 pytest 都會在這裡
    失敗，不必等 Windows CI。"""
    checks = kernel_checks()
    page = read(checks["PAGE"])
    notebook = json.loads(read(checks["NOTEBOOK"]))
    try:
        old, new, _ = checks["exercise"](page)
        checks["replace_input_line"](notebook, old, new)
        checks["one_sentence"](checks["W6_FIRST_LINE"], page, "the first W.6 output line")
    except SystemExit as error:
        pytest.fail(f"KERNEL_CHECKS 在頁面或 Notebook 上找不到它要的句子或那一行：{error}")


def test_workflow_triggers_and_shells():
    document = load_workflow()
    triggers = document["on"]
    assert "workflow_dispatch" in triggers
    for event in ("push", "pull_request"):
        assert "paths-ignore" not in triggers[event], f"{event} 不用 paths-ignore；排除寫成 paths 裡 '**' 之後的 '!'"
    push, pull = triggers["push"]["paths"], triggers["pull_request"]["paths"]
    assert push == pull, f"push 與 pull_request 的 paths 要完全相同：\npush：{push}\npull_request：{pull}"
    assert push[:1] == ["**"], f"paths 第一條要是 '**'，改到任何檔都跑，之後只用 '!' 排除不讀的樹；現在是 {push[:1]}"
    malformed = [path for path in push[1:] if not TREE_EXCLUSION.fullmatch(path)]
    assert not malformed, f"'**' 之後每一條都要寫成 '!<樹>/**'，一次排除一整棵樹：{malformed}"
    missing = [tree for tree in excluded_trees() if not (ROOT / tree).is_dir()]
    assert not missing, f"這些排除的樹在 repo 裡不存在，排除與旁邊的理由都已經沒有對象：{missing}"
    for name, job in document["jobs"].items():
        assert job["runs-on"] == "windows-latest", f"{name} 要在 windows-latest 跑"
        shells = job["strategy"]["matrix"]["shell"]
        assert shells == ["powershell", "pwsh"], (
            f"{name} 要用 matrix 跑 Windows PowerShell 5.1 與 PowerShell 7：{shells}"
        )
        assert job["defaults"]["run"]["shell"] == "${{ matrix.shell }}", f"{name} 的 run 步驟要照 matrix 選 shell"
        overridden = [step.get("name", "") for step in job["steps"] if "run" in step and "shell" in step]
        assert not overridden, f"這些步驟改用別的 shell，沒有照 matrix 跑兩種 PowerShell：{overridden}"


def test_excluded_trees_are_not_named_where_the_workflow_reads():
    """排除的樹的名字以一整串出現（各段在檔案文字裡以單一個 / 或反斜線相連，前後不接路徑字元）時報錯。掃的是四頁的程式
    區塊、workflow 的 env 與每一步的 run:、env、KERNEL_CHECKS 執行的 Notebook 程式格，以及 python_run_by_workflow 列的檔；
    on: 區塊本身寫著排除樣式，不掃。不是一整串寫出的路徑這裡看不到，見檔頭〈不涵蓋〉。"""
    document = load_workflow()
    sources = []
    for page in PAGES:
        blocks, _ = split_page(read(page))
        sources += [(f"{page}:{start} 的程式區塊", "\n".join(line for _, line in body)) for start, _, body in blocks]
    sources += [(f"{WORKFLOW} 的 env {key}", value) for key, value in (document.get("env") or {}).items()]
    for job in document["jobs"].values():
        for step in job.get("steps", []):
            where = f"{WORKFLOW} 步驟「{step.get('name', '')}」"
            if "run" in step:
                sources.append((f"{where}的 run", step["run"]))
            sources += [(f"{where}的 env {key}", value) for key, value in (step.get("env") or {}).items()]
    notebook = kernel_checks()["NOTEBOOK"]
    sources += [
        (f"{notebook} 的程式格 {number}", "".join(cell["source"]))
        for number, cell in enumerate(json.loads(read(notebook))["cells"])
        if cell["cell_type"] == "code"
    ]
    sources += [(str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")) for path in python_run_by_workflow()]
    problems = []
    for tree in excluded_trees():
        # 用 / 或 \ 分隔都算；前後還接著路徑字元的不算，例如名字以這棵樹開頭的另一棵樹。
        written = re.compile(r"(?<![\w.-])" + r"[/\\]".join(map(re.escape, tree.split("/"))) + r"(?![\w.-])")
        problems += [f"{where}：寫到排除的樹 {tree}" for where, text in sources if written.search(text)]
    assert not problems, (
        "workflow 會讀的地方寫到觸發條件排除的樹；這些樹改了，Windows CI 不會跑。拿掉那幾條排除，或改掉這些路徑：\n"
        + "\n".join(problems)
    )


def test_known_workflow_reads_are_outside_excluded_trees():
    """workflow 檔本身、四頁、KERNEL_CHECKS 執行的 Notebook，以及 python_run_by_workflow 列的檔，這些已知會讀的檔不在
    排除的樹底下。比對的是 repo 內的 POSIX 路徑是否以「<樹>/」開頭；這依賴 TREE_EXCLUSION 把每條排除限定成 '!<樹>/**'，
    寫法不符的排除不在 excluded_trees() 裡，由 test_workflow_triggers_and_shells 報錯。排除的樹裡另有哪些檔被讀，這裡看
    不到。"""
    known = [WORKFLOW, *PAGES, kernel_checks()["NOTEBOOK"]]
    # 用 as_posix()，不用 str()：ci.yml 在 windows-latest 也跑這個測試，str() 在 Windows 用反斜線，比對永遠不會命中。
    known += [path.relative_to(ROOT).as_posix() for path in python_run_by_workflow()]
    problems = [
        f"{path} 在排除的樹 {tree} 底下" for tree in excluded_trees() for path in known if path.startswith(tree + "/")
    ]
    assert not problems, (
        "觸發條件排除了 workflow 已知會讀的檔所在的樹；只改到這些檔時，Windows CI 不會跑。拿掉那幾條排除：\n"
        + "\n".join(problems)
    )
