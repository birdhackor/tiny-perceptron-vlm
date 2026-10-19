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
相同、第一條是 '**'、其餘每條都是 '!<樹>/**'（<樹> 的寫法見 TREE_EXCLUSION）而且那棵樹在 repo 裡。排除的樹的名字，不以
一整串（各段在檔案文字裡以單一個 / 或反斜線相連）出現在四頁的程式區塊、workflow 的 env 與每一步的 run:、env、
KERNEL_CHECKS 執行的 Notebook 程式格，以及 python_run_by_workflow 列的檔裡；on: 區塊本身寫著排除樣式，不掃。workflow 檔
本身、四頁、KERNEL_CHECKS 執行的 Notebook 與 python_run_by_workflow 列的檔，repo 內的 POSIX 路徑都不以任何排除的樹加 /
開頭。workflow 每一處 `# 頁面說：` 的引文，以及下面〈不涵蓋〉裡 not_covered_page_items 收的各項中的「」引文，都要在頁面上找得到，
比對方式見 test_page_quotes_appear_on_the_pages。
這些比對裡，引文往頁面查的函式（test_page_quotes_appear_on_the_pages 直接或間接呼叫的，read 除外）與 KERNEL_CHECKS 裡
main、run 以外的函式是單向比對：抽少了或挑錯了，比對仍會通過。所以它們的 docstring 與錯誤訊息宣稱的每個結果，都要有
一例餵假資料的案例，放在以該函式命名的參數化測試裡。推出頁面命令與取出 workflow 副本的函式不逐個結果補這種案例：compare 對
頁面推出的命令與 workflow 的副本做序列比對，一邊漏抽或多抽，另一邊多出或缺少的那條就會報出；兩邊同時看不到的寫法比對
不出來，〈不涵蓋〉列了其中一些。下面講觸發條件讀檔集合的兩道測試，用到的函式也是單向，同樣不逐個結果補這種案例，理由見
〈不涵蓋〉裡講排除安不安全的那句。

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
排除的樹的名字不以一整串出現在上面列的地方；
test_workflow_file_pages_notebook_and_python_run_by_workflow_are_outside_excluded_trees 核對 workflow 檔本身、四頁、
KERNEL_CHECKS 執行的 Notebook 與 python_run_by_workflow 列的檔不在排除的樹底下。(a) 實際開啟的檔、(b) 與 (c) 裡不是一整串
寫出的路徑，以及下面〈不涵蓋〉列的看不到的地方，要在新增或改動 Invoke-Page、Get-PageCode 或 KERNEL_CHECKS，或 runner 那一側
新讀了檔時，照三種來源重推一次，確認讀到的檔都不在排除的樹裡。

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
條件（「看到…就表示」），以及寫「希望」的答案；描述命令做什麼、工具自己核對的事、給維護者的流程與一般建議不算。
`# 頁面說：` 的引文裡每個子句，要嘛有核對程式比對它，要嘛緊接的註解寫明不核或只核一部分；其餘用「……」略去。只核一部分
時，要用收窄句說出核了哪一半，不能只靠略去把它藏起來。前提子句也算子句；workflow 照頁面做了的事（命令由命令的釘選核對）
算核到。只說同一則引文裡某個結果的意思、理由或用途，沒有引入新的可核對事實的子句，跟著那個結果算，寫在它前面或後面都
一樣；引入新事實（另一個輸出、另一個裝置、另一個檔）就是另一個子句，照同樣的規則核對或收窄。「緊接」指收窄句與引文在同一段
註解裡、中間沒有程式行，前後都可以。引文從哪裡開始、到哪裡結束見 workflow_quotes；頁面句子的句尾「。」在「」裡時，整句
引文讀不到句尾，引文以「……」收尾。含「。」的 inline code，引文要嘛把那段寫成 `…`，要嘛留著反引號，後者只限用一個反引號
包住的 code，原因見 cut_at_full_stop。引文和〈不涵蓋〉裡 not_covered_page_items 收的各項中的「」引文，由
test_page_quotes_appear_on_the_pages 核對還在頁面上。練習的改法或期望值只寫在句子裡時，要核對就得把句意寫死成 workflow
裡的程式，頁面改了練習，那份副本也不會跟著變，所以不核：
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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/windows-student-commands.yml"
PAGES = ("README.md", "README_en.md", "course/first-steps.md", "course/training.md")
# 觸發條件裡排除一棵樹的寫法：'!'、樹的路徑、'/**'。路徑以 / 分段，每一段都不能是空的、. 或 ..，也不能含 GitHub
# 路徑樣式的特殊字元（* ? + [ ] !）。含特殊字元時，排除的就不只那一棵樹；./x、a//b、a/../b 這類寫法，(ROOT / 樹).is_dir()
# 照樣通過，兩道排除測試卻都看不到，GitHub 是否先把它們正規化又沒有文件可查。
TREE_SEGMENT = r"(?!\.\.?/)[^/*?+\[\]!]+"
TREE_EXCLUSION = re.compile(rf"!(?P<tree>{TREE_SEGMENT}(?:/{TREE_SEGMENT})*)/\*\*")
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


@pytest.mark.parametrize(
    ("text", "expected", "message"),
    [
        pytest.param("執行後第一行輸出 `1.0 4.0 -4.0`，確認代價 4、梯度 -4。", "1.0 4.0 -4.0", None, id="剛好一句"),
        pytest.param("執行後輸出 `1.0 4.0 -4.0`。", None, "appears 0 times", id="找不到"),
        pytest.param(
            "第一行輸出 `1.0`，確認代價 4。改了之後第一行輸出 `2.0`，確認代價 1。", None, "appears 2 times", id="兩句"
        ),
    ],
)
def test_one_sentence_returns_the_code_of_the_only_match_or_fails(text, expected, message):
    """拿 KERNEL_CHECKS 的 W6_FIRST_LINE 與假的頁面文字，逐一核對 one_sentence：剛好一句時回傳那句的 inline code；找不到或
    有兩句時以 SystemExit 失敗，訊息寫出找到幾句。"""
    checks = kernel_checks()
    arguments = (checks["W6_FIRST_LINE"], text, "the first W.6 output line")
    if message is None:
        assert checks["one_sentence"](*arguments) == expected
    else:
        with pytest.raises(SystemExit, match=message):
            checks["one_sentence"](*arguments)


@pytest.mark.parametrize(
    ("text", "expected", "message"),
    [
        pytest.param(
            '再把輸入那行 `text = "狗看鳥"` 改成 `text = "鳥看狗"`，重啟並全部執行。'
            "程式最後的文字輸出應為 `鳥看狗`，而非舊句。",
            ('text = "狗看鳥"', 'text = "鳥看狗"', "鳥看狗"),
            None,
            id="依序回傳原輸入行、新輸入行與期望輸出",
        ),
        pytest.param(
            '再把輸入那行 `text = "狗看鳥"` 改成 `text = "狗看鳥"`，重啟並全部執行。'
            "程式最後的文字輸出應為 `狗看鳥`，而非舊句。",
            None,
            "replaces an input line with the same line",
            id="新舊輸入行相同",
        ),
    ],
)
def test_exercise_returns_the_old_and_new_lines_and_the_output_or_fails_on_the_same_line(text, expected, message):
    """拿假的頁面文字逐一核對 KERNEL_CHECKS 的 exercise：依序回傳 (原輸入行, 新輸入行, 期望輸出)；新舊輸入行相同時以
    SystemExit 失敗。"""
    checks = kernel_checks()
    if message is None:
        assert checks["exercise"](text) == expected
    else:
        with pytest.raises(SystemExit, match=message):
            checks["exercise"](text)


# replace_input_line 案例裡的原輸入行與新輸入行。
OLD_LINE, NEW_LINE = 'text = "狗看鳥"', 'text = "鳥看狗"'


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        pytest.param(
            [
                {"cell_type": "markdown", "source": ["# 1.1\n"]},
                {"cell_type": "code", "source": ["import tiny_perceptron\n"]},
                {"cell_type": "code", "source": [OLD_LINE + "\n", "print(text)"]},
            ],
            (2, [["# 1.1\n"], ["import tiny_perceptron\n"], NEW_LINE + "\nprint(text)"]),
            id="json 讀入的 Notebook：source 是逐行的清單",
        ),
        pytest.param(
            [
                {"cell_type": "markdown", "source": "# 1.1"},
                {"cell_type": "code", "source": "import tiny_perceptron"},
                {"cell_type": "code", "source": OLD_LINE + "\nprint(text)"},
            ],
            (2, ["# 1.1", "import tiny_perceptron", NEW_LINE + "\nprint(text)"]),
            id="nbformat 讀入的 Notebook：source 是字串",
        ),
        pytest.param(
            [
                {"cell_type": "code", "source": "# 原本是 " + OLD_LINE + "\nif True:\n    " + OLD_LINE},
                {"cell_type": "code", "source": OLD_LINE + "\nprint(text)"},
            ],
            (1, ["# 原本是 " + OLD_LINE + "\nif True:\n    " + OLD_LINE, NEW_LINE + "\nprint(text)"]),
            id="整行相等才算，只含有它或縮排過的行不算",
        ),
        pytest.param(
            [{"cell_type": "markdown", "source": OLD_LINE}, {"cell_type": "code", "source": OLD_LINE}],
            (1, [OLD_LINE, NEW_LINE]),
            id="Markdown 格裡的同一行不算",
        ),
        pytest.param([{"cell_type": "code", "source": 'text = "貓"'}], "0 code lines", id="找不到"),
        pytest.param(
            [{"cell_type": "code", "source": OLD_LINE + "\nprint(text)"}, {"cell_type": "code", "source": OLD_LINE}],
            "2 code lines",
            id="兩行相等",
        ),
    ],
)
def test_replace_input_line_replaces_the_only_equal_code_line_or_fails(cells, expected):
    """拿假的 Notebook 逐一核對 KERNEL_CHECKS 的 replace_input_line：程式格裡恰好一行等於原輸入行時，在傳進來的 Notebook
    裡把那一行換成新輸入行、其餘不動，回傳那一格在 cells 裡的索引（Markdown 格也算進索引）；source 是 json 讀入的清單或
    nbformat 讀入的字串都一樣。Markdown 格裡的、只含有它或縮排過的行不算；找不到或兩行相等時以 SystemExit 失敗，訊息寫出
    有幾行。"""
    checks = kernel_checks()
    notebook = {"cells": deepcopy(cells)}
    if isinstance(expected, str):
        with pytest.raises(SystemExit, match=expected):
            checks["replace_input_line"](notebook, OLD_LINE, NEW_LINE)
    else:
        index, sources = expected
        assert checks["replace_input_line"](notebook, OLD_LINE, NEW_LINE) == index
        assert [cell["source"] for cell in notebook["cells"]] == sources


@pytest.mark.parametrize(
    ("outputs", "expected", "message"),
    [
        pytest.param(
            [
                {"output_type": "stream", "name": "stdout", "text": "字表\n編號："},
                {"output_type": "stream", "name": "stdout", "text": "1 2 3\n鳥看狗\n"},
            ],
            ["字表", "編號：1 2 3", "鳥看狗"],
            None,
            id="一行分在兩段 stdout",
        ),
        pytest.param(
            [
                {"output_type": "stream", "name": "stdout", "text": "甲\n"},
                {"output_type": "stream", "name": "stderr", "text": "UserWarning: 警告\n"},
                {"output_type": "stream", "name": "stdout", "text": "乙\n"},
            ],
            ["甲", "乙"],
            None,
            id="stderr 不算",
        ),
        pytest.param(
            [
                {"output_type": "stream", "name": "stdout", "text": "甲\n"},
                {"output_type": "execute_result", "execution_count": 1, "data": {"text/plain": "3"}, "metadata": {}},
            ],
            None,
            "execute_result",
            id="還有運算式的值",
        ),
        pytest.param(
            [
                {"output_type": "stream", "name": "stdout", "text": "甲\n"},
                {"output_type": "display_data", "data": {"image/png": "iVBORw0KGgo="}, "metadata": {}},
            ],
            None,
            "display_data",
            id="還有圖",
        ),
        pytest.param(
            [
                {"output_type": "stream", "name": "stdout", "text": "甲\n"},
                {"output_type": "error", "ename": "ValueError", "evalue": "x", "traceback": []},
            ],
            None,
            "error",
            id="還有錯誤",
        ),
    ],
)
def test_printed_lines_join_stdout_before_splitting_or_fail_on_other_outputs(outputs, expected, message):
    """拿和 KERNEL_CHECKS 執行後同樣形狀的輸出（text 是字串；它的 NotebookClient 沒有開 coalesce_streams，每個 stream
    訊息各一筆）逐一核對 printed_lines：各段 stdout 依序接起來再分行，stderr 不算；還有運算式的值、圖或錯誤時以 SystemExit
    失敗，訊息寫出是哪一種。"""
    checks = kernel_checks()
    if message is None:
        assert checks["printed_lines"]({"outputs": outputs}) == expected
    else:
        with pytest.raises(SystemExit, match=message):
            checks["printed_lines"]({"outputs": outputs})


# workflow 引頁面原話的註解記號；引文從記號之後開始，範圍見 workflow_quotes。
PAGE_QUOTE = "# 頁面說："
# 同一行裡的 Markdown 行內連結 `[文字](網址)`（含圖片）：比對引文與頁面之前，只留方括號裡的文字。參考式連結
# `[文字][標籤]` 與自動連結 `<網址>` 原樣留著。
LINK = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)")


def cut_at_full_stop(text):
    """text 從開頭到反引號與「」以外的第一個「。」為止（含這個「。」）；沒有這種「。」時回傳 None。反引號一個一個算，每遇到
    一個就在 code 裡外切換，所以用兩個反引號包起來的 inline code，這裡當成 code 外的字。code 外的「」可以巢狀；code 裡的「」
    不算，前面沒有 `「` 可以對應的 `」` 也不算。反引號逐個切換、沒關的 code 與 `「` 一直算到 text 的結尾，是因為
    workflow_quotes 一行一行接起註解、每接一行就截一次，前一行可能停在 code 或「」裡；先成對遮掉 code（例如用 SPAN）或「」
    再找「。」，前一行沒關的那段裡的「。」就會在接上下一行之前被當成句尾。這裡只看 text 自己的字：引文引到含「。」的
    inline code 卻拿掉了反引號時，分不出那個「。」在 code 裡，引文會在那裡被截斷，後半不受
    test_page_quotes_appear_on_the_pages 核對，所以這種引文要留反引號（只限用一個反引號包住的 code），或把那段寫成 `…`。
    引文從 inline code 或「」的中間開始時，同樣分不出開頭那段的「。」在 code 或「」裡。"""
    code, depth = False, 0
    for position, char in enumerate(text):
        if char == "`":
            code = not code
        elif code:
            continue
        elif char == "「":
            depth += 1
        elif char == "」":
            depth = max(depth - 1, 0)
        elif char == "。" and depth == 0:
            return text[: position + 1]
    return None


def workflow_quotes():
    """workflow 每一處 `# 頁面說：` 的 (行號, 引文, 問題)。引文從記號之後開始，到 cut_at_full_stop 截到的「。」為止（反引號與
    「」以外的第一個「。」，含這個「。」）；在那之前先碰到以「……」結尾的行，就到那一行的結尾為止；兩者都還沒出現時，接著讀
    下一行註解（去掉 # 與前後空白）。下一行不是註解、或又是一處 `# 頁面說：` 時，引文沒有結束，問題寫明這件事。同一行在那個
    「。」之後的字是旁邊的說明，不算引文：要引兩句就寫兩行；頁面的句子沒在引到的地方結束時，引文以「……」收尾；表格的儲存格
    之間用「……」隔開。"""
    lines = read(WORKFLOW).split("\n")
    found = []
    for index, line in enumerate(lines):
        if PAGE_QUOTE not in line:
            continue
        quote, following, problem = line.split(PAGE_QUOTE, 1)[1].strip(), index + 1, None
        while cut_at_full_stop(quote) is None and not quote.endswith("……"):
            comment = lines[following].strip() if following < len(lines) else ""
            if not comment.startswith("#") or PAGE_QUOTE in comment:
                problem = "引文要在反引號與「」以外的第一個「。」或以「……」結尾的那一行結束"
                break
            quote += comment[1:].strip()
            following += 1
        found.append((index + 1, cut_at_full_stop(quote) or quote, problem))
    return found


def flatten(text):
    """比對用的寫法：同一行裡的 Markdown 行內連結（寫法見 LINK）只留方括號裡的文字，再去掉 `**`、反引號與所有空白。"""
    return re.sub(r"\s+", "", LINK.sub(r"\1", text).replace("**", "").replace("`", ""))


def in_order(quote, flat_page):
    """引文經 flatten 之後以「…」切段（一個或連續幾個；`…` 去掉反引號後也是），各段依序出現在 flat_page 裡時為真；
    切完一段都沒有時為假。"""
    parts = [part for part in re.split(r"…+", flatten(quote)) if part]
    if not parts:
        return False
    position = 0
    for part in parts:
        position = flat_page.find(part, position)
        if position < 0:
            return False
        position += len(part)
    return True


def bracketed(text):
    """text 裡最外層「」的內容，裡面可以再有「」；括號不成對時回傳 None。"""
    found, depth, start = [], 0, 0
    for position, char in enumerate(text):
        if char == "「":
            depth += 1
            if depth == 1:
                start = position + 1
        elif char == "」":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0:
                found.append(text[start:position])
    return found if depth == 0 else None


def not_covered_page_items():
    """檔頭「不涵蓋：」那一行之後，以「- 」開頭、用空白切開後第二個詞是四頁之一的每一行：(頁, 那一行)。頁名後面直接接
    標點的行（例如「- course/training.md：」）不算，也不報錯。每一項都寫成一行，這裡不讀接續行。"""
    lines = (__doc__ or "").split("\n")
    if "不涵蓋：" not in lines:
        return []
    items = [line for line in lines[lines.index("不涵蓋：") + 1 :] if line.startswith("- ")]
    return [(words[1], item) for item in items if len(words := item.split()) > 1 and words[1] in PAGES]


def test_page_quotes_appear_on_the_pages():
    """workflow 每一處 `# 頁面說：` 的引文（範圍見 workflow_quotes）要在四頁之一找得到；not_covered_page_items 從檔頭
    〈不涵蓋〉收的每一項，其中最外層「」裡的每段引文要在那一項的頁面找得到。比對前，引文與頁面都經 flatten，引文再以「…」切段，
    各段要在同一頁依序找得到。引文沒有照 workflow_quotes 的規則結束、某一項的「」不成對、workflow 沒有任何 `# 頁面說：`，
    或 not_covered_page_items 一項都沒收到時，也會失敗。這裡只核引文還在頁面上：引文旁的核對是否仍和頁面句子一樣強、
    〈不涵蓋〉那一項的理由是否仍成立，都不在這裡核；頁面留著同樣的字而意思改變時，這裡不會失敗。"""
    pages = {page: flatten(read(page)) for page in PAGES}
    quotes, malformed, missing = workflow_quotes(), [], []
    assert quotes, f"{WORKFLOW} 沒有任何 `# 頁面說：`"
    for number, quote, problem in quotes:
        if problem:
            malformed.append(f"{WORKFLOW}:{number}：{problem}：{quote}")
        elif not any(in_order(quote, text) for text in pages.values()):
            missing.append(f"{WORKFLOW}:{number}：「{quote}」不在四頁的任何一頁上")
    items = not_covered_page_items()
    assert items, "檔頭找不到「不涵蓋：」那一行，或它底下沒有 not_covered_page_items 收得到的項目"
    for page, item in items:
        found = bracketed(item)
        if found is None:
            malformed.append(f"檔頭〈不涵蓋〉這一項的「」不成對：{item}")
        missing += [
            f"檔頭〈不涵蓋〉：「{quote}」不在 {page} 上" for quote in found or [] if not in_order(quote, pages[page])
        ]
    sections = []
    if malformed:
        sections.append("這些引文的寫法不合規則，還沒有和頁面比對；照每一行寫的問題改寫法：\n" + "\n".join(malformed))
    if missing:
        sections.append(
            "頁面上找不到這些引文。頁面改了句子時，workflow 的引文要先重新判斷旁邊的核對是否仍和頁面的新句子一樣強，"
            "再照檔頭的引文規則改引文，而不是直接把新句子整句抄回來；〈不涵蓋〉的項目要先重新判斷它寫的理由是否仍成立，"
            "再改「」裡的引文。\n" + "\n".join(missing)
        )
    assert not sections, "\n\n".join(sections)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "安裝 Git 後在終端機執行 git --version，應顯示版本號。終端機是輸入指令的視窗。",
            "安裝 Git 後在終端機執行 git --version，應顯示版本號。",
            id="一般句子",
        ),
        pytest.param("第三行啟動 JupyterLab，終端機會提供瀏覽器網址", None, id="沒有「。」"),
        pytest.param("第三行啟動 JupyterLab……", None, id="以「……」結尾的片段"),
        pytest.param(
            '`text = "鳥看狗，狗看鳥。"` 是改後的輸入那行。旁邊的說明。',
            '`text = "鳥看狗，狗看鳥。"` 是改後的輸入那行。',
            id="反引號裡的「。」在開頭",
        ),
        pytest.param(
            "程式最後的文字輸出應為 `鳥看狗，狗看鳥。`，而非舊句；這就是要核對的還原結果。旁邊的說明。",
            "程式最後的文字輸出應為 `鳥看狗，狗看鳥。`，而非舊句；這就是要核對的還原結果。",
            id="反引號裡的「。」在中間",
        ),
        pytest.param(
            "程式最後的文字輸出應為 `鳥看狗，狗看鳥。`。旁邊的說明。",
            "程式最後的文字輸出應為 `鳥看狗，狗看鳥。`。",
            id="反引號裡的「。」緊接在句尾之前",
        ),
        pytest.param("程式最後的文字輸出應為 `鳥看狗，狗看鳥。`", None, id="反引號裡的「。」在結尾"),
        pytest.param("程式最後的文字輸出應為 `鳥看狗，狗看鳥。`……", None, id="以「……」結尾、反引號裡有「。」"),
        pytest.param("先輸入 `a`。再輸入 `b`。", "先輸入 `a`。", id="兩段反引號之間的「。」"),
        pytest.param(
            "「先執行一次。再改輸入那行」是練習的兩步。旁邊的說明。",
            "「先執行一次。再改輸入那行」是練習的兩步。",
            id="「」裡的「。」在開頭",
        ),
        pytest.param(
            "例如「顏色=紅；形狀=圓。」這種短文，共十二篇。旁邊的說明。",
            "例如「顏色=紅；形狀=圓。」這種短文，共十二篇。",
            id="「」裡的「。」在中間",
        ),
        pytest.param(
            "預設資料是短文，例如「顏色=紅；形狀=圓。」。旁邊的說明。",
            "預設資料是短文，例如「顏色=紅；形狀=圓。」。",
            id="「」裡的「。」緊接在句尾之前",
        ),
        pytest.param("預設資料是短文，例如「顏色=紅；形狀=圓。」", None, id="「」裡的「。」在結尾"),
        pytest.param("先讀「甲」。再讀「乙」。", "先讀「甲」。", id="兩組「」之間的「。」"),
        pytest.param(
            "先讀「題目「2+3」等於多少。」，再作答。旁邊的說明。",
            "先讀「題目「2+3」等於多少。」，再作答。",
            id="巢狀「」",
        ),
        pytest.param(
            "在「重啟。」之後輸入 `run。`，再按 Enter。旁邊的說明。",
            "在「重啟。」之後輸入 `run。`，再按 Enter。",
            id="反引號與「」都有",
        ),
        pytest.param("輸入 `「。` 之後按 Enter。旁邊的說明。", "輸入 `「。` 之後按 Enter。", id="反引號裡的 `「`"),
        pytest.param(
            "……Run All Cells」，清空工作桌再由上到下跑。旁邊的說明。",
            "……Run All Cells」，清空工作桌再由上到下跑。",
            id="前面沒有 `「` 的 `」`",
        ),
        pytest.param("程式最後的文字輸出應為 `鳥看狗，狗看鳥。", None, id="沒關的反引號之後的「。」"),
        pytest.param("例如「顏色=紅；形狀=圓。", None, id="沒關的 `「` 之後的「。」"),
        pytest.param(
            "程式最後的文字輸出應為 鳥看狗，狗看鳥。，而非舊句；這就是要核對的還原結果。",
            "程式最後的文字輸出應為 鳥看狗，狗看鳥。",
            id="限制：拿掉反引號的 code 裡的「。」",
        ),
        pytest.param("……狗看鳥。`，而非舊句；這就是要核對的還原結果。", "……狗看鳥。", id="限制：從 code 中間開始"),
        pytest.param("……形狀=圓。」。這個工具以字元為單位。", "……形狀=圓。", id="限制：從「」中間開始"),
        pytest.param(
            "輸入 ``a。b`` 之後按 Enter。旁邊的說明。", "輸入 ``a。", id="限制：兩個反引號包住的 code 裡的「。」"
        ),
    ],
)
def test_cut_at_full_stop_skips_backticks_and_brackets(text, expected):
    """逐一核對 cut_at_full_stop 截到哪裡：反引號裡的「。」在開頭、中間、緊接句尾之前與結尾，以及兩段反引號之間的「。」；
    「」裡的「。」在同樣四個位置，以及兩組「」之間的「。」；巢狀「」；反引號與「」都有、反引號裡的 `「`、前面沒有 `「` 的
    `」`；沒關的反引號與 `「` 之後的「。」；一般句子、沒有「。」、以「……」結尾的片段（另一例的反引號裡有「。」）；以及它
    docstring 寫的限制：拿掉反引號的 code、從 code 或「」中間開始、用兩個反引號包住的 code，裡面的「。」被當成句尾。"""
    assert cut_at_full_stop(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "# 頁面說：程式最後的文字輸出應為 `鳥看狗，狗看鳥。`，而非舊句；這就是要核對的還原結果。旁邊的說明。\n",
            [(1, "程式最後的文字輸出應為 `鳥看狗，狗看鳥。`，而非舊句；這就是要核對的還原結果。", None)],
            id="反引號裡的「。」",
        ),
        pytest.param(
            "# 頁面說：例如「顏色=紅；形狀=圓。」這種短文，\n# 共十二篇。旁邊的說明。\n",
            [(1, "例如「顏色=紅；形狀=圓。」這種短文，共十二篇。", None)],
            id="「」裡的「。」，引文接到下一行",
        ),
        pytest.param(
            "          # 頁面說：程式最後的文字輸出應為 `鳥看狗，狗看鳥。\n"
            "          # `，而非舊句；這就是要核對的還原結果。旁邊的說明。\n",
            [(1, "程式最後的文字輸出應為 `鳥看狗，狗看鳥。`，而非舊句；這就是要核對的還原結果。", None)],
            id="inline code 跨過註解換行，換行前的那段裡有「。」",
        ),
        pytest.param(
            "          # 頁面說：例如「顏色=紅；形狀=圓。\n          # 」這種短文，共十二篇。旁邊的說明。\n",
            [(1, "例如「顏色=紅；形狀=圓。」這種短文，共十二篇。", None)],
            id="「」跨過註解換行，換行前的那段裡有「。」",
        ),
        pytest.param(
            "          # 頁面說：先用 --list 查看名稱，\n          # 再下載所選包……\n          # 旁邊的說明。\n",
            [(1, "先用 --list 查看名稱，再下載所選包……", None)],
            id="接到以「……」結尾的那一行為止",
        ),
        pytest.param(
            "# 頁面說：這兩次沒有更新。--train 開啟更新……\n",
            [(1, "這兩次沒有更新。", None)],
            id="同一行先有「。」再以「……」結尾",
        ),
        pytest.param(
            "# 頁面說：mode……應為 dry-run-no-weight-update，\n# 有效目標數應大於零。\n",
            [(1, "mode……應為 dry-run-no-weight-update，有效目標數應大於零。", None)],
            id="行中間的「……」不結束引文",
        ),
        pytest.param(
            "x = 1\nAssert-Paths checkpoints/start.pt  # 頁面說：這兩次沒有更新。\n",
            [(2, "這兩次沒有更新。", None)],
            id="記號前面有程式",
        ),
        pytest.param(
            "# 頁面說：這兩次沒有更新。\nforeach ($name in 'a', 'b') { }\n# 頁面說：--train 開啟更新……\n",
            [(1, "這兩次沒有更新。", None), (3, "--train 開啟更新……", None)],
            id="兩處記號各一則",
        ),
    ],
)
def test_workflow_quotes_find_each_quote_and_where_it_ends(monkeypatch, text, expected):
    """餵一份假的 workflow 給 workflow_quotes，核對每一處 `# 頁面說：`（前面有程式也算）各回傳記號那一行的行號與引文；
    引文到 cut_at_full_stop 截到的「。」或以「……」結尾的那一行為止：反引號裡或「」裡的「。」不讓引文提早結束，code 或
    「」跨過註解換行時也一樣；同一行先有「。」、再以「……」結尾時截在「。」，行中間的「……」不結束引文。"""
    monkeypatch.setitem(globals(), "read", lambda relative: text)
    assert workflow_quotes() == expected


# workflow_quotes 回報引文沒有結束時的問題欄。
QUOTE_WITHOUT_END = "引文要在反引號與「」以外的第一個「。」或以「……」結尾的那一行結束"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "          # 頁面說：在專案根目錄執行時，後兩行應都是 True\n"
            "          Invoke-Page -Pages course/first-steps.md -Stdin $code -Commands '.\\.venv\\Scripts\\python.exe'\n",
            [(1, "在專案根目錄執行時，後兩行應都是 True", QUOTE_WITHOUT_END)],
            id="下一行不是註解",
        ),
        pytest.param(
            "          # 頁面說：先預測 pyproject.toml 是否還會被找到\n"
            "          # 頁面說：核對後用 exit() 退出，再 cd .. 回到上一層專案根目錄。\n",
            [
                (1, "先預測 pyproject.toml 是否還會被找到", QUOTE_WITHOUT_END),
                (2, "核對後用 exit() 退出，再 cd .. 回到上一層專案根目錄。", None),
            ],
            id="下一行又是一處 `# 頁面說：`",
        ),
        pytest.param(
            "# 頁面說：先用 --list 查看名稱，\n# 再下載所選包",
            [(1, "先用 --list 查看名稱，再下載所選包", QUOTE_WITHOUT_END)],
            id="引文還沒結束檔案就結束",
        ),
    ],
)
def test_workflow_quotes_report_a_quote_that_does_not_end(monkeypatch, text, expected):
    """餵一份假的 workflow 給 workflow_quotes，核對引文還沒碰到結束它的「。」或以「……」結尾的行，就遇到不是註解的行、
    又一處 `# 頁面說：`，或檔案結束時，問題欄寫明引文沒有結束，引文欄是到那時為止接起來的字。"""
    monkeypatch.setitem(globals(), "read", lambda relative: text)
    assert workflow_quotes() == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("見[入口頁](../README.md)的說明", "見入口頁的說明", id="連結只留連結文字"),
        pytest.param("![流程圖](figures/a.svg)之後", "流程圖之後", id="圖片只留方括號裡的文字"),
        pytest.param("見[入口\n頁](../README.md)", "見[入口頁](../README.md)", id="跨行的連結不換"),
        pytest.param("**粗體**與 2*3", "粗體與2*3", id="去掉 `**`，單一個 `*` 留著"),
        pytest.param("執行 `git lfs version` 核對", "執行gitlfsversion核對", id="去掉反引號"),
        pytest.param("a b\n\tc　d", "abcd", id="去掉所有空白：空格、換行、tab 與全形空白"),
        pytest.param("見[入口頁][readme]的說明", "見[入口頁][readme]的說明", id="參考式連結不換"),
        pytest.param("見 <https://example.com> 的說明", "見<https://example.com>的說明", id="自動連結不換"),
    ],
)
def test_flatten_keeps_link_text_and_drops_bold_marks_backticks_and_whitespace(text, expected):
    """逐一核對 flatten：同一行裡的行內連結（含圖片）只留方括號裡的文字，跨行的連結、參考式連結與自動連結不換；去掉
    `**`（單一個 `*` 留著）、反引號與所有空白。"""
    assert flatten(text) == expected


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        pytest.param("先用 --list 查看名稱……再用 --asset NAME 下載", True, id="連續的「……」切段，各段依序找得到"),
        pytest.param("先用 …--list 查看名稱", True, id="單一個「…」也切段"),
        pytest.param("**先用**`…`查看名稱", True, id="引文也先 flatten，`…` 去掉反引號後切段"),
        pytest.param("再用……先用", False, id="各段順序顛倒"),
        pytest.param("查看名稱……名稱，再用", False, id="後一段和前一段重疊"),
        pytest.param("先用……再用 --asset X 下載", False, id="有一段找不到"),
        pytest.param("……", False, id="切完一段都沒有"),
    ],
)
def test_in_order_is_true_only_when_every_part_is_found_in_order(quote, expected):
    """逐一核對 in_order：引文先 flatten，再以一個或連續幾個「…」切段（`…` 也算）；各段依序、不重疊地出現在頁面裡時為真，
    順序顛倒、和前一段重疊、有一段找不到，或切完一段都沒有時為假。"""
    page = flatten("先用 `--list` 查看名稱，再用 `--asset NAME` 下載、核對並解包所選包；將 NAME 換成清單中的名稱。")
    assert in_order(quote, page) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "- course/training.md T.9「希望答案是`blue`」與「這一題先讓你查看前後有沒有改變」：沒有期望值。",
            ["希望答案是`blue`", "這一題先讓你查看前後有沒有改變"],
            id="兩組「」各回傳裡面的字",
        ),
        pytest.param(
            "- course/training.md T.9「分別說明「保存變小了嗎」與「相同題目是否維持回答」」：沒有期望值。",
            ["分別說明「保存變小了嗎」與「相同題目是否維持回答」"],
            id="巢狀「」只回傳最外層的整段",
        ),
        pytest.param("- Skip-Page 的 -Reason 與註解寫的理由對不對。", [], id="沒有「」"),
        pytest.param(
            "- course/training.md T.1「打開幾筆，檢查 `text` 或 `messages` 等欄位：沒有期望值。", None, id="少一個 `」`"
        ),
        pytest.param("- course/training.md T.1 打開幾筆」：沒有期望值。", None, id="多一個 `」`"),
    ],
)
def test_bracketed_returns_the_outermost_contents_or_none_when_unbalanced(text, expected):
    """逐一核對 bracketed：每組最外層「」回傳裡面的字，巢狀時回傳外層的整段（裡面的「」留著）；沒有「」時回傳空的清單；
    少一個或多一個 `」` 時回傳 None。"""
    assert bracketed(text) == expected


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        pytest.param(
            "說明。\n\n不涵蓋：\n- README.md 甲「乙」：理由。\n- course/training.md T.1「丙」：理由。\n",
            [
                ("README.md", "- README.md 甲「乙」：理由。"),
                ("course/training.md", "- course/training.md T.1「丙」：理由。"),
            ],
            id="「不涵蓋：」之後以「- 」加頁名開頭的行",
        ),
        pytest.param(
            "- README.md 前面「甲」：理由。\n不涵蓋：\n- README_en.md 後面「乙」：理由。\n",
            [("README_en.md", "- README_en.md 後面「乙」：理由。")],
            id="「不涵蓋：」之前的項目不算",
        ),
        pytest.param(
            "不涵蓋：\n- 讀檔集合看不到 course/training.md 的「甲」。\n- course/first-steps.md W.7「乙」：理由。\n",
            [("course/first-steps.md", "- course/first-steps.md W.7「乙」：理由。")],
            id="「- 」之後不是頁名的項目不算",
        ),
        pytest.param(
            "不涵蓋：\n- README.md 甲「乙」：理由。\n\n另一段說明。\n- course/training.md T.3「丙」：理由。\n",
            [
                ("README.md", "- README.md 甲「乙」：理由。"),
                ("course/training.md", "- course/training.md T.3「丙」：理由。"),
            ],
            id="隔著一段說明的後一串項目也算",
        ),
        pytest.param(
            "不涵蓋：\n- README.md 甲「乙\n  丙」：理由。\n", [("README.md", "- README.md 甲「乙")], id="接續行不讀"
        ),
        pytest.param(
            "不涵蓋：\n- course/training.md：甲「乙」。\n- course/training.md T.1「丙」：理由。\n",
            [("course/training.md", "- course/training.md T.1「丙」：理由。")],
            id="頁名後面直接接標點的行不算",
        ),
        pytest.param("說明。\n- README.md 甲「乙」：理由。\n", [], id="沒有「不涵蓋：」那一行"),
    ],
)
def test_not_covered_page_items_list_the_page_lines_after_the_heading(monkeypatch, doc, expected):
    """把檔頭換成假的字串，逐一核對 not_covered_page_items：「不涵蓋：」那一行之後，以「- 」、頁名與空白開頭的每一行都算，
    隔著別的段落也算；那一行之前的、「- 」之後不是頁名的、頁名後面直接接標點的不算；接續行不讀；沒有「不涵蓋：」那一行時
    回傳空的清單。"""
    monkeypatch.setitem(globals(), "__doc__", doc)
    assert not_covered_page_items() == expected


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
    assert not malformed, (
        f"'**' 之後每一條都要寫成 '!<樹>/**'，一次排除一整棵樹；<樹> 以 / 分段，每一段都不能是空的、. 或 ..，"
        f"也不能含 * ? + [ ] !：{malformed}"
    )
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


@pytest.mark.parametrize(
    ("pattern", "accepted"),
    [
        pytest.param("!docs/technical-reviews/**", True, id="兩段一般的名字"),
        pytest.param("!.github/workflows/**", True, id="以點開頭的資料夾名"),
        pytest.param("!/docs/**", False, id="空段在第一段"),
        pytest.param("!docs//technical-reviews/**", False, id="空段在中間"),
        pytest.param("!notebooks//**", False, id="空段在最後一段"),
        pytest.param("!./notebooks/**", False, id=". 在第一段"),
        pytest.param("!docs/./technical-reviews/**", False, id=". 在中間"),
        pytest.param("!docs/./**", False, id=". 在最後一段"),
        pytest.param("!../notebooks/**", False, id=".. 在第一段"),
        pytest.param("!docs/../notebooks/**", False, id=".. 在中間"),
        pytest.param("!docs/../**", False, id=".. 在最後一段"),
        pytest.param("!docs/*-reviews/**", False, id="最後一段含 *"),
        pytest.param("!doc?/technical-reviews/**", False, id="第一段含 ?"),
        pytest.param("!docs/technical-reviews+/x/**", False, id="中間含 +"),
        pytest.param("!docs/[technical-reviews/**", False, id="最後一段含 ["),
        pytest.param("!docs]/technical-reviews/**", False, id="第一段含 ]"),
        pytest.param("!docs/!technical-reviews/x/**", False, id="中間含 !"),
    ],
)
def test_tree_exclusion_verdicts(pattern, accepted):
    """逐一核對 TREE_EXCLUSION 對幾種寫法的判定：兩段一般的名字、以點開頭的資料夾名要接受；空段、. 段、.. 段在第一段、
    中間、最後一段各一例，以及 * ? + [ ] ! 各一例（分在第一段、中間、最後一段），都要不接受。"""
    verdict = "接受" if accepted else "不接受"
    assert bool(TREE_EXCLUSION.fullmatch(pattern)) is accepted, f"TREE_EXCLUSION 應該{verdict} {pattern}"


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


def test_workflow_file_pages_notebook_and_python_run_by_workflow_are_outside_excluded_trees():
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
