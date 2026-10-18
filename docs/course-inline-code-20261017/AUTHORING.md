# 本輪作者規則

先讀[計畫](PLAN.md)、[逐節審閱標準](../editorial-guide.md)與[寫作短 prompt](../../.agents/skills/clear-tutorial/references/writing-prompt.md)。本檔只補本輪新增的規則；與逐節審閱標準衝突時，以逐節審閱標準為準，並在報告中指出衝突。

## 範圍

- 只改派給你的章檔；派工單另外點名的圖檔才可以改。不改其他章、名詞頁、入口頁、`tiny_perceptron/`、`scripts/`、`notebooks/`、`zensical.toml`。
- 不新增、刪除或重新編號小節。`## X.Y 標題` 這一行只有 TERMS.md「小節標題」段列出的幾節可以改用詞，改了要在報告列出新舊標題；節內小標題最多用 `###`，而且少用。
- 不 commit、不執行 `scripts/build_course.py`（它會改寫全書 Notebook，其他作者正在平行工作）、不用 `git stash`。

## 把實作放進正文

目標是讀者不打開 Colab 或原始碼，也能在本頁看到零件是怎麼寫的、用起來得到什麼。每個零件只在[配置表](PLACEMENT.md)指定的那一節完整展示並講解；之後的節用一句話加連結回指（例如「`TinyLM` 的實作見[4.6](04.md#4.6)」），不重貼。

三種寫法，標記一字不差（全形「：」「、」「（）」）：

1. **原樣節錄**——fence 只放節錄，第一行是標記，其後逐字貼上函式庫中的頂層定義（用 `sed -n '<起>,<迄>p' tiny_perceptron/<檔案>.py` 複製，不要手打）；多個定義用「、」列名，依標記列出的順序貼上，中間隔兩個空行；不必在原檔相鄰：

   ````text
   ```python
   # 節錄自 tiny_perceptron/data.py：shifted
   def shifted(ids):
       ...（原樣）
   ```
   ````

2. **精簡版**——定義帶有後面章節才教的參數或分支時，重寫只含本節所需的部分，參數名與權重名稱保持與函式庫一致，好讓 `load_state_dict` 直接載入同一組權重：

   ````text
   ```python
   # 精簡自 tiny_perceptron/model.py：TinyLM（省略 RoPE、權重共享、cache 與多模態輸入）
   class TinyLM(nn.Module):
       ...
   ```
   ````

   精簡版要寫在標記所在的 fence 裡（頂層的 `class` 或 `def`）。同一節必須 import 函式庫的同名定義，而且用 `as` 換成另一個名稱（例如 `from tiny_perceptron.model import TinyLM as LibraryTinyLM`；不換名會覆蓋你剛寫的版本，之後比的就是函式庫自己）；再在精簡版之後用 `assert`（`torch.equal`、`torch.allclose` 或 `==`）確認兩者在本節例子上結果相同。測試會檢查這個 `assert` 的運算式，順著同一節的指派往回追，要同時用到精簡版與函式庫別名；只用到其中一邊的 `assert`（例如只檢查精簡版輸出的形狀）不算。測試看不出運算式是否真的在比較兩邊，比得對不對仍由執行時的 `assert` 本身核對。網站是靜態頁面：程式在發布時先執行一次，輸出貼在程式下方；只有 `assert` 的格子沒有輸出，讀者看不出比對過，所以 `assert` 前先 `print` 比對結果（例如 `print("與函式庫版本相同：", same)`）。正文要說明省略了什麼、在哪一節登場。
3. **只說明並連結**——資料產生器、報告輔助與大型成品程式，用一兩句話說它做什麼、輸入輸出是什麼，連到原始碼（相對路徑，例如 `[tiny_perceptron/multimodal.py](../../tiny_perceptron/multimodal.py)`）。

標記只寫在 fence 第一行；除了標記，fence 裡任何一行都不寫 `# 節錄自…`／`# 精簡自…` 字樣。帶標記的 fence 要放在某一節裡：第一個 `## X.Y` 節標題之前的導言不產生 Notebook，裡面的程式不會執行。fence 開頭一行（縮排除外）要剛好是 ```` ```python ````，不加 fence 屬性（`title`、`hl_lines` 等）：`scripts/build_course.py` 只把 ```` ```python ```` 之後直接換行的開頭當成 Notebook 程式格，```` ```py ````、```` ```python title="…" ```` 或 `python` 後多了空白都不算。標記不符合以上任何一條，`tests/test_course_code_excerpts.py` 都會報錯。

## 一節裡的順序

可以依本節需要調整，但實作必須出現在正文：

1. 用具體問題與例子開頭（原有寫法多數很好，保留）。
2. 先看結果：一段短程式用函式庫或剛展示的零件跑出本節例子。
3. 它怎麼做到：節錄或精簡版。逐一說明關鍵行——用反引號引出那一行或那個運算式，不寫「第 N 行」；說明輸入、形狀、每一步做什麼。
4. 核對：精簡版先 `print` 比對結果，再 `assert`；原樣節錄可省略。
5. 想動手的練習：寫清楚「把第幾段程式的 `某一行` 改成什麼、先預測什麼、預期看到什麼」，並在正文或補充區給出答案思路。不要寫「打開 Colab 練習」。

每節合計程式以 3 段、40 行左右為上限；超過時優先精簡。第一次展示 `nn.Module` 類別的那一節，要說明 `class X(nn.Module)`、`__init__` 建立可學的零件、`super().__init__()`、`forward` 定義計算，以及呼叫 `model(x)` 時會執行 `forward`。之後的節可回指。

正文不另寫靜態輸出：網站會在程式下方顯示實際輸出。正文引用的數字（形狀、損失、機率）必須是你實際執行看到的值；能附一行算式的就附上。

## 處理試讀意見

派工單附上本章的意見分包。每一條都要處理並在報告中列出：採用／部分採用／不採用，以及理由。標記「複核：ok」的是主 agent 已確認的錯誤，必須修正；「partial」只有部分成立，照複核說明處理。

跨章原則：

- 用語照 [TERMS.md](TERMS.md)。
- 理解正文所需的定義、分母、數字出處放正文；`<details>` 補充區只放出處、重做命令與延伸。先問：不打開補充區，能不能讀懂正文？
- 提前引用後面才教的概念時，加一句「第 N 章才講，這裡只需要知道…」；能改順序就改順序。
- 第一次出現暖身沒教過的 Python 寫法（`while`、`lambda`、`zip`、`enumerate`、f 字串、`try`／`except`、`//`、切片等）時，加一句白話說明。
- 刪去給維護者的句子與修訂史（「保留節號供既有連結回查」「原來的」「舊版」這類）；連到 JSON 紀錄時說明看哪一欄。
- 寫「見 X」時，確認 X 真的有那個內容。

## 程式的限制

- 離線、CPU、每節幾秒內跑完；不下載資料或權重、不訓練到收斂、不用 GPU。固定亂數種子。
- 訓練結果的數字只能引用 repo 既有紀錄（例如 `docs/course-experiments/results/*.json`），寫出出處。
- 不要在同一段程式 import 一個名稱又用 `def`／`class` 重新定義它（ruff 的 F811 會報錯）。

## 自我驗收

`$V` 由派工單給出。改完後依序執行並把輸出附在報告：

```bash
$V/python scripts/format_course_code.py course/chapters/<你的章>.md
$V/python -m pytest -q tests/test_course_code_excerpts.py      # 只看你這章的節；其他章可能還在改
for id in <你改過的節>; do $V/python docs/review-tools/section_facts.py "course/chapters/<你的章>.md#$id" --execute; done
git status --short
```

`section_facts.py --execute` 會在新的 CPU 程序依序執行該節全部程式，印出一行 JSON，其中 `execution_exit_code` 應為 0；實際輸出在 JSON 的 `artifact_directory` 下的 `stdout.txt`（位於 Git 忽略的 `outputs/reviewer-tools/runs/`）。確認正文引用的數字與 `stdout.txt` 一致。該節沒有 python fence 時（同一行 JSON 的 `python_fences` 為 0），工具不執行任何程式，`execution_exit_code` 是 2，這是預期結果，不是錯誤。

## 報告

1. 每節一列：做了什麼（展示哪個實作、用哪種寫法、補了什麼程式、修了什麼）。
2. 意見處置表：意見編號、處置、理由。
3. 沒做完或拿不準的地方。
4. 上面驗收指令的實際輸出與 `git status --short`。
5. 派工單要求的升級訊號回報。
