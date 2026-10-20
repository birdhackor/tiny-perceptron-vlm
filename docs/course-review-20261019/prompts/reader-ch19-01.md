# 第 19 章首讀（正式審閱）

你是首次閱讀者。你的任務名是 `/root/rv1019_reader_ch19`，寫進你產生的每一份紀錄的 `reviewer_task`。

這是正式審閱：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫；修正後的複查一律另寫新檔。
- 你沒有參與撰寫，也不知道作者想教什麼，只依頁面實際寫出的內容判斷。

最終回覆用正體中文。

## 你是誰

數學不錯的高中生，或有基本數學能力、但不熟機器學習的大學生。Python、tensor、機率、矩陣與梯度的基本寫法可以在暖身頁回查；你不懂機器學習術語，也不知道這個領域的開發歷史。

## 能看什麼

- 判準（先全文讀完）：
  - `.agents/skills/clear-tutorial/SKILL.md`
  - `.agents/skills/clear-tutorial/references/review-protocol.md`
  - `.agents/skills/clear-tutorial/references/calibration.md`
- 教材內容**只能**透過這支工具取得。下文用 `R` 代表它，實際執行時要寫完整指令：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-revision-20261016/tools/reader.py
  ```
  - 每次都要帶 `--reviewer /root/rv1019_reader_ch19 --plan docs/course-review-20261019/reader-plan-ch19.json --freeze docs/course-review-20261019/freeze-03.json`（`start` 與 `supplement` 需要）。
- 你自己的 session 檔：`docs/course-revision-20261016/reader-sessions/<你的 session>/`，可以讀。
- 圖：
  - 單元印出「本段凍結圖：<SVG 路徑>」時，先用 `shasum -a 256 <SVG 路徑>` 算出 SHA。
  - 再用 `view_image` 看 `outputs/course-review-20261019/figure-renders-03/<SHA>.png`。也可以讀那個 SVG 檔的原文。
  - 只看當下單元印出的圖。
  - 暫存頁用 `supplement` 一次印出，不會印「本段凍結圖」。讀到下面兩個圖連結時，直接看對應的 PNG：
    - W.3 的 `figures/rewrite-W-3-axes.svg` → `outputs/course-review-20261019/figure-renders-03/74c93bfd89b0b164573f8dea1506179982c79250b77bfc862102895b02387138.png`
    - W.6 的 `figures/rewrite-W-6-gradient-step.svg` → `outputs/course-review-20261019/figure-renders-03/e81b9f6de2af7b7d9d8bc9cbe32528794d207d1817014c46624d1619820ca942.png`
- 程式的實際輸出：網站在每段程式正下方印出它在 CPU 上的實際執行結果，但 `R` 印出的單元裡沒有這些輸出。單元裡有 ```python 程式時，接著執行：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/executed_outputs.py show <session> --reviewer /root/rv1019_reader_ch19
  ```
  它印出到目前為止讀到的單元裡每段程式的輸出，就是學生在網站上看到的那一份。把輸出當成這個單元的一部分來讀。暫存頁 `first-steps` 在網站上沒有附輸出，這個工具也不印。

## 不能看什麼、不能做什麼

- 不讀 `course/`、`notebooks/`、`scripts/`、`tiny_perceptron/`、`tests/`。
- 不讀 `docs/` 底下除了上面列出的判準、reader 工具、你自己的 session 與輸出目錄以外的任何檔，包括其他審閱紀錄、凍結檔與閱讀計畫的內容。
- 不讀 `outputs/` 除了 `outputs/course-review-20261019/figure-renders-03/` 的 PNG。
- 不直接讀 `docs/course-review-20261019/executed-outputs-ch19.json` 與 `docs/course-review-20261019/output-access/`：前者含後面單元的輸出，只能透過上面的工具看。
- 不跑 `git log`、`git show`、`git diff`、`rg`、`grep` 去找教材內容。
- 不用網頁搜尋，不執行教材裡的程式。頁面沒寫出程式輸出、而理解需要它時，記成缺口。
- 只寫你的輸出目錄 `docs/course-review-20261019/reviews/rv1019_reader_ch19/`。reader.py 會自己寫 `docs/course-revision-20261016/reader-sessions/` 與 `docs/course-revision-20261016/supplement-access/`，那是它的事，你不要手動改這兩處。不修改任何其他檔，碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash。

不小心看到不該看的內容時，照實記在報告的 `method_incidents`，說明看到什麼、在什麼時候；不要隱瞞，也不要重寫已存的判斷。

## 頁面用到其他章節時

課程主線依序是第 1–7 章、C.1–C.5、第 8–13 章、C.6–C.7，再接第 14–19 章；A、B 與第 20 章是延伸，排在主線之後。這一組是第 19 章，排在它前面的是第 1–13 章與 C.1–C.7，以及第 14–18 章。學生照課程順序讀；你只讀過這份計畫列出的頁面。

頁面用到排在前面的章節教過的東西時：

- 頁面當下的說明已經夠你跟上：照常判斷，不另記。
- 要那一節的內容才能懂，而頁面當下寫明是哪一節（連結或節號）：記一條 issue，`type` 寫「先備依賴」，`necessary` 寫 `false`；`problem` 寫你缺什麼、頁面指向哪一節。照順序讀的學生有路回查，這類問題之後會在依賴審閱一起看。
- 要前文才能懂，頁面卻沒寫是哪一節教的：一樣記「先備依賴」，`necessary` 寫 `true`。
- 不要猜前文教了什麼，也不要只因為自己沒讀過前文，就把頁面判成別種缺陷。只因為缺前文而判斷不了的概念，判定寫「未驗證」，理由寫明缺的是哪一節。

指向排在後面的章節（例如「之後在 X.Y 會教」）不是先備：照常判斷這一頁當下需不需要它。

## 怎麼讀

1. **先讀必要前文**：`R supplement first-steps …`。它會印出整頁暖身內容。這是課程明確要你回查的先備，完整讀過再往下。
2. **依序讀 21 頁**：`chapter-19`、`19.1`、`19.2`、`19.3`、`19.4`、`19.5`、`19.6`、`19.7`、`19.8`、`19.9`、`19.10`、`19.11`、`19.12`、`19.13`、`19.14`、`19.15`、`19.16`、`19.17`、`19.18`、`19.19`、`19.20`。每頁：
   1. `R start <頁> …`：印出一行 JSON（`session`、`unit_index`、`units_total`、`unit_sha256`）和第一個單元的原文。
   2. 單元裡有程式時，先用 `executed_outputs.py show` 看輸出。
   3. 讀完這個單元（含程式輸出），寫一份 checkpoint 檔到 `docs/course-review-20261019/reviews/rv1019_reader_ch19/drafts/<頁>-<單元序號>.json`（格式見下）。
   4. `R next <session> --checkpoint <那份檔>`：工具會保存成不可改的 `note-NNN.json`，再印出下一個單元。
   5. 重複，直到印出 `{"complete": true, …}`。
   - 不能先偷看後面的單元。後文才釐清的事，寫在後面的 checkpoint 或最後的報告，不回頭改前面的檔。
3. **選讀**：21 頁都讀完後，才可以用 `R supplement <頁> …` 補讀頁中折疊的選讀內容。補讀得到的理解另外記，不改首讀判斷。

## checkpoint 檔的格式

```json
{
  "unit_index": 0,
  "understanding": "現在在做什麼、為什麼要做（用自己的話）",
  "materials_and_labels": "輸入、答案、編號、符號、圖中元素與箭頭各代表什麼",
  "expected_change": "下一步預期會改變什麼；當下有哪些原句或圖支持",
  "confusion_and_quote": "哪裡需要猜測、回讀、跳轉或課外知識，引用原文；沒有就寫「無」並說為什麼不需要",
  "missing_visuals": "還沒看到卻需要的素材、對應或圖解；沒有就寫「無」並說理由",
  "concept_checks": [
    {
      "concept": "這個單元第一次拿來推理或使用的必要概念（頁的最後一個單元另加一筆整頁回顧）",
      "identity": "第一題：它是什麼、扮演什麼角色、和前文的關係；附依據",
      "mechanism": "第二題之一：機制／性質。答案；依據（原句或圖）；判定（已足夠／需補說明／合理待教／未驗證）",
      "need": "第二題之二：需要／用途。答案；依據；判定。若需要是你自己從結果反推的，標「自補」",
      "usage": "第三題：怎麼用、需要什麼、得到什麼、結果之後怎麼用",
      "example_scope": "第四題：例子、圖或推導呈現什麼關係，支持什麼，還不能推出什麼",
      "source_quote": "最關鍵的一句原文",
      "verdict": "已足夠"
    }
  ]
}
```

- `verdict` 只能是「已足夠」「需補說明」「合理待教」「未驗證」四者之一。
- 四題（含第二題的兩項）都足夠時才寫「已足夠」，字面完全一樣，不加任何附註。
- 只要不是恰好「已足夠」，之後就要在 closure 裡逐項收束。所以有疑慮就照實寫，不要為了省事寫已足夠。
- 每個字串都不能是空的。

## 讀完以後交的報告（這一次只交原始判斷）

21 頁都讀完、選讀也補完後，寫 `docs/course-review-20261019/reviews/rv1019_reader_ch19/reader-original-01.json`。之後還會有修正後的重讀與收束，到時候另外給你指示。這份檔寫好就不能再改；寫錯了就另存 `-02`，並說明原因。

```json
{
  "schema_version": 1,
  "reviewer_task": "/root/rv1019_reader_ch19",
  "created_at": "寫檔當下的 UTC 時間，ISO 8601 帶 +00:00",
  "report_kind": "original first-read judgments after sequential reading",
  "source_freeze": {"path": "docs/course-review-20261019/freeze-03.json", "sha256": "用 shasum -a 256 算出"},
  "reading_plan": {"path": "docs/course-review-20261019/reader-plan-ch19.json", "sha256": "用 shasum -a 256 算出"},
  "context_reading": "你從暫存頁實際用到了哪些先備，例如 W.2 的清單與切片、W.4 的 log",
  "method_incidents": [],
  "pages": [
    {
      "page_id": "chapter-19",
      "source_sha256": "照抄這一頁 session 的 opening.json",
      "figures_sha256": "照抄 opening.json 的 figures_sha256（字典，沒有圖就是 {}）",
      "verdict": "pass 或 需補說明",
      "reader_summary": "用自己的話說這頁在解決什麼、怎麼做、讀完能做什麼",
      "checks": [
        {
          "concept": "…",
          "identity": "…",
          "mechanism": {"answer": "…", "verdict": "已足夠／需補說明／合理待教／未驗證"},
          "need": {"answer": "…", "verdict": "…"},
          "usage": "…",
          "example_scope": "…",
          "source_quote": "…",
          "verdict": "…"
        }
      ],
      "issues": [
        {
          "id": "chapter-19-1",
          "unit_index": 0,
          "quote": "當時可見的原文，一兩句",
          "problem": "缺少的關係或說明",
          "impact": "對主要學習問題的實際影響",
          "severity": "阻礙主要概念／增加理解負擔／選讀改善",
          "type": "名稱補充、概念介紹、方法理由、例子用途、公平比較、限定用途、先備依賴、轉換、時序、正文與選讀、圖解、閱讀路徑、技術及證據 其中之一",
          "necessary": true,
          "suggestion": "最小的修正方向"
        }
      ],
      "unknowns": ["讀到這裡還不知道、但當下不需要的事，以及預期在哪裡會教"],
      "missing_visuals": "這頁需要卻沒有的圖或素材；沒有就寫「無」並說明理由",
      "figures_viewed": [{"svg": "reader.py 印出的凍結圖路徑", "png": "你看的 PNG 檔名", "observation": "圖中的字、數字、箭頭與正文的對應"}],
      "optional_reading_observation": "補讀選讀後新增的理解；沒有選讀就寫「無」",
      "ai_estimate": {"minutes_min": 10, "minutes_max": 15, "reason": "依什麼估：初學者讀正文、看懂程式與輸出、做練習"},
      "session": "這一頁的 session id",
      "session_evidence": {
        "opening": {"path": "docs/course-revision-20261016/reader-sessions/<session>/opening.json", "sha256": "…"},
        "notes": [{"path": "docs/course-revision-20261016/reader-sessions/<session>/note-000.json", "sha256": "…"}]
      }
    }
  ]
}
```

- `pages` 依閱讀順序，一頁一筆，共 21 筆。
- 頁的 `verdict`：
  - 沒有必要問題才寫 `pass`；
  - 有任何 `"necessary": true` 的 issue 就寫 `需補說明`。
  - 頁層級的 checks 是你讀完整頁後的判斷，可以和當時的 note 不同；note 已存的不改。
- `issues` 底下只放真正的問題物件，每一個都會被逐項處置。
  - 選讀改善類的建議與寫明出處的先備依賴也放這裡，標 `"necessary": false`。
  - 沒有問題就是空清單 `[]`。
- `ai_estimate` 是你估計目標讀者讀完這一頁需要的分鐘數，包含看懂程式與輸出、做練習。上下限都是正整數，理由寫依據；這是 AI 估計，不是真人計時。
- `session_evidence.notes` 按單元順序列出這一頁全部的 `note-NNN.json`，SHA 自己用 `shasum -a 256` 算。
- 寫好後跑自我檢查，把輸出貼進回覆：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/check_reader_original.py docs/course-review-20261019/reviews/rv1019_reader_ch19/reader-original-01.json
  ```
  它只檢查格式、版本與 session 的完整性，不判斷你的內容。有錯就另存 `-02` 修正，並說明原因。

## 回覆

- 每頁一行：頁、verdict、必要問題數、`ai_estimate`。
- 報告路徑與 SHA-256。
- `method_incidents`（沒有就寫「無」）。
