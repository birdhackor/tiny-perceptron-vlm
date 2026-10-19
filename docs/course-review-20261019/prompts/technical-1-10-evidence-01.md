你的原始判斷 `docs/course-review-20261019/reviews/rv1019_technical_1-10/technical-1-10-original-01.json`（SHA-256 `b7a19eb099b992640fbcbc76c44ec6f7081dee2fbdba88fa6f84da57b904e5bf`）已收件，verdict 是 pass。之後作者為了另一節（1.4）的問題改了 `course/chapters/01.md`。你的報告在 `sources` 的 `s_course` 以全檔 SHA-256 `b479af60f95861dba2297789e7db3283b409bc0d9276c5a0e6c8e13949340097` 引用這個檔，所以發布閘門現在把這條證據判成「證據版本已改變」。請確認這次修改有沒有碰到你依賴的內容，並交一份現行判斷。

## 改了什麼

- commit `c889d619` 只改了 `course/chapters/01.md` 的一行，也就是 1.4「補充與重做」裡的一句，另外改了 `notebooks/01/1.4.ipynb` 裡對應的一行。完整 diff 用 `git show c889d619` 看。
  - 原句：「`Counter` 是一種字典：鍵是字，值是次數，所以字典的寫法都能照用；查沒出現過的字會得到 0 次，不會出現 `KeyError`。」
  - 新句：「`Counter` 是一種字典：鍵是字，值是次數，所以本節用到的 `.values()`、`.items()` 都能照用；查沒出現過的字會得到 0 次，不會出現 `KeyError`。」
- `course/chapters/01.md` 現在的 SHA-256 是 `d2fcd73563506b8d38f070dbc6ae23adf25050e4f0af86435170fd576c269787`；你審完之後，這個檔只有這一次修改。
- 你審的 1.10 沒有改：凍結檔 `docs/course-review-20261019/freeze-02.json`（SHA-256 `2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1`）裡 1.10 的 `source_sha256` 仍是 `11c391843416da38be771144c536cd0e04502b1bb1a43c8bf035841327b612af`，和你的報告相同。

## 請交

`docs/course-review-20261019/reviews/rv1019_technical_1-10/technical-1-10-current-01.json`，正式紀錄：寫好就以 SHA-256 綁住，不能改寫；寫錯就另存 `-02` 並在回覆說明原因。

- 格式照 `docs/technical-review-guide.md` 與你的原始判斷。`source_sha256` 不變；`source_freeze` 寫 `{"path": "docs/course-review-20261019/freeze-02.json", "sha256": "2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1"}`；`figure_sha256` 照凍結檔裡這一頁的圖清單（這一頁的圖是 `course/figures/rewrite-01-chain.svg`（SHA-256 `65f29ee754f2bb61393cb2cbf79048329058c16ec24b9b16dbb2e142fa2c4e92`））。
- `sources` 裡 `s_course` 那一條的 `sha256` 換成現行值。先讀上面的 diff，確認它的 `inspection_note` 依賴的段落沒有被改到。其他 sources 的 SHA 仍對得上的可以沿用原值。
- 主張照你的原判斷；凡是依賴被改內容的，要重新核實。全部主張核實才寫 `verdict: pass`；仍有 `unresolved` 或 `contradicted` 就寫 `revise`，並照原格式寫 `issues`。
- 另加 `recheck` 物件：`previous_report`（原始判斷的 path 與 sha256）、`scope`（這次重看了什麼）、`source_comparison`（你怎麼確認被改的那一行和你依賴的段落無關）、`unresolved_claims`、`contradicted_claims`。
- 新寫的程式與輸出放進你的 artifact 目錄，用新檔名，不覆寫既有檔。
- 寫好後跑自我檢查，把輸出貼進回覆：
  ```
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-review-20261019/tools/check_technical_report.py 1.10 docs/course-review-20261019/reviews/rv1019_technical_1-10/technical-1-10-current-01.json --freeze docs/course-review-20261019/freeze-02.json --reader-task /root/rv1019_reader_ch01 --reader-task /root/rv1019_reader_ch01b
  ```

能看什麼、不能做什麼、只寫哪裡，都照原指示；獨立性的要求也照舊，不看其他角色的報告。

## 回覆

- verdict；你依賴的段落有沒有被改到、怎麼確認的。
- 報告與每個新 artifact 的路徑與 SHA-256。
- 自我檢查的輸出。
- 你沒能驗證的範圍。
