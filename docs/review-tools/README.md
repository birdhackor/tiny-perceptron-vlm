# 單節原文與執行事實

這個工具只提取原始 UTF-8 小節、程式區塊與 SVG 指紋；加 `--execute` 才在新的 CPU 程序執行本節 Python。它不產生審閱報告或通過判定。審閱者仍須自行閱讀文本、圖、原始論文與實測證據，依 [技術審閱規約](../technical-review-guide.md)判斷主張。

在專案根目錄執行：

```bash
.venv/bin/python docs/review-tools/section_facts.py 'course/chapters/14.md#14.1'
.venv/bin/python docs/review-tools/section_facts.py 'course/chapters/14.md#14.1' --execute --timeout 120
```

每次建立新的 ignored `outputs/reviewer-tools/runs/` 目錄；也可用 `--output` 指定新的 outputs 或 /tmp 路徑。輸出包括原文、原始 fences、SVG 快照、當次 bootstrap、指令、Python／PyTorch 版本、stdout、stderr 與實際 exit code。原文 SHA 不正規化換行；同一節程式依順序共享 namespace，各次執行互不共享。沒有 Python 的小節不會執行 bash，明記 `executed=false` 並返回 2。

執行使用目前 `build_course.py` 的 CPU bootstrap 與實際 repo 模組，不讀生成的 Notebook。工具的 Python audit guard 禁止網路、子程序與當次輸出目錄外的寫入。需要編譯器等子程序的程式可能因此被拒絕；這是工具限制，不能判為教材錯誤。審閱者可在檢查該命令後，另設有界的 CPU 執行並保留失敗及新證據。

退出碼 0 只表示程式返回 0。真正引用執行證據時，需將必要的小份輸出與環境記錄保存到 `docs/technical-reviews/artifacts/`，記錄相對路徑、SHA、當次指令與支持範圍；ignored 目錄或 /tmp 本身不構成持久證據。

`check_review_round.py` 另外核對本輪審閱身分與導言。基準清單保存實驗改寫前固定 Git 版本的248份舊讀者報告指紋；新報告不能沿用任何舊身分，技術審閱也不能沿用新讀者身分。每份來源的首節須連同導言實際閱讀，記錄 `intro_sha256` 與自己的 `intro_summary`。此核對不產生報告、摘要或判定，也不能證明署名的代理真的執行過；協調者仍需查實際派工與閱讀紀錄。

```bash
# 讀者階段全部完成後
.venv/bin/python docs/review-tools/check_review_round.py --stage reader
# 兩個獨立階段完成後，再連同既有正文、圖與證據 checker 執行
.venv/bin/python docs/review-tools/check_review_round.py
```
