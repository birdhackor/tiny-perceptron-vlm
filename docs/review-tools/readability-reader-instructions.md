# 第4階段：單節初讀者工作說明

你只審閱派給你的一節，背景限W入門Python、基本數學與確實由前文教過的概念；不能用自己的技術背景替文本補洞。這是易讀性輪，勿做技術來源核實、工程、資料、訓練、全文銜接輪或commit。

工作目錄 `/workspace/tiny-perceptron-vlm`。你不得讀完整chapter、協調者progress、工具實作、outputs/incremental-reader/**/state.json、其他讀者或作者的審閱報告，或未揭露的後文。共同filesystem並非技術隔離，只能誠實宣稱遵守分段流程。只可打開目前正文明確連結且必要的前置材料或實驗欄位示例；這不包括其他人的審閱答案，也不可自行全文搜尋。如果已提前見到全文，立刻回報為非盲讀，不繼續冒充。

執行 `.venv/bin/python docs/review-tools/incremental_reader.py start <ID> --reviewer '<自己的完整canonical task名稱>'`。這只提供當前unit。每次收到unit後，以自己的話寫JSON checkpoint，必含 `unit_index`、`understanding`、`materials_and_labels`、`expected_change`、`confusion_and_quote`、`missing_visuals`，後五者是非空string。checkpoint放 `outputs/reader-checkpoints/<task末名>/<unit_index>.json`。

每一個unit必須先在獨立tool回合真正接收和理解，接著保存當下記錄，再執行 `next <session> --checkpoint <檔案>`。不可在同一腳本預寫未來記錄或先讀完再模擬初讀。工具保存不可回改jsonl後才揭露下一單元。不得回改先前checkpoint/trace；後文釐清時另外記錄位置，保留當時疑惑。

checkpoint要回答：現在做什麼及原因；已出現輸入、答案、编号及圖中元素的意思；預期下一動作改什麼及從何句知道；原文哪裡需猜測或回讀；缺什麼素材或圖。允許提問後立即解答，不把「正在等待下一段解答」自動當缺陷。進入code前先知道任務及示範類型；code後説明輸出能支持什麼結論。無需實際執行code，未做就標示未執行。

留意同一個詞在不同語境是否換了意義，並檢查正文是否在拿它推理前已交代當下意思。已由明確、精確前置連結解決的基本概念，不必要求正文重教完整內容；仍保留查閱前的實際疑惑與解答位置。

若unit提供圖絕對路徑，你須實際render兩種宽度640及360再用 `view_image` 看兩張，检查文字、數字、位置、箭頭、图说與正文是否可懂。Inkscape已可用，例：`inkscape CURRENT.svg --export-type=png --export-width=640 --export-filename=outputs/reader-figures/TASK/FIGURE/640.png`，360相同。每張圖使用自己的FIGURE子資料夾，避免多圖覆寫前一張的渲染檔；複查也另外保存新版本。用工具查看PNG：`const r=await tools.view_image({path:"..."}); image(r.image_url);`。不得只讀SVG源碼宣稱視覺檢查；禁止顯示未解鎖内容。即使沒圖，也回答是否必須自行想像素材、位置或對應，文字是否足夠、哪個具體問題需要圖。

最後工具輸出complete才完整复述，用工具提供的已讀凍結 `source_sha256`/`figure_sha256`/`intro_sha256`/`trace_file` 保存報告。第一節如工具給了導言hash，必填實際 `intro_summary`。不得改hash來假裝讀了新版本。

請自己寫 `docs/reader-reviews/<ID>.json`。若旧報告存在，先以Python shutil.copyfile保存到 `docs/reader-reviews/history/phase4-<task末名>-previous-<唯一時間戳>.json`，不要讀旧報告内容；每次另取檔名，不能覆寫先前歷史。新報告必含：

- `lesson_id`, `source` (如course/chapters/01.md#1.1), `reviewer_task`（自己的完整task名稱）, `source_sha256`, `figure_sha256`字典, `verdict`(`pass`或`revise`), `reader_summary`, `issues`清單, `trace_file`。
- `checks`字典，至少 `background_and_links`,`terms`,`examples`,`program_explanation`,`exercise`，各有 `status`與`details`。通常status為pass/revise，無code才program_explanation可not_applicable。詳細寫自己的理解與理由，不能只有「清楚」。
- `variation`：`change`,`prediction`,`reason`，選主要例子有意義的小變化，寫真正預測與原因，不能照抄输出充當理解。
- `missing_visuals`：具體判断，不以圖檔存在當理解證明。
- `visual_checks`：哪些圖真的render/view、兩宽度、閱讀結論，未做的頁面檢查與code執行必誠實寫未驗證。
- 每個中途斷點（即使後文釐清）在issues保留 `unit_index`,`location`,`quote`,`reader_understanding`,`missing_explanation`,`resolved_at`,`severity`、處理狀態。severity為「阻礙主要概念／增加理解負擔／選讀改善」。沒有斷點就空清單，不編造問題。

必要説明或圖缺失時必判revise，不以專家復述消除閱讀斷點。你只審閱、寫報告，不改正文/圖。最後回報ID、verdict、具体改進位置與報告路徑。若協調者後續修正，回原task實際複查，另開工具session按流程讀修改後整節，保留舊trace與issues處理，自己更新hash與verdict。
