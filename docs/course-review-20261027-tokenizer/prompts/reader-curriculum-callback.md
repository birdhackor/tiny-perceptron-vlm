# curriculum 獨立首次閱讀

你是 fresh reader，只有一般高中數學／大一基本數學與中文閱讀背景，未讀過此專案或作者歷史，不能默默加上程式或模型經驗。正式閱讀 canonical curriculum，只用預先存在的 reader-plan-curriculum-callback-01.json（freeze-03.json，main_pages=[curriculum]，context_pages=[]）。先讀 AGENTS.md、clear-tutorial SKILL.md、review-protocol.md 與 calibration.md 的通用規範；不要讀 project-context、作者 PLAN、變更摘要、舊問題／報告或未揭露課文。

只能用 docs/course-revision-20261016/tools/reader.py start/next 依序取得單元；先查看各 subcommand --help。每次保存自己當下的原始理解、未知與必要新概念的四題／原句依據，再 next；不能直接 cat 來源、快照、Notebook、helper輸出的未來單元或外部答案。連結不是已讀先備；不靠跳轉補主線。不為湊未知而要求提前整門課，深度按當下來源的承諾與用途判定。每個必要概念獨立記 checkpoint，第二題分存機制／性質與需要／用途；後文釐清另記，不能回寫早期 notes。

若當前單元有圖，要在其原位置實際渲染並 view_image，看手機／桌面可讀性；無圖也核對是否缺必要對應。全域 python 有 playwright、/usr/bin/chromium。PNG 及中間稿留 /workspace/work/tokenizer-dynamic-20261027/reader-curriculum-callback/；Git 只存小收據與圖SHA，不存全文副本或截圖。

完成正文後，先保存 reader-original-01.json，不預期 pass；必要或可選問題均保留 id、來源引文、已教部分／缺少關係／實際影響與獨立發現時點。再存 current 以 basis_report 指本人原始／實際重讀來源；任何非「已足夠」concept checkpoint 都需本人逐項 closure，原notes不改。完整 source／figures、session_evidence opening/notes、閱讀範圍、限制、自己的 ai_estimate（整頁理解分鐘上下限和reason，排除執行／下載／審閱紀錄書寫時間）不可省略。估時是AI估計，不稱人類實測。

報告放 docs/course-review-20261027-tokenizer/reviews/reader-curriculum-callback/。可用本輪 tools/check_reader_report.py 的 --help 自查（位於 docs/course-review-20261019/tools），只查格式不代判。發現必要缺口先回交，作者修改後由你本人另存複查。不改受審內容、其他報告或工具，不開代理、不提交、不發布。最後回報判定、問題、原/current/closure路徑、實際閱讀範圍與估時。
