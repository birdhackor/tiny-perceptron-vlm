# index 獨立技術核對筆記

審閱者：`/root/tech_index_tokenizer_20261027`；日期：2026-10-27。

完整讀取 freeze-01 的 `index` canonical slice。原來源是
`scripts/export_course.py` 的 `home_introduction`；snapshot 僅是凍結原文，
不把它當成原來源。首頁沒有 SVG、程式圍欄、研究結果或效能比較。

本文的實質主張是目前教材及網站提供什麼，不是在首頁論證某個模型方法。
因此查本專案的直接來源與實際產物，不附不相關的 BLT／H-Net 引文。

核對位置與範圍：

- `course/chapters/01.md` 的章首敘事與 1.1 原文：由字與編號、問題與答案，
  到可調數字表；索引的 1.4、1.5、1.12 對應接字猜法與參數更新。
  這支持「用接字表、具體數字與短程式，逐步解釋」的路線說明，
  不是重新驗收全章的數學正確性或教學效果。
- `course/chapters/10.md` 章首：像素、圖塊與位置轉成數字，
  10.7 展開圖片位置，10.8 組成 MultiModalLM。
  `course/chapters/12.md` 章首：聲音表示、聲音入口，最後與文字共用對話。
  支持「之後再看圖片與聲音怎麼接進同一套系統」的內容預告，
  不把程式接通當作模型已學會看圖或聽懂。
- 完整閱讀 `course/README.md`：R.1 明列閱讀及本機／Colab 路線；
  R.2 的主題表有文風／安全、圖片、儲存與壓縮；R.3 明列第 8、9、10、17 章。
  `course/first-steps.md` 的入口及 W.1 確為動手時才安裝的暖身操作。
- `scripts/export_course.py:223` 起：附執行副本時才承諾「已有實際結果」；
  未附執行副本時清楚寫出沒有結果。`checked_notebook` 比對所有 cell
  型別與程式／文字原文，要求 code 有 execution_count 且無 error。
  輸出器在每個教材程式格後插入 `output_html`，並在每節加入 `section_footer`。
  本次自行逐份核對 313 份現行執行副本；400 個教材 code cell 中 341 個有
  可渲染輸出。其餘靜默賦值／斷言並非有意義的預期輸出缺漏。
- `scripts/build_course.py` 的 `build(check=True)` 實際執行通過，
  證明原始課文、目前 Notebook、索引及 lessons.md 同步。
- `zensical.toml` 全量解析的 lesson nav 與 313 個現行編號完全相等；
  theme 有搜尋、導覽、手機抽屜；CSS 補充窄螢幕的側欄行為。
- `.github/workflows/pages.yml` 的發布工作先核對審閱、執行獨立 CPU kernel，
  再以 `--executed` 匯出與檢查。只讀此流程，未觸發 Actions 或發布。
- 本輪本機站點 `outputs/tokenizer-20261027/site` 的 revision 是 `main`，
  不能當成公開站已部署本輪工作樹的證據。

執行與視覺證據另見 inventory-output.json、site-check-output.txt、
browser-output.json。瀏覽器探針適配了目前 Zensical 的開放 shadow DOM 搜尋、
絕對化 href、搜尋結果的 `?h=` 參數、標題 permalink 及實際頁首選單 selector；
適配前的斷言失敗是探針 selector 假設，不是教材或站點缺陷。

未驗證範圍：Colab 雲端執行、公網連結與部署、讀者裝置的安裝步驟、
其他瀏覽器、每節圖文版面、教學理解效果、模型品質、完整訓練或論文效益。
本輪沒有扮演首次讀者，也沒有讀作者 PLAN、規劃對話或 reader 結論。
