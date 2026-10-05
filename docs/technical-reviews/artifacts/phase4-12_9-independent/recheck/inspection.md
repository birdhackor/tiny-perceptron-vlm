# 12.9 本人修訂後複查

身分仍為 `/root/phase4_factual_coordinator/factual_12_9`，沒有改用其他審閱者。原問題、初稿 revise、原始執行與工具事件全部保留；自己的初稿副本 `own-initial-revise.json` 為 SHA `61c337daa4b6362ed7757481d7decac74fe625860af443c5931674650a77fa38`，協調者另外保存的正式 history 路徑也記於新報告。

本人重新親讀目前 `course/chapters/12.md#12.9` 第 265–307 行，包含全文、原 fence、練習與 details。新版原 bytes 完整存 section.md，SHA `0930ed04d140c96d076a3ca8af5d3334f20d684293d2f6eaff8f04830d991149`。新舊 raw section 的差异精確只有第 295 行段落；source.diff.txt 保存差異。原 fence SHA 仍 `f0fc0a23c0bf898e6a45cbfef92f84fb49ea1df78ce01515be828819a78280e9`。

重新讀的原 JSON pointers 是 `/results/test/examples`、`/results/test/correct`、`/results/test/samples`、`/results/data/splits/train/records`、`/results/data/splits/validation/records`、`/results/data/splits/test/records`，沒有讀原 JSON scope、notes 或作者額外解釋。原完整 audio.json SHA 仍 `cb4f46e2643ef4a92c85795af54103efd8edc9d2d12832d5a405776c7b650419`。

本人重新以原 generated_ids 比對原答案 bytes、按實際 records 核 >300 Hz 規則；不是沿用協調者文句當作答案。真實重新彙總為 11/14；錯例 row 4、6、7 的頻率依序是 290、300、300 Hz。每個 split 的每個 frequency 都有且只有 (振幅,秒數)=(0.25,0.1)、(0.5,0.12)。時長與振幅完全共變，不能分離兩因素的獨立影響。新版只說三個錯例頻率與這個限制，沒有再宣稱獨立時長效果，原 issue 已解決。

逐一核自己的七個 claims 支持範圍：其餘六項入口、長度/詞表/目標、−100、teacher forcing、梯度/未更新/200 Hz、歷史 11/14 的原文都沒有改變。必要 data/model/multimodal/attention/recipe 原碼與 raw JSON 的目前 SHA 都與初輪本人檢查的 frozen originals 相等；前置兩 SVG SHA 也相等。本輪沒有再執行模型 forward/backward、既有模型評測、訓練、GPU 或下載。初輪確實執行的原 fence/短 CPU 檢查按相同 fence 及程式契約仍支持未變動主張；不是把未執行工作冒充本輪重跑。

只明確 copy 四個小型 extract 檔及本人初稿報告，沒有 copytree 或跟隨 runtime workspace symlinks。原初輪工具操作事件沒有抹除，未觸碰其他工作者資料。正式 artifact 樹沒有 .pt。

另由新 section 原 bytes 用 Markdown fenced_code 渲染 `current-artifact.html`，頁首明標 current source artifact 與新 SHA，沒有宣稱 8765 或 production parity。本人實際 view 1280×800 與 390×844 全頁 screenshot：修訂的 11/14、290/300 Hz、振幅/時長限制及梯度/訓練區分都可見；原 fence 和 rendered pre code 相等。本節仍無引用圖，手機 code 橫向滾動；這是此次修訂內容的畫面證據，不替 production 網站排版背書。

新 report 保留原 issue quote 與初稿 revise 指紋，改為 resolved 並連結本人 resolution-receipt.json。只有成功寫入本人完整新 report 並核 reviewer_task 後才執行單節 checker。
