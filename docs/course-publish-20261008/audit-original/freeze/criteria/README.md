# clear-tutorial：通用準則與 VLM 專案參考

使用入口為 [SKILL.md](SKILL.md)。通用寫作、例子與審閱判準放在主入口及配套 prompt；本課背景、操作與工作順序另放專案參考，已知問題保留在歷史資料。

| 文件 | 用途與使用者 |
| --- | --- |
| [SKILL.md](SKILL.md) | 作者與審閱者的通用教學與驗收準則 |
| [寫作 prompt](references/writing-prompt.md) | 寫作時使用的通用指令 |
| [審閱流程](references/review-protocol.md) | 協調者安排逐段閱讀、獨立初判、交叉覆核與歷史比對 |
| [跨領域校準例](references/calibration.md) | 用不同領域的例子校準證據充分性，不提供待審教材答案 |
| [VLM 專案背景與核對](references/project-context.md) | 作者、協調者及技術審閱者按需要查核本 repo 約定 |
| [UPSTREAM.json](UPSTREAM.json) | 固定來源、本地基準、原始檔案雜湊與歷史快照 |

首次閱讀者只取得通用判準、實際入口背景與當前閱讀包，不提前取得專案技術參考、舊問題、同款案例或修正答案。各角色先保存來源初判，再交叉覆核，最後由協調者比對歷史問題。

## 專案約定的採用與取捨

2026-10-08 依使用者要求整合 learn_to_yolo 的通用準則，並核對本 repo 的公開入口、編輯與發布約定：

| 約定 | 處置與理由 |
| --- | --- |
| 高中生／大學生背景與按需暖身 | 保留本課設定，以[實際閱讀路線](../../../course/README.md)與使用者指定為準；不搬入 YOLO 的 Python class 或 NN／CNN 經驗設定。 |
| 先教學、三輪審閱、再處理成品工程 | 保留六階段原規約與本輪計畫的先後關係，移到專案參考並標明適用範圍；局部維護不自動重啟完整六階段。 |
| 主線神經權重自行訓練，成熟模型作延伸 | 保留；這是第 19／20 章的成品分工與驗收約束，上游能力不能算成本課從零學會。 |
| 成熟知識與本課小實驗分開 | 保留；方法依原始來源教學，小實驗只支持其實際呈現的機制或能力，不用微型得分決定業界知識。 |
| 正文、Notebook、網站同源，審閱綁定版本 | 保留；本地已有來源、圖與凍結判準檢查，不能搬入 YOLO 的課程索引或證據目錄取代它們。 |
| 按改動做相稱驗證與實際頁面檢查 | 採用通用準則，沿用本地操作；區分內容、能力、頁面與公開材料的核對，不因 skill 或文句更新重跑訓練。 |
| 固定來源與雜湊追蹤 | 採用；雜湊描述 pinned upstream 原檔，與本地適用版分開，方便日後比對而不覆蓋專案約定。 |
| YOLO 座標、IoU／AP、版本線、section-map 與發布 tag 約定 | 不採用；與本課目前來源結構和成品驗收不相符。相關的通用比較、表示與證據原則已由主入口保留。 |
| active skill 中的 10.5 既有問題案例 | 移入歷史快照；保留查核價值，避免把待審領域答案當成首讀提示。 |

## 來源與維護

通用主入口、審閱流程、寫作 prompt 與校準例採用 [learn_to_yolo 的固定版本](https://github.com/birdhackor/learn_to_yolo/tree/147c0e986c54e440d75daf34254f48bcfa05c8f4/.agents/skills/clear-tutorial)。本次以 `tiny-perceptron-vlm@5224564f52b0bbe04c6958ea2693ad5afd188420` 為本地基準；原 skill 最近變更為 `3c0f3254d4762f5ed87e77bd55e5eaf9be2ec248`，也曾參考較早的 YOLO `19520ba`。這次重新追蹤直接來源，不照抄來源 repo 自己的 UPSTREAM.json。

更新前的四個檔案逐位元組保存於[本地 skill 歷史快照](../../../docs/reader-reviews/skill-history/clear-tutorial-3c0f3254/SKILL.md)，包含 [10.5 案例](../../../docs/reader-reviews/skill-history/clear-tutorial-3c0f3254/references/review-example-10-5.md)。它們是歷史方法資料，不是新版驗收證據，不提供給下一輪首次閱讀者。

同步來源時先比較差異，再評估專案參考；來源檔案的雜湊不能冒充本地檔案雜湊。只改 skill 不表示教材已改寫或重審。本地發布檢查會核對凍結的 skill／protocol 指紋；判準更新後，舊批次不能直接滿足新版發布條件，須另做適用範圍的新審閱，不修改舊指紋冒充完成。
