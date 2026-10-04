# 原圖圖框：作者呈現證據

作者：`natural_final_photo_diagrams`（圖片呈現協作作者）。這是作者本人對圖片呈現的自檢，沒有擔任 reader 或 fact reviewer，也沒有閱讀其他人的評判。

作者先以 `tools.view_image` 查看 `test_00729.jpg` 與 `test_01103.jpg` 兩张原圖，再製作 `course/figures/natural_actual_bike.svg` 與 `course/figures/natural_actual_glass.svg`。兩個 SVG 都完整嵌入原 JPEG bytes，沒有 JPEG 解碼後重編碼、裁切、修圖、重畫或圖片內標記。圖框的顯示縮放不改變模型輸入。

來源為 [DOCCI／Google](https://google.github.io/docci/)，授權為 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。每張照片的 sample id、來源、授權 URL、原始尺寸、位元組數與完整 SHA-256 保存在 SVG metadata 及 `actual-photo-verification.json`。

兩張圖在實際 Chromium 151.0.7922.173 中，各以 720px 與 358px 顯示寬度渲染，作者以 `tools.view_image` 逐張查看全部四個 PNG。兩行標題與兩行來源文字沒有出界，照片完整可見；原圖中本來被畫面邊界截斷的物件保持原樣。所有 SVG 文字的字號至少 32px；358px 顯示時按整個 SVG 的比例縮放，最小實際字號約 15.91px。圖片顯示邊界與原始長寬比相符，沒有 `clipPath` 或 `filter`。

| sample id | 原始 JPEG 大小 | 原始尺寸 | 原始與嵌入 bytes 的 SHA-256 |
| --- | ---: | --- | --- |
| `test_00729` | 112881 bytes | 1024 × 768 | `8fff2387a62580e6716d9e312ffd32e5058ae07af051d466c7aff941f6a94b60` |
| `test_01103` | 107589 bytes | 767 × 1024 | `6db6e5306f9ee2f46e86667cc87705a59c40826b45773a3d3397ed60ca3ee9f7` |

`actual-photo-verification.json` 保存原圖 SHA、嵌入解回 SHA、逐位元組相等結果、SVG SHA、圖片邊界、各文字實際邊界、瀏覽器版本及四個 PNG 的尺寸與 SHA。可重現的製作與驗證程式、HTML 包裝、PNG 截圖及同一份 JSON 位於 ignored `outputs/natural-extension/actual-photo-diagrams/`。

本作者只寫上述兩個 SVG 與 `actual-photo-*` 證據；沒有修改 index、書、報告、程式來源、環境、Git 或 GPU 狀態。後續獨立 reader 與事實核對由整合作者安排。
