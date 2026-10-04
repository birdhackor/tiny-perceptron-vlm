# 實際渲染與讀圖紀錄

本節 13.15 本身沒有 SVG。实际需要使用的前置圖只有下列兩張；兩張都先真正渲染 PNG，再以 tools.view_image 親自查看，不以 SHA 判斷可讀性。

## course/figures/ppo_clip.svg

命令：`inkscape course/figures/ppo_clip.svg --export-type=png --export-filename=docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/ppo_clip.png`。Inkscape exit 0；產生 PNG。渲染時有 PangoFT2FontMap / GtkRecentManager 的 warning，PNG 中中文字完整可讀，未觀察到缺字或裁切。view_image 路徑為 `/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/ppo_clip.png`。

親自看圖：左半優勢 +1，藍實線随 ratio 上升，1.2 之後平；灰虛線繼續升。右半優勢 −1，棕實線在 0.8 以下平，之後随 ratio 下降；灰虛線往左延伸。兩圖 ratio 刻度為 0.4、0.8、1、1.2、1.6，垂直虛線標 0.8/1.2。底下文字明說平台只是暫停額外鼓勵，新機率沒有被硬鎖。這讓我能理解 13.15 的 ratio=1 不觸發裁切，也不能據此宣稱 20% 的硬限制。

## course/figures/ppo_roles.svg

命令：`inkscape course/figures/ppo_roles.svg --export-type=png --export-filename=docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/ppo_roles.png`。Inkscape exit 0；同類 warning 未造成觀察到的文字缺失。view_image 路徑為 `/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/ppo_roles.png`。

親自看圖：三欄是角色、選卡任務中做甚麼、何時改變；四個橫列分别是黃色獎勵模型、藍色 critic/value、綠色 old log 機率、紫色 reference。獎勵模型看情境與卡，先教比較後固定；critic 先看情境並每輪教估計；old 只記作答當時機率且同一輪固定、下輪再記；reference 保存示範起點、整段不動。底句說 policy 才是最後選卡的模型。所有欄位及底句可讀；配色並非辨識角色的唯一依據，文字已標角色。這使我能把 13.15 程式裡 optimizer、no_grad 與 requires_grad_(False) 分別對上角色。
