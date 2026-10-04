# Round 2 當前 SVG 核對與再次親自讀圖

13.15 本身仍沒有圖。我完整重讀必要前置 13.13/13.14 後核對兩個原始 SVG SHA，均與我初輪實際渲染時保存的 SHA 一樣，故本輪重用自己的原 Inkscape PNG。沒有重新執行 Inkscape，沒有把沿用渲染說成本輪重做；本輪對兩張 PNG 都再次呼叫 view_image 並親自看整圖。原渲染命令與初輪讀圖記錄保留在 figure-view-record.md。

## 裁切圖

原 SVG：course/figures/ppo_clip.svg。當前與初輪 SHA256 同為 4783554593c9f68ef2a50ba671a82cced05340812f1df7652f6ad22439aadfcc。再次 view 的路徑：/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/ppo_clip.png。

本輪看見左側 +1 優勢的藍線在 ratio 1.2 後變平，虛線繼續上升；右側 −1 優勢的棕線在 0.8 前變平，右方繼續下降。0.8/1.2 的虛線位置與上下文字一致。底句明确说明新機率没有被硬鎖。沒有觀察到缺字、遮蓋或切掉標籤。我把本節 ratio 1.0 放在兩門檻中間，理解第一次 step 前既沒有機率偏移，也不能從這個輸出推出 20% 硬限。

## 四角色圖

原 SVG：course/figures/ppo_roles.svg。當前與初輪 SHA256 同為 ea7619a702fa3965caa4c78756febe8c71b1b281dd21e800b759e9dbb6d8b2af。再次 view 的路徑：/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/ppo_roles.png。

本輪重新按每列閱讀：reward 看情境與已選卡給這次分數、先學比較後固定；critic 看情境估預期回饋、每輪學；old log 保存收集時機率、同輪固定而下輪重新記；reference 保留示範起點、整段不變。下方 policy 是最終選卡模型的句子可讀，欄位間沒有混淆。這次 SFT 已在本節解釋，因此「保留示範訓練起點」也能直接與前述監督式微調連起來。圖文沒有新阻礙。
