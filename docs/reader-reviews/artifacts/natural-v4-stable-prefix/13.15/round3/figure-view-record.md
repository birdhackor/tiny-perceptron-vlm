# Round 3 真正渲染／再次親自讀圖

本節13.15本身沒有SVG；我在本輪實際必要前置中使用13.13、13.14與新增直接連結7.17的三張圖。圖SHA來自原SVG bytes，實際讀圖來自下面的view_image，不用SHA替代看圖。

## PPO裁切圖

原SVG course/figures/ppo_clip.svg 的當前SHA256為4783554593c9f68ef2a50ba671a82cced05340812f1df7652f6ad22439aadfcc，与我最初Inkscape渲染時記錄一样；因此本輪沿用既有的ppo_clip.png，沒有重跑Inkscape。原實際渲染命令与讀圖记录保留于上層figure-view-record.md。當前SVG原bytes另存本輪ppo_clip.svg。

本輪再次view_image：/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/ppo_clip.png。我親自看見左圖正優勢+1，蓝線在ratio1.2後平台；右圖負優勢−1，棕線在ratio0.8前平台。灰虛線示未裁切方向，0.8/1.2虛線與文字都清楚，底句說新機率沒有硬鎖。這讓我把本節ratio1.0理解為第一次step前尚未偏移，並不把平台誤讀成參數上限。没有文字裁切/遮住或圖文理解阻礙。

## 四角色圖

原SVG course/figures/ppo_roles.svg 的當前SHA256为ea7619a702fa3965caa4c78756febe8c71b1b281dd21e800b759e9dbb6d8b2af，与本人原渲染時記錄一样；沿用原ppo_roles.png，本輪沒有重跑Inkscape。原渲染命令在上層figure-view-record.md，當前SVG另存本輪ppo_roles.svg。

本輪再次view_image：/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/ppo_roles.png。我重新按三欄四列讀出reward先學比較後固定、critic每輪學估計、old log同一輪固定下輪重新記錄、reference整段保留起點。底部policy是最後選卡模型的句子可讀；有文字角色標籤，不必只靠四色區分。本節短例的reference是随机policy起點副本，完整示範訓練流程中的reference则保存其示範後起點；都以整段固定作对照理解。没有觀察到新障礙。

## 7.17的預訓練／後訓練階段圖

這是本輪新讀的必要圖。原SVG course/figures/posttrain_stages.svg，原bytes SHA256为24dab47bbd625020b1269c2609730488964dd66367dcbceeb4f6bc0b4f6cac2c。本輪真執行：inkscape course/figures/posttrain_stages.svg --export-type=png --export-filename=docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/round3/posttrain_stages.png。exit0并生成PNG；完整argv/stdout/stderr保存于posttrain_stages-render-execution.json，SVG原bytes也另存本輪posttrain_stages.svg。

本輪view_image：/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-stable-prefix/13.15/round3/posttrain_stages.png。我看到左上一般文章提供前文→原文下一項的預測目標，箭頭到綠色預訓練底座：保存權重、可另存副本再继续更新。底座向下分到客服/程式/圖文三個助理，各框都写自己的资料與評測。底句說後訓練可學任務能力、效果需由新題驗證，不能由箭頭保證。標題、各框字句與箭頭完整可读，没有缺字或遮蓋。此圖支撑我理解13.15新增的起點區分：LLM SFT接續已有底座，当前四卡小網路只借用示范教法、從隨機開始。
