# 有限助手訓練資料

這個快照只支援教材明定的有限情境。文字、工具對話和字卡組合由本專案撰寫，採專案 MIT 授權；未使用成熟模型生成答案。

服飾像素取自 Fashion-MNIST（Han Xiao、Kashif Rasul、Roland Vollgraf，Zalando Research，2017），固定上游版本 b2617bb6d3ffa2e429640350f613e3291e10b141，採 MIT 授權。快照保留原始授權與出處。只取褲子、包、短靴三類；左右／上下位置由本專案合成。

繁體字卡使用 Noto Sans/Serif CJK TC，固定版本 f8d157532fbfaeda587e826d4cd5b21a49186f7c，字型採 SIL Open Font License 1.1；保留原始授權。快照包含渲染字卡，不包含字型檔。

真人中文錄音取自 PolyAI MInDS-14 zh-CN，固定版本 40ce77cb32a384e4d50a568e1ec39ac804019d33，採 Creative Commons Attribution 4.0： https://creativecommons.org/licenses/by/4.0/ 。原始 model/data card 及專案 attribution 隨包保留。來源： https://huggingface.co/datasets/PolyAI/minds14/tree/40ce77cb32a384e4d50a568e1ec39ac804019d33 。本快照選取 107 段原始 WAV；未裁剪、重新錄音或換聲；新增格式與續答對話。錄音只訓練地址更新、App 問題與卡片問題三類；專案提供的短答是教學例句，並非銀行正式操作流程。沒有 speaker/session ID，不能宣稱測試是未見過的說話人。

所有資料沿來源群組切分 train、validation、test。圖像、錄音與文字示範標籤是訓練／評估資料；推論只讀使用者訊息、公開位置參數與實際像素／波形。公開資料授權不等於範圍外能力已通過驗證。AI 原創內容及自動檢查沒有被標成已通過人工校對。
