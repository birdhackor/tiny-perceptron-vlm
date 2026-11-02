# 本人實際先備路線

真實讀者：/root/omni_reader_entries。依reader-entries-plan-v1先以supplement取得全部38個context頁；這些頁的正文及工具一併提供的折疊內容均已讀，並非零先備入口測試。未開作者PLAN/OUTLINE/CONTRACTS/SOURCES/project-context或其他讀者答案，未開README_en。初讀只評理解，沒有重證模型數字與執行正式訓練。

- first-steps：讀過清單/字典/迴圈、tensor軸與平均、機率與-log、矩陣加權、Linear、導數/梯度/學習率、backward與更新分工、kernel及路徑。W.3可由每人/每次平均辨認消失的軸；W.6大步可跨目標甚至變差。
- environment、asset-storage：知道.venv與.venv-natural分工，CPU/短程式與正式訓練不同；程式/資料/權重、Git/LFS/HF、推論包/續訓包、指紋與版本各自用途。
- 1.1、1.7、1.8、1.12：ID可逆但無字義；softmax共分母與平移不變；CE用原分數與答案、代價非錯題數；求導不改參數、更新後須重算。
- 7.1、7.17、7.18、7.26：角色與內容分欄、X/Y移一位、IGNORE只在Y；文章自帶下一字答案、問答需核對；示範/偏好/作答回饋、DPO與RLHF/PPO角色不同；候選刪除/缺資訊須改文字標籤，none非固定答案，家族不拆跨份。
- 9.11、9.12、9.14–9.18：候選意義共享打分、模型預測/程式動作分工；重排須保ID與標籤、增刪候選影響分母；Choice/info/rubric、CE/Brier、校準/決策/final分工、最大機率與集中度不同；成本門檻、pointer獨立分支與對稱none；coverage/risk及零接受未定義。部分章節以未分配更早節為前置，我只取當段已寫出的機制與此路線可用內容，不宣稱已順序學過所有中間章。
- 12.17–12.18、14.19：局部音塊支持/錨點與共同時間、固定序列不隨移時重排；固定單碼本最近向量與有損重建；物理t/h/w旋轉與因果index分工。
- 16.15–16.16、17.16–17.17：state-only記憶M唯讀共享、交叉注意力與分支污染負控制、KV/卷積/循環狀態不同；流式尾部要接續且不能讀未來；STE近似梯度、三路loss更新對象、codec凍結後產targets；RVQ補同框殘差與文字MTP不同，參考重建不等於條件自猜。
- 19.11–19.13、19.19–19.24：舊A/B/C成品和分項失敗；固定推論包與完整續訓狀態；新版A局部影音分支、同一canonical semantic record、完整TinyAV參考計算但無能力成績；固定codec→targets、GRU Talker教師強制與自由生成界線；新C byte入口/唯讀M/獨立候選/對稱none；gold路由與真路由整鏈分母。
- 20.1、20.14–20.16：Qwen3-VL+Whisper先轉寫再共用聊天歷史、無朗讀；Kev不是官方Jev且題內因果可影響重排，混合快取要保全部狀態；兩代Omni條件與解碼分工；MoE啟用量不等於儲存量；人工指標表不是成熟模型實測。

實看context圖：rewrite-W-3-axes、rewrite-W-6-gradient-step、rewrite-01-character-ids、rewrite-01-update、rewrite-07-17-prepost-routes、rewrite-07-18-feedback-materials、omni-12.17-shift-pair、omni-12.18-fixed-codebook、omni-14.19-axis-time-index、omni-16.15-readonly-memory、omni-16.16-stream-tail、omni-17.16-gradient-paths、omni-17.17-rvq-axes、p6-19-delivery-package-roles、omni-19.21-av-module、omni-19.22-sound-chain、rewrite-20-input-routes、omni-20-architecture。SVG依freeze固定snapshot渲染，PNG在/tmp/omni-reader-entries。兩張暖身圖原環境中文字方框，按root提供的純fontconfig重渲染後另記；完整網站桌面/手機排版未驗。所有外部研究與下載連結未開，不用外部答案補入口。
