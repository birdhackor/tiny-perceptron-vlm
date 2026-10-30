# P1審查後的具體契約

本文件補足PRODUCT-SPEC的定義；衝突時以本文件的具體定義為準。所有門檻都是開啟final前的目標，尚無新版能力結果。P1規格與正式訓練recipe／final開封是不同凍結，後兩者仍須在三輪教材審閱與development工程後完成。

## A的模組與輸入

新版A是助理產品的模組擴充：固定自家舊A的權重、tokenizer與前處理revision，新增隨機初始化的影音局部編碼／融合／有限答案head，再接有限聲音模組。舊任務仍走原A；不能稱舊LM重新預訓練或LM本身已學會影片理解。新版manifest分列每個artifact與路由。影音head的真預測是canonical semantic result；共享對話可記錄它，舊LM若潤飾文字不能覆寫用來驗收或發聲的結果。

video-audio-v1最低欄位：frames[T,1,32,32]、frame_ms[T]、waveform[N]、sample_rate=16000、audio_start_ms、duration_ms、有效長度與presence mask。T<=16，整個共同時間軸<=4000ms；聲音起點與完整有效區間必須在共同時間內。timestamps有限、非負、影格嚴格遞增；影格採sample-and-hold到下一格／片尾。事件標籤用發聲區段中心，整段事件都不跨形狀切換，中心離切換至少125ms。前置無影格、越界、超限、非法mask或非有限輸入明確拒絕，不靜默截斷。

聲音先切成200ms區塊，每塊獨立log-mel／CNN；採固定尺度或塊內正規化，STFT center=False，禁止沿用整段錄音mean/std或跨塊卷積／pooling。每個局部token附support_start_ms、support_end_ms與anchor_ms；支持範圍最多200ms（滿足原250ms上限），padding不改anchor。融合後token已讀別塊，不能再稱局部觀測。development必驗窗外波形擾動不改局部token。

反事實配對固定影格、waveform、token數、模態區塊序列化和causal index，只改合法audio_start_ms／物理時間；這表示同一段聲音相對畫面移時。影格始終同順序，不能按新時間重新交錯後仍稱只有位置法變了。物理時間與token索引是兩個座標；本例是離線觀察整段材料，不宣稱輸入逐sample因果。

family包含同一底層情境所有移時、時長／間距、位置、振幅及線索刪除；變體不得跨split。200個主要final家族各有一個預指定配對，兩題都對才算成功：至少170/200。四個時長×間距主面板預定各50個家族，面板內至少75%；同家族其他診斷不增加獨立n。刪圖、刪聲音各200個診斷，要求不可判定各至少90%。若實際生成器無法滿足面板且無含糊答案，開final前先修資料凍結並公開理由。

只用dev選representation／seed作升版候選，再以固定85/75/90目標評它；其他五個正式下游訓練也全報，不看final換seed。兩表示三seed共用舊A時，只是三次下游訓練，不是三次獨立預訓練。研究差異按家族聚類與seed分別報。

## codec、Talker與整鏈

採VQ-VAE的loss更新配方，不用EMA：z_st=z_e+stopgrad(z_q-z_e)；重建梯度更新encoder/decoder，codebook loss更新碼本，beta=0.25的commitment loss更新encoder。padding不進loss／碼本統計。K、latent width、block hop、decoder state、碼率／EOS與最大長度在development-only profile後凍結，再正式訓練；P1不假稱已由profile決定。

codec的encoder/codebook/decoder及前處理全部以dev選定並凍結hash後，才產Talker train targets。Talker optimizer不含codec；eval/final不更新碼本、統計或死碼重啟。teacher forcing只用於訓練；正式生成從BOS自主預測到EOS／上限，絕不餵參考codes或前綴。block decoder只發出不受未來code改寫的樣本；公布hop、lookahead、state reset／flush，驗逐塊／整段等價與未來code擾動不改已發prefix。只有實際可播放chunk介面才量首段等待。

聲音程式以完整頻率／事件時值／停頓組合作family，phase/amplitude/noise是同family變體；codec與Talker都看不到final組合。Talker targets可採固定公開初相位，避免把同條件下任意相位當唯一答案；重建仍評原相位。固定生成器不用Qwen/Kev資料、權重、輸出或增強。

分析器在development固定：依參考事件區間算重建SNR=10log10(sum(ref²)/(sum((ref-reconstruction)²)+1e-12))，不重對齊／調增益，不用預測有聲區選分母；每個非靜音程式先算SNR，再按family等權平均。純靜音另報false sound，不能提高平均。生成評事件的頻率、順序、onset/offset、時值與必要停頓；預設頻率容忍5%、時間容忍30ms，extra/missing事件計錯，整個程式所有條件均對才成功。分析器須先通過解析正例、錯頻率／順序／節奏、漏／多事件和全靜音負例，再凍結。不得因模型失敗而放寬容忍。

合法輸出是有限數值、8kHz、<=2秒、格式／code值合法且分析器辨認<=4事件；100%是本次final目標，不是所有未來輸入保證。SNR>=10dB、gold條件程式成功>=85%、實際A整鏈>=80%分開驗收，前兩者不推導第三者。

公開semantic→音型字典先凍結。整鏈用每個A有解final家族的預指定題，實跑A→映射→Talker自由生成→decoder→分析器；分母含A錯答／拒答、解析錯、靜音及生成錯，不只挑A答對的題。刪線索的合法unknown獨立報，不用靜音拉高有解整鏈。分列A、映射、gold條件、真條件、整鏈及合法率；若音型已在Talker train見過，就稱未見AV輸入接已見音型，另由Talker自己的組合留出測新聲音組合。

## C的新介面與統計

C-v2直接收state、typed question及question_text、實質候選stable ID/自足description；不經舊C/B parse_state。UTF-8 bytes使用完整0–255及獨立PAD／特殊ID；非法surrogate、空必填字串、超512/128/128 bytes明確拒絕。Choice有1–7個實質候選，加一個none總數<=8；ID及完整description不得重複，不支援跨候選指涉。info固定yes/no，rubric固定0/1/2；每題唯一gold，語義等價重複項需先去重而非硬挑其中一項。平手按stable ID固定規則，不能按清單位置。

M=state_encoder(state bytes)，完全不讀問題／候選；各分支以question/candidate query cross-attend唯讀M，產生h_i／s_i。none在分支後讀masked mean/max/count等對稱彙整及question；實質候選不讀彼此。對稱只保重排，不保none答對。增加候選會改softmax分母和none；不要求增減時機率保持不變。固定fp32、eval、dropout=0下，重排一一對應及重算／共享M／中間態／機率容忍1e-5；Q1→Q2→Q1不得污染。cache key綁state bytes、encoder權重／tokenizer／前處理revision、padding／position／dtype/device；只用凍結推論，任一變動即失效。

family由可見任務事實與指定等價關係建立；同情境的問題、缺欄、換說法、翻譯、候選增減／重排同split。不可見entity/random ID不製造新family；完全相同可見輸入及token觀測跨split衝突必須為零，重複觀測單列或去重。多entity資料的指涉／欄位須真的影響至少一個問題，不能只塞無關文字湊獨立n。source/parent/family/split/generator lineage及跨模組交集都要audit；同生成器新seed只稱模板／組合留出。未見措辭、組合、概念三者分開。

C核心final為60家族，每家族預指定Choice/info/rubric各一個代表題；每型n=60但三型互相關聯，不加總成180獨立樣本。Choice>=54/60，info/rubric各>=51/60。主要Choice面板事前固定為48個有支援且資訊完整的動作情境、6個正解不在列表、6個支援世界內有明定required-field契約的缺資訊情境。核心不含info/rubric無法定義的未知任務；unsupported只在獨立Auto100中的Choice／路由診斷，不硬貼no/grade0。具體清單、同family次要變體及各面板交集在生成器freeze列明，不能由模型預測選分母。unsupported與列表none原因分列；單一none輸出不能宣稱已識別兩者。

全域T在T=0.25×2^(k/8)、k=0..32固定網格上，以calibration80家族三型代表題的等家族／等型NLL選取；相同NLL選最接近1者，再同分選較小T。報候選數2–8的範圍，raw與calibrated分列，不保證改善。policy60從tau=[0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.975,0.99]選coverage最大且empirical accepted error<=5%的規則，同coverage選較高tau；無可行tau就記無可行策略。接受只讀模型輸出：校準p_max>=tau、非none且資訊足夠預測為yes。評分時accepted error定義為已接受列中Choice錯誤、gold世界不支援、或gold資訊不足任一成立；缺資訊被誤判yes即使Choice恰好對仍算誤收。coverage分母始終是全部60個預定主要Choice，none/轉交留在分母但不算自動接受。主要目標accepted>=30且errors/accepted<=5%；診斷交叉面板分列n，零接受risk未定義。B深度沿用固定舊policy，不用此policy集重挑。小樣本區間以family為單位；經驗5%不當作未來保證。

## Auto與保留能力

C核心60以外，另有100個預先留出的Auto整合家族；同一100家族成對跑gold/真C路由，不拿60的variants湊100。兩面板及A/聲音面板的共同祖先與交集都列出，成本均預留。C路由只讀文字及附件存在資訊，不讀gold task或影音內容；人工指定仍需過輸入契約。

B adapter只將公開支援、無歧義、值域／事件數／tokenizer／max_length都合法的輸入轉到原文法，其他拒絕／轉交；原B權重、config、tokenizer與depth policy固定。舊算式世界只有84個獨立三數家族，不湊100、不擴值域。保留能力面板跑新版wrapper與固定舊版，逐任務列correct/n/差值；歷史family的新措辭只稱retention regression，不稱新unseen。任一主要舊任務退步>5pp就保留舊版預設。

## 費用、凍結與發布

在12個正式jobs外，development profile、targets產生、infra、final、輸出傳輸與收尾都入同一US$60新round，先預留核心final。每job在P4發起前有實際rates／CPU/RAM/GPU、最長updates/batch/media/tokens、硬runtime／absolute deadline、啟停與preemption重放、傳輸／保存期間上界，且經離線預約守門。沒有development實測前不編造吞吐或保證60美元足夠；不能在開機後才算帳。

P1採純client Modal inventory／GitHub CAS初始化，不啟容器。新版權重與完整optimizer/RNG包採已批准的GitHub Release版本化資產通道，先從Actions標準token驗證draft傳輸；本機uploads host credential不適用的失敗保留。舊HF權重固定revision不變，新HF鏡像可選。P4小型新模型checkpoint用Modal結果blob回到Actions，不建新長期named Volume；如實作需要新持續儲存，必須先列完整ownership／retention／費用／cleanup契約，不能臨時偷加。

凍結依序是資料／generator與split、度量／分析器、模型／recipe與最大嘗試、dev選出的候選／codec／T／tau／音型映射、首次final receipt。工程修錯只用development，保留舊新hash、理由、時間及final尚未開封狀態；能力差不算量測錯誤。final開封後不換seed、不改門檻，修錯另列研究輪次保留原結果。三輪教材審閱、工程測試、能力升版、公開發布是不同事實。

## P1審查處置

S01：以P4逐job上界與核心final保留為正式派工門檻，P1不宣稱profile已做。S02：C60與獨立Auto100。S03：可見語義family／跨模組lineage與觀測去重。S04：局部200ms前處理與雙座標／固定序列化。S05：200配對分母／四面板／dev選候選。S06：loss配方、凍codec、BOS自由生成與chunk。S07：事件時值停頓／SNR及合法率。S08：真A結果與整鏈全分母。S09：C-v2新入口／byte與候選契約。S10：唯讀M／對稱none／cache失效。S11：三型60、T/tau、接受定義與風險。S12：模組A／B84母體例外／適配。S13：四階段freeze與開封規則。O01無時間訓練控制延後；現輪先保留核心12jobs和final，CPU/chance/線索刪除對照足以支持指定範圍，不新增未批准架構網格。
