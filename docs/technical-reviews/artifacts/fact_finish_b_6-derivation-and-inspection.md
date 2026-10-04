# B.6 本人推導與來源定位

reviewer_task=/root/fact_finish_b_6；本輪只審B.6，未讀他人審稿結論。
讀時source由scripts.check_technical_reviews.sections取未正規化UTF-8段落：B.6在0B.md第六個##，不是首##，無intro要求。已完整親讀B.6、B.5、7.3、7.11、B.1、6.1、B.7、W.1並保存各原段。

ByteTokenizer有8個special ID，ordinary ID=byte+8。TOOL為ASCII [84,79,79,76]，因此ID=[92,87,87,84]；assistant的content後加EOS=2，有效labels=[92,87,87,84,2]，4+1=5。ASK=[65,83,75]+8=[73,91,83]，加EOS為4。render_chat把原ids去末項作x、原targets去首項作labels，故TOOL的五個targets位於127..131；x和labels都是132位置int64。ASK將最後assistant内容縮一byte，x=131，labels有效4。decode只取ID>=8，EOS=2自然被略去。user問題UTF-8內容仍在x而不是-100，-100只在y。

交叉熵對logit位置n的直接loss為1[y_n!=-100]*(-log softmax(logits_n)[y_n])。無class權重時mean分母為有效target個數；本項sum再除count也相同。忽略位置對該位置logits直接梯度=0，不能推出前文embedding無梯度：後續答案的causal attention仍讀前文。CPU小模型中有效logits梯度和>0，user-only數字1的embedding梯度norm>0，保存實值於execution；這只是機制算例。

SFT：親讀Ouyang et al. arXiv:2203.02155v1 (2022-03-04) §3.1 Step 1與§3.5 Supervised fine-tuning，作者明寫labeler demonstration後fine-tune pretrained GPT-3。B.6前置7.11分清基模接續和random baseline；本節正式probe亦明寫從隨機權重開始，不能借用InstructGPT能力或把這次說成載入既有基模。

Toolformer：親查2302.04761 abs submission history只有v1 (2023-02-09 16:49:57UTC)，並下載v1 PDF親讀abstract、§1、§2 Approach全部、§3與§4開頭。§2/圖2先sample位置和API呼叫，實際執行API，再以L_i(z)=-Σ w_(j-i)log p_M(x_j|z,x_1:j-1)比較L_i^+=L_i(e(c_i,r_i))及L_i^-=min(L_i(ε),L_i(e(c_i,ε)))，只保留L^- - L^+ >= τ_f，合併到原文本後用standard LM objective fine-tune。Inference看到→才中斷執行API並回填。這与B.6直接人工三卡labels、assistant-only mask、random TinyLM訓練不同；B.6只說也研究工具學習且未重做方法/成績，引用範圍成立。沒有把研究標題當自我認知的證據。

正式raw與模型核對：原bytes保存formal-tool-choice.json與dataset.json，全部928筆原records（44/5/6家族的704/80/96筆加48 paraphrase）按可用/停用配對產生464行完整讀取投影，本人全部讀完；沒有CALC/COPY user prefix。所有fields與親讀程式重建records精確相同。checkpoint SHA及state SHA匹配formal，29個float32 tensor逐項保存shape/hash/finite及strict load equality，step和optimizer state=900。900次每批24的seed42 sampler有效labels數重算121297。224個validation/test/diagnostic原生成逐筆重播，輸入ID、生成ID、EOS精確相同；只證routing卡，沒有工具參數、工具執行、final answer或自我認知測量。B.7的六個新工具問法全ASK也確實可重播，本節已要求新問法另查。

FP32記錄：第一次用1e-8絕對容忍度檢查保存final NLL失敗，差1.6100121417431026e-8。保存first-replay-failure.json，未改教材/raw/weights。FP32減法和累積只當近似，以1e-7絕對誤差核對初/末NLL；初始差5.3945e-8，末差1.6100e-8。此浮點近似與B.6未聲稱的NLL數值無衝突，生成/label/weights hash核對仍用exact。沒有宣稱loss bitwise重現；未確定差異的底層原因。

執行界線：documented CLI以.venv/bin/python在repo根目錄啟動（對應W.1啟用後python），5秒timeout只測bootstrap，已生成完整同內容dataset。另本人短CPU fit_lm只更新1步並保存其結果，不重跑900次訓練、不測GPU速度、不改env。CLI完整路由由原碼run.py→tool_choice.run及原900-step checkpoint/raw支撐，不將timeout寫成complete_run。

圖：B.6沒有直接圖；B.5圖額外完整親讀XML。核對root先render的manifest SVG與PNG SHA，再以view_image本輪親看持久copy。各條件、卡片、箭頭、人工策略footer均吻合。詳見figure-receipt.json。
