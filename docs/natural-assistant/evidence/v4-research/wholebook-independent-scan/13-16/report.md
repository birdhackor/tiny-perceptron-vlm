# 第13–16章独立逐节预检

实际任务身份：`/root/v4_wholebook_scan_13_16`。本报告是整本教材检查中的既有章节切片；不涉及仍在写作／训练的第20章，也不是整本最终验收。

依序完整读取第13–16章全部47节，读法模拟数学不错的高中生／有基本数学的大学生初次接触LLM；不是假称真人学生调查。先读教材，再检查本章直接依赖源码与重要原始资料，未读取旧review verdict或作者历史来补文本。

主线机制与小例子多数能接上，未发现确定的核心公式错误。初读发现第13章路线的13.5前置冲突、一处正規化圖標籤和少數修訂語句，root已修正且下方完整重读closure確認关闭。正文run方法與統計過密仍是P2编辑建议。以下初读发现保留，便于核对改动理由。

## 可执行建议

### P1：R13-route

位置：`course/chapters/13.md:5, 517, 552`。

導讀給PPO路線13.1–13.2 → 13.10–13.16 → 13.17；但13.16與13.17均要求讀過13.5。13.17又說接著回13.3–13.6。沿指定路線的初學者會先遇到被跳過的必要DPO代價。

建议：選一種一致安排：最小修法是导讀將13.1–13.6列為共同基礎，再選讀13.10–13.17；若要保留PPO獨立路，將13.16的DPO結果標為可選对照，13.17用兩卡例补齊log与margin並取消13.5必需前置。

验收：按導讀逐節走一遍，不需在未說明處回補另一支線；每節前置均為该路線已读或明列先補。

### P2：R13-revision-language

位置：`course/chapters/13.md:5, 501, 567`。

「新增的PPO實驗」「新增的兩行顯示」「與本章新增的有限選卡實驗分開」是在說修訂時間，對理解當前方法沒有增益。

建议：直接改為「PPO實驗」「後兩行顯示」「與有限選卡實驗分開」。13.17所說「新增偏好含示範階段沒教的條件」则保留方法含义，寫成「偏好資料包含示範階段未教的條件」，不可删掉SFT比較限制。

验收：學生不必知道教材曾新增哪些章節或輸出行，也仍能分清語言模型DPO與有限選卡PPO。

### P2：R14-normalization-label

位置：`course/figures/normalization.svg`。

圖把「gamma / beta」都放在「可學縮放」框；beta是加法偏移，正文已正确解释，图標籤卻使兩者看似同一種縮放。

建议：框名改成「可學縮放／偏移」，下行標「gamma乘、beta加（RMS常只有gamma）」或拆成兩個短標籤。

验收：單看圖也不會把beta當作乘法係數，且與正文LN／RMS約定一致。

### P2：R1316-experiment-density

位置：`尤其course/chapters/13.md:33, 503–513；course/chapters/16.md:36–44`。

核心小例子後，经常接精密run流程、五位小數、每批／每回合曝光次数和記憶體管理邊界；16.1在prefill等主題之前就連續解說TF32、六支訓練、allocator、reserved、子程序與全報告范围。這些細節可核對，但多次插入会讓第一次學原理的讀者把方法与本機記錄混在一起。

建议：正文保留一個能支持概念的真實結果、分母定义與关键限制；完整排程／指紋／精密逐題數值／backend清單／每支記憶體方法移至T.8或各節「實驗細節」可選區。13.1的家族切分保留，測前後是否算挑參數用一句說明即可；16.1留self/total、暖機及權重不等峰值，L4方法總卡放在效率實驗頁。不要刪除toy和真模型、候選排序和生成、記憶體起點和增量的概念区别。

验收：讀者先完成概念→小例子→現象→練習主線，只有想重跑／核對時才進入精密run資料；原實報與限制仍可回查。

## 每节实际阅读记录

| 節ID | 我理解到的概念與問題 |
| --- | --- |
| 13.1 | 偏好是同一問題、同一要求下的比較，兩篇都答對時格式也可決定勝者；家族切分能避免交換加數洩漏，但一長段測試流程例外解說會打斷最基本比較單位。 |
| 13.2 | prompt共享，chosen/rejected各自render成已shift輸入與答案目標；助手角色預測第一個答案、答案預測EOS，不能再shift或混搭兩篇前文；截短標籤的限制說清楚。 |
| 13.3 | 回答機率是條件鏈連乘，log把它換成有效答案格的加總，含約定EOS且忽略prompt/PAD；兩項0.7870得0.6193，程式實跑一致。 |
| 13.4 | reference是獨立複製並凍結的固定SFT基準，policy相對候選差距改善不代表正確候選已排前；clone、deepcopy、eval與凍結責任分得清楚。 |
| 13.5 | DPO對相對margin取負log-sigmoid，負chosen梯度與正rejected梯度使下降更新改進差距；共同模型權重與二選一排序／自由生成的差異有交代。 |
| 13.6 | beta既是KL理論目標係數，也在DPO局部導數中影響sigmoid飽和，固定正margin時梯度幅度可不單調；三組數值實跑一致，沒有把局部結果外推。 |
| 13.7 | 正文bytes與含EOS目標長度分开，資料偏好長篇和sum本身的長度效應不能靠偷換mean解決；控制支線只支持格式排序，生成仍未算對。 |
| 13.8 | 溫和修正可查證錯誤和粗魯反駁、無條件附和不是同一目標；字典只是偏好資料展示，還未訓練，正確前提與主觀喜好對照有學習價值。 |
| 13.9 | 配方對數不等於有效目標曝光量，未知None不等於失敗0，偏好改善要另測其他能力；自然資料截短與未測生成的界線清楚。 |
| 13.10 | 獎勵模型從同題分數差估计比較胜率，sigmoid差0／2的結果可手算；有限回答卡讀結構特徵且標籤由規則產生，沒有冒稱讀中文或真人RLHF。 |
| 13.11 | advantage是本次reward減收集時的value，當作固定評語教策略，critic另外學；一次選卡的bandit與逐token多步任務有明確區分。 |
| 13.12 | ratio是目前機率除收集時舊機率，old紀錄一輪內固定、下一輪更新，reference全程固定；兩格是兩次樣本，沒有假裝它們组成完整分布。 |
| 13.13 | min(rA,clip(r)A)在正A的高ratio與負A的低ratio側變平，錯方向仍被懲罰；六格目標與導數實跑一致，圖沒有把PPO平台說成機率硬上限。 |
| 13.14 | critic看情境估計策略預期reward，獎勵模型看已選回答；平方誤差與枚舉KL(p||q)分别更新估計／限制策略，四角色圖可對回各自凍結時間。 |
| 13.15 | 舊策略抽卡、凍結評分、固定advantage、新策略求導及critic更新串成一次；原程式實跑選0、ratio1、policy改變且RM/reference無梯度，但「新增的兩行」屬無用修訂語句。 |
| 13.16 | 長度獎勵可與真值／格式衝突；正式結果是RM排序正確但選卡仍沒完成句子要求，沒有把人工漏洞反例冒充實驗觀察；其13.5前置與導讀PPO跳讀路線衝突。 |
| 13.17 | DPO省獨立RM但仍用偏好標籤與固定reference，兩卡0.7／0.3可直接手算；同360更新不能稱同算力，其13.5前置再與導讀路線衝突，還留「本章新增」修訂語句。 |
| 14.1 | 成對向量按位置旋轉，共同平移保留角度差與點積，sin/cos與弧度有從單位箭頭導入；兩圖和旋轉程式一致，沒有保證超長外推。 |
| 14.2 | LN消去共同偏移，RMS只調均方根尺度，兩列結果能手算；正文分清gamma乘法／beta加法，但圖將gamma/beta合稱可學縮放，應修正圖標籤。 |
| 14.3 | 方向相同時點積仍受向量長度影響，L2正規化拿掉長度、另有scale調softmax尖銳度；特徵軸／候選軸-1的差別講清楚，未冒稱整模型加速。 |
| 14.4 | 矩陣線性層連乘可合成，activation提供非線性；GELU用常態累積比例、ReLU平方处理負／正值的規則和五個固定輸出可理解。 |
| 14.5 | SwiGLU的SiLU gate可負或大於1，逐項控制content，三張矩陣比普通MLP兩張多；等hidden不等參數及未等容量的正式比較限制明確。 |
| 14.6 | 輸入表／輸出表共享須是同Parameter，單純copy初值會各自變動；三候選打分與改表後4／1／3可追蹤，起點分布不同的實驗限制有說明。 |
| 15.1 | 同FFN權重供每個token使用，增加列數增加工作而不增加權重；up、relu、down首列4／5可手算，玩具12參數與真模型規模分得清楚。 |
| 15.2 | expert是獨立儲存的規則，剛初始化並無學科專長，ModuleList重複引用不增加容量；三組變換與12／4計數清楚。 |
| 15.3 | router線性分數經softmax成混合係數，所有expert皆算的soft routing尚未省計算；log2／exp2轉換與2.25倍輸出可以逐步理解。 |
| 15.4 | top-k逐token挑整數expert索引並保留連續比例，topk本身不跑expert也不重算比例；離散選擇和連續學習路徑有解釋。 |
| 15.5 | 多個expert沿相同特徵加權相加，top2重新正規化與top1保留比例約定明确；圖和例子沒有把expert相加與外層殘差混成同一件事。 |
| 15.6 | dispatch需保存來源token列号，combine按原列累加，覆蓋會漏掉同token第二份貢獻；図中列1零是刻意缺工作而非正常dropless遺失。 |
| 15.7 | top1把p除自身後gate恆1，任務梯度會斷；保留softmax比例時兩個scores都有相反梯度，原碼1.1852／-1.1852與0／0已實跑核對。 |
| 15.8 | 幾乎均勻softmax仍可全送同expert，實際被選次数和平均機率必須分開；熵從-f log f及零份額規則導入，明確只總括全驗證集。 |
| 15.9 | N∑f_i p_i將離散負載視固定、由連續平均機率教router，toy均衡1／集中2.4可核對；也清楚說明auxiliary不是完美不均量且可小於1。 |
| 15.10 | 總參數決定儲存、每token啟動只是容量代理，router隨expert數增加，完整字表列入proxy並非真正讀取格數或FLOPs；計數口徑充分交代。 |
| 15.11 | 分派、排名、小矩陣与合併開銷可讓MoE較慢，CPU只量前向、GPU量更新是不同範圍；峰值起點差異有交代，但长方法記錄適合移往實驗頁。 |
| 15.12 | 固定capacity按平均工作×factor向上取整，總容量足夠仍會單expert溢出；drop／fallback／reroute與殘差需另定，教材dropless並未實測drop收益。 |
| 15.13 | 同總容量和同使用代理會得到不同Dense尺寸，需要相同資料與曝光量並另量速度；正式表承認約3.15%近似差距、單seed与proxy不等FLOPs。 |
| 16.1 | profiler的self／total／calls与權重bytes不一樣，CPU例子可以定位瓶頸；但暖機之後接TF32、六支訓練、allocator／reserved／獨立子程序等多段方法細節，讀者尚未進入效率主線。 |
| 16.2 | prefill算完整提示、decode算新Query但讀過去KV，四軸形状可追蹤；TTFT与單call時間、含prefill總時間與純decode吞吐量分清楚。 |
| 16.3 | 因果前文各層不變可缓存K/V，新位置仍投影与读前文；完整／cache最後logit實跑差2.38e-7，位置offset与改前文要重建都說清楚。 |
| 16.4 | 多Query共用少KV能縮cache，4／1KV的384／96bytes與形状實跑一致；MQA是端點、重新初始化不等原論文平均轉換，品質和儲存收益分開。 |
| 16.5 | EOS不自動隔離文件，packing須同segment及因果交集、重置位置並避跨文件下一字目標；圖的四乘四矩陣正確，正式兩段等长例只檢語意未省PAD。 |
| 16.6 | microbatch的誤差和除共同有效目標總数，最後一次step与裁剪才等价同大batch；原碼整批／正确累積9.3333、错平均7.5，實跑核對一致。 |
| 16.7 | FP16精度細但範围窄，BF16範围大但捨入较粗，autocast不改FP32原參數儲存；scaler先放大後还原、遇非有限梯度可跳步，有限不等答對。 |
| 16.8 | SDPA自带1/√D縮放、布林True允許且dropout需傳0，實跑手寫差5.96e-8；API不等指定Flash，正式FP32memory-efficient與低精度Flash補驗各自分開。 |
| 16.9 | online softmax更新最大值時同縮舊分母与分子，两塊得到24.2857；原程式與图數值一致，Flash仍完整注意力並減HBM搬移，灰基線圖不把新增峰值當整GPU用量。 |
| 16.10 | activation checkpoint保存區域邊界再在反向重算中間值，以時間換峰值；例子只比獨立輸入梯度而不錯比已累加權重梯度，與存模型checkpoint區別清楚。 |
| 16.11 | 編譯首次準備與穩態每call要分量，500次回本是算式，eager backend只是接口示範；真Inductor例成功但未加速，null回本與未测整模型界線明確。 |

## 图、来源与实际核对

16张实际SVG全部用现有Inkscape 1.4渲染，实际查看四张contact sheet，并对照图内数字、矩阵、箭头与正文。DPO梯度、PPO两侧平台、旋轉分量、dispatch回归、GQA连线、packing／causal布尔矩阵、online softmax、Flash灰基线图均相符；normalization图的问题已列出。源SVG的完整SHA、关联节ID和解读在inspection.json。第一次ImageMagick依赖rsvg-convert的渲染失败，改用已有Inkscape后全部成功，沒有把失败当完成。

实际运行10段原文CPU Python fence：13.3、13.6、13.13、13.15、15.7、16.3、16.4、16.6、16.8、16.9。Python 3.13.5／PyTorch 2.14.1+cpu，退出0。PPO目标及导数为`[0.7,1,1.2,-0.8,-1,-1.3]`／`[-1,-1,0,0,1,1]`；top-1 gate梯度为约`[1.1852,-1.1852]`或重算后的零；KV储存384／96bytes；大批與正確累積9.3333、错平均7.5；online结果24.2857。实际stdout、stderr、执行原文fence的SHA分别保存，不能把这十段抽核说成全部notebookrun。

实际核对的權威来源（下载原件、抽取文本、URL与SHA已保存sources/receipts.json）：

- [dpo](https://arxiv.org/pdf/2305.18290v3)：原论文第3–4节，公式1–7（extracted text 75–255）。reference KL目标的beta、reward比较负log-sigmoid和DPO相对log概率公式一致；正文未把不同实现保证为同模型。
- [ppo](https://arxiv.org/pdf/1707.06347)：第3节公式6–7和Figure1说明，Algorithm1（101–146、238–249）。ratio与固定采样记录、多epoch复用和正负优势不对称裁切一致。
- [switch](https://arxiv.org/pdf/2101.03961v3)：2.1–2.2相关gate说明、容量式3、平衡项式4–6（243–330、347–436）。top1原softmax gate、capacity与N∑fP形式一致；本书top2改归一化與扩展分母已明示。
- [roformer](https://arxiv.org/pdf/2104.09864v5)：3.2.2公式14–16（257–293）。二维对的正交旋转与相对点積明确，书中没有把其不变性当成超长泛化保证。
- [gqa](https://arxiv.org/pdf/2305.13245v3)：2.1–2.2转換和端点（38–114）。GQA-1相当MQA，论文转换用KV平均；本书明确随机新KV不是复现同配方。
- [flash](https://arxiv.org/pdf/2205.14135v2)：3.1 tiling/recomputation、Algorithm1和Theorem1（236–322）。online统计共同重缩放、完整注意力及HBM/SRAM限制一致。
- [pytorch-sdpa](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)：2.14官方页面GQA段、布林mask与dropout警告，摘录存documentation-excerpts.json。目前2.14确列Flash、cuDNN、math与NVIDIA CUDA memory-efficient支持GQA；True允许、dropout不随eval自动清零正确。
- [pytorch-amp](https://docs.pytorch.org/docs/2.14/amp.html)：Gradient Scaling及FP16 underflow/max65504段落。损失扩大带同比例梯度、更新前缩回与学习率区别正确。
- [pytorch-grad-scaler-official](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/amp/grad_scaler.py)：官方2.14.1 commit的GradScaler docstring；infs/NaNs skip段。非有限梯度时跳过optimizer.step、成功尝试与真正更新次数需分计的说法正确；生成API页尝试404后改查真实源文件。


本地源码核对了DPO、RM／PPO、MoEFFN gate与dispatch、attention cache／mask和模型默认形状；architecture.py的有效输入forward hook确实在正式实验中重算不含PAD的辅助项，因此没有把stock MoEFFN内部未收PAD mask误报为实报错误。相关文件SHA和读取范围在inspection.json。

## 限制与不变证据

未重新执行L4训练、CUDA后端、AMP／compile性能，也未完整重审每个实验JSON；正文旧实测数字没有冒称是本次实验。完整notebook由root统一执行。图是静态render，尚未验证动画、手机或整站渲染。没有重复全文阅读前置章节，也没有读取第20章。

初读的四章与16图原始完整字节已保存在original-sources，全部核对初始SHA吻合。root收到发现后授权修改13章导读／13.15／13.17、14.2与normalization图；因此最终canonical与初读SHA并非全部相同，详见下面closure。此review agent只写指定outputs目录，没有修改教材、基线检查器、既有review或阅读时间元数据。fullfile／section／figure／code／authority原始SHA均在inspection.json及initial-manifest.json中，hash仅用于版本锚定，不代替上述完整阅读。


## root修稿后的完整重读closure

- `R13-route`：closed_in_rechecked_revision。导读把13.1–13.6列为共同基础，再分延伸13.7–9/PPO13.10–17；13.16/17要求的13.5已在路线内，13.17可回看也不再暗示尚未首次读过。
- `R13-revision-language`：closed_in_rechecked_revision。13intro不再称新增PPO，13.15用另外两行，13.17用本章有限选卡实验／可回看／偏好資料，均描述当前内容；新增求補資訊指實際能力变化，保留合理。
- `R14-normalization-label`：closed_in_rechecked_revision。14.2说明及真实新图明确缩放／偏移，gamma／LN有beta分开，内容与公式一致。
- `R1316-experiment-density`：optional_editorial_improvement_open。P2建议，不是公式错误；最值得移动的是原16.1第36行实验设置和40–44行allocator/reserved/子程序scope，保留38行中位数定义、34行权重不等峰值与结果必要分母／限制。其他节指标对照不要求机械删除。

完整重读顺序：13intro → 13.15 → 13.17 → 14.2 → 新normalization.svg源 → 实际新PNG。新图已Inkscape渲染并真实查看，文字无溢出。修正前／后的全文件与节／图SHA、精确source快照及判断都存在closure.json与inspection.json；没有拿hash更新替代阅读。Python fence字节相同，未不必要地重跑原十段CPU日志。

初读47节的原文字节已精确保存；root授权更动的13／14章及normalization图有双版本记录，其余章／图按SHA核对保持初读版本。此review agent没改canonical。closure是同一agent复读，不是另一个fresh reader，也不替代后续正式逐节reader／factual／time流程。
