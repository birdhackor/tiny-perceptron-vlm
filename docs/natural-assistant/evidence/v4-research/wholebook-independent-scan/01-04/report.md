# 第01–04章獨立完整閱讀預檢

任務身分：`/root/v4_wholebook_scan_01_04`；parent：`/root`。這是 supplemental 預檢，不替代既有 formal reader／technical review，也不宣稱全書 final pass。沒有改教材、基線 review、checker 或時間 metadata。

實讀順序為01→02→03→04，35節全部完整閱讀；01與04依行段分兩次讀完，不靠搜尋或雜湊代替閱讀。之後直接讀必要實作、12張SVG來源、W.1–W.6明確前置文字，再查看全部12張浏览器渲染圖。沒有讀舊review verdict或作者歷史來補背景。逐次實讀事件、每節範圍和SHA、原始來源快照及每張圖的SHA都在[inspection.json](inspection.json)与[source-manifest.json](source-manifest.json)。

本範圍目前沒有未解決的重大技術或閱讀問題。發現的2項P2問題都已向root回報，由root修文後實讀新版2.5確認解除。教材中的失敗例是概念反例，沒有找到應刪的無學習價值事故歷史。

## 發現與實際修正後重讀

1. **P2：2.5首次用MLP但未介紹。** 初稿說「讓三個MLP分別看最近1、3、5字」，2.1–2.4此前沒有為網路命名。建議在表格前加「查表→拼接→線性與非線性小網路」的淺白定義。現稿實際讀到「這種由線性層與非線性轉換串成的網路，叫多層感知器（multilayer perceptron，MLP）」並連回2.4的彎曲規則，已能理解實驗名稱。
2. **P2：中途補述把實驗激活說成ReLU。** 實讀中途修文的「線性層與2.4的ReLU串成」後，直接讀`tiny_perceptron/simple.py`，發現`ContextMLP.forward`實際使用`torch.tanh`。建議泛稱非線性轉換並連原理。現稿已照此處理，完整重讀2.5到末尾，與實作相容。中途版在又一次改文前沒有保存SHA，因此沒有偽造该版source身份；只留真實讀到的引句。

初讀與最後完整重讀的2.5 section SHA：`53818c6179720b62b0de43d41eeca967bed091e77a575e7485351a35e9bcdbc6` → `a0b287525b957a468d435301866ea476208d6611a9be3e56c5dd0f5043eff0ec`。初稿快照保留於`sources.initial/course/chapters/02.md`；現稿快照在`sources/course/chapters/02.md`。其他34節與12張圖内容仍與初讀相同。

## 每節自己的理解／問題

| sectionID | 實際理解 | 問題 |
| --- | --- | --- |
| 1.1 | 字與ID是可逆查找約定；Unicode排序固定編號，不產生字義；反序字表會變ID但不變還原文字。 | 無未解決問題 |
| 1.2 | 先去重並分完整來源，再切短片段；train調參、validation選設定、test作最後檢查，交集檢查只排除完全重複。 | 無未解決問題 |
| 1.3 | s[:-1]和s[1:]建立每題下一字答案；長度相等仍可能學錯任務，shift與限制可見前文分工不同。 | 無未解決問題 |
| 1.4 | unigram將次數除總數，所有前文得到同一分配；最大值選取與依機率抽樣是另一層規則。 | 無未解決問題 |
| 1.5 | bigram每列是目前字、每欄是下一字；貓列有兩次看，加一後看為3/9、其他各1/9，每列用自己的分母。 | 無未解決問題 |
| 1.6 | Embedding在此是V×V可調候選分數表；輸入選列，no_grad只管手動填表，查表本身未更新參數。 | 無未解決問題 |
| 1.7 | exp後除總和得到softmax；先减最大值保留比例而避免溢位，候選軸各自正規化，高機率不等於答案正確。 | 無未解決問題 |
| 1.8 | 答案ID指定正確機率那格，-log(p)衡量代價；F.cross_entropy收原始logits，三候選示例loss約0.2395。 | 無未解決問題 |
| 1.9 | 有限差分估附近敏感度；w=1的平方代價導數為-4，符號說微調方向，h過小會放大捨入影響。 | 無未解決問題 |
| 1.10 | 前向w=3→u=6→L=36，反向局部敏感度12乘2得24；計算圖記運算關係，多路影響要相加。 | 無未解決問題 |
| 1.11 | 每參數偏導排成梯度[-4,2]；總代價6不是更新量，參數與梯度形狀相同且數值用途不同。 | 無未解決問題 |
| 1.12 | backward只求導；no_grad內按w−ηgrad更新，再用新w重算loss，η=0.1得1.4和2.56，大步可能惡化。 | 無未解決問題 |
| 1.13 | 每次重算平方的新圖都把梯度加進同一.grad；清除桶得4/8/4，這與對舊圖第二次backward的釋放問題不同。 | 無未解決問題 |
| 1.14 | 逐步取當前字那列、抽一字、接回輸入；固定十二次決定長度，實訓錯填形狀由等號前文相同解釋。 | 無未解決問題 |
| 1.15 | 正溫度縮放logits後再softmax，改抽樣分配不改參數；greedy排序仍相同，T=0不代入除法公式。 | 無未解決問題 |
| 2.1 | 同一ID在[V,D]表取同一特徵向量，梯度可回到用過的列；它與經上下文混合後的表示不同。 | 無未解決問題 |
| 2.2 | 按位置拼接[B,C,D]→[B,C×D]保留順序，reshape的-1推軸長；固定窗口使下一層輸入格數固定。 | 無未解決問題 |
| 2.3 | Linear每個輸出一套加權配方和bias；權重[out,in]需轉置，最後特徵軸作用而前置筆/位置軸保留。 | 無未解決問題 |
| 2.4 | 兩層純線性仍可合成一層；手設x和−x再ReLU後相加得到x的絕對值，三點形成單一直線無法表示的V形。 | 無未解決問題 |
| 2.5 | 窗口1/3/4缺顏色線索、5字才可分兩題；新版先命名查表拼接後的MLP及非線性作用，表格顯示更長窗口不保證驗證較好。 | 無未解決問題 |
| 3.1 | 非負且和為1的讀取比例加權混合values，位置軸被消去而特徵保留；比例大小不等於實際數值貢獻。 | 無未解決問題 |
| 3.2 | query與各key點積得到匹配分數，再softmax；改query改偏好，點積還受向量長度影響。 | 無未解決問題 |
| 3.3 | Q的列是查詢、K的列是候選，QKᵀ輸出[Tq,Tk]；同源自注意力仍需分開需求與匹配角色。 | 無未解決問題 |
| 3.4 | Q/K決定候選比例，再用相同比例混合V；輸出特徵數由V決定，候選與V列需對齊，例值約7.042/14.083/21.125。 | 無未解決問題 |
| 3.5 | 在獨立零均值單位變異數假設下，D項加總變異數D、標準差√D；除√D控制特徵數引起的尺度，並非訓練後一律保證std=1。 | 無未解決問題 |
| 3.6 | j≤i許可目前和過去、禁止未來；-inf在softmax後成0，最後value加100不影響前3位置，shift與mask缺一不可。 | 無未解決問題 |
| 3.7 | 先投影Q/K/V再拆head，每頭仍讀全部位置；[1,4,8]拆為[1,2,4,4]再拼回，K快取不是4×4權重表。 | 無未解決問題 |
| 4.1 | 字表特徵加同寬絕對位置查表向量，使同ID不同位置起始表示不同；位置表上限與學會語序是兩種限制。 | 無未解決問題 |
| 4.2 | y=x+f(x)的原路與修正路相加；f=2x時數值[3,6]、對每格梯度1+2=3，殘差不保證無損保留一切。 | 無未解決問題 |
| 4.3 | LN每位置最後特徵軸單獨算均值/biased variance，epsilon避免除零，再作每特徵gamma/beta；整排平移與改一格不同。 | 無未解決問題 |
| 4.4 | 逐位置FFN先4→16、GELU、16→4，共用配方但不跨位置讀取；Dense讓每位置使用完整配方參數。 | 無未解決問題 |
| 4.5 | pre-norm先整理分支再Attention/FFN，加回未被替換的主路；shape保持，cache供後續生成，auxiliary=0不是語言loss=0。 | 無未解決問題 |
| 4.6 | TinyLM每位置D特徵轉V候選logits；三輸入位置各有不同可見前文，續寫用最後列，loss先展成三題二軸，約3.336。 | 無未解決問題 |
| 4.7 | 原序列[1,2,3,4]只shift一次成x=[1,2,3],y=[2,3,4]；因果許可使三題可平行，masked_loss不再shift，backward仍未更新。 | 無未解決問題 |
| 4.8 | 深度只複製block而非固定入口出口；width8每block840、固定1360，總2200/3040/3880，成本與實際準確率分開驗證。 | 無未解決問題 |

## 來源與可核對證據

- 原論文[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)實際下載HTTP200，讀§3.2.1、footnote4、§3.2.3與§3.3的原文。`primary-sources/transformer.txt`第150–205、227–264行支持attention的縮放、独立假设、因果遮罩和逐位置FFN。沒有把原論文的ReLU或權重共享當成此專案必然配置。
- PyTorch官方`LayerNorm`網頁stable入口只給HTML redirect，2.14目的頁403；沒有把redirect當正文。改讀已安裝官方PyTorch2.14.1+cpu的`LayerNorm`完整docstring，確認最後特徵軸、correction=0變異數、epsilon及γ/β初始化。原始class來源與SHA保存在`primary-sources/installed-pytorch-layernorm.py.txt`及receipt。
- 直接讀本專案`attention.py`、`model.py`、`modern.py`、`simple.py`、`data.py`核對手寫attention、K/V介面、DenseFFN、ContextMLP(tanh)、shift及split。來源快照與SHA見inspection的supporting_sources；未參考任何既有review結論。

## 真實执行與圖檢查

用既有`.venv/bin/python`執行1.8、2.4、3.4、3.6、4.6、4.8的原樣Python fences，共7段、6程序，全部exit0。环境實錄是Python3.13.5、Torch2.14.1+cpu、CUDA不可用。4.6平均loss=`3.335638999938965`；4.8參數格數=`2200,3040,3880`。數值、ReLU三點、attention混合及因果遮罩检查皆符合正文；原码、argv、exit、stdout/stderr与SHA在[numeric-execution.json](numeric-execution.json)，不是事后虚构日志。

12張图已從真实SVG渲染成PNG並实际查看：字ID、shift、bigram、chain rule、embedding、ReLU、窗口曲線、Q/K/V、causal mask、多頭、residual、各位置下一字预测。标签、箭头、数据与正文一致，沒有发現裁切或把shape／模型能力画錯的問題。每圖的獨立理解、來源SHA、render SHA及檔案都在inspection.figures。

## 範圍、未跑與限制

没有跑全部notebook或所有练习，也没有重训T.3/T.4；训练报告中的loss数字不在本次独立重跑的保证范围。root会另外跑完整kernel。沒有讀第20章，未检验自然語言能力、公共網站／Colab的端到端可用性。W.1–W.6仅作为明确前置文字读完，未对前置图进行视觉复审。没有对全书宣布最终通过。

## 最後正文檔案身份

| 完整來源檔案 | SHA-256 |
| --- | --- |
| `course/chapters/01.md` | `3292bcd7a9719f2ae84e86f8cf80e4f80704eddb5d9c1d75f0370b0efeb8af17` |
| `course/chapters/02.md` | `edb5102476271b0b39e0018ff9f41bc261b672773d28262d8cc791ce593938a1` |
| `course/chapters/03.md` | `9caca4915a56a6cfa7f42900bd45668303750cb695469809a9c237919022e95b` |
| `course/chapters/04.md` | `dc522bcf051fd3c2a3f9914fe98a22832fde88336fcab1b3d783b4ec72631c1c` |

每个section和figure的完整SHA另见inspection.json；这些身份记录只是绑定真正已读文字与图，未用来替代阅读。
