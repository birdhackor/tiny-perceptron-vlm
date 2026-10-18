# 遇到陌生名詞時，回到一個例子

這頁是查找入口，不是開課前的背誦清單。找到陌生詞時，先看它在一個小例子裡做什麼，再點進對應小節看完整過程。同一個中文譬喻有時涵蓋好幾個步驟，所以也要核對它究竟對應哪個英文名詞。

四節依主題排列：G.1 文字怎麼變成數字，G.2 模型怎麼猜、怎麼學、資料怎麼分，G.3 模型怎麼混合各位置的線索、圖片與聲音怎麼接進來，G.4 後面章節的主題。中文或英文都可以用瀏覽器的頁內搜尋（Ctrl+F，Mac 是 ⌘F）找；英文名詞與縮寫另有一份字母索引，在 G.4 最後。

## G.1 文字與數字怎麼對上

拿「貓看狗」作例子。先把可用字排成一份清單，這份清單叫字表；約定「狗」是 0、「看」是 1、「貓」是 2，原句便寫成 `[2, 1, 0]`。這些編號叫 ID，只負責指出是字表裡的哪一項，不表示貓比狗大。如果再拿 ID 2 去查一張可調的數字表，取出 `[0.2, -0.1]`：整張表叫 embedding 表，查出來的這一排叫 embedding 向量，中文也說嵌入向量。編號與向量是兩個步驟，不能混為一談。

| 名詞 | 在例子裡做什麼 | 完整講解 |
| --- | --- | --- |
| 字表（vocabulary，也叫詞表） | 所有可用字的清單；換成 token 之後仍叫字表，每一項是一種 token | [1.1](chapters/01.md#1.1)、[6.1](chapters/06.md#6.1) |
| ID、編碼（encode）、解碼（decode） | 替字表裡每種字安排的整數叫 ID；文字變 ID 叫編碼，ID 變回文字叫解碼。ID 只在同一張字表裡有效。第 16 章的逐步生成階段英文也叫 decode，和這裡的解碼是兩件事 | [1.1](chapters/01.md#1.1)、[16.2](chapters/16.md#16.2) |
| token、tokenizer、切詞（tokenization） | token 是模型一次處理的一個單位；tokenizer 把原文切成 token，再把每個 token 換成 ID，這個動作叫切詞。一個 token 可以是一個字、好幾個字，或一個字的幾個 byte | [6.1](chapters/06.md#6.1)、[6.7](chapters/06.md#6.7) |
| 字元、碼點（code point）、UTF-8 | 一個字、一個字母或一個圖案叫一個字元；Unicode 替每個字元訂的編號叫碼點；UTF-8 把每個碼點存成 1 到 4 個 byte，常見中文字用 3 個 | [6.1](chapters/06.md#6.1) |
| byte（位元組）、bit（位元） | 一個 byte 有 8 個 bit，能存 0 到 255 之間的一個整數；儲存空間以 byte 計 | [5.9](chapters/05.md#5.9)、[6.1](chapters/06.md#6.1)、[17.1](chapters/17.md#17.1) |
| 特殊 token（special token）、角色標記 | 不代表任何原文、專門標示結構的 token。本課 ByteTokenizer 的 ID 0 到 7 是 8 個特殊 token，包括 PAD、BOS、EOS，以及標示接下來由誰說話的角色標記 | [6.6](chapters/06.md#6.6) |
| BOS（開始標記）、EOS（結束標記） | BOS 放在序列開頭，只當前文、不計分；EOS 表示這段到此結束。回答尾端的 EOS 也是有效目標，生成時模型選到它就停 | [6.5](chapters/06.md#6.5)、[7.9](chapters/07.md#7.9) |
| embedding 表、embedding 向量 | 每個 ID 對應一列可調數字的表叫 embedding 表；用 ID 選出的那一列叫 embedding 向量（也叫嵌入向量）。ID 只選列，數字本身不代表字義 | [2.1](chapters/02.md#2.1) |
| tensor、shape、軸（axis） | tensor 是 PyTorch 保存有規則分組數字的容器，中文常譯為張量；每個可編索引的方向叫一個軸，shape 記每個軸有幾格，兩列三欄就是 `[2, 3]`。文字模型常見的 `[B, T, D]` 依序是幾筆句子、每筆幾個文字位置、每個位置幾個特徵數字 | [W.3](first-steps.md#W.3) |
| 特徵寬度（D，width）、中間寬度（hidden） | 每個位置存幾個特徵數字叫特徵寬度；FFN 內部先擴大到的特徵數叫中間寬度，例如 4.4 的特徵寬度 4、中間寬度 16 | [2.1](chapters/02.md#2.1)、[4.4](chapters/04.md#4.4) |
| 參數（parameter） | 模型裡可調、訓練時要改的數字。B 單元說的「工具的參數」是交給工具的輸入值，和這裡的參數是兩回事 | [W.6](first-steps.md#W.6)、[1.2](chapters/01.md#1.2)、[B.1](chapters/0B.md#B.1) |
| 權重（weight）、偏移（bias，也叫偏置） | Linear 裡各輸入要乘的數叫權重，加總後再加的數叫偏移，兩者都是參數；習慣上也把全部參數統稱權重，例如保存參數的權重檔 | [2.3](chapters/02.md#2.3) |
| Linear（線性層） | 每套配方把各輸入乘自己的權重、加總，再加偏移，產生一個輸出；只沿最後一個特徵軸計算 | [W.5](first-steps.md#W.5)、[2.3](chapters/02.md#2.3) |
| 啟用函式（activation function） | 插在線性配方之間、會轉折的非線性規則，例如 2.4 的 ReLU 保留正數、把負數改為 0；4.4 的 FFN 用 GELU | [2.4](chapters/02.md#2.4)、[4.4](chapters/04.md#4.4) |
| 中間特徵（activation） | 送進一層的輸入，以及計算途中產生、再交給下一層的數字。英文 activation 也用來稱呼上一列的啟用函式；本課把數值叫中間特徵 | [17.11](chapters/17.md#17.11) |
| JSON | 用文字寫資料的通用格式：大括號包住一組組「欄位名: 值」，寫法和 Python 字典很像，但它是一段字串 | [7.13](chapters/07.md#7.13)、[B.1](chapters/0B.md#B.1) |

token 不一定是一個中文字。逐 byte 切詞時，一個常見中文字是 3 個 byte，也就是 3 個 token；合併常見片段的切法，一個 token 也可能包含好幾個字。閱讀時要問「這裡採哪種切法」，不能看到 token 就一律算中文字數。

試著回答：`2` 與 `[0.2, -0.1]` 哪個是 ID、哪個是向量？前者指出要查表的哪一列，後者才是查回的數字。若仍分不清，建議先讀 1.1，看看文字如何換成編號；再讀 2.1，看看如何用編號查出一排向量。

## G.2 猜法、學習與保存

假設模型替三個候選字打分 `[2, 1, 0]`。這些分數叫 logits，還不是機率。softmax 把分數換成加總為 1 的比例，這個例子約為 `[0.665, 0.245, 0.090]`。若正確答案是第三個字，模型只分給它 0.090，猜錯代價就偏高。訓練用這個代價告訴模型：內部哪些數字應往哪個方向改。

| 名詞 | 可以怎麼理解 | 完整講解 |
| --- | --- | --- |
| 接字表（bigram 表） | 列是目前字、欄是下一字，記錄每對相鄰字出現幾次；每列除以自己的總數，得到下一字的條件機率 | [1.5](chapters/01.md#1.5) |
| logits（候選分數）、softmax | logits 是還沒轉成機率的原始候選分數，能正能負；softmax 把每個分數取指數、再除以總和，換成非負、總和為 1 的機率 | [1.7](chapters/01.md#1.7)、[1.8](chapters/01.md#1.8) |
| 代價（loss，也叫損失）、交叉熵（cross entropy） | 取正確答案那一格的機率 p，算 `-log(p)`；正確答案得到的機率越少，代價越大，代價不是猜錯題數。多題的代價取平均叫交叉熵 | [W.4](first-steps.md#W.4)、[1.8](chapters/01.md#1.8) |
| 誤差、成本 | 數值之間的差叫誤差，例如原值與還原值的差叫捨入誤差；9.16 做決策時人工訂的得失叫成本。兩者和代價（loss）是不同的量 | [17.2](chapters/17.md#17.2)、[9.16](chapters/09.md#9.16) |
| NLL、平均 NLL | 把全部有效目標的代價加總叫 NLL（負對數似然），本課的 NLL 指這個加總；加總後再除以有效目標數叫平均 NLL，訓練時印出的代價就是平均 | [6.5](chapters/06.md#6.5) |
| 正確答案（ground truth，也叫真值） | 拿來和預測比對的答案；代價取的就是它那一格的機率 | [1.8](chapters/01.md#1.8) |
| 目標、labels、有效目標 | 每個位置要對上的答案 ID 叫目標，整串叫 labels；寫成 −100 的目標不計代價，其餘真的要計分的位置叫有效目標，個數叫有效目標數。決定哪些位置計代價的設定也叫代價遮罩（loss mask） | [5.2](chapters/05.md#5.2)、[7.1](chapters/07.md#7.1)、[7.24](chapters/07.md#7.24) |
| 下一位置對齊（shift） | 讓每個位置配上它的下一個位置當答案；做法是建立兩份錯開一格的切片，不修改原文 | [1.3](chapters/01.md#1.3) |
| batch | 同一次計算裡一起處理的幾筆例子 | [5.2](chapters/05.md#5.2) |
| 梯度（gradient）、導數（derivative） | 參數微動時，「代價改變÷參數改變」在步子趨近零時的極限比例叫導數；對每個參數各算一個導數、排回原位置，就是梯度 | [W.6](first-steps.md#W.6)、[1.9](chapters/01.md#1.9) |
| 前向計算（forward）、反向傳播（backpropagation） | 從參數依序算到代價叫前向計算；`backward()` 從代價往回走、把每段的敏感度相乘求出梯度，叫反向傳播 | [1.10](chapters/01.md#1.10) |
| 梯度下降（gradient descent）、學習率（learning rate，lr） | 更新規則是 `新參數=舊參數−學習率×梯度`；學習率是我們設定的每次步子的比例 | [W.6](first-steps.md#W.6)、[1.12](chapters/01.md#1.12)、[5.1](chapters/05.md#5.1) |
| 步幅 | 參數在一次更新中實際移動的量，由學習率和更新方向一起算出。用 Adam 時各格的步幅可以不同，共同的學習率不等於每格的步幅 | [5.4](chapters/05.md#5.4) |
| 最佳化器（optimizer，也叫優化器）、Adam | 管理參數更新的工具：`zero_grad()` 清梯度、`step()` 更新參數。Adam 記下每格過去的梯度通常多大，用這個大小決定這一格這一步走多遠 | [1.13](chapters/01.md#1.13)、[5.4](chapters/05.md#5.4) |
| 最佳化器狀態（optimizer state）、m、v | Adam 替每格參數保存的兩份歷史：m 是過去梯度的加權平均，對應 5.3 的動量（momentum）；v 是梯度平方的加權平均。要接續訓練，得連同已更新的次數一起保存 | [5.3](chapters/05.md#5.3)、[5.4](chapters/05.md#5.4)、[5.7](chapters/05.md#5.7)、[15.10](chapters/15.md#15.10) |
| 檢查點（checkpoint）、父檢查點 | 訓練途中存下的檔案；只恢復參數能照樣預測，要讓下一步也相同，還要恢復最佳化器狀態。上一階段保存、這一階段載入當起點的檢查點叫父檢查點。16.10 的中間特徵重算（activation checkpointing）是另一件事 | [5.7](chapters/05.md#5.7)、[T.4](training.md#T.4)、[16.10](chapters/16.md#16.10) |
| 輔助代價（auxiliary loss） | 主要代價之外另加進訓練的代價，例如 MTP 距離 2 分支的代價，或 MoE 用來鼓勵各組 FFN 分到相近工作量的那一項 | [4.5](chapters/04.md#4.5)、[7.21](chapters/07.md#7.21)、[15.9](chapters/15.md#15.9) |
| NaN | not a number，不是有效數字；例如一整列分數都是負無限時，softmax 會算出 NaN | [3.6](chapters/03.md#3.6)、[16.7](chapters/16.md#16.7) |

比例高與回答可靠也是兩回事。模型可能很肯定地接出一個錯字；要知道它的信心是否與正確率一致，需要另外檢查，[9.9](chapters/09.md#9.9)會用具體例子說明。

看到某次代價下降，先找出那個代價算在哪些題目、哪些位置，再判斷能說什麼。可以用[5.8](chapters/05.md#5.8)的例子練習：寫一句「這個結果支持什麼」，以及還需要哪種新題檢查。這比把代價當成所有能力的總成績更準確。

### 資料怎麼分、實驗怎麼比

| 名詞 | 可以怎麼理解 | 完整講解 |
| --- | --- | --- |
| 訓練資料、驗證資料、測試資料 | 訓練資料用來調整參數，驗證資料用來選設定，測試資料留到設定選完後作最後檢查；各份裡的題目叫訓練題、驗證題、測試題 | [1.2](chapters/01.md#1.2) |
| 留出（held-out） | 沒有拿來更新參數的資料，例如驗證資料與測試資料 | [1.12](chapters/01.md#1.12) |
| 家族（family） | 從同一個來源或同一組屬性衍生出來的題目；切分資料時整個家族放在同一邊，全部是訓練資料、驗證資料或測試資料 | [5.12](chapters/05.md#5.12)、[7.12](chapters/07.md#7.12)、[19.3](chapters/19.md#19.3) |
| 支線 | 為了比較而分開訓練的版本；比較時要寫明它們差在哪些條件，例如起點、資料或更新次數 | [7.11](chapters/07.md#7.11) |
| 對照（control） | 只改一個條件、其他保持不變的比較 | [12.4](chapters/12.md#12.4) |
| 實驗紀錄（JSON） | 實驗的完整結果，存成機器可讀的 JSON 檔 | [1.12](chapters/01.md#1.12) |

### 生成與後訓練

| 名詞 | 可以怎麼理解 | 完整講解 |
| --- | --- | --- |
| 生成、逐 token 生成（也叫自回歸，autoregressive） | 算下一字分配、選一字、接回句尾，再把新字當下一次前文；每次只生成一個 token，接回輸入再算下一個 | [1.14](chapters/01.md#1.14)、[7.20](chapters/07.md#7.20) |
| 選字規則、greedy、抽樣（sampling，也叫採樣）、溫度（temperature，τ） | 挑下一個字的方法叫選字規則：greedy 總選最高分；抽樣按機率隨機抽，比例越大的字越常被選中。溫度 τ 是先拿來除原分數、再做 softmax 的正數，只控制分配的尖銳程度 | [1.14](chapters/01.md#1.14)、[1.15](chapters/01.md#1.15) |
| 推論（inference）、推理（reasoning） | 用現在的參數算候選分數；生成就是反覆推論，再選字接回。改溫度改的是推論時用的機率，更新參數才是訓練。推理（reasoning）是另一個詞，例如 7.23 的推理 token | [1.15](chapters/01.md#1.15)、[7.23](chapters/07.md#7.23) |
| 預訓練（pretraining）、底座（base model） | 先在大量文章上練續寫的階段叫預訓練；已經訓練好、拿來當下一階段起點的模型叫底座，例如預訓練後保存、再拿去做 SFT 的模型 | [7.11](chapters/07.md#7.11) |
| 後訓練（post-training） | 接在底座之後的學習。教法有幾種：用理想回答當示範（SFT），用同題兩份回答的偏好比較，或用作答後得到的分數（獎勵） | [7.17](chapters/07.md#7.17)、[7.18](chapters/07.md#7.18) |
| 監督式微調（SFT） | 拿正確的助手回答當示範，繼續訓練已學會續寫的語言模型；問題留在輸入裡，只用回答位置的目標 | [7.1](chapters/07.md#7.1)、[7.11](chapters/07.md#7.11) |
| 直接偏好最佳化（DPO） | 有同題兩份回答的偏好比較時，直接調整模型對兩份回答的傾向；它需要可靠的比較，以及一個固定參考 | [7.18](chapters/07.md#7.18)、[13.5](chapters/13.md#13.5) |
| LoRA、rank、adapter | 原權重 W 凍結不更新，另學兩個小矩陣 A、B 組成的修正路，輸出與原路相加；修正路中間剩下的數字個數叫 rank（秩）。可另存的一套 A、B 與設定叫 adapter（適配器），8.9 的兩套叫短答套、比喻套 | [8.8](chapters/08.md#8.8)、[8.9](chapters/08.md#8.9) |
| MTP（Multi-Token Prediction，多 token 預測） | 在同一個前文位置，要求模型預測不止一個未來的 token；分支依距離命名，例如距離 1、距離 2 | [7.20](chapters/07.md#7.20)、[7.21](chapters/07.md#7.21) |
| teacher forcing | 訓練時用資料中的標準序列當前文；只預測下一個 token 的一般訓練也這樣做，所以它不等於 MTP | [7.22](chapters/07.md#7.22) |

## G.3 模型如何混合線索

看到「那隻狗追著球，它跑得很快」，先暫把每個字當成一個位置，「目前位置」就是正在處理的「它」。實際模型的每個位置放一個 token，也就是切詞後的文字小單位；一個字可能拆成好幾個 token，佔好幾個位置。每個 token 先換成 ID，再查 embedding 表取得一排數字，這排數字就是文字特徵，讓模型能用數字處理文字。

模型把每個位置的特徵交給三套可學習的加權配方，分別產生 Q、K、V。「它」的 Q 與可讀的各個位置的 K 比較，例如前面「狗」與「球」的 K，得到每個位置的讀取比例，再按比例混合各位置的 V。注意力（attention）就是這套比較、計算比例、再混合數字的方法。「查筆記」可以幫你記住流程；哪些位置真能提供正確線索，仍要用模型的實際結果檢驗。

| 名詞 | 在流程裡的角色 | 完整講解 |
| --- | --- | --- |
| 上下文（context）、上下文視窗（context window，簡稱視窗，也寫作窗口） | 預測時能用的前文叫上下文；實際取出最近幾個位置送進模型的範圍叫上下文視窗。7.19 把前文加上新生成 token 的總上限叫上下文視窗的長度 T；位置算得出，不等於長文讀得好 | [2.5](chapters/02.md#2.5)、[7.19](chapters/07.md#7.19)、[14.7](chapters/14.md#14.7) |
| 訓練視窗 | 從長序列裡取出、當作一筆訓練樣本的一段；長度 T 的訓練視窗取 T＋1 個 ID，用 shift 錯開成 T 個輸入與 T 個答案。切詞在這之前就做完了 | [6.8](chapters/06.md#6.8) |
| MLP（Multilayer Perceptron，多層感知器） | 2.5 的固定視窗模型：查表、把最近幾個位置拼接，再經線性與非線性層猜下一字；T.3 訓練的就是它。block 裡在每個位置各套用一次的那一層叫 FFN | [2.5](chapters/02.md#2.5)、[T.3](training.md#T.3) |
| 注意力（attention）、讀取比例 | 按比例混合幾個位置的數字；這組比例叫讀取比例，程式與文獻常叫注意力權重（attention weights） | [3.1](chapters/03.md#3.1) |
| Q（query）、K（key）、V（value） | query（Q）提出需求，key（K）提供匹配線索，value（V）是按比例帶回的數字；三者由同一組位置產生，所以叫自注意力 | [3.2](chapters/03.md#3.2)、[3.3](chapters/03.md#3.3)、[3.4](chapters/03.md#3.4) |
| 注意力頭（attention head）、多頭注意力 | 一組各自的 Q/K/V 配方叫一個注意力頭；準備幾組、各算匹配與混合，再把結果拼回來，叫多頭注意力 | [3.7](chapters/03.md#3.7) |
| 因果遮罩（causal mask） | 讀取許可表：位置 i 只准讀自己與更早的位置，整段一起算時也讀不到後面的答案 | [3.6](chapters/03.md#3.6) |
| PAD、padding（補齊） | 把長短不同的序列補成一樣長時補上的位置，PAD 的 ID 是 0；PAD 位置的目標設為 −100、不計代價，`valid` 標出哪些位置是真實 token（非填充位置）。7.10 程式說明裡的「有效位置」有兩種：attention 的指非填充位置，loss 的指有效目標 | [6.6](chapters/06.md#6.6)、[7.6](chapters/07.md#7.6)、[7.7](chapters/07.md#7.7)、[7.10](chapters/07.md#7.10) |
| 殘差連線（residual connection）、主路、修正路 | 寫成 `y=x+f(x)`：x 沿主路原樣帶到後面，修正路算出的修正 f(x) 逐格加回；兩邊必須同形狀 | [4.2](chapters/04.md#4.2) |
| 層正規化（LayerNorm，LN）、RMSNorm | 每個位置先減自己的平均、再除自己的標準差來整理尺度，避免數值任意放大；14.2 的均方根正規化（RMSNorm）改除以均方根，不先減平均 | [4.3](chapters/04.md#4.3)、[14.2](chapters/14.md#14.2) |
| L2 正規化、文字正規化 | 名字裡也有正規化，卻不是上一列的層：L2 正規化把一個向量除以自己的長度，讓長度變成 1；文字正規化整理文字本身，例如把連續的空白合成一個空格 | [10.9](chapters/10.md#10.9)、[14.3](chapters/14.md#14.3)、[5.11](chapters/05.md#5.11) |
| 前饋網路（Feed-Forward Network，FFN） | 每個位置各自把手上的特徵組合成新特徵。結構就是 2.5 的 MLP（Linear、非線性、Linear），只是在每個位置各套用一次：先擴張到中間寬度，再縮回特徵寬度 | [4.4](chapters/04.md#4.4) |
| 輸出層（output layer，也叫 LM head）、分類頭（classification head） | 輸出層把最後的特徵變成每個候選字的分數；分類頭是為每一類各算一個分數的輸出層。這個 head 和注意力頭不是同一個零件；7.20 引用的論文把每條預測分支自己的那一小段網路叫 head，它也不是共用的輸出層 | [4.6](chapters/04.md#4.6)、[10.5](chapters/10.md#10.5)、[7.20](chapters/07.md#7.20) |
| 核心、kernel（計算程式） | 7.20 說的核心，是替每個位置算出特徵的主體，也就是 embedding、注意力與 FFN 這些層；kernel 則是實際執行底層乘加與資料讀取的程式，兩者不是同一件事 | [7.20](chapters/07.md#7.20)、[17.10](chapters/17.md#17.10) |

幾種「遮罩」也要分開：不讓模型看未來，與不讓某些位置計代價，是不同工作。提示文字可能不計代價，卻仍提供回答線索。試著用一句話說出「看得到」和「需要作答」的不同，再到[7.5](chapters/07.md#7.5)核對。

### 圖片與聲音

| 名詞 | 在流程裡的角色 | 完整講解 |
| --- | --- | --- |
| 文字核心（language backbone）、接頭（projector） | 多模態模型裡負責讀寫文字的語言模型部分叫文字核心；把每個位置的圖片特徵重新組合成文字核心所需寬度的線性層叫接頭。底座說的是訓練的起點，文字核心說的是模型裡的一個部分 | [10.6](chapters/10.md#10.6) |
| 模態（modality）、模態對齊 | 模態指資料的種類，例如文字、圖片、聲音；拿成對的圖片與正確描述訓練接頭，讓圖片的差異變成文字核心可用的線索，叫模態對齊 | [6.6](chapters/06.md#6.6)、[10.6](chapters/10.md#10.6) |
| 圖片入口（image encoder，也叫編碼器）、聲音入口、語音入口、感知入口 | 把素材轉成特徵的部分，依素材叫圖片入口、聲音入口（audio encoder）、語音入口（speech encoder）；第 19 章把圖片入口、讀字入口與語音入口合稱感知入口 | [10.5](chapters/10.md#10.5)、[12.8](chapters/12.md#12.8)、[12.15](chapters/12.md#12.15)、[第 19 章開頭](chapters/19.md) |
| patch、patch embedding | 圖片切成的小方塊叫 patch；把每塊像素算成特徵的線性層叫 patch embedding | [10.2](chapters/10.md#10.2)、[10.3](chapters/10.md#10.3) |
| 頻譜圖（spectrogram）、mel 頻帶 | 縱軸是頻率格、橫軸是時間框、亮度表示功率的圖叫頻譜圖；依 mel 尺度把相鄰頻率格的功率加權合成的較寬頻帶叫 mel 頻帶，在低頻窄而密、在高頻寬而疏 | [12.5](chapters/12.md#12.5)、[12.6](chapters/12.md#12.6) |
| 取樣率（sample rate） | 聲音每秒記下幾個取樣點，是讀懂時間軸的必要資料；和隨機抽字的抽樣（sampling）不同 | [12.2](chapters/12.md#12.2) |
| 語音辨識（Automatic Speech Recognition，ASR）、轉寫文字（transcript） | 把語音轉成文字的任務叫語音辨識；輸出的文字叫轉寫文字 | [12.11](chapters/12.md#12.11) |

頻率是聲音起伏每秒重複的次數，440Hz就是每秒440次；像單音這樣規律的聲音，頻率較高通常聽起來較高。頻譜圖按時間顯示各頻率成分的強弱；mel 頻帶則用一組可以重疊的三角形權重，把鄰近頻率加權彙整，得到每個時間框中各 mel 頻帶的強弱。前者讓你看各個頻率位置，後者把附近的頻率合成較少的頻帶。

## G.4 後段主題先對應到問題

後段名詞很多，下表先把主要方法各對應到一個問題：想少算、想少存、想學較好回答，或想取得模型原本不知道的資料。這些目的可以組合，也可能互相影響；要先知道自己想改善什麼，再選方法。後面幾張表再收偏好學習、多次嘗試與機率判斷，以及第 19、20 章成品的用語。

| 名詞 | 想解決的問題 | 完整講解 |
| --- | --- | --- |
| Dense、MoE（Mixture of Experts，專家混合） | 想保留多套規則，但每個 token 只花少數幾套的計算：Dense 讓每個 token 都用同一組完整 FFN；MoE 準備多組 FFN（每組叫一位 expert），由 router 替每個 token 打分，只留比例最大的 k 位（top-k routing） | [第 15 章開頭](chapters/15.md)、[15.1](chapters/15.md#15.1)、[15.2](chapters/15.md#15.2)、[15.3](chapters/15.md#15.3)、[15.4](chapters/15.md#15.4) |
| 每 token 使用估計、每 token 參數代理 | 報告 MoE 有多大時，要寫明數了哪些部分：每 token 使用估計只數一層 MoE FFN 裡選中的 expert 加 router；每 token 參數代理是全部非 expert 參數加上每層選中 expert 的參數（實驗紀錄欄位 `active_parameter_proxy_per_token`），連整張 embedding 表與位置表都算入。沒被選中的 expert 仍要存著 | [15.10](chapters/15.md#15.10) |
| thinking effort（也叫 reasoning effort），思考投入；Auto | 使用者對解題資源的偏好；傳入的請求檔位、服務換成的執行設定與實際耗用要分開記。Auto 是服務先根據輸入替每題選預算 | [7.23](chapters/07.md#7.23)、[7.25](chapters/07.md#7.25)、[19.17](chapters/19.md#19.17) |
| 離線理想上界（oracle） | 多花計算最多能好多少：事後看正確答案，替每題挑答對又最省的檔位；部署時看不到正確答案，所以它不是可以上線的成績。C.3 的已知答案命中率（oracle coverage）是另一個量 | [7.25](chapters/07.md#7.25)、[19.18](chapters/19.md#19.18)、[C.3](chapters/0C.md#C.3) |
| Looped Transformer（循環式 Transformer） | 想少存參數：只保留一份區塊參數，把上一輪輸出的狀態交回它反覆處理。省下的是參數的儲存，每多一輪仍要多執行一次區塊 | [14.11](chapters/14.md#14.11) |
| 權重共享（weight tying） | 想少存一張表：輸出層直接用字的 embedding 表當權重。Looped 讓同一個區塊在各輪共用一組參數，是另一種讓同一份參數用在不止一處的做法 | [14.6](chapters/14.md#14.6)、[14.11](chapters/14.md#14.11) |
| 終止深度（exit depth）、逐輪監督（deep supervision） | 讀出答案的那一輪叫終止深度；一次展開到最大輪數，在多個中間狀態各算代價、再加權成一個總代價，叫逐輪監督 | [14.13](chapters/14.md#14.13)、[14.14](chapters/14.md#14.14) |
| 讀提示階段（prefill）、逐步生成階段（decode） | 找出等待花在哪裡：一開始處理整段提示叫讀提示階段，之後每次只處理剛加入的 token 叫逐步生成階段。這裡的 decode 不是 1.1 把 ID 轉回文字的解碼 | [16.2](chapters/16.md#16.2) |
| KV cache（快取） | 保存每層已算好的 K、V，生成下一個 token 時只替新位置計算 | [16.2](chapters/16.md#16.2)、[16.3](chapters/16.md#16.3) |
| FlashAttention | 把分數、softmax 與 Value 加權分塊安排，減少大型中間表的保存與搬移，仍得到原來的加權結果 | [16.9](chapters/16.md#16.9) |
| CUDA | 想在 NVIDIA GPU 上計算：CUDA 是 NVIDIA GPU 的運算環境，命令寫 `--device cuda` 的配方需要它 | [16.8](chapters/16.md#16.8)、[操作頁開頭](training.md) |
| max_new_tokens | 這次最多新增幾個 token，不包含已送入的前文 | [7.19](chapters/07.md#7.19) |
| RoPE（旋轉位置嵌入）、位置插值（Position Interpolation） | 把向量的數字兩兩分組、依位置旋轉來表示位置；延長視窗時，先縮放位置座標再算位置資訊，叫位置插值 | [14.1](chapters/14.md#14.1)、[14.8](chapters/14.md#14.8) |
| 滑動視窗注意力（sliding-window attention） | 每層每個位置只直接讀附近固定寬度的位置；較早的資訊可能經其他位置一層層傳過來 | [16.12](chapters/16.md#16.12) |
| 量化（quantization）、碼、刻度（scale）、還原值 | 想減少表示成本：把原數值換成少數幾個離散選項之一，存下的整數叫碼；相鄰兩個整數代表的數值差叫刻度，用刻度把碼換回的浮點數叫還原值 | [17.1](chapters/17.md#17.1)、[17.2](chapters/17.md#17.2) |
| 蒸餾（distillation）、教師模型 | 讓學生透過教師提供的訊號學習，例如教師的回答或候選比例；教師模型是另一個已訓練好、替新模型提供答案或機率的模型 | [9.13](chapters/09.md#9.13)、[18.1](chapters/18.md#18.1) |
| 屬性教師、風格教師、MoE 教師、圖片教師、圖音教師 | 第 18 章實驗用的五位教師：前三位只讀文字，圖片教師與圖音教師用在多模態實驗，會看圖片 | [18.1](chapters/18.md#18.1)、[18.13](chapters/18.md#18.13) |
| 字表對齊 | 逐欄比較教師與學生的機率之前，先確保兩邊同一欄代表同一個 token；欄數相同還不夠，每一欄的候選身份也要一致 | [18.1](chapters/18.md#18.1)、[18.6](chapters/18.md#18.6) |
| KL 散度（Kullback–Leibler divergence）、熵（H(p)） | 量兩個機率分布差多少：`KL(p‖q)=Σ p_i(log p_i-log q_i)`，兩邊完全相同時是 0。13.14 用它量策略模型偏離參考模型多少，18.8 用它讓學生的比例接近教師。18.8 選讀的熵 `H(p)=−Σ p_i log p_i` 用分布自己的機率衡量它的不確定程度 | [13.14](chapters/13.md#13.14)、[18.8](chapters/18.md#18.8) |
| 推測解碼（speculative decoding） | 想加速既有生成：草稿方便宜地提出候選，主模型（目標模型）決定哪些符合原本的生成規則 | [16.14](chapters/16.md#16.14) |
| 安全、安全標籤 | 想讓回答守住任務明定的邊界：安全標籤問每份回答各自是否通過規則，和問哪份相對較好的偏好標籤不能互相代替；只憑字面拒絕，會擋掉正常的請求 | [9.1](chapters/09.md#9.1)、[9.4](chapters/09.md#9.4) |
| 檢索增強生成（Retrieval-Augmented Generation，RAG） | 想用模型原本不知道的資料：先依問題找出相關段落，再放進模型的前文讓它回答；這只是這一次的輸入，不會寫進模型的參數 | [A.2](chapters/0A.md#A.2)、[A.3](chapters/0A.md#A.3) |
| 工具請求（英文也叫 tool call） | 想讓程式代算：模型生成外層程式讀得懂的工具名稱與參數（一段 JSON 文字），外層先解析、再驗證，兩關都通過才真正執行 | [B.1](chapters/0B.md#B.1)、[B.2](chapters/0B.md#B.2) |

例如「模型檔案太大」可先看量化與蒸餾；「模型已放得下，但接話很慢」可先查時間花在哪裡，再看 cache。檔案變小不等於所有硬體上都變快；少啟動幾位專家也不等於所有權重都能不存。

挑一個你目前想解決的問題，在上表選起點，讀那節的具體數值例子。最後用那個例子解釋為何選它；若只是因為名詞熱門，先回到要改善的量是儲存、時間還是回答品質。

### 偏好與強化學習

| 名詞 | 意思 | 完整講解 |
| --- | --- | --- |
| 獎勵（reward）、獎勵模型（reward model） | 作答後依指定規則打的分數叫獎勵，不自動等於回答的真實品質；先用偏好比較訓練、讀問題與回答後輸出分數的模型叫獎勵模型 | [7.18](chapters/07.md#7.18)、[13.10](chapters/13.md#13.10) |
| 強化學習（Reinforcement Learning，RL）、RLHF（人類回饋強化學習） | 讓模型自己作答、依得到的分數調整，叫強化學習；分數的源頭是人類偏好時，整個流程叫 RLHF，這個名稱交代回饋從哪裡來、流程怎麼接 | [7.18](chapters/07.md#7.18)、[13.17](chapters/13.md#13.17) |
| 近端策略最佳化（PPO）、舊策略、ratio、替代目標 | 強化學習的一種更新方法，每次只讓回答的機率改變一小步。收集作答時的策略叫舊策略，同一筆的新機率÷舊機率叫 ratio（機率比）；每筆算「ratio×優勢」與「裁切後的 ratio×優勢」兩項，取較小者當要提高的替代目標（surrogate objective） | [7.18](chapters/07.md#7.18)、[13.12](chapters/13.md#13.12)、[13.13](chapters/13.md#13.13) |
| 策略模型（policy） | 正在更新、之後實際作答的模型：讀入目前的情境，輸出每個可選動作的機率。在語言模型裡，動作是下一個 token；第 13 章的選卡例子裡，動作是選哪張回答卡 | [13.4](chapters/13.md#13.4)、[13.11](chapters/13.md#13.11) |
| 基準值（baseline）、優勢（advantage）、價值模型（critic） | 作答前估計平均會拿多少分叫基準值；實際獎勵減基準值叫優勢，正負表示高於或低於預期；只看題目情境、預測這個平均的模型叫價值模型（英文也叫 value model） | [13.11](chapters/13.md#13.11)、[13.14](chapters/13.md#13.14) |
| 參考模型（reference，也叫固定參考） | 保留這次訓練的起點、不跟著更新的副本：DPO 用它作固定基準，第 13 章的 PPO 用 KL 散度量策略模型偏離它多少 | [7.18](chapters/07.md#7.18)、[13.4](chapters/13.md#13.4)、[13.14](chapters/13.md#13.14) |
| 回答卡、選卡 | 事先寫好的候選回答，模型的回答就是其中一張；從幾張中挑出一張叫選卡。第 13 章用選回答卡的小任務把評分和作答分開；B 單元的每張卡是一個表示下一步動作的固定標記 | [13.10](chapters/13.md#13.10)、[B.5](chapters/0B.md#B.5) |
| 策略梯度（policy gradient）、REINFORCE | 用「優勢 × 所選動作的 log 機率」決定更新方向的方法；C.7 補充裡訓練用的 REINFORCE 是一種策略梯度方法 | [C.7](chapters/0C.md#C.7) |
| 可驗證獎勵強化學習（RLVR） | 用能直接驗算的正確答案給獎勵的強化學習；它會更新參數，和 C.5 參數不變的作答計算不同 | [C.7](chapters/0C.md#C.7) |
| 寬鬆評分、嚴格評分、獎勵鑽漏洞（reward hacking） | C.6 的兩條評分規則：嚴格評分要求整段文字是一個整數，而且等於正確答案；寬鬆評分只要按空白切開後有一段等於正確答案就給分，把 0 到 15 全部列出來的回答也得分。策略利用評分弱點拿分、卻沒完成真正任務，叫獎勵鑽漏洞 | [C.6](chapters/0C.md#C.6)、[13.16](chapters/13.md#13.16) |

查後訓練方法時，先分清材料來源與更新方法：人類回答比較可以提供回饋，PPO則說明一種更新方式。第13章用預寫回答卡讓一次選擇與得分清楚可見，實際文字生成需要再處理逐 token 的選擇。直接用偏好對更新的DPO，可從G.2的連結查起。

### 多試幾次、驗算與機率判斷

| 名詞 | 意思 | 完整講解 |
| --- | --- | --- |
| 驗證器（verifier）、驗算 | 依明確規則核對回答的程式叫驗證器，它做的核對叫驗算；說一個驗證器可靠時，要說清它驗了哪幾項 | [C.2](chapters/0C.md#C.2) |
| 集合命中、pass@k | 每題抽 k 份候選，至少一份正確叫集合命中；評測生成程式的模型時常用的 pass@k 問的是同一件事。原論文的估計式另有候選池大小 n，可以比 k 大 | [C.3](chapters/0C.md#C.3) |
| 已知答案命中率（oracle coverage） | 用已知的正確答案逐份檢查候選，集合命中的題數占全部題數的比例；要事先知道答案才算得出，不能當成能實際部署的選法的成績 | [C.3](chapters/0C.md#C.3) |
| 有型別的機率判斷、Jev、Choice、Noul、Score | 一次判斷由狀態材料、問題與候選三部分組成，回傳型別固定的機率。Jev 是 TypeSafe 公司提供的判斷服務，官方分三種型別：Choice 回傳選項分布，Noul 回傳 yes 機率，Score 回傳有次序的有限等級分布 | [9.11](chapters/09.md#9.11) |
| 信心（confidence）、集中度（concentration） | 第 9 章把 softmax 後最大的那個機率叫信心，它只比較在場候選，不查正確答案；Jev 的 Choice 回傳的 confidence 欄位，本課叫集中度，n 個候選時算 `(最大機率−1/n)÷(1−1/n)`，描述分布相對均勻基準有多集中，不是由答對紀錄算出的頻率 | [9.9](chapters/09.md#9.9)、[9.15](chapters/09.md#9.15) |
| 校準（calibration）、裁判核對、範圍校準（range calibration） | 校準指讓報出的機率和實際答對的比例相稱，例如報 0.8 的那些題大約八成答對。8.10 先用已知例子核對裁判的判準，叫裁判核對；17.12 用一批輸入記下中間特徵的範圍，叫範圍校準，和第 9 章的機率校準是兩件事 | [9.9](chapters/09.md#9.9)、[8.10](chapters/08.md#8.10)、[17.12](chapters/17.md#17.12) |
| 溫度縮放（temperature scaling）、校準資料（calibration split） | 常用的校準方法：選一個校準溫度 τ，把各候選的原始分數除以 τ，再做 softmax；τ 要在另一份沒拿來訓練的校準資料上選 | [9.10](chapters/09.md#9.10) |
| 決策資料（policy split）、決策規則 | 第 9 章說實際資料至少分成四份：訓練資料更新參數，校準資料選溫度，決策資料選行動門檻，測試資料最後只報凍結後的成績。這裡的 policy 和第 13 章的策略模型無關。決定下一步怎麼做的規則叫決策規則：B.5 依問題與工具狀態選下一個動作，19.17 把機率判斷器的判斷變成預算 | [9.14](chapters/09.md#9.14)、[B.5](chapters/0B.md#B.5)、[19.17](chapters/19.md#19.17) |

### 第 19、20 章的成品與外部模型

| 名詞 | 意思 | 完整講解 |
| --- | --- | --- |
| MoE 助理、Looped 解題器、機率判斷器 | 第 19 章從隨機參數訓練的三個成品：MoE 助理讀文字、商品圖、字卡與錄音後用文字回答；Looped 解題器讀固定語法的事件或加法題，算出答案；機率判斷器依狀態、問題與候選意義回傳機率 | [第 19 章開頭](chapters/19.md)、[19.13](chapters/19.md#19.13) |
| 助手（assistant）、助理 | 對話資料替每段話記一個角色（role），寫明這段話是誰說的；模型那一方的角色叫助手，角色欄位寫 assistant。第 19 章的 MoE 助理與第 20 章的助理，指的是整個成品 | [6.6](chapters/06.md#6.6)、[7.1](chapters/07.md#7.1)、[19.13](chapters/19.md#19.13)、[20.1](chapters/20.md#20.1) |
| 字卡 | 讀字用的中文字圖片，由作者製作；和第 13 章、B 單元的回答卡不同 | [11.18](chapters/11.md#11.18)、[19.13](chapters/19.md#19.13) |
| 跨切分重複的輸入句 | 同一段輸入（連同問題與候選）出現在不同資料份的列裡；家族沒有跨份，不等於模型從未看過這段文字 | [19.14](chapters/19.md#19.14) |
| 前處理器（processor） | Qwen 附的前處理程式：把圖片換成模型能讀的輸入，並套用 Qwen 的對話模板標出角色 | [20.7](chapters/20.md#20.7) |
| Qwen、Whisper | 第 20 章接手、別人已經訓練好的兩個成熟模型：Qwen3-VL-2B-Instruct 讀照片與文字、生成回答，也是這一章的底座；Whisper 做語音辨識，輸出轉寫文字 | [第 20 章開頭](chapters/20.md)、[20.1](chapters/20.md#20.1) |

### 英文與縮寫索引

只收上面各表的英文名詞與縮寫，依英文字母順序排列，不分大小寫，空格、連字號與符號不計。每個詞列一次，連到下定義的地方；括號裡是本課用的中文名稱或符號。decode 在本課有兩個意思，兩個都列出。

**A**　[activation](chapters/17.md#17.11)（中間特徵） · [activation checkpointing](chapters/16.md#16.10)（中間特徵重算） · [activation function](chapters/02.md#2.4)（啟用函式） · [Adam](chapters/05.md#5.4) · [adapter](chapters/08.md#8.9)（適配器） · [advantage](chapters/13.md#13.11)（優勢） · [ASR](chapters/12.md#12.11)（語音辨識） · [attention](chapters/03.md#3.1)（注意力） · [attention head](chapters/03.md#3.7)（注意力頭） · [attention weights](chapters/03.md#3.1)（讀取比例） · [audio encoder](chapters/12.md#12.8)（聲音入口） · [Auto](chapters/07.md#7.25) · [autoregressive](chapters/07.md#7.20)（逐 token 生成） · [auxiliary loss](chapters/04.md#4.5)（輔助代價） · [axis](first-steps.md#W.3)（軸）

**B**　[backpropagation](chapters/01.md#1.10)（反向傳播） · [baseline](chapters/13.md#13.11)（基準值） · [base model](chapters/07.md#7.11)（底座） · [batch](chapters/05.md#5.2) · [bias](chapters/02.md#2.3)（偏移） · [bigram](chapters/01.md#1.5)（接字表） · [bit](chapters/05.md#5.9)（位元） · [BOS](chapters/06.md#6.5)（開始標記） · [byte](chapters/05.md#5.9)（位元組）

**C**　[calibration](chapters/09.md#9.9)（校準） · [calibration split](chapters/09.md#9.10)（校準資料） · [causal mask](chapters/03.md#3.6)（因果遮罩） · [checkpoint](chapters/05.md#5.7)（檢查點） · [Choice](chapters/09.md#9.11) · [classification head](chapters/10.md#10.5)（分類頭） · [code point](chapters/06.md#6.1)（碼點） · [concentration](chapters/09.md#9.15)（集中度） · [confidence](chapters/09.md#9.9)（信心） · [context](chapters/02.md#2.5)（上下文） · [context window](chapters/02.md#2.5)（上下文視窗） · [control](chapters/12.md#12.4)（對照） · [critic](chapters/13.md#13.14)（價值模型） · [cross entropy](chapters/01.md#1.8)（交叉熵） · [CUDA](chapters/16.md#16.8)

**D**　decode（[解碼](chapters/01.md#1.1)、[逐步生成階段](chapters/16.md#16.2)） · [deep supervision](chapters/14.md#14.14)（逐輪監督） · [Dense](chapters/15.md#15.1) · [derivative](chapters/01.md#1.9)（導數） · [distillation](chapters/18.md#18.1)（蒸餾） · [DPO](chapters/07.md#7.18)（直接偏好最佳化）

**E**　[embedding](chapters/02.md#2.1)（embedding 表、embedding 向量） · [encode](chapters/01.md#1.1)（編碼） · [EOS](chapters/06.md#6.5)（結束標記） · [exit depth](chapters/14.md#14.13)（終止深度） · [expert](chapters/15.md#15.2)（專家）

**F**　[family](chapters/05.md#5.12)（家族） · [FFN](chapters/04.md#4.4)（前饋網路） · [FlashAttention](chapters/16.md#16.9) · [forward](chapters/01.md#1.10)（前向計算）

**G**　[gradient](chapters/01.md#1.9)（梯度） · [gradient descent](chapters/01.md#1.12)（梯度下降） · [greedy](chapters/01.md#1.15) · [ground truth](chapters/01.md#1.8)（正確答案）

**H**　[held-out](chapters/01.md#1.12)（留出） · [hidden](chapters/04.md#4.4)（中間寬度）

**I**　[ID](chapters/01.md#1.1) · [image encoder](chapters/10.md#10.5)（圖片入口） · [inference](chapters/01.md#1.15)（推論）

**J**　[Jev](chapters/09.md#9.11) · [JSON](chapters/07.md#7.13)

**K**　[kernel](chapters/17.md#17.10)（計算程式） · [key](chapters/03.md#3.2)（K） · [KL](chapters/13.md#13.14)（KL 散度） · [KV cache](chapters/16.md#16.2)（快取）

**L**　[labels](chapters/05.md#5.2) · [language backbone](chapters/10.md#10.6)（文字核心） · [LayerNorm](chapters/04.md#4.3)（層正規化） · [learning rate](chapters/01.md#1.12)（學習率，lr） · [Linear](chapters/02.md#2.3)（線性層） · [LM head](chapters/04.md#4.6)（輸出層） · [logits](chapters/01.md#1.8)（候選分數） · [Looped Transformer](chapters/14.md#14.11)（循環式 Transformer） · [LoRA](chapters/08.md#8.8) · [loss](chapters/01.md#1.8)（代價） · [loss mask](chapters/07.md#7.24)（代價遮罩）

**M**　[max_new_tokens](chapters/07.md#7.19) · [mel](chapters/12.md#12.6)（mel 頻帶） · [MLP](chapters/02.md#2.5)（多層感知器） · [modality](chapters/06.md#6.6)（模態） · [MoE](chapters/15.md)（專家混合） · [momentum](chapters/05.md#5.3)（動量） · [MTP](chapters/07.md#7.20)（多 token 預測）

**N**　[NaN](chapters/03.md#3.6) · [NLL](chapters/06.md#6.5)（負對數似然） · [Noul](chapters/09.md#9.11)

**O**　[optimizer](chapters/01.md#1.13)（最佳化器） · [optimizer state](chapters/05.md#5.7)（最佳化器狀態） · [oracle](chapters/07.md#7.25)（離線理想上界） · [oracle coverage](chapters/0C.md#C.3)（已知答案命中率） · [output layer](chapters/04.md#4.6)（輸出層）

**P**　[PAD](chapters/06.md#6.6)（填充） · [padding](chapters/07.md#7.6)（補齊） · [parameter](chapters/01.md#1.2)（參數） · [pass@k](chapters/0C.md#C.3) · [patch](chapters/10.md#10.2) · [patch embedding](chapters/10.md#10.3) · [policy](chapters/13.md#13.4)（策略模型） · [policy gradient](chapters/0C.md#C.7)（策略梯度） · [policy split](chapters/09.md#9.14)（決策資料） · [Position Interpolation](chapters/14.md#14.8)（位置插值） · [post-training](chapters/07.md#7.17)（後訓練） · [PPO](chapters/07.md#7.18)（近端策略最佳化） · [prefill](chapters/16.md#16.2)（讀提示階段） · [pretraining](chapters/07.md#7.11)（預訓練） · [processor](chapters/20.md#20.7)（前處理器） · [projector](chapters/10.md#10.6)（接頭）

**Q**　[quantization](chapters/17.md#17.1)（量化） · [query](chapters/03.md#3.2)（Q） · [Qwen](chapters/20.md#20.1)

**R**　[RAG](chapters/0A.md#A.2)（檢索增強生成） · [range calibration](chapters/17.md#17.12)（範圍校準） · [rank](chapters/08.md#8.8)（秩） · [ratio](chapters/13.md#13.12)（機率比） · [reasoning effort](chapters/07.md#7.23)（思考投入） · [reference](chapters/13.md#13.4)（參考模型） · [REINFORCE](chapters/0C.md#C.7) · [residual connection](chapters/04.md#4.2)（殘差連線） · [reward](chapters/07.md#7.18)（獎勵） · [reward hacking](chapters/13.md#13.16)（獎勵鑽漏洞） · [reward model](chapters/13.md#13.10)（獎勵模型） · [RL](chapters/07.md#7.18)（強化學習） · [RLHF](chapters/07.md#7.18)（人類回饋強化學習） · [RLVR](chapters/0C.md#C.7)（可驗證獎勵強化學習） · [RMSNorm](chapters/14.md#14.2)（均方根正規化） · [RoPE](chapters/14.md#14.1)（旋轉位置嵌入） · [router](chapters/15.md#15.3)（路由器）

**S**　[sample rate](chapters/12.md#12.2)（取樣率） · [sampling](chapters/01.md#1.14)（抽樣） · [scale](chapters/17.md#17.2)（刻度） · [Score](chapters/09.md#9.11) · [SFT](chapters/07.md#7.1)（監督式微調） · [shape](first-steps.md#W.3) · [shift](chapters/01.md#1.3)（下一位置對齊） · [sliding-window attention](chapters/16.md#16.12)（滑動視窗注意力） · [softmax](chapters/01.md#1.7) · [special token](chapters/06.md#6.6)（特殊 token） · [spectrogram](chapters/12.md#12.5)（頻譜圖） · [speculative decoding](chapters/16.md#16.14)（推測解碼） · [speech encoder](chapters/12.md#12.15)（語音入口） · [surrogate objective](chapters/13.md#13.13)（替代目標）

**T**　[teacher forcing](chapters/07.md#7.22) · [temperature](chapters/01.md#1.15)（溫度） · [temperature scaling](chapters/09.md#9.10)（溫度縮放） · [tensor](first-steps.md#W.3)（張量） · [thinking effort](chapters/07.md#7.23)（思考投入） · [token](chapters/06.md#6.1) · [tokenization](chapters/06.md#6.1)（切詞） · [tokenizer](chapters/06.md#6.1) · [tool call](chapters/0B.md#B.1)（工具請求） · [top-k routing](chapters/15.md#15.4) · [transcript](chapters/12.md#12.11)（轉寫文字）

**U**　[UTF-8](chapters/06.md#6.1)

**V**　[value](chapters/03.md#3.4)（V） · [value model](chapters/13.md#13.14)（價值模型） · [verifier](chapters/0C.md#C.2)（驗證器） · [vocabulary](chapters/01.md#1.1)（字表）

**W**　[weight](chapters/02.md#2.3)（權重） · [weight tying](chapters/14.md#14.6)（權重共享） · [Whisper](chapters/20.md#20.1) · [width](chapters/02.md#2.1)（特徵寬度）
