# 2.1 獨立查證記錄

審閱者：`/root/phase4_factual_coordinator/factual_2_1`，fresh context。未讀舊 2.1 審核報告或其他歷史判定。

原稿親讀：`course/chapters/02.md` 行 1–200；判定只涵蓋 2.1（行 5–63）與本章導言（行 1–4）。另親讀 `01.md` 行 202–234、539–548，核對「上一章直接查候選分數、只看最後一字」的必要前提；未重審那一章的實測成績。親讀 `tiny_perceptron/simple.py` 全檔，`scripts/build_course.py` BOOTSTRAP、`section_facts.py` 與 checker 全部契約。導言 bytes、section bytes 與 SVG 原檔均已永久保存。

自己的導言摘要：上一章只取最近一字，兩個欄位前綴同以等號結尾時無法給不同分配。本章計畫先把每字轉成可調向量，按位置合併，再送進可調運算；會區分輸入包含線索與模型是否已學會使用線索。這是章節路線與模型輸入限制的說明，不是本節已驗收訓練品質。

## 原始權威來源的親讀範圍

- Bengio et al., JMLR 3 (2003), 1137–1155，出版社原 PDF。親讀 abstract、§1.1（p.1139），以及 §2（pp.1141–1143）、Figure 1、Eq.(1)、梯度更新式。C 的每列是詞的 m 維實數向量，C 是 |V|×m 個自由參數；後續 g 用表示產生下一詞機率，C 與後續參數共同學習。m 與 |V| 可以不同，C 共用於上下文內各詞位置。特徵座標沒有要求使用人類命名的語義欄位。這只支持方法概念，不把教材人工的 0/1 表視為已學到的語義或論文成績。
- PyTorch 官方 GitHub `v2.9.0`，`torch/nn/modules/sparse.py` lines 15–204：Embedding 的查表契約、weight shape (num_embeddings, embedding_dim)、預設可求導 Parameter、N(0,1) 初值、reset_parameters 與 forward。`torch/nn/functional.py` lines 2428–2546：F.embedding 的 indices、weight、輸出附加 embedding_dim 的軸，以及 default padding_idx/max_norm。`aten/src/ATen/native/Embedding.cpp` lines 37–54、112–178：前向 index_select、附加特徵軸與 CPU 反向累加至被索引列。只套用本例 default padding_idx=None、max_norm=None、scale_grad_by_freq=False、sparse=False；未聲稱帶 padding 或 weight decay 的任何更新都只影響使用列。
- PyTorch 官方 GitHub `v2.9.0`，`torch/autograd/grad_mode.py` lines 21–86：no_grad 關閉區塊內求導，離開區塊恢復原設定。`torch/_tensor_docs.py` lines 1204–1224：copy_ 把來源元素寫入原 Tensor。`torch/_torch_docs.py` lines 4064–4085、9223–9275：equal 是尺寸與元素相等判斷，tensor 從 data 建立 Tensor、型別推斷與預設 requires_grad=False。本節 no_grad 只用於填表，不會把原先 Embedding Parameter 永久凍結。
- Vaswani et al., Attention Is All You Need, arXiv:1706.03762v7。親讀 §3.2.3（p.5，encoder/decoder self-attention 的上下文範圍）、§3.4（p.5，learned token embedding）與 §3.5（p.6，把位置信息加入 token embedding）。支持同字查表起點相同、加入位置與上下文後中間表示可能不同；不聲稱 2.1 已建立 attention/位置層，也不聲稱各位置必然不同。

來源 URL、實際 HTTP 200、存取日與下載 bytes 的 SHA-256 見 `sources/receipts.json`。官方 PyTorch 來源固定為 v2.9.0；實際安裝與執行是 Python 3.13.5 / torch 2.14.1+cpu，git hash 5c4886908584029761b579af026dcfb627c84070。來源契約與安裝行為分開核對，不冒稱兩者同版本。

## 執行與推導

原始 fence 逐 byte 擷取後由 helper 在 CPU 執行，exit 0，僅 1 個 Python fence。stdout 的 shape 是 [1,3,3]（B=1 筆、C=3 位置、D=3 特徵），內容依 [1,2,1] 取貓/看/貓；首尾 exact equal。原碼沒有候選層、softmax、代價、backward 或 optimizer.step；no_grad 內 copy_ 是人工賦值，不是從答案學習。

`probe.py` 在既有 .venv CPU 上另做有界檢查，exit 0。中間 ID 由 2 改 1 後，三位置全為 [1,0,0]、shape 不變。重新編號並搬動对应表列得到相同表示，說明 ID 只作選列約定。nn.Embedding(5,3) 的參數是 5×3=15，而 BigramLM 的表為 5×5；BigramLM 對最後 ID 相同的兩個不同前綴，產生完全相同分數。

手算候選配方是點積 [2,0,1]：貓 [1,0,0] 得 2×1+0=2，看 [0,1,0] 得 2×0+0=0。改貓首格 1→1.5 得 2×1.5+0=3；另把配方首乘數 2→3，原貓向量得 3。量是無單位的示意特徵/分數，沒有樣本統計分母。這些 0、1、1.5、2、3 都可用浮點精確表示，lookup/shape/count/score 採 exact equality，不引入任意 tolerance。

sum(x) 的獨立預期梯度為：貓列出現兩次→[2,2,2]，看列一次→[1,1,1]，其他列→0；實際精確相等。forward 與 backward 後未 step，表格仍精確不變。兩個 seed 的新建 Embedding 權重不同，只作亂數初始化的行為佐證。

另用 1 筆、1 位置、5 候選、目標 ID 3 的人工 head 做單次 SGD（lr=0.1）更新。正確候選概率 0.6487856507301331→0.6976850032806396；交叉熵 0.43265295028686523→0.3599875569343567。查表梯度僅進使用的貓列，step 前參數不變，step 後該列與 head 都改變。這個數字是審閱者自建的機制檢查，不是教材原碼的輸出、重訓、語言能力或準確率。其 softmax 分母是五個候選 exp(logit) 的總和；只要求本例方向成立，不從一小步推論訓練必然成功。

## 圖的親看

已讀原 SVG，實際以 Inkscape 1.4 匯出 640×1060 PNG，exit 0，並用 view_image 親看保存的畫面。五列「。/貓/看/狗/，」與數值表一致，貓列突出且箭頭由查表接到共用候選狗配方；2、0 與假設改值後 3 均和原文吻合。圖上「手動指定的表」「不是已訓練的字義」與底部流程吻合本节範圍。所有文字可見，沒有裁切、重疊或把 ID 當特徵的箭頭錯置。使用 Inkscape，未做瀏覽器渲染；其 stderr 有 PangoFT2FontMap/GtkRecentManager 初始化警告，但輸出畫面可判讀，未阻止此靜態 SVG 查證。

判定：目前本節無未確定或錯誤的實質主張，pass。手算配方與原碼範圍一致，不宣稱本節學習成功；本次沒有改教材或 SVG。
