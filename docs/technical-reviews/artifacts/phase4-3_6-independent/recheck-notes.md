# 修訂後本審閱者的實際複查

親讀目前course/chapters/03.md#3.6完整原始UTF-8小節（179–222行），不是只更新fingerprint。
原問題所在句現在明說[None,None]為許可表增加B/H兩軸，遮罩[1,1,4,4]的最後兩軸為四個
query位置與四個候選位置；另外明說Q/K/V資料[1,1,4,2]的尾軸為四位置與兩特徵。

親自重新對照本次已存官方來源：PyTorch installed commit的torch.tril docstring
12067–12088行說保留主對角線及以下；functional.py 6383–6405行明示L,S、
Q @ K.transpose(-2,-1)、softmax最後候選軸及@V；6485–6512行明示Q(N,Hq,L,E)、
K(N,H,S,E)、V(N,H,S,Ev)、mask/weight(N,Hq,L,S)。
因此本例矩陣尺寸[4,2]@[2,4]→[4,4]，再[4,4]@[4,2]→[4,2]；B=H=1只包住這兩位置軸。
`tril()[None,None]`從bool[4,4]得到[1,1,4,4]，不能變成[1,1,4,2]。
新文字已把原本模糊的資料/許可區分清楚；issue可resolved。

另親讀真正attention.py 10–28行：allowed=query×key的位置不等式；分數先除sqrt(Dk)，
~allowed填−inf再softmax(dim=-1)，權重column對應同index的V row，返回out,weights。
重新讀原論文p3 §3.1 Decoder與p5 §3.2.3 decoder bullet：shift與mask須合用，當前位置可讀自身。
全文其餘claim仍正確：Q/K全零合法候選均分，四組原V均值，clone未來V+100前3輸出不變，
去掉tril使所有輸出改变並讓assert失敗；這只測資訊範圍，無參數更新，低loss不證明因果防偷看。

本次沿用首次審閱的CPU原fence/有意義變體/官方SDPA對照與Inkscape圖證據；沒有重新執行fence、
沒有重新render或view_image。新/舊唯一fence bytes與SHA、SVG bytes與SHA、attention.py bytes與SHA
均親自用程式核對完全相同，初稿evidence manifest全部原檔SHA保持相同。圖解既有親看記錄仍有效。
本次新执行只做extract/hash/diff/證據完整性/schema checker；不冒稱重新跑CPU計算或圖截圖。
