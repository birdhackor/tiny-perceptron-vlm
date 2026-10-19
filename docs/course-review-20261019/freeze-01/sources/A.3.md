## A.3 文件多了怎麼找相關段落？

查詢「書店地址」，兩份文件是 d1「貓喜歡曬太陽」和 d2「書店地址是青街8號」。最簡單的找法是看共同片段：書、店、地、址都在 d2 出現，四項吻合；d1 一項也沒有。數吻合的片段可以替文件排序，但這個分數不是答案正確的機率。下面用課程函式庫的 `lexical_terms` 與 `retrieve` 跑這個例子。

```python
from tiny_perceptron.retrieval import lexical_terms, retrieve

query = "書店地址"
docs = [
    {"id": "d1", "text": "貓喜歡曬太陽"},
    {"id": "d2", "text": "書店地址是青街8號"},
]
print("查詢片段", lexical_terms(query))
print("d2 的片段", lexical_terms(docs[1]["text"]))
print("取回", retrieve(query, docs, k=1))
print("改問法", retrieve("營業處在哪", docs, k=1))
```

`lexical_terms` 把文字切成片段：中文每個字一段，「8」這種數字也自成一段。`retrieve` 依片段替每份文件打分，`k=1` 表示最多取回一份，所以只交回 d2。改問「營業處在哪」時，兩份文件都沒有共同片段，結果是空清單：這套規則只比對字面，不知道「營業處」和「書店地址」意思相近。

這兩個函式是怎麼寫的？下面是原始碼。原始碼檔案開頭有 `import re` 與 `from collections import Counter`，提供裡面用到的兩個工具：

```python
# 節錄自 tiny_perceptron/retrieval.py：lexical_terms、retrieve
def lexical_terms(text):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def retrieve(query, documents, k=2):
    terms = Counter(lexical_terms(query))
    scored = []
    for document in documents:
        counts = Counter(lexical_terms(document["text"]))
        score = sum(min(counts[t], amount) for t, amount in terms.items())
        if score:
            scored.append((score, document))
    return [d for _, d in sorted(scored, key=lambda item: (-item[0], item[1]["id"]))[:k]]
```

`lexical_terms` 只有一行。`text.lower()` 先把英文字母轉成小寫，`K017` 與 `k017` 因此算同一段。`re` 是 Python 的正規表示式（regular expression）工具，用一串符號描述「要找什麼樣的文字」。這裡的樣式有兩種，用 `|`（或）隔開：`[a-z0-9]+` 是「一個以上連續的小寫字母或數字」，所以 `k017` 整段算一個片段；`[\u4e00-\u9fff]` 是「一個常用中文字」，`\u4e00` 到 `\u9fff` 是 Unicode 碼點（[6.1](06.md#6.1)）裡的一段範圍，涵蓋常用漢字。樣式字串引號前的 `r` 讓反斜線照字面保留（見 [C.6](0C.md#C.6)），`\u4e00` 這種寫法由 `re` 自己解讀。`re.findall` 由左到右找出所有符合的片段，排成清單；既不屬於 a–z、0–9，也不在這段範圍內的字元（例如標點與空白），都不符合任何一種，就被略過。

`retrieve` 先把查詢的片段數成一張表：`Counter` 記下每種項目出現幾次，查不到的項目當成 0 次，不會報錯。接著對每份文件做同一件事，再用 `sum(min(counts[t], amount) for t, amount in terms.items())` 打分：`terms.items()` 逐一取出查詢的（片段, 次數），每種片段取「文件裡的次數」與「查詢裡的次數」兩者較小的那個，再全部相加。所以查詢只有一個「書」時，文件就算寫了五次「書」，也只算 1 分；文件不能靠重複關鍵字灌分。

`if score:` 把 0 分當成不成立，零分文件不放進 `scored`，所以完全沒有共同片段時回傳空清單，不會硬交一份；有分數的文件則以（分數, 文件）一對放進 `scored`。最後一行排序：`key=lambda item: (-item[0], item[1]["id"])`（`key=`、`lambda` 見 W.2）裡的 `item` 就是這一對，`item[0]` 是分數，`item[1]["id"]` 是文件編號。它替每份文件做出一組兩個值，Python 比較兩組值時先比第一個，相同才比第二個。`sorted` 預設由小到大排（見 W.2），第一個值是分數加上負號，所以分數越高、負得越多，排得越前；同分時再依文件編號排，例如 d1、d2 同分時 d1 在前，排序結果因此固定。`[:k]` 只留前 k 份（切片見 W.2），`[d for _, d in ...]` 丟掉分數、只交回文件本身；`_` 是「這個值用不到」的習慣命名。

下面這段要接在前兩段後面執行（見 [W.1](../first-steps.md#W.1)）：`query` 來自第一段；執行過上一段節錄後，`lexical_terms` 指的是節錄的定義，取代了第一段從函式庫匯入的同名函式，跳過上一段則仍是函式庫版。這段照 `retrieve` 的算法數片段、打分，拆開看三段文字各得幾分。節錄的 `lexical_terms` 呼叫時要用 `re`，數片段要用 `Counter`，所以先載入：

```python
import re
from collections import Counter

terms = Counter(lexical_terms(query))
print("查詢的計數", dict(terms))
for text in ["書店地址是青街8號", "書書書書書店", "K017 address=C2"]:
    counts = Counter(lexical_terms(text))
    score = sum(min(counts[t], amount) for t, amount in terms.items())
    print(text, lexical_terms(text), "分數", score)
```

查詢的計數是四個字各 1 次。「書店地址是青街8號」切出的片段，和第一段用函式庫版印出的「d2 的片段」相同，得 4 分。「書書書書書店」的「書」雖然出現五次，`min` 只算 1，加上「店」共 2 分。`K017 address=C2` 被切成 `k017`、`address`、`c2` 三段（大寫轉成小寫，字母和數字連成一段），和中文查詢沒有共同片段，得 0 分；補充區 RAG 實驗的文件就是這種寫法。

被選中不等於資料足夠。若 d2 寫的是「書店地址尚未公布」，書、店、地、址照樣吻合，分數仍是 4，它照樣會被取回，卻沒有地址可答。不論用哪種排序方法，都要拿標明了正確來源的題目，核對所需的文件有沒有被取回，不能因為結果不是空的，就當成答案可靠。

練習只把第一段 d2 的 `text` 改成 `"書店地址尚未公布"`，先預測：「d2 的片段」那一行變成八個字，取回的仍是 d2；再執行核對。接著把 `k=1` 改成 `k=2`，先預測 d1 仍不會出現，因為零分文件不回傳。

<details>
<summary>補充：RAG 實驗的檢索命中</summary>

RAG 實驗用的就是這個 `retrieve`，每次只取一份（`k=1`）。12 家測試店的公告混進 60 份其他店的公告，共 72 份文件，每份都是 `K017 address=C2` 這種寫法。12 題中有 9 題用 `K017` 這種原本的店名查詢；另外 3 題刻意改用 `shop003` 這種別名，字詞規則找不到共同片段。結果正確來源命中 9/12：那 3 次落空是預先安排的查詢限制，不是語言模型的生成錯誤。檢索文件有固定的編號，例如 `doc-K017`；放進提示時改用短的引用編號，例如 `D1`，兩者的對應逐題保存，才能回查。

數字在[實驗紀錄 rag.json](../../docs/course-experiments/results/rag.json)的 `results.retrieval_recall_at_1`；逐題的查詢、取回的文件與編號對應，在 `results.samples` 裡 `mode` 為 `retrieved_context` 的 12 筆，欄位是 `query`、`retrieved_ids` 與 `citation_id_map`。

</details>

