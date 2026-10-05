## 16.4 KV cache 怎麼縮小？

如果每個注意力頭都保存一份Key、Value，長對話會讓快取占用持續增加。可以保留多個Query頭，讓幾個Query頭共用同一組K、V嗎？一個[注意力頭](03.md#3.7)用一小組特徵做比對；[KV快取](16.md#16.3)則按每層、每頭、每個過去位置保存K、V。現在固定四個Query頭，只改保存的KV組數。多個Query頭共用較少的KV頭，叫Grouped-Query Attention（GQA，分組查詢注意力）。

一般分組設定要求Query頭數能被KV頭數整除，例如四個Query、兩個KV，每組有兩個Query；全部Query共用一個KV的情況也常稱Multi-Query Attention（MQA）。投影權重的形狀與共享方式會改變，不能隨意把訓練好的四頭KV重排成一頭，就宣稱原模型功能保持不變。
