## 固定資料與授權

v2資料清單固定SHA-256為 `3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b`。Git LFS包為程式庫commit `08761dac87a6ef360db95883d9bcd338c44fe76d` 的 `assets/training/selftrained-v2.tar.gz`，壓縮大小67,862,431 bytes，解包內容127,161,811 bytes，包SHA-256為 `0976073a3bc7c331a65cedb4c31d54f5c6e9a5014ff2443698a8e2b8cad7e78a`。

包內12個JSONL與8,950個其他檔案（包含圖片、錄音與來源說明），共8,962個固定檔案，逐檔指紋在[manifest](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/v2-manifest.json)。訓練／validation／test各28,876／2,435／3,734題；錄音各62／15／30段。問法、格式與衍生材料可能共享來源，題數不是獨立素材數；錄音沒有已知speaker ID，不宣稱不同側由不同說話者錄製。

本課原創文字與工具題採MIT；Fashion-MNIST由Han Xiao、Kashif Rasul、Roland Vollgraf與Zalando Research提供，採MIT；[PolyAI MInDS-14](https://huggingface.co/datasets/PolyAI/minds14/tree/40ce77cb32a384e4d50a568e1ec39ac804019d33)中文錄音採CC BY 4.0；Noto CJK Sans／Serif字型採SIL OFL 1.1。包內保留來源、固定版本、授權及修改說明。公開模型權重的MIT授權不會取代原始資料或字型授權。

