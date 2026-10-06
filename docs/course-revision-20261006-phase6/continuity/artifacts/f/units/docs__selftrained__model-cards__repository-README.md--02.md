## 自行訓練v2：四組固定推論輸出

公開revision固定為 **`979cdfacc588ad0536f1c64fff96f264571cf054`**，四組位於 `selftrained/v2/`，共16個檔案、81,501,640 bytes，採MIT授權。每組包含 `model.safetensors`、`model-config.json`、`tokenizer.json` 和 `inference-manifest.json`。請將四個檔案作為一組，由本課自訂PyTorch模型和 `scripts/selftrained/chat.py` 載入。

| 公開路徑與固定下載 | 用途 | 該段完成步數 | 選定步數 | inference manifest SHA-256 |
| --- | --- | ---: | ---: | --- |
| [selftrained/v2/moe-pretrain](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-pretrain) | 文字預訓練里程碑 | 1,000 | 250 | `6c5fbeb80491d32743ed6739b5e0dc0622d3d31b168b396af5677c2df35a6d15` |
| [selftrained/v2/moe-sft](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-sft) | 對話示範訓練里程碑 | 8,000 | 8,000 | `b909f7347b6f49e6ca746a32ec1e834afe6b95ec9fa38715f5bf54e194109a79` |
| [selftrained/v2/moe-joint](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-joint) | 最後選定MoE：native joint段 | 4,000 | 1,000 | `f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e` |
| [selftrained/v2/dense-joint](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/dense-joint) | 最後選定Dense：weighted joint段 | 10,000 | 1,000 | `dcb8ff538a958cd1a69b765056522e421df95b991566209aeb85046817596269` |

選定權重依validation比較保存，步數是該段中的位置，不是整條訓練歷史的累計步數。pretrain和SFT是中間里程碑，下面的最後測試表只評最後兩組joint權重，不能套到前兩組。

MoE最後段的tool／numeric／native-voice loss weight為4／1／4，Dense最後段為4／4／1；兩版的選定歷史也不同。完整12段baseline以及最後目標分支的來源見[訓練索引](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/v2-training-stage-index.json)和[最後權重選擇](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/results/v2-final-training-source-selection.json)。

