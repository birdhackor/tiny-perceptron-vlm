# capstone_deployment：教學模型

這是一個 328,128 參數的小世界教學模型：用短句、RGB 幾何圖形、高低音與受限計算器，讓你比較預訓練、監督式微調、多模態整合、偏好訓練及量化。第一次使用請選 **joint**。我們在正式 test 前依 validation 選定它；DPO 留作比較分支與另一個固定蒸餾實驗的教師。

| 版本 | 完成更新次數 | Validation 完整答對／84 | 正式 Test 完整答對／90 |
| --- | ---: | ---: | ---: |
| pretrain | 300 | 0/84 | 0/90 |
| sft | 1,400 | 42/84 | 45/90 |
| joint | 600 | 75/84 | 78/90 |
| dpo | 100 | 71/84 | 78/90 |
| joint-int4／joint-int8 | 完成的 Joint 做 PTQ | 未安排獨立 validation | 各 78/90 |
| dpo-int4／dpo-int8 | 完成的 DPO 做 PTQ | 未安排獨立 validation | 各 78/90 |

未訓練基準為 0/90。預訓練的 0/90 只指本教材助理動作與回答協定的測試；這次沒有獨立評估一般文字續寫品質，也不能據此說沒有學到序列規律。完整答對要求動作、EOS、計算器參數與工具讀回後的第二次回答都符合教材標準，不能只看 loss。

**最後一站不是自動最好。** Joint 在 validation 的圖音聯合題是 18/18，DPO 是 14/18；風格都維持 3/3，形狀都為 0/9。DPO 是 beta 0.1 的 DPO 加 0.2 倍助理答案 CE 重播，參照為固定的上一站 Joint，另有 valid-token router balance；偏好資料是合成的，沒有真人偏好評分、獎勵模型或 PPO。正式 test 的 Joint 與 DPO 都為 78/90，沒有改變先前的選擇。

**請看失敗，不只看總分。** Joint 的工具請求 12/12 正確，完整答案卻只有 10/12：計算器正確回傳 1 後，0+1 被模型讀回成 0，1+0 被讀回成 111；另有一題工具結果讀回失敗（5/6）。正式形狀題仍為 0/9。換圖後題目 36/36 正確，但原圖和換圖都對只有 27/36，形狀配對仍為 0/9；Joint 的圖音聯合換音題為 18/18。固定答案標籤可能使窄任務分數很高，這些受控結果不代表一般圖片、音訊或聊天能力。

**量化是儲存壓縮。** Linear 權重採逐輸出通道對稱 int4／int8 打包，MoE router、embedding、norm 與 bias 保持 float32；載入後解量化成 float32 計算，沒有整數推論核心。FP32 的 tensor bytes 為 1,312,512，int4 為 248,864，int8 為 402,720。部署時的 Joint FP32／int4／int8 檔案分別為 1,345,023／275,381／429,365 bytes；DPO 分別為 1,341,513／275,381／429,193 bytes。這是公開清理前的推論檔，正式下載 bytes 與 SHA 以公開收據為準。四個量化版都是 78/90，但 Joint-int4 有一題、DPO-int4 有兩題失敗輸出的生成 ID 改變；兩個 int8 在本次 90 題則與各自 FP32 的生成 ID 相同。

**快取與 MoE 的實測範圍很小。** 十二類 validation 各取第一題，KV cache 與完整重算 ID 相同，logits 在 atol=rtol=1e-4 內一致。固定第一筆 validation 風格題照抄 15，產生 DIRECT:15 加 EOS 共十個新 token；L4 上完整重算／快取各暖身三次、測十次，中位數為 0.06805／0.05998 秒（本次約 1.135 倍），沒有更新權重，也沒量生成記憶體峰值。Forward+backward 的 MoE／隨機 Dense80 同 batch 中位數為 0.02146／0.01134 秒；此實作的 MoE 約慢 1.89 倍。CUDA allocator 相對各自開始時新增的峰值為 80,174,592／38,589,952 bytes，不是完整 GPU 用量。Dense80 只有 189,520 參數且沒有訓練能力對照，不能據此聲稱 Dense 與 MoE 能力相等，或 MoE 必然省硬體。

**公開內容與資料。** 合成資料版本 capstone-small-world-v2、seed 42，共 552 train／84 validation／90 test；按家族切分，1+2 與 2+1 留在 test，低高音各 split 都有覆蓋。data.json 的 RGB／波形規格可由公開程式重建。公開 checkpoint 只保留推論設定、tokenizer、架構、權重與必要來源，剔除 optimizer、RNG、參照模型及內嵌訓練資料；續訓完整狀態留在維護者私有備份。

程式及安裝說明：[tiny-perceptron-vlm](https://github.com/birdhackor/tiny-perceptron-vlm)。逐題結果、反事實 trace 與原始速度資料隨教材保存在 docs/course-experiments/capstone-evidence/deployment/；主摘要在 docs/course-experiments/results/capstone_deployment.json。下載使用專案自訂載入器，固定 HF commit 並核對每檔 SHA／bytes；不是 Transformers pipeline 模型。


一次 seed 42 的受控小世界教學實驗。兩層、width64、4experts/top2 MoE，328,128 總參數、195,776 邏輯啟用參數；後者不是速度、FLOPs 或記憶體保證。264-ID byte tokenizer，context192，16×16 RGB 圖像投影與真實波形的16-band log-mel投影，共用一個文字核心。

限制：

- 只在短模板、0–9 加法工具、RGB 幾何形狀及高低音等受控任務上訓練；沒有一般 VLM、自然對話、人格或可靠自我校準的證據。
- Validation 與 test 的窄任務可能有常量答案標籤；圖像、音訊反事實與多數標籤基準須連同完整失敗一起看。形狀原圖失敗仍為0/9。
- 模型會產生工具請求，但真正加法由外部受限計算器做；工具回傳後第二次生成仍會錯。提供文字再回答沒有外部檢索索引。
- DPO 是合成偏好加監督式重播，不是 PPO／真人 RLHF；本次 validation 能力退步，沒有安全性或風格改善證據。
- SDPA 不保證每次使用 Flash Attention，沒有 Flash Attention 的獨立速度比較。
- PTQ 載入後以 float32 計算；儲存變小不代表整數推論、更快或執行時記憶體等比例減少。
- 單一短 validation prompt 的生成速度及隨機 Dense80 機制對照不是一般 benchmark，也不是等能力的 Dense/MoE 訓練配方比較。
- 公開檔只有推論狀態；無法從公開推論檔精確接續維護者 optimizer／RNG 的中途狀態。

訓練程式版本：`6ffc653199a71ad83afcee24aa8a2388122771bd`。授權逐檔列於 export-manifest.json。

權重許可及上游來源聲明見 LICENSE 與 THIRD_PARTY_NOTICES.md；程式碼採 MIT。

下載的 checkpoint 已移除 optimizer、RNG 與內嵌的訓練參照模型；自訂模型的 architecture 保留在檔案中。清單若列有教師基準，它是可單獨載入的推論權重。

已審閱的實測資料：

```json
{
  "status": "completed",
  "batch_id": "course-integration-v2",
  "experiment_id": "capstone_deployment",
  "git_revision": "6ffc653199a71ad83afcee24aa8a2388122771bd",
  "run_id": "gha-37169529991-1",
  "device": "NVIDIA L4",
  "seed": 42,
  "data_version": "capstone-small-world-v2",
  "dataset_manifest_sha256": "77fdbe79dfa27236a7cc06c083245276875ac750fe15d4d776096323ef302cd9",
  "counts": {
    "train": 552,
    "validation": 84,
    "test": 90
  },
  "recommended_stage": "joint",
  "recommendation_selected_before_official_test": true,
  "validation_correct": {
    "pretrain": 0,
    "sft": 42,
    "joint": 75,
    "dpo": 71
  },
  "test": {
    "untrained": {
      "count": 90,
      "action_correct": 0,
      "end_to_end_correct": 0,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "unavailable": {
          "count": 12,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "tool_return": {
          "count": 6,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "concept": {
          "count": 6,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "image_color": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "audio": {
          "count": 6,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "rag": {
          "count": 3,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "missing": {
          "count": 3,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "safety": {
          "count": 3,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "style": {
          "count": 3,
          "action_correct": 0,
          "end_to_end_correct": 0
        }
      }
    },
    "pretrain": {
      "count": 90,
      "action_correct": 0,
      "end_to_end_correct": 0,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "unavailable": {
          "count": 12,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "tool_return": {
          "count": 6,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "concept": {
          "count": 6,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "image_color": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "audio": {
          "count": 6,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "rag": {
          "count": 3,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "missing": {
          "count": 3,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "safety": {
          "count": 3,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "style": {
          "count": 3,
          "action_correct": 0,
          "end_to_end_correct": 0
        }
      }
    },
    "sft": {
      "count": 90,
      "action_correct": 47,
      "end_to_end_correct": 45,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 10
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 5,
          "end_to_end_correct": 5
        },
        "concept": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "image_color": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "audio": {
          "count": 6,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "rag": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "missing": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "safety": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "style": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        }
      }
    },
    "joint": {
      "count": 90,
      "action_correct": 80,
      "end_to_end_correct": 78,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 10
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 5,
          "end_to_end_correct": 5
        },
        "concept": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "image_color": {
          "count": 9,
          "action_correct": 9,
          "end_to_end_correct": 9
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 18,
          "end_to_end_correct": 18
        },
        "audio": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "rag": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "missing": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "safety": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "style": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        }
      }
    },
    "dpo": {
      "count": 90,
      "action_correct": 80,
      "end_to_end_correct": 78,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 10
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 5,
          "end_to_end_correct": 5
        },
        "concept": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "image_color": {
          "count": 9,
          "action_correct": 9,
          "end_to_end_correct": 9
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 18,
          "end_to_end_correct": 18
        },
        "audio": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "rag": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "missing": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "safety": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "style": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        }
      }
    },
    "joint-int4": {
      "count": 90,
      "action_correct": 80,
      "end_to_end_correct": 78,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 10
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 5,
          "end_to_end_correct": 5
        },
        "concept": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "image_color": {
          "count": 9,
          "action_correct": 9,
          "end_to_end_correct": 9
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 18,
          "end_to_end_correct": 18
        },
        "audio": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "rag": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "missing": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "safety": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "style": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        }
      }
    },
    "joint-int8": {
      "count": 90,
      "action_correct": 80,
      "end_to_end_correct": 78,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 10
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 5,
          "end_to_end_correct": 5
        },
        "concept": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "image_color": {
          "count": 9,
          "action_correct": 9,
          "end_to_end_correct": 9
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 18,
          "end_to_end_correct": 18
        },
        "audio": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "rag": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "missing": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "safety": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "style": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        }
      }
    },
    "dpo-int4": {
      "count": 90,
      "action_correct": 80,
      "end_to_end_correct": 78,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 10
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 5,
          "end_to_end_correct": 5
        },
        "concept": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "image_color": {
          "count": 9,
          "action_correct": 9,
          "end_to_end_correct": 9
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 18,
          "end_to_end_correct": 18
        },
        "audio": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "rag": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "missing": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "safety": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "style": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        }
      }
    },
    "dpo-int8": {
      "count": 90,
      "action_correct": 80,
      "end_to_end_correct": 78,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 10
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 5,
          "end_to_end_correct": 5
        },
        "concept": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "image_color": {
          "count": 9,
          "action_correct": 9,
          "end_to_end_correct": 9
        },
        "image_shape": {
          "count": 9,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "joint": {
          "count": 18,
          "action_correct": 18,
          "end_to_end_correct": 18
        },
        "audio": {
          "count": 6,
          "action_correct": 6,
          "end_to_end_correct": 6
        },
        "rag": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "missing": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "safety": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        },
        "style": {
          "count": 3,
          "action_correct": 3,
          "end_to_end_correct": 3
        }
      }
    }
  },
  "quantization": {
    "joint-int4": {
      "bits": 4,
      "file_bytes": 275381,
      "source_file_bytes": 1345023,
      "tensor_bytes": 248864,
      "float_tensor_bytes": 1312512,
      "router_dtype": "float32",
      "compute": "dequantize-to-float32"
    },
    "joint-int8": {
      "bits": 8,
      "file_bytes": 429365,
      "source_file_bytes": 1345023,
      "tensor_bytes": 402720,
      "float_tensor_bytes": 1312512,
      "router_dtype": "float32",
      "compute": "dequantize-to-float32"
    },
    "dpo-int4": {
      "bits": 4,
      "file_bytes": 275381,
      "source_file_bytes": 1345023,
      "tensor_bytes": 248864,
      "float_tensor_bytes": 1312512,
      "router_dtype": "float32",
      "compute": "dequantize-to-float32"
    },
    "dpo-int8": {
      "bits": 8,
      "file_bytes": 429193,
      "source_file_bytes": 1345023,
      "tensor_bytes": 402720,
      "float_tensor_bytes": 1312512,
      "router_dtype": "float32",
      "compute": "dequantize-to-float32"
    }
  },
  "deployment_file_sizes_are_not_final_public_export_sizes": true,
  "joint_image_swap": {
    "correct": 36,
    "count": 36,
    "both_original_and_swapped_correct": 27,
    "pair_count": 36,
    "shape_pair_correct": 0,
    "shape_pair_count": 9
  },
  "joint_audio_swap": {
    "correct": 18,
    "count": 18
  },
  "dpo_audio_swap": {
    "correct": 18,
    "count": 18
  },
  "cache_consistency": {
    "count": 12,
    "selection": "first validation row for each task, sorted by task name; selected without generation results",
    "atol": 0.0001,
    "rtol": 0.0001,
    "max_new_tokens": 16,
    "all_generated_ids_equal": true,
    "all_logits_close": true,
    "diagnostic_stage": "joint",
    "checkpoint_sha256": "8f7e85820bd70bf1f16e7ef789b11f2d10873cf26a2688d651e23136bd7c3b47"
  },
  "generation_benchmark": {
    "scope": "first frozen validation style row; FP32 Joint/L4; warm3 measured10; max_new_tokens16; CUDA sync; no weight updates",
    "row_id": "691c9de656c01fa2df60",
    "generated_tokens_including_eos": 10,
    "eos": true,
    "all_generated_ids_equal": true,
    "full_median_seconds": 0.06805368149999858,
    "cache_median_seconds": 0.05997750799999935,
    "memory_peak_measured": false
  },
  "mechanism_benchmark": {
    "moe": {
      "parameters": {
        "config": {
          "vocab_size": 264,
          "width": 64,
          "layers": 2,
          "heads": 2,
          "max_length": 192,
          "norm": "rms",
          "activation": "gelu",
          "rotary": true,
          "tied": false,
          "kv_heads": 1,
          "backend": "sdpa",
          "experts": 4,
          "top_k": 2
        },
        "parameters": 328128,
        "logical_active_parameters": 195776
      },
      "warmup_iterations": 3,
      "measured_iterations": 10,
      "median_seconds": 0.021464479999998787,
      "optimizer_updates": 0,
      "weights_unchanged": true,
      "cuda_peak_extra_bytes": 80174592
    },
    "dense80": {
      "parameters": {
        "config": {
          "vocab_size": 264,
          "width": 80,
          "layers": 2,
          "heads": 2,
          "max_length": 192,
          "norm": "rms",
          "activation": "gelu",
          "rotary": true,
          "tied": false,
          "kv_heads": 1,
          "backend": "sdpa",
          "experts": 0,
          "top_k": 2
        },
        "parameters": 189520,
        "logical_active_parameters": 189520
      },
      "warmup_iterations": 3,
      "measured_iterations": 10,
      "median_seconds": 0.011338393500000876,
      "optimizer_updates": 0,
      "weights_unchanged": true,
      "cuda_peak_extra_bytes": 38589952
    }
  },
  "evidence": "docs/course-experiments/capstone-evidence/deployment/"
}
```

固定版本下載：

- [pretrain/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/pretrain/model.pt?download=true)
- [sft/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/sft/model.pt?download=true)
- [joint/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/joint/model.pt?download=true)
- [dpo/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/dpo/model.pt?download=true)
- [joint-int4/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/joint-int4/model.pt?download=true)
- [joint-int8/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/joint-int8/model.pt?download=true)
- [dpo-int4/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/dpo-int4/model.pt?download=true)
- [dpo-int8/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/dpo-int8/model.pt?download=true)
- [data.json](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/bf6902a50e8ffd2d4c5a9428d0df40b593834860/course/course-integration-v2/capstone_deployment/data.json?download=true)

使用設定：

```json
{
  "loader": "scripts/capstone.py; fp32 load_capstone / packed load_quantized_capstone",
  "download_manifest": "docs/course-experiments/capstone-public.json",
  "recommended_stage": "joint",
  "commands": [
    "python scripts/fetch_capstone.py --list",
    "python scripts/fetch_capstone.py --stage joint",
    "python scripts/capstone.py infer --checkpoint checkpoints/capstone/joint/model.pt --prompt \"1+2等於多少？\" --device cpu"
  ]
}
```

在專案根目錄執行：

```bash
python scripts/fetch_capstone.py --list
python scripts/fetch_capstone.py --stage joint
python scripts/capstone.py infer --checkpoint checkpoints/capstone/joint/model.pt --prompt "1+2等於多少？" --device cpu
```

各權重的使用設定：

### pretrain/model.pt

```json
{
  "download_stage": "pretrain",
  "format": "capstone-v1",
  "compute": "float32"
}
```

### sft/model.pt

```json
{
  "download_stage": "sft",
  "format": "capstone-v1",
  "compute": "float32"
}
```

### joint/model.pt

```json
{
  "download_stage": "joint",
  "format": "capstone-v1",
  "compute": "float32"
}
```

### dpo/model.pt

```json
{
  "download_stage": "dpo",
  "format": "capstone-v1",
  "compute": "float32"
}
```

### joint-int4/model.pt

```json
{
  "download_stage": "joint-int4",
  "format": "capstone-ptq-v1",
  "compute": "float32 (packed weights dequantized at load)"
}
```

### joint-int8/model.pt

```json
{
  "download_stage": "joint-int8",
  "format": "capstone-ptq-v1",
  "compute": "float32 (packed weights dequantized at load)"
}
```

### dpo-int4/model.pt

```json
{
  "download_stage": "dpo-int4",
  "format": "capstone-ptq-v1",
  "compute": "float32 (packed weights dequantized at load)"
}
```

### dpo-int8/model.pt

```json
{
  "download_stage": "dpo-int8",
  "format": "capstone-ptq-v1",
  "compute": "float32 (packed weights dequantized at load)"
}
```
