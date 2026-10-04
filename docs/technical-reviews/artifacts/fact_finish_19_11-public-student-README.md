# capstone_student：教學模型

這一組權重讓你比較兩個一樣小的 Dense 模型：CE 只看標準答案，KD 還看固定教師對 token 的機率分布，再把 KD 做 4-bit 儲存壓縮。**這次蒸餾沒有勝過 CE**：validation 兩者都是 59/84，正式 test 的 CE 是 62/90、KD 是 61/90。

主教材推薦的最終模型仍是 **Joint MoE**。這裡是獨立的壓縮比較分支，教師按事前固定的設計使用 **DPO**，不因 test 結果重新挑教師，也不改配方重跑。它不是推薦 Joint 的蒸餾。DPO 教師 validation 為71/84、test為78/90；validation 圖音聯合題14/18、形狀0/9、風格3/3。

| 版本 | 參數量 | 完成更新次數 | Validation 完整答對／84 | Test 完整答對／90 |
| --- | ---: | ---: | ---: | ---: |
| DPO教師 | 328,128 | 主教材第四站已完成 | 71/84 | 78/90 |
| student-ce | 79,920 | 350 | 59/84 | 62/90 |
| student-kd | 79,920 | 350 | 59/84 | 61/90 |
| student-kd-int4 | 同一KD的量化權重 | 沒有另外重訓 | 未安排獨立validation | 61/90 |

**同一起跑點，比較不同損失。** 學生都是兩層、width48 Dense，沒有直接繼承主MoE權重。兩分支共用初始隨機權重，初始state SHA已由CPU重建核對；抽樣器以同一seed、相同凍結程式設計為相同順序，每個分支有效target token皆為145,163。沒有逐batch獨立紀錄或hash，不把這項配方設計說成已取得每批回放證據。CE是助理答案CE；KD是0.5 CE+0.5 KL(teacher||student)，T=2且含T²。教師不更新。L4上訓練區段CE為10.931293秒、KD為12.063112秒；KD包括教師forward與額外損失，這不是推論速度比較。

**把失敗拆開看。** 正式test計算器題CE只有1/12、KD為0/12；工具結果讀回兩者各1/6，風格各0/3，形狀各0/9。例如1+8，CE產生正確TOOL:calculator:1+8並回答9；KD卻請求1+9並答11，這是兩者答對狀態唯一不同的test題。其餘測試任務分數相同：工具不可用12/12、概念6/6、顏色9/9、圖音聯合18/18、聲音6/6、提供短資料再回答3/3、缺資訊3/3、拒答模板3/3。這些是受控短模板，固定答案標籤可以造成高分，不能當一般VLM、對話、個性或安全能力的證據。

**壓縮總分相同，輸出仍會改變。** KD-int4真正重載後仍為61/90，但逐題核對action與工具後續final生成ID，有6/90題不同；這六題在量化前後都已失敗。例如工具結果讀回題FP32產生DIRECT:11，int4改成DIRECT:10，仍都不符標準答案。KD的FP32 tensor bytes為319,680，int4為91,680；公開清理前推論檔bytes為342,451與106,229，正式下載檔大小以公開收據為準。Linear逐通道對稱打包int4，embedding/norm/bias保留float32；學生沒有MoErouter。載入後解量化成float32，沒有整數推論核心或已測得的加速。

**資料與公開範圍。** capstone-small-world-v2由本專案MIT程式生成，seed42、552train/84validation/90test，與主模型共用同一凍結manifest、家族切分及高低音分層。統一下載清單核對後會共享主模型data.json，RGB圖形與波形按規格重建。公開權重剔除optimizer、RNG、內嵌訓練資料與教師參照state；完整續訓狀態保持私有。學生並未獨立做圖像／音訊交換或生成效率基準，不能借用主模型的通過數字。

程式與安裝：[教材專案](https://github.com/birdhackor/tiny-perceptron-vlm)。正式完整逐題紀錄位於docs/course-experiments/capstone-evidence/student/，主摘要為docs/course-experiments/results/capstone_student.json。公共下載固定HFcommit並核SHA／bytes，再用自訂載入器重建CPU前向；不是Transformers pipeline模型。


已完成的獨立 Dense CE/KD 教學比較：79,920參數，兩層width48、264-ID byte tokenizer、context192、2query heads/1KV head與相同RGB／log-mel投影。CE／KD各350更新，KD-int4由完成KD做PTQ。三個公開ID student-ce/student-kd/student-kd-int4由student_branch區分，不因格式內stage=joint而合併；Dense沒有MoErouter。

限制：

- 一次seed42的小型受控實驗：本次KD test61/90低於CE62/90，不能聲稱蒸餾改善，也不能概括所有蒸餾配方。
- 固定DPO教師不同於主教材推薦Joint；不得把這組當推薦Joint的蒸餾，教師與配方不因test重新選擇。
- 相同抽樣順序來自凍結sampler程式與同seed/RNG設計；只有有效target總數與共同初始state SHA，沒有真實每batchhash可供獨立回放比對。
- 工具請求、結果讀回、風格與形狀仍有明顯失敗；受控split的答案標籤可能固定，成功不能延伸成一般VLM、數學可靠性、人格、安全對齊或自然對話。
- 學生沒有獨立圖像／音訊交換或生成效率benchmark，不能借用主模型診斷結果。
- KD-int4以float32解量化推論；總分相同但有六題生成ID改變。只測儲存壓縮，沒有整數加速或等能力速度結論。
- KD訓練時間包括教師forward／額外KL損失，CE與KD訓練秒數不是教師／學生推論速度對照。
- 公開推論檔剔除optimizer/RNG/參照教師state；不能靠公開推論檔精確恢復中途完整續訓。

訓練程式版本：`43c78c2608be89d6072606be516c233e46abef31`。授權逐檔列於 export-manifest.json。

權重許可及上游來源聲明見 LICENSE 與 THIRD_PARTY_NOTICES.md；程式碼採 MIT。

下載的 checkpoint 已移除 optimizer、RNG 與內嵌的訓練參照模型；自訂模型的 architecture 保留在檔案中。清單若列有教師基準，它是可單獨載入的推論權重。

已審閱的實測資料：

```json
{
  "status": "completed",
  "batch_id": "course-integration-v2",
  "experiment_id": "capstone_student",
  "git_revision": "43c78c2608be89d6072606be516c233e46abef31",
  "run_id": "gha-37170200956-1",
  "device": "NVIDIA L4",
  "seed": 42,
  "recipe": {
    "steps_per_branch": 350,
    "temperature": 2.0,
    "alpha": 0.5,
    "same_initial_state_by_shared_initialization_code": true,
    "initial_state_sha256": "3c1018fd7da00fe36dc2cb3ade40709793330ea636f4d667db493be5c3f7d24d",
    "initial_state_cpu_reconstruction_verified": true,
    "same_sample_sequence_by_frozen_sampler_seed_design": true,
    "per_batch_sequence_hashes_recorded": false,
    "effective_target_tokens_per_branch": 145163,
    "teacher_stage": "dpo",
    "teacher_frozen": true,
    "recipe_frozen_before_test": true,
    "teacher_or_recipe_reselected_after_test": false
  },
  "teacher_checkpoint_sha256": "b2428be8adfdbc60dff589ec8ca9270f3377e1688da863169c75d40326396b1f",
  "teacher_validation_correct": 71,
  "teacher_validation_count": 84,
  "teacher_test_correct": 78,
  "teacher_test_count": 90,
  "student_parameters": 79920,
  "branches": {
    "ce": {
      "steps": 350,
      "requested_steps": 350,
      "schedule_completed": true,
      "seconds": 10.931293079,
      "parameters": {
        "config": {
          "vocab_size": 264,
          "width": 48,
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
        "parameters": 79920,
        "logical_active_parameters": 79920
      },
      "effective_tokens": 145163,
      "teacher_checkpoint_sha256": "b2428be8adfdbc60dff589ec8ca9270f3377e1688da863169c75d40326396b1f",
      "objective": "answer-only CE",
      "initialization": "identical random Dense state, not inherited MoE weights"
    },
    "kd": {
      "steps": 350,
      "requested_steps": 350,
      "schedule_completed": true,
      "seconds": 12.063111922999997,
      "parameters": {
        "config": {
          "vocab_size": 264,
          "width": 48,
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
        "parameters": 79920,
        "logical_active_parameters": 79920
      },
      "effective_tokens": 145163,
      "teacher_checkpoint_sha256": "b2428be8adfdbc60dff589ec8ca9270f3377e1688da863169c75d40326396b1f",
      "objective": "0.5 answer-only CE + 0.5 KL(teacher||student), T=2, includes T²",
      "initialization": "identical random Dense state, not inherited MoE weights"
    }
  },
  "student_ce": {
    "validation": {
      "count": 84,
      "action_correct": 59,
      "end_to_end_correct": 59,
      "by_task": {
        "calculator": {
          "count": 10,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "unavailable": {
          "count": 10,
          "action_correct": 10,
          "end_to_end_correct": 10
        },
        "tool_return": {
          "count": 5,
          "action_correct": 2,
          "end_to_end_correct": 2
        },
        "concept": {
          "count": 5,
          "action_correct": 5,
          "end_to_end_correct": 5
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
          "action_correct": 0,
          "end_to_end_correct": 0
        }
      },
      "protocol": "exact generated action + EOS; requested calculator arguments must match; runtime return is input to a second generation"
    },
    "test": {
      "count": 90,
      "action_correct": 62,
      "end_to_end_correct": 62,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 1,
          "end_to_end_correct": 1
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 1,
          "end_to_end_correct": 1
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
          "action_correct": 0,
          "end_to_end_correct": 0
        }
      },
      "protocol": "exact generated action + EOS; requested calculator arguments must match; runtime return is input to a second generation"
    },
    "deployment_inference_file_bytes": 342451,
    "tensor_bytes": 319680
  },
  "student_kd": {
    "validation": {
      "count": 84,
      "action_correct": 59,
      "end_to_end_correct": 59,
      "by_task": {
        "calculator": {
          "count": 10,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "unavailable": {
          "count": 10,
          "action_correct": 10,
          "end_to_end_correct": 10
        },
        "tool_return": {
          "count": 5,
          "action_correct": 2,
          "end_to_end_correct": 2
        },
        "concept": {
          "count": 5,
          "action_correct": 5,
          "end_to_end_correct": 5
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
          "action_correct": 0,
          "end_to_end_correct": 0
        }
      },
      "protocol": "exact generated action + EOS; requested calculator arguments must match; runtime return is input to a second generation"
    },
    "test": {
      "count": 90,
      "action_correct": 61,
      "end_to_end_correct": 61,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 1,
          "end_to_end_correct": 1
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
          "action_correct": 0,
          "end_to_end_correct": 0
        }
      },
      "protocol": "exact generated action + EOS; requested calculator arguments must match; runtime return is input to a second generation"
    },
    "deployment_inference_file_bytes": 342451,
    "tensor_bytes": 319680
  },
  "student_kd_int4": {
    "validation_status": "not_run_by_recipe",
    "test": {
      "count": 90,
      "action_correct": 61,
      "end_to_end_correct": 61,
      "by_task": {
        "calculator": {
          "count": 12,
          "action_correct": 0,
          "end_to_end_correct": 0
        },
        "unavailable": {
          "count": 12,
          "action_correct": 12,
          "end_to_end_correct": 12
        },
        "tool_return": {
          "count": 6,
          "action_correct": 1,
          "end_to_end_correct": 1
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
          "action_correct": 0,
          "end_to_end_correct": 0
        }
      },
      "protocol": "exact generated action + EOS; requested calculator arguments must match; runtime return is input to a second generation"
    },
    "bits": 4,
    "deployment_inference_file_bytes": 106229,
    "tensor_bytes": 91680,
    "float_tensor_bytes": 319680,
    "compute": "dequantize-to-float32",
    "router_present": false
  },
  "quantization_full_trace_comparison": {
    "count": 90,
    "comparison": "action generated IDs plus final generated IDs after tool result when present",
    "changed_record_count": 6,
    "changed_record_ids": [
      "9f3b6e22324a70ac8c2c",
      "c99474b4859b0c371dec",
      "4bd37b7c74680ad6ffa1",
      "db82339effa74cbb01f9",
      "246ec17075366e4bea5f",
      "9acc36f77f3ae65654a7"
    ],
    "all_changed_records_failed_both_before_and_after": true,
    "identical_total_score_is_not_identical_generation": true
  },
  "ce_kd_correctness_difference": {
    "id": "ffe914bb5e9abc53f39f",
    "expected_action": "TOOL:calculator:1+8",
    "ce_action": "TOOL:calculator:1+8",
    "ce_final": "9",
    "kd_action": "TOOL:calculator:1+9",
    "kd_final": "11",
    "ce_correct": true,
    "kd_correct": false
  },
  "deployment_file_sizes_are_not_final_public_export_sizes": true,
  "data_version": "capstone-small-world-v2",
  "dataset_manifest_sha256": "77fdbe79dfa27236a7cc06c083245276875ac750fe15d4d776096323ef302cd9",
  "counts": {
    "train": 552,
    "validation": 84,
    "test": 90
  },
  "evidence": "docs/course-experiments/capstone-evidence/student/"
}
```

固定版本下載：

- [student-ce/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/f3a407438c5311ccc07f455c7d5fccc944e6ad28/course/course-integration-v2/capstone_student/student-ce/model.pt?download=true)
- [student-kd/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/f3a407438c5311ccc07f455c7d5fccc944e6ad28/course/course-integration-v2/capstone_student/student-kd/model.pt?download=true)
- [student-kd-int4/model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/f3a407438c5311ccc07f455c7d5fccc944e6ad28/course/course-integration-v2/capstone_student/student-kd-int4/model.pt?download=true)

使用設定：

```json
{
  "loader": "scripts/capstone.py; custom repository format",
  "download_manifest": "docs/course-experiments/capstone-public.json",
  "commands": [
    "python scripts/fetch_capstone.py --stage student-ce",
    "python scripts/fetch_capstone.py --stage student-kd",
    "python scripts/fetch_capstone.py --stage student-kd-int4",
    "python scripts/capstone.py infer --checkpoint checkpoints/capstone/student-kd/model.pt --prompt \"1+2等於多少？\" --device cpu"
  ]
}
```

在專案根目錄執行：

```bash
python scripts/fetch_capstone.py --stage student-ce
python scripts/fetch_capstone.py --stage student-kd
python scripts/fetch_capstone.py --stage student-kd-int4
python scripts/capstone.py infer --checkpoint checkpoints/capstone/student-kd/model.pt --prompt "1+2等於多少？" --device cpu
```

各權重的使用設定：

### student-ce/model.pt

```json
{
  "download_stage": "student-ce",
  "format": "capstone-v1",
  "compute": "float32"
}
```

### student-kd/model.pt

```json
{
  "download_stage": "student-kd",
  "format": "capstone-v1",
  "compute": "float32"
}
```

### student-kd-int4/model.pt

```json
{
  "download_stage": "student-kd-int4",
  "format": "capstone-ptq-v1",
  "compute": "float32"
}
```
