# photos-3 讀取與評分紀錄

Grader canonical task_path: `/root/v4_lower_lr_blind_photo_3`。

已先讀 `outputs/natural-v4/review-orchestration/fresh-grader-contract.txt`，完整讀取本 partition 的 `packet.json`（81 個 case/candidate judgement 的 question、model_user、system、history、rubric、reference_example、實際 prediction、decoder_complete 及來源圖欄位）與 `grades-template.json`。每張圖的九項答案均在讀取實際來源圖時逐項判讀。

completion 規則採本次事前固定的「Any actual truncation or unknown completion is incorrect」。本 partition 的 81 項 decoder_complete 皆為 true，無 completion 失敗項；語意判斷依各項預先 rubric 獨立完成。

整包盲評指紋保留為 `3e972679f69cc905299dbb013a62548a981f4fa386a141311891c6e08cf68bc1`。
指派 partition 原 packet SHA256 為 `d5c7ed7697eb9deaf9d60480a47547bdbf2f9f2f1b3971d389fba958428aefb7`。
原 grades-template SHA256 為 `8cbda0dc61cc64e54dfcdc467a37af3ea8c2ee644f6cee6c35e065cc3943869f`。

9 張來源圖片已本人逐張呼叫 view_image，detail=original，觀看完整原始影像各一次；尺寸、檔案大小、來源 SHA256、檢視順序與具體可見觀察記於 actual-image-inspection.json。沒有使用拼圖或文字 caption 替代讀圖。以下是實際視察順序：

1. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/test_02509.jpg` — 1024×768；38925 bytes；SHA256 `044154ebb3f48dff0bc627e02b577a1116edef32d56cba42232500202f7ccb8b`。
2. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/test_04816.jpg` — 768×1024；116089 bytes；SHA256 `9906a3ea032dfc81ea01d9199aaf6e805cea440011cae6196d3b1e30eae72988`。
3. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/test_01214.jpg` — 1024×768；164103 bytes；SHA256 `41c3f06bbb506da39ebea15189ed8ee2f3fa8764f781d53362a82ab26b90bf78`。
4. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/test_03092.jpg` — 1024×768；76864 bytes；SHA256 `404fa0a5126431c35370d9a8a3f7bed73a1dbf76a0f5dd94fd178a6af796de56`。
5. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/test_01604.jpg` — 1024×768；229381 bytes；SHA256 `82a21e37ae38f0e98504bf2b438d7fd20df6960c520e5ccd8f3717a3d0fadde1`。
6. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/test_04238.jpg` — 767×1024；230197 bytes；SHA256 `807e01ab11147242465acc361cb7e710822c3167158ef20cd394e69ec7011229`。
7. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/test_02113.jpg` — 767×1024；111117 bytes；SHA256 `9476c6e5c2da69da998265591ac948223233f226356abb9fc8be7acd573b5a6d`。
8. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/test_00711.jpg` — 1024×768；195654 bytes；SHA256 `b3aa020bafe93aed74c8acebcc03a8130852af0b643d3f109df9d3e1616d20ac`。
9. `/workspace/tiny-perceptron-vlm/outputs/natural-v4/data/vision/images/train_05669.jpg` — 1024×768；95470 bytes；SHA256 `e777958a2186bbb050a99a3ae6c65ce2d392622c6bd0c017ad165a9bead081c2`。

內容讀取範圍限於上述契約、自己的 packet/template 及 packet 明列的 9 張來源照片。未查閱 private mapping、舊評分、訓練或選版紀錄、其它 partition、其它模型 artifact、父 agent 的過往結果；沒有推測 candidate 身份，沒有執行 training、inference、publish 或向外發訊息。

只有本 partition 的 grades.completed.json、actual-image-inspection.json、read-record.md 由本人寫入。81 個 passed 及 source_image_inspected 都是 Boolean，reason 均為本人對該項答案的具體判斷；沒有用預設通過或字串比對替代語意評分。

自查完成：81 項 packet/template/grade 的 (case_id, candidate) 順序與集合一致、無遺漏或重複，原整包 SHA 綁定保留，9 張來源圖 coverage 完整，全部 passed/source_image_inspected 為 Boolean。完成後再次確認原 packet/template 位元組 SHA 未改動。

限制：依照片只能判讀可見場景，不能確證隱藏的年齡、地點、名稱、象徵意義或事件。這是模型獨立語意盲評，沒有進行人類受試研究。
