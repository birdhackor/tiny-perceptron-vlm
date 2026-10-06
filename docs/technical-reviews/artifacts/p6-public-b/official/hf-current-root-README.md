# Tiny Perceptron：課程教學模型

這些 tiny 模型用來練習從零訓練、下載與推論。各模型的任務、實測結果與限制以其模型卡為準；小型受控任務的結果不能推廣為通用能力或高性能保證。

本頁僅列出課程 `docs/course-experiments/public-models.json` 中已發布的模型，不代表所有課程實驗都已完成。下載連結固定在各次發布的 HF commit；根 README 的後續更新不改變那些版本。

程式碼採 **MIT**。**權重與資料按檔案適用個別授權**；下載前請閱讀各模型的 `LICENSE`、`THIRD_PARTY_NOTICES.md` 與 `export-manifest.json`，不能把程式碼的 MIT 當成所有檔案的權重授權。

| 已發布模型 | 模型卡 | 固定 HF revision | 逐檔授權 |
| --- | --- | --- | --- |
| `simple_models` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models/README.md) | [`fbbff36990db`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/fbbff36990db0d95a6e0af5ecdbc593f920938d8) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models/export-manifest.json) |
| `text_foundation` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/README.md) | [`2ba1278993a6`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/2ba1278993a68d4ab30091534376a955b188df4c) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/export-manifest.json) |
| `real_text` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/c8416bcf4d54cf40fbd270d3f636655a522310a0/course/course-v1/real_text/README.md) | [`c8416bcf4d54`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/c8416bcf4d54cf40fbd270d3f636655a522310a0) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/c8416bcf4d54cf40fbd270d3f636655a522310a0/course/course-v1/real_text/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/c8416bcf4d54cf40fbd270d3f636655a522310a0/course/course-v1/real_text/export-manifest.json) |
| `tokenizer` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/58eb946c0eb9a37b8abb8d1930900685b42790bf/course/course-v1/tokenizer/README.md) | [`58eb946c0eb9`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/58eb946c0eb9a37b8abb8d1930900685b42790bf) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/58eb946c0eb9a37b8abb8d1930900685b42790bf/course/course-v1/tokenizer/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/58eb946c0eb9a37b8abb8d1930900685b42790bf/course/course-v1/tokenizer/export-manifest.json) |
| `sft` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft/README.md) | [`14293af762e7`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/14293af762e79c5a65da5bdc5afed1bea68175a6) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft/export-manifest.json) |
| `sft_ablation` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation/README.md) | [`e30b15712b79`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/e30b15712b798cecacf47b0787a92590d2a7d876) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation/export-manifest.json) |
| `style` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style/README.md) | [`23b58c077a2d`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/23b58c077a2da2ec6b94f4902e88510c0364e3fc) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style/export-manifest.json) |
| `lora` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora/README.md) | [`1bfb0ed028d3`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/1bfb0ed028d3a980711a29cb38a993f58d127d02) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora/export-manifest.json) |
| `safety` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e/course/course-v1/safety/README.md) | [`308aa207d691`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e/course/course-v1/safety/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e/course/course-v1/safety/export-manifest.json) |
| `dpo` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo/README.md) | [`9c3601de9cd4`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo/export-manifest.json) |
| `encoders` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/93e45c77a7f6f21047bcd9f6e7ac3b5a7ad6c294/course/course-v1/encoders/README.md) | [`93e45c77a7f6`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/93e45c77a7f6f21047bcd9f6e7ac3b5a7ad6c294) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/93e45c77a7f6f21047bcd9f6e7ac3b5a7ad6c294/course/course-v1/encoders/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/93e45c77a7f6f21047bcd9f6e7ac3b5a7ad6c294/course/course-v1/encoders/export-manifest.json) |
| `contrastive` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/03cddf7bf0a2cdd69c1df876d3cff4b5e88cd621/course/course-v1/contrastive/README.md) | [`03cddf7bf0a2`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/03cddf7bf0a2cdd69c1df876d3cff4b5e88cd621) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/03cddf7bf0a2cdd69c1df876d3cff4b5e88cd621/course/course-v1/contrastive/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/03cddf7bf0a2cdd69c1df876d3cff4b5e88cd621/course/course-v1/contrastive/export-manifest.json) |
| `projector` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/6d115860fc0af4669fccaa44bcdf2fad7c97e91e/course/course-v1/projector/README.md) | [`6d115860fc0a`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/6d115860fc0af4669fccaa44bcdf2fad7c97e91e) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/6d115860fc0af4669fccaa44bcdf2fad7c97e91e/course/course-v1/projector/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/6d115860fc0af4669fccaa44bcdf2fad7c97e91e/course/course-v1/projector/export-manifest.json) |
| `vqa` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa/README.md) | [`e17db92f5504`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/e17db92f550449d1fc195655dfda1a4a49af9a43) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa/export-manifest.json) |
| `vision_ablation` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf/course/course-v1/vision_ablation/README.md) | [`71b6a22c6934`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf/course/course-v1/vision_ablation/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf/course/course-v1/vision_ablation/export-manifest.json) |
| `ocr` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ce803a4c1eb75c428397fc4a49352354d9c82e8e/course/course-v1/ocr/README.md) | [`ce803a4c1eb7`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/ce803a4c1eb75c428397fc4a49352354d9c82e8e) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ce803a4c1eb75c428397fc4a49352354d9c82e8e/course/course-v1/ocr/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ce803a4c1eb75c428397fc4a49352354d9c82e8e/course/course-v1/ocr/export-manifest.json) |
| `audio` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/1c946fcb7f1d75948711962f22559b18d879e8b9/course/course-v1/audio/README.md) | [`1c946fcb7f1d`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/1c946fcb7f1d75948711962f22559b18d879e8b9) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/1c946fcb7f1d75948711962f22559b18d879e8b9/course/course-v1/audio/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/1c946fcb7f1d75948711962f22559b18d879e8b9/course/course-v1/audio/export-manifest.json) |
| `joint` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/b7908e15691515c15ab93caa9e71327b7ffda87b/course/course-v1/joint/README.md) | [`b7908e156915`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/b7908e15691515c15ab93caa9e71327b7ffda87b) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/b7908e15691515c15ab93caa9e71327b7ffda87b/course/course-v1/joint/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/b7908e15691515c15ab93caa9e71327b7ffda87b/course/course-v1/joint/export-manifest.json) |
| `real_modal` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/0c284d926ee729291fa023976fc03d9260db4068/course/course-v1/real_modal/README.md) | [`0c284d926ee7`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/0c284d926ee729291fa023976fc03d9260db4068) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/0c284d926ee729291fa023976fc03d9260db4068/course/course-v1/real_modal/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/0c284d926ee729291fa023976fc03d9260db4068/course/course-v1/real_modal/export-manifest.json) |
| `modern` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/README.md) | [`59d993473b51`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/59d993473b5155842cea7b40d37be91a7b0bc033) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/export-manifest.json) |
| `moe` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/README.md) | [`17fc5f20686f`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/export-manifest.json) |
| `efficiency` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/README.md) | [`0e1628d51964`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/0e1628d51964fdbd6f6dbf6373ce1324927addba) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/export-manifest.json) |
| `precision` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fb1d740aa140204efd41329a2c28e2a87daae6a4/course/course-v1/precision/README.md) | [`fb1d740aa140`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/fb1d740aa140204efd41329a2c28e2a87daae6a4) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fb1d740aa140204efd41329a2c28e2a87daae6a4/course/course-v1/precision/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fb1d740aa140204efd41329a2c28e2a87daae6a4/course/course-v1/precision/export-manifest.json) |
| `quantization` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/b41d097e4c10b29f6e6218728cbe1bafcc9db93c/course/course-v1/quantization/README.md) | [`b41d097e4c10`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/b41d097e4c10b29f6e6218728cbe1bafcc9db93c) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/b41d097e4c10b29f6e6218728cbe1bafcc9db93c/course/course-v1/quantization/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/b41d097e4c10b29f6e6218728cbe1bafcc9db93c/course/course-v1/quantization/export-manifest.json) |
| `qat` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/README.md) | [`30ea5592953d`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/30ea5592953d0f0db4a0827fe6b7f30bc53805f0) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/export-manifest.json) |
| `distillation` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/README.md) | [`fa69713c7dec`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/fa69713c7dece0177b5088e14f49581d8ae97b7f) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/export-manifest.json) |
| `multimodal_distillation` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/README.md) | [`ae3c4263c4f9`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/export-manifest.json) |
| `rag` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/660fa1d1fa67770b651c8397b1c4f44b11d200e5/course/course-v1/rag/README.md) | [`660fa1d1fa67`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/660fa1d1fa67770b651c8397b1c4f44b11d200e5) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/660fa1d1fa67770b651c8397b1c4f44b11d200e5/course/course-v1/rag/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/660fa1d1fa67770b651c8397b1c4f44b11d200e5/course/course-v1/rag/export-manifest.json) |
| `tools` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/973d02736f4ebbfd7188e7f032fff1dfeddbc66c/course/course-v1/tools/README.md) | [`973d02736f4e`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/973d02736f4ebbfd7188e7f032fff1dfeddbc66c) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/973d02736f4ebbfd7188e7f032fff1dfeddbc66c/course/course-v1/tools/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/973d02736f4ebbfd7188e7f032fff1dfeddbc66c/course/course-v1/tools/export-manifest.json) |
| `reasoning` | [任務與實測限制](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning/README.md) | [`ac5ac599faab`](https://huggingface.co/birdhackor/tiny-perceptron-course-models/commit/ac5ac599faabcb026a159af09433ec0490a397f7) | [LICENSE](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning/LICENSE) / [逐檔清單](https://huggingface.co/birdhackor/tiny-perceptron-course-models/blob/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning/export-manifest.json) |

## 在課程 checkout 下載與推論

使用本課程程式碼與配對的推論腳本。下載器只取選定模型，匿名下載並核對固定 revision、檔案大小與 SHA-256。

```bash
uv sync --extra cpu
uv run --extra cpu python scripts/fetch_course_models.py --list
```

下面列出公開清單中已提供的推論設定；生成輸出仍需核對，不從下載成功推論模型能力。
RAG 的文字推論範例直接提供文件；工具模型的文字推論只產生請求，沒有自行執行計算。檢索、嚴格驗證、工具執行與結果回填仍是外層程式的工作；完整操作請依各模型卡的流程。
OCR 與真實媒體組下方的合成形狀／單音輸入只檢查程式入口，不能用來評估數字辨識或人聲辨識。這兩組請依模型卡準備數字圖片或固定媒體，再使用其 `--image`／`--audio` 範例。

### simple_models

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model simple_models
uv run --extra cpu python scripts/infer_simple.py checkpoints/course/simple_models/mlp3.pt --prompt '顏色=' --tokens 24 --device cpu
```

固定版本權重：

- [bigram.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models/bigram.pt?download=true) · 3,618 bytes · SHA-256 `2a51d84afd66eb2a3d528d2447cfe978e8bbbadd2faf451525f50b900d8bddd6`
- [mlp1.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models/mlp1.pt?download=true) · 6,584 bytes · SHA-256 `8eecee287fbd149e5d1202d16f9e6cbcf6de20f93ac38db2f4cb1075a96836a0`
- [mlp3.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models/mlp3.pt?download=true) · 8,632 bytes · SHA-256 `9de6d06eb4ef9894562e7565d917db913b590cd7663f4510c74c17f4cf5b7332`
- [mlp5.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models/mlp5.pt?download=true) · 10,680 bytes · SHA-256 `b4563451e72b61fdb0cdc91809908ae8157751dd511c66d4a4cc53559d4c772b`

### text_foundation

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model text_foundation
uv run --extra cpu python scripts/infer.py checkpoints/course/text_foundation/model.pt --prompt 'color=blue;shape=circle;' --tokens 32
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/model.pt?download=true) · 575,494 bytes · SHA-256 `98bfedd6cf596a3bb90a21f4acd67fd9c4c2caf67435b2a381b6cfc11ed80834`
- [scaling-w16-n16.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/scaling-w16-n16.pt?download=true) · 62,376 bytes · SHA-256 `e449a2480196ab0cc80a324ec894d7fae24cdf02204169e33c08ffe7ab067889`
- [scaling-w16-n64.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/scaling-w16-n64.pt?download=true) · 62,376 bytes · SHA-256 `67d867a814f4254e28db07ec8c53ef62ea2eb842d992e79a9d01a510dd17ab63`
- [scaling-w32-n16.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/scaling-w32-n16.pt?download=true) · 141,928 bytes · SHA-256 `7a1d7c93a7cc275d2eb789d893103e4cd420058559902ba4feba877aa0cabfad`
- [scaling-w32-n64.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/scaling-w32-n64.pt?download=true) · 141,928 bytes · SHA-256 `5cb931b7857c4204b7634334b284da7322dd230fec48aca93ebdbe107e8f5af4`
- [start.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation/start.pt?download=true) · 575,494 bytes · SHA-256 `90a6a6dd093aafd3cf8588c7db76647aaf50f48fe643d117a9d161e301b191a0`

### real_text

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model real_text
uv run --extra cpu python scripts/infer.py checkpoints/course/real_text/tinystories.pt --prompt 'Once upon a time' --tokens 32 --temperature 0.0 --device cpu
```

固定版本權重：

- [chinese-poetry.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/c8416bcf4d54cf40fbd270d3f636655a522310a0/course/course-v1/real_text/chinese-poetry.pt?download=true) · 577,729 bytes · SHA-256 `66090e65760ee09450c920563c837c568534684902f90cf98d1f6aeacc2b96ae`
- [tinystories.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/c8416bcf4d54cf40fbd270d3f636655a522310a0/course/course-v1/real_text/tinystories.pt?download=true) · 577,560 bytes · SHA-256 `ca6d734aa3fe2eb1f97196982fd057ba6bd14078579d43d4756991621a54b7ed`

### tokenizer

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model tokenizer
uv run --extra cpu python scripts/infer.py checkpoints/course/tokenizer/bpe512.pt --prompt Once --tokens 32 --temperature 0.0 --device cpu --tokenizer checkpoints/course/tokenizer/tokenizer-bpe512.json
```

固定版本權重：

- [bpe512.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/58eb946c0eb9a37b8abb8d1930900685b42790bf/course/course-v1/tokenizer/bpe512.pt?download=true) · 237,081 bytes · SHA-256 `5e1d520fdd97ddc0e1ae15de06e5e596dd19d0f5d5e6d4edc00a0f0e6857d3cc`
- [byte256.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/58eb946c0eb9a37b8abb8d1930900685b42790bf/course/course-v1/tokenizer/byte256.pt?download=true) · 174,128 bytes · SHA-256 `40d7b2e53c55e8b7c36b44e137c5ef40e3e62b75afd6193fb384e8d546a36ee2`

### sft

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model sft
uv run --extra cpu python scripts/infer.py checkpoints/course/sft/model.pt --prompt 'color=blue;shape=circle;pitch=low;shape?' --tokens 32 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft/model.pt?download=true) · 575,622 bytes · SHA-256 `751fe9448d209fc34fcd723ee69aa1f10fe40d8c38c2c689c11decfae8f4e132`
- [pretrain-sft.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft/pretrain-sft.pt?download=true) · 577,723 bytes · SHA-256 `5a157b8b0c767f3b7041e6bd1f2a47826649b270412a7b9ecce5d9443326d6d3`
- [pretrain.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft/pretrain.pt?download=true) · 577,583 bytes · SHA-256 `bb68217d83d197e59dd6cf3a8a2cb4eb73a1dba703c38d9805ed0abc6be62817`
- [ultrachat-pilot.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft/ultrachat-pilot.pt?download=true) · 158,440 bytes · SHA-256 `d2e7e6b9fab8e6d111d446f5253cc85a71e42b84360c42e7dd592373fb0ae888`

### sft_ablation

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model sft_ablation
uv run --extra cpu python scripts/infer.py checkpoints/course/sft_ablation/replay.pt --prompt '1+2=?' --tokens 32 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [b-only.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation/b-only.pt?download=true) · 575,593 bytes · SHA-256 `bd1c2ff521093a50c571dbbceffd876be029803ce0aba7cac92358a9e92adcf5`
- [clean.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation/clean.pt?download=true) · 575,558 bytes · SHA-256 `d947d654d84c5bca5324f2ea5fefa8072ecedb67da77b93f75f8a3c6a9471ea8`
- [noisy.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation/noisy.pt?download=true) · 575,558 bytes · SHA-256 `e00105d154e84078aa69770f925ba55c7930cdb00bfc22fbb96cced5349de28f`
- [replay.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation/replay.pt?download=true) · 575,593 bytes · SHA-256 `e94b7f364917b48249081f9a87f3418b49cfa8cf644bed83af18754c7ecbb39e`

### style

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model style
uv run --extra cpu python scripts/infer.py checkpoints/course/style/model.pt --prompt 'style=json; 1+2=?' --tokens 96 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [content.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style/content.pt?download=true) · 576,908 bytes · SHA-256 `57a49e9bcdfb1e411d9a64c683bae90ad1fba41072bace0f1bd68e0e1f3a35f6`
- [default-concise.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style/default-concise.pt?download=true) · 577,764 bytes · SHA-256 `5ff737a4376c42b675f9198d9f1e0cfb888c412cdf54d203cbd474861c5924a7`
- [default-vivid.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style/default-vivid.pt?download=true) · 577,694 bytes · SHA-256 `7b48a2e8e1e70fb016f00886cdf949d81cc9b6952d530c2a7fbc33851b96e875`
- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style/model.pt?download=true) · 575,558 bytes · SHA-256 `a3481cd6ee5415563f107d6ec07aeecba92f2520ab7d45fdc5373840f8a78607`

### lora

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model lora
uv run --extra cpu python scripts/infer.py checkpoints/course/lora/merged-vivid.pt --prompt '1+2=?' --tokens 96 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [adapter-concise.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora/adapter-concise.pt?download=true) · 46,721 bytes · SHA-256 `a17192b40050ea891063a4e29255fee3c77783800be58b5187662853337703fa`
- [adapter-vivid.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora/adapter-vivid.pt?download=true) · 46,665 bytes · SHA-256 `dd2525010901218db8bd90fa5ab0e863e10424b2bddb8ec13515bc4c49abbfbe`
- [full-sft.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora/full-sft.pt?download=true) · 577,519 bytes · SHA-256 `b9fdce59a0ba96e270709110aedc97b00ee98275db8631e193d2caf7ba3c9406`
- [merged-concise.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora/merged-concise.pt?download=true) · 577,921 bytes · SHA-256 `1e6059dd4a56c5729f906670b24cbc35012267ca9bb79f78d91080d8540551ad`
- [merged-vivid.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora/merged-vivid.pt?download=true) · 577,787 bytes · SHA-256 `30c82b7a8d50c8429b623394eccbc2f36940c1b15403167308e42abbe23cfbe6`

### safety

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model safety
uv run --extra cpu python scripts/infer.py checkpoints/course/safety/model.pt --prompt '盒子0；permission=True；請提供秘密碼。' --tokens 128 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e/course/course-v1/safety/model.pt?download=true) · 575,558 bytes · SHA-256 `0271ddaf1320059110cfaf0e10d0477613f510b741f4299b7976250b36cf09a8`
- [safety-only.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e/course/course-v1/safety/safety-only.pt?download=true) · 577,624 bytes · SHA-256 `719316bb22c664d260321a1a151ec6c9f00cfb5b47689a6897c369c9cea69e5e`

### dpo

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model dpo
uv run --extra cpu python scripts/infer.py checkpoints/course/dpo/model.pt --prompt '1+2=?' --tokens 32 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [beta1.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo/beta1.pt?download=true) · 575,558 bytes · SHA-256 `e25fea870fa479d7acdbad29cdb91d716cf741c754470cbec72793023912d1a2`
- [format-model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo/format-model.pt?download=true) · 577,659 bytes · SHA-256 `a8b10e2de373d50c94e209aa12302c04c29cd25071594009c0de110da0474c2d`
- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo/model.pt?download=true) · 575,558 bytes · SHA-256 `278c537de53c2f991a040c9edd7bc103a91cfd25a501e73a0b6b777881ec7a73`
- [ultrafeedback-pilot.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo/ultrafeedback-pilot.pt?download=true) · 158,468 bytes · SHA-256 `e7a8927fc5d96232f8514f05bca55b5656273d86a63bd34bdd1383eb7baa801a`
- [ultrafeedback-sft.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo/ultrafeedback-sft.pt?download=true) · 158,422 bytes · SHA-256 `242e5753ebf11027e7c1fa2f5cf9dd08df9c055490d1ed8ecb87c33fc8cdf563`

### encoders

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model encoders
```

此項公開清單未指定推論 CLI；請依模型卡的使用設定與 architecture 載入。

固定版本權重：

- [audio.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/93e45c77a7f6f21047bcd9f6e7ac3b5a7ad6c294/course/course-v1/encoders/audio.pt?download=true) · 6,385 bytes · SHA-256 `23e4c805224191e9745c35b338bd745944984c3cae7621930179db7a66025a79`
- [vision.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/93e45c77a7f6f21047bcd9f6e7ac3b5a7ad6c294/course/course-v1/encoders/vision.pt?download=true) · 10,106 bytes · SHA-256 `a94424e267950bd2e253aa00231613f7bbc19eb10f2f6a4a9154ca05a98333ec`

### contrastive

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model contrastive
```

此項公開清單未指定推論 CLI；請依模型卡的使用設定與 architecture 載入。

固定版本權重：

- [one_way.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/03cddf7bf0a2cdd69c1df876d3cff4b5e88cd621/course/course-v1/contrastive/one_way.pt?download=true) · 26,253 bytes · SHA-256 `6ab3643d6a0f041982db2cf3103896d198644f1c11429adbb09e052b854c7691`
- [two_way.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/03cddf7bf0a2cdd69c1df876d3cff4b5e88cd621/course/course-v1/contrastive/two_way.pt?download=true) · 26,253 bytes · SHA-256 `77a8d58118c9395756ba691266df8d57e160398969ac6261d367eb3e9b28c5c8`

### projector

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model projector
uv run --extra cpu python scripts/infer_modal.py checkpoints/course/projector/model.pt --device cpu --tokens 16 --prompt describe --color red --shape square
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/6d115860fc0af4669fccaa44bcdf2fad7c97e91e/course/course-v1/projector/model.pt?download=true) · 597,105 bytes · SHA-256 `09e7bedda6145f190b5df41184ada2834517fa6d8ec01b8593a7615f2039b5ea`

### vqa

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model vqa
uv run --extra cpu python scripts/infer_modal.py checkpoints/course/vqa/all.pt --device cpu --tokens 16 --prompt 'shape?' --color red --shape square
```

固定版本權重：

- [all.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa/all.pt?download=true) · 597,001 bytes · SHA-256 `c623e0195d41224ec44bf99fdb56a070b4dfd3235e44682c8dabba7a41f36b2f`
- [all_replay.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa/all_replay.pt?download=true) · 600,309 bytes · SHA-256 `6a5634a802798201ea2f9b25dc6910980b9e69e50f33a05e845d0462c87be7de`
- [direct_vqa.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa/direct_vqa.pt?download=true) · 600,309 bytes · SHA-256 `bf703a40a3da1d6b508ca3bda31e987cbf4482e85dcdc08899c3d4a8011f1e5e`
- [partial.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa/partial.pt?download=true) · 599,577 bytes · SHA-256 `e8f998aa89c3d94caa5a6ceb9270e4dcd3872e17517fdd548711669533895bd4`
- [projector_only.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa/projector_only.pt?download=true) · 600,581 bytes · SHA-256 `c9ea03a0810a3d17aa16e3d98083f46a20ec8dc996570e7822bf3d20ebca832c`

### vision_ablation

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model vision_ablation
uv run --extra cpu python scripts/infer_modal.py checkpoints/course/vision_ablation/model.pt --device cpu --tokens 16 --prompt 'shape?' --color red --shape square
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf/course/course-v1/vision_ablation/model.pt?download=true) · 597,105 bytes · SHA-256 `e206c6e5956c75de6505899eadde291706969eb76d548b025ca4be36330cee2d`
- [patch-4.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf/course/course-v1/vision_ablation/patch-4.pt?download=true) · 599,577 bytes · SHA-256 `f867ca9aa401ae3efe69aa764dc8f214f9587ac392861d295c30950468b91a91`
- [patch-8.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf/course/course-v1/vision_ablation/patch-8.pt?download=true) · 608,025 bytes · SHA-256 `4eadde6b99d7345d404756a420a042ae0fee3ddc51066b7e5c39ab54083aec3f`

### ocr

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model ocr
uv run --extra cpu python scripts/infer_modal.py checkpoints/course/ocr/model.pt --device cpu --tokens 5 --prompt 'read digits' --color red --shape square
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ce803a4c1eb75c428397fc4a49352354d9c82e8e/course/course-v1/ocr/model.pt?download=true) · 597,105 bytes · SHA-256 `ffbec1ca9affb7c53ea3133464f3b60a84c315cf2e4f8dadb108f8b3876a0dcf`

### audio

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model audio
uv run --extra cpu python scripts/infer_modal.py checkpoints/course/audio/model.pt --device cpu --tokens 6 --prompt 'pitch?' --frequency 220.0
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/1c946fcb7f1d75948711962f22559b18d879e8b9/course/course-v1/audio/model.pt?download=true) · 597,105 bytes · SHA-256 `76c271934541c7005d3b1f450697a1d673cb827bab22fe015fb074a57df8d950`

### joint

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model joint
uv run --extra cpu python scripts/infer_modal.py checkpoints/course/joint/model.pt --device cpu --tokens 16 --prompt 'joint?' --color red --shape square --frequency 220.0
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/b7908e15691515c15ab93caa9e71327b7ffda87b/course/course-v1/joint/model.pt?download=true) · 597,105 bytes · SHA-256 `2d8c41239c0fd08ae8fdf3128dd05effa54ac7f76960bed4fc963eb956b2339a`

### real_modal

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model real_modal
uv run --extra cpu python scripts/infer_modal.py checkpoints/course/real_modal/fsdd.pt --device cpu --tokens 32 --prompt 'digit?' --frequency 220.0
```

固定版本權重：

- [fashion-mnist.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0c284d926ee729291fa023976fc03d9260db4068/course/course-v1/real_modal/fashion-mnist.pt?download=true) · 600,529 bytes · SHA-256 `2c9a795b838fb06bb287ce704570588783d42470a71e51ddf444d3a34da545a8`
- [fsdd.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0c284d926ee729291fa023976fc03d9260db4068/course/course-v1/real_modal/fsdd.pt?download=true) · 609,341 bytes · SHA-256 `a80abb51f06dc935e35f5fcede478474bf4ede11b9e250e16421abedecbb3fb5`

### modern

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model modern
uv run --extra cpu python scripts/infer.py checkpoints/course/modern/baseline.pt --prompt 'Once upon a time' --tokens 32 --temperature 0.0 --device cpu
```

固定版本權重：

- [baseline.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/baseline.pt?download=true) · 577,583 bytes · SHA-256 `29790d1c2ded1863d2dcda66665a8176648055078fb185611380633513fce0d7`
- [relu2.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/relu2.pt?download=true) · 575,622 bytes · SHA-256 `e08dbabca1880f3e1112a06526a69413f9e7dd450a3422abc165e6f8e974d9f4`
- [rmsnorm.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/rmsnorm.pt?download=true) · 574,171 bytes · SHA-256 `6cf49f2b23dd7b5887e4a630a6b96126bbc65402a07db0d8fdcf0fda0266e273`
- [rope.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/rope.pt?download=true) · 542,569 bytes · SHA-256 `d92168652cf55fa9b7335af91e5d2bc84e70e0684b0c9797407bdbbe2335d568`
- [swiglu.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/swiglu.pt?download=true) · 709,785 bytes · SHA-256 `a4bcd1f974a64c7b87264478308344f56e8ef295e1be236272a62b31407a438d`
- [tied.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern/tied.pt?download=true) · 507,881 bytes · SHA-256 `9648792bc322eba668baf7b93c1800f2093451dc8f5c5a698711f6454a50edf5`

### moe

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model moe
uv run --extra cpu python scripts/infer.py checkpoints/course/moe/top2_aux0.01.pt --prompt 'Once upon a time' --tokens 32 --temperature 0.0 --device cpu
```

固定版本權重：

- [dense_active_top1.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/dense_active_top1.pt?download=true) · 577,898 bytes · SHA-256 `7740bb89da376e8b8c335086c72bfe128cada4548bbf00ff77cfa2611c36c292`
- [dense_active_top2.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/dense_active_top2.pt?download=true) · 842,346 bytes · SHA-256 `e20da254f9aed4a803e46c8b368c178aa08ec881fb04c40a595c6993ab93abcb`
- [dense_total.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/dense_total.pt?download=true) · 1,330,584 bytes · SHA-256 `24bd49a51ccf7d4e3d34b91709b13f02f7d178c17e7e9fe9d2b76d56775af2e0`
- [top1_aux0.01.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/top1_aux0.01.pt?download=true) · 1,383,215 bytes · SHA-256 `8a97133cbf9ab9f5d2f75864d98dc86f66214560b2a155548065ada7b44c727b`
- [top1_aux0.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/top1_aux0.pt?download=true) · 1,383,032 bytes · SHA-256 `f3df1cd360f2341f24def64e19d84241362ab14e518201d693b6da08521c5e21`
- [top2_aux0.01.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/top2_aux0.01.pt?download=true) · 1,383,215 bytes · SHA-256 `6ecad573c367f46e6eab4a5a7a08ac7917706d1ee159931d4e8e950163f2f3c7`
- [top2_aux0.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe/top2_aux0.pt?download=true) · 1,383,032 bytes · SHA-256 `42009e9265cf89b5de7de208ec8468960599e4da9005c19af4938b4488cb56d2`

### efficiency

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model efficiency
uv run --extra cpu python scripts/infer.py checkpoints/course/efficiency/ordinary.pt --prompt 'color=blue;shape=circle;pitch=low;shape?' --tokens 32 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [accumulated.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/accumulated.pt?download=true) · 577,688 bytes · SHA-256 `91ec3c9467448201e5df31d7241257c6966cd9dee9483f1736633794c7474eb5`
- [activation_checkpoint.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/activation_checkpoint.pt?download=true) · 578,038 bytes · SHA-256 `e1fafe2efa762585d18f88af06b7318a34bfd2d0e8c431d11425b02bd2fc6517`
- [gqa.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/gqa.pt?download=true) · 526,400 bytes · SHA-256 `f37fc52d5bea892acee79c73d794798396bfa20d2070f5386b05aad21a213a80`
- [mha.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/mha.pt?download=true) · 575,552 bytes · SHA-256 `db869d795edc0fa24fcc04025db29685f151ff0bd5b9a963981e2907d529064b`
- [ordinary.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/ordinary.pt?download=true) · 577,583 bytes · SHA-256 `28e6eb93acc4ced4efb19fc77f620dcc032a08ba753fe64d3cc6e6465245237c`
- [packed.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/packed.pt?download=true) · 575,657 bytes · SHA-256 `630ef043d062a62202bbdafc7098cdc53ad434de128f0c051b96d79e38f0022d`
- [padded.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/padded.pt?download=true) · 575,657 bytes · SHA-256 `bb53820720a43ee534e2fc91fee16645dbc814c02b4607693c0132a94d1c6e0a`
- [sdpa.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency/sdpa.pt?download=true) · 575,587 bytes · SHA-256 `65078eb0fd98d4cdcf29bc89cb308cf17715a855c3fcfc705755172d70f1fc22`

### precision

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model precision
uv run --extra cpu python scripts/infer.py checkpoints/course/precision/fp32.pt --prompt 'color=blue;shape=circle;pitch=low;shape?' --tokens 32 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [bf16.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fb1d740aa140204efd41329a2c28e2a87daae6a4/course/course-v1/precision/bf16.pt?download=true) · 575,523 bytes · SHA-256 `741c0d63882442d547e74bc484d67ce78364a60212a4f567202037472ce4ffad`
- [fp16.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fb1d740aa140204efd41329a2c28e2a87daae6a4/course/course-v1/precision/fp16.pt?download=true) · 575,523 bytes · SHA-256 `20daeb4fbde13a7374ea823346664b37075b5d19a14427615c732c148a09bcce`
- [fp32.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fb1d740aa140204efd41329a2c28e2a87daae6a4/course/course-v1/precision/fp32.pt?download=true) · 575,523 bytes · SHA-256 `d3c6edc501b94d0288387b8792f9d9f0098b4033524495839917743454f42c19`

### quantization

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model quantization
uv run --extra cpu python scripts/infer.py checkpoints/course/quantization/model.pt --prompt 'color=blue;shape=circle;pitch=low;shape?' --tokens 32 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [fp32.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/b41d097e4c10b29f6e6218728cbe1bafcc9db93c/course/course-v1/quantization/fp32.pt?download=true) · 575,587 bytes · SHA-256 `74e8bab39917102d92fd988470dde4281da2a3e647a115b4e05e5edee6f058c1`
- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/b41d097e4c10b29f6e6218728cbe1bafcc9db93c/course/course-v1/quantization/model.pt?download=true) · 181,381 bytes · SHA-256 `17e7f0e824b8bebb8396c9384a1887463b30575f337ac88e1072f30177da41cb`
- [packed8.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/b41d097e4c10b29f6e6218728cbe1bafcc9db93c/course/course-v1/quantization/packed8.pt?download=true) · 241,189 bytes · SHA-256 `27b487ce26e4da4ec7f406269a59d5cbf551064b94d32ab95ef1b10eb25bf0a0`

### qat

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model qat
uv run --extra cpu python scripts/infer.py checkpoints/course/qat/model.pt --prompt 'color=blue;shape=circle;pitch=low;shape?' --tokens 32 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [fp_finetuned.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/fp_finetuned.pt?download=true) · 577,723 bytes · SHA-256 `becc1eb7247cfb867140f56cb8ad711e247127967f5ca745c1ce170a0a1ac708`
- [initial_fp32.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/initial_fp32.pt?download=true) · 577,787 bytes · SHA-256 `3f7103f8b8df0cfe0712866964a1e9a6116b271000f6803068c7fc2043e086bc`
- [initial_ptq4.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/initial_ptq4.pt?download=true) · 184,341 bytes · SHA-256 `17d9db0bbd66eced55800a2b338a0a2ad8967844feb3477824b4218affa26a27`
- [matched_ptq4.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/matched_ptq4.pt?download=true) · 184,341 bytes · SHA-256 `0de4b8d25cf9f3e5caa2b5d342523ac3928cdf449262ff3ade7f4c3f27b34008`
- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/model.pt?download=true) · 181,381 bytes · SHA-256 `d4753c4d944bf2629f8e589a76f2bc809c996c0002628238f9f9153a0dcdeedf`
- [qat_float.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat/qat_float.pt?download=true) · 577,618 bytes · SHA-256 `af2ab595d987dc404fa30795d518843e824a2533e90841ae11b379f187e056a0`

### distillation

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model distillation
uv run --extra cpu python scripts/infer.py checkpoints/course/distillation/sft-w32-ce_kl.pt --prompt 'color=blue;shape=circle;pitch=low;shape?' --tokens 32 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [moe-teacher.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/moe-teacher.pt?download=true) · 1,383,218 bytes · SHA-256 `a4efa3a172910542052b10c8b71412e090d22625fe0d1d8ee19531ac69812b5a`
- [moe-w32-ce.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/moe-w32-ce.pt?download=true) · 141,941 bytes · SHA-256 `a58d0a915e15574bbb1f88def20c5931f54a01564be4fc8ee18bf9a0e3d6ea5a`
- [moe-w32-ce_kl-packed4.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/moe-w32-ce_kl-packed4.pt?download=true) · 74,111 bytes · SHA-256 `91155d5e00478420bd478cddd0800d210def584c01c43ce09aeb7e45c654f2b4`
- [moe-w32-ce_kl.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/moe-w32-ce_kl.pt?download=true) · 142,010 bytes · SHA-256 `11d2251b976ed91bdbe9c0da415eeabf8d34bb4a940e722b3ff9997364d9bb2e`
- [sft-teacher.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-teacher.pt?download=true) · 577,752 bytes · SHA-256 `0a0d86ee5ca383a7c1fdf185eaf2c23d6af90fe3c6fa22dec437967ac246f028`
- [sft-w16-ce.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-w16-ce.pt?download=true) · 62,389 bytes · SHA-256 `f2e8b39bc982c4847e3ab022ffd83bedfac5237a63aa53f074a6747cf8aafe21`
- [sft-w16-ce_kl-packed4.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-w16-ce_kl-packed4.pt?download=true) · 41,023 bytes · SHA-256 `998573ab716e6c873875c06bbf296738d91136fdcc81be184a763dfa5a234a90`
- [sft-w16-ce_kl.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-w16-ce_kl.pt?download=true) · 62,458 bytes · SHA-256 `2013439c43a02f1d970ed538f5001e590afbc61c6163bde37608c64712e785e2`
- [sft-w16-teacher_hard.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-w16-teacher_hard.pt?download=true) · 62,619 bytes · SHA-256 `63118322858e2cf1ec563e8c064059c2210dfd8f976b3a8ee0cc7df6ba578eb4`
- [sft-w32-ce.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-w32-ce.pt?download=true) · 141,941 bytes · SHA-256 `a727a99bbc0bee44744ba0deb387490b8921fb6def8f88184a4d39113b25e888`
- [sft-w32-ce_kl-packed4.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-w32-ce_kl-packed4.pt?download=true) · 74,111 bytes · SHA-256 `e5a2bde7122991cfab7c7f28f59793695e8f5ce5cd12bb662d2ecea527bde65b`
- [sft-w32-ce_kl.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-w32-ce_kl.pt?download=true) · 142,010 bytes · SHA-256 `addc52576558ccead03851d86e748873bdb52c9b13fd7b30f34a082c2d453f0e`
- [sft-w32-teacher_hard.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/sft-w32-teacher_hard.pt?download=true) · 142,171 bytes · SHA-256 `cbca0d92444150baf75820dba01d5e64e5a11d5459b8387466fdcce78b9c2fcb`
- [style-teacher.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/style-teacher.pt?download=true) · 577,822 bytes · SHA-256 `1ac6baf3ea200aa407946e29d372b677eacdc510e8490cfd52e52fe3a1e7bfa0`
- [style-w32-ce.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/style-w32-ce.pt?download=true) · 141,987 bytes · SHA-256 `ba5dcf0f6e06a0be495366a7e959fb672e5a60c2679f377eb58b7ad7e9f0e270`
- [style-w32-ce_kl-packed4.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/style-w32-ce_kl-packed4.pt?download=true) · 74,235 bytes · SHA-256 `24345ebe6b5b1e3a227a1edab2fa8038dc1a3f3830abf3aa5b2bde61664db3a9`
- [style-w32-ce_kl.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/style-w32-ce_kl.pt?download=true) · 142,056 bytes · SHA-256 `ca85ea2a6dcfe2ae20fb045122263af581c0f26d0b4e260723a43d6807404f48`
- [style-w32-teacher_hard.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation/style-w32-teacher_hard.pt?download=true) · 142,281 bytes · SHA-256 `f07f6ad0c139d98c84376391f16417c63e5137fcdd033b63a70a631a49f77736`

### multimodal_distillation

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model multimodal_distillation
uv run --extra cpu python scripts/infer_modal.py checkpoints/course/multimodal_distillation/joint-ce_kl.pt --device cpu --tokens 16 --prompt 'joint?' --color red --shape square --frequency 220.0
```

固定版本權重：

- [joint-ce.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/joint-ce.pt?download=true) · 157,605 bytes · SHA-256 `3969e8912358d070384f56c436a47d19fcf21e0a11322c4b79755de738a4dc41`
- [joint-ce_kl.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/joint-ce_kl.pt?download=true) · 157,725 bytes · SHA-256 `b8a04ef072275087e3e01b4201986f409c83d97afd2033b5bcbc4da74ed96761`
- [joint-teacher.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/joint-teacher.pt?download=true) · 600,657 bytes · SHA-256 `a0165c8108fb895e0987c87465ff0f4cb845693c039f411567dfccdfd816640f`
- [vqa-ce.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/vqa-ce.pt?download=true) · 155,861 bytes · SHA-256 `5e3e84d83704e469a22ffdbd212f61eca7c3efb8f81e08c3b13c342356f73e63`
- [vqa-ce_kl.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/vqa-ce_kl.pt?download=true) · 157,645 bytes · SHA-256 `5b60922fe03249a9f643c4ba89140f9108841ec9a52084a8fbefb58471f8e153`
- [vqa-teacher.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation/vqa-teacher.pt?download=true) · 600,553 bytes · SHA-256 `7f4f3797ab18cd2e7b0790f7bf3cd4eddd4f9039be6de8c7e8771681264474bd`

### rag

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model rag
uv run --extra cpu python scripts/infer.py checkpoints/course/rag/model.pt --prompt 'Docs:
[D0] K000 address=A1
Find:K000
Reply:address[source] or UNKNOWN' --tokens 64 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/660fa1d1fa67770b651c8397b1c4f44b11d200e5/course/course-v1/rag/model.pt?download=true) · 583,814 bytes · SHA-256 `37f4a570e878a85949d7670d5da38148779811dcc4cc5bb3f2df0e5fb7e5b194`

### tools

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model tools
uv run --extra cpu python scripts/infer.py checkpoints/course/tools/model.pt --prompt 'CALC:add(2,3)' --tokens 64 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [model.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/973d02736f4ebbfd7188e7f032fff1dfeddbc66c/course/course-v1/tools/model.pt?download=true) · 641,158 bytes · SHA-256 `c8789df280e4a4cba06994ee94f1e96c7199b30b9806dc005dd9c32d32d4bdbf`

### reasoning

```bash
uv run --extra cpu python scripts/fetch_course_models.py --model reasoning
uv run --extra cpu python scripts/infer.py checkpoints/course/reasoning/steps.pt --prompt '(1+2)+3=?' --tokens 48 --chat --temperature 0.0 --device cpu
```

固定版本權重：

- [direct.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning/direct.pt?download=true) · 575,657 bytes · SHA-256 `2724ea905a92871e2d8216a57408191de6437b4bd78fb37f6b134615c371101a`
- [gsm8k-pilot.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning/gsm8k-pilot.pt?download=true) · 141,964 bytes · SHA-256 `9070fabfa6cc62e2e1cd923fc148a6eb7c11556b5b8c9fccf93bfcd92e4f8ada`
- [policy-strict.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning/policy-strict.pt?download=true) · 10,521 bytes · SHA-256 `01eaa3895041a8c17001a700089b6a476ab1af8aed19f66e9763f8dae09cbd94`
- [policy-weak_proxy.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning/policy-weak_proxy.pt?download=true) · 10,561 bytes · SHA-256 `3247f10643a85f9c121ecad47240f80833317d859efa34d207cae6b89842d650`
- [steps.pt](https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning/steps.pt?download=true) · 575,622 bytes · SHA-256 `55f2c3b54a76c4a2b65feb24c8e0ee4b36ba3415f00999fba201e75ec81d91c3`
