最终独立工程结论：接受本次单文件 wrapper 的最小修复，限定于本机可信 checkpoint 串接与执行证据；没有剩余阻塞发现。

最终审阅文件：`frozen-final/train_local_stage.py`，SHA256 `675e9f63f264365d59a5104811d6899354b412f46b876b8128ed63e40dff2561`，255行。作者初始版本的三个实质问题和原始负例均保留在 `review-initial.md`、`frozen/`、`independent-actual-results.json`、`runs/` 及旧 raw logs；没有改写旧失败结果。

最终修复仅涉及新 wrapper：wrapper 与 trainer 解析对象均禁止 argparse 缩写；fresh init 只接受自己的可信 `best.pt`，exact resume 只接受自己的可信 `latest.pt`；启动 child 前用现有 read_records / asset_fingerprints 确认实际引用的图像和音频均被 manifest assets 的精确 SHA 覆盖。没有改动原训练、模型、evaluate 或 chat 文件。

独立实际验证共9个真 CLI subprocess，完整命令、退出状态和 raw log SHA 保存于 `final-independent-actual-results.json`，SHA256 `357a0d59fdb379f7ec8c7b9fdbba6ed923e7a5fc1d42455b275e353b337d330e`。

自己的 width16 / layers1 / batch16 / context512 / CPU 合成数据完成随机 pretrain1 → SFT1 → SFT latest exact resume 到target2，共3个新的 optimizer steps。三个正例 outer/child exit0，execution completed，training interrupted=false / completed_requested_steps=true；各同目录 execution/receipt 字段及所有记录的 raw output bytes/SHA 均核验通过。SFT 与 resume 的 parent 字段绑定自己的 checkpoint hash、run_id 和 source receipt SHA，全部使用新的 attempt 目录。

六个负例在 child 启动前拒绝，均没有创建新 output：从自己的 best.pt resume；`--record` 加入未声明 records；manifest 漏列引用 assets；外部 manifest SHA 错误；当前 Git tracked source 的 `/tmp` 镜像篡改；自己的 latest.pt 附加字节篡改。前三个负例逐一闭合初始审阅的发现。

真实 native 前置合同位于 `scripts/selftrained/train.py` 与 `scripts/selftrained/modal_runner.py`；不存在被误提及的 `tiny_perceptron/selftrained/training.py`。读取 guard 确认它要求同目录 completed execution/receipt、4/4的有效job参数、selected best 与 safetensors/export/train receipt 哈希及身份一致。作者初版三个 joint subprocess（1/1→4/4→4/1/native4）、真实 SIGTERM 后 child exit0 但 outer failed、fresh promotion 拒绝和 own latest resume 的 raw artifacts 已只读独立核验，见 `author-raw-evidence-inspection.json`；该核验确认为11个实际新增 train metrics rows，且 native receipt 真绑定 weighted source best、execution和receipt的哈希。

范围与限制：本次准备性检查仅用独立生成的 CPU 合成数据、自己的合成权重，以及作者 CPU synthetic 原始证据；没有完整重跑 vision/ocr/audio/final 模型，也没有质量、生产数据谱系或 GPU 性能声明。没有 GPU、Modal、paid dispatch、HF write、生产 private checkpoint、生产 test 数据、教材变更或 commit。公开 safetensors 保持推论用途，未转换到训练状态；source 路径及文档明确限制可信 own checkpoint。

wrapper 保存实际 child command、有效 parser job、真实 cwd/解释器、当前 Git revision/tracked source hashes、wrapper hash、manifest hash、run_id、parent pin、完整 raw runner.log 和全部输出哈希。它没有内建 wall timer；外部 SIGTERM 可以请求现有 trainer 在 step boundary 保存。未完成/中断的运行不能 fresh promote。不能闭合 execution/receipt 的 hard kill attempt 保持 running，后续 fresh init 会拒绝；本审阅没有把所有 hard kill 情况描述为成功保存。

最终作者 patch SHA256 `e7b6e66ebdb948340beac9a378db63d7f1fdfc3b19e51619083bb1943d09a00d`、proof SHA256 `d48f8caa05c6b611d2170acda0070ac6f54e6a7c27776aa417f1902ad9c71c82`，以及最终六个 joint/raw attempts 的文件哈希和 native 来源绑定已核验，见 `final-author-closure.json` 与 `final-author-raw-evidence-inspection.json`。作者另七个真实 CLI 拒绝日志的 SHA 和无 output 状态均核验通过。操作文档 SHA `cc27b1a3d12fbc840b47e65876192396716f4fb6877c82cd166e63a442a6ab92` 的 manifest/job/selected index 数值与本机入口合同一致；安装与下载指令没有重跑。Root 后续已提交并从真实 repo 文件路径、无 --repo-root 参数执行自己合成 pretrain1，原始 execution 确认 revision `0e79975a`、script SHA `675e9f63...` 且新的 committed wrapper 被 source hashes 记录，所有输出哈希核验通过；本 reviewer 没有重跑该 smoke 或提交 commit。
