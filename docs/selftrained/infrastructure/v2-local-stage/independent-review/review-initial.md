独立工程审阅，2026-10-06 UTC。

审阅对象是作者冻结的单文件 local stage subprocess wrapper；没有改作者候选或真实仓库核心文件。仅使用 `/workspace/tiny-perceptron-vlm/.venv/bin/python`、自己生成的 width16 / layers1 / batch16 / context512 CPU 数据与权重。没有 GPU、Modal、paid dispatch、HF 写入、生产 private checkpoint、生产 test 数据、教材修改或 commit。本审阅不构成最终模型训练或质量验证。

初始冻结 wrapper SHA256：`8aee10c5f6a1d38896980346ac69ff5e79eb1b84b2f47360aeaa448f2b337e62`。
初始冻结 patch SHA256：`adb76d378ce34a16b226b7884d2b3ee9dd8df74926a53905533620ffed409e2f`。
冻结副本在 `frozen/`；实际独立执行命令与结果在 `independent-actual-results.json`、各 `*.raw.log` 和 `runs/*` 原始文件。

初始结论：需要修正下列三个实际复现问题后再接受。

1. `local_source` 第53行将 `best.pt` 与 `latest.pt` 都允许作为 resume 输入。独立 `resume-selected-best` 真 subprocess 从自己 SFT 的 `best.pt` 续到 target2，outer exit0 / completed；合同要求 exact resume 只从自己的可信 latest 状态。
2. 第129–131行仅禁止管理选项的精确名字，底层 trainer argparse 可缩写。独立 `undeclared-record-abbreviation` 用 `--record data/undeclared.jsonl` 加入 manifest 外文件，真 subprocess exit0 / completed，实际 training data_sha256 包含该未声明文件。
3. 第139–145行仅验证已声明 assets，没有确认 records 引用的资产都在 manifest 内。独立 `undeclared-assets` 将 manifest.assets 留空，records 仍引用 `image.png` 与 `audio.wav`，真 subprocess exit0 / completed，实际 training asset_sha256 包含两个未声明 assets。manifest SHA 无法冻结这些实际输入。

独立正向结果：自己的随机 pretrain1 → SFT1 通过自己的 selected `best.pt` 串接，均 outer exit0 / completed。镜像当前 Git 元数据和源码后只在 `/tmp` 镜像 train.py 附加注释，`reject-modified-source` prelaunch exit1 且不创建 output；真实仓库没有源码篡改。

作者 raw 证据另经独立只读检查：6个 attempts 的 execution/receipt 身份一致，全部记录的 output 文件 bytes/SHA 验证通过，真实 child command/cwd/解释器正确，当前 tracked core 文件 SHA 一致。native 训练 receipt 真正绑定 weighted4/4 selected best、source execution 和 source receipt 的哈希。新 train metrics rows 总数为11，包含 interrupted 的1步和 resume 的7个新步；不是把 inherited resume step 重新计入。作者的 SIGTERM 例子真实 child exit0、training interrupted=true、saved step1<target8，但 outer failed，fresh promotion 被拒绝。

已有边界：没有内建 wall timer，外部监督器可发 SIGTERM；SIGKILL 等不能完成 outer receipt 的 attempt 保持 running，因此不会成为 fresh init 来源。该结论没有把所有 hard kill 情况称作成功保存。公开 safetensors 未转换为训练 checkpoint；wrapper 文档及 source filename gate 保持其推论用途。

对应只读证据检查 `author-raw-evidence-inspection.json` SHA256：`0cff86612bfc08605e481adc3f61c2d3c290095c3b63598eb4ae51c083598348`。
