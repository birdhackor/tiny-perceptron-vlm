# 原 R.1 审阅者的窄范围 context 复查

实际身份仍是 `/root/phase4_factual_coordinator/factual_r_1`；这是同一审阅者的必要回读，不宣称新的 initial/fresh 审阅轮。

我亲读当前章首导言和完整 R.1，随后读原 frozen 输入与现稿 R.2/R.4 的真实差异，以及现稿 course/README.md 第 34 行、74 行、76 行的必要导航上下文。没有读其他审阅者的报告或判定。

原初次 PASS 报告先按原始 bytes 保存为 initial-pass-R.1.json，SHA-256 为 `147229327e73233da180fbdaa456bf06d0c797341ddb3be7f76a49f853c84e6a`。原 full frozen input 仍是上层 frozen-input/course/README.md，SHA-256 `9629a929f6f15f672f4768f7a20fa17bdd01b4d370d03c7a43c0bee6a21c24c9`，未改写。frozen-current-README.md 是本次窄复查读取时的完整 bytes 快照，SHA-256 `7db6338ee157abb290eee02930d1ea8b57af4eec2ab1e477ce380ce9264e637e`；它没有替代原 full frozen input，正式 source_sha256 仍只对应 R.1。

真正观察：R.1 与导言逐 bytes 相同。原第一节 1.1、W.1/W.2/W.3、1.1 Notebook 原始 bytes、第一节 inventory entry、四条 R.1 链接均未改变。Notebook 生成所需方法/常数与 exporter 同页/页尾的原 AST 也逐项相同。check_context.py 是实际运行的 CPU 字节/文本/AST 检查，stdout、stderr、版本环境和具名 measurements 留存；没有重跑原 fence、模型、训练或远端服务。

我对每项原主张的影响判断：

- `r1-entry-material`：第 34、76 行的改动没有改变第一节入口或字表材料；原 1.1 raw bytes 及四条导航的实际目标未变，原支持范围保持。
- `r1-same-notebook`：第 34 行只是把回读位置说成正文与末尾补充；第 76 行涉及较后面旧合成示范的下载入口。两处都没有改变教学节与同节 Notebook 的生成或页尾文件对应；原方法、常数、Notebook 与第一节 inventory entry 未变。
- `r1-warmup-entries`：R.2 的回读描述与 R.1 已给出的正文暖身链接相容。W.1–W.3 的原文和 anchors 未变，仍是 R.1 所指两种打开方式、清单字典和形状主题。
- `r1-readable-page`：R.4 现在把已公开合成示范的下载/重做指向 19.11，把局部实验操作指向操作页；这不扩大 R.1「先读一节、不先安装」的静态阅读范围，也不把所有动手执行说成无安装。当前 R.4 第 74 行仍区分旧示范与尚待完成的成品。我只判断此段对 R.1 读法范围的影响，未替 R.4 评判模型、实验或 19.11/19.12 全部事实。

导言仍是学习题材的开场，未因这两处导航改动新增已完成模型能力或成绩。R.1 原四项主张及其支持范围保持；没有新发现的实质未定事项，本人维持 PASS。没有对 R.2/R.4 作独立节判定，也没有要求重新派遣原 R.1 reader。
