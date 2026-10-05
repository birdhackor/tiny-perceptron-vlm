本轮独立事实审阅 initial findings，reviewer `/root/phase4_factual_coordinator/factual_c_4`。

本节原始 UTF-8 slice SHA-256：89e972e9cfad8ff829d9365e87281e25eba46a967d5ec2aea69c7ba76f97b030。

唯一实质问题位于 C.4 补充最后一段：

> 精確驗證器能直接計算這個窄任務的真值，並選第一份全部檢查通過的候選，所以它的選擇成績等於「集合至少有一份正解」的比例（oracle coverage）。

现稿 C.3 和原 `_reasoning_samples` 的 oracle coverage 用候选的 `final_correct`；原 `_verify_reasoning` 的步骤模式还要求格式完整、两式算术正确、前后相连及题目操作数一致，才能 `fully_verified`。因此当前条件不足以推出一般等同性。

有界反例只替换手工候选，无生成或训练：题目 `(2+2)+0=?` 的候选 `2+2=3;3+0=4;answer=4`，最终答案4正确，集合 coverage=True，但中间式错，原生产验算器拒绝，不能选中，返回 None。recompute.py 通过 AST 实际执行原验算函数，并用独立解析器交叉核对，stdout 的 equality_counterexample 和 variants 保存此区别。

已有八格 raw candidates 全部重算及原方法离线 replay 都与原数值一致。其中被覆盖集合都已有至少一份完整通过的候选，所以在本批数据中 verifier accuracy 确实等于 oracle coverage。k=8 步骤组为14/24、12/14、12/24，两例与平票也吻合。问题是把本批巧合写成规则推出的必然性。

建议只限定该句为本次候选记录的结果，并写出原因：本批每个最终答案命中的集合，也至少含一份完整验算通过候选；若集合只有最终对而步骤错的候选，严格验算器可能拒绝。也可另定义完整通过的 coverage，但不要暗中改 C.3 的最终答案判准。

图 rewrite-C-candidate-choice.svg 是具体人工清单 [3,3,4] 的两路比较；图中的直接真值4与两路箭头、投票3、验算4均与原 fence 完全一致。已真实 Inkscape 渲染并查看，图不提出步骤候选的一般等同性，不需要修改。

原算例、原已有数据与图均保留有教学价值，问题不要求重新训练、追加模型生成或工程修改。
