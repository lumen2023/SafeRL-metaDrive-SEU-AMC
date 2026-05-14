# 风险场文档索引

这个目录集中放置 SafeMetaDrive 风险场相关的分析文档。建议阅读顺序如下：

1. [风险场训练语义与门控风险设计](RISK_FIELD_TRAINING_SEMANTICS.md)
   - 讨论连续风险场长期累积为什么会掩盖离散事件 cost，为什么标准 Safe RL 环境偏向离散事件 cost，以及如何用门控风险场作为辅助训练信号。

2. [当前风险场公式与参数速查](RISK_FIELD_FORMULA_AND_PARAMS.md)
   - 梳理当前 `env.py` 和 `risk_field.py` 中真实生效的风险场公式、参数默认值、量级和调参方向。

3. [风险场 Cost 与 cost_limit 设计说明](RISK_FIELD_COST_LIMIT.md)
   - 解释风险场公式、`env.py` 中的 cost 映射、collector 聚合、cost critic / Lagrange 约束，以及 IDM 标定方法。

4. [风险场版本对比与回退指南](RISK_FIELD_COMPARISON.md)
   - 对比旧版硬编码风险场和新版 MetaDrive-native 风险场，说明公式、几何来源和调参差异。

5. [车道线类型详解与风险惩罚对比](LANE_LINE_TYPES_EXPLAINED.md)
   - 解释 MetaDrive 车道线类型、线型风险权重、可视化语义和交通规则含义。

## 当前推荐结论

- 离散事件 cost 适合作为 Safe RL 的主约束，因为它有清晰物理语义，例如碰撞、越界、撞障碍物。
- 风险场适合作为早期预警、near-miss 度量、可视化调参和辅助约束。
- 不推荐把每一步低风险巡航状态的 dense risk 直接长期累加进同一个 `cost_limit`。
- 如果风险场要参与训练，优先考虑门控风险、平均单步风险、超阈值暴露率或多约束 Lagrange。
- 当前最小改动建议是保留风险场计算和日志，但将主约束切回事件语义，例如设置 `risk_field_cost_combine="event_only"`。
