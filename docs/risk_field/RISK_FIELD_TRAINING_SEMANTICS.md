# 风险场训练语义与门控风险设计

本文档讨论一个比公式本身更重要的问题：

> 连续风险场是否应该直接加入 `info["cost"]`，和离散事件 cost 使用同一个 `cost_limit`？

当前结论：

- 风险场有意义，但不应该无条件、长时域地逐步累加到主 `info["cost"]`。
- 离散事件 cost 应继续作为 Safe RL 的主约束，因为它有明确物理语义。
- 风险场更适合作为预警、near-miss、辅助约束、可视化调参和多约束安全信号。
- 如果风险场要进入训练，应优先使用门控、归一化、阈值超额、episode max 或独立约束，避免对 episode 长度敏感。

## 当前核心问题

标准事件 cost 是稀疏的：

```text
正常驾驶 step: cost = 0
碰撞车辆:      cost = 1
撞障碍物:      cost = 1
偏离道路:      cost = 1
```

如果一个 episode 中发生 3 次不安全事件，那么 episode cost 大约就是 3。此时：

```text
cost_limit = 10
```

可以被直观解释为：平均每个 episode 最多允许约 10 个安全违规事件。

加入 dense risk field 后，正常驾驶也可能每一步都有小 cost：

```text
靠近车道线:      cost = 0.005 ~ 0.05
靠近周车:        cost = 0.02 ~ 0.2
跟车略近:        cost = 0.01 ~ 0.3
TTC 略低:        cost = 0.05 ~ 0.5
```

这些单步值看起来都不大，但如果 episode 长度为 `H=1000`：

```text
平均每步 dense cost = 0.03
episode dense cost = 30
```

于是一个没有碰撞、没有越界、只是长期处在轻微风险中的 episode，可能比发生几次离散事件的 episode cost 还大。这就是“连续风险长时域累积掩盖离散事件 cost”的根源。

## 为什么这会破坏 cost_limit 的物理意义

在当前 FSRL 训练链路中：

```text
FastCollector:
    stats_train["cost"] = sum_t info["cost"] / episode_count

LagrangianPolicy:
    lagrangian.step(stats_train["cost"], cost_limit)
```

也就是说，`cost_limit` 的单位是：

```text
平均单个 episode 的累计 cost
```

如果 `info["cost"]` 是离散事件：

```text
cost_limit = 10
```

代表“每个 episode 平均最多 10 次违规事件”。

如果 `info["cost"]` 混入 dense risk：

```text
cost_limit = 10
```

可能代表：

- 10 次碰撞。
- 1000 步里每步 0.01 的风险暴露。
- 100 步里每步 0.1 的 near-miss 暴露。
- 1 次事件 + 大量轻微风险。

这些语义混在一起后，`cost_limit` 不再是一个清晰的安全阈值，而变成了“某种风险积分预算”。这不是不能用，但必须重新命名、重新标定，并接受它不再等价于事件次数。

## 添加风险场还有意义吗？

有意义，但要明确它的角色。

### 风险场的价值

风险场能解决离散事件 cost 的几个问题：

- **提前预警**：离散事件只有碰撞后才有 cost，风险场可以在危险接近时给信号。
- **near-miss 度量**：没有碰撞但贴得很近、TTC 很低的行为，事件 cost 看不到，风险场能看到。
- **可视化调参**：topdown overlay 可以直接显示 agent 为什么被认为危险。
- **安全泛化**：在不同交通密度、不同道路几何下，风险场能提供连续的安全梯度。
- **辅助训练**：在早期训练中，它能比稀疏事件 cost 更快告诉 agent 哪些区域危险。

### 风险场的风险

如果直接把风险场逐步加进主 cost，也会带来问题：

- **长时域偏置**：episode 越长，正常小风险累积越多。
- **事件语义被稀释**：一次真实碰撞可能被大量轻微风险淹没。
- **过度保守**：agent 可能学会低速、远离一切、频繁停滞。
- **benchmark 不可比**：和标准 MetaDrive / Safety Gym 的 cost_limit 不能直接比较。
- **Lagrange 误调节**：乘子根据 episode 总 cost 更新，dense cost 偏大时会长期压制 reward 学习。

所以正确问题不是“要不要风险场”，而是：

```text
风险场应该以什么语义进入训练？
```

## 为什么标准 Safe RL 环境多用离散 cost

标准 MetaDrive、Safety Gymnasium / Safety Gym 一类 benchmark 通常把 cost 设计为事件、接触、越界、碰撞或 constraint violation，而不是完整 dense potential field。主要原因如下。

### 1. cost_limit 必须有清晰物理含义

安全强化学习通常建模为 CMDP：

```text
maximize     expected return
subject to   expected cumulative cost <= cost_limit
```

如果 cost 是碰撞、越界、撞障碍物：

```text
cost_limit = 25
```

可以解释为一个 episode 或一批 episode 允许的违规预算。

如果 cost 是 dense potential：

```text
cost_limit = 25
```

含义就依赖：

- episode 长度。
- 时间步长。
- 风险场尺度。
- sigma。
- 交通密度。
- 正常车道保持时是否也有背景风险。

这会让 benchmark 难以公平比较。

### 2. 离散事件 cost 更稳定

离散事件 cost 的范围通常是：

```text
0 or 1
```

这对 cost critic 和 Lagrange 更新更稳定。dense risk 可能因为参数调得过宽、交通密度变大、episode 变长而整体漂移。

### 3. 奖励和约束需要分工

标准环境一般让 reward 处理“效率和任务完成”，让 cost 处理“明确安全违规”。

如果把车道中心、速度、舒适性、距离风险全部塞进 cost，cost 会变成另一个 reward shaping 项，安全约束和行为偏好会混在一起。

### 4. benchmark 需要可复现和可解释

离散事件 cost 方便统计：

- crash rate。
- out-of-road rate。
- collision count。
- safety violation count。

这些指标人能直接理解。dense risk 的绝对数值则必须绑定具体公式和参数。

## 风险场如何高效加入训练

下面按推荐程度排序。

## 方案 A：事件 cost 做主约束，风险场只做日志和可视化

这是最稳妥、最接近标准 Safe RL benchmark 的设计。

```text
info["cost"] = event_cost
info["risk_field_cost"] = raw continuous risk
info["risk_field_event_equivalent_cost"] = mapped dense risk
```

训练时：

```text
Lagrange 只约束 info["cost"]
风险场只上传到 wandb / progress.txt
```

优点：

- `cost_limit` 保持物理意义。
- 可以和标准 MetaDrive / Safety Gym 结果比较。
- 风险场仍然能用于分析、筛选、可视化和论文图。

缺点：

- 风险场不直接影响策略更新。
- 早期训练仍可能依赖稀疏事件信号，学习安全行为较慢。

当前代码中最小改动：

```python
"risk_field_cost_combine": "event_only"
```

或者：

```python
"use_risk_field_cost": False
```

区别：

- `event_only` 可以继续保留风险场 info。
- `use_risk_field_cost=False` 会跳过风险场计算，日志中没有真实风险分解。

因此如果还想观察风险场，推荐 `event_only` 而不是完全关闭。

## 方案 B：门控风险，只统计 near-miss 区域

不要累计所有小风险，只累计超过阈值的部分：

```text
gated_risk = max(risk_event_equivalent_cost - tau, 0)
```

或：

```text
gated_risk = 1[risk_event_equivalent_cost > tau]
```

例如：

```text
tau = 0.2
```

含义：

- 单步风险低于 0.2 的正常轻微风险不进入主 cost。
- 只有接近周车、TTC 低、压线严重、越界边缘等 near-miss 情况才记 cost。

优点：

- 保留风险场的提前预警能力。
- 大幅降低长时域小风险累积。
- `cost_limit` 可以解释为“near-miss 暴露预算”。

缺点：

- 需要选择阈值 `tau`。
- 不同风险组件可能需要不同阈值。

推荐形式：

```text
event_cost = discrete violation cost
risk_excess = max(risk_event_equivalent_cost - tau, 0)
info["cost"] = max(event_cost, risk_excess)
```

更保守的形式：

```text
info["cost"] = event_cost + lambda_risk * risk_excess
```

但如果使用相加，需要重新标定 `cost_limit`。

## 方案 C：按时间归一化，使用平均单步风险

如果希望风险场表达“平均暴露水平”，就不要用 episode sum，而是用：

```text
episode_risk_mean = sum_t risk_t / episode_length
```

或：

```text
exposure_ratio = sum_t 1[risk_t > tau] / episode_length
```

这样对 episode 长度不敏感。

优点：

- 不会因为 horizon 变长而线性变大。
- `cost_limit` 可以解释为“平均风险强度”或“风险暴露比例”。

缺点：

- 当前 FSRL collector 的 `stats_train["cost"]` 默认是 episode sum，不是 step mean。
- 如果要让 Lagrange 直接约束 step mean，需要修改 collector 或额外提供 risk-specific constraint。

当前已有日志中已经有：

```text
risk_field_cost_step_mean
risk_field_event_equivalent_cost_step_mean
```

但默认 PPO/SAC 的 Lagrange 仍然使用 `stats_train["cost"]`，不会自动用这些 step mean。

## 方案 D：episode max 风险，而不是 sum 风险

如果关心“最危险时刻”，可以用：

```text
episode_risk_max = max_t risk_t
```

含义：

- 一个 episode 中只要出现过一次高风险 near-miss，就会被标记。
- 长时间低风险巡航不会无限累积。

优点：

- 对 episode 长度不敏感。
- 很适合做 near-miss safety metric。

缺点：

- 对风险持续时间不敏感。
- 需要 collector 支持 max 聚合，而不是 sum 聚合。

## 方案 E：多约束 Lagrange，事件和风险分开约束

不要把事件 cost 和风险场 cost 混成一个标量，而是使用两个约束：

```text
cost_vector = [
    event_cost,
    risk_exposure_cost
]

cost_limit = [
    event_limit,
    risk_limit
]
```

例如：

```text
event_limit = 5
risk_limit = 0.05  # 平均单步风险或暴露率
```

优点：

- 事件语义不被风险场淹没。
- 风险场也能真正参与策略约束。
- 分析更清楚：一个 Lagrange 乘子管碰撞，一个管 near-miss。

缺点：

- 当前训练链路主要按单标量 cost 使用，需要扩展 collector、policy 和 logger 的多约束支持。
- 调参复杂度更高。

## 方案 F：风险场用于 reward shaping 或 auxiliary loss

风险场不进入 Safe RL 的 `cost_limit`，而是作为辅助训练信号：

```text
reward = task_reward - alpha * gated_risk
cost = event_cost
```

或作为模型辅助任务：

```text
critic / encoder 同时预测 risk_field_cost
```

优点：

- `cost_limit` 保持事件语义。
- 风险场仍可影响策略偏好。

缺点：

- 如果放进 reward，就不再是严格 safety constraint。
- reward shaping 可能改变任务目标，需要谨慎写论文表述。

## 推荐实现路线

### 第一阶段：恢复主约束物理语义

目标：

```text
离散事件 cost = 主安全约束
风险场 = 日志 + 可视化 + 标定观察
```

推荐配置：

```python
"use_risk_field_cost": True,
"risk_field_cost_combine": "event_only",
"risk_field_event_cost_weight": 1.0,
```

这样：

- `RiskFieldCalculator` 仍然运行。
- `info["risk_field_*"]` 仍然可记录。
- `info["cost"]` 回到事件语义。
- `cost_limit` 可继续解释为事件预算。

训练时：

```bash
python train_sacl.py --task SafeMetaDrive --cost_limit 10
```

这时 `10` 又有相对清晰的事件语义。

### 第二阶段：加入 near-miss 门控风险

在 `env.py` 中增加一种新模式：

```python
"risk_field_cost_combine": "event_plus_gated_risk"
"risk_field_gate_threshold": 0.2
"risk_field_gate_weight": 0.2
```

单步：

```text
risk_excess = max(risk_event_equivalent_cost - gate_threshold, 0)
final_cost = event_cost + gate_weight * risk_excess
```

或者：

```text
final_cost = max(event_cost, risk_excess)
```

推荐先用 `max`，再尝试加权相加。

### 第三阶段：如果需要，做多约束

扩展为：

```text
cost_0 = event_cost
cost_1 = risk_exposure_step_mean or risk_excess_sum
```

对应：

```text
cost_limit_0 = event budget
cost_limit_1 = risk exposure budget
```

这更适合论文中强调“事件安全 + 风险暴露”的方法，但实现量也最大。

## 为什么不推荐无门控 dense risk 直接进 cost

无门控 dense risk 的形式是：

```text
final_cost = max(event_cost, risk_event_equivalent_cost)
episode_cost = sum_t final_cost
```

它的问题：

- `episode_cost` 强依赖 horizon。
- 正常巡航也会积累成本。
- 轻微长时风险可能超过短时严重事件。
- `cost_limit` 需要随地图长度、traffic density、episode horizon 重新解释。
- 和标准 benchmark 的 event cost 不可直接比较。

这并不是说它完全不能用。它可以被定义成：

```text
episode integrated risk exposure
```

但这时需要明确告诉读者：

- `cost_limit` 不是碰撞次数。
- 它是风险暴露积分。
- 标定必须基于 IDM 或专家策略。
- 不同 horizon 下的结果不能直接比较。

## 高效加入风险场的推荐配置矩阵

| 目标 | 推荐 cost 设计 | 是否改 `cost_limit` | 适用阶段 |
| --- | --- | --- | --- |
| 和标准 Safe RL 对齐 | `event_only` | 用事件预算，例如 10/15 | baseline |
| 保留风险场分析 | `event_only` + 记录 `risk_field_*` | 不因风险场改变 | 可视化和日志 |
| 惩罚 near-miss | `event + gated_risk` | 需要重新标定 | 方法改进 |
| 对 horizon 不敏感 | `risk_step_mean` 或 `exposure_ratio` | 使用均值/比例阈值 | 进阶实现 |
| 事件和风险都约束 | 多约束 Lagrange | 分别设置两个 limit | 论文方法 |
| 只想提高安全偏好 | reward shaping | `cost_limit` 不变 | ablation |

## 训练日志应该如何看

如果采用事件主约束，重点看：

```text
train/cost
train/event_cost
train/success_rate
train/safe_success_rate
train/crash_free_rate
train/no_out_of_road_rate
```

如果采用风险场辅助，额外看：

```text
train/risk_field_event_equivalent_cost
train/risk_field_vehicle_cost
train/risk_field_headway_cost
train/risk_field_ttc_cost
train/risk_field_boundary_cost
train/risk_field_lane_cost
```

如果后续加入 step mean 或 gated risk，应重点看：

```text
train/risk_field_event_equivalent_cost_step_mean
train/risk_field_excess_cost
train/risk_field_exposure_ratio
```

理想状态不是风险场完全为 0，而是：

- 事件 cost 接近 0。
- near-miss 风险下降。
- success rate 不明显崩溃。
- average speed 不被过度压低。
- Lagrange multiplier 不长期爆炸。

## 当前建议

基于现在的问题，我建议短期这样做：

1. 主训练先不要把全量 dense risk 直接混入 `info["cost"]`。
2. 保留风险场计算和 wandb 日志。
3. 将主约束切回事件语义，保证 `cost_limit` 可解释。
4. 用风险场做分析：看 agent 在没有碰撞时是否长期贴车、压线、TTC 低。
5. 下一步再实现 `event_plus_gated_risk`，只让 near-miss 风险进入约束。

这条路线最稳：既不浪费已经做好的风险场，也不会让长时域 dense cost 把安全强化学习的核心约束语义冲掉。
