# 风险场 Cost 与 cost_limit 设计说明

本文档用于解释当前项目中风险场 cost 的完整链路，以及在加入 dense risk field 后，安全强化学习里的 `cost_limit` 应该如何重新设计。

核心结论：

- `risk_field.py` 负责计算原始连续风险场，不直接决定训练里的最终 `info["cost"]`。
- `env.py` 会把原始风险场压缩到“碰撞等价”的单步 cost，默认单步风险场 cost 不超过 `1.0`。
- `FastCollector` 上传给 PPO-Lag / SAC-Lag 的 `train/cost` 是“平均单个 episode 的累计 cost”。
- `cost_limit` 的单位也是“单个 episode 的累计 cost”，不是单步 cost，也不是原始风险场数值。
- 如果继续把 dense risk 混入主 `info["cost"]`，必须用 IDM 跑一批 episodes，取最终 episode total cost 的 `p90 * 1.1` 作为 `cost_limit` 起点。
- 更推荐的训练语义见 [风险场训练语义与门控风险设计](RISK_FIELD_TRAINING_SEMANTICS.md)：事件 cost 做主约束，风险场做日志、门控辅助约束或 near-miss 指标。

## 代码位置

当前风险场和训练约束链路由以下文件组成：

- `risk_field.py`: `RiskFieldCalculator`，从 MetaDrive 的路网、车道线、周车和障碍物读取状态，计算原始连续风险 `risk_field_cost`。
- `env.py`: `SafeMetaDriveEnv_mini.cost_function()`，调用 `RiskFieldCalculator.calculate()`，再把原始风险映射成碰撞等价 cost，并写入 `info["cost"]`。
- `fsrl/fsrl/data/fast_collector.py`: `FastCollector.collect()`，按 episode 聚合 `info["cost"]` 和风险场分解指标。
- `fsrl/fsrl/policy/lagrangian_base.py`: `LagrangianPolicy.pre_update_fn()`，用 collector 返回的 `stats_train["cost"]` 和 `cost_limit` 更新 Lagrange 乘子。
- `fsrl/fsrl/policy/ppo_lag.py`: PPO-Lag 的 reward critic / cost critic 和 actor safety loss。
- `fsrl/fsrl/policy/sac_lag.py`: SAC-Lag 的 reward Q critic / cost Q critic 和 actor safety loss。
- `calibrate_cost_limit_idm.py`: 用官方 IDMPolicy 批量跑 episodes，并推荐 `cost_limit`。

## 风险场原始公式

`RiskFieldCalculator.calculate(env, vehicle)` 会先分别计算道路、车辆、障碍物、headway、TTC 等组件，再加权求和得到原始连续风险：

```text
risk_raw =
    w_boundary * boundary_cost
  + w_lane     * lane_cost
  + w_offroad  * offroad_cost
  + w_vehicle  * vehicle_cost
  + w_object   * object_cost
  + w_headway  * headway_cost
  + w_ttc      * ttc_cost
```

然后裁剪：

```text
risk_field_cost = clip_nonnegative(risk_raw, risk_field_raw_clip)
```

默认 `env.py` 中 `risk_field_raw_clip = 10.0`。

## 相关论文中的风险场设计

自动驾驶风险场的共同思想是：把道路、车辆、障碍物、行人、车道边界、驾驶行为等交通元素转化为连续空间中的风险分布，然后用该分布做风险评估、预警、轨迹规划或强化学习约束。

### Driver's Risk Field: 概率场乘以后果代价

Kolekar 等人在 Nature Communications 发表的 Driver's Risk Field (DRF) 模型中，将风险理解为：

```text
perceived_risk = sum_over_grid(probability_field(x, y) * consequence_cost(x, y))
```

其中：

- `probability_field` 表示驾驶员认为车辆未来出现在某位置的概率。
- `consequence_cost` 表示该位置或物体的危险后果。
- 驾驶员控制目标不是严格最小化风险，而是让风险保持在某个阈值以下。

这和我们当前设计的关系：

- 我们没有显式建模“驾驶员未来位置概率”，但车辆、道路和障碍物势场也在表达空间上的危险分布。
- 我们把连续风险压缩成 `risk_field_event_equivalent_cost`，再作为 Safe RL 的约束信号，本质上也是“风险低于阈值”的控制思想。
- 我们的 `cost_limit` 就相当于 episode 级别的风险阈值。

参考：Kolekar et al., 2020, [Human-like driving behaviour emerges from a risk-based driver model](https://www.nature.com/articles/s41467-020-18353-4)。

### Probabilistic Driving Risk Field: 碰撞概率乘以碰撞后果

Mullakkal-Babu 等人在 Transportation Research Part C 的 PDRF 模型中，将风险建模为：

```text
risk = collision_probability * expected_crash_energy
```

这个模型的特点：

- 静态物体用 potential field 表示。
- 周围车辆用 kinematic field 表示。
- 车辆未来运动不确定性由纵向和横向加速度分布建模。
- 风险可做单步评估，也可沿多个未来时间步累积。

这和我们当前设计的关系：

- 我们当前 `vehicle_cost` 使用静态超高斯 + 动态速度势场，和“静态 potential + 动态 kinematic field”的结构一致。
- 我们目前没有显式乘上碰撞能量或车辆质量，因此车辆风险主要表达“接近与相对速度风险”，不是完整的“概率 × 损伤严重度”。
- 如果后续要进一步贴近论文，可以给大车、重车或高速相对速度加入 severity 权重。

参考：Mullakkal-Babu et al., 2020, [Probabilistic field approach for motorway driving risk assessment](https://www.sciencedirect.com/science/article/pii/S0968090X20306318)。

### Driving Safety Field: 势场、动能场、行为场叠加

Driving Safety Field (DSF) 系列工作通常把总安全场拆成几个层次：

```text
safety_field = potential_field + kinetic_field + behavior_field
```

常见解释：

- `potential_field`: 静态道路元素，例如车道线、道路边界、路侧物体。
- `kinetic_field`: 动态交通参与者，例如车辆速度、相对运动、加速度。
- `behavior_field`: 驾驶员行为、意图、转向、加减速等因素。

这和我们当前设计的关系：

- `boundary_cost`、`lane_cost`、`offroad_cost` 接近 potential field。
- `vehicle_cost` 的动态项、`headway_cost`、`ttc_cost` 接近 kinetic field。
- 我们暂时没有显式的 driver behavior field，但 PPO/SAC agent 的策略行为会通过 cost critic 间接学习“什么行为导致未来风险更高”。

参考：Wang et al., 2016, [Driving safety field theory modeling and its application in pre-collision warning system](https://www.sciencedirect.com/science/article/pii/S0968090X16301887)。

### Modified Driving Risk Field: 道路场、静态车辆场、动态车辆场

面向换道规划的 MDRF 模型通常会把风险拆成：

```text
modified_driving_risk_field = road_risk_field + static_vehicle_risk_field + dynamic_vehicle_risk_field
```

其中道路场又常拆成：

```text
road_risk_field = roadside_risk_field + lane_line_or_centerline_risk_field
```

车辆静态风险常使用高阶二维高斯或超高斯：

```text
U_static = A_static * exp(
    -((x - x_j)^2 / sigma_x^2)^beta
    -((y - y_j)^2 / sigma_y^2)^beta
)
```

车辆动态风险常额外引入相对速度和方向的非对称项：

```text
U_dynamic =
A_dynamic * exp(
    -(x - x_j)^2 / sigma_v^2
    -(y - y_j)^2 / sigma_y^2
)
/ (1 + exp(-rel_v * (x - x_j - alpha * L_j * rel_v)))
```

这和我们当前实现几乎是一一对应的：

- 我们的 `lane_cost` / `boundary_cost` 对应 road risk field。
- 我们的 `vehicle_potential_components()["static"]` 对应 static vehicle risk field。
- 我们的 `vehicle_potential_components()["dynamic"]` 对应 dynamic vehicle risk field。
- 我们使用 MetaDrive 真实车道线语义和周车信息，而不是写死高速路 `y=-2,2,6,10,14` 这类固定坐标。

参考：Zheng et al., 2024, [A Dynamic Lane-Changing Trajectory Planning Algorithm for Intelligent Connected Vehicles Based on Modified Driving Risk Field Model](https://www.mdpi.com/2076-0825/13/10/380)。

### Generalized Risk Field: 多车冲突用重叠风险场表示

Joo 等人在高速公路场景中提出用有限标量风险场描述车辆产生的风险，并用车辆间风险场的重叠来表达 conflict field：

```text
conflict_field = overlap(risk_field_i, risk_field_j)
```

它的意义是：

- 可以同时描述跟驰、让行、换道等多种并发冲突。
- 风险场可视化能比 PET、MTTC、DRAC 等离散 surrogate safety measures 更连续。
- 适合用在混合交通流和自动驾驶风险评估中。

这和我们当前设计的关系：

- 真实训练 cost 中，多个车辆风险会累加，因此同一步多个邻车会提高 `vehicle_cost`。
- topdown overlay 可视化中，同一像素采用 `max()` 而不是 alpha 叠加，目的是避免图像透明度叠加制造“假高风险”。这只是可视化合成策略，不改变真实 cost。

参考：Joo et al., 2023, [A generalized driving risk assessment on high-speed highways using field theory](https://www.sciencedirect.com/science/article/pii/S2213665723000386)。

### Crash-Injury-Aware DSF: 风险不仅是概率，也包含后果严重度

Guo 等人在 crash-injury-aware DSF 中强调，风险不应只考虑碰撞概率，还应考虑潜在伤害严重度。该类方法会把车辆几何、碰撞区域、速度/能量、事故严重度模型等因素融入势场强度。

这和我们当前设计的关系：

- 现在的 `risk_field_collision_equivalent_cost = 1.0` 是一个统一的“碰撞等价”尺度，并没有区分低速擦碰和高速重撞。
- 如果后续要更贴近事故严重度，可以把 `vehicle_cost` 或最终 `risk_field_event_equivalent_cost` 与速度、相对速度、车辆质量、冲击角等 severity proxy 关联。
- 但这会显著改变 cost 尺度，必须重新跑 IDM 标定。

参考：Guo et al., 2025, [A crash-injury-aware driving safety field for real-time risk assessment and its application in autonomous vehicle motion planning](https://www.sciencedirect.com/science/article/pii/S0001457525003124)。

### 道路边界风险

道路边界风险使用一维高斯衰减：

```text
R_1d(d, sigma) = exp(-max(d, 0)^2 / (2 * sigma^2))
```

距离边界越近，`d` 越小，风险越接近 `1`。距离越远，风险快速衰减。

边界风险：

```text
boundary_cost = R_1d(min(dist_to_left_boundary, dist_to_right_boundary), boundary_sigma)
```

如果车辆已经越界，则使用越界距离的绝对值作为边缘带衰减，避免整片路外区域全部变成高风险。

### 车道线风险

MetaDrive 中一条 lane 的左右边线使用 `side=0` 和 `side=1` 表示：

```text
side0 gap = lateral + lane_width / 2
side1 gap = lane_width / 2 - lateral
```

每条边线根据线型给不同风险系数：

```text
broken   = risk_field_broken_line_factor    # 默认 0.05
solid    = risk_field_solid_line_factor     # 默认 0.60
boundary = risk_field_boundary_line_factor  # 默认 1.00
oncoming = risk_field_oncoming_line_factor  # 默认 1.50
```

车道线风险取两侧边线的最大值：

```text
lane_cost = max(
    side0_factor * R_1d(side0_gap, side0_sigma),
    side1_factor * R_1d(side1_gap, side1_sigma)
)
```

这样做的含义是：虚线换道风险最低，普通实线中等，道路边缘高，黄色/对向分隔线最高。

### 偏离道路风险

如果当前位置不在 MetaDrive 路网可行驶 surface 上：

```text
offroad_cost = risk_field_offroad_cost * R_1d(offroad_distance, offroad_sigma)
```

这里也是边缘带模型，只在道路边缘附近给风险，不把整个路外区域涂满。

### 车辆静态风险

周车静态势场使用超高斯分布。对每辆周车，先把 ego 位置转到该车朝向坐标系下，得到：

```text
longitudinal = ego 相对周车的纵向距离
lateral      = ego 相对周车的横向距离
```

静态车辆风险：

```text
static_vehicle_cost =
exp(-(
    ((longitudinal^2) / sigma_long^2)^beta
  + ((lateral^2)      / sigma_lat^2)^beta
))
```

默认参数：

```text
sigma_long = risk_field_vehicle_longitudinal_sigma = 5.0
sigma_lat  = risk_field_vehicle_lateral_sigma      = 1.6
beta       = risk_field_vehicle_beta               = 2.0
```

`sigma_long` 控制前后风险长度，`sigma_lat` 控制横向风险宽度，`beta` 控制形状尖锐程度。

### 车辆动态风险

动态车辆风险额外考虑相对速度：

```text
speed_delta = abs(v_other_forward - v_ego_forward)
dynamic_sigma = max(
    risk_field_vehicle_dynamic_sigma_scale * speed_delta,
    risk_field_vehicle_min_dynamic_sigma
)
```

动态风险基础项：

```text
dynamic_base =
exp(-(
    longitudinal^2 / dynamic_sigma^2
  + lateral^2      / lateral_sigma^2
))
```

再用相对速度方向做非对称调制：

```text
relv = 1   if v_other_forward >= v_ego_forward
relv = -1  otherwise

sigmoid_arg = -relv * (longitudinal - alpha * other_length * relv)

dynamic_vehicle_cost =
dynamic_base / (1 + exp(clip(sigmoid_arg, -60, 60)))
```

总车辆势场：

```text
vehicle_potential = static_vehicle_cost + dynamic_vehicle_cost
```

多辆车在真实 cost 中会累加为 `vehicle_cost`；在 topdown overlay 可视化里，为了避免透明叠加制造假高风险，同一像素通常取 `max()`。

### 静态障碍物风险

静态障碍物使用和车辆静态势场类似的超高斯模型，但没有动态项：

```text
object_cost =
exp(-(
    ((longitudinal^2) / object_sigma_long^2)^object_beta
  + ((lateral^2)      / object_sigma_lat^2)^object_beta
))
```

### Headway 风险

Headway 只针对同一行驶走廊内的前车计算。

```text
front_gap = max(delta_long - (ego_length + other_length) / 2, 0)
headway_time = front_gap / max(abs(ego_forward_speed), risk_field_min_speed)
```

如果 `headway_time` 小于阈值：

```text
headway_cost = clip(-log(headway_time / headway_threshold), 0, headway_cost_clip)
```

默认 `headway_threshold = 1.2s`。

### TTC 风险

TTC 也只针对同一行驶走廊内的前车计算。

```text
closing_speed = ego_forward_speed - other_forward_speed
ttc = front_gap / closing_speed
```

仅当 `closing_speed > 0` 时计算 TTC。如果 `ttc` 小于阈值：

```text
ttc_cost = clip(-log(ttc / ttc_threshold), 0, ttc_cost_clip)
```

默认 `ttc_threshold = 3.0s`。

## 当前各风险组件的量级

下面的量级基于当前 `env.py` 默认配置。这里先讨论单步原始风险，再讨论压缩后的 `info["cost"]`。

### 单步原始组件范围

| 组件 | 原始范围直觉 | 当前默认权重后的最大贡献直觉 | 说明 |
| --- | ---: | ---: | --- |
| `boundary_cost` | `0 ~ 1` | `0 ~ 1.0` | 距离道路边界越近越接近 1。 |
| `lane_cost` broken | `0 ~ 0.05` | `0 ~ 0.01` | 虚线允许换道，默认非常低。 |
| `lane_cost` solid | `0 ~ 0.60` | `0 ~ 0.12` | 普通实线中等惩罚。 |
| `lane_cost` boundary | `0 ~ 1.00` | `0 ~ 0.20` | 道路边缘线高惩罚，但还会同时受到 boundary cost 影响。 |
| `lane_cost` oncoming | `0 ~ 1.50` | `0 ~ 0.30` | 黄色/对向分隔线最高。 |
| `offroad_cost` | `0 ~ 1` | `0 ~ 1.0` | 仅在路外边缘带附近产生。 |
| 单车 `vehicle_cost` | `0 ~ 2` | `0 ~ 2.0` | 静态项最高约 1，动态项最高约 1，多辆车会累加。 |
| 单个 `object_cost` | `0 ~ 1` | `0 ~ 0.8` | 静态障碍物无动态项。 |
| 单个 `headway_cost` | `0 ~ 3` | `0 ~ 3.0` | 低于 1.2s 后按 `-log()` 增长并裁剪。 |
| 单个 `ttc_cost` | `0 ~ 3` | `0 ~ 3.0` | 低于 3.0s 后按 `-log()` 增长并裁剪。 |

注意：

- `lane_cost` 先根据线型得到 `0~factor`，再乘 `risk_field_lane_weight=0.2` 进入总风险。
- `vehicle_cost` 会对感知范围内多辆车求和，所以多车拥挤时原始车辆风险可能明显大于 2。
- `headway_cost` 和 `ttc_cost` 是最容易把 raw risk 撑大的预测性指标，因为单个前车就可能贡献到 3。
- 最终原始风险会被 `risk_field_raw_clip=10.0` 裁剪。

### 原始风险到碰撞等价 cost 的量级

当前默认：

```text
risk_field_cost_weight = 0.2
upper = 1.0
risk_event_equivalent = 1 - exp(-0.2 * risk_field_cost)
```

因此原始风险和单步碰撞等价 cost 的对应关系约为：

| 单步 `risk_field_cost` | 单步 `risk_field_event_equivalent_cost` |
| ---: | ---: |
| `0.1` | `0.020` |
| `0.5` | `0.095` |
| `1.0` | `0.181` |
| `2.0` | `0.330` |
| `5.0` | `0.632` |
| `10.0` | `0.865` |

含义：

- 即使 raw risk 到达裁剪上限 `10`，默认 `event_squash` 下单步风险场 cost 也约为 `0.865`，不会超过一次碰撞 `1.0`。
- 但如果一个 episode 有 800 到 1000 步，每步平均 `0.03 ~ 0.06` 的 cost 也会累计成 `24 ~ 60` 的 episode cost。
- 所以加入 dense risk field 后，`cost_limit` 从 `10/15` 上升到几十并不奇怪。

### 当前 smoke test 的量级参考

在当前环境中用 2 个 IDM episode 做 smoke test，结果为：

```text
episode 1: steps=899, total_cost=60.9028, risk_equiv=52.3066
episode 2: steps=643, total_cost=12.5683, risk_equiv=12.5683
recommended cost_limit = 61.6763
```

这只是 smoke test，不是正式标定。它说明：

- 当前 dense risk 的 episode cost 量级已经明显高于旧的 sparse event-only `10/15`。
- 最终训练前必须用更多 episodes 重新标定。
- 推荐至少使用 `--episodes 100`，并观察 p50、p80、p90、p95，而不是只看均值。

## env.py 中的碰撞等价映射

原始风险场 `risk_field_cost` 不能直接作为 Safe RL cost，因为它是连续势场强度，不是事件 cost。当前 `env.py` 使用 `event_squash` 把它压到碰撞等价尺度。

先缩放：

```text
x = risk_field_cost_weight * risk_field_cost
```

默认：

```text
risk_field_cost_weight = 0.2
```

再确定单步上限：

```text
upper = min(risk_field_collision_equivalent_cost, risk_field_cost_clip)
```

默认：

```text
risk_field_collision_equivalent_cost = 1.0
risk_field_cost_clip = 1.0
upper = 1.0
```

默认映射：

```text
risk_field_event_equivalent_cost =
upper * (1 - exp(-x / upper))
```

因此单步风险场 cost 被压缩到 `[0, 1]`，其中 `1` 表示“一次碰撞等价”的单步风险上限。

如果配置为 `linear_clip`，则使用：

```text
risk_field_event_equivalent_cost = clip(x, 0, upper)
```

## 最终 info["cost"]

MetaDrive 原始事件 cost 包括碰撞、撞障碍物、偏离道路等离散事件：

```text
event_cost = super().cost_function(...)
```

当前默认组合方式：

```text
info["cost"] = max(
    risk_field_event_cost_weight * event_cost,
    risk_field_event_equivalent_cost
)
```

默认：

```text
risk_field_event_cost_weight = 1.0
risk_field_cost_combine = "max"
```

使用 `max` 的目的：当碰撞已经发生时，不再把事件 cost 和风险场 cost 相加，避免同一步双重计数。

也可以配置为：

```text
sum        # event + risk
risk_only  # 只用风险场
event_only # 只用事件
```

但训练阶段推荐先保留默认 `event_squash + max`，因为它最接近“一步最多一次碰撞等价 cost”的语义。

## cost_limit 的真实单位

这是最重要的部分。

`FastCollector.collect()` 中：

```text
total_cost += sum(info["cost"])
stats_train["cost"] = total_cost / episode_count
```

所以 `train/cost` 和 Lagrange 更新使用的 `stats_train["cost"]` 都是：

```text
平均单个 episode 的累计 cost
```

它不是：

- 单步 cost。
- 原始风险场 `risk_field_cost`。
- `risk_field_event_equivalent_cost` 的最大值。
- cost critic 的网络输出。

如果一个 episode 长度约为 `H`，那么：

```text
平均单步 cost 预算 ≈ cost_limit / H
```

例子：如果 `H = 1000`，`cost_limit = 20`，那么平均每步只能有约 `0.02` 的 cost。即使单步风险场已经压缩到 `<= 1`，长期的小风险也会在 episode 里累积成很大的总 cost。

## cost critic 和 Lagrange 的作用

PPO-Lag 和 SAC-Lag 都有 reward critic 和 cost critic：

- reward critic 学习未来奖励回报。
- cost critic 学习未来 cost 回报。
- actor 更新时，一方面最大化 reward，另一方面通过 Lagrange 乘子惩罚高 cost 动作。

Lagrange 乘子的更新在 `LagrangianPolicy.pre_update_fn()` 中：

```text
cost_values = stats_train["cost"]
lagrangian.step(cost_values, cost_limit)
```

含义：

- 如果采样到的平均 episode cost 高于 `cost_limit`，Lagrange 乘子增大。
- 乘子增大后，actor 的 safety loss 权重变大，更倾向于选择 cost critic 预测较低的动作。
- 如果平均 episode cost 低于 `cost_limit`，Lagrange 乘子会逐渐降低，训练会更重视 reward。

因此，`cost_limit` 不是一个随便的惩罚系数，而是算法认为“可接受的平均 episode cost 上限”。

## 为什么加入风险场后必须重新标定

旧的 event-only cost 通常是稀疏的：

```text
碰撞一次 cost = 1
偏离道路一次 cost = 1
大部分正常行驶 step cost = 0
```

加入风险场后，cost 变成 dense signal：

```text
靠近车道边线有小 cost
靠近周车有小 cost
headway / TTC 变小时有预测性 cost
即使没碰撞，也可能每一步都有非零 cost
```

所以旧的 `cost_limit=10` 或 `cost_limit=15` 不能直接沿用。它们原本表示“允许若干次离散事件”，现在可能表示“允许多少个碰撞等价的风险步累计量”。

## 推荐标定方式：IDM p90 * 1.1

推荐用官方 IDMPolicy 作为可接受安全驾驶参考线：

```bash
python calibrate_cost_limit_idm.py \
  --episodes 100 \
  --num-scenarios 50 \
  --start-seed 100 \
  --recommend-percentile 90 \
  --margin 1.1
```

脚本会统计每个 episode 的：

- `sum_cost`: 最终进入 Safe RL 的 episode 累计 cost。
- `sum_event_cost`: 原始离散事件累计 cost。
- `sum_risk_field_cost`: 原始风险场累计值。
- `sum_risk_field_event_equivalent_cost`: 碰撞等价风险累计值。
- 各组件累计值：boundary、lane、offroad、vehicle、object、headway、ttc。

推荐值：

```text
cost_limit = percentile_90(sum_cost over IDM episodes) * 1.1
```

为什么用 p90：

- p50 太乐观，可能让训练约束过紧。
- p95 / max 太宽松，可能放过明显危险行为。
- p90 再乘 `1.1` 是一个比较稳的起点，允许少数复杂场景，但不会完全失去约束。

如果想更严格：

```text
cost_limit = p80
```

如果训练一开始就被安全约束压死：

```text
cost_limit = p95 * 1.1
```

## 如何把标定结果用于训练

标定完成后，脚本会打印：

```text
recommended cost_limit: <value>
```

训练时显式传入：

```bash
python train_ppol.py --task SafeMetaDrive --cost_limit <value>
python train_sacl.py --task SafeMetaDrive --cost_limit <value>
```

不要只看默认配置里的 `cost_limit`。默认值可能来自旧的 sparse event cost 时代，不一定适合当前风险场尺度。

## 如何判断 cost 设计是否合理

优先观察这些指标：

- `train/cost`: 最终用于 Lagrange 约束的 episode 平均累计 cost。
- `train/event_cost`: 原始离散事件累计 cost。
- `train/risk_field_cost`: 原始连续风险场累计值。
- `train/risk_field_event_equivalent_cost`: 压缩后的碰撞等价风险累计值。
- `train/risk_field_vehicle_cost`: 车辆势场贡献。
- `train/risk_field_lane_cost`: 车道线贡献。
- `train/risk_field_headway_cost`: 跟车时距贡献。
- `train/risk_field_ttc_cost`: TTC 贡献。

一个比较健康的状态：

- 安全策略的 `event_cost` 接近 0。
- `risk_field_event_equivalent_cost` 有连续变化，而不是长期全 0 或长期饱和。
- `train/cost` 的量级接近 IDM 标定结果。
- Lagrange 乘子不会一直爆炸，也不会长期为 0。

## 参数调整指南

如果标定出来的 cost 太大，先调这些参数：

| 现象 | 优先调整 |
| --- | --- |
| 所有场景 cost 都很高 | 降低 `risk_field_cost_weight` |
| 单步风险经常接近 1 | 降低 `risk_field_cost_weight` 或 `risk_field_cost_clip` |
| 换道 cost 过高 | 降低 `risk_field_lane_weight` 或 `risk_field_broken_line_factor` |
| 实线/黄线约束太弱 | 提高 `risk_field_solid_line_factor` 或 `risk_field_oncoming_line_factor` |
| 跟车 cost 过高 | 降低 `risk_field_headway_weight`、`risk_field_ttc_weight`，或降低阈值 |
| 周车风险范围太大 | 降低 `risk_field_vehicle_longitudinal_sigma` 或 `risk_field_vehicle_lateral_sigma` |
| 动态风险太敏感 | 降低 `risk_field_vehicle_dynamic_sigma_scale` |
| 车辆几乎不避障 | 提高 `risk_field_vehicle_weight` 或车辆 sigma |

推荐调参顺序：

1. 先用 GIF 可视化确认风险场形状合理。
2. 再用 IDM 标定确认 episode cost 尺度合理。
3. 最后才改 `cost_limit` 进入训练。

## 推荐工作流

1. 生成 full overlay GIF，确认图上风险来源合理：

```bash
python debug_risk_field_topdown_overlay_gif.py \
  --output debug/risk_full_overlay.gif \
  --frames 120 \
  --fps 10
```

2. 跑短 IDM smoke test：

```bash
python calibrate_cost_limit_idm.py \
  --episodes 2 \
  --num-scenarios 2 \
  --start-seed 100 \
  --recommend-percentile 90 \
  --margin 1.1 \
  --no-save
```

3. 跑正式 IDM 标定：

```bash
python calibrate_cost_limit_idm.py \
  --episodes 100 \
  --num-scenarios 50 \
  --start-seed 100 \
  --recommend-percentile 90 \
  --margin 1.1
```

4. 如果继续使用 dense-risk mixed 主 cost，则使用推荐值训练：

```bash
python train_sacl.py --task SafeMetaDrive --cost_limit <recommended_cost_limit>
```

5. 如果采用推荐的事件主约束路线，则保留风险场日志，但把主 cost 切回事件语义：

```python
{
    "use_risk_field_cost": True,
    "risk_field_cost_combine": "event_only",
}
```

6. 对比 PPO/SAC 的 `train/cost`、`test/cost`、`risk_field_*` 分解和成功率。

## 常见误区

### 误区 1：单步风险场已经小于 1，所以 cost_limit 也应该小于 1

不对。`cost_limit` 是 episode 累计尺度。如果 episode 有 1000 步，即使每步平均 cost 只有 `0.02`，episode total cost 也会达到 `20`。

### 误区 2：cost critic 会自动适配任何尺度

不完全对。critic 可以学习当前尺度，但 Lagrange 乘子会把 `stats_train["cost"]` 和 `cost_limit` 比较。如果 `cost_limit` 尺度错了，算法会长期过度保守或过度冒险。

### 误区 3：风险场越大越安全

不一定。风险场过大可能让 agent 认为所有动作都危险，导致学习不到有效驾驶策略。风险场应该提供“可区分”的安全梯度，而不是把所有正常行为都压成高 cost。

### 误区 4：只看总 cost 就够了

不够。必须同时看 `risk_field_boundary_cost`、`risk_field_lane_cost`、`risk_field_vehicle_cost`、`risk_field_headway_cost`、`risk_field_ttc_cost`。否则不知道 cost 是由车道线、周车、headway 还是 TTC 主导。

## 当前推荐默认策略

在没有新的标定结果前，建议：

- 不建议把 dense risk 长期积分直接混入主 `cost_limit`。
- 更稳妥的训练默认是：事件 cost 做主约束，风险场继续做日志、可视化和门控辅助指标。
- 如果为了复现实验仍保留 `event_squash + max`，必须用 IDM `p90 * 1.1` 重新设置 `cost_limit`，不要沿用旧的 `10/15`。
- 每次修改风险场参数后都重新跑 IDM 标定，并同时观察 step mean、exposure rate、max risk 这类 horizon-insensitive 指标。

这样可以避免离散事件 cost 被长时域 dense risk 掩盖，同时保留风险场对 near-miss 和早期危险的表达能力。
