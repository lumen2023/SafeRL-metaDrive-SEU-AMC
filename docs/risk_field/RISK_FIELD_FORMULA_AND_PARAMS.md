# 当前风险场公式与参数速查

本文档只回答一个问题：

> 当前代码中，风险场到底怎么算？每个参数现在是什么值？它对训练 cost 有什么影响？

对应代码：

- `env.py`: 默认参数、风险场到 `info["cost"]` 的映射、最终 cost 合成。
- `risk_field.py`: MetaDrive-native 风险场公式。

## 当前总览

当前默认配置中：

```python
"use_risk_field_cost": True
"risk_field_cost_combine": "max"
"risk_field_cost_transform": "event_squash"
"risk_field_cost_weight": 0.2
"risk_field_collision_equivalent_cost": 1.0
"risk_field_cost_clip": 1.0
```

因此，当前训练时的最终 `info["cost"]` 不是纯事件 cost，而是：

```text
info["cost"] = max(event_cost, risk_event_equivalent_cost)
```

这意味着：

- 离散事件 cost 仍保留。
- dense risk 当前仍会进入主安全约束。
- 单步风险场 cost 被压缩到不超过 `1.0`。
- 但 episode 内仍会长时域累积，所以 `cost_limit` 仍会被 dense risk 影响。

当前预测性指标：

```python
"risk_field_headway_weight": 0.0
"risk_field_ttc_weight": 0.0
```

所以：

- headway 和 TTC 会被计算并写入 `info`。
- 但它们当前不参与 `risk_field_cost` 的加权求和。

## 完整计算链路

单步计算顺序如下：

```text
MetaDrive scene
    ↓
RiskFieldCalculator.calculate(env, vehicle)
    ↓
raw component costs:
    boundary_cost
    lane_cost
    offroad_cost
    vehicle_cost
    object_cost
    headway_cost
    ttc_cost
    ↓
weighted raw risk:
    risk_field_cost
    ↓
env.py event-equivalent mapping:
    risk_event_equivalent_cost
    ↓
event/risk combine:
    info["cost"]
    ↓
FastCollector:
    episode accumulated train/cost
```

## 原始风险总公式

`risk_field.py` 中的总原始风险为：

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
risk_field_cost = min(max(risk_raw, 0), risk_field_raw_clip)
```

当前参数：

```python
"risk_field_raw_clip": 10.0
```

## 当前组件权重

| 参数 | 当前值 | 作用 | 当前影响 |
| --- | ---: | --- | --- |
| `risk_field_boundary_weight` | `1.0` | 道路边界风险权重 | 参与 raw risk |
| `risk_field_lane_weight` | `0.2` | 车道线风险权重 | 参与 raw risk，较低 |
| `risk_field_offroad_weight` | `1.0` | 路外风险权重 | 参与 raw risk |
| `risk_field_vehicle_weight` | `1.0` | 周车风险权重 | 参与 raw risk |
| `risk_field_object_weight` | `0.8` | 静态障碍物风险权重 | 参与 raw risk |
| `risk_field_headway_weight` | `0.0` | 车头时距风险权重 | 当前只记录，不贡献 |
| `risk_field_ttc_weight` | `0.0` | TTC 风险权重 | 当前只记录，不贡献 |

当前真正主导 `risk_field_cost` 的是：

```text
boundary + lane + offroad + vehicle + object
```

当前不会主导 `risk_field_cost` 的是：

```text
headway + ttc
```

但它们仍然在 `info["risk_field_headway_cost"]` 和 `info["risk_field_ttc_cost"]` 中可见。

## 一维高斯风险函数

道路边界、车道线、offroad 都使用一维高斯衰减：

```text
R_1d(d, sigma) = exp(-max(d, 0)^2 / (2 * sigma^2))
```

性质：

- `d = 0` 时，风险为 `1.0`。
- `d = sigma` 时，风险约为 `0.607`。
- `d = 2 * sigma` 时，风险约为 `0.135`。
- `sigma` 越大，风险影响范围越宽。

当前道路几何 sigma：

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `risk_field_boundary_sigma` | `0.75` m | 道路边界风险衰减宽度 |
| `risk_field_lane_edge_sigma` | `0.75` m | 车道线风险衰减宽度 |
| `risk_field_offroad_sigma` | `1.0` m | 路外边缘带衰减宽度 |

## 道路边界风险

代码位置：`RiskFieldCalculator._road_risk()`。

优先使用 MetaDrive vehicle 的道路边界距离：

```text
left_dist  = vehicle.dist_to_left_side
right_dist = vehicle.dist_to_right_side
```

如果没有，则回退到当前 lane 的宽度和横向偏移：

```text
left_dist  = lateral + lane_width / 2
right_dist = lane_width / 2 - lateral
```

边界风险：

```text
d_boundary = min(left_dist, right_dist)

if d_boundary >= 0:
    boundary_cost = R_1d(d_boundary, risk_field_boundary_sigma)
else:
    boundary_cost = R_1d(abs(d_boundary), risk_field_boundary_sigma)
```

当前参数：

```python
"risk_field_boundary_weight": 1.0
"risk_field_boundary_sigma": 0.75
```

量级：

```text
boundary_cost ∈ [0, 1]
加权贡献 ∈ [0, 1]
```

注意：

- 越靠近道路外侧边界，风险越高。
- 该项使用道路边界距离，不等同于所有车道线。

## 车道线风险

代码位置：`RiskFieldCalculator._road_risk()` 和 `lane_line_risk_profile()`。

MetaDrive lane 的两侧边线：

```text
side0: lateral = -lane_width / 2
side1: lateral = +lane_width / 2
```

当前车辆相对两侧边线的 gap：

```text
side0_gap = lateral + lane_width / 2
side1_gap = lane_width / 2 - lateral
```

每条线根据线型得到风险系数：

| 线型 kind | 参数 | 当前值 | 语义 |
| --- | --- | ---: | --- |
| `broken` | `risk_field_broken_line_factor` | `0.05` | 虚线，允许换道，几乎不惩罚 |
| `solid` | `risk_field_solid_line_factor` | `0.60` | 普通实线，禁止变道，中等惩罚 |
| `boundary` | `risk_field_boundary_line_factor` | `1.00` | 道路边缘线，高惩罚 |
| `oncoming` | `risk_field_oncoming_line_factor` | `1.50` | 黄色/对向分隔线，最高惩罚 |

单条边线风险：

```text
side_i_cost = side_i_factor * R_1d(side_i_gap, side_i_sigma)
```

车道线风险取两侧最大值：

```text
lane_cost = max(side0_cost, side1_cost)
```

再进入总风险时乘组件权重：

```text
weighted_lane = risk_field_lane_weight * lane_cost
```

当前参数：

```python
"risk_field_lane_weight": 0.2
"risk_field_lane_edge_sigma": 0.75
```

当前最大贡献直觉：

| 线型 | `lane_cost` 最大值 | 乘 `lane_weight=0.2` 后最大贡献 |
| --- | ---: | ---: |
| 虚线 broken | `0.05` | `0.01` |
| 实线 solid | `0.60` | `0.12` |
| 边界线 boundary | `1.00` | `0.20` |
| 对向线 oncoming | `1.50` | `0.30` |

注意：

- 当前虚线换道几乎不惩罚。
- 对向线虽然最高，但经过 `lane_weight=0.2` 后最大单步 raw 贡献约 `0.30`。
- 道路最外侧还会同时受到 `boundary_cost` 影响，所以实际边界风险不只来自 `lane_cost`。

## Offroad 风险

代码位置：`RiskFieldCalculator._lane_surface_state()` 和 `_road_risk()`。

先用 MetaDrive road network 判断当前位置是否在任意 candidate lane surface 上。

如果不在道路上：

```text
offroad_cost =
    risk_field_offroad_cost
  * R_1d(offroad_distance, risk_field_offroad_sigma)
```

当前参数：

```python
"risk_field_offroad_weight": 1.0
"risk_field_offroad_cost": 1.0
"risk_field_offroad_sigma": 1.0
"risk_field_on_lane_margin": 0.05
```

量级：

```text
offroad_cost ∈ [0, 1]
加权贡献 ∈ [0, 1]
```

注意：

- 当前 offroad 是“路外边缘带”模型，不是整片路外区域都高风险。
- `_road_risk()` 中还有：

```text
boundary_cost = max(boundary_cost, offroad_cost)
```

所以如果已经 offroad，`offroad_cost` 可能同时抬高 `boundary_cost`。在当前总风险中 `boundary_weight` 和 `offroad_weight` 都是 `1.0`，因此路外边缘区域可能出现边界与 offroad 的双重贡献。这是当前代码真实行为。

## 车辆风险

代码位置：`RiskFieldCalculator._vehicle_risk()` 和 `vehicle_potential_components()`。

只统计距离 ego 小于：

```python
"risk_field_max_distance": 50.0
```

的周围车辆。

每辆周车风险由两部分组成：

```text
vehicle_potential = static_vehicle_cost + dynamic_vehicle_cost
```

多辆车求和：

```text
vehicle_cost = sum_over_surrounding_vehicles(vehicle_potential)
```

进入总风险：

```text
weighted_vehicle = risk_field_vehicle_weight * vehicle_cost
```

当前：

```python
"risk_field_vehicle_weight": 1.0
```

### 车辆坐标系

对每辆周车，先把 ego 位置投影到周车自身朝向坐标系：

```text
longitudinal = dot(ego_pos - other_pos, other_forward)
lateral      = dot(ego_pos - other_pos, other_left)
```

因此车辆风险场中心在周车位置，方向跟随周车 heading。

### 车辆静态势场

当前代码公式：

```text
static_vehicle_cost =
exp(-(
    ((longitudinal^2) / sigma_long^2)^beta
  + ((lateral^2)      / sigma_lat^2)^beta
))
```

当前参数：

```python
"risk_field_vehicle_longitudinal_sigma": 5.0
"risk_field_vehicle_lateral_sigma": 1.6
"risk_field_vehicle_beta": 2.0
```

量级：

```text
static_vehicle_cost ∈ [0, 1]
```

重要注意：

- 按当前代码公式，`beta=1` 更接近标准二维高斯的平方距离形式。
- 当前 `beta=2` 是更尖锐的超高斯，会让车辆附近风险更集中，远处衰减更快。
- `env.py` 注释里“beta=2 为标准高斯”的说法从数学上不精确；当前真实实现以 `risk_field.py` 为准。

### 车辆动态势场

动态势场先计算相对速度导致的纵向扩散：

```text
speed_delta = abs(other_forward_speed - ego_forward_speed)

dynamic_sigma = max(
    risk_field_vehicle_dynamic_sigma_scale * speed_delta,
    risk_field_vehicle_min_dynamic_sigma
)
```

当前参数：

```python
"risk_field_vehicle_dynamic_sigma_scale": 2.0
"risk_field_vehicle_min_dynamic_sigma": 0.5
```

动态基础项：

```text
dynamic_base =
exp(-(
    longitudinal^2 / dynamic_sigma^2
  + lateral^2      / sigma_lat^2
))
```

再加入相对速度方向的 sigmoid 非对称项：

```text
relv = 1.0  if other_forward_speed >= ego_forward_speed
relv = -1.0 otherwise

sigmoid_arg = -relv * (longitudinal - alpha * other_length * relv)

dynamic_vehicle_cost =
dynamic_base / (1 + exp(clip(sigmoid_arg, -60, 60)))
```

当前参数：

```python
"risk_field_vehicle_dynamic_alpha": 0.9
```

量级：

```text
dynamic_vehicle_cost ∈ [0, 1]
单车 vehicle_potential ∈ [0, 2]
```

调参直觉：

- 增大 `risk_field_vehicle_longitudinal_sigma`: 车辆前后风险范围变长。
- 增大 `risk_field_vehicle_lateral_sigma`: 车辆横向风险范围变宽，更保守。
- 增大 `risk_field_vehicle_beta`: 风险更集中、更像“方块中心高、边缘快衰减”。
- 增大 `risk_field_vehicle_dynamic_sigma_scale`: 相对速度导致的动态风险范围更大。
- 增大 `risk_field_vehicle_dynamic_alpha`: 非对称风险沿车辆长度方向偏移更明显。

## 静态障碍物风险

代码位置：`RiskFieldCalculator._object_risk()`。

只统计距离 ego 小于：

```python
"risk_field_max_distance": 50.0
```

的 traffic objects。

公式：

```text
object_cost_single =
exp(-(
    ((longitudinal^2) / object_sigma_long^2)^object_beta
  + ((lateral^2)      / object_sigma_lat^2)^object_beta
))
```

多障碍物求和：

```text
object_cost = sum_over_objects(object_cost_single)
```

进入总风险：

```text
weighted_object = risk_field_object_weight * object_cost
```

当前参数：

```python
"risk_field_object_weight": 0.8
"risk_field_object_longitudinal_sigma": 5.0
"risk_field_object_lateral_sigma": 1.6
"risk_field_object_beta": 2.0
```

量级：

```text
单个 object_cost_single ∈ [0, 1]
单个障碍物加权最大贡献约 0.8
```

## Headway 风险

代码位置：`RiskFieldCalculator._vehicle_risk()` 和 `_time_threshold_cost()`。

虽然当前不进入总风险，但仍会计算并写入 `info`。

只对同一行驶走廊中的前车计算：

```text
is_front_vehicle = delta_long > 0
is_same_corridor = abs(delta_lat) <= max(ego_lane_width * 0.75, (ego_width + other_width) / 2)
```

车头间距：

```text
front_gap = max(delta_long - (ego_length + other_length) / 2, 0)
```

车头时距：

```text
headway_time = front_gap / max(abs(ego_forward_speed), risk_field_min_speed)
```

阈值代价：

```text
if headway_time < risk_field_headway_time_threshold:
    headway_cost = clip(
        -log(headway_time / risk_field_headway_time_threshold),
        0,
        risk_field_headway_cost_clip
    )
else:
    headway_cost = 0
```

当前参数：

```python
"risk_field_headway_weight": 0.0
"risk_field_headway_time_threshold": 1.2
"risk_field_headway_cost_clip": 3.0
"risk_field_min_speed": 0.5
```

当前影响：

```text
info["risk_field_headway_cost"] 有值
但 risk_field_cost 中贡献为 0
```

## TTC 风险

代码位置：`RiskFieldCalculator._vehicle_risk()` 和 `_time_threshold_cost()`。

只对同一行驶走廊中的前车计算，并且 ego 正在接近前车：

```text
closing_speed = ego_forward_speed - other_forward_speed
```

当：

```text
closing_speed > 0
```

时：

```text
ttc = front_gap / closing_speed
```

阈值代价：

```text
if ttc < risk_field_ttc_threshold:
    ttc_cost = clip(
        -log(ttc / risk_field_ttc_threshold),
        0,
        risk_field_ttc_cost_clip
    )
else:
    ttc_cost = 0
```

当前参数：

```python
"risk_field_ttc_weight": 0.0
"risk_field_ttc_threshold": 3.0
"risk_field_ttc_cost_clip": 3.0
```

当前影响：

```text
info["risk_field_ttc_cost"] 有值
但 risk_field_cost 中贡献为 0
```

## 原始风险到碰撞等价 cost

`env.py` 不直接把 `risk_field_cost` 放进 `info["cost"]`，而是先压缩：

```text
scaled_risk = risk_field_cost_weight * risk_field_cost
```

当前：

```python
"risk_field_cost_weight": 0.2
```

单步上限：

```text
upper = min(
    risk_field_collision_equivalent_cost,
    risk_field_cost_clip
)
```

当前：

```python
"risk_field_collision_equivalent_cost": 1.0
"risk_field_cost_clip": 1.0
upper = 1.0
```

当前 transform：

```python
"risk_field_cost_transform": "event_squash"
```

公式：

```text
risk_event_equivalent_cost =
upper * (1 - exp(-scaled_risk / upper))
```

代入当前值：

```text
risk_event_equivalent_cost =
1 - exp(-0.2 * risk_field_cost)
```

映射表：

| 单步 `risk_field_cost` | 单步 `risk_event_equivalent_cost` |
| ---: | ---: |
| `0.1` | `0.020` |
| `0.5` | `0.095` |
| `1.0` | `0.181` |
| `2.0` | `0.330` |
| `5.0` | `0.632` |
| `10.0` | `0.865` |

如果 transform 改为：

```python
"risk_field_cost_transform": "linear_clip"
```

则：

```text
risk_event_equivalent_cost = clip(scaled_risk, 0, upper)
```

## 最终 info["cost"]

MetaDrive 原始事件 cost：

```text
event_cost =
    out_of_road_cost       if out of road
    crash_vehicle_cost     if crash vehicle
    crash_object_cost      if crash object
    0                      otherwise
```

当前事件参数：

```python
"crash_vehicle_cost": 1.0
"crash_object_cost": 1.0
"out_of_road_cost": 1.0
"risk_field_event_cost_weight": 1.0
```

当前组合方式：

```python
"risk_field_cost_combine": "max"
```

最终公式：

```text
info["cost"] =
max(
    risk_field_event_cost_weight * event_cost,
    risk_event_equivalent_cost
)
```

代入当前值：

```text
info["cost"] = max(event_cost, risk_event_equivalent_cost)
```

支持的其他组合方式：

| 模式 | 公式 | 语义 |
| --- | --- | --- |
| `max` | `max(event, risk)` | 当前默认；避免同一步双重计数 |
| `sum` | `event + risk` | 事件和风险相加，episode cost 更大 |
| `risk_only` | `risk` | 只训练风险场约束 |
| `event_only` | `event` | 主约束回到离散事件，风险场可保留日志 |

## 当前阶段最重要的事实

### 1. 当前风险场仍然影响训练 cost

因为：

```python
"use_risk_field_cost": True
"risk_field_cost_combine": "max"
```

所以最终 `info["cost"]` 会受到 dense risk 影响。

如果你希望主约束回到事件语义，应改为：

```python
"risk_field_cost_combine": "event_only"
```

### 2. Headway/TTC 当前不影响训练 cost

因为：

```python
"risk_field_headway_weight": 0.0
"risk_field_ttc_weight": 0.0
```

所以它们只作为日志和诊断指标。

### 3. 当前 raw risk 主要来自道路、车辆、障碍物

当前参与 raw risk 的项：

```text
boundary + lane + offroad + vehicle + object
```

其中最可能撑大 episode cost 的是：

- `vehicle_cost`: 多车累加，单车最高约 2。
- `boundary_cost` / `offroad_cost`: 靠近边界或路外时接近 1。
- `object_cost`: 静态障碍物附近会贡献风险。

### 4. 单步 cost 不大，不代表 episode cost 不大

当前单步 risk 被压缩到不超过 1：

```text
risk_event_equivalent_cost <= 1
```

但 episode cost 是逐步累加：

```text
episode_cost = sum_t info["cost"]
```

因此长期小风险会积累，导致 `cost_limit` 失去原来的“事件次数”语义。

## 参数调节速查

| 想调整的现象 | 优先调节参数 | 方向 |
| --- | --- | --- |
| 总体风险太大 | `risk_field_cost_weight` | 降低 |
| 单步风险经常接近 1 | `risk_field_cost_weight` 或 `risk_field_cost_clip` | 降低 |
| 训练想回到事件约束 | `risk_field_cost_combine` | 改为 `event_only` |
| 道路边界太敏感 | `risk_field_boundary_weight` / `risk_field_boundary_sigma` | 降低权重或 sigma |
| 虚线换道仍被惩罚 | `risk_field_broken_line_factor` / `risk_field_lane_weight` | 降低 |
| 实线/黄线约束不够 | `risk_field_solid_line_factor` / `risk_field_oncoming_line_factor` | 提高 |
| 周车风险太宽 | `risk_field_vehicle_longitudinal_sigma` / `risk_field_vehicle_lateral_sigma` | 降低 |
| 周车风险太弱 | `risk_field_vehicle_weight` | 提高 |
| 动态速度风险太敏感 | `risk_field_vehicle_dynamic_sigma_scale` | 降低 |
| 静态障碍物风险太强 | `risk_field_object_weight` | 降低 |
| 想启用 headway/TTC 训练约束 | `risk_field_headway_weight` / `risk_field_ttc_weight` | 从 0 慢慢增大 |

## 建议的当前解释口径

如果写论文或实验说明，当前风险场可以描述为：

```text
The risk field is a MetaDrive-native continuous safety potential built from road boundary,
lane-line semantics, off-road exposure, surrounding vehicles, and static traffic objects.
The raw potential is clipped and mapped through an event-equivalent squash function before
being combined with discrete event cost by a max operator.
```

对应中文：

```text
当前风险场是一个基于 MetaDrive 原生路网、车道线语义、路外暴露、周围车辆和静态障碍物构建的连续安全势场。
原始势场经过裁剪后，通过碰撞等价的饱和映射压缩到单步不超过一次事件 cost 的尺度，
再用 max 操作与离散事件 cost 合成最终约束信号。
```

但如果强调标准 Safe RL benchmark 可比性，应同时说明：

```text
无门控 dense risk 会改变 cost_limit 的事件语义，因此推荐在主实验中使用 event_only，
或将风险场作为 near-miss 辅助约束单独标定。
```
