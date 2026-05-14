# SafeMetaDrive 环境封装 (env.py) 详细说明

## 📋 目录
- [概述](#概述)
- [直接运行效果](#直接运行效果)
- [核心功能](#核心功能)
- [配置参数详解](#配置参数详解)
- [使用示例](#使用示例)
- [架构设计](#架构设计)
- [调试与可视化](#调试与可视化)

---

## 概述

`env.py` 是 SafeMetaDrive 项目的**环境封装入口文件**，基于 MetaDrive 仿真平台构建了支持安全强化学习研究的定制化驾驶环境。该文件提供了三个核心环境类和丰富的配置系统，主要特性包括：

### ✨ 核心特性
1. **风险场成本系统** - 将连续的风险势场转化为 Safe RL 可用的成本信号
2. **偏离道路宽容模式** - 允许智能体短暂偏离后自我纠正，避免过度惩罚
3. **单场景随机化环境** - 支持 5 种典型交通场景的程序化生成与可控随机化
4. **灵活的成本组合策略** - 支持多种事件成本与风险成本的融合方式

---

## 直接运行效果

### 执行命令
```bash
python3 env.py
```

### 运行效果
执行后会启动一个**交互式手控驾驶演示**：

1. **环境初始化**
   - 创建训练环境 (`get_training_env`)
   - 启用手控模式 (`manual_control=True`)
   - 启用实时渲染 (`use_render=True`)

2. **可视化界面**
   - 打开 MetaDrive 渲染窗口
   - 以**俯视角度 (topdown)** 显示车辆和道路
   - 目标智能体（自车）始终朝上显示
   - 窗口中显示帮助信息（按键说明）

3. **交互控制**
   - 用户可通过**键盘**控制车辆行驶
   - 常见按键：WASD 或方向键控制加减速和转向
   - 空格键刹车，R 键重置等（具体见窗口帮助信息）

4. **自动重置**
   - 当 episode 结束（成功/失败/超时）时
   - 环境自动重置到新的初始状态
   - 可无限次重复体验

### 适用场景
- ✅ 快速验证环境配置是否正确
- ✅ 熟悉 MetaDrive 的操作手感
- ✅ 观察不同场景的地图结构
- ✅ 调试奖励和成本函数的即时反馈

---

## 核心功能

### 1️⃣ 风险场成本系统 (Risk Field Cost)

#### 设计理念
传统的 Safe RL 仅使用**离散事件成本**（碰撞=1, 偏离=1），缺乏对"接近危险但未发生事故"状态的感知。风险场系统通过计算周围物体产生的**连续势场强度**，为智能体提供更细粒度的安全约束信号。

#### 工作原理
```
总风险成本 = Σ (各组件权重 × 组件风险值)
U_total = w_boundary·E_boundary + w_lane·E_lane + w_vehicle·E_vehicle + ...
```

#### 风险组件详解

| 组件 | 物理含义 | 计算公式 | 默认权重 |
|------|---------|---------|---------|
| **边界风险** | 到道路左右边界的距离惩罚 | `exp(-d²/(2σ²))` | 2.0 |
| **车道线风险** | 到车道边缘的距离惩罚 | `exp(-gap²/(2σ²))` | 0.2 |
| **偏离道路风险** | 在不可行驶区域的风险带惩罚 | 边缘带衰减模型 | 2.0 |
| **车辆风险** | 周围车辆的静态占据+动态速度风险 | 超高斯分布 + 非对称动态调整 | 1.0 |
| **障碍物风险** | 静态障碍物（锥桶、路障）的风险 | 超高斯分布（无动态部分） | 0.8 |
| **车头时距** | 跟车过近的预警（行业标准指标） | `-ln(t_headway/t_threshold)` | 0.0 |
| **碰撞时间 TTC** | 预测性碰撞风险预警 | `-ln(TTC/TTC_threshold)` | 0.0 |

#### 成本映射策略

**linear_scale 模式（推荐）**
```python
normalized_risk = clip(risk_raw / risk_field_raw_clip, 0, 1)
event_equivalent_cost = risk_field_cost_scale * normalized_risk
```
- 简单线性归一化
- `risk_field_cost_scale` 控制单步最大事件等价成本
- 例如：scale=4.0 表示风险场最大值等价于 4 次碰撞事件

**legacy 模式（旧版兼容）**
- 支持 `event_squash` 和 `linear_clip` 两种变换
- 通过饱和函数压缩极端值
- 保留用于复现历史实验

#### 成本组合策略

| 模式 | 公式 | 适用场景 |
|------|------|---------|
| **max** | `max(event_cost, risk_cost)` | 避免双重计数（推荐） |
| **sum** | `event_cost + risk_cost` | 累加所有风险信号 |
| **risk_only** | `risk_cost` | 仅使用风险场 |
| **event_only** | `event_cost` | 禁用风险场 |

#### 关键配置参数
```python
"use_risk_field_cost": False,           # 总开关：是否启用风险场
"risk_field_cost_scale": 4.0,           # 尺度映射：单步最大事件等价cost
"risk_field_raw_clip": 1.0,             # 原始风险上限裁剪
"risk_field_event_cost_weight": 1.0,    # 离散事件成本的保留权重
"risk_field_cost_combine": "max",       # 组合策略
```

---

### 2️⃣ 偏离道路宽容模式 (Out-of-Road Warning Budget)

#### 问题背景
原始 MetaDrive 环境中，车辆一旦偏离道路立即终止 episode。这种**零容忍策略**导致：
- 智能体学习到过度保守的策略（不敢变道、不敢靠近边界）
- 无法区分"短暂越线后立即纠正"和"彻底失控"的差异
- 训练效率低下，样本利用率低

#### 解决方案：Warning Budget 机制

**工作流程**
```
检测到偏离道路
    ↓
是否在纠正步数内回到道路？
    ├─ 是 → 不计入warning，继续行驶
    └─ 否 → 触发一次warning
            ↓
        warning次数 < 上限？
            ├─ 是 → 给予warning惩罚，允许继续
            └─ 否 → 终止episode
```

**两种模式对比**

| 特性 | legacy 模式 | warning_budget 模式 |
|------|------------|-------------------|
| 偏离后立即终止 | ✅ | ❌ |
| 允许自我纠正 | ❌ | ✅ (默认15步) |
| warning次数限制 | N/A | 默认5次 |
| 超出预算后终止 | N/A | ✅ (可配置) |
| 独立的warning惩罚 | ❌ | ✅ |

#### 状态追踪
每次 step 返回的 `info` 字典包含：
```python
{
    "out_of_road_mode": "warning_budget",          # 当前模式
    "out_of_road_warning_count": 2,                # 已触发warning次数
    "out_of_road_recovery_remaining": 8,           # 剩余纠正步数
    "out_of_road_budget_exhausted": False,         # 预算是否耗尽
    "out_of_road_warning_triggered": True,         # 本步是否触发warning
    "out_of_road_timeout_terminated": False,       # 是否因超时而终止
}
```

#### 关键配置参数
```python
"out_of_road_mode": "legacy",               # 模式选择："legacy" 或 "warning_budget"
"out_of_road_warning_limit": 5,             # 允许的warning次数上限
"out_of_road_recovery_steps": 15,           # 每次偏离后的纠正步数
"out_of_road_warning_penalty": 1.0,         # 每次warning的奖励惩罚
"out_of_road_warning_cost": 1.0,            # 每次warning的成本值
"out_of_road_terminate_after_budget": True, # 超出预算后是否终止
```

---

### 3️⃣ 单场景随机化环境 (Single Scene Environment)

#### 设计理念
相比完全随机的程序化地图，单场景环境提供：
- **固定的几何结构** - 便于分析特定场景下的算法表现
- **可控的随机化** - 自车位置、交通流密度等可在预设范围内变化
- **确定性种子** - 保证实验可复现性

#### 支持的 5 种场景

##### 1. straight_4lane - 四车道直道
```
特点：
- 4条平行车道
- 总长度约420米（入口60m + 主直道300m + buffer 60m）
- 适合测试基础跟车、变道能力

随机化：
- 自车可在4条车道任意位置出生（5-30m范围）
- 默认要求自车同物理车道前方存在一辆车，最近前车 gap 落在 20-45m
- 交通车辆5-8辆，速度8-15 m/s
```

##### 2. ramp_merge - 匝道汇入
```
特点：
- 主路2车道 + 匝道汇入
- 包含加速段、合并段、后续路段
- 适合测试汇入决策和让行行为

随机化：
- 自车在主路approach段出生（5-25m）
- 强制在匝道和合并区生成车辆
- 交通车辆4-7辆
```

##### 3. t_intersection - T型交叉口
```
特点：
- 标准T型路口，默认自车导航目标固定为左转
- 包含入口、approach段、三个出口
- 适合测试左转路口通行策略

随机化：
- 自车在approach段出生（5-24m）
- 默认只从左转出口采样目的地；可通过 `single_scene.navigation.t_intersection_turn_policy` 改为 `right_only` 或 `mixed`
- 各方向均有交通流
```

##### 4. roundabout - 环岛
```
特点：
- 圆形环岛，3个出口
- 内径30m，出口半径10m
- 适合测试环岛的进入、绕行、退出策略

随机化：
- 自车在入口段出生（5-18m）
- 随机选择一个出口作为目的地
- 环岛内外均有交通车辆
```

##### 5. lane_change_bottleneck - 变道瓶颈路段
```
特点：
- 4车道收窄为2车道，再扩展回4车道
- 瓶颈段长度25米
- 适合测试拥堵场景下的变道博弈

随机化：
- 自车固定在右侧第2车道出生（5-12m）
- 强制在瓶颈前后生成车辆制造拥堵
- 横向抖动极小（±0.05m）保证车辆对齐
```

#### 随机化机制详解

**自车初始化**
```python
# 1. 随机选择车道（从候选列表中）
lane_candidates = [{"road": "entry", "lane": 0}, 
                   {"road": "entry", "lane": 1}, ...]
selected_lane = rng.choice(lane_candidates)

# 2. 随机纵向位置
spawn_longitude = uniform(long_low, long_high)

# 3. 随机初始速度（通常设为0）
spawn_speed = uniform(speed_low, speed_high)

# 4. 随机目的地（某些场景支持多目的地）
destination = sample_from_available_exits()
```

**交通流生成**
```python
# 1. 确定车辆数量
vehicle_count = randint(count_low, count_high)

# 2. 从预设槽位中选择。单场景环境使用自定义slot交通层，不再由MetaDrive通用PGTrafficManager直接摆车。
road_min_counts = {"approach": 2, "ramp": 2, "merge": 2, "post": 1}
mandatory_slots = [{"road": "ramp", "lane": 0, "longitude": 2.0}, {"road": "ramp", "lane": 0, "longitude": 12.0}]
optional_slots = [...]   # 可选位置池
selected = satisfy_road_min_counts_first(road_min_counts)
selected += random_sample(optional, vehicle_count - len(selected))

# 3. 应用随机扰动
for slot in selected:
    longitude += uniform(-3.0, 3.0)  # 纵向抖动
    lateral += uniform(-0.15, 0.15)  # 横向抖动
    speed = uniform(6.0, 12.0)       # 速度随机化
    
# 4. 最小间距检查（避免重叠）
if abs(new_long - existing_long) < min_gap:
    skip_this_slot

# 5. 运行中如果周车离开道路，单场景补车逻辑会继续从可用slot中补车，
#    并尽量维持目标车辆数和road_min_counts。
```

**确定性种子**
```python
# 不显式传seed时，每次reset会推进episode_index：
# train 模式：episode种子 = 10000 + scene_index * 1000 + episode_index * 97
# val   模式：episode种子 = 50000 + scene_index * 1000 + episode_index * 97
#
# 显式传seed时，固定几何地图仍使用该场景自己的start_seed，
# 但自车和周车布局使用用户传入的seed：
obs, info = env.reset(seed=12345)  # 重复调用会复现相同自车/周车初始化
```

**官方MetaDrive随机化开关**
```python
random_traffic      # 官方PG交通流是否每次reset都随机化
random_lane_width   # 官方PG地图是否随机车道宽度
random_lane_num     # 官方PG地图是否随机车道数量
random_agent_model  # 自车车型是否随机
start_seed          # 场景种子起点
num_scenarios       # 可采样场景数量
reset(seed=...)     # 指定某个场景/布局种子
```
> 单场景环境默认仍使用自己的 episode 级随机化和安全补车逻辑；当 `single_scene.traffic_backend="official"` 时，`traffic_density` / `traffic_mode` / `random_traffic` / `need_inverse_traffic` / `traffic_vehicle_config` 会直接按 MetaDrive 官方 `PGTrafficManager` 语义生效。

**周车出现与消失逻辑**
```python
TrafficMode.Basic    # reset时生成一批车；离开道路后消失，不补车
TrafficMode.Trigger  # 自车接近某个block时触发该block车辆；触发一次后不持续补
TrafficMode.Respawn  # reset时生成车；车辆离开道路后官方逻辑会在respawn lane补一辆
TrafficMode.Hybrid   # Trigger + Respawn 的混合模式
```
单场景环境现在支持两套周车后端：

```python
single_scene["traffic_backend"] = "single_scene"  # 默认：自定义安全补车
single_scene["traffic_backend"] = "official"      # 官方：直接使用 PGTrafficManager
```

`single_scene` 后端：
- 初始化用 `traffic_density` 按官方 10m 间隔候选点公式生成车辆。
- 运行中从 `replenish_roads` 做安全补车，避免突然刷在自车前方。
- 需要旧版固定 slot 数量逻辑时，将 `single_scene.randomization.traffic_count_mode` 设为 `"range"`。

`official` 后端：
- 初始化、触发、消失和补车全部交给官方 `PGTrafficManager`。
- `straight_4lane` 已把 `entry` 和各个 `straight_*` 段加入官方 respawn roads，官方 `Respawn/Basic` 不会再只在入口刷车。

`straight_4lane` 之前“自车前方同车道车辆很少”的主要原因有 3 个：
- 默认密度是按整段直道均匀抽样，100m 内同物理车道前车本来就不多。
- 旧的 ego profile 只看完全相同的 `lane_index`，拆段后的 `straight_1/2/...` 前车不会被视作同车道前车。
- 自定义安全补车禁止在自车同车道前方补车，运行中前车会自然变少。

现在 `straight_4lane` 默认启用了“同物理车道走廊”前车 profile：会跨 `entry -> straight_* -> buffer` 统计同 lane 编号的前后车，并让自车出生时前方保留一辆 20-45m 的前车。

#### 使用接口
```python
# 获取单场景训练环境
env = get_single_scene_training_env("straight_4lane")

# 获取单场景验证环境（不同的种子区间）
env = get_single_scene_validation_env("roundabout")

# 自定义额外配置
env = get_single_scene_training_env(
    "t_intersection",
    extra_config={"traffic_density": 0.1}
)

# 切到官方MetaDrive交通管理器
env = get_single_scene_training_env(
    "straight_4lane",
    extra_config={
        "single_scene": {"traffic_backend": "official"},
        "traffic_density": 0.2,
        "traffic_mode": "respawn",
        "random_traffic": True,
    },
)

# 命令行快速调密度 / 切换官方后端
python run_single_scene.py --scene straight_4lane --traffic-density 0.2
python run_single_scene.py --scene straight_4lane --traffic-backend official --traffic-density 0.2
```

---

## 配置参数详解

### 基础配置 (DEFAULT_CONFIG)

#### 环境难度
```python
"accident_prob": 0.8,        # 事故/障碍物场景概率（控制锥桶、故障车、路障等静态障碍密度）
"traffic_density": 0.05,     # 交通密度（车辆生成概率）
```

#### 终止条件
```python
"crash_vehicle_done": False,  # 碰撞车辆后是否结束episode
"crash_object_done": False,   # 碰撞障碍物后是否结束episode
"out_of_road_done": True,     # 偏离道路后是否结束episode
```
> ⚠️ 注意：在 `warning_budget` 模式下，`out_of_road_done` 会在 step 中被临时覆盖

#### 奖励设置
```python
"success_reward": 10.0,       # 成功到达目的地的奖励
"driving_reward": 1.0,        # 持续驾驶的奖励（鼓励前进）
"speed_reward": 0.1,          # 速度奖励（鼓励保持合理速度）
```

#### 惩罚设置（取负值加入奖励）
```python
"out_of_road_penalty": 5.0,   # 偏离道路的惩罚
"crash_vehicle_penalty": 1.0, # 碰撞车辆的惩罚
"crash_object_penalty": 1.0,  # 碰撞障碍物的惩罚
```

#### 成本设置（用于 Safe RL 约束优化）
```python
"crash_vehicle_cost": 1.0,    # 碰撞车辆的成本
"crash_object_cost": 1.0,     # 碰撞障碍物的成本
"out_of_road_cost": 1.0,      # 偏离道路的成本
```

---

### 风险场配置

#### 总开关与尺度
```python
"use_risk_field_cost": False,              # 是否启用风险场
"risk_field_max_distance": 50.0,           # 感知半径（米）

# 成本映射（linear_scale 模式）
"risk_field_cost_scale": 4.0,              # 单步最大事件等价cost
"risk_field_raw_clip": 1.0,                # 原始风险总和上限

# 成本组合
"risk_field_event_cost_weight": 1.0,       # 离散事件成本权重
"risk_field_cost_combine": "max",          # 组合策略：max/sum/risk_only/event_only
```

#### 风险组件权重
```python
# 道路几何风险
"risk_field_boundary_weight": 2.0,         # 道路边界
"risk_field_lane_weight": 0.2,             # 车道线（较低以避免限制变道）
"risk_field_offroad_weight": 2.0,          # 偏离道路

# 动态物体风险
"risk_field_vehicle_weight": 1.0,          # 周围车辆
"risk_field_object_weight": 0.8,           # 静态障碍物

# 预测性指标
"risk_field_headway_weight": 0.0,          # 车头时距（默认关闭）
"risk_field_ttc_weight": 0.0,              # 碰撞时间TTC（默认关闭）
```

#### 形状参数 (Sigma)
```python
# 道路几何 sigma
"risk_field_boundary_sigma": 0.5,          # 边界风险扩散范围（米）
"risk_field_lane_edge_sigma": 0.5,         # 车道线风险扩散范围

# 车辆风险 sigma（超高斯分布）
"risk_field_vehicle_longitudinal_sigma": 3.0,  # 纵向扩散（前后）
"risk_field_vehicle_lateral_sigma": 1.8,       # 横向扩散（左右）
"risk_field_vehicle_beta": 2.0,                # 超高斯指数（2=标准高斯）

# 动态风险调整
"risk_field_vehicle_dynamic_sigma_scale": 0.6, # 速度差对sigma的影响系数
"risk_field_vehicle_dynamic_alpha": 0.9,       # 前车/后车风险非对称性
"risk_field_vehicle_min_dynamic_sigma": 0.5,   # 动态sigma最小值

# 障碍物 sigma
"risk_field_object_longitudinal_sigma": 2.0,
"risk_field_object_lateral_sigma": 0.6,
"risk_field_object_beta": 2.0,
```

#### 车道线类型权重
```python
"risk_field_broken_line_factor": 0.05,   # 虚线（允许变道，几乎不惩罚）
"risk_field_solid_line_factor": 0.60,    # 实线（禁止变道，中等惩罚）
"risk_field_boundary_line_factor": 1.0,  # 边界线（道路边缘，高惩罚）
"risk_field_oncoming_line_factor": 1.50, # 对向线（逆行，最高惩罚）
```

#### 偏离道路参数
```python
"risk_field_offroad_cost": 1.0,          # 偏离道路基准成本
"risk_field_offroad_sigma": 1.0,         # 风险带扩散范围
"risk_field_on_lane_margin": 0.05,       # "在道路上"判定的容差
```

#### 预测性指标阈值
```python
"risk_field_headway_time_threshold": 1.2,    # 车头时距阈值（秒）
"risk_field_ttc_threshold": 3.0,             # TTC阈值（秒）
"risk_field_min_speed": 0.5,                 # 最小速度（低于此跳过计算）

# 成本裁剪
"risk_field_headway_cost_clip": 3.0,         # Headway成本上限
"risk_field_ttc_cost_clip": 3.0,             # TTC成本上限
```

---

### 偏离道路宽容模式配置

```python
"out_of_road_mode": "legacy",                # 模式选择
"out_of_road_warning_limit": 5,              # warning次数上限
"out_of_road_warning_penalty": 1.0,          # 每次warning的奖励惩罚
"out_of_road_warning_cost": 1.0,             # 每次warning的成本
"out_of_road_recovery_steps": 15,            # 纠正步数
"out_of_road_terminate_after_budget": True,  # 超出预算后终止
```

---

### IDM 调试策略配置

```python
"enable_idm_lane_change": True,              # IDM自车是否允许变道
"disable_idm_deceleration": False,           # 是否禁用IDM减速逻辑
```
> 💡 这些参数用于调试 IDM 策略的行为，通常在训练 learned policy 时不使用

---

## 使用示例

### 1. 创建训练环境
```python
from env import get_training_env

# 使用默认配置
env = get_training_env()

# 自定义配置
env = get_training_env(extra_config={
    "traffic_density": 0.1,
    "use_risk_field_cost": True,
    "risk_field_cost_scale": 2.0,
})

# 启用手控和渲染（用于调试）
env = get_training_env({
    "manual_control": True,
    "use_render": True,
})
```

### 2. 创建验证环境
```python
from env import get_validation_env

env = get_validation_env()
```

### 3. 创建单场景环境
```python
from env import get_single_scene_training_env, get_single_scene_validation_env

# 训练环境
env = get_single_scene_training_env("straight_4lane")
env = get_single_scene_training_env("ramp_merge")
env = get_single_scene_training_env("t_intersection")
env = get_single_scene_training_env("roundabout")
env = get_single_scene_training_env("lane_change_bottleneck")

# 验证环境（使用不同的种子区间）
val_env = get_single_scene_validation_env("roundabout")

# 自定义配置
env = get_single_scene_training_env(
    "t_intersection",
    extra_config={"traffic_density": 0.15}
)
```

### 4. 标准训练循环
```python
import numpy as np
from env import get_training_env

env = get_training_env({
    "use_risk_field_cost": True,
    "out_of_road_mode": "warning_budget",
})

obs, info = env.reset()
episode_reward = 0
episode_cost = 0

while True:
    # 替换为你的策略动作
    action = np.array([0.0, 0.0])  # [加速度, 转向角]
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_reward += reward
    episode_cost += info.get("cost", 0.0)
    
    # 访问风险场详细信息
    if info.get("risk_field_cost") is not None:
        print(f"风险场成本: {info['risk_field_cost']:.3f}")
        print(f"  - 边界风险: {info.get('risk_field_boundary_cost', 0):.3f}")
        print(f"  - 车辆风险: {info.get('risk_field_vehicle_cost', 0):.3f}")
        print(f"  - 事件等价成本: {info['risk_field_event_equivalent_cost']:.3f}")
    
    # 访问偏离道路状态
    if info.get("out_of_road_warning_triggered"):
        print(f"⚠️  触发warning! 累计次数: {info['out_of_road_warning_count']}")
    
    done = terminated or truncated
    if done:
        print(f"Episode结束! 总奖励: {episode_reward:.2f}, 总成本: {episode_cost:.2f}")
        break

env.close()
```

### 5. 渲染与可视化
```python
env = get_training_env({"use_render": True})
obs, _ = env.reset()

for _ in range(1000):
    action = np.array([0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 俯视渲染（目标智能体朝上）
    env.render(mode="topdown", target_agent_heading_up=True)
    
    if terminated or truncated:
        break

env.close()
```

---

## 架构设计

### 类继承关系
```
SafeMetaDriveEnv (MetaDrive原生基类)
    │
    ├── SafeMetaDriveEnv_mini (本文件实现)
    │   ├── 集成风险场成本计算
    │   ├── 实现偏离道路宽容模式
    │   └── 重写 cost_function 和 step
    │
    └── SafeMetaDriveSingleSceneEnv (本文件实现)
        ├── 固定几何单场景地图
        ├── 程序化生成5种场景
        └── 可控的随机化机制
```

### 关键方法重写

#### SafeMetaDriveEnv_mini

**`cost_function(vehicle_id)`**
```python
# 1. 调用父类计算离散事件成本
event_cost, step_info = super().cost_function(vehicle_id)

# 2. 如果启用风险场，计算连续风险成本
if use_risk_field_cost:
    risk_cost, risk_info = risk_calculator.calculate(self, vehicle)
    weighted_risk = _risk_field_event_equivalent_cost(risk_cost)
    final_cost = _combine_event_and_risk_cost(event_cost, weighted_risk)
else:
    final_cost = event_cost

# 3. 更新累计成本和info
step_info["cost"] = final_cost
step_info["total_cost"] = self.episode_cost
return final_cost, step_info
```

**`step(actions)`**
```python
# 如果是 legacy 模式，直接调用父类
if not _is_warning_budget_mode():
    return super().step(actions)

# 如果是 warning_budget 模式
original_out_of_road_done = self.config["out_of_road_done"]
self.config["out_of_road_done"] = False  # 临时禁用立即终止

obs, reward, terminated, truncated, info = super().step(actions)

self.config["out_of_road_done"] = original_out_of_road_done  # 恢复

# 处理偏离道路逻辑
if raw_out_of_road and not severe_crash:
    if not _out_of_road_warning_active:
        # 首次检测到偏离，启动warning状态
        _out_of_road_warning_count += 1
        _out_of_road_recovery_remaining = recovery_steps
    
    if _budget_exhausted or _timeout:
        # 超出预算或超时，终止episode
        terminated = True
        reward = -out_of_road_penalty
        cost = out_of_road_cost
    else:
        # 仍在纠正期内，给予warning惩罚
        if warning_triggered:
            reward -= out_of_road_warning_penalty
            cost = out_of_road_warning_cost
        
        _out_of_road_recovery_remaining -= 1

return obs, reward, terminated, truncated, info
```

**`reset()`**
```python
obs, info = super().reset()

# 重置episode成本计数器
self.episode_cost = 0

# 重置偏离道路状态
_reset_out_of_road_warning_state()

# 注入初始状态到info
info["total_cost"] = 0.0
return obs, _annotate_out_of_road_info(info)
```

#### SafeMetaDriveSingleSceneEnv

**`reset()`**
```python
obs, info = super().reset()

# 应用单场景随机化
_apply_single_scene_randomization()

# 重新观察和计算初始状态
agent = self.agents[agent_id]
obs = self.observations[agent_id].observe(agent)

reward, reward_info = self.reward_function(agent_id)
terminated, done_info = self.done_function(agent_id)
_, cost_info = self.cost_function(agent_id)

info.update(reward_info)
info.update(done_info)
info.update(cost_info)
info["total_cost"] = 0.0

return obs, info
```

**`_apply_single_scene_randomization()`**
```python
# 1. 生成确定性种子
rng = _next_single_scene_episode_rng()

# 2. 随机采样自车初始状态
lane_index, lane, spawn_longitude, spawn_speed = _sample_ego_spawn(rng)
destination = _sample_ego_destination(rng)

# 3. 更新自车配置并重置
ego_config.update({
    "spawn_lane_index": lane_index,
    "spawn_longitude": spawn_longitude,
    "spawn_velocity": [spawn_speed, 0.0],
    "destination": destination,
})
agent.reset(vehicle_config=ego_config)

# 4. 生成交通流
_respawn_single_scene_traffic(rng, occupied_by_ego)
```

### 地图生成器

#### SingleScenePGMap
负责程序化生成5种场景的地图结构：

```python
def _generate(self):
    scene_name = self.config["scene_name"]
    
    if scene_name == "straight_4lane":
        self._build_straight_4lane(...)
    elif scene_name == "ramp_merge":
        self._build_ramp_merge(...)
    elif scene_name == "t_intersection":
        self._build_t_intersection(...)
    elif scene_name == "roundabout":
        self._build_roundabout(...)
    elif scene_name == "lane_change_bottleneck":
        self._build_lane_change_bottleneck(...)
    
    # 记录场景元数据（道路别名、目的地节点等）
    self._finalize_scene(scene_name, road_aliases, destinations, spawn_road)
```

#### SingleScenePGMapManager
管理单场景地图的加载和重置，确保每次 reset 都使用相同的地图结构。

---

## 调试与可视化

奖励塑形调参时，建议同时参考：

- [docs/REWARD_SHAPING_OBSERVABILITY.md](docs/REWARD_SHAPING_OBSERVABILITY.md)

这份文档专门整理了当前 `metadrive_env.py` 链路下哪些指标是真正有效的、哪些旧字段不要拿来判断，以及“出车道 / 转向震荡 / 速度 / 风险场 / 周车风险”五类问题的固定观察口径。

### 1. Info 字典完整结构

每次 `step` 返回的 `info` 包含以下关键字段：

#### 基础信息
```python
{
    "cost": 0.5,                    # 当前步成本
    "total_cost": 2.3,              # 累计episode成本
    "episode_reward": 15.7,         # 累计episode奖励
    "step_reward": 0.8,             # 当前步奖励
    "success": False,               # 是否成功到达目的地
    "crash_vehicle": False,         # 是否碰撞车辆
    "crash_object": False,          # 是否碰撞障碍物
    "out_of_road": False,           # 是否偏离道路
}
```

#### 风险场分解信息（当启用时）
```python
{
    # 原始风险值
    "risk_field_cost": 0.45,                    # 加权前的总风险
    "risk_field_road_cost": 0.12,               # 道路几何风险总和
    "risk_field_boundary_cost": 0.08,           # 边界风险
    "risk_field_lane_cost": 0.02,               # 车道线风险
    "risk_field_offroad_cost": 0.02,            # 偏离道路风险
    "risk_field_vehicle_cost": 0.25,            # 车辆风险
    "risk_field_object_cost": 0.05,             # 障碍物风险
    "risk_field_headway_cost": 0.0,             # 车头时距风险
    "risk_field_ttc_cost": 0.03,                # TTC风险
    
    # 映射后的成本
    "risk_field_weighted_cost": 0.45,           # 事件等价成本
    "risk_field_event_equivalent_cost": 0.45,   # 同上（兼容字段）
    
    # 配置参数
    "risk_field_cost_scale": 4.0,               # 尺度参数
    "risk_field_collision_equivalent_cost": 1.0,# 碰撞等价成本
    "risk_field_cost_mapping": "linear_scale",  # 映射模式
    "risk_field_cost_transform": "event_squash",# 变换函数
    "risk_field_cost_combine": "max",           # 组合策略
}
```

#### 偏离道路状态
```python
{
    "out_of_road_mode": "warning_budget",       # 当前模式
    "out_of_road_warning_count": 2,             # 已触发warning次数
    "out_of_road_recovery_remaining": 8,        # 剩余纠正步数
    "out_of_road_budget_exhausted": False,      # 预算是否耗尽
    "out_of_road_warning_triggered": True,      # 本步是否触发warning
    "out_of_road_warning_active": True,         # 是否处于warning状态
    "out_of_road_timeout_terminated": False,    # 是否因超时而终止
}
```

### 2. WandB 日志记录

在训练脚本中，可以记录详细的风险信息：

```python
import wandb

# 记录每个风险组件的贡献
wandb.log({
    "cost/risk_field_total": info["risk_field_cost"],
    "cost/risk_field_boundary": info["risk_field_boundary_cost"],
    "cost/risk_field_vehicle": info["risk_field_vehicle_cost"],
    "cost/risk_field_headway": info["risk_field_headway_cost"],
    "cost/event_equivalent": info["risk_field_event_equivalent_cost"],
    "safety/out_of_road_warnings": info["out_of_road_warning_count"],
})
```

### 3. 风险场计算器独立调用

可以在不运行完整环境的情况下测试风险场计算：

```python
from risk_field import RiskFieldCalculator
from env import get_training_env

# 创建环境和计算器
env = get_training_env()
calculator = RiskFieldCalculator(env.config)

# 重置环境
obs, _ = env.reset()

# 手动计算风险
vehicle = env.agents["default_agent"]
risk_cost, risk_info = calculator.calculate(env, vehicle)

print(f"总风险: {risk_cost:.3f}")
print(f"边界风险: {risk_info['boundary_cost']:.3f}")
print(f"车辆风险: {risk_info['vehicle_cost']:.3f}")

env.close()
```

### 4. Topdown 渲染调试

```python
env = get_training_env({"use_render": True})
obs, _ = env.reset()

for step in range(500):
    action = np.array([0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 俯视渲染，自车朝上
    env.render(mode="topdown", target_agent_heading_up=True)
    
    # 在控制台打印关键信息
    if step % 50 == 0:
        print(f"Step {step}: reward={reward:.2f}, cost={info['cost']:.2f}")
        if info.get("risk_field_cost"):
            print(f"  风险场: {info['risk_field_cost']:.3f}")
    
    if terminated or truncated:
        print(f"Episode结束于step {step}")
        break

env.close()
```

### 5. 常见调试技巧

#### 检查风险场是否生效
```python
env = get_training_env({"use_risk_field_cost": True})
obs, info = env.reset()

# 检查info中是否有风险场字段
assert "risk_field_cost" in info, "风险场未正确初始化"
assert "risk_field_boundary_cost" in info, "边界风险未计算"
```

#### 测试偏离道路宽容模式
```python
env = get_training_env({
    "out_of_road_mode": "warning_budget",
    "out_of_road_warning_limit": 3,
    "out_of_road_recovery_steps": 10,
})

obs, info = env.reset()
print(f"初始warning计数: {info['out_of_road_warning_count']}")  # 应为0

# 故意偏离道路多次，观察warning计数变化
for _ in range(100):
    action = np.array([0.0, 0.5])  # 大幅转向
    obs, reward, terminated, truncated, info = env.step(action)
    
    if info.get("out_of_road_warning_triggered"):
        print(f"Step {_}: 触发warning #{info['out_of_road_warning_count']}")
    
    if terminated:
        print(f"Episode终止! 原因: budget_exhausted={info['out_of_road_budget_exhausted']}")
        break
```

#### 验证单场景随机化的确定性
```python
# 显式传入相同seed，应该得到相同的初始状态
env1 = get_single_scene_training_env("straight_4lane")
obs1, info1 = env1.reset(seed=12345)
pos1 = env1.agents["default_agent"].position

env2 = get_single_scene_training_env("straight_4lane")
obs2, info2 = env2.reset(seed=12345)
pos2 = env2.agents["default_agent"].position

print(f"位置1: {pos1}")
print(f"位置2: {pos2}")
print(f"是否相同: {np.allclose(pos1, pos2)}")  # 应为True

env1.close()
env2.close()
```

---

## 常见问题 (FAQ)

### Q1: 如何启用风险场成本？
```python
env = get_training_env({
    "use_risk_field_cost": True,
    "risk_field_cost_scale": 4.0,  # 调整尺度
})
```

### Q2: 如何调整风险场的敏感度？
- **增大权重** → 更敏感，更早规避风险
- **减小 sigma** → 风险场更集中，只在近距离生效
- **增大 sigma** → 风险场更宽泛，远距离就开始预警

### Q3: warning_budget 模式下，如何知道智能体是否充分利用了纠正机会？
检查 `info["out_of_road_recovery_remaining"]`：
- 如果经常为 0 → 智能体未能及时纠正
- 如果经常接近初始值 → 智能体很少偏离或快速纠正

### Q4: 单场景环境的随机化范围在哪里配置？
在 `SINGLE_SCENE_COMMON_RANDOMIZATION` 和各场景的 `_SINGLE_SCENE_RANDOMIZATION_OVERRIDES` 字典中修改：
```python
"ego_longitude_range": (5.0, 30.0),  # 自车纵向范围
"traffic_count_mode": "density",     # 默认按 traffic_density 控制周车数量
```
默认推荐通过顶层 `traffic_density` 调周车密度，单场景会在 `traffic_roads` 覆盖的道路上按 10m 间隔生成候选点，并按 `ceil(traffic_density * candidate_count)` 抽样。旧版 `traffic_vehicle_count_range` 只在 `traffic_count_mode="range"` 时生效。

### Q5: 如何复现特定的随机化结果？
单场景环境默认每次 `reset()` 都推进 episode 级随机种子，因此同一个环境连续 reset 会得到不同的自车/周车初始化。需要复现时显式传入 seed：
```python
obs, info = env.reset(seed=12345)
```
对同一个场景重复使用相同 seed，会复现相同的自车出生点、目标点和周车 slot 布局。

### Q6: 风险场计算会影响性能吗？
是的，风险场计算涉及遍历所有周围车辆和障碍物，并进行指数运算。建议：
- 仅在训练时启用 (`use_risk_field_cost=True`)
- 评估时可禁用以加速
- 调整 `risk_field_max_distance` 限制感知范围

---

## 总结

`env.py` 提供了 SafeMetaDrive 项目的**核心环境抽象层**，通过三大创新功能显著提升了安全强化学习研究的便利性和有效性：

1. **风险场成本系统** - 从离散事件到连续势场，提供更细粒度的安全约束
2. **偏离道路宽容模式** - 从零容忍到有限纠错，平衡安全性与探索性
3. **单场景随机化环境** - 从完全随机到可控变化，提升实验可复现性

通过灵活的配置系统和清晰的接口设计，研究者可以快速原型化各种安全驾驶场景，并深入分析智能体的决策行为。

---

## 参考资源

- **MetaDrive 官方文档**: https://github.com/metadriverse/metadrive
- **FSRL 框架文档**: https://github.com/liuzuxin/fsrl
- **风险场实现细节**: 参见项目中的 `risk_field.py`
- **训练脚本示例**: 参见 `train_ppol.py` 和 `train_sacl.py`
