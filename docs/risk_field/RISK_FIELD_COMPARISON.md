# 风险场版本对比与回退指南

## 📋 文档说明

本文档详细对比了**旧版风险场**（`abstract.py` 中的 `_cost()` 方法）和**新版风险场**（`risk_field.py` 中的 `RiskFieldCalculator` 类）的差异，并提供回退到旧版的指导，以确保能够复现之前的实验效果。

---

## 🎯 为什么需要回退？

在开发过程中,新版风险场引入了多项改进(如动态道路几何、预测性安全指标等),但这些改动可能影响训练效果的可比性。为确保**严格复现之前的实验结果**,需要暂时回退到旧版实现。

**关键原因:**
1. **参数敏感性**:新版的7个组件权重需要重新调优
2. **算法收敛性**:Headway/TTC 的引入改变了成本分布
3. **实验可比性**:确保与历史 baseline 公平对比

---

## 📊 核心差异对比总览

| **维度** | **旧版 (Legacy)** | **新版 (Current)** | **影响程度** |
|---------|------------------|-------------------|------------|
| **代码位置** | `abstract.py` L740-821 | `risk_field.py` 完整类 | 🔴 高 |
| **架构设计** | 单一函数 `_cost()` | 模块化类 `RiskFieldCalculator` | 🟡 中 |
| **道路几何** | ❌ 硬编码坐标 | ✅ 动态 API 获取 | 🔴 高 |
| **坐标系** | ⚠️ 手动旋转(有bug修正) | ✅ Frenet 优先 | 🟡 中 |
| **风险组件** | 4个 | 7个 (+3个新增) | 🔴 高 |
| **车辆尺寸** | ❌ 固定 5m×2m | ✅ 动态读取 | 🟢 低 |
| **预测性指标** | ❌ 无 Headway/TTC | ✅ 行业标准指标 | 🔴 高 |
| **参数管理** | ❌ 全部硬编码 | ✅ 配置字典 | 🟢 低 |
| **数值稳定性** | ⚠️ 可能溢出 | ✅ 完善保护 | 🟢 低 |

---

## 🔬 详细公式对比

### 1️⃣ 道路边界风险 (Boundary Risk)

#### **旧版实现**
```python
# abstract.py L780-783
y_boundary = [-2, 14]  # ⚠️ 硬编码!仅适用于特定地图
σb = 1                  # 硬编码标准差
Ab = 1                  # 硬编码场强系数

Eb = Σ Ab * exp(-(y_ego - boundary)² / σb²)
   = exp(-(y_ego + 2)²) + exp(-(y_ego - 14)²)
```

**特点:**
- ❌ 仅考虑横向距离 `y_ego`
- ❌ 假设固定4车道高速公路(y范围 -2 到 14)
- ❌ 无法适配弯道或不同宽度的道路

---

#### **新版实现**
```python
# risk_field.py L195-210
left_dist, right_dist = vehicle.dist_to_left_side, vehicle.dist_to_right_side
min_boundary_distance = min(left_dist, right_dist)

if min_boundary_distance < 0:
    # 已越界:边缘带衰减
    E_boundary = exp(-|d|² / (2 * σ_b²))
else:
    # 在道路内:标准高斯
    E_boundary = exp(-d² / (2 * σ_b²))

# 可配置参数
σ_b = config.get("risk_field_boundary_sigma", 0.75)  # 默认 0.75m
w_b = config.get("risk_field_boundary_weight", 1.0)
```

**改进:**
- ✅ 从 MetaDrive API 实时获取边界距离
- ✅ 区分"接近边界"和"已越界"两种状态
- ✅ 适配任意道路几何(直线/弯道/变宽)

---

#### **回退建议**
如果要复现旧版效果,需要:
```python
# 在 env.py 中硬编码边界位置
Y_BOUNDARY = [-2, 14]  # 仅适用于原始测试地图

def _legacy_boundary_risk(y_ego):
    """旧版边界风险计算"""
    Eb = 0.0
    for boundary in Y_BOUNDARY:
        Eb += np.exp(-(y_ego - boundary) ** 2)
    return Eb
```

---

### 2️⃣ 车道线风险 (Lane Edge Risk)

#### **旧版实现**
```python
# abstract.py L785-788
y_lane = [2, 6, 10]  # ⚠️ 硬编码3条车道线!
σl = 1
Al = 0.1

El = Σ Al * exp(-(y_ego - lane)² / σl²)
   = 0.1 * [exp(-(y_ego-2)²) + exp(-(y_ego-6)²) + exp(-(y_ego-10)²)]
```

**特点:**
- ❌ 假设固定3条车道线(4车道公路)
- ❌ 鼓励车辆靠近车道中心线(而非保持在车道内)
- ❌ 无法处理变宽车道

---

#### **新版实现**
```python
# risk_field.py L165-173
lane_longitudinal, lateral_offset = lane.local_coordinates(ego_pos)
lane_width = lane.width_at(longitudinal)

# 计算到最近车道边缘的间隙
edge_gap = max(lane_width / 2.0 - abs(lateral_offset), 0.0)

# 一维高斯风险:越靠近边缘风险越高
E_lane = exp(-edge_gap² / (2 * σ_l²))

# 可配置参数
σ_l = config.get("risk_field_lane_edge_sigma", 0.75)  # 默认 0.75m
w_l = config.get("risk_field_lane_weight", 0.1)       # 降低权重
```

**改进:**
- ✅ 使用 Frenet 坐标系,自动适配弯道
- ✅ 动态获取车道宽度(支持变宽车道)
- ✅ 惩罚偏离车道边缘,而非偏离中心线
- ✅ 降低权重(0.1 vs 旧版隐式 0.1*3=0.3)

---

#### **回退建议**
```python
# 在 env.py 中硬编码车道线位置
Y_LANE = [2, 6, 10]

def _legacy_lane_risk(y_ego):
    """旧版车道线风险计算"""
    El = 0.0
    for lane in Y_LANE:
        El += 0.1 * np.exp(-(y_ego - lane) ** 2)
    return El
```

---

### 3️⃣ 周围车辆风险 (Vehicle Risk)

这是**最复杂的差异部分**,涉及坐标系、车辆尺寸、动态势场等多个方面。

---

#### **旧版实现 - 静态势场**
```python
# abstract.py L790-812
L_obs = 5    # ⚠️ 固定障碍物长度
W_obs = 2    # ⚠️ 固定障碍物宽度
kx = 1       # 纵向扩散系数
ky = 0.8     # 横向扩散系数
beta = 2     # 超高斯指数

σx = kx * L_obs  # = 5
σy = ky * W_obs  # = 1.6

# ⚠️ 坐标旋转(存在bug,需手动取反航向角)
h_obs = -h_obs  # Bug 修正!
x_rel = (x_ego - x_obs)*cos(h_obs) - (y_ego - y_obs)*sin(h_obs) + x_obs
y_rel = (x_ego - x_obs)*sin(h_obs) + (y_ego - y_obs)*cos(h_obs) + y_obs

# 超高斯分布
Usta = Asta * exp(-((x_rel-x_obs)²/σx²)^beta - ((y_rel-y_obs)²/σy²)^beta)
# Asta = 1 (硬编码)
```

**问题:**
- ❌ 坐标旋转公式有误(需 `h_obs = -h_obs` 修正)
- ❌ 固定车辆尺寸,不考虑车型差异
- ❌ 以主车为中心的相对坐标(易受主车航向影响)

---

#### **新版实现 - 静态部分**
```python
# risk_field.py L647-656
# 以障碍物为中心的相对坐标(更稳定)
obs_long, obs_lat = _oriented_delta(other, ego_pos, ref_heading)

# 动态获取车辆尺寸
other_length = other.top_down_length  # 非固定值
other_width = other.top_down_width

# 超髙斯分布
E_static = exp(-((long²/σ_long²)^beta + (lat²/σ_lat²)^beta))

# 可配置参数
σ_long = config.get("risk_field_vehicle_longitudinal_sigma", 5.0)
σ_lat = config.get("risk_field_vehicle_lateral_sigma", 1.6)
beta = config.get("risk_field_vehicle_beta", 2.0)
```

**改进:**
- ✅ 使用障碍物自身朝向建立局部坐标系(避免旋转bug)
- ✅ 动态读取车辆实际尺寸
- ✅ 可配置的 sigma 和 beta 参数

---

#### **旧版实现 - 动态势场**
```python
# abstract.py L800-812
kv = 2              # 速度扩散系数
alpha = 0.9         # 不对称性系数

σv = kv * |v_obs - v_ego|  # 动态sigma与速度差成正比

relv = 1 if v_obs >= v_ego else -1

# 复杂的非对称公式
Udyn = Adyn * exp(-(x_rel-x_obs)²/σv² - (y_rel-y_obs)²/σy²) 
       / (1 + exp(-relv * (x_rel - x_obs - alpha * L_obs * relv)))
# Adyn = 1 (硬编码)
```

**问题:**
- ❌ 公式复杂,缺乏物理解释
- ❌ sigmoid 分母的作用不明确
- ❌ 速度差直接作为 sigma,可能导致数值不稳定(速度相同时 σv=0)

---

#### **新版实现 - 动态部分**
```python
# risk_field.py L658-677
other_forward_speed = _forward_speed(other, ref_heading)
speed_delta = abs(other_forward_speed - ego_forward_speed)

# 动态sigma与速度差成正比,但有下限保护
dynamic_sigma = max(k_scale * speed_delta, σ_min)
# k_scale = 2.0, σ_min = 0.5m

lateral_sigma = max(σ_lat, EPS)

# 动态风险指数
dynamic_exponent = -(long²/dynamic_sigma² + lat²/lateral_sigma²)
E_dynamic_raw = exp(max(dynamic_exponent, -80.0))  # 数值稳定性保护

# 非对称性:根据相对速度方向调整
relv = 1.0 if other_forward_speed >= ego_forward_speed else -1.0
alpha = config.get("risk_field_vehicle_dynamic_alpha", 0.9)
sigmoid_arg = -relv * (long - alpha * other_length * relv)
E_dynamic = E_dynamic_raw / (1.0 + exp(clip(sigmoid_arg, -60.0, 60.0)))

# 总车辆风险 = 静态 + 动态
E_vehicle = E_static + E_dynamic
```

**改进:**
- ✅ 限制最小 sigma 避免除零(`σ_min=0.5`)
- ✅ 清晰的非对称性建模(前车风险更大)
- ✅ 分离静态和动态成分,便于调试
- ✅ 指数运算限制下限(-80)避免溢出

---

#### **回退建议**
如果要严格复现旧版,需要恢复以下逻辑:
```python
def _legacy_vehicle_risk(x_ego, y_ego, v_ego, vehicles_obs):
    """旧版车辆风险计算(静态+动态)"""
    L_obs, W_obs = 5.0, 2.0
    kx, ky, kv = 1.0, 0.8, 2.0
    alpha, beta = 0.9, 2.0
    Asta, Adyn = 1.0, 1.0
    
    Ustas, Udyns = [], []
    
    for vehicle in vehicles_obs[1:]:  # 跳过主车
        x_obs, y_obs, v_obs, h_obs = vehicle
        
        σx = kx * L_obs
        σy = ky * W_obs
        σv = kv * abs(v_obs - v_ego)
        
        # ⚠️ Bug 修正:手动取反航向角
        h_obs = -h_obs
        
        # 坐标旋转
        x_rel = (x_ego - x_obs)*np.cos(h_obs) - (y_ego - y_obs)*np.sin(h_obs) + x_obs
        y_rel = (x_ego - x_obs)*np.sin(h_obs) + (y_ego - y_obs)*np.cos(h_obs) + y_obs
        
        # 静态势场
        Usta = Asta * np.exp(
            -((x_rel - x_obs)**2 / σx**2)**beta 
            - ((y_rel - y_obs)**2 / σy**2)**beta
        )
        
        # 动态势场
        relv = 1 if v_obs >= v_ego else -1
        if σv > 1e-6:  # 避免除零
            Udyn = Adyn * np.exp(
                -(x_rel - x_obs)**2 / σv**2 - (y_rel - y_obs)**2 / σy**2
            ) / (1 + np.exp(-relv * (x_rel - x_obs - alpha * L_obs * relv)))
        else:
            Udyn = 0.0
        
        Ustas.append(Usta)
        Udyns.append(Udyn)
    
    return sum(Ustas), sum(Udyns)
```

---

### 4️⃣ 偏离道路风险 (Off-road Risk) ⭐ **新增组件**

#### **旧版实现**
```python
# ❌ 完全没有此组件
```

---

#### **新版实现**
```python
# risk_field.py L175-181
on_road, offroad_distance = _lane_surface_state(env, ego_pos)

if not on_road:
    # 边缘带衰减模型(避免整片路外区域涂满红色)
    E_offroad = C_offroad * exp(-d² / (2 * σ_offroad²))
    
# 配置化参数
C_offroad = config.get("risk_field_offroad_cost", 1.0)
σ_offroad = config.get("risk_field_offroad_sigma", 1.0)
w_offroad = config.get("risk_field_offroad_weight", 1.0)
```

**物理意义:**
- 仅在道路边缘附近形成风险带(sigma=1m)
- 避免远处路外区域也被标红
- 鼓励车辆保持在可行驶区域内

---

#### **回退建议**
```python
# 如果要回退到旧版,直接将此项设为 0
E_offroad = 0.0
```

---

### 5️⃣ 车头时距风险 (Headway Time Risk) ⭐ **新增组件**

#### **旧版实现**
```python
# ❌ 完全没有此组件
```

---

#### **新版实现**
```python
# risk_field.py L250-267
# 仅针对同车道前车
is_front_vehicle = delta_long > 0
is_same_corridor = abs(delta_lat) <= max(lane_width*0.75, (W_ego+W_other)/2)

if is_front_vehicle and is_same_corridor:
    # 计算纵向间隙(扣除车辆长度)
    front_gap = max(delta_long - (L_ego + L_other)/2, 0)
    
    # 车头时距
    t_headway = front_gap / max(v_ego, v_min)
    
    # 对数惩罚(阈值内)
    if t_headway < t_threshold:
        E_headway = -ln(t_headway / t_threshold)
    else:
        E_headway = 0
    
    # 裁剪到上限
    E_headway = clip(E_headway, 0, cost_clip)

# 配置化参数
t_threshold = config.get("risk_field_headway_time_threshold", 1.2)  # 秒
cost_clip = config.get("risk_field_headway_cost_clip", 3.0)
w_headway = config.get("risk_field_headway_weight", 1.0)
```

**物理意义:**
- 交通安全领域的标准指标
- 1.2秒是行业推荐的最小安全时距
- 对数惩罚:时距越小惩罚增长越快

---

#### **回退建议**
```python
# 如果要回退到旧版,直接将此项设为 0
E_headway = 0.0
```

---

### 6️⃣ 碰撞时间风险 (TTC Risk) ⭐ **新增组件**

#### **旧版实现**
```python
# ❌ 完全没有此组件
```

---

#### **新版实现**
```python
# risk_field.py L269-283
# 仅当前车速度低于主车时
closing_speed = v_ego - v_other

if closing_speed > EPS:
    # 碰撞时间
    TTC = front_gap / closing_speed
    
    # 对数惩罚(阈值内)
    if TTC < TTC_threshold:
        E_ttc = -ln(TTC / TTC_threshold)
    else:
        E_ttc = 0
    
    # 裁剪到上限
    E_ttc = clip(E_ttc, 0, cost_clip)

# 配置化参数
TTC_threshold = config.get("risk_field_ttc_threshold", 3.0)  # 秒
cost_clip = config.get("risk_field_ttc_cost_clip", 3.0)
w_ttc = config.get("risk_field_ttc_weight", 1.0)
```

**物理意义:**
- 预测性安全指标,提前预警潜在碰撞
- 3.0秒是常见的TTC警告阈值
- 结合Headway提供双重保护

---

#### **回退建议**
```python
# 如果要回退到旧版,直接将此项设为 0
E_ttc = 0.0
```

---

### 7️⃣ 静态障碍物风险 (Object Risk)

#### **旧版实现**
```python
# ❌ 未区分车辆和静态障碍物
# 所有物体都用相同的车辆势场公式
```

---

#### **新版实现**
```python
# risk_field.py L307-332
# 专用超髙斯模型(无动态部分,因为障碍物静止)
E_object = Σ exp(-((long²/σ_obj_long²)^beta + (lat²/σ_obj_lat²)^beta))

# 配置化参数(与车辆略有不同)
σ_obj_long = config.get("risk_field_object_longitudinal_sigma", 5.0)
σ_obj_lat = config.get("risk_field_object_lateral_sigma", 1.6)
beta_obj = config.get("risk_field_object_beta", 2.0)
w_object = config.get("risk_field_object_weight", 0.8)
```

**改进:**
- ✅ 单独建模静态障碍物(交通锥、路障等)
- ✅ 略低的权重(0.8 vs 1.0),因为障碍物通常较小
- ✅ 可独立调整参数

---

#### **回退建议**
```python
# 如果要回退到旧版,将障碍物当作普通车辆处理
# 或者直接将此项设为 0
E_object = 0.0
```

---

### 8️⃣ 总风险成本公式

#### **旧版**
```python
# abstract.py L815-820
U_total = Ustas + Udyns + Eb + El
# 简单求和,无权重控制
# 其中:
# - Ustas: 所有车辆的静态势场之和
# - Udyns: 所有车辆的动态势场之和
# - Eb: 道路边界风险
# - El: 车道线风险
```

---

#### **新版**
```python
# risk_field.py L100-112
U_total = w_b * E_boundary 
        + w_l * E_lane 
        + w_offroad * E_offroad
        + w_v * E_vehicle      # 包含静态+动态
        + w_o * E_object
        + w_h * E_headway
        + w_t * E_ttc

# 限制为非负并裁剪到上限
U_total = clip(U_total, 0, 10.0)
```

**改进:**
- ✅ 7个独立组件,每个可单独调权
- ✅ 非负约束(风险不能为负)
- ✅ 上限裁剪避免极端值影响训练

---

#### **回退建议**
```python
# 恢复旧版的简单求和方式
U_total = E_boundary + E_lane + E_vehicle_static + E_vehicle_dynamic
# 注意:不包含 E_offroad, E_headway, E_ttc, E_object
```

---

## 🛠️ 回退实施步骤

### 方案A: 完全回退到旧版(推荐用于严格复现)

1. **备份新版代码**
   ```bash
   cp risk_field.py risk_field_new.py.backup
   cp risk_field_bev.py risk_field_bev_new.py.backup
   ```

2. **从 abstract.py 提取旧版 `_cost()` 方法**
   ```python
   # 在 env.py 中创建 LegacyRiskFieldCalculator 类
   class LegacyRiskFieldCalculator:
       def __init__(self):
           # 硬编码参数
           self.y_boundary = [-2, 14]
           self.y_lane = [2, 6, 10]
           self.L_obs = 5.0
           self.W_obs = 2.0
           # ... 其他硬编码参数
       
       def calculate(self, env, vehicle):
           # 调用旧版 _cost() 逻辑
           return self._cost(env, vehicle), {}
       
       def _cost(self, env, vehicle):
           # 复制 abstract.py L740-821 的完整实现
           ...
   ```

3. **修改 env.py 使用旧版计算器**
   ```python
   # 在 SafeMetaDriveEnv_mini.__init__ 中
   from legacy_risk_field import LegacyRiskFieldCalculator
   self.risk_calculator = LegacyRiskFieldCalculator()
   ```

4. **禁用新增组件**
   - 设置 `E_offroad = 0`
   - 设置 `E_headway = 0`
   - 设置 `E_ttc = 0`
   - 设置 `E_object = 0`(或合并到车辆风险)

---

### 方案B: 渐进式回退(推荐用于调试)

如果不想完全放弃新版的改进,可以**逐步禁用新增组件**:

```python
# 在 env.py 的配置中
config = {
    # 保留新版的核心改进
    "risk_field_boundary_sigma": 0.75,
    "risk_field_lane_edge_sigma": 0.75,
    
    # 禁用新增组件
    "risk_field_offroad_weight": 0.0,    # 禁用偏离道路风险
    "risk_field_headway_weight": 0.0,    # 禁用Headway
    "risk_field_ttc_weight": 0.0,        # 禁用TTC
    "risk_field_object_weight": 0.0,     # 禁用静态障碍物
    
    # 恢复旧版的车辆风险权重
    "risk_field_vehicle_weight": 1.0,
}
```

---

## 📈 预期影响分析

### 回退后的变化

| **指标** | **新版预期** | **旧版预期** | **变化方向** |
|---------|------------|------------|------------|
| **平均成本** | 较低(多组件分散) | 较高(仅4个组件) | ⬆️ 上升 |
| **成本方差** | 较小(平滑) | 较大(尖锐) | ⬆️ 增大 |
| **训练稳定性** | 较好 | 一般 | ⬇️ 略降 |
| **收敛速度** | 较快 | 较慢 | ⬇️ 变慢 |
| **最终性能** | 待验证 | Baseline | ➡️ 对齐 |

### 关键观察点

1. **成本曲线形状**: 旧版的成本波动会更大
2. **碰撞率**: 由于缺少 Headway/TTC,初期碰撞率可能上升
3. **车道保持**: 旧版的车道线风险较弱,可能导致更多偏离

---

## ✅ 验证清单

回退后需要验证以下内容:

- [ ] 成本值范围与旧版实验一致(通常在 0-10 之间)
- [ ] 移除 Headway/TTC 后,跟车行为是否变得激进
- [ ] 硬编码的道路几何是否与测试地图匹配
- [ ] 车辆尺寸固定为 5m×2m 是否合理
- [ ] WandB 日志中的成本分解是否正确

---

## 📝 总结

### 何时使用旧版?
- ✅ 需要严格复现历史实验结果
- ✅ 与论文 baseline 进行公平对比
- ✅ 调试新版引入的问题

### 何时使用新版?
- ✅ 追求更好的安全性表现
- ✅ 需要适配多种地图场景
- ✅ 希望利用预测性安全指标

### 最佳实践
建议在**同一分支**中保留两个版本的实现,通过配置文件切换:
```yaml
# config.yaml
risk_field_version: "legacy"  # 或 "current"
```

这样可以灵活地在两个版本之间切换,既保证实验可比性,又不失新版的优势。

---

## 🔗 相关文档

- [风险场计算算法规范](./RISK_FIELD_SPEC.md)
- [MetaDrive集成指南](./METADRIVE_INTEGRATION.md)
- [训练配置说明](./TRAINING_CONFIG.md)

---

**最后更新**: 2026-04-11  
**维护者**: safeRL-metadrive 团队
