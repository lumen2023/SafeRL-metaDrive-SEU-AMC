# 车道线类型详解与风险惩罚对比

## 📋 概述

在MetaDrive中，车道线不仅是视觉元素，更是**交通规则和安全约束**的体现。不同类型的车道线对应不同的**碰撞掩码（Collision Mask）**和**风险惩罚权重**。

---

## 🎯 四种车道线类型对比

| **类型** | **英文名称** | **风险权重** | **物理含义** | **是否允许换道** | **碰撞检测** |
|---------|------------|------------|------------|---------------|------------|
| **虚线** | `BROKEN` | 0.05 | 同向车道分隔线 | ✅ 允许 | ❌ 无碰撞 |
| **实线** | `SOLID/CONTINUOUS` | 0.60 | 禁止变道分隔线 | ❌ 禁止 | ⚠️ 软碰撞 |
| **边界** | `BOUNDARY/SIDE` | 1.00 | 道路边缘/路肩 | ❌ 禁止 | ✅ 硬碰撞 |
| **对向线** | `ONCOMING/YELLOW` | 1.50 | 对向车道分隔线 | ❌ 严禁 | ✅ 硬碰撞+逆行惩罚 |

---

## 🔬 详细解析

### 1️⃣ **虚线 (BROKEN)** - 权重 0.05

#### **定义位置**
```python
# metadrive/constants.py L286
PGLineType.BROKEN = MetaDriveType.LINE_BROKEN_SINGLE_WHITE
```

#### **视觉特征**
- **颜色**: 白色或灰色 (`#cbd5e1`)
- **样式**: 短划线 `(0, (4.0, 4.0))` - 4像素线 + 4像素间隔
- **线宽**: 0.65像素

#### **物理意义**
- ✅ **允许换道**：车辆可以安全跨越
- ✅ **无碰撞检测**：穿过虚线不会触发碰撞
- 🟢 **极低风险惩罚**：仅作为视觉参考，几乎不惩罚

#### **典型场景**
- 高速公路同向车道之间的分隔线
- 城市道路的多车道分隔

#### **代码实现**
```python
# risk_field_bev.py L637-643
if "broken" in line_text:
    return {
        "color": "#cbd5e1",              # 浅灰色
        "linewidth": 0.65,
        "alpha": 0.80,
        "linestyle": (0, (4.0, 4.0)),    # 虚线模式
    }
```

#### **碰撞掩码**
```python
# base_block.py L319 & L506
mask = PGDrivableAreaProperty.BROKEN_COLLISION_MASK  # 不与车辆碰撞
```

---

### 2️⃣ **实线 (SOLID/CONTINUOUS)** - 权重 0.60

#### **定义位置**
```python
# metadrive/constants.py L287
PGLineType.CONTINUOUS = MetaDriveType.LINE_SOLID_SINGLE_WHITE
```

#### **视觉特征**
- **颜色**: 红色 (`#ef4444`) - 表示警告
- **样式**: 实线 `"-"`
- **线宽**: 1.0像素

#### **物理意义**
- ❌ **禁止换道**：交通规则不允许跨越
- ⚠️ **软碰撞检测**：跨越会触发轻微碰撞警告
- 🟡 **中等风险惩罚**：显著惩罚但不致命

#### **典型场景**
- 路口前的导向车道线
- 禁止变道路段（如隧道、桥梁）
- 匝道入口/出口的导流线

#### **代码实现**
```python
# risk_field_bev.py L646-652
if "boundary" in line_text or "solid" in line_text or "guardrail" in line_text:
    return {
        "color": "#ef4444",              # 红色边界
        "linewidth": 1.0,                # 只描边，不铺满
        "alpha": 0.75,
        "linestyle": "-",                # 实线
    }
```

#### **碰撞掩码**
```python
# base_block.py L319 & L506
mask = PGDrivableAreaProperty.CONTINUOUS_COLLISION_MASK  # 与车辆有碰撞
```

#### **与BOUNDARY的关键区别** ⭐
| **维度** | **SOLID (实线)** | **BOUNDARY (边界)** |
|---------|-----------------|-------------------|
| **位置** | 车道之间 | 道路最外侧 |
| **另一侧** | 仍是可行驶道路 | 路外区域（草地、人行道等） |
| **碰撞后果** | 轻微碰撞，可能继续行驶 | 严重碰撞，通常导致episode结束 |
| **风险权重** | 0.60（中等） | 1.00（高） |
| **交通规则** | 禁止变道 | 禁止驶出道路 |
| **物理障碍** | 无实体障碍 | 可能有护栏、路缘石 |

**举例说明：**
```
[车道1] ===实线=== [车道2] ===实线=== [车道3] |||边界||| [路外]
         ↑                              ↑              ↑
      禁止换道                       禁止换道       禁止驶出道路
      (仍在道路上)                  (仍在道路上)    (离开道路=事故)
```

---

### 3️⃣ **边界 (BOUNDARY/SIDE)** - 权重 1.00

#### **定义位置**
```python
# metadrive/constants.py L288
PGLineType.SIDE = MetaDriveType.BOUNDARY_LINE
```

#### **视觉特征**
- **颜色**: 红色 (`#ef4444`)
- **样式**: 实线 `"-"`
- **线宽**: 1.0像素（可能更粗）

#### **物理意义**
- ❌ **严禁跨越**：道路的最外侧边界
- ✅ **硬碰撞检测**：跨越即碰撞，通常导致episode终止
- 🔴 **高风险惩罚**：严重违反安全约束

#### **典型场景**
- 道路左右边缘
- 路肩与行车道的分界
- 桥梁/隧道的侧壁

#### **代码实现**
```python
# 与SOLID共用相同样式（见上面代码）
# 但在碰撞检测和风险评估中有更高优先级
```

#### **碰撞掩码**
```python
# 通常使用 CONTINUOUS_COLLISION_MASK 或专门的 BOUNDARY_MASK
# 取决于具体实现
```

#### **风险场中的特殊处理**
```python
# risk_field.py L195-210
left_dist, right_dist = vehicle.dist_to_left_side, vehicle.dist_to_right_side
min_boundary_distance = min(left_dist, right_dist)

# 边界风险是道路风险的组成部分
boundary_cost = exp(-d² / (2 * σ_b²))
```

---

### 4️⃣ **对向线 (ONCOMING/YELLOW)** - 权重 1.50 ⚠️ **最高**

#### **定义位置**
```python
# metadrive/type.py L27-L30
LINE_BROKEN_SINGLE_YELLOW = "ROAD_LINE_BROKEN_SINGLE_YELLOW"
LINE_SOLID_SINGLE_YELLOW = "ROAD_LINE_SOLID_SINGLE_YELLOW"
LINE_SOLID_DOUBLE_YELLOW = "ROAD_LINE_SOLID_DOUBLE_YELLOW"
```

#### **视觉特征**
- **颜色**: 黄色 (`YELLOW = (255/255, 200/255, 0/255, 1)`)
- **样式**: 实线或双黄线
- **线宽**: 可能更粗（1.5-2.0像素）

#### **物理意义**
- ❌ **绝对禁止跨越**：对向车道分隔线
- ✅ **硬碰撞检测**：跨越=逆行=严重事故
- 🔴🔴 **最高风险惩罚**：违反交通规则且极度危险

#### **典型场景**
- 双向两车道的中央分隔线
- 山路、乡村道路的中央线
- 禁止超车的路段（双黄线）

#### **代码实现**
```python
# pg_map.py L157-164
def _line_type_to_node_name(type, color):
    if type == PGLineType.CONTINUOUS and color == PGLineColor.YELLOW:
        return MetaDriveType.LINE_SOLID_SINGLE_YELLOW  # 单黄实线
    elif type == PGLineType.BROKEN and color == PGLineColor.YELLOW:
        return MetaDriveType.LINE_BROKEN_SINGLE_YELLOW  # 单黄虚线
    # ...
```

#### **风险场中的特殊处理**
```python
# 可能需要额外的惩罚项
if is_oncoming_lane_crossing:
    oncoming_penalty = 1.50 * base_risk  # 1.5倍惩罚
```

#### **为什么权重最高？**
1. **法律层面**：逆行是严重交通违法
2. **安全层面**：对向碰撞通常是正面撞击，死亡率极高
3. **规则层面**：智能体必须学会严格遵守交通规则

---

## 📊 风险权重设计原理

### **权重比例关系**
```
虚线 : 实线 : 边界 : 对向线 = 0.05 : 0.60 : 1.00 : 1.50
     = 1   : 12   : 20   : 30
```

### **设计逻辑**

#### **1. 虚线 (0.05) - 基准值**
- 几乎不惩罚，仅作为视觉引导
- 允许自由换道，符合交通规则

#### **2. 实线 (0.60) - 12倍于虚线**
- 显著惩罚但非致命
- 鼓励遵守"禁止变道"规则
- 偶尔跨越不会立即终止训练

#### **3. 边界 (1.00) - 20倍于虚线**
- 高风险惩罚
- 代表"离开可行驶区域"
- 通常伴随out_of_road事件

#### **4. 对向线 (1.50) - 30倍于虚线**
- 最高惩罚
- 代表"逆行"这一极端危险行为
- 必须严格避免

---

## 🎨 可视化差异

### **BEV热力图中的表现**

```python
# 查看不同组件的风险分布
python debug_risk_field_bev_gif.py --component lane --output debug/lane_risk.gif
```

- **虚线区域**: 几乎透明（风险≈0.05）
- **实线区域**: 橙红色带状（风险≈0.60）
- **边界区域**: 深红色带状（风险≈1.00）
- **对向线区域**: 最深的红色（风险≈1.50）

### **Topdown Overlay的表现**

```python
# 快速预览
python debug_risk_field_topdown_overlay_gif.py --no-vehicle-risk --output debug/lines.gif
```

- **虚线**: 浅灰短划线 `---- ----`
- **实线**: 红色实线 `────────`
- **边界**: 红色粗实线 `════════`
- **对向线**: 黄色实线/双黄线 `━━━━━━━━` 或 `══════`

---

## 💡 实际应用建议

### **1. 训练配置**
```yaml
# env.yaml
risk_field_config:
  risk_field_lane_weight: 0.1          # 车道线总权重
  risk_field_boundary_weight: 1.0      # 边界权重
  # 内部会自动根据线型调整：
  # - broken: 0.05 * 0.1 = 0.005
  # - solid: 0.60 * 0.1 = 0.06
  # - boundary: 1.00 * 1.0 = 1.0
  # - oncoming: 1.50 * 1.0 = 1.5
```

### **2. 调试技巧**
```bash
# 单独查看车道线风险
python debug_risk_field_bev_gif.py \
  --component lane \
  --output debug/lane_types.gif \
  --draw-lane-centers  # 显示中心线便于对比

# 观察智能体如何处理不同类型
python debug_risk_field_topdown_overlay_gif.py \
  --vehicle-only \
  --output debug/agent_behavior.gif
```

### **3. 常见问题排查**

#### **问题1：智能体频繁跨越实线**
```yaml
# 解决方案：增加实线惩罚
risk_field_config:
  risk_field_lane_weight: 0.3  # 从0.1提升到0.3
```

#### **问题2：智能体不敢换道（即使虚线）**
```yaml
# 解决方案：检查虚线权重是否过高
risk_field_config:
  risk_field_lane_weight: 0.05  # 降低总权重
```

#### **问题3：智能体经常偏离道路**
```yaml
# 解决方案：增强边界惩罚
risk_field_config:
  risk_field_boundary_weight: 2.0  # 从1.0提升到2.0
  risk_field_offroad_weight: 1.5   # 启用偏离道路风险
```

---

## 🔗 相关代码位置

### **核心定义**
- **常量定义**: [`metadrive/constants.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/metadrive/metadrive/constants.py) L282-296
- **类型枚举**: [`metadrive/type.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/metadrive/metadrive/type.py) L24-42
- **碰撞掩码**: [`metadrive/component/block/base_block.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/metadrive/metadrive/component/block/base_block.py) L319, L506

### **风险场实现**
- **车道线样式**: [`risk_field_bev.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/risk_field_bev.py) L606-655 `_lane_line_style()`
- **边界风险计算**: [`risk_field.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/risk_field.py) L195-210 `_road_risk()`
- **车道线风险计算**: [`risk_field.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/risk_field.py) L165-173

### **可视化工具**
- **BEV精确采样**: [`debug_risk_field_bev_gif.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/debug_risk_field_bev_gif.py)
- **Topdown快速预览**: [`debug_risk_field_topdown_overlay_gif.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/debug_risk_field_topdown_overlay_gif.py)
- **道路专用工具**: [`debug_risk_field_road_gif.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/debug_risk_field_road_gif.py)

---

## 📝 总结

### **关键记忆点**

1. **虚线 (0.05)**: "可以走" - 同向换道，无碰撞
2. **实线 (0.60)**: "别走" - 禁止变道，软碰撞
3. **边界 (1.00)**: "不能走" - 道路边缘，硬碰撞
4. **对向线 (1.50)**: "绝对不能走" - 逆行，最高惩罚

### **Solid vs Boundary 核心区别**

| **对比项** | **Solid (实线)** | **Boundary (边界)** |
|-----------|-----------------|-------------------|
| **位置** | 车道之间 | 道路最外侧 |
| **另一侧是什么** | 仍是道路 | 路外区域 |
| **碰撞后能否继续** | 可能可以 | 通常不能 |
| **风险等级** | 中等 (0.60) | 高 (1.00) |
| **类比** | "禁止跨越的黄线" | "悬崖边缘" |

**形象比喻：**
- **虚线** = 人行横道的白线（可以跨）
- **实线** = 足球场的边线（出界犯规但比赛继续）
- **边界** = 游泳池的边缘（掉下去就湿了）
- **对向线** = 高速公路的中央隔离带（穿越=自杀）

---

**最后更新**: 2026-04-12  
**维护者**: safeRL-metadrive 团队
