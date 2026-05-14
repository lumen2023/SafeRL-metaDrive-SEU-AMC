# 车道线风险分布改进：从标准高斯到超髙斯 📊

## 🎯 改进目标

将车道线风险从**标准一维高斯分布**改为**超髙斯分布**，实现更陡峭的边缘惩罚效果，鼓励智能体自然居中行驶。

## 📝 核心改动

### 1. 新增配置参数

在 [`risk_field.py`](file:///home/ac/@Lyz-Code/safeRL-metadrive/risk_field.py) 的 `DEFAULTS` 中添加：

```python
"risk_field_lane_beta": 2.0,  # 车道线超高斯指数（控制边缘惩罚陡峭程度）
```

**推荐范围**: 1.5 ~ 2.5（建议起点为 2.0）

### 2. 新增计算方法

添加 `_super_gaussian_risk_1d(distance, sigma, beta)` 方法：

```python
@staticmethod
def _super_gaussian_risk_1d(distance: float, sigma: float, beta: float) -> float:
    """计算一维超髙斯风险
    
    公式：exp(-(distance² / sigma²)^beta)
    
    当beta=1时为标准高斯，beta>1时曲线更陡峭，边缘惩罚更强。
    """
    sigma = max(float(sigma), RiskFieldCalculator.EPS)
    beta = max(float(beta), RiskFieldCalculator.EPS)
    exponent = -((max(distance, 0.0) ** 2) / (sigma ** 2)) ** beta
    return float(math.exp(max(exponent, -80.0)))
```

### 3. 应用位置

在 `_road_risk()` 方法中替换原有的 `_one_dimensional_risk()` 调用：

```python
# 使用超髙斯分布替代标准高斯分布，使边缘惩罚更陡峭
lane_beta = self._cfg("risk_field_lane_beta")
lane_cost = max(
    side0_line_profile["factor"] * self._super_gaussian_risk_1d(side_0_gap, side0_line_profile["sigma"], lane_beta),
    side1_line_profile["factor"] * self._super_gaussian_risk_1d(side_1_gap, side1_line_profile["sigma"], lane_beta),
)
```

## 📈 效果对比

### 数值分析（σ=0.35米）

| 距离(米) | β=1.0 (标准高斯) | β=1.5 (温和) | **β=2.0 (推荐)** | β=2.5 (陡峭) |
|---------|-----------------|-------------|-----------------|-------------|
| 0.0     | 1.000           | 1.000       | **1.000**       | 1.000       |
| 0.1     | 0.960           | 0.977       | **0.993**       | 0.998       |
| 0.2     | 0.849           | 0.830       | **0.899**       | 0.941       |
| **0.35**| **0.607**       | **0.368**   | **0.368**       | **0.368**   |
| 0.5     | 0.360           | 0.054       | **0.016**       | 0.003       |
| 0.7     | 0.135           | 0.000       | **0.000**       | 0.000       |
| 1.0     | 0.017           | 0.000       | **0.000**       | 0.000       |

### 关键观察

✅ **β=1.0 (标准高斯)**: 衰减平缓，边缘惩罚不够明显  
✅ **β=2.0 (推荐)**: 在σ附近快速衰减，紧贴车道线时显著惩罚  
⚠️ **β=2.5 (陡峭)**: 边缘惩罚极强，但可能导致梯度消失  

## 🔧 参数调整策略

### 调整顺序（重要！）

1. **优先调整 β** (`risk_field_lane_beta`)
   - 控制曲线形状
   - 推荐范围：1.5 ~ 2.5
   - 建议起点：2.0

2. **再调 σ** (`risk_field_lane_edge_sigma`)
   - 控制影响范围
   - 推荐范围：0.3 ~ 0.6 米
   - 当前默认：0.6 米

3. **最后调节权重** (`risk_field_lane_weight`)
   - 控制整体强度
   - 推荐范围：0.05 ~ 0.10
   - 当前默认：0.05

### ⚠️ 避免极端组合

❌ **不要同时使用**: `beta > 3.0` 且 `sigma < 0.3`  
→ 会导致梯度消失，智能体无法学习

## 🎨 可视化验证

运行测试脚本生成对比图：

```bash
python test_lane_risk_distribution.py
```

生成的图表包括：
- 不同β值的曲线对比
- 安全区/警告区/危险区标注
- 关键参数说明

输出文件：`lane_risk_comparison.png`

## 📋 设计原则

### 目标行为

实现 **"紧贴车道线时显著惩罚，稍远即快速衰减"** 的效果：

- ✅ 靠近车道线（< 0.2m）：高风险，强烈惩罚
- ✅ 中等距离（0.2~0.5m）：风险快速下降
- ✅ 远离车道线（> 0.5m）：风险接近零，不干扰正常驾驶

### 物理意义

- **β值增大** → 曲线更"方"，边缘更陡峭
- **σ值减小** → 影响范围更窄，风险更集中
- **权重调节** → 整体强度缩放

## 🔍 调试建议

### 观察指标

在 WandB 或训练日志中关注：

1. `risk_field_lane_cost`: 车道线风险成本
2. `risk_field_lateral_offset`: 横向偏移量
3. `episode/return`: 总体回报

### 预期效果

改进后应该看到：
- ✅ 智能体更倾向于保持在车道中心
- ✅ 变道行为更加平滑和谨慎
- ✅ 减少"贴边行驶"的危险行为

## 📚 相关文档

- [风险场系统架构规范](./ENV_README.md)
- [测试脚本](./test_lane_risk_distribution.py)
- [核心实现](./risk_field.py)

---

**最后更新**: 2026-05-05  
**作者**: SafeRL-MetaDrive Team
