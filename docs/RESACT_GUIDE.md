# ResAct (Residual Action) 集成指南

## 📖 什么是 ResAct?

**ResAct (Residual Action)** 是一种动作平滑化技术,核心思想是:

> **策略不直接输出最终动作,而是输出"动作的修正量"(残差),再结合上一步动作得到最终动作。**

### 数学公式

```python
# 传统方式
final_action = policy(state)

# ResAct 方式
residual = policy(state, prev_action)
final_action = tanh(residual + prev_action)
```

其中 `tanh` 确保动作在合法范围内 `[-1, 1]`。

---

## 🎯 为什么使用 ResAct?

### 优势

1. **动作平滑性**: 残差通常很小,避免动作突变
2. **训练稳定性**: 平滑的动作序列有助于价值函数学习
3. **物理合理性**: 真实驾驶中,方向盘和油门都是连续变化的
4. **可解释性**: `action_distance` 指标直观反映动作平滑度

### 适用场景

- ✅ 连续控制任务 (如驾驶、机器人)
- ✅ 需要平滑动作的场景
- ✅ 策略震荡严重的问题

---

## 🛠️ 实现方案: Env Wrapper

### 核心设计原则

**零侵入**: 不修改现有 SACL/PPO/collector/policy 代码,通过包装器隔离。

```
策略网络 → ResActWrapper → SafeMetaDriveEnv_mini
   ↓            ↓                  ↓
输出残差   tanh(res+prev)    执行真实动作
```

### 文件结构

```
safeRL-metadrive/
├── resact_wrapper.py          # ResAct Wrapper 实现
├── train_resact_example.py    # 使用示例脚本
├── docs/
│   └── RESACT_GUIDE.md        # 本文档
└── agents/
    ├── agent_sacl/            # 无需修改
    └── agent_ppol/            # 无需修改
```

---

## 📝 快速开始

### Step 1: 创建带 ResAct 的环境

```python
from resact_wrapper import ResActWrapper
from env import get_training_env

# 1. 创建基础环境
base_env = get_training_env(extra_config={
    "accident_prob": 0.8,
    "traffic_density": 0.05,
    # ... 其他配置
})

# 2. 包裹 ResAct Wrapper
env = ResActWrapper(
    base_env, 
    action_dim=2,              # steering + throttle
    use_throttle_clip=True     # 对throttle进行[0,1]裁剪
)
```

### Step 2: 训练 (无需修改现有代码!)

```python
# 现有的训练流程完全不变!
# 策略仍输出 2D 向量,但 Wrapper 会将其解释为残差

for episode in range(num_episodes):
    obs, _ = env.reset()
    
    for step in range(max_steps):
        # 策略输出 (实际上是残差)
        action = agent.select_action(obs)  # shape: (2,)
        
        # Wrapper 自动合成真实动作并执行
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 可以记录 ResAct 相关指标
        if 'action_distance' in info:
            wandb.log({
                'action_distance': info['action_distance'],
                'final_action_steering': info['final_action'][0],
                'final_action_throttle': info['final_action'][1],
            })
        
        if terminated or truncated:
            break
```

### Step 3: 评估平滑性

```python
# Wrapper 提供统计信息
stats = env.get_episode_stats()
print(f"平均动作距离: {stats['mean']:.4f}")
print(f"标准差: {stats['std']:.4f}")

# action_distance 越小,表示动作越平滑
```

---

## 🔧 高级配置

### 1. 调整动作维度

```python
# 如果只希望对 steering 应用 ResAct
env = ResActWrapper(base_env, action_dim=1)

# 默认是两个维度都用
env = ResActWrapper(base_env, action_dim=2)
```

### 2. Throttle 裁剪

```python
# MetaDrive 的 throttle 范围是 [0, 1],建议启用裁剪
env = ResActWrapper(base_env, use_throttle_clip=True)

# 如果希望 throttle 也能取负值 (类似刹车)
env = ResActWrapper(base_env, use_throttle_clip=False)
```

### 3. 向量化环境

```python
from resact_wrapper import ResActVectorizedWrapper

# 支持批量环境 (并行训练)
env = ResActVectorizedWrapper(vectorized_env, action_dim=2)
```

---

## ⚠️ 注意事项

### 1. 与横向奖励的冲突

**问题**: `use_lateral_reward` 会惩罚偏离车道中心的行为,可能抑制变道。

**建议**: 使用 ResAct 时关闭此选项:

```python
env_config = {
    "use_lateral_reward": False,  # ⚠️ 关闭
    # 改用风险场的车道线风险作为 Cost 信号
    "use_risk_field_cost": True,
    "risk_field_lane_weight": 0.6,
}
```

### 2. 学习率调整

由于残差通常比绝对动作值小,可能需要调整学习率:

```yaml
# agents/agent_sacl/config.yaml
actor_lr: 5e-4  # 可能需要从 3e-4 调大
critic_lr: 5e-4
```

### 3. Replay Buffer 扩展

如果需要完整实现 ResAct (如原论文),需要在 buffer 中存储 `prev_action`:

```python
# 扩展现有的 buffer
buffer.add(
    obs=obs,
    action=residual_action,      # 策略输出的残差
    prev_action=prev_action,     # ⭐ 新增
    reward=reward,
    next_obs=next_obs,
    ...
)
```

但在当前的 Wrapper 方案中,**这不是必须的**,因为:
- Wrapper 内部维护 `prev_action`
- 策略网络不需要显式接收 `prev_action` 作为输入
- 简化实现,便于快速实验

### 4. 初始化

每次 `reset()` 时,`prev_action` 会自动清零,无需手动处理。

---

## 📊 实验对比

### 建议的对比实验

| 实验组 | 配置 | 预期效果 |
|--------|------|----------|
| Baseline | 无 ResAct, 无平滑惩罚 | 基准性能 |
| Smoothness Only | 有平滑惩罚, 无 ResAct | 动作略平滑 |
| ResAct Only | 有 ResAct, 无平滑惩罚 | 动作明显平滑 |
| Both | ResAct + 平滑惩罚 | 最佳平滑性? |

### 关键指标

```python
# 安全性指标
- collision_rate: 碰撞率
- out_of_road_rate: 偏离道路率
- avg_cost: 平均成本

# 性能指标
- success_rate: 成功率
- avg_reward: 平均奖励
- route_completion: 路线完成度

# 平滑性指标 (ResAct 特有)
- avg_action_distance: 平均动作距离 (越小越好)
- action_variance: 动作方差
```

---

## 🔬 与原论文的差异

### LiuZhenxian123/ResAct 原始实现

```python
# Actor 网络显式接收 prev_action
mu = actor(prev_obs, obs, prev_action)  # 输出残差
final_action = tanh(mu + prev_action)

# Buffer 存储 prev_action
buffer.add(obs, residual, prev_action, reward, next_obs)
```

### 我们的 Wrapper 方案

```python
# Actor 网络不知道 ResAct 的存在
action = actor(obs)  # 输出被 Wrapper 解释为残差

# Wrapper 内部维护 prev_action
final_action = tanh(action + wrapper.prev_action)

# Buffer 只需存储 action (即残差)
buffer.add(obs, action, reward, next_obs)
```

### 优缺点对比

| 特性 | 原始实现 | Wrapper 方案 |
|------|---------|-------------|
| 代码侵入性 | 高 (需改 agent/buffer) | 低 (仅加 wrapper) |
| 实现复杂度 | 高 | 低 |
| 灵活性 | 高 (可定制网络结构) | 中 |
| 实验速度 | 慢 (需大量修改) | 快 (即插即用) |
| 性能上限 | 可能更高 | 略低 (简化版) |

**结论**: Wrapper 方案适合快速验证 ResAct 的有效性,如果效果显著,再考虑完整实现。

---

## 🐛 常见问题

### Q1: ResAct 会不会限制策略的表达能力?

**A**: 理论上 `tanh` 的输出范围是 `[-1, 1]`,与 MetaDrive 的动作范围一致,不会限制表达能力。但残差的累积效应可能导致某些极端动作难以达到。如果发现性能下降,可以尝试:
- 增大初始探索噪声
- 调整学习率
- 禁用 `use_throttle_clip`

### Q2: 如何可视化动作平滑性?

**A**: 使用 WandB 记录:

```python
wandb.log({
    'action_distance': info['action_distance'],
    'final_action/steering': info['final_action'][0],
    'final_action/throttle': info['final_action'][1],
    'residual/steering': info['residual_action'][0],
    'residual/throttle': info['residual_action'][1],
})
```

然后绘制时间序列图,观察动作是否平滑。

### Q3: ResAct 和风险场成本有冲突吗?

**A**: 没有冲突。ResAct 影响动作执行层面,风险场影响成本计算层面,两者职责分离。实际上可能有协同效应:
- ResAct 使动作平滑 → 减少突然转向/加减速
- 风险场惩罚危险行为 → 引导安全驾驶
- 两者结合可能获得更安全、更平滑的策略

### Q4: 能否只在特定场景使用 ResAct?

**A**: 可以!Wrapper 是可插拔的:

```python
if use_resact:
    env = ResActWrapper(base_env)
else:
    env = base_env
```

这样可以轻松对比有/无 ResAct 的效果。

---

## 📚 参考资料

1. **原论文**: "Visual Reinforcement Learning with Residual Action" (Liu et al.)
2. **原代码**: https://github.com/LiuZhenxian123/ResAct
3. **本项目实现**: 
   - `resact_wrapper.py`: Wrapper 核心实现
   - `train_resact_example.py`: 使用示例
   - `docs/RESACT_GUIDE.md`: 本文档

---

## 🚀 下一步

1. **运行示例**: `python3 train_resact_example.py`
2. **集成到训练**: 修改 `train_sacl.py` 或 `train_ppol.py`,添加 Wrapper
3. **对比实验**: 有/无 ResAct 的性能对比
4. **超参数搜索**: 学习率、平滑惩罚权重等
5. **完整实现** (可选): 如果效果显著,考虑扩展 buffer 和 actor 网络

---

**祝实验顺利! 🎉**
