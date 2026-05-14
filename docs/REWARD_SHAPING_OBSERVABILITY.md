# 奖励塑形观测口径（`metadrive_env.py` 链路）

这份文档专门服务于**奖励塑形调参**。目标不是罗列所有日志字段，而是给出一套稳定的观察口径，帮助我们回答 5 个问题：

- 出车道问题
- 转向震荡 / 左右切换
- 速度问题
- 风险场整体问题
- 周车风险问题

## 1. 适用范围

本口径只适用于当前 **`metadrive_env.py` 环境链**：

- 环境逻辑来源：`metadrive/metadrive/envs/metadrive_env.py`
- 指标聚合来源：`fsrl/fsrl/data/fast_collector.py`、`fsrl/fsrl/trainer/base_trainer.py`

如果切回 `env.py` 那套自定义平滑性环境，这份文档里的“有效指标池”不再完全成立。

---

## 2. 去哪里看这些指标

### WandB / 训练日志

优先看这些聚合后的字段：

- `train/*_step_mean`
- `test/*_step_mean`
- `train/safe_success_rate`
- `test/safe_success_rate`

### periodic eval 的 `metrics.json`

结构上要看：

```json
{
  "test_stats": {
    "safe_success_rate": ...,
    "risk_field_vehicle_cost_step_mean": ...,
    "steering_switch_step_mean": ...
  }
}
```

也就是说，**主要诊断值都在 `test_stats` 下面**，不要只看顶层字段。

---

## 3. 当前链路的有效指标池

下面这些是当前 `metadrive_env.py` 实际写进 `step_info`，并且 collector / trainer 会聚合的主指标。

### 行为结果

- `success_rate`
- `safe_success_rate`
- `crash_free_rate`
- `no_out_of_road_rate`
- `avg_speed_km_h`

### 奖励分解

- `base_reward_step_mean`
- `driving_component_reward_step_mean`
- `speed_component_reward_step_mean`
- `normalized_speed_reward_step_mean`
- `longitudinal_progress_step_mean`
- `lateral_reward_step_mean`
- `final_reward_step_mean`
- `reward_override_active_step_mean`
- `reward_override_delta_step_mean`

### 转向相关

- `steering_delta_step_mean`
- `steering_switch_step_mean`
- `steering_smoothness_penalty_step_mean`
- `steering_switch_penalty_step_mean`
- `steering_penalty_step_mean`

### 横向位置相关

- `lateral_now_step_mean`
- `lateral_norm_step_mean`
- `lateral_score_step_mean`
- `lateral_speed_gate_step_mean`
- `lateral_forward_gate_step_mean`
- `lateral_broken_line_gate_step_mean`

### 风险场

- `risk_field_cost_step_mean`
- `risk_field_road_cost_step_mean`
- `risk_field_vehicle_cost_step_mean`
- `risk_field_object_cost_step_mean`

---

## 4. 当前链路不要用来判断的指标

下面这些字段虽然在历史白名单、部分旧 run、或 `metrics.json` 里可能出现，但**当前 `metadrive_env.py` 并不把它们作为主诊断源**。如果它们是 0，不能直接说明“行为健康”；如果它们非 0，也应视作旧链路残留或其他环境来源。

- `steering_jerk*`
- `lateral_velocity*`
- `lateral_acceleration*`
- `lateral_jerk*`
- `yaw_rate*`
- `yaw_acceleration*`
- `smoothness_penalty*`
- `action_smoothness_penalty*`

这组字段最容易造成“日志里有，但当前环境并不真依赖它们”的误读。

---

## 5. 五类问题的固定观察面板

## A. 出车道问题

主指标：

- `no_out_of_road_rate`
- `lateral_norm_step_mean`
- `risk_field_road_cost_step_mean`

辅助指标：

- `lateral_score_step_mean`
- `lateral_reward_step_mean`
- `reward_override_active_step_mean`
- `reward_override_delta_step_mean`

解读：

- `no_out_of_road_rate` 下降，同时 `lateral_norm_step_mean` 上升、`risk_field_road_cost_step_mean` 上升：典型车道保持不足
- `reward_override_active_step_mean` 高：大量 step 已被终止事件覆盖，问题不只是“贴边”，而是已经发展成真实失败
- `lateral_score_step_mean` 很低但 `lateral_reward_step_mean` 仍非零：横向奖励门控还在发，但位置质量不够好

## B. 转向震荡 / 左右切换

主指标：

- `steering_switch_step_mean`
- `steering_delta_step_mean`
- `steering_penalty_step_mean`

辅助指标：

- `avg_speed_km_h`
- `longitudinal_progress_step_mean`
- `lateral_norm_step_mean`

解读：

- `steering_switch_step_mean` 是最直接的“左右来回切换”指标
- `steering_delta_step_mean` 高但 `steering_switch_step_mean` 低：更像单向大修正，不一定是抖动
- `steering_switch_step_mean` 高，同时 `avg_speed_km_h` 和 `longitudinal_progress_step_mean` 低：典型“原地抖方向 / 低速犹豫”
- `steering_penalty_step_mean` 接近或超过 `driving_component_reward_step_mean`：惩罚可能压住前进激励

## C. 速度问题

主指标：

- `avg_speed_km_h`
- `speed_component_reward_step_mean`
- `longitudinal_progress_step_mean`
- `driving_component_reward_step_mean`

辅助指标：

- `final_reward_step_mean`
- `steering_switch_step_mean`
- `risk_field_vehicle_cost_step_mean`

解读：

- `avg_speed_km_h` 低，同时 `speed_component_reward_step_mean` 和 `longitudinal_progress_step_mean` 低：速度激励不足，或被其他惩罚压住
- `avg_speed_km_h` 低，但 `steering_switch_step_mean` 高：不是单纯慢，而是抖动导致走不出去
- `avg_speed_km_h` 低，同时 `risk_field_vehicle_cost_step_mean` 高：更像周车交互压力下的保守减速

## D. 风险场整体问题

主指标：

- `risk_field_cost_step_mean`
- `risk_field_road_cost_step_mean`
- `risk_field_vehicle_cost_step_mean`
- `risk_field_object_cost_step_mean`

辅助指标：

- `final_reward_step_mean`
- `crash_free_rate`
- `no_out_of_road_rate`

解读：

- `risk_field_cost_step_mean` 高，且主要由 `risk_field_road_cost_step_mean` 驱动：问题在道路边界 / 车道保持
- 主要由 `risk_field_vehicle_cost_step_mean` 驱动：问题在周车交互
- 主要由 `risk_field_object_cost_step_mean` 驱动：问题在静态障碍物避让
- `risk_field_cost_step_mean` 高，但 `crash_free_rate` 也高：当前更多是“高压但没出事”，属于 shaping 约束阶段
- `risk_field_cost_step_mean` 高，且 `crash_free_rate`、`no_out_of_road_rate` 都差：风险场和真实失败事件已经同向恶化

## E. 周车风险问题

主指标：

- `risk_field_vehicle_cost_step_mean`
- `crash_free_rate`
- `safe_success_rate`

辅助指标：

- `avg_speed_km_h`
- `steering_switch_step_mean`
- `risk_field_cost_step_mean`

解读：

- `risk_field_vehicle_cost_step_mean` 高，`crash_free_rate` 低：周车交互处理差
- `risk_field_vehicle_cost_step_mean` 高，同时 `steering_switch_step_mean` 高：典型“看到周车后左右犹豫”
- `risk_field_vehicle_cost_step_mean` 高，但 `avg_speed_km_h` 很低：策略可能在用过度保守减速规避周车

---

## 6. 固定做三组量纲对比

看 reward shaping 不能只看高低，还要看**量纲关系**：

- `steering_penalty_step_mean` vs `driving_component_reward_step_mean`
- `lateral_reward_step_mean` vs `speed_component_reward_step_mean`
- `risk_field_cost_step_mean` vs `crash_free_rate` / `no_out_of_road_rate`

判读规则：

- 惩罚项长期接近或压过主前进奖励：优先怀疑 shaping 过强
- 风险场指标已经明显下降，但行为结果没改善：优先怀疑 shaping 方向不对，而不是强度不够
- 行为结果改善，但某个 reward 分解长期接近 0：说明那项 shaping 实际没有参与学习

---

## 7. 固定观察顺序

每次看一个 run，都按同样顺序走，避免思路飘：

1. 先看结果
   - `safe_success_rate`
   - `crash_free_rate`
   - `no_out_of_road_rate`
   - `avg_speed_km_h`

2. 再看主奖励是否健康
   - `driving_component_reward_step_mean`
   - `speed_component_reward_step_mean`
   - `final_reward_step_mean`

3. 再看横向和出车道
   - `lateral_norm_step_mean`
   - `lateral_reward_step_mean`
   - `risk_field_road_cost_step_mean`

4. 再看转向抖动
   - `steering_switch_step_mean`
   - `steering_delta_step_mean`
   - `steering_penalty_step_mean`

5. 最后看风险归因
   - `risk_field_cost_step_mean`
   - `risk_field_vehicle_cost_step_mean`
   - `risk_field_object_cost_step_mean`

---

## 8. 每次调参后的 sanity check

至少做下面 3 个检查：

1. `steering_penalty_step_mean` 是否明显小于 `driving_component_reward_step_mean`
2. `risk_field_vehicle_cost_step_mean` 和 `risk_field_road_cost_step_mean` 是否真的能区分“周车问题”和“出车道问题”
3. 是否误把全 0 的旧平滑性字段当成“已经收敛”

---

## 9. 下一轮如果要补指标，优先补什么

本轮不补代码。如果下一轮要增强可解释性，最值得优先补的是：

- 周车数量
- 静态障碍物数量
- `risk_field_reward_penalty`

这三项最能帮助判断：

- 为什么 `risk_field_vehicle_cost_step_mean` 变高
- 是场景里“真有更多车/障碍物”，还是策略真的更糟
- 风险场 shaping 的强度是不是已经压过了主奖励
