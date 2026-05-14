# r2Dreamer 多模态编码器迁移分析

## 目标

这份文档总结 r2Dreamer 中多模态编码器的工作方式，并给出迁移到 safeRL-MetaDrive 的第一版设计。当前只做分析和接口设计，不修改 safeRL 训练代码，也不改 r2Dreamer。

## r2Dreamer 的观测编码数据流

r2Dreamer 的环境观测是一个 dict，例如：

```python
{
    "state": ...,
    "image": ...,
    "reward": ...,
    "is_first": ...,
    "is_last": ...,
    "is_terminal": ...,
    "log_cost": ...,
}
```

`MultiEncoder` 会根据 key、shape 和配置里的正则表达式，把观测分成两类：

- CNN 分支：shape 长度为 3 且匹配 `cnn_keys`，典型输入是 `image`，形状类似 `(H, W, C)`。
- MLP 分支：shape 长度为 1 或 2 且匹配 `mlp_keys`，典型输入是 `state`，MetaDrive 当前是 `(259,)`。

以下字段不会进入 encoder：

- `is_first`
- `is_last`
- `is_terminal`
- `reward`
- 所有 `log_*`

这些字段是训练控制信号或日志指标，不应该被策略当成环境状态使用。

## 多模态融合方式

`MultiEncoder` 的融合逻辑很直接：

1. 多个 CNN 观测沿 channel 维拼接，再交给同一个 `ConvEncoder`。
2. 多个 MLP 观测沿 feature 维拼接，再交给同一个 MLP encoder。
3. 如果同时有 CNN 和 MLP 分支，两个分支的输出 feature 在最后一维拼接。
4. 如果只有一个分支，就直接返回该分支的输出。

因此 r2Dreamer 的 encoder 输出是统一的 embedding：

```python
embed = encoder(obs)
```

这个 `embed` 不直接输出动作，而是进入 RSSM 世界模型。

## Dreamer 中 embedding 的作用

r2Dreamer 的核心不是直接用观测做 actor/critic，而是先用 RSSM 建模 latent state：

```python
embed = encoder(obs)
post_stoch, post_deter, post_logit = rssm.observe(
    embed,
    action,
    initial,
    is_first,
)
feat = rssm.get_feat(post_stoch, post_deter)
```

之后几个头都基于 `feat`：

- actor：从 latent feature 输出动作分布。
- value：预测价值。
- reward：预测 reward。
- cont：预测 continuation。

在 `rep_loss=r2dreamer` 下，还有一个 projector，把 RSSM latent feature 投影到 encoder embedding 空间，并用 Barlow Twins 风格损失约束表征：

```python
x1 = projector(feat)
x2 = embed.detach()
loss = invariance_loss + lambda * redundancy_loss
```

所以 r2Dreamer 的 encoder 不只是 policy 前处理器，它还参与世界模型训练和表征学习。

## 当前 r2Dreamer MetaDrive 配置

当前 MetaDrive 接入是 state-only：

```yaml
encoder:
  mlp_keys: '^state$'
  cnn_keys: '$^'
decoder:
  mlp_keys: '^state$'
  cnn_keys: '$^'
```

这表示：

- `state` 进入 MLP encoder。
- 没有任何 key 进入 CNN encoder。
- 图像分支目前没有启用。

但架构本身已经支持未来扩展到：

```python
{
    "state": np.ndarray(259,),
    "image": np.ndarray(H, W, C),
}
```

## safeRL 当前网络接口

safeRL-MetaDrive 当前 PPO-Lag / SAC-Lag 基本使用 Tianshou/FSRL 风格的 `Net` 作为 preprocess net：

```python
net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
actor = ActorProb(net, action_shape, ...)
critic = Critic(Net(state_shape, ...), ...)
```

`Net.forward()` 的接口是：

```python
forward(obs, state=None, info=None) -> (logits, state)
```

当前假设 observation 是一个扁平向量，MetaDrive 为 259 维。actor 和 critic 都依赖 preprocess net 的 `output_dim` 来构建后续 MLP。

## 推荐迁移方案

第一版不要把 Dreamer 的 RSSM 搬到 safeRL，也不要替换默认训练路径。推荐新增一个可选的 preprocess net：

```python
class MultiModalPreprocessNet(nn.Module):
    def __init__(
        self,
        obs_shapes,
        mlp_keys="^state$",
        cnn_keys="^image$",
        hidden_sizes=(256, 256),
        cnn_depth=16,
        device="cuda",
    ):
        ...

    def forward(self, obs, state=None, info=None):
        ...
        return feature, state
```

设计要点：

- 兼容现有 `np.ndarray(259,)` 输入，把它自动视为 `{"state": obs}`。
- 兼容未来 dict 输入，例如 `{"state": ..., "image": ...}`。
- 保留 `output_dim` 属性，让 `ActorProb`、`Critic` 可以无缝复用。
- 不处理 `reward`、`done`、`cost`、`log_*` 等训练或日志字段。
- CNN/MLP/fuser 的逻辑参考 r2Dreamer `MultiEncoder`，但实现放在 safeRL 自己的模块里，避免直接 import r2Dreamer。

## 分阶段落地建议

第一阶段：只新增模块和配置开关。

- 默认仍使用原有 MLP `Net`。
- 新增 `encoder_type=mlp|multimodal`。
- 当 `encoder_type=multimodal` 且 observation 是 dict 时启用多模态 encoder。

第二阶段：让环境可选返回 dict observation。

- `state`: 当前 259 维 LidarStateObservation。
- `image`: 可选 front camera 或 top-down image。
- 继续保证评估接口能退回纯 state，避免破坏已有提交格式。

第三阶段：分别评估 actor/critic 共享 encoder 与独立 encoder。

- PPO/SAC 的 actor 和 critic 可以各自使用独立 encoder，最安全但参数更多。
- 也可以共享 encoder，但需要谨慎处理 actor loss 和 critic loss 对表征的共同更新。

## 与 r2Dreamer 的差异

迁移到 safeRL 时只建议迁移 encoder/fuser，不建议迁移以下 Dreamer 组件：

- RSSM
- reward predictor
- continuation predictor
- decoder
- r2dreamer projector / Barlow loss

原因是 safeRL 当前是 model-free PPO/SAC 训练范式。强行加入 RSSM 会变成新的 model-based safe RL 算法，改动范围远大于“多模态 encoder 迁移”。

## 验证标准

未来实现时，至少需要这些 smoke tests：

- 纯 state 输入仍能通过原 PPO/SAC 网络。
- dict 输入 `{state}` 能通过 `MultiModalPreprocessNet`。
- dict 输入 `{state, image}` 能产生固定维度 feature。
- `ActorProb` 和 `Critic` 能读取 `output_dim` 并正常 forward。
- 不改变默认训练命令的行为。

