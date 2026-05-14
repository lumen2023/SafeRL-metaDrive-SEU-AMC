# safeRL-metadrive

这是一个面向 **SafeMetaDrive 安全强化学习** 的实验仓库，重点封装了三块可迁移能力：

- **奖励函数**：道路行驶奖励、速度奖励、横向居中奖励、转向平滑惩罚、事件惩罚和风险场 reward shaping。
- **风险场**：基于 MetaDrive 原生道路、车辆、障碍物状态计算连续风险成本，支持调试快照和 topdown 叠加可视化。
- **残差动作 ResAct**：把策略输出解释为上一动作附近的残差控制量，方便限制转向和油门变化幅度。

核心代码已经整理到 `safe_metadrive_adapter/`。旧入口仍然保留，所以原来的训练、评估、调试脚本不需要改命令。

## 目录结构

```text
safe_metadrive_adapter/
  config.py          默认环境、奖励、风险场、ResAct 配置
  env.py             SafeMetaDriveAdapterEnv，覆盖 reward/cost 逻辑
  factory.py         make_safe_metadrive_env / get_training_env / get_validation_env
  risk_field.py      风险场计算器 RiskFieldCalculator
  reward.py          奖励函数和 cost 聚合
  resact.py          残差动作 wrapper
  local_import.py    优先使用仓库内本地 MetaDrive

env.py               旧环境入口兼容层
risk_field.py        旧风险场入口兼容层
resact_metadrive.py  旧 ResAct 入口兼容层
train_sacl.py        SAC-Lag 训练入口
train_ppol.py        PPO-Lag 训练入口
eval.py              模型评估入口
debug_*.py           风险场和路线可视化调试脚本
examples/            其他算法接入示例
tests/               轻量单元测试和 smoke 测试
```

## 环境安装

推荐 Linux + conda + Python 3.10。MetaDrive 依赖 Panda3D，远程服务器建议使用无渲染模式或配置好图形驱动。

```bash
conda create -n metadrive python=3.10 -y
conda activate metadrive
python -m pip install -U pip setuptools wheel
```

安装本仓库依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

如果你希望显式安装本地 vendored 包，也可以单独执行：

```bash
pip install -e ./metadrive
pip install -e ./fsrl
```

如果 `metadrive/metadrive/assets/` 不存在，先下载 MetaDrive 资源：

```bash
python -m metadrive.pull_asset
```

`assets/` 和 `assets.zip` 体积较大，默认不建议提交到 GitHub。

PyTorch 请按你的 CUDA 版本安装。CPU 版本示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`bullet_safety_gym` 和 `safety_gymnasium` 只在运行非 MetaDrive 的 Bullet/Mujoco 任务时需要。只训练或评估 SafeMetaDrive 时可以不安装。

## 快速验证

导入兼容层：

```bash
python -c "from env import get_training_env; from risk_field import RiskFieldCalculator; from resact_metadrive import wrap_residual_action_env; print('imports ok')"
```

编译主要入口：

```bash
python -m py_compile env.py risk_field.py resact_metadrive.py train_sacl.py train_ppol.py eval.py debug_risk_field_snapshot.py safe_metadrive_adapter/*.py
```

运行轻量测试：

```bash
pytest -q tests/test_risk_field_lane_profile.py tests/test_resact_action_wrapper.py
```

运行本地 MetaDrive smoke test：

```bash
python - <<'PY'
from env import get_training_env
from risk_field import RiskFieldCalculator

env = get_training_env({"log_level": 50, "num_scenarios": 1, "start_seed": 100})
try:
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    _, risk_info = RiskFieldCalculator(env.config).calculate(env, env.agent)
    print("obs:", getattr(obs, "shape", type(obs)))
    print("reward:", reward)
    print("risk_field_cost:", risk_info["risk_field_cost"])
    print("risk_field_object_cost:", risk_info["risk_field_object_cost"])
finally:
    env.close()
PY
```

## 训练与评估

保留旧命令：

```bash
python train_sacl.py --task SafeMetaDrive
python train_ppol.py --task SafeMetaDrive
python eval.py
```

评估脚本默认读取 `eval.py` 顶部的 `EVAL_CONFIG`。如果 GitHub 克隆环境里没有模型权重，请先训练得到 checkpoint，或者把 `EVAL_CONFIG["model_path"]` 改成你本机已有的模型路径。

保存风险场快照：

```bash
python debug_risk_field_snapshot.py --warmup-steps 40 --traffic-density 0.1 --accident-prob 1.0
```

## 新接口示例

其他算法推荐直接使用 adapter 工厂：

```python
from safe_metadrive_adapter import make_safe_metadrive_env

env = make_safe_metadrive_env(
    split="train",
    config={
        "traffic_density": 0.05,
        "use_risk_field_reward": True,
        "risk_field_reward_scale": 25.0,
    },
    resact={
        "enabled": True,
        "resact_steer_delta_scale": 0.15,
        "resact_throttle_delta_scale": 0.10,
    },
)
```

旧导入仍然可用：

```python
from env import DEFAULT_CONFIG, get_training_env, get_validation_env
from risk_field import RiskFieldCalculator
from resact_metadrive import ResidualActionWrapper, wrap_residual_action_env
```

## 常用配置

奖励相关：

- `use_lateral_reward`：是否启用横向居中奖励。
- `lateral_reward_weight`：横向奖励权重。
- `use_steering_penalty`：是否启用转向变化惩罚。
- `use_absolute_steering_penalty`：是否启用绝对转向幅值惩罚。
- `crash_vehicle_penalty`、`crash_object_penalty`、`out_of_road_penalty`：事件 reward 覆盖惩罚。

风险场相关：

- `use_risk_field_reward`：是否把风险场作为 reward shaping。
- `risk_field_reward_scale`：风险场 reward 惩罚缩放。
- `use_risk_field_cost`：是否把风险场计入 Safe RL cost。
- `risk_field_lane_weight`、`risk_field_vehicle_weight`、`risk_field_object_weight`：车道线、车辆、静态障碍物风险权重。
- `risk_field_headway_time_threshold`、`risk_field_ttc_threshold`：跟车时距和 TTC 阈值。

ResAct 相关：

- `resact_enabled`：是否通过工厂自动包裹残差动作。
- `resact_steer_delta_scale`：单步转向残差尺度。
- `resact_throttle_delta_scale`：单步油门/刹车残差尺度。
- `resact_initial_action`：初始动作。

## GitHub 上传前检查

本仓库默认不提交训练日志、wandb 目录、GIF、PNG、视频和模型权重。上传前建议执行：

```bash
find . -type f -size +20M
git status --short
```

确认没有以下内容等待提交：

```text
logs/
wandb/
data/
data2/
data_sacrc/
debug/
*.pt
*.pth
*.ckpt
*.gif
*.mp4
*.png
metadrive/metadrive/assets/
metadrive/metadrive/assets.zip
```

如果确实需要公开模型，请优先放在 GitHub Release、网盘或 Git LFS 中，并在 README 中给出下载位置。

## 说明

`safe_metadrive_adapter/local_import.py` 会优先把仓库内 `metadrive/` 放到 `sys.path` 前面，所以本项目脚本默认使用本地仿真环境，而不是系统里可能安装的其他 MetaDrive 版本。
