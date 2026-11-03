# 训练与部署指南

[项目首页](../README.md) · [环境安装](setup_zh.md) · [专家动作格式](../datasets/go2_amp/README.md)

本指南覆盖 HIM+AMP、HIM 和 PPO 的训练、导出、MuJoCo 仿真及真机键盘控制。三条路线使用配套的模型、接口参数和真机配置，共用部署入口。

所有命令从仓库根目录执行，先完成[环境安装](setup_zh.md)：

```bash
conda activate unitree-rl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## 训练

### HIM+AMP

先按[数据格式](../datasets/go2_amp/README.md)准备 Go2 专家动作及 `datasets/go2_amp/manifest.json`。数据保存在本地，不随仓库分发。

```bash
python legged_gym/scripts/train_him_from_scratch.py \
  --profile him-amp --run-dir logs/go2_him_amp/my_run \
  --amp-manifest datasets/go2_amp/manifest.json --amp-weight 1.0 \
  --headless --num_envs 2048 --max_iterations 3000 --seed 42
```

`him-amp` 使用独立平地任务，为原 HIM 任务奖励增加 `amp_weight × 控制周期 × 风格奖励`。程序核对数据来源、文件摘要、Go2 URDF、关节顺序与足端运动学。部署模型只包含 Source Encoder 与 Actor，不加载判别器或专家数据。

### HIM

使用平地、有界动作、固定学习率的配置：

```bash
python legged_gym/scripts/train_him_from_scratch.py \
  --profile flat-bounded --run-dir logs/him/my_run \
  --headless --num_envs 2048 --max_iterations 3000 --seed 42
```

`candidate-rough` 提供混合地形及地形课程，`candidate-flat` 为对应的平地候选。基础 HIM 任务可通过 `python legged_gym/scripts/train.py --task go2_him --headless` 运行。

### 参数、导出与续训

HIM 和 HIM+AMP 使用相同的训练脚本：

| 参数 | 作用 |
|---|---|
| `--profile` | `him-amp` 或 `flat-bounded` 等训练配置 |
| `--run-dir` | 独立运行目录，每次新训练使用新目录 |
| `--num_envs` | 并行环境数，按显存选择 |
| `--max_iterations` | 训练轮数 |
| `--seed` | 随机种子 |
| `--background` | 后台运行，输出保存在运行目录 |
| `--amp-manifest` / `--amp-weight` | HIM+AMP 专家动作清单及风格奖励权重 |

首次检查训练链路时，可将环境数和轮数改为 `--num_envs 64 --max_iterations 3`。正式训练的目录保存配置、源码快照、断点和完整导出模型：

```text
logs/go2_him_amp/my_run/
├── env.yaml
├── train.yaml
├── progress.json
├── model_3000.pt
└── exported/
    ├── policies/policy.pt
    └── params/deploy.yaml
```

将已有 HIM+AMP 运行继续训练至总计 5000 轮：

```bash
python legged_gym/scripts/continue_him_training.py \
  --source-run logs/go2_him_amp/my_run \
  --run-dir logs/go2_him_amp/my_run_extended \
  --target-iteration 5000 --background
```

HIM 使用同一续训入口，替换运行目录即可。HIM+AMP 会恢复判别器、优化器、归一化、Replay 和随机状态，并核对专家数据身份。Runner 的 `load(..., load_optimizer=False)` 只加载网络权重，重置优化器、迭代和 AMP 状态，不属于完整续训。

### PPO

使用原 PPO 训练与回放导出入口：

```bash
python legged_gym/scripts/train.py \
  --task go2 --headless --num_envs 1024 --max_iterations 1500

python legged_gym/scripts/play.py \
  --task go2 --headless --load_run RUN_DIRECTORY --checkpoint 1500 --export_only
```

将 `RUN_DIRECTORY` 替换为本次训练目录名。程序打印导出的模型与 `deploy.yaml` 路径，后续使用同一组配套文件。

## MuJoCo运行

运行仓库内的 HIM+AMP 模型：

```bash
python -m deploy.deploy_mujoco.deploy_go2 \
  --config deploy/deploy_mujoco/configs/go2_him_flat_bounded.yaml \
  --policy deploy/pre_train/go2_him_amp/policy.pt \
  --deploy-config deploy/pre_train/go2_him_amp/deploy.yaml \
  --command 0.3 0 0
```

HIM 与 HIM+AMP 共用相同的仿真控制接口。运行 HIM 时，只需指定 `--config deploy/deploy_mujoco/configs/go2_him_flat_bounded.yaml`；运行 PPO 使用 `--config deploy/deploy_mujoco/configs/go2.yaml`。两者均可省略 `--policy` 和 `--deploy-config`，读取所选配置中的配套文件。

运行自己的训练结果时，将 `--policy` 和 `--deploy-config` 分别替换为运行目录下的 `exported/policies/policy.pt` 与 `exported/params/deploy.yaml`。

`--command VX VY WZ` 依次为前进、侧移、偏航指令，单位为 m/s、m/s、rad/s。添加 `--headless --no-realtime --duration 10` 可运行无窗口仿真并输出统计。

## 真机键盘控制

### 选择模型与配置

通过网线连接机器人，使用同一子网内的电脑地址，DDS domain 为 0。下文网卡以 `eno1` 为例，替换为实际有线网卡名。机器人从原厂站稳姿态接管，现场保持平坦空旷，并准备停止终端。

| 路线 | 配置 | 版本清单 |
|---|---|---|
| HIM+AMP | `go2_him_amp.yaml` | `go2_him_amp_baseline.json` |
| HIM | `go2_him_flat_bounded.yaml` | `go2_him_flat_bounded_baseline.json` |
| PPO | `go2.yaml` | `go2_verified_baseline.json` |

以上文件均位于 `deploy/deploy_real/configs/`。模型与配置必须成套选择。以下默认使用 HIM+AMP；切换路线时同步替换两个变量：

```bash
GO2_CONFIG=deploy/deploy_real/configs/go2_him_amp.yaml
GO2_BASELINE=deploy/deploy_real/configs/go2_him_amp_baseline.json
```

检查本地配置和模型，不初始化 DDS：

```bash
python -m deploy.deploy_real.deploy_real_go2 --config "$GO2_CONFIG" --check
```

机器人连接后，只读查看状态：

```bash
python -m deploy.deploy_real.deploy_real_go2 eno1 \
  --config "$GO2_CONFIG" --read-only
```

按 Ctrl+C 结束只读采集。

### HIM与HIM+AMP启动

核对配置和版本清单后，启动键盘控制：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --config "$GO2_CONFIG" --baseline "$GO2_BASELINE" --check

python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --config "$GO2_CONFIG" --baseline "$GO2_BASELINE" \
  --speed-profile legacy --run
```

HIM+AMP 当前作为独立试用策略，现场先使用 W/S/A/D 的 0.30 m/s 平移指令；低速原地转向仍待改进。

### PPO首次配置与启动

PPO 需要与当前模型、配置匹配的站姿及零速记录，不能直接套用 HIM 的前置流程。先将上面的变量切换为 PPO 配置与版本清单，再进行站姿接管：

```bash
python -m deploy.deploy_real.go2_commission eno1 \
  --config "$GO2_CONFIG" \
  --stand-check --start-standing --confirm-standing \
  --release-mode --allow-body-movement
```

流程结束后，按[恢复原厂模式](#停止与恢复原厂模式)重新站稳。将 `STAND_JSONL` 替换为刚生成的日志路径，运行零速策略：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --config "$GO2_CONFIG" --baseline "$GO2_BASELINE" \
  --mode zero --readiness STAND_JSONL --run
```

后续运动使用该步骤生成的 `ZERO_JSONL`。恢复原厂站稳后检查并启动：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --config "$GO2_CONFIG" --baseline "$GO2_BASELINE" \
  --readiness ZERO_JSONL --check

python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --config "$GO2_CONFIG" --baseline "$GO2_BASELINE" \
  --readiness ZERO_JSONL --run
```

已有匹配本机记录时可省略 `--readiness`；更换模型或配置后重新生成记录。

### 按键与速度

启动时按终端提示输入 `START`，等待按键窗口开启后控制。

| 按键 | 操作 | HIM / HIM+AMP legacy | PPO 默认统一模式 |
|---|---|---|---|
| W / S | 前进 / 后退 | ±0.30 m/s | ±0.50 m/s |
| A / D | 左移 / 右移 | ±0.30 m/s | ±0.40 m/s |
| Q / E | 前进并左右转 | 前进 0.30 m/s，偏航 ±0.30 rad/s | 前进 0.50 m/s，偏航 ±0.50 rad/s |
| X / 空格 | 速度归零，策略继续运行 | 相同 | 相同 |
| C | 受控降低并退出 | 相同 | 相同 |
| Z / Ctrl+C | 立即阻尼停止并退出 | 相同 | 相同 |

表中为指令值，实际速度随策略和运行条件变化。最后一个方向键生效；超过 0.6 秒未收到方向输入会归零，主动停步使用 X 或空格。默认持续运行至主动退出，受控退出中仍可按 Z 停止。接管期间手机 App 不作为停止入口。

## 停止与恢复原厂模式

第二终端可请求阻尼停止：

```bash
python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0
```

C 正常降低后退出；Z、Ctrl+C 和第二终端停止立即撤除位置支撑并进入阻尼，机身会下沉。独立倾角保护触发后同样锁定阻尼。

控制进程退出、机器人已趴稳后恢复原厂模式：

```bash
python -m deploy.deploy_real.go2_commission eno1 \
  --restore-normal --confirm-prone --allow-factory-stand
```

恢复操作可能直接使机器人站起，完成后可重新使用 App 控制。

## 模型接口与权重替换

| 接口 | HIM / HIM+AMP | PPO |
|---|---|---|
| 仓库模型文件 | `policy.pt` | `motion.pt` |
| 模型输入 | 270维，六帧45维观测，最新帧在前 | 45维当前观测 |
| Actor输入 | 45维观测＋3维估计速度＋16维隐变量 | 45维当前观测 |
| 动作输出 | 12维关节位置动作 | 12维关节位置动作 |
| 策略周期 | 20 ms | 20 ms |
| 策略PD增益 | 20 / 0.5 | 20 / 0.5 |

HIM 每帧观测顺序为指令3、角速度3、投影重力3、相对关节位置12、关节速度12、上一策略动作12；PPO 顺序为角速度3、投影重力3、指令3、关节位置12、关节速度12、上一动作12。具体顺序、缩放及历史语义以各模型的 `deploy.yaml` 为准。

策略关节顺序为 FL/FR/RL/RR，SDK 顺序为 FR/FL/RR/RL，共用接口完成映射。关节参考目标为默认关节角加 `0.25 × 动作`；实际参数以模型导出配置为准。HIM/HIM+AMP 的 Critic 使用238维输入，仅训练时运行。

训练导出的 `deploy.yaml` 是模型接口参数，不能直接替代包含网卡、停止与控制流程参数的完整真机 YAML。更换权重时，同步更新模型包、真机接口字段与匹配的 baseline 摘要。真机配置的部署状态字段控制是否允许启动。

HIM 的[独立权重目录](../release/go2_him_flat_bounded_3000/README.md)还包含训练断点与实际训练配置。训练及真机日志保存在本地 `logs/`，不随仓库分发。

## 可选工具

| 入口 | 用途 |
|---|---|
| `go2_commission --analyze TRIAL_JSONL` | 分析本机控制日志 |
| `go2_velocity_diagnostics --analyze TRIAL_JSONL` | 查看速度观测 |
| `go2_policy_replay TRIAL_JSONL --output replay.json` | 离线回放策略输入输出 |
| `go2_keyboard --velocity-source kinematic --velocity-feedback shadow` | 记录运动学速度估计 |

以上入口位于 `deploy.deploy_real`，使用 `python -m deploy.deploy_real.<入口>` 运行，通过 `--help` 查看参数。速度反馈选项与所选键盘模式配套使用。
