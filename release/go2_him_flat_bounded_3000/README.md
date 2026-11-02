# Go2 平地HIM模型（3000轮）

面向Unitree Go2的平地策略权重。训练及部署代码：https://github.com/KehangWu-hub/unitree_rl_gym ，配套发布代码提交 `5816791`。

|文件|用途|
|---|---|
|policy.pt|完整TorchScript推理图，包含历史编码器和Actor|
|model_3000.pt|训练断点，含Actor/Critic、估计器及优化器状态|
|deploy.yaml|观测顺序、缩放、关节映射、动作和PD参数|
|training_env.yaml / training_ppo.yaml|本轮实际训练配置|
|training_source_hashes.json|训练时源码摘要|
|SHA256SUMS|包内文件校验摘要|

输入为270维、最新帧在前的6×45维本体观测历史；Actor使用当前观测、估计速度和16维隐变量。输出12维有界动作，范围±4；关节目标为默认角度加动作×0.25，PD为20/0.5，策略周期20ms。启动时历史旧帧置零，顺序与缩放见deploy.yaml。

训练：平地、2048环境、每轮100步、3000轮、seed42、从随机参数开始；固定学习率3e-4，scaled-tanh动作分布。完整运行共614400000条样本；权重并非旧失败模型的续训结果。

在仓库根目录按项目安装说明配置环境后，运行随仓库配套的同一权重：

```bash
python -m deploy.deploy_mujoco.deploy_go2 --config deploy/deploy_mujoco/configs/go2_him_flat_bounded.yaml --command 0.3 0 0
```

使用本目录下载的权重和参数时，从仓库根目录运行（替换绝对路径）：

```bash
python -m deploy.deploy_mujoco.deploy_go2 --config deploy/deploy_mujoco/configs/go2_him_flat_bounded.yaml --policy /absolute/path/policy.pt --deploy-config /absolute/path/deploy.yaml --command 0.3 0 0
```

真机配置为仓库的deploy/deploy_real/configs/go2_him_flat_bounded.yaml及同名baseline.json；连接、站姿接管、START及停止操作见deploy/deploy_real/README_GO2_HIM.zh.md。deploy.yaml仅定义模型契约，不是独立真机控制程序。用户已现场运行并反馈满意；该反馈不等于所有地形、速度与工况均已验证。

许可证与仓库一致，见LICENSE及LICENSE_LEGGED_GYM；上游来源见upstream.json。保留原有版权及许可证声明。
