# Go2 平地HIM策略权重

`policy.pt`用于部署，`model_3000.pt`用于继续训练。在项目根目录运行：

MuJoCo仿真：

```bash
python -m deploy.deploy_mujoco.deploy_go2 --config deploy/deploy_mujoco/configs/go2_him_flat_bounded.yaml --policy release/go2_him_flat_bounded_3000/policy.pt --deploy-config release/go2_him_flat_bounded_3000/deploy.yaml --command 0.3 0 0
```

真机键盘控制（网卡`eno1`）：

```bash
python -m deploy.deploy_real.go2_keyboard --network eno1 --config deploy/deploy_real/configs/go2_him_flat_bounded.yaml --baseline deploy/deploy_real/configs/go2_him_flat_bounded_baseline.json --run
```
