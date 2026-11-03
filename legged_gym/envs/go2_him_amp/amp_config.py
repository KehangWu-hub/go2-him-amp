"""AMP is an additive training objective on the approved bounded HIM recipe."""
from legged_gym.envs.go2_him.flat_bounded_config import Go2HIMFlatBoundedCfg, Go2HIMFlatBoundedCfgPPO


class Go2HIMAMPCfg(Go2HIMFlatBoundedCfg):
    training_profile = 'go2_him_amp_v1'
    training_task = 'go2_him_amp'
    hardware_deployment_blocked = True
    hardware_deployment_block_reason = 'HIM+AMP experimental training; not qualified for hardware'


class Go2HIMAMPCfgPPO(Go2HIMFlatBoundedCfgPPO):
    runner_class_name = 'HIMAMPOnPolicyRunner'

    class runner(Go2HIMFlatBoundedCfgPPO.runner):
        runner_class_name = 'HIMAMPOnPolicyRunner'
        experiment_name = 'go2_him_amp'

    class amp:
        # Engineering starting values; no claim of a selected successful recipe.
        weight = 1.0
        manifest = 'datasets/go2_amp/manifest.json'
        hidden_dims = [1024, 512]
        learning_rate = 3e-4
        gradient_penalty = 10.
        replay_capacity = 100000
        batch_size = 512
        updates_per_iteration = 20
        seed = 42
        allow_diagnostic_data = False
