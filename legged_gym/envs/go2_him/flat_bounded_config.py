"""Flat quality retraining with restored bounded actions and fixed learning rate."""
from .candidate_config import Go2HIMFlatQualityCfg, Go2HIMFlatQualityCfgPPO


class Go2HIMFlatBoundedCfg(Go2HIMFlatQualityCfg):
    training_profile = 'go2_him_flat_bounded_fixed_v1'
    training_task = 'go2_him_flat_bounded'
    action_distribution = 'scaled_tanh_normal'

    class normalization(Go2HIMFlatQualityCfg.normalization):
        clip_actions = 4.


class Go2HIMFlatBoundedCfgPPO(Go2HIMFlatQualityCfgPPO):
    class policy(Go2HIMFlatQualityCfgPPO.policy):
        action_bound = 4.

    class algorithm(Go2HIMFlatQualityCfgPPO.algorithm):
        learning_rate = 3e-4
        schedule = 'fixed'

    class runner(Go2HIMFlatQualityCfgPPO.runner):
        policy_class_name = 'BoundedHIMActorCritic'
        experiment_name = 'go2_him_flat_bounded'
        max_iterations = 3000
        num_steps_per_env = 100
