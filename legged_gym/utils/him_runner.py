"""Lazy selection shared by training, playback and continuation entries."""


def get_him_runner_class(train_cfg):
    name = train_cfg.get('runner_class_name', train_cfg['runner'].get('runner_class_name', 'HIMOnPolicyRunner'))
    if name == 'HIMOnPolicyRunner':
        from third_party.himloco_rsl_rl.rsl_rl.runners import HIMOnPolicyRunner
        return HIMOnPolicyRunner
    if name == 'HIMAMPOnPolicyRunner':
        from third_party.himloco_rsl_rl.rsl_rl.amp.runner import HIMAMPOnPolicyRunner
        return HIMAMPOnPolicyRunner
    raise ValueError('Unsupported HIM runner: '+str(name))
